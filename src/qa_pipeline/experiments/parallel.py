"""多卡实验调度：按依赖分波。教师走已配置的远程接口，空闲 GPU 只跑 NLI、向量和 LoRA。"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def split_waves(entries: list[dict[str, Any]]) -> tuple[list, list, list, list]:
    """拆成互不依赖的管线、依赖上游缓存的管线、纯 CPU 后处理、拒答评测。"""
    independent: list[dict[str, Any]] = []
    dependent: list[dict[str, Any]] = []
    light: list[dict[str, Any]] = []
    refusal: list[dict[str, Any]] = []
    for entry in entries:
        kind = entry.get("kind") or "pipeline"
        if kind == "refusal":
            refusal.append(entry)
        elif kind in {"annotate", "cost", "subset", "skipped"}:
            light.append(entry)
        elif entry.get("from_questions") or entry.get("from_kept") or entry.get("from_cache") or entry.get("from_accepted"):
            dependent.append(entry)
        else:
            independent.append(entry)
    return independent, dependent, light, refusal


def _error_row(entry: dict[str, Any], message: str) -> dict[str, Any]:
    text = " ".join(message.split())
    if len(text) > 500:
        text = text[:500] + "…"
    return {
        "id": entry.get("id"),
        "purpose": entry.get("purpose", ""),
        "recipe": entry.get("recipe", ""),
        "recipe_snapshot": {},
        "metrics": {},
        "sft": {},
        "note": f"失败：{text}",
    }


def _deferred_sft(entry: dict[str, Any], skip_sft: bool) -> dict[str, Any]:
    if skip_sft or not entry.get("sft"):
        return {"skipped": True, "reason": "skip_sft" if skip_sft else "not_requested"}
    return {"skipped": True, "reason": "deferred"}


def _load_pairs(path: Path):
    from ..schemas import QAPair
    from ..store import read_jsonl

    if not path.is_file():
        return []
    return [QAPair.model_validate(row) for row in read_jsonl(path)]


def _load_questions(path: Path):
    from ..schemas import Question
    from ..store import read_jsonl

    if not path.is_file():
        return []
    return [Question.model_validate(row) for row in read_jsonl(path)]


def _store_cache(entry: dict[str, Any], result, cache_dir: Path) -> None:
    from ..store import write_jsonl

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = entry.get("cache_as")
    if key:
        write_jsonl(cache_dir / f"{key}.kept.jsonl", [p.model_dump() for p in result.pairs])
        write_jsonl(cache_dir / f"{key}.raw.jsonl", [p.model_dump() for p in (result.raw_pairs or [])])
    qkey = entry.get("cache_questions")
    if qkey:
        write_jsonl(cache_dir / f"{qkey}.questions.jsonl", [q.model_dump() for q in result.questions])


def _run_pipeline_job(job: dict[str, Any], client) -> dict[str, Any]:
    from ..adapters.zhixun import export_zhixun
    from ..pipeline import Pipeline
    from ..store import save_result
    from .metrics import compute_metrics
    from .runner import _apply_grades, _recipe_from_entry, _reset_pairs, recipe_snapshot

    entry = job["entry"]
    exp_dir = Path(job["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(client, "reset_usage"):
        client.reset_usage()
    recipe = _recipe_from_entry(entry, Path(job["recipes_dir"]))
    pipe = Pipeline(recipe, llm=client)
    if job.get("questions_path"):
        questions = _load_questions(Path(job["questions_path"]))
        if not questions:
            raise FileNotFoundError(f"缺少问题缓存 {job['questions_path']}")
        result = pipe.distill_from_questions(questions)
        result.recipe_name = recipe.name
    elif job.get("accepted_path"):
        accepted = _load_pairs(Path(job["accepted_path"]))
        if not accepted:
            raise FileNotFoundError(f"缺少共同合格池 {job['accepted_path']}")
        result = pipe.select_only(accepted)
        result.recipe_name = recipe.name
    elif job.get("raw_path"):
        raw = _load_pairs(Path(job["raw_path"]))
        if not raw:
            raise FileNotFoundError(f"缺少原始问答缓存 {job['raw_path']}")
        result = pipe.refilter(_reset_pairs(raw), llm=client)
        result.recipe_name = recipe.name
    else:
        result = pipe.run(job["input_path"])

    _store_cache(entry, result, Path(job["cache_dir"]))
    grades = entry.get("grades")
    if grades:
        result.pairs = _apply_grades(list(result.pairs), grades)
    save_result(result, exp_dir)
    export_zhixun(result.pairs, exp_dir / "zhixun.jsonl")
    metrics = compute_metrics(result)
    row = {
        "id": entry["id"],
        "purpose": entry.get("purpose", ""),
        "recipe": recipe.name,
        "recipe_snapshot": recipe_snapshot(recipe),
        "metrics": metrics,
        "sft": _deferred_sft(entry, bool(job.get("skip_sft"))),
        "note": "",
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(json.dumps(row, ensure_ascii=False))


def _gpu_worker(jobs: list[dict[str, Any]], model_path: str, result_queue) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logging.info("worker cuda=%s teacher=remote jobs=%s", visible, [job["entry"]["id"] for job in jobs])
    del model_path
    from ..llm import LLMClient

    cfg = (jobs[0].get("client") if jobs else None) or {}
    kwargs = {key: cfg[key] for key in ("api_key", "base_url", "default_model", "timeout") if cfg.get(key)}
    client = LLMClient(**kwargs)
    try:
        for job in jobs:
            exp_id = job["entry"]["id"]
            try:
                result_queue.put(_run_pipeline_job(job, client))
                logging.info("finished %s", exp_id)
            except Exception:
                logging.exception("job %s failed", exp_id)
                result_queue.put(_error_row(job["entry"], traceback.format_exc()))
    finally:
        if hasattr(client, "release"):
            client.release()
        result_queue.put(None)


def _sft_worker(jobs: list[dict[str, Any]], model_path: str, result_queue) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    del model_path
    from ..llm import LLMClient
    from .sft import evaluate_sft

    cfg = (jobs[0].get("client") if jobs else None) or {}
    kwargs = {key: cfg[key] for key in ("api_key", "base_url", "default_model", "timeout") if cfg.get(key)}
    client = LLMClient(**kwargs)
    try:
        for job in jobs:
            try:
                pairs = _load_pairs(Path(job["pairs_path"]))
                report = evaluate_sft(
                    pairs,
                    job["heldout"],
                    Path(job["out_dir"]),
                    llm=client,
                    base_model=job["base_model"],
                    skip_sft=False,
                )
                adapter = (report.get("train") or {}).get("adapter")
                result_queue.put({"id": job["id"], "sft": report, "adapter": adapter})
            except Exception:
                logging.exception("sft %s failed", job.get("id"))
                result_queue.put(
                    {
                        "id": job["id"],
                        "sft": {"skipped": True, "reason": traceback.format_exc()[-500:]},
                        "adapter": None,
                    }
                )
    finally:
        if hasattr(client, "release"):
            client.release()
        result_queue.put(None)


def _spawn(ctx, target, args, device: int):
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        proc = ctx.Process(target=target, args=args)
        proc.start()
        return proc
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def _collect(procs: list[tuple[int, Any, list[str]]], result_queue, on_message) -> None:
    finished = 0
    expected = len(procs)
    seen: set[str] = set()
    while finished < expected:
        try:
            msg = result_queue.get(timeout=60)
        except queue.Empty:
            if procs and all(not proc.is_alive() for _, proc, _ in procs):
                break
            continue
        if msg is None:
            finished += 1
            continue
        if msg.get("id"):
            seen.add(msg["id"])
        on_message(msg)
    for device, proc, ids in procs:
        proc.join(timeout=60)
        if proc.exitcode not in (0, None):
            for exp_id in ids:
                if exp_id not in seen:
                    on_message(
                        _error_row(
                            {"id": exp_id},
                            f"GPU {device} 进程退出码 {proc.exitcode}",
                        )
                    )


def _pipeline_job(entry: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(ctx["cache_dir"])
    job = {
        "entry": entry,
        "recipes_dir": ctx["recipes_dir"],
        "input_path": ctx["input_path"],
        "exp_dir": str(Path(ctx["out"]) / entry["id"]),
        "cache_dir": str(cache_dir),
        "skip_sft": ctx["skip_sft"],
    }
    if entry.get("from_questions"):
        job["questions_path"] = str(cache_dir / f"{entry['from_questions']}.questions.jsonl")
    if entry.get("from_accepted"):
        job["accepted_path"] = str(cache_dir / f"{entry['from_accepted']}.kept.jsonl")
    if entry.get("from_cache"):
        job["raw_path"] = str(cache_dir / f"{entry['from_cache']}.raw.jsonl")
    job["client"] = ctx.get("client") or {}
    return job


def _assign(jobs: list[dict[str, Any]], devices: list[int]) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        buckets[devices[index % len(devices)]].append(job)
    return buckets


def _run_workers(jobs: list[dict[str, Any]], devices: list[int], model_path: str, target, on_message) -> None:
    if not jobs:
        return
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for device, bucket in _assign(jobs, devices).items():
        if not bucket:
            continue
        proc = _spawn(ctx, target, (bucket, model_path, result_queue), device)
        procs.append((device, proc, [job["entry"]["id"] if "entry" in job else job["id"] for job in bucket]))
    _collect(procs, result_queue, on_message)


def _write_payload(
    out: Path,
    suite: dict[str, Any],
    input_path: Path,
    student: str,
    devices: list[int],
    notes: list[str],
    order: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    teacher_model: str = "qwen3.8-27b",
) -> dict[str, Any]:
    from .runner import _render_report

    rows = [rows_by_id[exp_id] for exp_id in order if exp_id in rows_by_id]
    payload = {
        "suite": suite.get("name"),
        "input": str(input_path),
        "llm": "live",
        "teacher_model": teacher_model,
        "base_model": student,
        "devices": devices,
        "limitations": notes,
        "progress": {"done": len(rows), "total": len(order)},
        "experiments": rows,
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        _render_report(suite.get("name", "suite"), rows, notes),
        encoding="utf-8",
    )
    return payload


def _run_light_entry(entry: dict[str, Any], ctx: dict[str, Any], rows_by_id: dict[str, dict[str, Any]], cache_kept: dict) -> dict[str, Any]:
    from ..adapters.zhixun import export_zhixun
    from ..store import save_result
    from .metrics import compute_metrics
    from .runner import _apply_grades, _empty_result, _export_annotation

    exp_id = entry["id"]
    kind = entry.get("kind") or "pipeline"
    exp_dir = Path(ctx["out"]) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    if kind == "skipped":
        return {
            "id": exp_id,
            "purpose": entry.get("purpose", ""),
            "recipe": "",
            "recipe_snapshot": {},
            "metrics": {},
            "sft": {},
            "note": entry.get("reason") or "未执行",
        }
    if kind == "annotate":
        source = cache_kept.get(entry.get("from_kept") or "") or []
        ann = _export_annotation(source, exp_dir / "annotation.jsonl", entry.get("grades"))
        return {
            "id": exp_id,
            "purpose": entry.get("purpose", ""),
            "recipe": "",
            "recipe_snapshot": {},
            "metrics": {"kept": ann["exported"]},
            "sft": {},
            "note": ann.get("reason") or "",
            "annotation": ann,
        }
    if kind == "cost":
        found = rows_by_id.get(entry.get("source") or "")
        metrics = dict((found or {}).get("metrics") or {})
        metrics["api_cost_usd"] = metrics.get("estimated_cost_usd")
        metrics["cost_note"] = "教师 qwen3.8-27b 为内网接口，API 费用按 0 计，tokens_per_10k_kept 为等价 token。"
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "id": exp_id,
            "purpose": entry.get("purpose", ""),
            "recipe": "",
            "recipe_snapshot": {},
            "metrics": metrics,
            "sft": {},
            "note": metrics["cost_note"],
        }
    source = cache_kept.get(entry.get("from_kept") or "") or []
    pairs = _apply_grades(source, entry.get("grades"))
    result = _empty_result(entry.get("name") or exp_id, pairs)
    if entry.get("grades"):
        result.pairs = _apply_grades(list(result.pairs), entry.get("grades"))
    save_result(result, exp_dir)
    export_zhixun(result.pairs, exp_dir / "zhixun.jsonl")
    metrics = compute_metrics(result)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "id": exp_id,
        "purpose": entry.get("purpose", ""),
        "recipe": entry.get("recipe", ""),
        "recipe_snapshot": {},
        "metrics": metrics,
        "sft": _deferred_sft(entry, bool(ctx["skip_sft"])),
        "note": "",
    }


def _load_cache_kept(cache_dir: Path) -> dict[str, list]:
    kept = {}
    if not cache_dir.is_dir():
        return kept
    for path in cache_dir.glob("*.kept.jsonl"):
        kept[path.name[: -len(".kept.jsonl")]] = _load_pairs(path)
    return kept


def _sft_plan(entries: list[dict[str, Any]], out: Path, heldout: Path, student: str) -> tuple[list[dict], list[tuple[str, str]]]:
    from .scoring import cache_signature

    owners: dict[str, str] = {}
    jobs: list[dict[str, Any]] = []
    reuse: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.get("sft"):
            continue
        exp_id = entry["id"]
        pairs = _load_pairs(out / exp_id / "qa.kept.jsonl")
        if not pairs:
            continue
        sig = cache_signature(
            {
                "samples": [
                    {
                        "id": pair.qa_id,
                        "question": pair.question,
                        "answer": pair.answer,
                        "context": pair.metadata.get("student_context", pair.chunk_text),
                        "loss_weight": pair.loss_weight,
                        "included": pair.included_in_this_run,
                        "hash": pair.validation_subject_hash,
                    }
                    for pair in pairs
                ],
                "model": student,
                "heldout": str(heldout),
            }
        )
        if sig in owners:
            reuse.append((exp_id, owners[sig]))
            continue
        owners[sig] = exp_id
        jobs.append(
            {
                "id": exp_id,
                "entry": {"id": exp_id},
                "pairs_path": str(out / exp_id / "qa.kept.jsonl"),
                "heldout": str(heldout),
                "out_dir": str(out / exp_id / "sft"),
                "base_model": student,
                "client": {},
            }
        )
    return jobs, reuse


def run_parallel(
    suite: dict[str, Any],
    out: Path,
    recipes_dir: Path,
    input_path: Path,
    heldout: Path,
    refusal_path: Path,
    model_path: str,
    student: str,
    devices: list[int],
    skip_sft: bool,
    notes: list[str],
    client_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .runner import abort_missing_heldout, gate_heldout, write_run_meta

    gate = gate_heldout(heldout)
    if not gate["ok"]:
        return abort_missing_heldout(suite, heldout)
    entries = list(suite.get("experiments") or [])
    order = [entry["id"] for entry in entries]
    independent, dependent, light, refusal = split_waves(entries)
    cache_dir = out / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher_model = (client_config or {}).get("default_model") or "qwen3.8-27b"
    notes = list(notes) + [f"管线并行于 GPU {devices}。教师为 {teacher_model}，实验卡只加载 NLI、向量和 LoRA。"]
    if gate["heldout_status"] == "empty" and not any("主评测不加载模型" in item for item in notes):
        notes.append("主测试清单存在但没有可评分题目。允许训练和训练原题探针，主评测不加载模型，delta_f1 为空。")
    ctx = {
        "out": str(out),
        "recipes_dir": str(recipes_dir),
        "input_path": str(input_path),
        "cache_dir": str(cache_dir),
        "skip_sft": skip_sft,
        "client": client_config or {},
    }
    rows_by_id: dict[str, dict[str, Any]] = {}
    _write_payload(out, suite, input_path, student, devices, notes, order, rows_by_id, teacher_model)

    def accept_row(row: dict[str, Any]) -> None:
        if not row.get("id"):
            return
        rows_by_id[row["id"]] = row
        _write_payload(out, suite, input_path, student, devices, notes, order, rows_by_id, teacher_model)

    def on_pipeline(msg: dict[str, Any]) -> None:
        if "recipe" in msg or "note" in msg:
            accept_row(msg)

    for wave in (independent, dependent):
        jobs = [_pipeline_job(entry, ctx) for entry in wave]
        logger.info("wave %s", [job["entry"]["id"] for job in jobs])
        _run_workers(jobs, devices, model_path, _gpu_worker, on_pipeline)

    cache_kept = _load_cache_kept(cache_dir)
    for entry in light:
        try:
            accept_row(_run_light_entry(entry, ctx, rows_by_id, cache_kept))
        except Exception:
            logger.exception("light %s failed", entry.get("id"))
            accept_row(_error_row(entry, traceback.format_exc()))

    adapters: dict[str, str] = {}
    if not skip_sft:
        jobs, reuse = _sft_plan(entries, out, heldout, student)
        for job in jobs:
            job["client"] = client_config or {}
        for entry in entries:
            if not entry.get("sft"):
                continue
            path = out / entry["id"] / "qa.kept.jsonl"
            if entry["id"] in {job["id"] for job in jobs} or entry["id"] in {item[0] for item in reuse}:
                continue
            if entry["id"] in rows_by_id and not _load_pairs(path):
                rows_by_id[entry["id"]]["sft"] = {"skipped": True, "reason": "empty_train"}
                accept_row(rows_by_id[entry["id"]])

        def on_sft(msg: dict[str, Any]) -> None:
            exp_id = msg.get("id")
            row = rows_by_id.get(exp_id)
            if not row:
                return
            if "sft" in msg:
                row["sft"] = msg.get("sft") or {}
                adapter = msg.get("adapter")
                if adapter:
                    adapters[exp_id] = adapter
            elif msg.get("note"):
                row["sft"] = {"skipped": True, "reason": msg["note"]}
            accept_row(row)

        logger.info("sft %s", [job["id"] for job in jobs])
        _run_workers(jobs, devices, model_path, _sft_worker, on_sft)
        from .sft import assemble_answer_book

        assemble_answer_book(out, entries)
        for exp_id, owner in reuse:
            row = rows_by_id.get(exp_id)
            source = rows_by_id.get(owner) or {}
            if not row:
                continue
            row["sft"] = {**(source.get("sft") or {}), "reused": True}
            if owner in adapters:
                adapters[exp_id] = adapters[owner]
            accept_row(row)

    if refusal:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        from .refusal import evaluate_refusal, load_refusal_set

        for entry in refusal:
            report: dict[str, Any] = {"skipped": skip_sft, "arms": {}}
            exp_dir = out / entry["id"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            if skip_sft or not refusal_path.is_file():
                report["reason"] = "skip_sft" if skip_sft else f"missing {refusal_path}"
            else:
                refusal_rows = load_refusal_set(refusal_path)
                for arm in entry.get("compare") or []:
                    adapter = adapters.get(arm)
                    if not adapter:
                        report["arms"][arm] = {"skipped": True, "reason": "no_adapter"}
                        continue
                    report["arms"][arm] = evaluate_refusal(refusal_rows, student, adapter)
            (exp_dir / "refusal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            accept_row(
                {
                    "id": entry["id"],
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {},
                    "sft": {},
                    "note": json.dumps(report.get("arms") or report.get("reason"), ensure_ascii=False),
                    "refusal": report,
                }
            )
    payload = _write_payload(out, suite, input_path, student, devices, notes, order, rows_by_id, teacher_model)
    write_run_meta(out, suite, heldout, student)
    return payload
