"""实验套件：按 YAML 跑多组 recipe，产出 metrics.json 与 report.md。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..config import Recipe, StrategySpec, load_recipe
from ..llm import FakeLLM, LLMClient
from ..pipeline import Pipeline, PipelineResult
from ..schemas import QAPair
from ..store import save_result
from .metrics import compute_metrics
from .sft import evaluate_sft


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def gate_heldout(path: str | Path) -> dict[str, Any]:
    """生成前检查主测试清单。缺失则不能开跑；空文件允许流程验证。"""
    heldout = Path(path)
    if not heldout.is_file():
        return {
            "ok": False,
            "reason": "missing_heldout",
            "heldout": str(heldout),
            "heldout_status": "missing",
        }
    rows = [line for line in heldout.read_text(encoding="utf-8").splitlines() if line.strip()]
    status = "empty" if not rows else "ready"
    return {"ok": True, "reason": None, "heldout": str(heldout), "heldout_status": status, "n": len(rows)}


def abort_missing_heldout(suite: dict[str, Any], heldout: str | Path) -> dict[str, Any]:
    return {
        "suite": suite.get("name"),
        "aborted": True,
        "reason": "missing_heldout",
        "heldout": str(heldout),
        "experiments": [],
        "note": "主测试清单缺失。不开始生成，也不回退到历史题干清单。",
    }


def write_run_meta(out: Path, suite: dict[str, Any], heldout: str | Path, student: str) -> Path:
    import hashlib
    import subprocess

    root = _repo_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    def digest(path: Path) -> str | None:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    trains = {}
    if out.is_dir():
        for path in sorted(out.glob("*/sft/train.jsonl")):
            trains[str(path.relative_to(out))] = digest(path)
    version: dict[str, Any] = {"path": student}
    config = Path(student) / "config.json"
    if config.is_file():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        version.update(
            {
                "model_type": data.get("model_type"),
                "architectures": data.get("architectures"),
                "name_or_path": data.get("_name_or_path") or data.get("name_or_path"),
            }
        )
    meta = {
        "run_id": suite.get("run_id") or out.name,
        "suite": suite.get("name"),
        "git_commit": commit,
        "heldout_protocol_sha256": digest(Path(heldout)),
        "train_jsonl_sha256": trains or None,
        "base_model": student,
        "model_version": version,
    }
    dest = out / "run_meta.json"
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def load_suite(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def recipe_snapshot(recipe: Recipe) -> dict[str, Any]:
    """记录一组实验实际使用的各阶段策略，供对照台解释分数差异。"""

    def spec(item: StrategySpec) -> dict[str, Any]:
        return {"name": item.name, **item.params}

    return {
        "name": recipe.name,
        "description": recipe.description,
        "chunking": spec(recipe.chunking),
        "anchor": spec(recipe.anchor),
        "question_gen": spec(recipe.question_gen),
        "evolution": spec(recipe.evolution),
        "question_filter": spec(recipe.question_filter),
        "distillation": spec(recipe.distillation),
        "teacher_router": spec(recipe.teacher_router),
        "filters": [spec(item) for item in recipe.filters],
        "grading": recipe.grading,
    }


def _recipe_from_entry(entry: dict[str, Any], recipes_dir: Path) -> Recipe:
    if "recipe" in entry:
        return load_recipe(recipes_dir / entry["recipe"] if not Path(entry["recipe"]).is_file() else entry["recipe"])
    if "from" in entry:
        base = load_recipe(recipes_dir / entry["from"] if not Path(entry["from"]).is_file() else entry["from"])
        data = base.dump()
        overlay = deepcopy(entry.get("override") or {})
        data.update({k: v for k, v in overlay.items() if k != "filters"})
        if "filters" in overlay:
            data["filters"] = overlay["filters"]
        if entry.get("name"):
            data["name"] = entry["name"]
        return Recipe.from_dict(data)
    return Recipe.from_dict(entry)


def _render_report(suite_name: str, rows: list[dict[str, Any]], notes: list[str]) -> str:
    keys = [
        "id",
        "kept",
        "retention",
        "answerability",
        "evidence_grounded",
        "nli_mean",
        "judge_mean",
        "type_entropy",
        "s_ratio",
        "tokens_per_10k_kept",
        "delta_f1",
    ]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [f"# {suite_name} 实验报告", "", header, sep]
    for row in rows:
        m = row.get("metrics") or {}
        sft = row.get("sft") or {}
        values = {
            "id": row.get("id"),
            "kept": m.get("kept"),
            "retention": m.get("retention"),
            "answerability": m.get("answerability"),
            "evidence_grounded": m.get("evidence_grounded"),
            "nli_mean": m.get("nli_mean"),
            "judge_mean": m.get("judge_mean"),
            "type_entropy": m.get("type_entropy"),
            "s_ratio": m.get("s_ratio"),
            "tokens_per_10k_kept": m.get("tokens_per_10k_kept"),
            "delta_f1": sft.get("delta_f1"),
        }
        lines.append("| " + " | ".join(str(values[k]) for k in keys) + " |")
    lines += ["", "## 说明", ""]
    for row in rows:
        extra = row.get("note") or ""
        lines.append(f"- **{row.get('id')}**：{row.get('purpose') or ''} {extra}".rstrip())
    if notes:
        lines += ["", "## 运行约束", ""]
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines) + "\n"


def _empty_result(name: str, pairs: list[QAPair]) -> PipelineResult:
    from ..schemas import UsageStats

    return PipelineResult(
        recipe_name=name,
        documents=[],
        chunks=[],
        questions=[],
        pairs=pairs,
        rejected=[],
        stats=UsageStats(produced={"distilled": len(pairs), "kept": len(pairs), "questions": len(pairs)}),
        raw_pairs=list(pairs),
    )


def _apply_grades(pairs: list[QAPair], grades: list[str] | None) -> list[QAPair]:
    if not grades:
        return list(pairs)
    allow = set(grades)
    return [p.model_copy(deep=True) for p in pairs if p.grade in allow]


def _export_annotation(pairs: list[QAPair], path: Path, grades: list[str] | None) -> dict[str, Any]:
    chosen = _apply_grades(pairs, grades or ["S", "A"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in chosen:
            handle.write(
                json.dumps(
                    {
                        "qa_id": pair.qa_id,
                        "grade": pair.grade,
                        "question": pair.question,
                        "answer": pair.answer,
                        "evidence": pair.evidence_span,
                        "q_type": pair.q_type,
                        "labels": {
                            "factuality": None,
                            "answerability": None,
                            "completeness": None,
                            "language": None,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    labeled = [p for p in chosen if (p.metadata or {}).get("human_labels")]
    return {
        "exported": len(chosen),
        "path": str(path),
        "annotated": bool(labeled),
        "reason": None if labeled else "未提供人工标签，不计算 Cohen's Kappa",
        "kappa": None,
    }


def _reset_pairs(pairs: list[QAPair]) -> list[QAPair]:
    copies = []
    for p in pairs:
        c = p.model_copy(deep=True)
        c.action = "pass"
        c.grade = None
        c.filter_trace = {}
        copies.append(c)
    return copies


def run_suite(
    suite_path: str | Path,
    out_dir: str | Path | None = None,
    fake: bool = False,
    skip_sft: bool = True,
    llm: LLMClient | None = None,
    base_model: str | None = None,
    devices: list[int] | None = None,
) -> dict[str, Any]:
    from ..adapters.zhixun import export_zhixun
    from .sft import DEFAULT_BASE

    suite = load_suite(suite_path)
    root = _repo_root()
    recipes_dir = Path(suite.get("recipes_dir") or root / "configs" / "recipes")
    if not recipes_dir.is_absolute():
        recipes_dir = (root / recipes_dir).resolve()
    input_path = Path(suite.get("input") or root / "fixtures" / "sample_manual.md")
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()
    heldout = Path(suite.get("heldout") or root / "fixtures" / "heldout.jsonl")
    if not heldout.is_absolute():
        heldout = (root / heldout).resolve()
    refusal_path = Path(suite.get("refusal") or root / "fixtures" / "refusal.jsonl")
    if not refusal_path.is_absolute():
        refusal_path = (root / refusal_path).resolve()
    gate = gate_heldout(heldout)
    if not gate["ok"]:
        return abort_missing_heldout(suite, heldout)
    out = Path(out_dir or root / "runs" / suite.get("name", "suite"))
    out.mkdir(parents=True, exist_ok=True)
    client = llm or (FakeLLM() if fake else LLMClient())
    student = base_model or DEFAULT_BASE
    write_run_meta(out, suite, heldout, student)
    llm_label = "fake" if isinstance(client, FakeLLM) else ("local" if type(client).__name__ == "LocalLLM" else "live")

    cache_pairs: dict[str, list[QAPair]] = {}
    cache_kept: dict[str, list[QAPair]] = {}
    cache_questions: dict[str, list] = {}
    sft_done: dict[str, dict[str, Any]] = {}
    adapters: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    notes = list(suite.get("notes") or [
        "语料仅为 fixtures/sample_manual.md，规模小于设计稿的 500 chunk / 1 万条。",
        "教师为 192.168.4.110:4000 的 qwen3.8-27b，API 费用记 0 美元，并报告等价 token。学生基座为本地 Qwen2.5-7B-Instruct。",
        "E5 无人工标签，不计算 Cohen's Kappa。事实遵循使用 NLI 与证据子串，未接入 RAGAS。",
    ])
    if gate["heldout_status"] == "empty":
        notes.append("主测试清单存在但没有可评分题目。允许训练和训练原题探针，主评测不加载模型，delta_f1 为空。")
    if not fake and devices and len(devices) >= 2:
        from .parallel import run_parallel

        return run_parallel(
            suite=suite,
            out=out,
            recipes_dir=recipes_dir,
            input_path=input_path,
            heldout=heldout,
            refusal_path=refusal_path,
            model_path=student,
            student=student,
            devices=list(devices),
            skip_sft=skip_sft,
            notes=notes,
            client_config={
                "api_key": getattr(client, "api_key", None),
                "base_url": getattr(client, "base_url", None),
                "default_model": getattr(client, "default_model", None),
                "timeout": getattr(client, "timeout", None),
            },
        )
    if not fake and devices and len(devices) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        notes.append(f"单卡运行，CUDA_VISIBLE_DEVICES={devices[0]}。")

    def remember(entry: dict[str, Any], result: PipelineResult) -> None:
        key = entry.get("cache_as")
        if key:
            cache_pairs[key] = list(result.raw_pairs or (list(result.pairs) + list(result.rejected)))
            cache_kept[key] = list(result.pairs)
        qkey = entry.get("cache_questions")
        if qkey:
            cache_questions[qkey] = list(result.questions)

    def maybe_sft(entry: dict[str, Any], pairs: list[QAPair], exp_dir: Path, recipe: Recipe | None = None) -> dict[str, Any]:
        if skip_sft or not entry.get("sft"):
            return {"skipped": True, "reason": "skip_sft" if skip_sft else "not_requested"}
        if hasattr(client, "release"):
            client.release()
        from .scoring import cache_signature

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
                "recipe": recipe.dump() if recipe else {},
                "model": student,
                "seed": recipe.seed if recipe else None,
            }
        )
        if not pairs:
            return {"skipped": True, "reason": "empty_train"}
        if not Path(heldout).is_file():
            raise FileNotFoundError(f"missing_heldout: {heldout}")
        if sig in sft_done:
            return {**sft_done[sig], "reused": True}
        report = evaluate_sft(pairs, heldout, exp_dir / "sft", llm=client, base_model=student, skip_sft=False)
        sft_done[sig] = report
        adapter = (report.get("train") or {}).get("adapter")
        if adapter:
            adapters[entry["id"]] = adapter
        sheet = (exp_dir / "sft" / "answers.md")
        if entry.get("answer_sheet") and sheet.is_file():
            target = exp_dir.parent / "answers.md"
            target.write_text(sheet.read_text(encoding="utf-8"), encoding="utf-8")
        return report

    def finalize(entry: dict[str, Any], result: PipelineResult, recipe: Recipe | None, exp_dir: Path, note: str = "") -> None:
        grades = entry.get("grades")
        full_kept = list(result.pairs)
        remember(entry, result)
        if grades:
            result.pairs = _apply_grades(full_kept, grades)
        save_result(result, exp_dir)
        export_zhixun(result.pairs, exp_dir / "zhixun.jsonl")
        metrics = compute_metrics(result)
        row = {
            "id": entry["id"],
            "purpose": entry.get("purpose", ""),
            "recipe": recipe.name if recipe else entry.get("recipe", ""),
            "recipe_snapshot": recipe_snapshot(recipe) if recipe else {},
            "metrics": metrics,
            "sft": maybe_sft(entry, result.pairs, exp_dir, recipe),
            "note": note,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)

    for entry in suite.get("experiments") or []:
        exp_id = entry["id"]
        kind = entry.get("kind") or "pipeline"
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(client, "reset_usage"):
            client.reset_usage()

        if kind == "skipped":
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {},
                    "sft": {},
                    "note": entry.get("reason") or "未执行",
                }
            )
            continue

        if kind == "annotate":
            source = cache_kept.get(entry.get("from_kept") or "") or []
            ann = _export_annotation(source, exp_dir / "annotation.jsonl", entry.get("grades"))
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {"kept": ann["exported"]},
                    "sft": {},
                    "note": ann.get("reason") or "",
                    "annotation": ann,
                }
            )
            continue

        if kind == "cost":
            source_id = entry.get("source")
            found = next((row for row in rows if row["id"] == source_id), None)
            metrics = dict((found or {}).get("metrics") or {})
            metrics["api_cost_usd"] = metrics.get("estimated_cost_usd")
            metrics["cost_note"] = "教师 qwen3.8-27b 为内网接口，API 费用按 0 计，tokens_per_10k_kept 为等价 token。"
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": metrics,
                    "sft": {},
                    "note": metrics["cost_note"],
                }
            )
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        if kind == "refusal":
            report: dict[str, Any] = {"skipped": skip_sft, "arms": {}}
            if skip_sft or not refusal_path.is_file():
                report["reason"] = "skip_sft" if skip_sft else f"missing {refusal_path}"
            else:
                if hasattr(client, "release"):
                    client.release()
                from .refusal import evaluate_refusal, load_refusal_set

                refusal_rows = load_refusal_set(refusal_path)
                for arm in entry.get("compare") or []:
                    adapter = adapters.get(arm)
                    if not adapter:
                        report["arms"][arm] = {"skipped": True, "reason": "no_adapter"}
                        continue
                    report["arms"][arm] = evaluate_refusal(refusal_rows, student, adapter)
            (exp_dir / "refusal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {},
                    "sft": {},
                    "note": json.dumps(report.get("arms") or report.get("reason"), ensure_ascii=False),
                    "refusal": report,
                }
            )
            continue

        if kind == "subset":
            source = cache_kept.get(entry.get("from_kept") or "") or []
            pairs = _apply_grades(source, entry.get("grades"))
            result = _empty_result(entry.get("name") or exp_id, pairs)
            finalize(entry, result, None, exp_dir)
            continue

        recipe = _recipe_from_entry(entry, recipes_dir)
        pipe = Pipeline(recipe, llm=client)
        q_source = entry.get("from_questions")
        accepted_source = entry.get("from_accepted")
        kept_source = entry.get("from_kept")
        raw_source = entry.get("from_cache")
        if q_source:
            if q_source not in cache_questions:
                rows.append(
                    {
                        "id": exp_id,
                        "purpose": entry.get("purpose", ""),
                        "recipe": recipe.name,
                        "recipe_snapshot": recipe_snapshot(recipe),
                        "metrics": {},
                        "sft": {"skipped": True, "reason": "missing_cache"},
                        "note": f"缺少必需问题缓存 {q_source}，已停止而不是重新生成",
                    }
                )
                continue
            result = pipe.distill_from_questions(cache_questions[q_source])
            result.recipe_name = recipe.name
        elif accepted_source:
            if accepted_source not in cache_kept:
                rows.append(
                    {
                        "id": exp_id,
                        "purpose": entry.get("purpose", ""),
                        "recipe": recipe.name,
                        "recipe_snapshot": recipe_snapshot(recipe),
                        "metrics": {},
                        "sft": {"skipped": True, "reason": "missing_cache"},
                        "note": f"缺少共同合格池 {accepted_source}",
                    }
                )
                continue
            result = pipe.select_only(cache_kept[accepted_source])
            result.recipe_name = recipe.name
        elif kept_source and kept_source in cache_kept:
            pairs = _apply_grades(cache_kept[kept_source], entry.get("grades"))
            result = _empty_result(recipe.name, pairs)
            entry = {**entry, "grades": None}
        elif raw_source:
            if raw_source not in cache_pairs:
                rows.append(
                    {
                        "id": exp_id,
                        "purpose": entry.get("purpose", ""),
                        "recipe": recipe.name,
                        "recipe_snapshot": recipe_snapshot(recipe),
                        "metrics": {},
                        "sft": {"skipped": True, "reason": "missing_cache"},
                        "note": f"缺少必需原始缓存 {raw_source}",
                    }
                )
                continue
            result = pipe.refilter(_reset_pairs(cache_pairs[raw_source]), llm=client)
            result.recipe_name = recipe.name
        else:
            result = pipe.run(input_path)
        finalize(entry, result, recipe, exp_dir)

    payload = {
        "suite": suite.get("name"),
        "input": str(input_path),
        "llm": llm_label,
        "teacher_model": getattr(client, "default_model", ""),
        "base_model": student,
        "limitations": notes,
        "experiments": rows,
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _render_report(suite.get("name", "suite"), rows, notes)
    (out / "report.md").write_text(report, encoding="utf-8")
    from .sft import assemble_answer_book

    assemble_answer_book(out, list(suite.get("experiments") or []))
    write_run_meta(out, suite, heldout, student)
    return payload
