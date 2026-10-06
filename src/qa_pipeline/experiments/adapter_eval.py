"""评测已经训好的 adapter。不导出训练集，也不调用 train_lora。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .devices import select_trainable_gpus
from .scoring import (
    SCORER_V1,
    SCORER_V2,
    aggregate_layered,
    align_prediction_groups,
    prediction_item_signature,
    score_task,
    score_task_v2,
)
from .sft import eval_messages

EXPLORATORY_DISCLAIMER = "未完成人工审核，仅供探索"
PROTECTED_PROTOCOL_NAMES = frozenset({"heldout_protocol.jsonl", "heldout.jsonl"})
REVIEWED_STATUS = frozenset({"agreed", "dual_agreed"})
DEFAULT_INFER = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "dtype": "bfloat16",
}
CARRY_FIELDS = (
    "family_id",
    "source_family_id",
    "source_hash",
    "dataset_version",
    "split",
    "task_mode",
    "evidence_state",
    "expected_action",
    "stratum",
    "source",
    "generic_family_id",
)


class ProtocolError(Exception):
    """协议不合法。应在加载模型之前抛出。"""


def case_id_of(case: dict[str, Any]) -> str:
    return str(case.get("case_id") or case.get("id") or "")


def case_question(case: dict[str, Any]) -> str:
    if str(case.get("question") or "").strip():
        return str(case["question"]).strip()
    for message in case.get("messages") or []:
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def review_content_hash(case: dict[str, Any]) -> str:
    from .scoring import cache_signature

    payload = {
        "case_id": case_id_of(case),
        "question": case.get("question"),
        "context": case.get("context"),
        "messages": case.get("messages"),
        "answer": case.get("answer"),
        "answer_points": case.get("answer_points"),
        "expected_action": case.get("expected_action"),
        "evidence_state": case.get("evidence_state"),
        "source_family_id": case.get("source_family_id"),
        "family_id": case.get("family_id"),
    }
    return cache_signature(payload)


def validate_protocol(cases: Any, mode: str) -> dict[str, Any]:
    if mode not in {"formal", "exploratory"}:
        raise ProtocolError(f"unknown_mode:{mode}")
    if not isinstance(cases, list):
        raise ProtocolError("invalid_object")
    failures = []
    eligible = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        reasons: list[str] = []
        if not isinstance(case, dict):
            failures.append({"id": index, "reasons": ["invalid_object"]})
            continue
        case_id = case_id_of(case)
        if not case_id:
            reasons.append("missing_id")
        elif case_id in seen:
            reasons.append("duplicate_id")
        else:
            seen.add(case_id)
        if not case_question(case):
            reasons.append("empty_question")
        if mode == "formal":
            if not str(case.get("answer") or "").strip() and not (case.get("answer_points") or []):
                reasons.append("missing_gold")
            if case.get("review_status") not in REVIEWED_STATUS:
                reasons.append("unreviewed")
            digest = str(case.get("review_content_hash") or "")
            if not digest:
                reasons.append("missing_review_hash")
            elif digest != review_content_hash(case):
                reasons.append("review_hash_mismatch")
            if not (case.get("source_family_id") or case.get("source")):
                reasons.append("missing_source")
            if not case.get("expected_action"):
                reasons.append("missing_expected_action")
            if case.get("source_overlap_train"):
                reasons.append("source_overlap")
        if reasons:
            failures.append({"id": case_id or index, "reasons": reasons})
        else:
            eligible.append(case)
    return {"mode": mode, "eligible": eligible, "failures": failures, "planned_n": len(cases)}


def assert_candidate_destination(path: str | Path, protected: list[str | Path] | None = None) -> None:
    dest = Path(path)
    if dest.name in PROTECTED_PROTOCOL_NAMES:
        raise ProtocolError("refuse_overwrite_formal_protocol")
    resolved = dest.resolve() if dest.exists() else dest.absolute()
    for item in protected or []:
        guard = Path(item)
        if guard.exists() and guard.resolve() == resolved:
            raise ProtocolError("refuse_overwrite_formal_protocol")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    file = Path(path)
    if not file.is_file():
        return rows
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _cache_map(path: Path) -> dict[str, dict[str, Any]]:
    found = {}
    for row in read_jsonl(path):
        signature = str(row.get("signature") or "")
        if signature:
            found[signature] = row
    return found


def _carry(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case.get(key) for key in CARRY_FIELDS if case.get(key) not in {None, ""}}


def _score_one(prediction: str, case: dict[str, Any], scorer_version: str) -> dict[str, Any]:
    if scorer_version == SCORER_V1:
        scored = score_task(prediction, case)
        scored["scorer_version"] = SCORER_V1
        scored["semantic_accuracy_verified"] = False
        return scored
    if scorer_version == SCORER_V2:
        return score_task_v2(prediction, case)
    raise ProtocolError(f"unknown_scorer:{scorer_version}")


def _not_executed(reason: str, mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {
        "status": "not_executed",
        "executed": False,
        "reason": reason,
        "mode": mode,
        "result_class": "not_executed",
        "enters_formal_metric": False,
        "formal_main_metric": "not_executed",
        "formal_gold": False,
        "delta_f1": None,
    }
    if mode == "exploratory":
        report["disclaimer"] = EXPLORATORY_DISCLAIMER
    if extra:
        report.update(extra)
    return report


def _apply_scores(
    alignment: dict[str, Any],
    scorer_version: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for item in alignment["aligned"]:
        case = item["case"]
        base = item["predictions"]["base"]
        tuned = item["predictions"]["adapter"]
        before = _score_one(str(base.get("text") or ""), case, scorer_version)
        after = _score_one(str(tuned.get("text") or ""), case, scorer_version)
        row = {
            "case_id": case_id_of(case),
            **_carry(case),
            "before": before,
            "after": after,
            "content_error": (not before.get("passed")) or (not after.get("passed")),
            "technical_failure": False,
        }
        if mode == "exploratory":
            row["disclaimer"] = EXPLORATORY_DISCLAIMER
            row["review_status"] = case.get("review_status") or "unreviewed"
            row["enters_formal_metric"] = False
        else:
            row["review_status"] = case.get("review_status")
            row["enters_formal_metric"] = True
        rows.append(row)
    return rows


def eval_adapter(
    *,
    base_model: str,
    base_id: str,
    adapter: str,
    adapter_id: str,
    protocol_path: str | Path,
    out_dir: str | Path,
    infer_config: dict[str, Any] | None = None,
    scorer_version: str = SCORER_V2,
    mode: str = "exploratory",
    template_id: str = "unspecified-template",
    generate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
    device: int | None = None,
    reuse_predictions: bool = True,
) -> dict[str, Any]:
    """推理与评分分开。正式协议没有有效金标时，在调用生成器之前停止。"""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = dict(DEFAULT_INFER)
    config.update(infer_config or {})
    cases = read_jsonl(protocol_path)
    checked = validate_protocol(cases, mode)
    (destination / "protocol_check.json").write_text(
        json.dumps(
            {"failures": checked["failures"], "eligible_n": len(checked["eligible"]), "planned_n": checked["planned_n"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if checked["failures"] and mode == "exploratory":
        report = _not_executed("invalid_protocol", mode, {"protocol_failures": checked["failures"]})
        _dump_report(destination, report)
        raise ProtocolError("invalid_protocol")
    if mode == "formal" and not checked["eligible"]:
        report = _not_executed(
            "no_valid_gold",
            mode,
            {"protocol_failures": checked["failures"], "main_metric": "not_executed"},
        )
        _dump_report(destination, report)
        raise ProtocolError("no_valid_gold")
    target = checked["eligible"]
    cache_path = destination / "prediction_cache.jsonl"
    cache = _cache_map(cache_path) if reuse_predictions else {}
    pending: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for case in target:
        messages = list(case.get("messages") or eval_messages(case))
        for role in ("base", "adapter"):
            signature = prediction_item_signature(
                case, base_id, adapter_id, template_id, config, role, model_input=messages
            )
            hit = cache.get(signature)
            if hit is not None:
                selected.append(hit)
                continue
            pending.append(
                {
                    "case_id": case_id_of(case),
                    "model_role": role,
                    "messages": messages,
                    "signature": signature,
                    "model_id": base_id,
                    "adapter_id": adapter_id if role == "adapter" else "",
                    **_carry(case),
                }
            )
    cache_hits = len(selected)
    if pending:
        if generate_fn is None:
            produced = generate_with_model(
                pending,
                base_model=base_model,
                adapter=adapter,
                infer_config=config,
                device=device,
            )
        else:
            produced = generate_fn(pending)
        if isinstance(produced, dict):
            report = _not_executed(str(produced.get("reason") or "generation_failed"), mode, {"cache_hits": cache_hits})
            _dump_report(destination, report)
            return report
        if len(produced) != len(pending):
            report = _not_executed(
                "prediction_count_mismatch",
                mode,
                {"expected": len(pending), "got": len(produced), "cache_hits": cache_hits},
            )
            _dump_report(destination, report)
            return report
        stored = []
        for item, spec in zip(produced, pending):
            if not isinstance(item, dict) or "text" not in item:
                report = _not_executed("invalid_generation_row", mode)
                _dump_report(destination, report)
                return report
            row = {
                **spec,
                "text": str(item.get("text") or ""),
                "prompt_tokens": item.get("prompt_tokens"),
                "completion_tokens": item.get("completion_tokens"),
                "finish_reason": item.get("finish_reason"),
                "truncated": bool(item.get("truncated")),
                "device": item.get("device"),
                "elapsed_sec": item.get("elapsed_sec"),
                "error": item.get("error"),
            }
            stored.append(row)
            cache[row["signature"]] = row
            selected.append(row)
        previous = [row for row in read_jsonl(cache_path) if row.get("signature") not in {item["signature"] for item in stored}]
        write_jsonl(cache_path, previous + stored)
    by_role: dict[str, list[dict[str, Any]]] = {"base": [], "adapter": []}
    for row in selected:
        role = str(row.get("model_role") or "")
        if role in by_role:
            by_role[role].append(row)
    write_jsonl(destination / "predictions.jsonl", selected)
    errored = [row for row in selected if row.get("error")]
    if errored:
        report = _not_executed(
            "generation_item_failed",
            mode,
            {
                "planned_n": len(target),
                "cache_hits": cache_hits,
                "failed_n": len({row.get("case_id") for row in errored}),
                "failures": [
                    {"case_id": row.get("case_id"), "model_role": row.get("model_role"), "reason": row.get("error"), "kind": "technical"}
                    for row in errored
                ],
            },
        )
        _dump_report(destination, report)
        return report
    alignment = align_prediction_groups(target, by_role)
    (destination / "alignment.json").write_text(json.dumps({k: v for k, v in alignment.items() if k != "aligned"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not alignment["ok"]:
        report = _not_executed(
            "alignment_failed",
            mode,
            {
                "planned_n": alignment["planned_n"],
                "generated_n": alignment["generated_n"],
                "scored_n": alignment["scored_n"],
                "failed_n": alignment["failed_n"],
                "failures": alignment["failures"],
                "cache_hits": cache_hits,
                "task_accuracy": None,
            },
        )
        _dump_report(destination, report)
        return report
    scored = _apply_scores(alignment, scorer_version, mode)
    write_jsonl(destination / "scores.jsonl", scored)
    passed_rows = [{**row, "passed": row["after"].get("passed")} for row in scored]
    layered = aggregate_layered(passed_rows)
    exploratory = mode == "exploratory"
    report = {
        "status": "executed" if not exploratory else "exploratory_executed",
        "executed": True,
        "reason": None,
        "mode": mode,
        "result_class": "exploratory_auxiliary" if exploratory else "formal",
        "disclaimer": EXPLORATORY_DISCLAIMER if exploratory else None,
        "enters_formal_metric": not exploratory,
        "formal_main_metric": "not_executed" if exploratory else "executed",
        "formal_gold": False if exploratory else True,
        "semantic_accuracy_verified": False,
        "scorer_version": scorer_version,
        "base_id": base_id,
        "adapter_id": adapter_id,
        "template_id": template_id,
        "infer_config": config,
        "planned_n": alignment["planned_n"],
        "generated_n": alignment["generated_n"],
        "scored_n": alignment["scored_n"],
        "failed_n": alignment["failed_n"],
        "cache_hits": cache_hits,
        "protocol_failures": checked["failures"],
        "content_error_n": sum(1 for row in scored if row["content_error"]),
        "technical_failure_n": 0,
        "layered": layered,
        "combined_generalization_score": None,
        "delta_f1": None,
        "judge": {"mean": None, "reason": "not_called", "note": "本轮不为填空值调用模型评委，评委结果也不是人工金标。"},
    }
    _dump_report(destination, report)
    return report


def _dump_report(destination: Path, report: dict[str, Any]) -> None:
    (destination / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def rescore_saved(
    *,
    protocol_path: str | Path,
    predictions_path: str | Path,
    out_dir: str | Path,
    scorer_version: str,
    mode: str,
) -> dict[str, Any]:
    """只读取已经落盘的预测。不加载模型。"""
    cases = read_jsonl(protocol_path)
    checked = validate_protocol(cases, mode)
    if mode == "formal" and not checked["eligible"]:
        raise ProtocolError("no_valid_gold")
    if mode == "exploratory" and checked["failures"]:
        raise ProtocolError("invalid_protocol")
    groups: dict[str, list[dict[str, Any]]] = {"base": [], "adapter": []}
    for row in read_jsonl(predictions_path):
        role = str(row.get("model_role") or "")
        if role in groups:
            groups[role].append(row)
    alignment = align_prediction_groups(checked["eligible"], groups)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not alignment["ok"]:
        report = _not_executed("alignment_failed", mode, {"failures": alignment["failures"], "planned_n": alignment["planned_n"], "failed_n": alignment["failed_n"], "scored_n": 0, "task_accuracy": None})
        _dump_report(destination, report)
        return report
    scored = _apply_scores(alignment, scorer_version, mode)
    write_jsonl(destination / "scores.jsonl", scored)
    report = {
        "status": "rescored",
        "executed": True,
        "mode": mode,
        "result_class": "exploratory_auxiliary" if mode == "exploratory" else "formal",
        "disclaimer": EXPLORATORY_DISCLAIMER if mode == "exploratory" else None,
        "enters_formal_metric": mode == "formal",
        "formal_gold": mode == "formal",
        "formal_main_metric": "not_executed" if mode == "exploratory" else "executed",
        "scorer_version": scorer_version,
        "planned_n": alignment["planned_n"],
        "scored_n": alignment["scored_n"],
        "failed_n": 0,
        "model_loaded": False,
    }
    _dump_report(destination, report)
    return report


def generate_with_model(
    pending: list[dict[str, Any]],
    *,
    base_model: str,
    adapter: str,
    infer_config: dict[str, Any],
    device: int | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """在一张空闲 GPU 上串行跑基座和 adapter。没有足够显存时不回退到 CPU。"""
    requested = None if device is None else [int(device)]
    chosen, error = select_trainable_gpus(requested)
    if not chosen:
        return {"skipped": True, "reason": error or "no_gpu"}
    index = chosen[0]
    try:
        import time

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"missing_infer_deps: {exc}"}
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "cuda_unavailable"}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": index},
    )
    model.eval()
    max_new = int(infer_config.get("max_new_tokens") or 512)
    do_sample = bool(infer_config.get("do_sample"))
    outputs: dict[tuple[str, str], dict[str, Any]] = {}

    def _run(spec: dict[str, Any], current) -> dict[str, Any]:
        started = time.perf_counter()
        messages = spec["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(current.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = current.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
            )
        new_tokens = generated[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(new_tokens.shape[0])
        truncated = completion_tokens >= max_new
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "length" if truncated else "stop",
            "truncated": truncated,
            "device": index,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }

    try:
        base_pending = [item for item in pending if item.get("model_role") == "base"]
        adapter_pending = [item for item in pending if item.get("model_role") == "adapter"]
        def _safe(spec, current):
            try:
                return _run(spec, current)
            except Exception as exc:
                return {
                    "text": "",
                    "error": str(exc),
                    "finish_reason": "error",
                    "truncated": False,
                    "device": index,
                }

        for spec in base_pending:
            outputs[(spec["case_id"], "base")] = _safe(spec, model)
        if adapter_pending:
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()
            for spec in adapter_pending:
                outputs[(spec["case_id"], "adapter")] = _safe(spec, model)
    except Exception as exc:
        return {"skipped": True, "reason": f"generation_exception: {exc}"}
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = []
    for spec in pending:
        key = (spec["case_id"], spec["model_role"])
        if key not in outputs:
            return {"skipped": True, "reason": "missing_generated_row"}
        rows.append(outputs[key])
    return rows
