"""把已有预测按身份对齐成结果页数据。不调用模型，不按行号拼接。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapter_eval import canonical_messages, case_id_of, resolve_infer
from .scoring import prediction_item_signature, score_task_v3

HISTORICAL_BANNER = "历史真实预测"
HISTORICAL_LIMITATION = (
    "旧 adapter 来自脏工作区，tokenizer 与基座配置不一致。"
    "两侧差异不能直接归因于微调。"
)
PENDING_TEACHER = "待自动评估"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _signature_formulas(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    declared = dict(report.get("infer_config") or {})
    formulas = {"declared": declared}
    resolved, _unknown = resolve_infer(declared)
    if resolved:
        formulas["effective"] = dict(resolved["effective"])
    return formulas


def match_prediction(
    case: dict[str, Any],
    row: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """重算签名并核对 messages。任一身份条件失败则不并入该题的回答。"""
    role = str(row.get("model_role") or "")
    reasons: list[str] = []
    if role not in {"base", "adapter"}:
        reasons.append("unknown_role")
    expected_messages = canonical_messages(case)
    if row.get("messages") != expected_messages:
        reasons.append("messages_mismatch")
    base_id = str(report.get("base_id") or "")
    if base_id and str(row.get("model_id") or "") != base_id:
        reasons.append("model_id_mismatch")
    adapter_id = str(report.get("adapter_id") or "")
    stored_adapter = str(row.get("adapter_id") or "")
    if role == "adapter" and adapter_id and stored_adapter != adapter_id:
        reasons.append("adapter_id_mismatch")
    if role == "base" and stored_adapter:
        reasons.append("base_has_adapter")
    template = row.get("template_id") or report.get("template_id")
    hits: list[str] = []
    if not reasons:
        for name, infer in _signature_formulas(report).items():
            signature = prediction_item_signature(
                case,
                base_id or str(row.get("model_id") or ""),
                adapter_id,
                template,
                infer,
                role,
                model_input=row.get("messages"),
            )
            if signature == row.get("signature"):
                hits.append(name)
        if not hits:
            reasons.append("signature_mismatch")
    return {
        "matched": not reasons,
        "reasons": reasons,
        "signature_formulas": hits,
        "template_id": template,
    }


def _rule_status(answers: dict[str, Any]) -> str:
    present = [slot for slot in answers.values() if slot.get("match_status") == "matched"]
    if len(present) < 2:
        return "unscored"
    flags = [slot.get("rule_passed") for slot in present]
    if any(flag is False for flag in flags):
        return "fail"
    if all(flag is True for flag in flags):
        return "pass"
    return "unscored"


def _answer_slot(row: dict[str, Any], case: dict[str, Any], source: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    scored = score_task_v3(str(row.get("text") or ""), case)
    return {
        "match_status": "matched",
        "model_role": row.get("model_role"),
        "text": row.get("text") or "",
        "signature": row.get("signature"),
        "signature_formulas": identity["signature_formulas"],
        "model_id": row.get("model_id"),
        "adapter_id": row.get("adapter_id") or "",
        "template_id": identity["template_id"],
        "source_run": source.get("label"),
        "prediction_subject_id": f"{case_id_of(case)}::{row.get('model_role')}",
        "rule_score": scored,
        "rule_passed": scored.get("passed"),
        "rule_role": "auxiliary",
        "teacher": None,
    }


def _empty_slot(role: str, case_id: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "match_status": "unmatched",
        "model_role": role,
        "text": None,
        "signature": None,
        "signature_formulas": [],
        "model_id": None,
        "adapter_id": None,
        "template_id": None,
        "source_run": None,
        "prediction_subject_id": f"{case_id}::{role}",
        "rule_score": None,
        "rule_passed": None,
        "rule_role": "auxiliary",
        "teacher": None,
        "unmatch_reasons": reasons or ["missing_prediction"],
    }


def assemble_cases(
    cases: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    block: str = "historical",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {case_id_of(case): case for case in cases}
    slots: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: dict[tuple[str, str], list[str]] = {}
    pending_reasons: dict[tuple[str, str], list[str]] = {}
    unmatched: list[dict[str, Any]] = []
    for source in sources:
        report = source.get("report") if isinstance(source.get("report"), dict) else {}
        for row in source.get("predictions") or []:
            case_id = str(row.get("case_id") or "")
            case = by_id.get(case_id)
            if case is None:
                continue
            identity = match_prediction(case, row, report)
            key = (case_id, str(row.get("model_role") or ""))
            if not identity["matched"]:
                unmatched.append({"case_id": case_id, "model_role": row.get("model_role"), "reasons": identity["reasons"], "source_run": source.get("label")})
                pending_reasons.setdefault(key, identity["reasons"])
                continue
            if key in rejected:
                unmatched.append({"case_id": case_id, "model_role": row.get("model_role"), "reasons": ["duplicate_match"], "source_run": source.get("label")})
                continue
            if key in slots:
                previous = slots.pop(key)
                rejected[key] = ["duplicate_match"]
                unmatched.append({"case_id": case_id, "model_role": row.get("model_role"), "reasons": ["duplicate_match"], "source_run": previous.get("source_run")})
                unmatched.append({"case_id": case_id, "model_role": row.get("model_role"), "reasons": ["duplicate_match"], "source_run": source.get("label")})
                continue
            pending_reasons.pop(key, None)
            slots[key] = _answer_slot(row, case, source, identity)
    built: list[dict[str, Any]] = []
    for case in cases:
        case_id = case_id_of(case)
        answers = {}
        for role in ("base", "adapter"):
            reasons = rejected.get((case_id, role)) or pending_reasons.get((case_id, role), [])
            answers[role] = slots.get((case_id, role)) or _empty_slot(role, case_id, reasons)
        built.append(
            {
                "case_id": case_id,
                "block": block,
                "question": case.get("question") or "",
                "student_context": case.get("context") or "",
                "messages": canonical_messages(case),
                "candidate_answer": case.get("answer") or "",
                "expected_action": case.get("expected_action") or "",
                "evidence_state": case.get("evidence_state") or "",
                "pattern": case.get("pattern") or "",
                "task_mode": case.get("task_mode") or "",
                "source_family_id": case.get("source_family_id") or "",
                "source": case.get("source") or "",
                "human_review_status": case.get("review_status") or "unreviewed",
                "core_question_id": case.get("core_question_id") or "",
                "answers": answers,
                "rule_status": _rule_status(answers),
                "teacher_gold": None,
            }
        )
    summary = {
        "protocol_n": len(cases),
        "matched_predictions": sum(1 for slot in slots.values()),
        "unmatched_predictions": len(unmatched),
        "unmatched": unmatched,
    }
    return built, summary


def write_view(
    out_dir: str | Path,
    cases: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "banner": meta.get("banner") or HISTORICAL_BANNER,
        "limitation": meta.get("limitation") or HISTORICAL_LIMITATION,
        "teacher_pending_label": PENDING_TEACHER,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    (dest / "view_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (dest / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def _teacher_view(record: dict[str, Any] | None, aggregate: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None and aggregate is None:
        return None
    dimensions = (record or {}).get("dimensions") or {}
    status = (aggregate or {}).get("teacher_review_status") or (record or {}).get("decision") or "unreviewed"
    return {
        "status": status,
        "label": status,
        "decision": (record or {}).get("decision"),
        "execution_status": (record or {}).get("execution_status"),
        "behavior_appropriate": dimensions.get("behavior_appropriate"),
        "required_points_complete": dimensions.get("required_points_complete"),
        "evidence_adequate": dimensions.get("evidence_adequate"),
        "no_unsupported_claims": dimensions.get("no_unsupported_claims"),
        "reason_codes": (record or {}).get("reason_codes") or [],
        "ready_for_formal_human_eval": bool((aggregate or {}).get("ready_for_formal_human_eval")),
        "readiness_scope": (aggregate or {}).get("readiness_scope"),
        "model_id": (record or {}).get("model_id"),
        "model_revision": (record or {}).get("model_revision"),
    }


def attach_reviews(cases: list[dict[str, Any]], records: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把已落盘的教师记录接到同一题上。没有记录时保持 null，不填 0。"""
    by_subject_records: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("execution_status") == "succeeded" or row.get("subject_id") not in by_subject_records:
            by_subject_records[str(row.get("subject_id") or "")] = row
    by_subject_agg = {str(row.get("subject_id") or ""): row for row in aggregates}
    attached: list[dict[str, Any]] = []
    for case in cases:
        item = json.loads(json.dumps(case))
        gold_id = str(item.get("case_id") or "")
        item["teacher_gold"] = _teacher_view(by_subject_records.get(gold_id), by_subject_agg.get(gold_id))
        for slot in (item.get("answers") or {}).values():
            subject_id = str(slot.get("prediction_subject_id") or "")
            slot["teacher"] = _teacher_view(by_subject_records.get(subject_id), by_subject_agg.get(subject_id))
        attached.append(item)
    return attached


def load_result_cases(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(root)
    cases = load_jsonl(directory / "cases.jsonl")
    meta_path = directory / "view_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    records = load_jsonl(directory / "review_records.jsonl") or load_jsonl(directory / "reviews.jsonl")
    aggregates = load_jsonl(directory / "review_aggregates.jsonl")
    if records or aggregates:
        cases = attach_reviews(cases, records, aggregates)
    return cases, meta


def select_cases(
    cases: list[dict[str, Any]],
    *,
    pattern: str = "",
    rule: str = "",
    review: str = "",
    block: str = "",
) -> list[dict[str, Any]]:
    selected = []
    for case in cases:
        if pattern and case.get("pattern") != pattern:
            continue
        if rule and case.get("rule_status") != rule:
            continue
        if review and case.get("human_review_status") != review:
            continue
        if block and case.get("block") != block:
            continue
        selected.append(case)
    return selected


def read_prediction_source(label: str, predictions: str | Path, report: str | Path) -> dict[str, Any]:
    return {
        "label": label,
        "predictions": load_jsonl(predictions),
        "report": json.loads(Path(report).read_text(encoding="utf-8")),
    }


def public_case(case: dict[str, Any], *, full: bool) -> dict[str, Any]:
    teacher = case.get("teacher_gold")
    summary = {
        "case_id": case.get("case_id"),
        "block": case.get("block"),
        "question": case.get("question"),
        "pattern": case.get("pattern"),
        "expected_action": case.get("expected_action"),
        "evidence_state": case.get("evidence_state"),
        "source_family_id": case.get("source_family_id"),
        "source": case.get("source"),
        "human_review_status": case.get("human_review_status"),
        "rule_status": case.get("rule_status"),
        "teacher_label": (teacher or {}).get("label") if teacher else PENDING_TEACHER,
        "teacher_status": (teacher or {}).get("status") if teacher else None,
        "base_matched": (case.get("answers") or {}).get("base", {}).get("match_status") == "matched",
        "adapter_matched": (case.get("answers") or {}).get("adapter", {}).get("match_status") == "matched",
        "base_rule_passed": (case.get("answers") or {}).get("base", {}).get("rule_passed"),
        "adapter_rule_passed": (case.get("answers") or {}).get("adapter", {}).get("rule_passed"),
    }
    if not full:
        return summary
    return {**case, "teacher_label": summary["teacher_label"]}
