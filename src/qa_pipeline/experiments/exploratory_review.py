"""探索性单模型审核的题目装配。不写人工审核表，也不把模型角色放进教师请求。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _mode(case: dict[str, Any]) -> str:
    from ..task_mode import canonical_task_mode

    return canonical_task_mode(str(case.get("task_mode") or case.get("goal") or "rag_grounded"))


def gold_subject(case: dict[str, Any], batch_id: str) -> dict[str, Any]:
    answer = case.get("candidate_answer") if case.get("candidate_answer") is not None else (case.get("answer") or "")
    return {
        "subject_type": "protocol_case",
        "subject_id": str(case.get("case_id") or case.get("subject_id") or ""),
        "subject_version": "1",
        "batch_id": batch_id,
        "question": case.get("question") or "",
        "student_context": case.get("student_context") if case.get("student_context") is not None else (case.get("context") or ""),
        "answer": answer,
        "expected_action": case.get("expected_action") or "",
        "condition_id": case.get("condition_id") or "",
        "required_points": list(case.get("required_points") or []),
        "unavailable_points": list(case.get("unavailable_points") or []),
        "task_mode": _mode(case),
        "review_mode": "reference_based" if str(answer or "").strip() or case.get("required_points") else "reference_free",
        "messages": case.get("messages"),
        "judge_source_context": case.get("judge_source_context") or "",
        "risk": "high",
    }


def prediction_subject(case: dict[str, Any], role: str, batch_id: str) -> dict[str, Any]:
    slot = (case.get("answers") or {}).get(role) or {}
    reference = case.get("candidate_answer") if case.get("candidate_answer") is not None else (case.get("answer") or "")
    points = list(case.get("required_points") or [])
    return {
        "subject_type": "prediction",
        "subject_id": str(slot.get("prediction_subject_id") or f"{case.get('case_id')}::{role}"),
        "subject_version": "1",
        "batch_id": batch_id,
        "question": case.get("question") or "",
        "student_context": case.get("student_context") if case.get("student_context") is not None else (case.get("context") or ""),
        "answer_text": slot.get("text") or "",
        "reference_answer": reference,
        "expected_action": case.get("expected_action") or "",
        "condition_id": case.get("condition_id") or "",
        "required_points": points,
        "unavailable_points": list(case.get("unavailable_points") or []),
        "task_mode": _mode(case),
        "review_mode": "reference_based" if str(reference or "").strip() or points else "reference_free",
        "messages": case.get("messages"),
        "judge_source_context": case.get("judge_source_context") or "",
        "risk": "high",
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def config_fingerprint(base_url: str) -> str:
    return hashlib.sha256(base_url.encode("utf-8")).hexdigest()


def effective_config(
    *,
    base_url: str,
    model_id: str,
    max_calls: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    return {
        "service_id": "teacher-openai-compatible",
        "endpoint_fingerprint": config_fingerprint(base_url),
        "model_id": model_id,
        "model_revision": "unknown",
        "authentication": "models 列表无需密钥；聊天以小试为准",
        "max_calls": max_calls,
        "max_total_tokens": max_tokens,
        "concurrency": concurrency,
        "max_technical_retries": 1,
        "estimated_cost": None,
        "currency": "unknown",
        "pricing_version": "unknown",
    }


def pilot_usable(out_dir: str | Path) -> tuple[bool, str]:
    """小试必须解析出状态和用量。401 或无法解析时停止后续调用。"""
    root = Path(out_dir)
    records = []
    path = root / "review_records.jsonl"
    if not path.is_file():
        return False, "missing_records"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if any("unauthorized" in (row.get("reason_codes") or []) for row in records):
        return False, "unauthorized"
    succeeded = [row for row in records if row.get("execution_status") == "succeeded"]
    if not succeeded:
        return False, "no_parsed_status"
    usage_path = root / "usage.jsonl"
    if not usage_path.is_file():
        return False, "missing_usage"
    counted = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counted = [row for row in counted if row.get("counted_call")]
    if not counted or any(not isinstance(row.get("prompt_tokens"), int) for row in counted):
        return False, "missing_token_usage"
    return True, "ok"
