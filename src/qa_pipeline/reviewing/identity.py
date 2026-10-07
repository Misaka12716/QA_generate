"""审核对象与缓存身份。客户端声明的哈希不能覆盖这里的重算结果。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..experiments.scoring import cache_signature
from ..task_mode import TaskModeError, canonical_task_mode

PROMPTS = Path(__file__).resolve().parent / "prompts"
INFERENCE_CONFIG = {"temperature": 0, "response_format": "json_object", "max_tokens": 2000}
REVIEW_IDENTITY_SCHEMA = "review-identity-v3"


def prompt_for(subject_type: str) -> tuple[str, str]:
    name = "answer_v1.txt" if subject_type == "prediction" else "gold_v1.txt"
    path = PROMPTS / name
    text = path.read_text(encoding="utf-8")
    return name, text


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_body(text: Any) -> str:
    return " ".join(str(text or "").split())


def canonical_answer_text(subject: dict[str, Any]) -> str:
    """预测正文只认 answer_text。金标没有该字段时才回退到 answer。"""
    if subject.get("subject_type") == "prediction" or "answer_text" in subject:
        if subject.get("answer_text") is not None:
            return normalize_body(subject.get("answer_text"))
    return normalize_body(subject.get("answer"))


def task_mode_field(subject: dict[str, Any]) -> str:
    raw = subject.get("task_mode") or subject.get("goal") or ""
    if not str(raw).strip():
        return ""
    try:
        return canonical_task_mode(str(raw))
    except TaskModeError:
        return str(raw)


def review_mode_of(subject: dict[str, Any]) -> str:
    declared = str(subject.get("review_mode") or "")
    if declared in {"reference_based", "reference_free"}:
        return declared
    if subject.get("subject_type") == "prediction":
        if normalize_body(subject.get("reference_answer")) or list(subject.get("required_points") or []):
            return "reference_based"
        return "reference_free"
    if canonical_answer_text(subject) or list(subject.get("required_points") or []):
        return "reference_based"
    return "reference_free"


def reference_answer_of(subject: dict[str, Any]) -> str:
    if subject.get("subject_type") == "prediction":
        return normalize_body(subject.get("reference_answer"))
    return canonical_answer_text(subject)


def subject_hash(subject: dict[str, Any]) -> str:
    """审核对象身份。回答正文、参考要求和任务模式都在其中。"""
    return cache_signature(
        {
            "schema": REVIEW_IDENTITY_SCHEMA,
            "subject_type": subject.get("subject_type"),
            "subject_id": subject.get("subject_id"),
            "subject_version": subject.get("subject_version"),
            "question": subject.get("question") or "",
            "student_context": subject.get("student_context") or "",
            "answer_text": canonical_answer_text(subject),
            "reference_answer": reference_answer_of(subject),
            "required_points": list(subject.get("required_points") or []),
            "answer_point_specs": subject.get("answer_point_specs") or [],
            "response_contract": subject.get("response_contract") or {},
            "unavailable_points": list(subject.get("unavailable_points") or []),
            "expected_action": subject.get("expected_action") or "",
            "condition_id": subject.get("condition_id") or "",
            "task_mode": task_mode_field(subject),
            "review_mode": review_mode_of(subject),
            "messages": subject.get("messages"),
        }
    )


def model_input_hash(subject: dict[str, Any]) -> str:
    """学生实际看到的输入。不含回答、参考答案和评分要点。"""
    return cache_signature(
        {
            "schema": REVIEW_IDENTITY_SCHEMA,
            "question": subject.get("question") or "",
            "student_context": subject.get("student_context") or "",
            "messages": subject.get("messages"),
            "task_mode": task_mode_field(subject),
        }
    )


def evidence_snapshot_hash(subject: dict[str, Any]) -> str:
    return cache_signature(
        {
            "student_context": subject.get("student_context") or "",
            "visible_evidence": subject.get("visible_evidence") or "",
        }
    )


def gold_hash(subject: dict[str, Any]) -> str:
    """参考标注身份。预测没有参考时为空，改参考不必重做学生推理。"""
    reference = reference_answer_of(subject)
    points = list(subject.get("required_points") or [])
    unavailable = list(subject.get("unavailable_points") or [])
    if subject.get("subject_type") == "prediction" and not reference and not points and not unavailable:
        return ""
    return cache_signature(
        {
            "schema": REVIEW_IDENTITY_SCHEMA,
            "reference_answer": reference,
            "required_points": points,
            "answer_point_specs": subject.get("answer_point_specs") or [],
            "response_contract": subject.get("response_contract") or {},
            "unavailable_points": unavailable,
            "expected_action": subject.get("expected_action") or "",
            "condition_id": subject.get("condition_id") or "",
        }
    )


def review_cache_key(
    *,
    subject_hash_value: str,
    review_kind: str,
    model_id: str,
    model_revision: str,
    prompt_digest: str,
    rubric_version: str,
    policy_version: str,
    evidence_hash: str,
    judge_role: str,
    reference_hash: str = "",
) -> str:
    return cache_signature(
        {
            "schema": REVIEW_IDENTITY_SCHEMA,
            "subject_hash": subject_hash_value,
            "reference_hash": reference_hash,
            "review_kind": review_kind,
            "model_id": model_id,
            "model_revision": model_revision or "unknown",
            "prompt_hash": prompt_digest,
            "rubric_version": rubric_version,
            "policy_version": policy_version,
            "inference_config": INFERENCE_CONFIG,
            "evidence_snapshot_hash": evidence_hash,
            "judge_role": judge_role,
        }
    )


def inference_config_hash() -> str:
    return cache_signature(INFERENCE_CONFIG)
