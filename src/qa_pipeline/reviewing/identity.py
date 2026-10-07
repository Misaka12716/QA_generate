"""审核对象与缓存身份。客户端声明的哈希不能覆盖这里的重算结果。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..experiments.scoring import cache_signature

PROMPTS = Path(__file__).resolve().parent / "prompts"
INFERENCE_CONFIG = {"temperature": 0, "response_format": "json_object", "max_tokens": 2000}


def prompt_for(subject_type: str) -> tuple[str, str]:
    name = "answer_v1.txt" if subject_type == "prediction" else "gold_v1.txt"
    path = PROMPTS / name
    text = path.read_text(encoding="utf-8")
    return name, text


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def subject_hash(subject: dict[str, Any]) -> str:
    return cache_signature(
        {
            "subject_type": subject.get("subject_type"),
            "subject_id": subject.get("subject_id"),
            "subject_version": subject.get("subject_version"),
            "question": subject.get("question") or "",
            "student_context": subject.get("student_context") or "",
            "answer": subject.get("answer") or "",
            "expected_action": subject.get("expected_action") or "",
            "condition_id": subject.get("condition_id") or "",
            "task_mode": subject.get("task_mode") or subject.get("goal") or "",
            "messages": subject.get("messages"),
        }
    )


def model_input_hash(subject: dict[str, Any]) -> str:
    return cache_signature(
        {
            "question": subject.get("question") or "",
            "student_context": subject.get("student_context") or "",
            "messages": subject.get("messages"),
            "task_mode": subject.get("task_mode") or subject.get("goal") or "",
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
    if subject.get("subject_type") == "prediction":
        return ""
    return cache_signature(
        {
            "answer": subject.get("answer") or "",
            "required_points": subject.get("required_points") or [],
            "expected_action": subject.get("expected_action") or "",
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
) -> str:
    return cache_signature(
        {
            "subject_hash": subject_hash_value,
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
