"""调用教师并解析结构化审核。解析失败保持技术失败，不改写成内容拒绝。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ..llm import FakeLLM, LLMClient, LLMResponse
from .schemas import ClaimRecord, DimensionScores, ReviewRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:16]


def teacher_credentials_configured() -> dict[str, Any]:
    """只报告是否配置，不返回密钥或内网地址。"""
    import os

    from ..llm import _read_gpt_api_file

    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QA_PIPELINE_API_KEY") or ""
    file_key, _base = _read_gpt_api_file()
    configured = bool(env_key or file_key) and (env_key or file_key) != "EMPTY"
    endpoint = "environment" if (os.environ.get("OPENAI_BASE_URL") or os.environ.get("QA_PIPELINE_BASE_URL")) else "builtin_default_not_verified"
    return {
        "configured": configured,
        "env_key_present": bool(env_key),
        "file_key_present": bool(file_key),
        "endpoint_source": endpoint,
        "pricing_version": "unknown",
        "currency": "unknown",
        "estimated_cost": None,
    }


def build_messages(subject: dict[str, Any], *, prompt_text: str) -> list[dict[str, str]]:
    payload = {
        "subject_type": subject.get("subject_type"),
        "question": subject.get("question") or "",
        "student_context": subject.get("student_context") or "",
        "answer": subject.get("answer") or "",
        "answer_text": subject.get("answer_text") or "",
        "expected_action": subject.get("expected_action") or "",
        "condition_id": subject.get("condition_id") or "",
        "task_mode": subject.get("task_mode") or subject.get("goal") or "rag_grounded",
        "required_points": subject.get("required_points") or [],
        "unavailable_points": subject.get("unavailable_points") or [],
    }
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "<subject>\n" + json.dumps(payload, ensure_ascii=False) + "\n</subject>"},
    ]


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def parse_teacher_payload(data: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(data, dict):
        return None, "schema_invalid"
    decision = data.get("decision")
    if decision not in {"accept", "revise", "reject", "abstain"}:
        return None, "schema_invalid"
    try:
        dimensions = DimensionScores.model_validate(data.get("dimensions") or {})
        raw_claims = data.get("claims") or []
        if not isinstance(raw_claims, list):
            return None, "schema_invalid"
        claims = []
        for item in raw_claims:
            if not isinstance(item, dict):
                return None, "schema_invalid"
            normalized = dict(item)
            for ref_key in ("visible_evidence_refs", "source_evidence_refs"):
                refs = _string_list(normalized.get(ref_key))
                if refs is None:
                    return None, "schema_invalid"
                normalized[ref_key] = refs
            claims.append(ClaimRecord.model_validate(normalized))
    except ValidationError:
        return None, "schema_invalid"
    patch = data.get("suggested_patch")
    if isinstance(patch, str):
        patch = {"note": patch} if patch.strip() else None
    elif patch is not None and not isinstance(patch, dict):
        return None, "schema_invalid"
    contradictions = _string_list(data.get("contradictions"))
    reason_codes = _string_list(data.get("reason_codes"))
    if contradictions is None or reason_codes is None:
        return None, "schema_invalid"
    return {
        "decision": decision,
        "dimensions": dimensions,
        "claims": claims,
        "answerable_points": [str(item) for item in data.get("answerable_points") or []],
        "unavailable_points": [str(item) for item in data.get("unavailable_points") or []],
        "numeric_bindings": list(data.get("numeric_bindings") or []),
        "required_conditions": [str(item) for item in data.get("required_conditions") or []],
        "contradictions": contradictions,
        "reason_codes": reason_codes,
        "evidence_summary": str(data.get("evidence_summary") or ""),
        "suggested_patch": patch,
        "confidence": data.get("confidence") if isinstance(data.get("confidence"), (int, float)) else None,
    }, None


def client_for(judge: dict[str, Any], *, fake: bool) -> LLMClient:
    if fake or judge.get("provider_id") == "fake":
        client = FakeLLM()
        client.script = [dict(item) for item in judge.get("script") or []]
        client.default_model = str(judge.get("model_id") or "fake-teacher")
        return client
    api_key = judge.get("api_key")
    if judge.get("authentication") == "models_list_no_key" and not api_key:
        api_key = "EMPTY"
    return LLMClient(
        api_key=api_key,
        base_url=judge.get("base_url"),
        default_model=str(judge.get("model_id") or ""),
        timeout=float(judge.get("timeout") or 60),
    )


def call_teacher(client: LLMClient, messages: list[dict[str, str]], model_id: str) -> LLMResponse:
    return client.chat_json(messages, model=model_id, temperature=0, max_tokens=2000)


def make_record(
    *,
    subject: dict[str, Any],
    policy: Any,
    judge: dict[str, Any],
    review_set_id: str,
    prompt_digest: str,
    inference_digest: str,
    decision: str,
    execution_status: str,
    reason_codes: list[str],
    parsed: dict[str, Any] | None = None,
    attempt: int = 1,
    cache_key: str = "",
    request_id: str = "",
    usage: dict[str, Any] | None = None,
) -> ReviewRecord:
    parsed = parsed or {}
    from .identity import evidence_snapshot_hash, gold_hash, model_input_hash, subject_hash

    gold = gold_hash(subject)
    return ReviewRecord(
        review_id=new_id("rev_"),
        batch_id=str(subject.get("batch_id") or ""),
        subject_type=subject["subject_type"],
        subject_id=str(subject["subject_id"]),
        subject_version=str(subject.get("subject_version") or "1"),
        review_source="teacher",
        review_mode=policy.review_mode,
        review_policy_id=policy.review_policy_id,
        policy_version=policy.policy_version,
        subject_hash=subject_hash(subject),
        model_input_hash=model_input_hash(subject),
        evidence_snapshot_hash=evidence_snapshot_hash(subject),
        gold_hash=gold or None,
        provider_id=str(judge.get("provider_id") or ""),
        model_id=str(judge.get("model_id") or ""),
        model_revision=str(judge.get("model_revision") or "unknown"),
        model_identity_verified=bool(judge.get("model_identity_verified")),
        prompt_hash=prompt_digest,
        rubric_version=policy.rubric_version,
        inference_config_hash=inference_digest,
        decision=decision,  # type: ignore[arg-type]
        execution_status=execution_status,  # type: ignore[arg-type]
        dimensions=parsed.get("dimensions") or DimensionScores(),
        claims=parsed.get("claims") or [],
        answerable_points=parsed.get("answerable_points") or [],
        unavailable_points=parsed.get("unavailable_points") or [],
        numeric_bindings=parsed.get("numeric_bindings") or [],
        required_conditions=parsed.get("required_conditions") or [],
        contradictions=parsed.get("contradictions") or [],
        reason_codes=list(reason_codes),
        evidence_summary=str(parsed.get("evidence_summary") or ""),
        suggested_patch=parsed.get("suggested_patch"),
        confidence=parsed.get("confidence"),
        confidence_calibrated=False,
        request_id=request_id or new_id("req_"),
        attempt=attempt,
        usage=usage or {},
        estimated_cost=None,
        currency="unknown",
        pricing_version="unknown",
        created_at=utc_now(),
        raw_response_artifact_id=None,
        review_set_id=review_set_id,
        judge_role=str(judge.get("judge_role") or "gold_reviewer"),
        independence_level=str(judge.get("independence_level") or "unverified_names"),
        cache_key=cache_key,
    )


def technical_reason(status: str) -> str:
    if status in {"transport_failed", "timeout"}:
        return "timeout" if "timeout" in status or status == "timeout" else "transport_failed"
    if status == "parse_failed":
        return "invalid_json"
    if status == "schema_failed":
        return "schema_invalid"
    return status or "technical_failed"
