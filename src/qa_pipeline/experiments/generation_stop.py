"""区分正常结束、上限截断、空响应和异常。没有 token 时不声称遇到了 EOS。"""

from __future__ import annotations

from typing import Any


def classify_generation_stop(
    *,
    completion_token_ids: list[int] | None = None,
    completion_tokens: int | None = None,
    max_new_tokens: int = 0,
    eos_token_id: int | None = None,
    stop_token_ids: list[int] | None = None,
    text: str = "",
    error: str = "",
) -> dict[str, Any]:
    if error:
        return {
            "finish_reason": "error",
            "finish_reason_evidence": "exception",
            "last_token_id": None,
            "last_token": None,
            "eos_matched": False,
            "stop_matched": False,
            "truncated": False,
            "completion_tokens": completion_tokens or 0,
            "hit_max_new_tokens": False,
        }
    count = int(completion_tokens if completion_tokens is not None else (0 if completion_token_ids is None else len(completion_token_ids)))
    if completion_token_ids is None:
        hit = bool(max_new_tokens) and count >= max_new_tokens
        if count <= 0 and not str(text or "").strip():
            reason = "empty"
        else:
            reason = "length" if hit else "stop"
        return {
            "finish_reason": reason,
            "finish_reason_evidence": "inferred_from_length",
            "last_token_id": None,
            "last_token": "not_recorded",
            "eos_matched": None,
            "stop_matched": None,
            "truncated": hit,
            "completion_tokens": count,
            "hit_max_new_tokens": hit,
        }
    last = int(completion_token_ids[-1]) if completion_token_ids else None
    if not completion_token_ids or (not str(text or "").strip() and count <= 0):
        return {
            "finish_reason": "empty",
            "finish_reason_evidence": "token_ids",
            "last_token_id": last,
            "last_token": last,
            "eos_matched": False,
            "stop_matched": False,
            "truncated": False,
            "completion_tokens": count,
            "hit_max_new_tokens": False,
        }
    eos_matched = eos_token_id is not None and last == int(eos_token_id)
    stop_ids = {int(item) for item in (stop_token_ids or [])}
    stop_matched = last in stop_ids if last is not None else False
    hit = bool(max_new_tokens) and count >= max_new_tokens and not eos_matched and not stop_matched
    if eos_matched:
        reason = "eos"
    elif stop_matched:
        reason = "stop"
    elif hit:
        reason = "length"
    elif not str(text or "").strip():
        reason = "empty"
    else:
        reason = "stop"
    return {
        "finish_reason": reason,
        "finish_reason_evidence": "token_ids",
        "last_token_id": last,
        "last_token": last,
        "eos_matched": eos_matched,
        "stop_matched": stop_matched,
        "truncated": reason == "length",
        "completion_tokens": count,
        "hit_max_new_tokens": bool(max_new_tokens) and count >= max_new_tokens,
    }
