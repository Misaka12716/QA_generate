"""已审核协议上的探索评测编排。未授权时不调用生成函数。"""

from __future__ import annotations

from typing import Any, Callable

from .adapter_eval import case_id_of
from .scoring import prediction_item_signature, score_item_signature
from ..reviewing.identity import subject_hash
from ..reviewing.policy import ReviewPolicy, release_block_reasons
from ..reviewing.schemas import ReviewAggregate
from ..reviewing.service import normalize_subject

GenerateFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def execute_reviewed_batch(
    *,
    cases: list[dict[str, Any]],
    aggregates: dict[str, ReviewAggregate],
    policy: ReviewPolicy,
    mode: str,
    allow_inference: bool,
    inference_authorized: bool,
    generate_fn: GenerateFn | None = None,
    predictions: list[dict[str, Any]] | None = None,
    base_id: str = "",
    adapter_id: str = "",
    template_id: str = "",
    infer_config: dict[str, Any] | None = None,
    scorer_version: str = "aux-rules-v3",
    result_class: str = "exploratory_teacher_reviewed",
    planned_n: int | None = None,
) -> dict[str, Any]:
    planned = len(cases) if planned_n is None else planned_n
    report = {
        "planned_n": planned,
        "case_n": len(cases),
        "executed": False,
        "model_loaded": False,
        "formal_main_metric": "not_executed",
        "result_class": "not_executed",
        "dropped_case_ids": [],
    }
    if planned != len(cases):
        report.update({"status": "blocked", "reason": "incomplete_group"})
        return report
    if mode == "formal":
        report.update({"status": "blocked", "reason": "formal_requires_human_gold"})
        return report
    if mode != "exploratory":
        report.update({"status": "blocked", "reason": "unknown_mode"})
        return report
    blocked = []
    for case in cases:
        case_id = case_id_of(case)
        aggregate = aggregates.get(case_id)
        current_hash = subject_hash(normalize_subject(case, "reviewed_eval"))
        sample = {
            "grade": "S",
            "action": "pass",
            "validation_subject_hash": current_hash,
        }
        reasons = release_block_reasons(sample, aggregate, policy)
        if aggregate is None or aggregate.subject_hash != current_hash:
            reasons.append("stale_hash")
        if aggregate is None or not aggregate.ready_for_exploratory_inference:
            reasons.append("teacher_review_not_ready")
        if aggregate is not None and aggregate.ready_for_formal_human_eval:
            reasons.append("formal_flag_not_allowed")
        if reasons:
            blocked.append({"case_id": case_id, "reasons": reasons})
    if blocked:
        report.update({"status": "blocked", "reason": "pending_review", "failures": blocked})
        return report
    infer = dict(infer_config or {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0, "top_p": 1.0})
    hits, misses, stale_scores = _split_cache(
        cases,
        predictions or [],
        base_id=base_id,
        adapter_id=adapter_id,
        template_id=template_id,
        infer_config=infer,
        scorer_version=scorer_version,
    )
    report["cache_hits"] = len(hits)
    report["cache_misses"] = len(misses)
    report["stale_score_n"] = len(stale_scores)
    if not allow_inference:
        report.update({"status": "not_started", "reason": "inference_not_requested"})
        return report
    if misses and not inference_authorized:
        report.update({"status": "blocked", "reason": "inference_budget_missing"})
        return report
    if not misses:
        report.update(
            {
                "status": "executed",
                "executed": True,
                "reason": "prediction_cache_reused",
                "result_class": result_class,
                "model_loaded": False,
            }
        )
        return report
    if generate_fn is None:
        report.update({"status": "not_started", "reason": "executor_missing"})
        return report
    outcome = generate_fn(misses)
    report.update(
        {
            "status": "executed" if outcome.get("executed") else str(outcome.get("status") or "not_started"),
            "executed": bool(outcome.get("executed")),
            "model_loaded": bool(outcome.get("model_loaded")),
            "reason": outcome.get("reason"),
            "result_class": result_class if outcome.get("executed") else "not_executed",
            "generated_n": outcome.get("generated_n"),
        }
    )
    return report


def _split_cache(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    base_id: str,
    adapter_id: str,
    template_id: str,
    infer_config: dict[str, Any],
    scorer_version: str,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    by_sig = {str(row.get("signature")): row for row in predictions if row.get("signature")}
    hits: list[str] = []
    misses: list[dict[str, Any]] = []
    stale_scores: list[str] = []
    for case in cases:
        matched = True
        for role, model_id, adapter in (("base", base_id, ""), ("adapter", base_id, adapter_id)):
            signature = prediction_item_signature(case, model_id, adapter, template_id, infer_config, role)
            row = by_sig.get(signature)
            if row is None:
                matched = False
                continue
            score_sig = score_item_signature(signature, case, scorer_version)
            if row.get("score_signature") not in {None, "", score_sig}:
                stale_scores.append(signature)
        if matched:
            hits.append(case_id_of(case))
        else:
            misses.append(case)
    return hits, misses, stale_scores
