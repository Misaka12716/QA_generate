"""审核政策。就绪标记只在这里根据记录计算。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .schemas import ReviewAggregate, ReviewRecord

CRITICAL = (
    "evidence_adequate",
    "factual_consistency",
    "behavior_appropriate",
    "no_unsupported_claims",
)


class ReviewPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_policy_id: str
    policy_version: str
    review_mode: Literal["teacher_only", "human", "hybrid"] = "teacher_only"
    require_dual: bool = True
    allow_single: bool = False
    rubric_version: str = "rubric-v1"
    prompt_version: str = "gold-v1"
    max_technical_retries: int = 1
    max_auto_revisions: int = 1


def normalize_judges(judges: list[dict]) -> list[dict]:
    """同名同 revision 不能靠配置字段伪装成不同模型族。"""
    normalized = [dict(item) for item in judges]
    identities = [(str(item.get("model_id") or ""), str(item.get("model_revision") or "unknown")) for item in normalized]
    for item, identity in zip(normalized, identities):
        revision = identity[1]
        declared = str(item.get("independence_level") or "")
        if identities.count(identity) > 1:
            item["independence_level"] = "same_model"
            item["model_identity_verified"] = False
        elif revision in {"", "unknown"} or declared != "distinct_family":
            item["independence_level"] = "unverified_names"
            item["model_identity_verified"] = False
        else:
            item["independence_level"] = "distinct_family"
            item["model_identity_verified"] = True
        item["model_revision"] = revision or "unknown"
    return normalized


def _critical_ok(record: ReviewRecord) -> bool:
    return all(getattr(record.dimensions, name) is True for name in CRITICAL)


def _evidence_ok(record: ReviewRecord, task_mode: str) -> bool:
    if record.contradictions:
        return False
    if record.decision != "accept":
        return True
    if not record.claims:
        return False
    for claim in record.claims:
        if claim.status == "contradicted":
            return False
        if claim.status != "supported":
            continue
        if task_mode == "closed_book_domain":
            if not (claim.source_evidence_refs or claim.visible_evidence_refs):
                return False
        elif not claim.visible_evidence_refs:
            return False
    return True


def _closed(
    *,
    subject_id: str,
    subject_hash_value: str,
    policy: ReviewPolicy,
    status: str,
    reason: str | None,
    accepted: bool,
    active_ids: list[str],
    review_set_id: str,
    human_review_status: str,
    scope: str,
) -> ReviewAggregate:
    ready = bool(accepted and status in {"teacher_consensus_accepted", "teacher_single_accepted"})
    return ReviewAggregate(
        subject_id=subject_id,
        subject_hash=subject_hash_value,
        active_review_ids=active_ids,
        human_review_status=human_review_status or "unreviewed",
        teacher_review_status=status,  # type: ignore[arg-type]
        accepted_by_policy=accepted,
        policy_version=policy.policy_version,
        quarantine_reason=None if accepted else reason,
        readiness_scope=scope if ready else "none",
        ready_for_training=ready and policy.review_mode == "teacher_only",
        ready_for_exploratory_inference=ready and policy.review_mode == "teacher_only",
        ready_for_formal_human_eval=False,
        review_source="teacher",
        review_set_id=review_set_id,
    )


def build_aggregate(
    records: list[ReviewRecord],
    *,
    subject_id: str,
    expected_hash: str,
    policy: ReviewPolicy,
    human_review_status: str = "unreviewed",
    task_mode: str = "rag_grounded",
    risk: str = "high",
) -> ReviewAggregate:
    """忽略调用方传入的 ready_*。教师记录永远不能打开正式人工门禁。"""
    if not records:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="unreviewed",
            reason="unreviewed",
            accepted=False,
            active_ids=[],
            review_set_id="",
            human_review_status=human_review_status,
            scope="none",
        )
    if any(item.subject_hash != expected_hash for item in records):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="stale",
            reason="subject_hash_mismatch",
            accepted=False,
            active_ids=[item.review_id for item in records],
            review_set_id=records[-1].review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    review_set_id = records[-1].review_set_id
    current = [item for item in records if item.review_set_id == review_set_id]
    active = [item for item in current if item.judge_role != "arbiter"]
    arbiters = [item for item in current if item.judge_role == "arbiter"]
    ids = [item.review_id for item in current]
    if any(item.execution_status == "budget_stopped" for item in active):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="budget_stopped",
            reason="budget_stopped",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if any(item.execution_status == "technical_failed" for item in active):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="technical_failed",
            reason="technical_failed",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if any(item.execution_status == "cancelled" for item in active):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="cancelled",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    decisions = {item.decision for item in active}
    if "abstain" in decisions:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="abstain",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if "accept" in decisions and "reject" in decisions:
        resolved = _arbiter_accepts(arbiters, active, task_mode)
        if resolved:
            return _closed(
                subject_id=subject_id,
                subject_hash_value=expected_hash,
                policy=policy,
                status="teacher_consensus_accepted",
                reason=None,
                accepted=True,
                active_ids=ids,
                review_set_id=review_set_id,
                human_review_status=human_review_status,
                scope="exploratory_teacher_reviewed",
            )
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="judge_disagreement",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if "revise" in decisions:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="needs_revision",
            reason="needs_revision",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if decisions == {"reject"}:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="rejected",
            reason="rejected",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if decisions != {"accept"} or not active:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="unresolved",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if not all(_critical_ok(item) and _evidence_ok(item, task_mode) for item in active):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="missing_evidence_or_critical_dimension",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    levels = {item.independence_level for item in active}
    if "same_model" in levels or (len(active) > 1 and len({(item.model_id, item.model_revision) for item in active}) < 2):
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="disputed",
            reason="same_model_not_independent",
            accepted=False,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="none",
        )
    if len(active) >= 2 and levels == {"distinct_family"}:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="teacher_consensus_accepted",
            reason=None,
            accepted=True,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="exploratory_teacher_reviewed",
        )
    if len(active) == 1 and policy.allow_single:
        return _closed(
            subject_id=subject_id,
            subject_hash_value=expected_hash,
            policy=policy,
            status="teacher_single_accepted",
            reason=None,
            accepted=True,
            active_ids=ids,
            review_set_id=review_set_id,
            human_review_status=human_review_status,
            scope="teacher_single_uncalibrated",
        )
    reason = "second_judge_missing" if len(active) < 2 else "independence_unverified"
    return _closed(
        subject_id=subject_id,
        subject_hash_value=expected_hash,
        policy=policy,
        status="disputed",
        reason=reason,
        accepted=False,
        active_ids=ids,
        review_set_id=review_set_id,
        human_review_status=human_review_status,
        scope="none",
    )


def _arbiter_accepts(arbiters: list[ReviewRecord], active: list[ReviewRecord], task_mode: str) -> bool:
    if len(arbiters) != 1:
        return False
    arbiter = arbiters[0]
    if arbiter.execution_status != "succeeded" or arbiter.decision != "accept":
        return False
    if arbiter.independence_level != "distinct_family":
        return False
    judge_ids = {(item.model_id, item.model_revision) for item in active}
    if (arbiter.model_id, arbiter.model_revision) in judge_ids:
        return False
    return _critical_ok(arbiter) and _evidence_ok(arbiter, task_mode)


def release_block_reasons(sample: dict, aggregate: ReviewAggregate | None, policy: ReviewPolicy | None) -> list[str]:
    """teacher_only 新运行的发布检查。未提供政策时不改变旧导出行为。"""
    if policy is None or policy.review_mode != "teacher_only":
        return []
    reasons: list[str] = []
    grade = sample.get("grade")
    action = sample.get("action")
    if grade in {None, "B", "reject", "quarantine"} or action in {"reject", "quarantine", "needs_escalation"}:
        reasons.append("release_ineligible")
    if sample.get("verification_status") == "pending" or sample.get("review_status") == "pending":
        reasons.append("pending_verification")
    if aggregate is None:
        reasons.append("legacy_unreviewed")
        return reasons
    expected = str(sample.get("validation_subject_hash") or "")
    if expected and aggregate.subject_hash != expected:
        reasons.append("stale_hash")
    if not aggregate.accepted_by_policy or not aggregate.ready_for_training:
        reasons.append("teacher_not_accepted")
    if aggregate.ready_for_formal_human_eval:
        reasons.append("formal_flag_not_allowed")
    return reasons
