"""审核记录契约。判定与执行状态分开，不能靠自由文本里的“通过”放行。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["accept", "revise", "reject", "abstain"]
ExecutionStatus = Literal["succeeded", "technical_failed", "budget_stopped", "cancelled"]
SubjectType = Literal["training_sample", "protocol_case", "prediction"]
ReviewSource = Literal["human", "teacher", "hybrid"]
TeacherStatus = Literal[
    "unreviewed",
    "pending",
    "teacher_single_accepted",
    "teacher_consensus_accepted",
    "needs_revision",
    "rejected",
    "disputed",
    "technical_failed",
    "budget_stopped",
    "stale",
]


class DimensionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_clear: bool | None = None
    semantic_preserved: bool | None = None
    evidence_adequate: bool | None = None
    factual_consistency: bool | None = None
    required_points_complete: bool | None = None
    behavior_appropriate: bool | None = None
    no_unsupported_claims: bool | None = None


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    status: Literal["supported", "contradicted", "insufficient"]
    visible_evidence_refs: list[str] = Field(default_factory=list)
    source_evidence_refs: list[str] = Field(default_factory=list)


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    batch_id: str
    subject_type: SubjectType
    subject_id: str
    subject_version: str
    review_source: ReviewSource
    review_mode: str
    review_policy_id: str
    policy_version: str
    subject_hash: str
    model_input_hash: str | None = None
    evidence_snapshot_hash: str | None = None
    gold_hash: str | None = None
    provider_id: str
    model_id: str
    model_revision: str
    model_identity_verified: bool = False
    prompt_hash: str
    rubric_version: str
    inference_config_hash: str
    decision: Decision
    execution_status: ExecutionStatus
    dimensions: DimensionScores = Field(default_factory=DimensionScores)
    claims: list[ClaimRecord] = Field(default_factory=list)
    answerable_points: list[str] = Field(default_factory=list)
    unavailable_points: list[str] = Field(default_factory=list)
    numeric_bindings: list[dict[str, Any]] = Field(default_factory=list)
    required_conditions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    suggested_patch: dict[str, Any] | None = None
    confidence: float | None = None
    confidence_calibrated: bool = False
    request_id: str = ""
    attempt: int = 1
    usage: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float | None = None
    currency: str = "unknown"
    pricing_version: str = "unknown"
    created_at: str
    raw_response_artifact_id: str | None = None
    review_set_id: str
    judge_role: str
    independence_level: str
    parent_review_id: str | None = None
    supersedes_review_id: str | None = None
    cache_key: str = ""


class ReviewAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    subject_hash: str
    active_review_ids: list[str] = Field(default_factory=list)
    human_review_status: str = "unreviewed"
    teacher_review_status: TeacherStatus = "unreviewed"
    accepted_by_policy: bool = False
    policy_version: str
    quarantine_reason: str | None = None
    readiness_scope: str = "none"
    ready_for_training: bool = False
    ready_for_exploratory_inference: bool = False
    ready_for_formal_human_eval: bool = False
    review_source: ReviewSource = "teacher"
    review_set_id: str = ""
