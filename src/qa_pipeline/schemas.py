"""管线各阶段的标准化数据结构。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def request_id(prefix: str = "req_") -> str:
    """随机 ID 只用于请求，不用于文档或切块。"""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _uid(prefix: str = "") -> str:
    return request_id(prefix)


def content_id(prefix: str, payload: Any) -> str:
    """由规范化内容生成稳定 ID。同输入同参数得到相同结果。"""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:20]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


Q_TYPES = ("factual", "procedural", "conditional", "comparative", "multihop")
_QTYPE_ALIAS = {"explanatory": "procedural", "reasoning": "multihop"}
QType = Literal["factual", "procedural", "conditional", "comparative", "multihop"]
INTENTS = (
    "lookup_explain",
    "procedure",
    "rule_decision",
    "compare_select",
    "diagnose_explain",
    "synthesize",
    "other_review",
)
Intent = Literal[
    "lookup_explain",
    "procedure",
    "rule_decision",
    "compare_select",
    "diagnose_explain",
    "synthesize",
    "other_review",
]
_INTENT_FROM_QTYPE = {
    "factual": "lookup_explain",
    "procedural": "procedure",
    "conditional": "rule_decision",
    "comparative": "compare_select",
    "multihop": "synthesize",
}
_QTYPE_FROM_INTENT = {
    "lookup_explain": "factual",
    "procedure": "procedural",
    "rule_decision": "conditional",
    "compare_select": "comparative",
    "diagnose_explain": "procedural",
    "synthesize": "multihop",
    "other_review": "factual",
}
Goal = Literal["rag_grounded", "closed_book_domain"]
EvidenceState = Literal["sufficient", "partial", "missing", "conflict", "ambiguous"]
EXPECTED_ACTIONS = (
    "answer",
    "partial_answer",
    "clarify",
    "state_insufficient",
    "state_conflict",
    "correct_premise",
    "state_scope",
)
_ACTION_ALIAS = {"partial": "partial_answer", "insufficient": "state_insufficient"}
ExpectedAction = Literal[
    "answer",
    "partial_answer",
    "clarify",
    "state_insufficient",
    "state_conflict",
    "correct_premise",
    "state_scope",
]
GENERATION_ROUTES = ("direct_grounded", "anchor_assisted", "k_sequential", "k_joint")
GenerationRoute = Literal["direct_grounded", "anchor_assisted", "k_sequential", "k_joint"]
DATA_STAGES = ("accepted", "selected", "released", "actually_trained")
DataStage = Literal["accepted", "selected", "released", "actually_trained"]
SelectionRole = Literal["learning", "retention", "behavior"]
Grade = Literal["S", "A", "B", "quarantine", "reject"]
FilterAction = Literal["pass", "reject", "downgrade", "quarantine", "needs_escalation"]
CALL_STATUSES = (
    "ok",
    "generated_empty",
    "parse_failed",
    "schema_failed",
    "transport_failed",
    "budget_stopped",
    "semantic_rejected",
)
CallStatus = Literal[
    "ok",
    "generated_empty",
    "parse_failed",
    "schema_failed",
    "transport_failed",
    "budget_stopped",
    "semantic_rejected",
]
VERIFICATION_STATUSES = ("evidence_located", "semantically_verified", "pending", "failed")
SCORING_TOKENIZER_ID = "locked-char-v1"
AnchorType = Literal["entity", "keyword", "sentence"]
ENTITY_TYPES = (
    "product",
    "medicinal_material",
    "source_plant",
    "constituent",
    "population",
    "unknown",
)
EntityType = Literal[
    "product",
    "medicinal_material",
    "source_plant",
    "constituent",
    "population",
    "unknown",
]
RESPONSE_STYLES = ("brief", "standard", "detailed")
ResponseStyle = Literal["brief", "standard", "detailed"]
TYPE_LABEL_ORIGINS = ("source", "backfilled", "inferred", "unknown")
TypeLabelOrigin = Literal["source", "backfilled", "inferred", "unknown"]
SELECTION_STATUSES = ("selected", "not_selected", "")
QUOTA_RATIOS = {
    "factual": 0.40,
    "procedural": 0.20,
    "conditional": 0.15,
    "comparative": 0.15,
    "multihop": 0.10,
}


def canon_qtype(value: object, default: str = "factual") -> str:
    """把旧题型名映射到五类兼容别名；无法识别时回落到 default。"""
    if not isinstance(value, str):
        return default
    name = _QTYPE_ALIAS.get(value.strip(), value.strip())
    return name if name in Q_TYPES else default


def canon_intent(value: object, default: str = "lookup_explain") -> str:
    if isinstance(value, str) and value.strip() in INTENTS:
        return value.strip()
    if isinstance(value, str):
        qtype = canon_qtype(value, default="")
        if qtype in _INTENT_FROM_QTYPE:
            return _INTENT_FROM_QTYPE[qtype]
    return default


def qtype_for_intent(intent: str) -> str:
    return _QTYPE_FROM_INTENT.get(intent, "factual")


def canon_action(value: object, default: str = "answer") -> str:
    """旧值 partial / insufficient 只作为读入别名。"""
    if not isinstance(value, str):
        return default
    name = _ACTION_ALIAS.get(value.strip(), value.strip())
    return name if name in EXPECTED_ACTIONS else default


def canon_route(value: object, default: str = "direct_grounded") -> str:
    if isinstance(value, str) and value.strip() in GENERATION_ROUTES:
        return value.strip()
    if value == "knowledge_unit":
        return "k_sequential"
    return default


class DocumentIdentity(BaseModel):
    """文档身份。标题和实体说明不写入正文，避免被当成连续原文。"""

    document_title: str = ""
    canonical_subject: str = ""
    entity_type: EntityType = "unknown"
    aliases: list[str] = Field(default_factory=list)
    dosage_form: str = ""
    strength: str = ""
    version: str = ""
    source_id: str = ""
    source_family_id: str = ""
    identity_notes: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("entity_type", mode="before")
    @classmethod
    def _canon_entity(cls, value: object) -> str:
        if isinstance(value, str) and value.strip() in ENTITY_TYPES:
            return value.strip()
        return "unknown"


class EvidenceQuote(BaseModel):
    quote_id: str = ""
    source_id: str = ""
    source_version: str = ""
    source_family_id: str = ""
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    quote: str = ""
    quote_hash: str = ""


class AnswerPointSpec(BaseModel):
    point_id: str = ""
    text: str = ""
    criticality: Literal["required", "optional"] = "required"
    support_quote_ids: list[str] = Field(default_factory=list)


class ResponseContract(BaseModel):
    """要点角色划分。文本以 AnswerPointSpec 为准，这里只保存 ID。"""

    required_point_ids: list[str] = Field(default_factory=list)
    optional_point_ids: list[str] = Field(default_factory=list)
    critical_conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    required_steps: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    unavailable_points: list[str] = Field(default_factory=list)
    response_style: ResponseStyle = "standard"

    @field_validator("response_style", mode="before")
    @classmethod
    def _canon_style(cls, value: object) -> str:
        if isinstance(value, str) and value.strip() in RESPONSE_STYLES:
            return value.strip()
        return "standard"


class Document(BaseModel):
    doc_id: str = ""
    path: str = ""
    title: str = ""
    text: str
    source_group: str = ""
    source_version: str = ""
    dataset_version: str = ""
    source_hash: str = ""
    source_family_id: str = ""
    clean_version: str = ""
    tokenizer_id: str = "approx-v1"
    location: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _content_identity(self) -> Document:
        if not self.source_hash:
            self.source_hash = hashlib.sha256((self.text or "").encode("utf-8")).hexdigest()
        if not self.doc_id:
            self.doc_id = content_id(
                "doc_",
                {
                    "source_hash": self.source_hash,
                    "clean_version": self.clean_version,
                    "text": self.text,
                    "tokenizer_id": self.tokenizer_id,
                },
            )
        return self


def chunk_identity_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "source_hash": chunk.source_hash,
        "clean_version": chunk.metadata.get("clean_version") or "",
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "chunking": chunk.metadata.get("chunking"),
        "chunk_params": chunk.metadata.get("chunk_params") or {},
        "tokenizer_id": chunk.tokenizer_id or "approx-v1",
    }


def refresh_chunk_id(chunk: Chunk) -> Chunk:
    chunk.chunk_id = content_id("chk_", chunk_identity_payload(chunk))
    return chunk


class Chunk(BaseModel):
    chunk_id: str = ""
    text: str
    doc_id: str = ""
    source_doc: str = ""
    title_path: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    source_hash: str = ""
    source_family_id: str = ""
    dataset_version: str = ""
    tokenizer_id: str = "approx-v1"
    location: str = ""
    document_identity: DocumentIdentity = Field(default_factory=DocumentIdentity)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _content_identity(self) -> Chunk:
        if not self.chunk_id:
            refresh_chunk_id(self)
        return self


class Anchor(BaseModel):
    anchor_text: str
    anchor_type: AnchorType = "keyword"
    position_in_chunk: int = 0
    chunk_id: str = ""
    score: float = 0.0


class Question(BaseModel):
    q_id: str = Field(default_factory=lambda: _uid("q_"))
    question: str
    chunk_id: str
    anchor: str = ""
    q_type: QType = "factual"
    evidence_span: str = ""
    evolution_type: str | None = None
    evol_level: int = 0
    difficulty: float = 0.0
    answer_hint: str = ""
    intent_primary: str = ""
    operations: list[str] = Field(default_factory=list)
    evidence_topology: str = "single"
    evidence_state: EvidenceState = "sufficient"
    expected_action: ExpectedAction = "answer"
    answer_points: list[str] = Field(default_factory=list)
    generation_route: GenerationRoute = "direct_grounded"
    family_id: str = ""
    task_variant_id: str = ""
    source_family_id: str = ""
    dataset_version: str = ""
    source_hash: str = ""
    location: str = ""
    requested_type: str = ""
    actual_type: str = ""
    requested_q_type: str = ""
    actual_q_type: str = ""
    type_label_origin: str = ""
    requested_evidence: str = ""
    located_evidence: str = ""
    repair_status: str = ""
    verification_status: str = ""
    rubric_version: str = ""
    review_status: str = ""
    critical_constraints: list[str] = Field(default_factory=list)
    acceptable_variants: list[str] = Field(default_factory=list)
    document_identity: DocumentIdentity = Field(default_factory=DocumentIdentity)
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    answer_point_specs: list[AnswerPointSpec] = Field(default_factory=list)
    response_contract: ResponseContract = Field(default_factory=ResponseContract)
    knowledge_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("q_type", mode="before")
    @classmethod
    def _canon_qtype(cls, value: object) -> str:
        return canon_qtype(value)

    @field_validator("expected_action", mode="before")
    @classmethod
    def _canon_action(cls, value: object) -> str:
        return canon_action(value)

    @field_validator("generation_route", mode="before")
    @classmethod
    def _canon_route(cls, value: object) -> str:
        return canon_route(value)


class QAPair(BaseModel):
    qa_id: str = Field(default_factory=lambda: _uid("qa_"))
    q_id: str = ""
    question: str
    answer: str
    reasoning: str = ""
    evidence_span: str = ""
    chunk_id: str = ""
    chunk_text: str = ""
    source_doc: str = ""
    q_type: QType = "factual"
    teacher_model: str = ""
    confidence: float = 0.0
    cot_enabled: bool = False
    evolution_type: str | None = None
    grade: Grade | None = None
    action: FilterAction = "pass"
    nli_score: float | None = None
    judge_scores: dict[str, float] = Field(default_factory=dict)
    judge_overall: float | None = None
    kb_gain: float | None = None
    goal: Goal = "rag_grounded"
    intent_primary: str = ""
    operations: list[str] = Field(default_factory=list)
    evidence_topology: str = "single"
    evidence_state: EvidenceState = "sufficient"
    expected_action: ExpectedAction = "answer"
    answer_points: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    selection_role: SelectionRole = "learning"
    generation_route: GenerationRoute = "direct_grounded"
    construction_evidence_refs: list[str] = Field(default_factory=list)
    visible_support_refs: list[str] = Field(default_factory=list)
    teacher_context_refs: list[str] = Field(default_factory=list)
    validation_subject_hash: str = ""
    data_stage: DataStage | None = None
    student_context_refs: list[str] = Field(default_factory=list)
    student_context_score: float | None = None
    student_closed_score: float | None = None
    family_id: str = ""
    task_variant_id: str = ""
    parent_sample_id: str = ""
    dataset_version: str = ""
    source_hash: str = ""
    source_family_id: str = ""
    location: str = ""
    rubric_version: str = ""
    review_status: str = ""
    critical_constraints: list[str] = Field(default_factory=list)
    acceptable_variants: list[str] = Field(default_factory=list)
    selection_weight: float = 1.0
    included_in_this_run: bool | None = None
    exclude_reason: str = ""
    requested_type: str = ""
    actual_type: str = ""
    requested_q_type: str = ""
    actual_q_type: str = ""
    type_label_origin: str = ""
    selection_status: str = ""
    requested_evidence: str = ""
    located_evidence: str = ""
    repair_status: str = ""
    verification_status: str = ""
    document_identity: DocumentIdentity = Field(default_factory=DocumentIdentity)
    evidence_quotes: list[EvidenceQuote] = Field(default_factory=list)
    answer_point_specs: list[AnswerPointSpec] = Field(default_factory=list)
    response_contract: ResponseContract = Field(default_factory=ResponseContract)
    knowledge_id: str = ""
    exposure_count: int = 0
    assistant_target_tokens: int = 0
    sequence_tokens: int = 0
    loss_weight: float = 1.0
    filter_trace: dict[str, Any] = Field(default_factory=dict)
    generation_trace: dict[str, Any] = Field(default_factory=dict)
    audit: list[dict[str, Any]] = Field(default_factory=list)
    split: Literal["train", "validation", "test"] = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("q_type", mode="before")
    @classmethod
    def _canon_qtype(cls, value: object) -> str:
        return canon_qtype(value)

    @field_validator("expected_action", mode="before")
    @classmethod
    def _canon_action(cls, value: object) -> str:
        return canon_action(value)

    @field_validator("generation_route", mode="before")
    @classmethod
    def _canon_route(cls, value: object) -> str:
        return canon_route(value)

    def log(self, stage: str, strategy: str, **payload: Any) -> None:
        self.audit.append(
            {
                "at": utc_now(),
                "stage": stage,
                "strategy": strategy,
                **payload,
            }
        )


class FilterDecision(BaseModel):
    action: FilterAction = "pass"
    reason: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    backend: str = ""


class UsageStats(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    produced: dict[str, int] = Field(default_factory=dict)
    rejected: dict[str, int] = Field(default_factory=dict)
    funnel: dict[str, dict[str, int]] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list)
    budget_stops: list[str] = Field(default_factory=list)
    ledger: dict[str, Any] = Field(default_factory=dict)
    call_statuses: dict[str, int] = Field(default_factory=dict)
    stage_counts: dict[str, Any] = Field(default_factory=dict)


def lineage_record(pair: QAPair) -> dict[str, Any]:
    return {
        "dataset_version": pair.dataset_version,
        "source_group": pair.metadata.get("source_group") or pair.source_doc,
        "source_hash": pair.source_hash,
        "chunk_id": pair.chunk_id,
        "location": pair.location,
        "split": pair.split,
        "sample_id": pair.qa_id,
        "family_id": pair.family_id,
        "task_variant_id": pair.task_variant_id,
        "parent_id": pair.parent_sample_id,
        "route": pair.generation_route,
        "teacher_revision": pair.teacher_model,
        "source_family_id": pair.source_family_id,
        "sample_family_id": pair.family_id,
        "knowledge_id": pair.knowledge_id,
    }


def sample_gold_record(pair: QAPair) -> dict[str, Any]:
    visible = pair.student_context_refs or ([pair.chunk_text] if pair.chunk_text else [])
    return {
        "question": pair.question,
        "visible_context_refs": visible,
        "context_hash": hashlib.sha256("\n".join(visible).encode("utf-8")).hexdigest(),
        "evidence_state": pair.evidence_state,
        "expected_action": pair.expected_action,
        "answer_points": list(pair.answer_points),
        "answer_point_specs": [item.model_dump() for item in pair.answer_point_specs],
        "response_contract": pair.response_contract.model_dump(),
        "critical_constraints": list(pair.critical_constraints),
        "acceptable_variants": list(pair.acceptable_variants),
        "rubric_version": pair.rubric_version,
        "review_status": pair.review_status,
    }


def training_consumption_record(pair: QAPair, run_id: str = "") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample_id": pair.qa_id,
        "validation_subject_hash": pair.validation_subject_hash,
        "selected": pair.included_in_this_run is not False and not pair.exclude_reason,
        "skip_reason": pair.exclude_reason,
        "exposure_count": pair.exposure_count,
        "assistant_target_tokens": pair.assistant_target_tokens,
        "sequence_tokens": pair.sequence_tokens,
        "loss_weight": pair.loss_weight,
    }


def project_required_texts(pair: QAPair) -> list[str]:
    """必答要点文本只从规格投影，避免和 response_contract 各写一套。"""
    by_id = {spec.point_id: spec.text for spec in pair.answer_point_specs if spec.point_id}
    ids = list(pair.response_contract.required_point_ids)
    if not ids:
        ids = [spec.point_id for spec in pair.answer_point_specs if spec.criticality == "required" and spec.point_id]
    projected = [by_id[item] for item in ids if by_id.get(item)]
    if projected:
        return projected
    return [text for text in pair.answer_points if str(text).strip()]


def prediction_record(
    *,
    case_id: str,
    model_revision: str = "",
    adapter_hash: str = "",
    input_hash: str = "",
    answer: str = "",
    generated_tokens: int = 0,
    stop_reason: str = "",
    per_point_scores: dict[str, Any] | None = None,
    critical_errors: list[str] | None = None,
    actual_action: str = "",
    judge_revision: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model_revision": model_revision,
        "adapter_hash": adapter_hash,
        "input_hash": input_hash,
        "answer": answer,
        "generated_tokens": generated_tokens,
        "stop_reason": stop_reason,
        "per_point_scores": per_point_scores or {},
        "critical_errors": critical_errors or [],
        "actual_action": actual_action,
        "judge_revision": judge_revision,
    }
