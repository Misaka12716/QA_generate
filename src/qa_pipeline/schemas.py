"""管线各阶段的标准化数据结构。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


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
)
_ACTION_ALIAS = {"partial": "partial_answer", "insufficient": "state_insufficient"}
ExpectedAction = Literal[
    "answer",
    "partial_answer",
    "clarify",
    "state_insufficient",
    "state_conflict",
]
GENERATION_ROUTES = ("direct_grounded", "anchor_assisted", "k_sequential", "k_joint")
GenerationRoute = Literal["direct_grounded", "anchor_assisted", "k_sequential", "k_joint"]
DATA_STAGES = ("accepted", "selected", "released", "actually_trained")
DataStage = Literal["accepted", "selected", "released", "actually_trained"]
SelectionRole = Literal["learning", "retention", "behavior"]
Grade = Literal["S", "A", "B", "quarantine", "reject"]
FilterAction = Literal["pass", "reject", "downgrade", "quarantine"]
AnchorType = Literal["entity", "keyword", "sentence"]


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


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: _uid("doc_"))
    path: str = ""
    title: str = ""
    text: str
    source_group: str = ""
    source_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: _uid("chk_"))
    text: str
    doc_id: str = ""
    source_doc: str = ""
    title_path: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    parent_sample_id: str = ""
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
