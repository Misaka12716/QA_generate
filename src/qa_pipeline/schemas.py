"""管线各阶段的标准化数据结构。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


QType = Literal["factual", "explanatory", "reasoning"]
Grade = Literal["S", "A", "B", "reject"]
FilterAction = Literal["pass", "reject", "downgrade"]
AnchorType = Literal["entity", "keyword", "sentence"]


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: _uid("doc_"))
    path: str = ""
    title: str = ""
    text: str
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
    difficulty: float = 0.0
    answer_hint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    filter_trace: dict[str, Any] = Field(default_factory=dict)
    generation_trace: dict[str, Any] = Field(default_factory=dict)
    audit: list[dict[str, Any]] = Field(default_factory=list)
    split: Literal["train", "validation", "test"] = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    fallbacks: list[str] = Field(default_factory=list)
