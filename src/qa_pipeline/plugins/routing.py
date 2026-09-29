"""教师路由：single / grade_by_qtype / always_strong。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Question
from ..textutil import approx_tokens


def _mark(q: Question, ctx, tier: str, *, hard: bool = False, cot: bool = False) -> None:
    q.metadata["teacher_tier"] = tier
    q.metadata["teacher_model"] = ctx.model_for(tier)
    q.metadata["hard"] = hard
    q.metadata["cot"] = cot or hard or q.q_type == "multihop" or int(q.evol_level or 0) >= 2


@register("teacher_router", "single")
class SingleRouter:
    name = "single"

    def __init__(self, tier: str = "default", **_: object) -> None:
        self.tier = tier

    def run(self, questions: list[Question], ctx) -> list[Question]:
        for q in questions:
            _mark(q, ctx, self.tier, hard=False, cot=False)
        return questions


@register("teacher_router", "grade_by_qtype")
class GradeByQTypeRouter:
    name = "grade_by_qtype"

    def __init__(
        self,
        factual: str = "cheap",
        procedural: str = "strong",
        conditional: str = "strong",
        comparative: str = "strong",
        multihop: str = "strong",
        explanatory: str | None = None,
        reasoning: str | None = None,
        long_chunk_tokens: int = 900,
        **_: object,
    ) -> None:
        self.map = {
            "factual": factual,
            "procedural": procedural if explanatory is None else explanatory,
            "conditional": conditional,
            "comparative": comparative,
            "multihop": multihop if reasoning is None else reasoning,
        }
        self.long_chunk_tokens = int(long_chunk_tokens)

    def run(self, questions: list[Question], ctx) -> list[Question]:
        for q in questions:
            tier = self.map.get(q.q_type, "default")
            chunk_text = q.metadata.get("chunk_text") or ""
            hard = q.q_type == "multihop" or int(q.evol_level or 0) >= 2
            if approx_tokens(chunk_text) >= self.long_chunk_tokens or hard:
                tier = "strong"
            _mark(q, ctx, tier, hard=hard, cot=hard or q.q_type in {"procedural", "conditional", "comparative", "multihop"})
        return questions


@register("teacher_router", "always_strong")
class AlwaysStrongRouter:
    name = "always_strong"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[Question]:
        for q in questions:
            hard = q.q_type == "multihop" or int(q.evol_level or 0) >= 2
            _mark(q, ctx, "strong", hard=hard, cot=True)
        return questions
