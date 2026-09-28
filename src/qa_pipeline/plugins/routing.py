"""教师路由：single / grade_by_qtype / always_strong。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Question
from ..textutil import approx_tokens


@register("teacher_router", "single")
class SingleRouter:
    name = "single"

    def __init__(self, tier: str = "default", **_: object) -> None:
        self.tier = tier

    def run(self, questions: list[Question], ctx) -> list[Question]:
        model = ctx.model_for(self.tier)
        for q in questions:
            q.metadata["teacher_model"] = model
            q.metadata["teacher_tier"] = self.tier
        return questions


@register("teacher_router", "grade_by_qtype")
class GradeByQTypeRouter:
    name = "grade_by_qtype"

    def __init__(
        self,
        factual: str = "cheap",
        explanatory: str = "default",
        reasoning: str = "strong",
        long_chunk_tokens: int = 900,
        **_: object,
    ) -> None:
        self.map = {
            "factual": factual,
            "explanatory": explanatory,
            "reasoning": reasoning,
        }
        self.long_chunk_tokens = int(long_chunk_tokens)

    def run(self, questions: list[Question], ctx) -> list[Question]:
        for q in questions:
            tier = self.map.get(q.q_type, "default")
            chunk_text = q.metadata.get("chunk_text") or ""
            if approx_tokens(chunk_text) >= self.long_chunk_tokens or q.evolution_type:
                tier = "strong"
            q.metadata["teacher_tier"] = tier
            q.metadata["teacher_model"] = ctx.model_for(tier)
        return questions


@register("teacher_router", "always_strong")
class AlwaysStrongRouter:
    name = "always_strong"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[Question]:
        model = ctx.model_for("strong")
        for q in questions:
            q.metadata["teacher_model"] = model
            q.metadata["teacher_tier"] = "strong"
        return questions
