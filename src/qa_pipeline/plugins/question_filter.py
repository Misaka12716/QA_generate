"""问题侧筛选：none / difficulty_sample（QueST 思路的不确定性采样）。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Question
from ..textutil import rouge_l, token_f1


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(token_f1(a, b), rouge_l(a, b))


@register("question_filter", "none")
class NoQuestionFilter:
    name = "none"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[Question]:
        return questions


@register("question_filter", "difficulty_sample")
class DifficultySample:
    """保留教师两次作答一致性接近 0.5 的题；教师完全确定时保留全部。"""

    name = "difficulty_sample"

    def __init__(self, k: int = 2, **_: object) -> None:
        self.k = max(2, int(k))

    def run(self, questions: list[Question], ctx) -> list[Question]:
        if not questions:
            return questions
        scored: list[tuple[Question, float]] = []
        for q in questions:
            chunk = q.metadata.get("chunk_text") or ""
            answers = []
            for _ in range(self.k):
                answers.append(
                    ctx.llm.chat(
                        [
                            {
                                "role": "system",
                                "content": "请只根据参考文本回答。若不确定，请说明无法确定。",
                            },
                            {"role": "user", "content": f"参考文本：{chunk}\n问题：{q.question}"},
                        ],
                        model=ctx.model_for("cheap"),
                        max_tokens=200,
                        temperature=0.8,
                    )
                )
            sims = [_sim(answers[i], answers[j]) for i in range(len(answers)) for j in range(i + 1, len(answers))]
            consistency = sum(sims) / max(1, len(sims))
            closeness = 1.0 - abs(consistency - 0.5) * 2
            q.metadata["difficulty_consistency"] = round(consistency, 4)
            q.difficulty = max(q.difficulty, round(closeness, 4))
            scored.append((q, closeness))
        if max(score for _, score in scored) < 0.25:
            return questions
        selected = [q for q, score in scored if score >= 0.5]
        return selected or questions
