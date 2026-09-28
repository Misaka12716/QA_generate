"""Evol-Instruct：none / evol_depth / evol_breadth / evol_both，带 Eliminator。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Question
from ..textutil import tokenize

_DEPTH_HINTS = {
    "加约束": "增加条件限制，使问题只在特定工况下成立",
    "深化": "把是什么改成为什么/如何，要求解释机制",
    "具体化": "引入具体场景或设备状态",
    "增推理步": "要求两到三步组合推理才能作答",
}

_BREADTH_HINTS = {
    "对比": "改写成对比 A 与 B 的差异",
    "总结": "改写成归纳该段要点",
    "排障": "改写成故障排查类长尾问题",
}


def _eliminate(question: Question, ctx) -> bool:
    chunk_text = question.metadata.get("chunk_text") or ""
    text = ctx.llm.chat(
        [
            {
                "role": "system",
                "content": "请根据参考文本回答问题。若材料不足以回答，只回复：无法确定。",
            },
            {"role": "user", "content": f"参考文本：{chunk_text}\n问题：{question.question}"},
        ],
        model=ctx.model_for("cheap"),
        max_tokens=200,
    )
    if not text or "无法确定" in text or len(tokenize(text)) < 8:
        return False
    question.metadata["eliminator_preview"] = text[:200]
    return True


def _evolve_one(question: Question, kind: str, hint: str, ctx) -> Question | None:
    if question.q_type == "reasoning" and kind.startswith("evol_depth"):
        pass
    data = ctx.llm.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你在做 Evol-Instruct 改写问题。必须仍能仅根据原文回答，禁止引入原文没有的实体。"
                    f"进化操作：{hint}。"
                    '输出 JSON：{"question":"...","evolution_type":"...","q_type":"factual|explanatory|reasoning"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原文：{question.metadata.get('chunk_text','')}\n"
                    f"问题：{question.question}\n锚点：{question.anchor}"
                ),
            },
        ],
        model=ctx.model_for("default"),
    ) or {}
    qtext = str(data.get("question") or "").strip()
    if len(tokenize(qtext)) < 5:
        return None
    evolved = question.model_copy(deep=True)
    evolved.q_id = Question().q_id
    evolved.question = qtext
    evolved.evolution_type = str(data.get("evolution_type") or hint[:8])
    if data.get("q_type") in {"factual", "explanatory", "reasoning"}:
        evolved.q_type = data["q_type"]
    elif "推理" in hint or "对比" in hint:
        evolved.q_type = "reasoning"
    evolved.difficulty = min(1.0, question.difficulty + 0.3)
    evolved.metadata = {**question.metadata, "parent_q_id": question.q_id, "evolution_kind": kind}
    if not _eliminate(evolved, ctx):
        return None
    return evolved


def _sample(questions: list[Question], ratio: float, rng) -> list[Question]:
    factual = [q for q in questions if q.q_type == "factual" and not q.evolution_type]
    pool = factual or questions
    k = max(0, int(len(pool) * ratio))
    if k == 0:
        return []
    picked = list(pool)
    rng.shuffle(picked)
    return picked[:k]


class _BaseEvol:
    kind = "none"
    hints: dict[str, str] = {}

    def __init__(self, sample_ratio: float = 0.25, **_: object) -> None:
        self.sample_ratio = float(sample_ratio)

    def run(self, questions: list[Question], ctx) -> list[Question]:
        if self.kind == "none" or self.sample_ratio <= 0:
            return questions
        extra: list[Question] = []
        hints = list(self.hints.items())
        for i, q in enumerate(_sample(questions, self.sample_ratio, ctx.rng)):
            name, hint = hints[i % len(hints)]
            evolved = _evolve_one(q, self.kind + ":" + name, hint, ctx)
            if evolved:
                extra.append(evolved)
        return questions + extra


@register("evolution", "none")
class NoEvol(_BaseEvol):
    name = "none"
    kind = "none"


@register("evolution", "evol_depth")
class EvolDepth(_BaseEvol):
    name = "evol_depth"
    kind = "evol_depth"
    hints = _DEPTH_HINTS


@register("evolution", "evol_breadth")
class EvolBreadth(_BaseEvol):
    name = "evol_breadth"
    kind = "evol_breadth"
    hints = _BREADTH_HINTS


@register("evolution", "evol_both")
class EvolBoth(_BaseEvol):
    name = "evol_both"
    kind = "evol_both"
    hints = {**_DEPTH_HINTS, **_BREADTH_HINTS}
