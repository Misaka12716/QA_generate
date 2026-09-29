"""Evol-Instruct：none / evol_depth / evol_breadth / evol_both，带 Eliminator。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Q_TYPES, Question, canon_qtype
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


def _evolve_one(
    question: Question,
    kind: str,
    hint: str,
    ctx,
    *,
    constrain: bool = True,
    eliminator: bool = True,
) -> Question | None:
    qtype_doc = "|".join(Q_TYPES)
    if constrain:
        system = (
            "你在做受控进化（Tag-Evol）。知识标签必须注入改写："
            f"锚点「{question.anchor or '本段'}」、题型「{question.q_type}」。"
            "进化后的问题仍必须仅根据原文回答，禁止引入原文没有的实体。"
            f"进化操作：{hint}。"
            f'输出 JSON：{{"question":"...","evolution_type":"...","q_type":"{qtype_doc}"}}'
        )
    else:
        system = (
            "你在做无约束 Evol-Instruct。可以引入原文之外的条件、实体或更广的任务，以提升难度。"
            f"进化操作：{hint}。"
            f'输出 JSON：{{"question":"...","evolution_type":"...","q_type":"{qtype_doc}"}}'
        )
    data = ctx.llm.chat_json(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"原文：{question.metadata.get('chunk_text','')}\n"
                    f"问题：{question.question}\n锚点：{question.anchor}\n题型：{question.q_type}"
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
    raw_type = data.get("q_type")
    if isinstance(raw_type, str) and (raw_type in Q_TYPES or raw_type in {"explanatory", "reasoning"}):
        evolved.q_type = canon_qtype(raw_type)
    elif "推理" in hint or "对比" in hint or "多跳" in hint:
        evolved.q_type = "multihop"
    breadth = any(token in kind for token in ("对比", "总结", "排障", "breadth"))
    evolved.evol_level = min(2, int(question.evol_level or 0) + (2 if breadth else 1))
    evolved.difficulty = min(1.0, question.difficulty + 0.3)
    evolved.metadata = {
        **question.metadata,
        "parent_q_id": question.q_id,
        "evolution_kind": kind,
        "evol_level": evolved.evol_level,
        "constrained": constrain,
    }
    if eliminator and not _eliminate(evolved, ctx):
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
    constrain = True
    eliminator = True

    def __init__(self, sample_ratio: float = 0.25, **_: object) -> None:
        self.sample_ratio = float(sample_ratio)

    def run(self, questions: list[Question], ctx) -> list[Question]:
        if self.kind == "none" or self.sample_ratio <= 0:
            return questions
        extra: list[Question] = []
        hints = list(self.hints.items())
        for i, q in enumerate(_sample(questions, self.sample_ratio, ctx.rng)):
            name, hint = hints[i % len(hints)]
            evolved = _evolve_one(
                q,
                self.kind + ":" + name,
                hint,
                ctx,
                constrain=self.constrain,
                eliminator=self.eliminator,
            )
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


@register("evolution", "tag_evol")
class TagEvol(_BaseEvol):
    """知识标签受控进化：锚点与题型注入提示，进化后做可回答性预筛。"""

    name = "tag_evol"
    kind = "tag_evol"
    hints = {**_DEPTH_HINTS, **_BREADTH_HINTS}
    constrain = True
    eliminator = True

    def __init__(self, sample_ratio: float = 0.3, **_: object) -> None:
        super().__init__(sample_ratio=sample_ratio)


@register("evolution", "evol_unconstrained")
class EvolUnconstrained(_BaseEvol):
    """无锚点约束的 Evol-Instruct，用于对照受控进化是否守住 grounding。"""

    name = "evol_unconstrained"
    kind = "evol_unconstrained"
    hints = {**_DEPTH_HINTS, **_BREADTH_HINTS}
    constrain = False
    eliminator = False
