"""答案蒸馏：concise_response / cot_mixed / multi_teacher_judge。"""

from __future__ import annotations

import json
import re

from ..registry import register
from ..schemas import QAPair, Question, utc_now
from ..textutil import is_substring, longest_overlap_span, sentences


def _parse_answer(raw: str | dict | None) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"answer": raw.strip()}


def _to_pair(q: Question, payload: dict, model: str, cot: bool) -> QAPair:
    chunk_text = q.metadata.get("chunk_text") or ""
    answer = str(payload.get("answer") or payload.get("a") or "").strip()
    if not answer and q.answer_hint:
        answer = q.answer_hint
    evidence = str(payload.get("evidence_span") or q.evidence_span or "").strip()
    if evidence and not is_substring(evidence, chunk_text):
        evidence = longest_overlap_span(evidence, chunk_text) or q.evidence_span
    if not evidence:
        evidence = longest_overlap_span(answer, chunk_text) or (sentences(chunk_text)[:1] or [""])[0]
    pair = QAPair(
        q_id=q.q_id,
        question=q.question,
        answer=answer,
        reasoning=str(payload.get("reasoning") or ""),
        evidence_span=evidence,
        chunk_id=q.chunk_id,
        chunk_text=chunk_text,
        source_doc=str(q.metadata.get("source_doc") or ""),
        q_type=q.q_type,
        teacher_model=model,
        confidence=float(payload.get("confidence") or 0.0),
        cot_enabled=cot,
        evolution_type=q.evolution_type,
        generation_trace={
            "teacher_model": model,
            "anchor": q.anchor,
            "evolution_type": q.evolution_type,
            "cot_enabled": cot,
            "generation_timestamp": utc_now(),
            "q_type": q.q_type,
            "evol_level": q.evol_level,
            "hard": bool(q.metadata.get("hard")),
        },
        metadata=dict(q.metadata),
    )
    pair.log("distillation", "write", teacher_model=model, cot=cot)
    return pair


def _needs_cot(q: Question) -> bool:
    if q.metadata.get("cot"):
        return True
    return q.q_type == "multihop" or int(q.evol_level or 0) >= 2


def _ask(q: Question, ctx, cot: bool, model: str) -> dict:
    if q.answer_hint and ctx.recipe.question_gen.name == "direct_qa":
        return {"answer": q.answer_hint, "evidence_span": q.evidence_span, "confidence": 0.8}
    chunk_text = q.metadata.get("chunk_text") or ""
    if cot:
        system = (
            "请根据给定的参考文本，逐步推理回答用户问题。\n"
            "要求：\n1. 先分析参考文本中的相关信息\n2. 展示推理过程（Chain of Thought）\n"
            "3. 最后给出明确结论\n4. 所有推理必须基于参考文本，不要引入外部知识\n"
            '输出 JSON：{"answer":"...","reasoning":"...","evidence_span":"...","confidence":0.0}'
        )
    else:
        system = (
            "请根据给定的参考文本，准确、简洁地回答用户问题。\n"
            "要求：\n1. 答案必须完全基于参考文本，不要编造\n"
            "2. 如果参考文本不足以回答，明确说明「根据给定材料无法确定」\n"
            "3. 语言简洁，不啰嗦\n"
            '输出 JSON：{"answer":"...","evidence_span":"...","confidence":0.0}'
        )
    data = ctx.llm.chat_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"参考文本：{chunk_text}\n问题：{q.question}"},
        ],
        model=model,
    )
    if data:
        return data
    raw = ctx.llm.chat(
        [
            {"role": "system", "content": system.replace("输出 JSON：", "直接回答：")},
            {"role": "user", "content": f"参考文本：{chunk_text}\n问题：{q.question}"},
        ],
        model=model,
    )
    return _parse_answer(raw)


@register("distillation", "concise_response")
class ConciseResponse:
    name = "concise_response"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        out = []
        for q in questions:
            model = q.metadata.get("teacher_model") or ctx.model_for("default")
            out.append(_to_pair(q, _ask(q, ctx, cot=False, model=model), model, False))
        return [p for p in out if p.answer]


@register("distillation", "cot_mixed")
class CotMixed:
    name = "cot_mixed"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        out = []
        for q in questions:
            model = q.metadata.get("teacher_model") or ctx.model_for("default")
            cot = _needs_cot(q)
            out.append(_to_pair(q, _ask(q, ctx, cot=cot, model=model), model, cot))
        return [p for p in out if p.answer]


@register("distillation", "multi_teacher_judge")
class MultiTeacherJudge:
    name = "multi_teacher_judge"

    def __init__(self, teachers: list[str] | None = None, **_: object) -> None:
        self.teachers = teachers

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        tiers = self.teachers or ["default", "strong"]
        out = []
        for q in questions:
            cands = []
            for tier in tiers:
                model = ctx.model_for(tier)
                payload = _ask(q, ctx, cot=_needs_cot(q), model=model)
                cands.append((model, payload))
            judged = []
            for model, payload in cands:
                score = ctx.llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是裁判。按事实准确性、完整性、清晰度 1-5 打分。"
                                '输出 JSON：{"accuracy":0,"completeness":0,"clarity":0}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"问题：{q.question}\n答案：{payload.get('answer')}\n"
                                f"参考：{q.metadata.get('chunk_text','')[:1500]}"
                            ),
                        },
                    ],
                    model=ctx.model_for("strong"),
                ) or {}
                total = float(score.get("accuracy") or 0) + float(score.get("completeness") or 0) + float(score.get("clarity") or 0)
                judged.append((total, model, payload, score))
            judged.sort(key=lambda x: -x[0])
            _, model, payload, score = judged[0]
            pair = _to_pair(q, payload, model, _needs_cot(q))
            pair.generation_trace["multi_teacher"] = [
                {"model": m, "score": sc} for _, m, _, sc in judged
            ]
            pair.log("distillation", self.name, picked=model, judge=score)
            out.append(pair)
        return [p for p in out if p.answer]


@register("distillation", "routed_teacher")
class RoutedTeacher:
    """简单题走单教师，hard 标记的难题才走多教师 + Judge。"""

    name = "routed_teacher"

    def __init__(self, teachers: list[str] | None = None, **_: object) -> None:
        self.teachers = teachers

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        easy = [q for q in questions if not q.metadata.get("hard")]
        hard = [q for q in questions if q.metadata.get("hard")]
        out = CotMixed().run(easy, ctx)
        out.extend(MultiTeacherJudge(teachers=self.teachers).run(hard, ctx))
        return out
