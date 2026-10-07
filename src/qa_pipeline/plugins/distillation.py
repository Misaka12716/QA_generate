"""答案蒸馏：concise_response / cot_mixed / multi_teacher_judge。"""

from __future__ import annotations

import json
import re

from ..llm import json_payload
from ..registry import register
from ..schemas import QAPair, Question, canon_intent, qtype_for_intent, utc_now
from ..textutil import is_substring


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
    requested = str(payload.get("evidence_span") or q.requested_evidence or q.evidence_span or "").strip()
    quote_count = len([item for item in q.evidence_quotes if item.quote])
    if quote_count > 1:
        evidence = ""
        requested = ""
    else:
        evidence = requested if requested and is_substring(requested, chunk_text) else ""
    repair = "located" if evidence else ("missing" if not requested else "repair_pending")
    intent = canon_intent(q.intent_primary or q.q_type)
    points = [str(item) for item in (q.answer_points or []) if str(item).strip()]
    pair = QAPair(
        q_id=q.q_id,
        question=q.question,
        answer=answer,
        reasoning=str(payload.get("reasoning") or payload.get("rationale_summary") or ""),
        evidence_span=evidence,
        chunk_id=q.chunk_id,
        chunk_text=chunk_text,
        source_doc=str(q.metadata.get("source_doc") or ""),
        q_type=q.actual_q_type or q.q_type or qtype_for_intent(intent),
        intent_primary=intent,
        operations=list(q.operations),
        evidence_topology=q.evidence_topology or "single",
        evidence_state=q.evidence_state,
        expected_action=q.expected_action,
        answer_points=points or [spec.text for spec in q.answer_point_specs if spec.criticality == "required"],
        answer_point_specs=list(q.answer_point_specs),
        response_contract=q.response_contract,
        document_identity=q.document_identity,
        evidence_quotes=list(q.evidence_quotes) if q.evidence_quotes else [],
        requested_q_type=q.requested_q_type,
        actual_q_type=q.actual_q_type or q.q_type,
        type_label_origin=q.type_label_origin or "source",
        knowledge_id=q.knowledge_id,
        selection_role=q.metadata.get("selection_role") or "learning",
        requested_evidence=requested,
        located_evidence=evidence,
        repair_status=repair,
        family_id=q.family_id,
        source_family_id=q.source_family_id,
        source_hash=q.source_hash,
        requested_type=q.requested_type,
        actual_type=q.actual_type or q.intent_primary,
        verification_status=q.verification_status,
        generation_route=q.generation_route,
        construction_evidence_refs=[q.evidence_span] if q.evidence_span else [],
        teacher_context_refs=[chunk_text] if chunk_text else [],
        visible_support_refs=[q.evidence_span] if q.evidence_span and q.evidence_state == "sufficient" else [],
        student_context_refs=[chunk_text] if chunk_text and q.metadata.get("goal") != "closed_book_domain" else [],
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
            "generation_route": q.generation_route,
            "verification_status": q.metadata.get("verification_status"),
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
    response = ctx.llm.chat_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"参考文本：{chunk_text}\n问题：{q.question}"},
        ],
        model=model,
    )
    ctx.stats.call_statuses[response.status] = ctx.stats.call_statuses.get(response.status, 0) + 1
    data = json_payload(response)
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

    def __init__(self, supervise_rationale: bool = False, **_: object) -> None:
        self.supervise_rationale = bool(supervise_rationale)

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        out = []
        for q in questions:
            model = q.metadata.get("teacher_model") or ctx.model_for("default")
            cot = _needs_cot(q)
            pair = _to_pair(q, _ask(q, ctx, cot=cot, model=model), model, cot)
            if self.supervise_rationale and pair.located_evidence and pair.repair_status == "located":
                pair.metadata["assistant_target"] = f"{pair.answer}\n依据：{pair.located_evidence}"
                pair.metadata["rationale_audited"] = True
                pair.generation_trace["supervise_rationale"] = True
            out.append(pair)
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
                score = json_payload(ctx.llm.chat_json(
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
                ))
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


@register("distillation", "reuse_candidate")
class ReuseCandidate:
    """候选答案已带可定位证据时直接采用，不再请教师重写同一答案。"""

    name = "reuse_candidate"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, questions: list[Question], ctx) -> list[QAPair]:
        out = []
        for q in questions:
            chunk_text = q.metadata.get("chunk_text") or ""
            hint = (q.answer_hint or "").strip()
            evidence = (q.evidence_span or "").strip()
            if hint and evidence and is_substring(evidence, chunk_text):
                model = q.metadata.get("teacher_model") or ctx.model_for("default")
                pair = _to_pair(
                    q,
                    {"answer": hint, "evidence_span": evidence, "confidence": 0.8},
                    model,
                    False,
                )
                pair.generation_trace["operation"] = "reuse_candidate"
                pair.generation_trace["reused_candidate"] = True
                pair.log("distillation", self.name, reused=True)
                out.append(pair)
                continue
            model = q.metadata.get("teacher_model") or ctx.model_for("default")
            out.append(_to_pair(q, _ask(q, ctx, cot=False, model=model), model, False))
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
