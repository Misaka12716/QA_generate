"""Question 生成：direct_qa / anchor_reverse / self_instruct / answer_aware。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Q_TYPES, Anchor, Chunk, Question, canon_qtype
from ..textutil import is_substring, longest_overlap_span, rouge_l, sentences, tokenize

_TYPE_CYCLE = list(Q_TYPES)
_QTYPE_DOC = "factual|procedural|conditional|comparative|multihop"


def _anchors_of(chunk: Chunk) -> list[Anchor]:
    raw = chunk.metadata.get("anchors") or []
    out = []
    for item in raw:
        try:
            out.append(Anchor.model_validate(item) if not isinstance(item, Anchor) else item)
        except Exception:
            continue
    return out


def _valid_question(text: str) -> bool:
    n = len(tokenize(text))
    return 5 <= n <= 128 and bool(text.strip())


def _dedup(questions: list[Question], threshold: float = 0.75) -> list[Question]:
    kept: list[Question] = []
    for q in questions:
        if any(rouge_l(q.question, k.question) > threshold for k in kept):
            continue
        kept.append(q)
    return kept


def _clip_evidence(span: str, chunk: Chunk) -> str:
    span = (span or "").strip()
    if span and is_substring(span, chunk.text):
        return span
    return longest_overlap_span(span or chunk.text[:80], chunk.text) or (sentences(chunk.text)[0] if sentences(chunk.text) else "")


def _quota_type(i: int) -> str:
    """设计稿默认配额：事实 40% / 步骤 20% / 条件 15% / 比较 15% / 多跳 10%。"""
    table = (
        ["factual"] * 8
        + ["procedural"] * 4
        + ["conditional"] * 3
        + ["comparative"] * 3
        + ["multihop"] * 2
    )
    return table[i % len(table)]


def _as_qtype(value: object, index: int) -> str:
    if isinstance(value, str) and canon_qtype(value, default="") in _TYPE_CYCLE and value.strip() in {
        *_TYPE_CYCLE,
        "explanatory",
        "reasoning",
    }:
        return canon_qtype(value)
    return _quota_type(index)


@register("question_gen", "direct_qa")
class DirectQA:
    """智训现状：一次由教师同时生成 Q + A。"""

    name = "direct_qa"

    def __init__(self, per_chunk: int | None = None, **_: object) -> None:
        self.per_chunk = per_chunk

    def run(self, chunks: list[Chunk], ctx) -> list[Question]:
        n = int(self.per_chunk or ctx.recipe.questions_per_chunk)
        out: list[Question] = []
        for chunk in chunks:
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是训练数据构造助手。生成问答对：问题应是使用者会真实提出的，"
                            "答案必须完全由资料支持、准确简洁。硬性规则：每条必须给出证据原文"
                            "（evidence 必须是片段中的连续原文，不得改写）。"
                            '只输出 JSON：{"samples":[{"q":"...","a":"...","evidence":"...","kind":"事实问答"}]}'
                        ),
                    },
                    {"role": "user", "content": f"请基于以下资料片段构造 {n} 条样本：\n\n{chunk.text}"},
                ],
                model=ctx.model_for("default"),
            ) or {}
            samples = data.get("samples") or data.get("questions") or []
            for i, item in enumerate(samples[:n]):
                if not isinstance(item, dict):
                    continue
                qtext = str(item.get("q") or item.get("question") or "").strip()
                if not _valid_question(qtext):
                    continue
                kind = str(item.get("kind") or item.get("q_type") or "")
                q_type = "factual"
                if "多跳" in kind or "推理" in kind:
                    q_type = "multihop"
                elif "比较" in kind:
                    q_type = "comparative"
                elif "条件" in kind:
                    q_type = "conditional"
                elif "步骤" in kind or "流程" in kind:
                    q_type = "procedural"
                elif kind in _TYPE_CYCLE or kind in {"explanatory", "reasoning"}:
                    q_type = canon_qtype(kind)
                ev = _clip_evidence(str(item.get("evidence") or item.get("evidence_span") or ""), chunk)
                q = Question(
                    question=qtext,
                    chunk_id=chunk.chunk_id,
                    q_type=q_type,
                    evidence_span=ev,
                    answer_hint=str(item.get("a") or item.get("answer") or "").strip(),
                    metadata={"source_doc": chunk.source_doc, "chunk_text": chunk.text, "strategy": self.name},
                )
                out.append(q)
        return _dedup(out)


@register("question_gen", "anchor_reverse")
class AnchorReverseQG:
    """锚点驱动反向提问，设计稿主管线。"""

    name = "anchor_reverse"

    def __init__(self, per_anchor: int = 2, **_: object) -> None:
        self.per_anchor = int(per_anchor)

    def run(self, chunks: list[Chunk], ctx) -> list[Question]:
        out: list[Question] = []
        existing: list[str] = []
        for chunk in chunks:
            anchors = _anchors_of(chunk)
            if not anchors:
                anchors = [Anchor(anchor_text=chunk.title_path[-1] if chunk.title_path else "本段", chunk_id=chunk.chunk_id)]
            payload = [{"anchor_text": a.anchor_text, "anchor_type": a.anchor_type} for a in anchors]
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业的QA数据生成器。请根据给定的文本块和锚点，生成能够从该文本块中找到答案的问题。\n"
                            "要求：\n1. 答案必须完全来自给定文本，不能引入外部知识\n"
                            "2. 每个锚点生成不同类型的问题，q_type 只能是 "
                            f"{_QTYPE_DOC}（事实/步骤/条件/比较/多跳）\n"
                            "3. 问题要自然、清晰，不重复，且只能依据文本块回答\n"
                            '4. 输出 JSON：{"questions":[{"question":"...","anchor":"...","q_type":"factual","evidence_span":"原文中的答案片段"}]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"文本块：{chunk.text}\n锚点列表：{payload}\n"
                            f"已有问题（避免重复）：{existing[-12:]}\n每个锚点最多 {self.per_anchor} 题。"
                        ),
                    },
                ],
                model=ctx.model_for("default"),
            ) or {}
            for i, item in enumerate(data.get("questions") or []):
                if not isinstance(item, dict):
                    continue
                qtext = str(item.get("question") or "").strip()
                if not _valid_question(qtext):
                    continue
                q_type = _as_qtype(item.get("q_type"), i)
                q = Question(
                    question=qtext,
                    chunk_id=chunk.chunk_id,
                    anchor=str(item.get("anchor") or (anchors[0].anchor_text if anchors else "")),
                    q_type=q_type,
                    evidence_span=_clip_evidence(str(item.get("evidence_span") or ""), chunk),
                    metadata={"source_doc": chunk.source_doc, "chunk_text": chunk.text, "strategy": self.name},
                )
                out.append(q)
                existing.append(qtext)
        return _dedup(out)


@register("question_gen", "self_instruct")
class SelfInstructQG:
    name = "self_instruct"

    def __init__(self, per_chunk: int | None = None, **_: object) -> None:
        self.per_chunk = per_chunk

    def run(self, chunks: list[Chunk], ctx) -> list[Question]:
        n = int(self.per_chunk or ctx.recipe.questions_per_chunk)
        seeds = [
            "请根据资料解释该模块的核心功能。",
            "出现告警时应执行哪些步骤？",
            "该参数的取值范围是多少？",
        ]
        out: list[Question] = []
        for chunk in chunks:
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你在做 Self-Instruct：参考种子问题风格，基于文本块生成新的、互不重复的问题。"
                            "问题必须能由文本块回答。"
                            f'输出 JSON：{{"questions":[{{"question":"...","q_type":"{_QTYPE_DOC}","evidence_span":"..."}}]}}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"文本块：{chunk.text}\n种子问题：{seeds}\n请生成 {n} 个问题。",
                    },
                ],
                model=ctx.model_for("default"),
            ) or {}
            for i, item in enumerate((data.get("questions") or [])[:n]):
                if not isinstance(item, dict):
                    continue
                qtext = str(item.get("question") or "").strip()
                if not _valid_question(qtext):
                    continue
                out.append(
                    Question(
                        question=qtext,
                        chunk_id=chunk.chunk_id,
                        q_type=_as_qtype(item.get("q_type"), i),
                        evidence_span=_clip_evidence(str(item.get("evidence_span") or ""), chunk),
                        metadata={"source_doc": chunk.source_doc, "chunk_text": chunk.text, "strategy": self.name},
                    )
                )
        return _dedup(out)


@register("question_gen", "answer_aware")
class AnswerAwareQG:
    """先抽 evidence span，再反向生成问题。"""

    name = "answer_aware"

    def __init__(self, per_chunk: int | None = None, **_: object) -> None:
        self.per_chunk = per_chunk

    def run(self, chunks: list[Chunk], ctx) -> list[Question]:
        n = int(self.per_chunk or ctx.recipe.questions_per_chunk)
        out: list[Question] = []
        for chunk in chunks:
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "先从文本块抽取可作为答案的原文片段（答案跨度），再为每个片段生成一个问题。"
                            f'输出 JSON：{{"questions":[{{"question":"...","evidence_span":"...","q_type":"{_QTYPE_DOC}"}}]}}'
                        ),
                    },
                    {"role": "user", "content": f"文本块：{chunk.text}\n请抽取并提问 {n} 组。"},
                ],
                model=ctx.model_for("default"),
            ) or {}
            for i, item in enumerate((data.get("questions") or [])[:n]):
                if not isinstance(item, dict):
                    continue
                qtext = str(item.get("question") or "").strip()
                if not _valid_question(qtext):
                    continue
                out.append(
                    Question(
                        question=qtext,
                        chunk_id=chunk.chunk_id,
                        q_type=_as_qtype(item.get("q_type"), i),
                        evidence_span=_clip_evidence(str(item.get("evidence_span") or ""), chunk),
                        metadata={"source_doc": chunk.source_doc, "chunk_text": chunk.text, "strategy": self.name},
                    )
                )
        return _dedup(out)
