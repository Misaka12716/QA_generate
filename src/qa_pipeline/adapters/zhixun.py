"""与智训平台 JSONL 互转（不修改 data_governance 代码）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..schemas import Chunk, Document, QAPair
from ..store import json_line, read_jsonl, write_jsonl
from ..textutil import is_substring, normalize

GROUNDED_POLICY = "仅依据提供的资料回答；资料不足时说明缺少的信息。"
CLOSED_POLICY = "可以运用已有领域知识回答。不确定时说明依据不足。"


class ReleaseRejected(Exception):
    """样本未通过发布资格检查。"""


def ensure_evidence(pair: QAPair) -> str:
    ev = (pair.located_evidence or pair.evidence_span or "").strip()
    if ev and is_substring(ev, pair.chunk_text):
        return ev
    return ""


def validation_subject(pair: QAPair) -> dict[str, Any]:
    goal = pair.goal or "rag_grounded"
    policy = CLOSED_POLICY if goal == "closed_book_domain" else GROUNDED_POLICY
    return {
        "goal": goal,
        "policy": policy,
        "question": pair.question,
        "student_context": _student_context(pair) if goal != "closed_book_domain" else "",
        "answer": assistant_content(pair),
        "expected_action": pair.expected_action,
        "evidence_span": pair.located_evidence or pair.evidence_span,
        "visible_support_refs": list(pair.visible_support_refs),
    }


def bind_validation_hash(pair: QAPair) -> str:
    digest = hashlib.sha256(
        json.dumps(validation_subject(pair), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if pair.validation_subject_hash and pair.validation_subject_hash != digest:
        raise ReleaseRejected("validation_subject_hash 与最终训练对象不一致，需要重新验证")
    pair.validation_subject_hash = digest
    return digest


def assistant_content(pair: QAPair) -> str:
    target = pair.metadata.get("assistant_target")
    if target:
        return str(target)
    return pair.answer


def build_messages(pair: QAPair, *, include_assistant: bool = True) -> list[dict[str, str]]:
    goal = pair.goal or "rag_grounded"
    if goal == "closed_book_domain":
        messages = [
            {"role": "system", "content": CLOSED_POLICY},
            {"role": "user", "content": pair.question},
        ]
    else:
        context = _student_context(pair)
        if context:
            user = f"资料：\n{context}\n\n问题：{pair.question}"
        else:
            user = f"资料：\n（当前未提供可回答该问题的资料）\n\n问题：{pair.question}"
        messages = [
            {"role": "system", "content": GROUNDED_POLICY},
            {"role": "user", "content": user},
        ]
    if include_assistant:
        messages.append({"role": "assistant", "content": assistant_content(pair)})
    return messages


def assert_releasable(pair: QAPair) -> None:
    if pair.grade in {None, "B", "reject", "quarantine"} or pair.action in {"reject", "quarantine", "needs_escalation"}:
        raise ReleaseRejected(f"样本 {pair.qa_id} 未达到发布等级")
    if pair.metadata.get("verification_status") == "pending" or pair.verification_status == "pending":
        raise ReleaseRejected(f"样本 {pair.qa_id} 仍待核验")
    if pair.grade not in {"S", "A"}:
        raise ReleaseRejected(f"样本 {pair.qa_id} 等级不可发布")
    bind_validation_hash(pair)


def _student_context(pair: QAPair) -> str:
    if "student_context" in pair.metadata:
        return str(pair.metadata.get("student_context") or "")
    if pair.student_context_refs:
        return "\n".join(pair.student_context_refs)
    return pair.chunk_text


def to_zhixun_row(pair: QAPair, split: str | None = None, *, bind: bool = True) -> dict[str, Any]:
    evidence = ensure_evidence(pair)
    goal = pair.goal or "rag_grounded"
    messages = build_messages(pair, include_assistant=True)
    if bind and not pair.validation_subject_hash:
        bind_validation_hash(pair)
    return {
        "id": pair.qa_id,
        "split": split or pair.split,
        "messages": messages,
        "metadata": {
            "source": pair.source_doc,
            "source_file_id": pair.metadata.get("source_file_id"),
            "location": " / ".join(pair.metadata.get("title_path") or []) or pair.chunk_id,
            "evidence": evidence,
            "goal": goal,
            "evidence_state": pair.evidence_state,
            "expected_action": pair.expected_action,
            "generation_route": pair.generation_route,
            "data_stage": pair.data_stage,
            "validation_subject_hash": pair.validation_subject_hash,
            "intent_primary": pair.intent_primary,
            "selection_role": pair.selection_role,
            "kind": {
                "factual": "事实问答",
                "procedural": "步骤说明",
                "conditional": "条件问答",
                "comparative": "比较问答",
                "multihop": "多跳问答",
                "explanatory": "步骤说明",
                "reasoning": "多跳问答",
            }.get(pair.q_type, "事实问答"),
            "grade": pair.grade,
            "chunk_id": pair.chunk_id,
            "q_type": pair.q_type,
            "generation_trace": pair.generation_trace,
            "filter_trace": pair.filter_trace,
            "nli_score": pair.nli_score,
            "judge_scores": pair.judge_scores,
            "judge_overall": pair.judge_overall,
            "kb_gain": pair.kb_gain,
            "teacher_model": pair.teacher_model,
            "review_history": pair.audit,
            "family_id": pair.family_id,
            "source_family_id": pair.source_family_id,
            "task_variant_id": pair.task_variant_id,
            "parent_id": pair.parent_sample_id,
        },
    }


def export_zhixun(pairs: Iterable[QAPair], path: str | Path, split: str | None = None) -> Path:
    path = Path(path)
    rows = []
    for pair in pairs:
        assert_releasable(pair)
        if pair.data_stage in {None, "accepted", "selected"}:
            pair.data_stage = "released"
        rows.append(to_zhixun_row(pair, split=split, bind=False))
    write_jsonl(path, rows)
    return path


def from_zhixun_blocks(blocks: Iterable[dict[str, Any]]) -> list[Chunk]:
    """把智训 Block/TaskBlock 字典转成 Chunk。"""
    chunks: list[Chunk] = []
    for i, b in enumerate(blocks):
        text = normalize(str(b.get("content") or b.get("text") or ""))
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=str(b.get("block_id") or b.get("id") or f"blk_{i:04d}"),
                text=text,
                doc_id=str(b.get("file_id") or b.get("doc_id") or ""),
                source_doc=str(b.get("file_name") or b.get("source") or ""),
                title_path=[p for p in str(b.get("loc") or "").split("·") if p.strip()],
                metadata={k: b.get(k) for k in ("loc", "page", "kind") if b.get(k) is not None},
            )
        )
    return chunks


def documents_from_zhixun_blocks(blocks: Iterable[dict[str, Any]]) -> list[Document]:
    grouped: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    for b in blocks:
        key = str(b.get("file_id") or b.get("file_name") or "doc")
        titles[key] = str(b.get("file_name") or key)
        grouped.setdefault(key, []).append(str(b.get("content") or ""))
    return [
        Document(doc_id=k, title=titles[k], text=normalize("\n\n".join(v)), metadata={"source": "zhixun_blocks"})
        for k, v in grouped.items()
    ]


def load_zhixun_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(path))


def zhixun_to_pairs(rows: Iterable[dict[str, Any]]) -> list[QAPair]:
    pairs = []
    for row in rows:
        msgs = row.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        meta = row.get("metadata") or {}
        pairs.append(
            QAPair(
                qa_id=str(row.get("id") or ""),
                question=user,
                answer=asst,
                evidence_span=str(meta.get("evidence") or ""),
                source_doc=str(meta.get("source") or ""),
                chunk_id=str(meta.get("chunk_id") or ""),
                grade=meta.get("grade"),
                split=row.get("split") or "train",
                filter_trace=meta.get("filter_trace") or {},
                generation_trace=meta.get("generation_trace") or {},
                metadata=meta,
            )
        )
    return pairs
