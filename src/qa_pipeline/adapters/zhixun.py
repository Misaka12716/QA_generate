"""与智训平台 JSONL 互转（不修改 data_governance 代码）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..schemas import Chunk, Document, QAPair
from ..store import json_line, read_jsonl, write_jsonl
from ..textutil import is_substring, longest_overlap_span, normalize


def ensure_evidence(pair: QAPair) -> str:
    ev = (pair.evidence_span or "").strip()
    if ev and is_substring(ev, pair.chunk_text):
        return ev
    recovered = longest_overlap_span(ev or pair.answer, pair.chunk_text)
    return recovered


def to_zhixun_row(pair: QAPair, split: str | None = None) -> dict[str, Any]:
    evidence = ensure_evidence(pair)
    return {
        "id": pair.qa_id,
        "split": split or pair.split,
        "messages": [
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.answer},
        ],
        "metadata": {
            "source": pair.source_doc,
            "source_file_id": pair.metadata.get("source_file_id"),
            "location": " / ".join(pair.metadata.get("title_path") or []) or pair.chunk_id,
            "evidence": evidence,
            "goal": "qa",
            "kind": {
                "factual": "事实问答",
                "explanatory": "步骤说明",
                "reasoning": "条件问答",
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
        },
    }


def export_zhixun(pairs: Iterable[QAPair], path: str | Path, split: str | None = None) -> Path:
    path = Path(path)
    rows = [to_zhixun_row(p, split=split) for p in pairs]
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
