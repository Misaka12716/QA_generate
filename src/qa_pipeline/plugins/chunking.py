"""Chunking 策略：fixed_overlap / heading_window / semantic_boundary。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Chunk, Document, refresh_chunk_id
from ..textutil import approx_tokens, heading_sections, sentences, sliding_windows

_MIN_TOKENS = 32
_MAX_TOKENS = 2048


def _complete_piece(text: str) -> tuple[str, str]:
    """尽量在句号处结束。无法保全规则或例外时标记，而不是静默截断结论。"""
    body = text.strip()
    if not body:
        return "", ""
    if body[-1] in "。！？!?；;":
        return body, ""
    parts = sentences(body)
    if len(parts) >= 2 and parts[-1] and parts[-1][-1] not in "。！？!?；;":
        kept = "".join(parts[:-1]).strip()
        if kept:
            return kept, "boundary_split"
    return body, "incomplete_boundary"


def _emit(
    doc: Document,
    text: str,
    path: list[str],
    start: int,
    end: int,
    strategy: str,
    params: dict | None = None,
) -> Chunk | None:
    body, boundary = _complete_piece(text)
    if not body:
        return None
    tokens = approx_tokens(body)
    if tokens < _MIN_TOKENS and path:
        return None
    identity = {
        "generic_name": doc.metadata.get("generic_name") or "",
        "strength": doc.metadata.get("strength") or "",
        "dosage_form": doc.metadata.get("dosage_form") or "",
        "version": doc.source_version or doc.metadata.get("version") or "",
    }
    chunk = Chunk(
        text=body,
        doc_id=doc.doc_id,
        source_doc=doc.title or doc.path or doc.doc_id,
        title_path=path,
        char_start=start,
        char_end=start + len(body),
        token_count=tokens,
        source_hash=doc.source_hash,
        source_family_id=doc.source_family_id,
        dataset_version=doc.dataset_version,
        tokenizer_id=doc.tokenizer_id or "approx-v1",
        location=" / ".join(path),
        metadata={
            "chunking": strategy,
            "source_group": doc.source_group or doc.doc_id,
            "clean_version": doc.clean_version,
            "chunk_params": params or {},
            "identity": identity,
            "boundary_note": boundary,
            "offset_origin": start,
        },
    )
    return chunk


def _merge_short(chunks: list[Chunk], min_tokens: int = 128) -> list[Chunk]:
    if not chunks:
        return []
    out: list[Chunk] = []
    buf: Chunk | None = None
    for ch in chunks:
        if buf is None:
            buf = ch
            continue
        same = buf.doc_id == ch.doc_id and buf.title_path == ch.title_path
        if buf.token_count < min_tokens and same:
            buf.text = (buf.text + "\n\n" + ch.text).strip()
            buf.char_end = ch.char_end
            buf.token_count = approx_tokens(buf.text)
            refresh_chunk_id(buf)
        else:
            out.append(buf)
            buf = ch
    if buf:
        out.append(buf)
    return [c for c in out if c.token_count >= _MIN_TOKENS]


@register("chunking", "fixed_overlap")
class FixedOverlapChunking:
    """对齐智训平台：按字符窗 + overlap 切分。"""

    name = "fixed_overlap"

    def __init__(self, split: int = 600, overlap: int = 80, **_: object) -> None:
        self.split = int(split)
        self.overlap = int(overlap)
        if self.overlap >= self.split:
            raise ValueError("overlap 必须小于 split")

    def run(self, docs: list[Document], ctx) -> list[Chunk]:
        out: list[Chunk] = []
        step = max(1, self.split - self.overlap)
        for doc in docs:
            text = doc.text
            if not text:
                continue
            i = 0
            idx = 0
            while i < len(text):
                piece = text[i : i + self.split]
                ch = _emit(doc, piece, [], i, i + len(piece), self.name)
                if ch:
                    ch.metadata["idx"] = idx
                    out.append(ch)
                    idx += 1
                if i + self.split >= len(text):
                    break
                i += step
        return _merge_short(out, min_tokens=40)


@register("chunking", "heading_window")
class HeadingWindowChunking:
    """标题感知 + 滑窗，设计稿默认切分。"""

    name = "heading_window"

    def __init__(self, max_tokens: int = 768, overlap: float = 0.12, min_tokens: int = 128, **_: object) -> None:
        self.max_tokens = int(max_tokens)
        self.overlap = float(overlap)
        self.min_tokens = int(min_tokens)

    def run(self, docs: list[Document], ctx) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            cursor = 0
            for path, body in heading_sections(doc.text):
                start = doc.text.find(body, cursor)
                if start < 0:
                    start = cursor
                if approx_tokens(body) <= self.max_tokens:
                    ch = _emit(doc, body, path, start, start + len(body), self.name)
                    if ch:
                        out.append(ch)
                else:
                    for s, e, piece in sliding_windows(body, self.max_tokens, self.overlap):
                        ch = _emit(doc, piece, path, start + s, start + e, self.name)
                        if ch:
                            out.append(ch)
                cursor = start + len(body)
        merged = _merge_short(out, min_tokens=self.min_tokens)
        kept = []
        for chunk in merged:
            if chunk.token_count <= _MAX_TOKENS:
                kept.append(chunk)
                continue
            chunk.metadata["dropped_reason"] = "over_max_tokens"
            if ctx is not None:
                dropped = ctx.extras.setdefault("dropped_chunks", [])
                dropped.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "reason": "over_max_tokens",
                        "token_count": chunk.token_count,
                    }
                )
        return kept


@register("chunking", "semantic_boundary")
class SemanticBoundaryChunking:
    """段落 embedding 转折处切分；无向量模型时退回 heading_window。"""

    name = "semantic_boundary"

    def __init__(
        self,
        max_tokens: int = 768,
        min_tokens: int = 128,
        drop_threshold: float = 0.35,
        **_: object,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self.drop_threshold = float(drop_threshold)

    def run(self, docs: list[Document], ctx) -> list[Chunk]:
        from ..embeddings import get_embedder
        from ..textutil import paragraphs

        embedder = get_embedder()
        if embedder.backend != "bge":
            ctx.note_fallback("semantic_boundary:backend=tfidf")
        out: list[Chunk] = []
        heading = HeadingWindowChunking(max_tokens=self.max_tokens, min_tokens=self.min_tokens)
        for doc in docs:
            for path, body in heading_sections(doc.text):
                paras = paragraphs(body)
                if len(paras) <= 1:
                    ch = _emit(doc, body, path, 0, len(body), self.name)
                    if ch:
                        out.append(ch)
                    continue
                vecs = embedder.encode(paras)
                groups: list[list[str]] = [[paras[0]]]
                for i in range(1, len(paras)):
                    sim = float(vecs[i] @ vecs[i - 1])
                    current_tokens = approx_tokens("\n\n".join(groups[-1]))
                    if sim < self.drop_threshold or current_tokens >= self.max_tokens:
                        groups.append([paras[i]])
                    else:
                        groups[-1].append(paras[i])
                offset = 0
                for g in groups:
                    text = "\n\n".join(g)
                    ch = _emit(doc, text, path, offset, offset + len(text), self.name)
                    if ch:
                        ch.metadata["backend"] = embedder.backend
                        out.append(ch)
                    offset += len(text)
        if not out:
            ctx.note_fallback("semantic_boundary:empty,fallback=heading_window")
            return heading.run(docs, ctx)
        return _merge_short(out, min_tokens=self.min_tokens)
