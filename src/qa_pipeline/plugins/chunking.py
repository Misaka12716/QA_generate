"""Chunking 策略：fixed_overlap / heading_window / semantic_boundary。"""

from __future__ import annotations

from ..registry import register
from ..schemas import Chunk, Document
from ..textutil import approx_tokens, heading_sections, sliding_windows

_MIN_TOKENS = 32
_MAX_TOKENS = 2048


def _emit(doc: Document, text: str, path: list[str], start: int, end: int, strategy: str) -> Chunk | None:
    body = text.strip()
    if not body:
        return None
    tokens = approx_tokens(body)
    if tokens < _MIN_TOKENS and path:
        return None
    return Chunk(
        text=body,
        doc_id=doc.doc_id,
        source_doc=doc.title or doc.path or doc.doc_id,
        title_path=path,
        char_start=start,
        char_end=end,
        token_count=tokens,
        metadata={"chunking": strategy},
    )


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
        return [c for c in merged if c.token_count <= _MAX_TOKENS or True]


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
