"""锚点抽取：none / tfidf_keyword / ner_rake_textrank / llm_extract。"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from ..llm import json_payload
from ..registry import register
from ..schemas import Anchor, Chunk
from ..textutil import sentences, tokenize

_ENTITY = re.compile(
    r"[A-Z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,12}(?:系统|模块|接口|告警|装置|规程|参数|协议)"
)
_STOP = {
    "以及", "如果", "然后", "可以", "进行", "这个", "一个", "我们", "他们", "不是",
    "需要", "应该", "对于", "由于", "同时", "通过", "出现", "使用", "相关", "包括",
}


def _pos(text: str, span: str) -> int:
    return max(0, text.find(span))


def _attach(chunk: Chunk, anchors: list[Anchor], extra: dict | None = None) -> Chunk:
    chunk.metadata["anchors"] = [a.model_dump() for a in anchors]
    if extra:
        chunk.metadata.update(extra)
    return chunk


@register("anchor", "none")
class NoAnchor:
    name = "none"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        for c in chunks:
            _attach(c, [])
        return chunks


@register("anchor", "tfidf_keyword")
class TfidfKeywordAnchor:
    name = "tfidf_keyword"

    def __init__(self, top_k: int = 5, **_: object) -> None:
        self.top_k = int(top_k)

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [c.text for c in chunks] or [""]
        vec = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
            max_features=4000,
        )
        try:
            mat = vec.fit_transform(corpus)
        except ValueError:
            for c in chunks:
                _attach(c, [])
            return chunks
        vocab = vec.get_feature_names_out()
        for i, c in enumerate(chunks):
            row = mat[i].toarray().ravel()
            idxs = row.argsort()[::-1]
            anchors: list[Anchor] = []
            for j in idxs:
                term = str(vocab[j])
                if term in _STOP or len(term) < 2 or row[j] <= 0:
                    continue
                anchors.append(
                    Anchor(
                        anchor_text=term,
                        anchor_type="keyword",
                        position_in_chunk=_pos(c.text, term),
                        chunk_id=c.chunk_id,
                        score=float(row[j]),
                    )
                )
                if len(anchors) >= self.top_k:
                    break
            _attach(c, anchors)
        return chunks


def _rake_phrases(text: str, top_k: int = 5) -> list[tuple[str, float]]:
    tokens = tokenize(text)
    phrases: list[list[str]] = []
    cur: list[str] = []
    for t in tokens:
        if t in _STOP or re.fullmatch(r"[\d.]+", t):
            if cur:
                phrases.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        phrases.append(cur)
    freq: dict[str, int] = defaultdict(int)
    degree: dict[str, int] = defaultdict(int)
    for ph in phrases:
        for t in ph:
            freq[t] += 1
            degree[t] += len(ph)
    scored = []
    for ph in phrases:
        if not ph:
            continue
        score = sum(degree[t] / max(1, freq[t]) for t in ph)
        phrase = "".join(ph) if all(len(t) == 1 or "\u4e00" <= t[0] <= "\u9fff" for t in ph) else " ".join(ph)
        if 2 <= len(phrase) <= 20:
            scored.append((phrase, score))
    scored.sort(key=lambda x: -x[1])
    seen = set()
    out = []
    for p, s in scored:
        if p in seen:
            continue
        seen.add(p)
        out.append((p, s))
        if len(out) >= top_k:
            break
    return out


def _textrank_sentences(text: str, top_k: int = 2) -> list[tuple[str, float]]:
    sents = [s for s in sentences(text) if 8 <= len(s) <= 120]
    if not sents:
        return []
    n = len(sents)
    if n == 1:
        return [(sents[0], 1.0)]
    toks = [set(tokenize(s)) for s in sents]
    scores = [1.0] * n
    for _ in range(12):
        nxt = [0.15] * n
        for i in range(n):
            for j in range(n):
                if i == j or not toks[i] or not toks[j]:
                    continue
                w = len(toks[i] & toks[j]) / math.sqrt(len(toks[i]) * len(toks[j]))
                nxt[i] += 0.85 * w * scores[j]
        scores = nxt
    ranked = sorted(zip(sents, scores), key=lambda x: -x[1])
    return ranked[:top_k]


@register("anchor", "ner_tfidf")
class NerTfidfAnchor:
    """实体 + TF-IDF，不含关键句。对应锚点消融 C3。"""

    name = "ner_tfidf"

    def __init__(self, top_k: int = 5, **_: object) -> None:
        self.top_k = int(top_k)

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        TfidfKeywordAnchor(top_k=self.top_k).run(chunks, ctx)
        for c in chunks:
            keywords = _anchors_from_meta(c)
            ents = []
            for m in _ENTITY.finditer(c.text):
                t = m.group(0)
                if t in _STOP:
                    continue
                ents.append(
                    Anchor(
                        anchor_text=t,
                        anchor_type="entity",
                        position_in_chunk=m.start(),
                        chunk_id=c.chunk_id,
                        score=1.0,
                    )
                )
            merged: list[Anchor] = []
            seen = set()
            for a in ents + keywords:
                if a.anchor_type == "sentence":
                    continue
                key = a.anchor_text.strip()
                if key in seen or len(key) < 2:
                    continue
                seen.add(key)
                merged.append(a)
                if len(merged) >= self.top_k:
                    break
            _attach(c, merged, {"anchor_backend": "ner_tfidf"})
        return chunks


def _anchors_from_meta(chunk: Chunk) -> list[Anchor]:
    raw = chunk.metadata.get("anchors") or []
    out = []
    for item in raw:
        try:
            out.append(Anchor.model_validate(item) if not isinstance(item, Anchor) else item)
        except Exception:
            continue
    return out


@register("anchor", "ner_rake_textrank")
class NerRakeTextrankAnchor:
    name = "ner_rake_textrank"

    def __init__(self, top_k: int = 5, **_: object) -> None:
        self.top_k = int(top_k)

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        backend = "heuristic"
        try:
            import jieba  # noqa: F401

            backend = "jieba"
        except Exception:
            ctx.note_fallback("anchor.ner_rake_textrank:backend=heuristic")
        for c in chunks:
            ents = []
            for m in _ENTITY.finditer(c.text):
                t = m.group(0)
                if t not in _STOP:
                    ents.append(Anchor(anchor_text=t, anchor_type="entity", position_in_chunk=m.start(), chunk_id=c.chunk_id, score=1.0))
            kws = [
                Anchor(anchor_text=p, anchor_type="keyword", position_in_chunk=_pos(c.text, p), chunk_id=c.chunk_id, score=s)
                for p, s in _rake_phrases(c.text, self.top_k)
            ]
            cores = [
                Anchor(anchor_text=s, anchor_type="sentence", position_in_chunk=_pos(c.text, s), chunk_id=c.chunk_id, score=sc)
                for s, sc in _textrank_sentences(c.text, 2)
            ]
            merged: list[Anchor] = []
            seen = set()
            for a in ents + kws + cores:
                key = a.anchor_text.strip()
                if key in seen or len(key) < 2:
                    continue
                seen.add(key)
                merged.append(a)
                if len(merged) >= self.top_k:
                    break
            _attach(c, merged, {"anchor_backend": backend})
        return chunks


@register("anchor", "llm_extract")
class LLMExtractAnchor:
    name = "llm_extract"

    def __init__(self, top_k: int = 5, **_: object) -> None:
        self.top_k = int(top_k)

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        for c in chunks:
            data = json_payload(ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是锚点抽取器。从文本块中抽取关键实体、术语或中心句。"
                            f'只输出 JSON：{{"anchors":[{{"anchor_text":"...","anchor_type":"entity|keyword|sentence","position_in_chunk":0}}]}}'
                            f"最多 {self.top_k} 个。"
                        ),
                    },
                    {"role": "user", "content": f"文本块：{c.text[:3000]}"},
                ],
                model=ctx.model_for("cheap"),
            ))
            anchors = []
            for i, item in enumerate((data.get("anchors") or [])[: self.top_k]):
                if not isinstance(item, dict) or not item.get("anchor_text"):
                    continue
                at = item.get("anchor_type") if item.get("anchor_type") in {"entity", "keyword", "sentence"} else "keyword"
                anchors.append(
                    Anchor(
                        anchor_text=str(item["anchor_text"])[:40],
                        anchor_type=at,
                        position_in_chunk=int(item.get("position_in_chunk") or _pos(c.text, str(item["anchor_text"]))),
                        chunk_id=c.chunk_id,
                        score=1.0 - i * 0.1,
                    )
                )
            if not anchors:
                ctx.note_fallback("anchor.llm_extract:empty,fallback=tfidf")
                TfidfKeywordAnchor(top_k=self.top_k).run([c], ctx)
            else:
                _attach(c, anchors, {"anchor_backend": "llm"})
        return chunks
