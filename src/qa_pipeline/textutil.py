"""轻量文本工具：切分、近似 token、规范化、相似度。不依赖重模型。"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def approx_tokens(text: str) -> int:
    """中英混合粗估：汉字按字，英文按词。"""
    if not text:
        return 0
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return max(1, han + latin)


def normalize(text: str) -> str:
    text = re.sub(r"!\[\]\([^)]*\)", "", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(text: str) -> str:
    return _WS.sub("", text or "")


def sha1(text: str) -> str:
    return hashlib.sha1(compact(text).encode("utf-8")).hexdigest()


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def tokenize(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        return [t.strip() for t in jieba.lcut(text or "") if t.strip()]
    except Exception:
        return _TOKEN.findall(text or "")


def ngrams(tokens: list[str], n: int = 3) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rouge_l(hyp: str, ref: str) -> float:
    """字符级 LCS / 较短串长度，用作问题去重。"""
    a, b = compact(hyp), compact(ref)
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[m]
    return lcs / min(n, m)


def token_f1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if compact(pred) == compact(gold) and pred.strip() else 0.0


def is_substring(needle: str, haystack: str) -> bool:
    n, h = compact(needle), compact(haystack)
    return bool(n) and n in h


def longest_overlap_span(needle: str, haystack: str, min_len: int = 8) -> str:
    """在 haystack 中找与 needle 最长的连续子串，用作 evidence 回退。"""
    n, h = compact(needle), haystack
    if not n:
        return ""
    raw = haystack
    compact_map = []
    buf = []
    for i, ch in enumerate(raw):
        if not ch.isspace():
            compact_map.append(i)
            buf.append(ch)
    packed = "".join(buf)
    if n in packed:
        i = packed.find(n)
        start, end = compact_map[i], compact_map[i + len(n) - 1] + 1
        return raw[start:end].strip()
    best = ""
    window = min(len(n), 80)
    for size in range(window, min_len - 1, -4):
        for i in range(0, len(n) - size + 1, max(1, size // 4)):
            frag = n[i : i + size]
            if frag in packed and size > len(compact(best)):
                j = packed.find(frag)
                start, end = compact_map[j], compact_map[j + size - 1] + 1
                best = raw[start:end].strip()
                break
        if best:
            break
    return best


def heading_sections(text: str) -> list[tuple[list[str], str]]:
    """按 Markdown 标题切成 (title_path, body) 段。无标题则整篇一段。"""
    matches = list(_HEADING.finditer(text or ""))
    if not matches:
        return [([], text or "")]
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(([], lead))
    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        path = [t for _, t in stack]
        if body:
            sections.append((path, body))
    return sections or [([], text or "")]


def sliding_windows(text: str, max_tokens: int, overlap: float) -> list[tuple[int, int, str]]:
    """按近似 token 滑窗，返回 (char_start, char_end, piece)。"""
    if not text.strip():
        return []
    tokens = list(_TOKEN.finditer(text))
    if not tokens:
        return [(0, len(text), text)]
    step = max(1, int(max_tokens * (1 - overlap)))
    windows = []
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + max_tokens)
        start = tokens[i].start()
        end = tokens[j - 1].end()
        windows.append((start, end, text[start:end].strip()))
        if j >= len(tokens):
            break
        i += step
    return [w for w in windows if w[2]]


def entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log(p, 2)
    return max(0.0, h)
