"""轻量文本工具：切分、近似 token、规范化、相似度。不依赖重模型。"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from typing import Any, Callable

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
_MANUAL_HEADING = re.compile(r"^【([^】\n]{1,40})】\s*$", re.M)
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
SCORING_TOKENIZER_ID = "locked-char-v1"
_REPLACEMENT = "\ufffd"


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
    """评分与去重使用锁定分词，不随 jieba 是否安装而改变口径。"""
    return _TOKEN.findall(text or "")


def student_token_count(text: str, tokenizer=None) -> int:
    """学生序列长度。传入真实 tokenizer 时用其计数，否则退回近似长度。"""
    if tokenizer is not None:
        return len(tokenizer.encode(text or ""))
    return approx_tokens(text)


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


def _heading_matches(text: str) -> list[tuple[int, int, int, str]]:
    """返回 (start, end, level, title)。Markdown 用井号级数，说明书栏目为 2 级。"""
    found: list[tuple[int, int, int, str]] = []
    for match in _MD_HEADING.finditer(text or ""):
        found.append((match.start(), match.end(), len(match.group(1)), match.group(2).strip()))
    for match in _MANUAL_HEADING.finditer(text or ""):
        found.append((match.start(), match.end(), 2, match.group(1).strip()))
    found.sort(key=lambda item: item[0])
    return found


def heading_sections(text: str) -> list[tuple[list[str], str]]:
    """按 Markdown 标题或说明书【栏目】切成 (title_path, body)。无标题则整篇一段。"""
    matches = _heading_matches(text or "")
    if not matches:
        return [([], text or "")]
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    if matches[0][0] > 0:
        lead = text[: matches[0][0]].strip()
        if lead:
            sections.append(([], lead))
    for idx, (start, end, level, title) in enumerate(matches):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        body_end = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        body = text[end:body_end].strip()
        path = [item for _, item in stack]
        if body:
            sections.append((path, body))
    return sections or [([], text or "")]


def offset_map(original: str, cleaned: str) -> list[int]:
    """清洗文本每个字符对应原文字符下标。无法对齐的位置记为 -1。"""
    mapping: list[int] = []
    i = 0
    for ch in cleaned:
        while i < len(original) and original[i] != ch:
            i += 1
        if i < len(original) and original[i] == ch:
            mapping.append(i)
            i += 1
        else:
            mapping.append(-1)
    return mapping


def char_ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    a = set(compact(left)[i : i + n] for i in range(max(0, len(compact(left)) - n + 1)))
    b = set(compact(right)[i : i + n] for i in range(max(0, len(compact(right)) - n + 1)))
    return jaccard(a, b)


def parse_quality(text: str) -> dict[str, Any]:
    """解析质量与“能否读出文本”分开。坏表格、缺主体、断句不进入正式测试。"""
    raw = text or ""
    readable = bool(raw.strip())
    reasons: list[str] = []
    if _REPLACEMENT in raw:
        reasons.append("replacement_char")
    if raw.count("目录") >= 3 and raw.count("【") <= 1:
        reasons.append("repeated_navigation")
    sections = heading_sections(raw)
    headings = [path[-1] for path, _body in sections if path]
    if readable and not headings and "【" not in raw and "#" not in raw:
        reasons.append("missing_section_boundary")
    if re.search(r"\d+(?:\.\d+)?\s*(?:mg|g|ml|μg).{0,8}\d+(?:\.\d+)?\s*(?:mg|g|ml|μg)", raw, re.I):
        if "规格" not in raw:
            reasons.append("strength_glued")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.count("|") >= 2]
    if pipe_lines:
        header = pipe_lines[0]
        if "---" not in raw and not re.search(r"[A-Za-z\u4e00-\u9fff]", header.replace("|", "")):
            reasons.append("table_missing_header")
        elif len(pipe_lines) >= 2 and any(cell.strip() == "" for cell in pipe_lines[1].split("|")):
            reasons.append("table_missing_row_key")
        if pipe_lines and max(line.count("|") for line in pipe_lines) >= 6:
            reasons.append("complex_table_isolated")
    if raw.rstrip().endswith(("，", ",", "、", "【")):
        reasons.append("truncated_sentence")
    identity = bool(re.search(r"(通用名称|药品名称|【成份】|【成分】)", raw))
    if readable and len(raw) > 400 and not identity and not headings:
        reasons.append("missing_subject")
    return {
        "readable": readable,
        "parse_ok": readable and not reasons,
        "reasons": reasons,
        "heading_count": len(headings),
    }


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


def balanced_take(items: list, limit: int | None, keyfn: Callable, seed: int = 0) -> list:
    """按来源轮转截取工作清单，而不是保留输入前缀。"""
    if not limit or limit <= 0 or len(items) <= int(limit):
        return list(items)
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(str(keyfn(item)), []).append(item)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out: list = []
    while len(out) < int(limit) and any(groups.values()):
        for key in keys:
            bucket = groups[key]
            if bucket and len(out) < int(limit):
                out.append(bucket.pop(0))
    return out


def entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log(p, 2)
    return max(0.0, h)
