"""可串联过滤策略。缺失重模型时自动降级并写入 backend=fallback。"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict

import numpy as np

from ..registry import register
from ..schemas import QAPair
from ..textutil import (
    compact,
    exact_match,
    is_substring,
    ngrams,
    sentences,
    sha1,
    token_f1,
    tokenize,
)

logger = logging.getLogger(__name__)


def _keep(pairs: list[QAPair], pred) -> list[QAPair]:
    out = []
    for p in pairs:
        if pred(p):
            out.append(p)
    return out


@register("filter", "rule_clean")
class RuleClean:
    name = "rule_clean"

    def __init__(self, min_answer_tokens: int = 8, max_answer_tokens: int = 2000, **_: object) -> None:
        self.min_answer_tokens = int(min_answer_tokens)
        self.max_answer_tokens = int(max_answer_tokens)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            n = len(tokenize(p.answer))
            reason = ""
            if n < self.min_answer_tokens or n > self.max_answer_tokens:
                reason = "length"
            elif not compact(p.answer) or set(p.answer) <= set("。.，,！!？?、-"):
                reason = "format"
            elif "无法确定" in p.answer and len(p.answer) < 20:
                reason = "unanswerable"
            if reason:
                p.action = "reject"
                p.filter_trace[self.name] = {"action": "reject", "reason": reason}
                p.log("filter", self.name, action="reject", reason=reason)
                continue
            p.filter_trace[self.name] = {"action": "pass"}
            p.log("filter", self.name, action="pass")
            kept.append(p)
        return kept


@register("filter", "evidence_substring")
class EvidenceSubstring:
    """智训硬门：evidence 必须是 chunk 连续子串。"""

    name = "evidence_substring"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            ok = is_substring(p.evidence_span, p.chunk_text)
            p.filter_trace[self.name] = {"action": "pass" if ok else "reject", "ok": ok}
            p.log("filter", self.name, ok=ok)
            if not ok:
                p.action = "reject"
                continue
            kept.append(p)
        return kept


@register("filter", "llm_supported")
class LLMSupported:
    """对齐智训 quality_check：教师判断答案是否被证据支持。"""

    name = "llm_supported"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": '你是数据质检员。判断「回答」是否被「证据」支持。只输出 JSON：{"supported": true/false, "reason": "一句话理由"}',
                    },
                    {"role": "user", "content": f"问题：{p.question}\n回答：{p.answer}\n证据：{p.evidence_span}"},
                ],
                model=ctx.model_for("cheap"),
                max_tokens=200,
            ) or {}
            supported = bool(data.get("supported"))
            p.filter_trace[self.name] = {"supported": supported, "reason": data.get("reason")}
            p.log("filter", self.name, supported=supported)
            if not supported:
                p.action = "reject"
                continue
            kept.append(p)
        return kept


@register("filter", "exact_hash_dedup")
class ExactHashDedup:
    name = "exact_hash_dedup"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        seen = set()
        kept = []
        for p in pairs:
            key = sha1(p.question + "\n" + p.answer)
            p.filter_trace[self.name] = {"dup": key in seen}
            if key in seen:
                p.action = "reject"
                p.log("filter", self.name, action="reject", reason="exact_dup")
                continue
            seen.add(key)
            kept.append(p)
        return kept


def _minhash(tokens: list[str], n: int = 3, bits: int = 64) -> int:
    shingles = ngrams(tokens, n) or [tuple(tokens)]
    sig = (1 << bits) - 1
    for sh in shingles:
        h = int(hashlib.md5(("\0".join(sh)).encode("utf-8")).hexdigest()[:16], 16)
        sig = min(sig, h)
    return sig


@register("filter", "minhash_dedup")
class MinHashDedup:
    name = "minhash_dedup"

    def __init__(self, threshold: float = 0.8, ngram: int = 3, **_: object) -> None:
        self.threshold = float(threshold)
        self.ngram = int(ngram)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept: list[QAPair] = []
        sigs: list[tuple[set[tuple], int]] = []
        for p in pairs:
            toks = tokenize(p.question + " " + p.answer)
            sh = set(ngrams(toks, self.ngram))
            mh = _minhash(toks, self.ngram)
            dup = False
            for other, other_mh in sigs:
                if mh == other_mh:
                    dup = True
                    break
                if sh and other:
                    j = len(sh & other) / len(sh | other)
                    if j >= self.threshold:
                        dup = True
                        break
            p.filter_trace[self.name] = {"dup": dup}
            p.log("filter", self.name, dup=dup)
            if dup:
                p.action = "reject"
                continue
            sigs.append((sh, mh))
            kept.append(p)
        return kept


@register("filter", "semdedup")
class SemDeDup:
    name = "semdedup"

    def __init__(self, threshold: float = 0.88, **_: object) -> None:
        self.threshold = float(threshold)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        if len(pairs) <= 1:
            for p in pairs:
                p.filter_trace[self.name] = {"dup": False, "backend": "skip"}
            return pairs
        from ..embeddings import get_embedder

        embedder = get_embedder()
        if embedder.backend != "bge":
            ctx.note_fallback("semdedup:backend=tfidf")
        vecs = embedder.encode([p.question for p in pairs])
        keep_idx = []
        kept_vecs = []
        out = []
        for i, p in enumerate(pairs):
            dup = False
            if kept_vecs:
                sims = np.asarray(kept_vecs) @ vecs[i]
                if float(np.max(sims)) >= self.threshold:
                    dup = True
            p.filter_trace[self.name] = {"dup": dup, "backend": embedder.backend}
            p.log("filter", self.name, dup=dup, backend=embedder.backend)
            if dup:
                p.action = "reject"
                continue
            keep_idx.append(i)
            kept_vecs.append(vecs[i])
            out.append(p)
        return out


class _NLIBackend:
    def __init__(self) -> None:
        self.backend = "llm"
        self.pipe = None
        model_name = os.environ.get("QA_PIPELINE_NLI_MODEL")
        if model_name:
            try:
                from transformers import pipeline  # type: ignore

                self.pipe = pipeline("text-classification", model=model_name)
                self.backend = "deberta"
            except Exception as exc:
                logger.info("NLI fallback to LLM: %s", exc)
                self.backend = "llm"

    def score(self, premise: str, hypothesis: str, llm, model: str) -> float:
        if self.pipe is not None:
            try:
                # sliding window if long
                windows = [premise[i : i + 1500] for i in range(0, max(1, len(premise)), 1200)] or [premise]
                scores = []
                for w in windows[:4]:
                    pred = self.pipe({"text": w, "text_pair": hypothesis}, truncation=True)[0]
                    label = str(pred.get("label", "")).lower()
                    sc = float(pred.get("score") or 0)
                    if "entail" in label:
                        scores.append(sc)
                    elif "neutral" in label:
                        scores.append(0.5 * sc)
                    else:
                        scores.append(1 - sc)
                return float(max(scores) if scores else 0.0)
            except Exception:
                self.backend = "llm"
        data = llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "判断前提是否蕴含假设（NLI）。"
                        '输出 JSON：{"entailment":0.0到1.0,"label":"entailment|neutral|contradiction"}'
                    ),
                },
                {"role": "user", "content": f"前提：{premise[:2000]}\n假设：{hypothesis}"},
            ],
            model=model,
            max_tokens=120,
        ) or {}
        if "entailment" in data:
            return float(data["entailment"])
        return 0.85 if data.get("label") == "entailment" or data.get("supported") else 0.4


_NLI: _NLIBackend | None = None


def get_nli() -> _NLIBackend:
    global _NLI
    if _NLI is None:
        _NLI = _NLIBackend()
    return _NLI


@register("filter", "nli_fact")
class NLIFact:
    name = "nli_fact"

    def __init__(self, pass_threshold: float = 0.8, downgrade: float = 0.6, **kwargs: object) -> None:
        self.pass_threshold = float(kwargs.get("pass", pass_threshold))
        self.downgrade = float(kwargs.get("downgrade", downgrade))

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        nli = get_nli()
        if nli.backend != "deberta":
            ctx.note_fallback("nli_fact:backend=llm")
        kept = []
        for p in pairs:
            hyps = sentences(p.answer) or [p.answer]
            scores = [nli.score(p.chunk_text, h, ctx.llm, ctx.model_for("cheap")) for h in hyps[:6]]
            avg = float(sum(scores) / max(1, len(scores)))
            p.nli_score = avg
            if avg < self.downgrade:
                action = "reject"
            elif avg < self.pass_threshold:
                action = "downgrade"
            else:
                action = "pass"
            p.filter_trace[self.name] = {"score": avg, "action": action, "backend": nli.backend}
            p.log("filter", self.name, score=avg, action=action, backend=nli.backend)
            if action == "reject":
                p.action = "reject"
                continue
            if action == "downgrade":
                p.action = "downgrade"
            kept.append(p)
        return kept


@register("filter", "llm_judge")
class LLMJudge:
    name = "llm_judge"

    def __init__(
        self,
        weights: dict | None = None,
        min_overall: float = 2.5,
        **_: object,
    ) -> None:
        self.weights = weights or {
            "correctness": 0.35,
            "relevance": 0.20,
            "completeness": 0.20,
            "fluency": 0.15,
            "reasoning": 0.10,
        }
        self.min_overall = float(min_overall)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            data = ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是 QA 数据裁判，对相关性/正确性/完整性/语言质量/推理质量 1-5 打分。"
                            '输出 JSON：{"relevance":1,"correctness":1,"completeness":1,"fluency":1,"reasoning":1}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"文本块：{p.chunk_text[:2000]}\n问题：{p.question}\n答案：{p.answer}\n"
                            f"推理：{p.reasoning or '无'}"
                        ),
                    },
                ],
                model=ctx.model_for("default"),
            ) or {}
            scores = {
                k: float(data.get(k) or 0)
                for k in ("relevance", "correctness", "completeness", "fluency", "reasoning")
            }
            overall = sum(scores[k] * w for k, w in self.weights.items())
            p.judge_scores = scores
            p.judge_overall = overall
            p.filter_trace[self.name] = {"scores": scores, "overall": overall}
            p.log("filter", self.name, overall=overall)
            if overall < self.min_overall:
                p.action = "reject"
                continue
            kept.append(p)
        return kept


@register("filter", "knowledge_ablation")
class KnowledgeAblation:
    name = "knowledge_ablation"

    def __init__(self, f1_drop: float = 0.8, **_: object) -> None:
        self.f1_drop = float(f1_drop)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            blind = ctx.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "请只根据自身知识回答。不要编造具体内部规程编号。若无法确定请说明。无上下文。不要给出参考文本。",
                    },
                    {"role": "user", "content": p.question},
                ],
                model=p.teacher_model or ctx.model_for("default"),
                max_tokens=300,
            )
            f1 = token_f1(blind, p.answer)
            em = exact_match(blind, p.answer)
            gain = 1.0 - max(f1, em)
            p.kb_gain = gain
            no_gain = max(f1, em) >= self.f1_drop
            p.filter_trace[self.name] = {"f1": f1, "em": em, "kb_gain": gain, "no_gain": no_gain}
            p.log("filter", self.name, f1=f1, em=em, no_gain=no_gain)
            if no_gain:
                p.action = "downgrade"
            kept.append(p)
        return kept


@register("filter", "diversity_sample")
class DiversitySample:
    name = "diversity_sample"

    def __init__(self, quota: dict | None = None, per_doc_cap: int | None = None, **_: object) -> None:
        self.quota = quota or {"factual": 0.5, "explanatory": 0.3, "reasoning": 0.2}
        self.per_doc_cap = per_doc_cap

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        if not pairs:
            return pairs
        by_type: dict[str, list[QAPair]] = defaultdict(list)
        for p in pairs:
            by_type[p.q_type].append(p)
        target_n = len(pairs)
        chosen: list[QAPair] = []
        for q_type, ratio in self.quota.items():
            bucket = by_type.get(q_type, [])
            k = max(1, int(target_n * ratio)) if bucket else 0
            bucket.sort(key=lambda x: (x.judge_overall or x.nli_score or 0), reverse=True)
            chosen.extend(bucket[:k])
        leftover = [p for p in pairs if p not in chosen]
        chosen.extend(leftover)
        if self.per_doc_cap:
            counts: dict[str, int] = defaultdict(int)
            trimmed = []
            for p in chosen:
                key = p.source_doc or p.chunk_id
                if counts[key] >= self.per_doc_cap:
                    p.action = "reject"
                    p.filter_trace[self.name] = {"dropped": "doc_cap"}
                    continue
                counts[key] += 1
                trimmed.append(p)
            chosen = trimmed
        ids = {id(p) for p in chosen}
        for p in pairs:
            p.filter_trace[self.name] = {"kept": id(p) in ids}
        return [p for p in chosen if p.action != "reject"]
