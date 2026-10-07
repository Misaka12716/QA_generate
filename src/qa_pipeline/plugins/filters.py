"""可串联过滤策略。缺失重模型时自动降级并写入 backend=fallback。"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict

import numpy as np

from ..llm import json_payload
from ..registry import register
from ..entity import subject_status
from ..qtypes import answerable, partial_quota_caps
from ..schemas import QAPair, _uid, content_id
from ..textutil import (
    balanced_take,
    compact,
    exact_match,
    is_substring,
    ngrams,
    rouge_l,
    sentences,
    sha1,
    token_f1,
    tokenize,
)

logger = logging.getLogger(__name__)


_JUDGE_ALIAS = {
    "accuracy": ("accuracy", "correctness"),
    "relevancy": ("relevancy", "relevance"),
    "completeness": ("completeness",),
    "quality": ("quality", "fluency"),
}


def _judge_dims(data: dict) -> dict[str, float]:
    scores = {}
    for name, aliases in _JUDGE_ALIAS.items():
        value = 0.0
        for alias in aliases:
            if data.get(alias) is not None:
                value = float(data.get(alias) or 0)
                break
        scores[name] = value
    return scores


def _answer_sim(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return max(token_f1(left, right), rouge_l(left, right))


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
    """智训硬门：evidence 必须是 chunk 连续子串。on_fail=quarantine 时留在流中，不发布。"""

    name = "evidence_substring"

    def __init__(self, on_fail: str = "reject", **_: object) -> None:
        self.on_fail = on_fail

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            quotes = [quote.quote for quote in p.evidence_quotes if quote.quote]
            haystacks = [p.chunk_text, *(p.metadata.get("evidence_chunk_texts") or [])]
            if len(quotes) > 1:
                ok = not p.evidence_span and all(any(text and is_substring(quote, text) for text in haystacks) for quote in quotes)
            else:
                ok = bool(p.evidence_span) and is_substring(p.evidence_span, p.chunk_text)
            if p.expected_action == "state_insufficient" and p.evidence_state == "missing" and not p.evidence_span:
                ok = True
            action = "pass" if ok else self.on_fail
            p.filter_trace[self.name] = {"action": action, "ok": ok}
            p.log("filter", self.name, ok=ok, action=action)
            if ok:
                kept.append(p)
                continue
            if self.on_fail == "quarantine":
                p.action = "quarantine"
                p.grade = "quarantine"
                kept.append(p)
                continue
            p.action = "reject"
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
            data = json_payload(ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": '你是数据质检员。判断「回答」是否被「证据」支持。只输出 JSON：{"supported": true/false, "reason": "一句话理由"}',
                    },
                    {"role": "user", "content": f"问题：{p.question}\n回答：{p.answer}\n证据：{p.evidence_span}"},
                ],
                model=ctx.model_for("cheap"),
                max_tokens=200,
            ))
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
        data = json_payload(llm.chat_json(
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
        ))
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
            "accuracy": 0.25,
            "relevancy": 0.25,
            "completeness": 0.25,
            "quality": 0.25,
        }
        self.min_overall = float(min_overall)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            data = json_payload(ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是 QA 数据裁判。按事实性、相关性、完整性、语言质量打 1-5 分。"
                            '输出 JSON：{"accuracy":1,"relevancy":1,"completeness":1,"quality":1}'
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
            ))
            scores = _judge_dims(data)
            overall = sum(scores[k] * self.weights.get(k, 0.0) for k in scores)
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


@register("filter", "roundtrip")
class RoundTrip:
    """k 次复现蒸馏答案，至少一次语义匹配或平均相似度达标。"""

    name = "roundtrip"

    def __init__(self, k: int = 3, pass_sim: float = 0.75, mean_sim: float = 0.6, **_: object) -> None:
        self.k = int(k)
        self.pass_sim = float(pass_sim)
        self.mean_sim = float(mean_sim)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            sims = []
            refused = 0
            for _ in range(self.k):
                text = ctx.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": "请只根据参考文本回答问题。若材料不足以回答，只回复：无法回答。",
                        },
                        {"role": "user", "content": f"参考文本：{p.chunk_text}\n问题：{p.question}"},
                    ],
                    model=ctx.model_for("cheap"),
                    max_tokens=300,
                    temperature=0.7,
                )
                if any(flag in text for flag in ("无法回答", "无法确定", "不足以回答")):
                    refused += 1
                sims.append(_answer_sim(text, p.answer))
            mean = sum(sims) / max(1, len(sims))
            ok = refused < self.k and (max(sims, default=0) >= self.pass_sim or mean >= self.mean_sim)
            p.filter_trace[self.name] = {"sims": [round(s, 4) for s in sims], "mean": round(mean, 4), "ok": ok}
            p.log("filter", self.name, mean=mean, ok=ok)
            if not ok:
                p.action = "reject"
                continue
            kept.append(p)
        return kept


@register("filter", "selfcheck")
class SelfCheck:
    """SelfCheckGPT：仅对高价值候选采样，一致性过低则丢弃。"""

    name = "selfcheck"

    def __init__(
        self,
        n: int = 5,
        min_consistency: float = 0.6,
        s_nli: float = 0.8,
        s_gain: float = 0.2,
        **_: object,
    ) -> None:
        self.n = int(n)
        self.min_consistency = float(min_consistency)
        self.s_nli = float(s_nli)
        self.s_gain = float(s_gain)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for p in pairs:
            gain = 1.0 if p.kb_gain is None else p.kb_gain
            candidate = (p.nli_score or 0) >= self.s_nli and gain >= self.s_gain and p.action == "pass"
            if not candidate:
                p.filter_trace[self.name] = {"skipped": True}
                kept.append(p)
                continue
            answers = []
            for _ in range(self.n):
                answers.append(
                    ctx.llm.chat(
                        [
                            {
                                "role": "system",
                                "content": "请只根据参考文本回答。回答要简短。",
                            },
                            {"role": "user", "content": f"参考文本：{p.chunk_text}\n问题：{p.question}"},
                        ],
                        model=ctx.model_for("cheap"),
                        max_tokens=240,
                        temperature=0.8,
                    )
                )
            sims = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sims.append(_answer_sim(answers[i], answers[j]))
            consistency = sum(sims) / max(1, len(sims))
            ok = consistency >= self.min_consistency
            p.filter_trace[self.name] = {"consistency": round(consistency, 4), "ok": ok, "n": self.n}
            p.log("filter", self.name, consistency=consistency, ok=ok)
            if not ok:
                p.action = "reject"
                continue
            kept.append(p)
        return kept


@register("filter", "diversity_sample")
class DiversitySample:
    name = "diversity_sample"

    def __init__(self, quota: dict | None = None, per_doc_cap: int | None = None, **_: object) -> None:
        self.quota = quota or {
            "factual": 0.4,
            "procedural": 0.2,
            "conditional": 0.15,
            "comparative": 0.15,
            "multihop": 0.1,
        }
        self.per_doc_cap = per_doc_cap

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        if not pairs:
            return pairs
        targets = dict(getattr(ctx, "extras", {}).get("quota_targets") or {})
        answer_rows = [pair for pair in pairs if answerable(pair)]
        by_type: dict[str, list[QAPair]] = defaultdict(list)
        for pair in answer_rows:
            by_type[pair.actual_q_type or pair.q_type].append(pair)
        if not targets:
            targets = partial_quota_caps(len(answer_rows), self.quota)
        chosen: list[QAPair] = []
        for q_type, limit in targets.items():
            bucket = by_type.get(q_type, [])
            bucket.sort(key=lambda x: ((x.judge_overall or x.nli_score or 0), x.qa_id), reverse=True)
            chosen.extend(bucket[: int(limit)])
        if self.per_doc_cap:
            counts: dict[str, int] = defaultdict(int)
            trimmed = []
            for pair in chosen:
                key = pair.source_family_id or pair.source_doc or pair.chunk_id
                if counts[key] >= self.per_doc_cap:
                    pair.selection_status = "not_selected"
                    pair.included_in_this_run = False
                    pair.exclude_reason = pair.exclude_reason or "doc_cap"
                    pair.filter_trace[self.name] = {"kept": False, "reason": "doc_cap"}
                    continue
                counts[key] += 1
                trimmed.append(pair)
            chosen = trimmed
        ids = {id(pair) for pair in chosen}
        for pair in pairs:
            if not answerable(pair):
                pair.filter_trace[self.name] = {"kept": pair.action == "pass", "quota": "excluded_behavior"}
                continue
            selected = id(pair) in ids and pair.action != "reject"
            pair.filter_trace[self.name] = {"kept": selected}
            if selected:
                pair.included_in_this_run = True
                pair.selection_status = "selected"
            else:
                pair.included_in_this_run = False
                pair.selection_status = "not_selected"
                if pair.action != "reject":
                    pair.exclude_reason = pair.exclude_reason or "quota"
        return pairs


_OK_CLAIM = {"supported"}


def _critical_conflict(answer: str, points: list[str]) -> str:
    import re

    for point in points:
        gold_nums = set(re.findall(r"\d+(?:\.\d+)?", point or ""))
        pred_nums = set(re.findall(r"\d+(?:\.\d+)?", answer or ""))
        if gold_nums and pred_nums and not gold_nums.issubset(pred_nums):
            return "numeric"
        if any(token in (point or "") for token in ("不得", "禁用", "尚未", "未进行")) and not any(
            token in (answer or "") for token in ("不", "未", "无", "非", "禁用")
        ):
            return "negation"
    return ""


def _high_risk(pair: QAPair) -> str:
    if pair.evidence_state == "conflict":
        return "conflict"
    if any(ch.isdigit() for ch in pair.answer or ""):
        return "numeric"
    if any(token in (pair.answer or "") for token in ("禁用", "慎用", "孕妇", "儿童", "肾功能", "不得")):
        return "population_or_negation"
    return ""


def _claims_cover(pair: QAPair) -> tuple[bool, str]:
    if not pair.claims:
        return False, "no_claims"
    targets = _claim_texts(pair)
    for text in targets:
        matched = [
            item
            for item in pair.claims
            if item.get("text")
            and (
                is_substring(str(item["text"]), text)
                or is_substring(text, str(item["text"]))
                or token_f1(str(item["text"]), text) >= 0.45
            )
        ]
        if not matched:
            return False, "claim_gap"
        if any(item.get("status") == "not_applicable" for item in matched):
            return False, "not_applicable_substantive"
        if any(item.get("status") not in _OK_CLAIM for item in matched):
            return False, str(matched[0].get("status") or "unsupported")
    points = [point for point in pair.answer_points if point]
    if points:
        hit = sum(1 for point in points if point in pair.answer or token_f1(pair.answer, point) >= 0.35)
        if hit < len(points):
            return False, "missing_point"
    critical = _critical_conflict(pair.answer, points)
    if critical:
        return False, critical
    return True, "pass"


def _claim_texts(pair: QAPair) -> list[str]:
    points = [item.strip() for item in pair.answer_points if item and item.strip()]
    if points:
        return points
    parts = [item.strip() for item in sentences(pair.answer) if item.strip()]
    return parts or ([pair.answer.strip()] if pair.answer.strip() else [])


def _apply_claim_payload(pair: QAPair, data: dict) -> None:
    raw = data.get("claims") or []
    claims = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                status = str(item.get("status") or "insufficient")
                claims.append({"text": str(item.get("text") or ""), "status": status})
            elif isinstance(item, str):
                claims.append({"text": item, "status": "insufficient"})
    if not claims:
        claims = [{"text": text, "status": "insufficient"} for text in _claim_texts(pair)]
    pair.claims = claims


@register("filter", "claim_evidence")
class ClaimEvidence:
    """一次独立验证：按主张与证据的支持关系判定，相似度不参与通过条件。"""

    name = "claim_evidence"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        kept = []
        for pair in pairs:
            if pair.action == "quarantine":
                pair.filter_trace[self.name] = {"skipped": "already_quarantine"}
                kept.append(pair)
                continue
            visible = pair.metadata.get("student_context")
            if visible is None:
                visible = "\n".join(pair.student_context_refs) or pair.chunk_text
            visible = str(visible)
            if pair.goal != "closed_book_domain" and pair.evidence_state == "sufficient":
                if not pair.evidence_span or not is_substring(pair.evidence_span, visible):
                    pair.action = "quarantine"
                    pair.grade = "quarantine"
                    pair.filter_trace[self.name] = {"action": "quarantine", "reason": "hidden_or_missing_support"}
                    kept.append(pair)
                    continue
            elif pair.evidence_state == "sufficient" and not is_substring(pair.evidence_span, pair.chunk_text):
                pair.action = "quarantine"
                pair.grade = "quarantine"
                pair.filter_trace[self.name] = {"action": "quarantine", "reason": "evidence_not_located"}
                kept.append(pair)
                continue
            data = json_payload(ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是独立验证器。把答案拆成主张，逐条判断证据的支持关系。"
                            "status 只能是 supported、contradicted、insufficient、not_applicable。"
                            '只输出 JSON：{"claims":[{"text":"...","status":"supported"}]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"问题：{pair.question}\n最终可见资料：{visible}\n答案：{pair.answer}\n"
                            f"必答要点：{'；'.join(_claim_texts(pair))}\n来源位置：{pair.evidence_span}"
                        ),
                    },
                ],
                model=ctx.model_for("cheap"),
                max_tokens=400,
            ))
            _apply_claim_payload(pair, data)
            covered, reason = _claims_cover(pair)
            risk = _high_risk(pair)
            if not covered or risk:
                pair.action = "needs_escalation"
                pair.grade = None
                pair.metadata["escalation_reason"] = risk or reason
                action = "needs_escalation"
            else:
                action = "pass"
                pair.action = "pass"
            if pair.generation_route == "k_joint":
                from .question_gen import _unit_status

                class _ChunkView:
                    def __init__(self, text: str) -> None:
                        self.text = text

                units = pair.metadata.get("knowledge_units") or []
                view = _ChunkView(pair.chunk_text)
                statuses = [_unit_status(unit, view) for unit in units if isinstance(unit, dict)]
                if units and action == "pass" and statuses and all(item == "semantically_verified" for item in statuses):
                    pair.metadata["verification_status"] = "semantically_verified"
                    pair.verification_status = "semantically_verified"
                else:
                    pair.metadata["verification_status"] = pair.metadata.get("verification_status") or "pending"
            pair.filter_trace[self.name] = {"action": action, "claims": pair.claims, "reason": reason}
            pair.log("filter", self.name, action=action)
            kept.append(pair)
        return kept


@register("filter", "risk_escalate")
class RiskEscalate:
    """只对矛盾、数值或验证分歧样本再验一次。"""

    name = "risk_escalate"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        limit = int(getattr(ctx.recipe, "escalation_max", 1) or 0)
        cheap = ctx.model_for("cheap")
        strong = ctx.model_for("strong")
        for pair in pairs:
            if pair.action != "needs_escalation" or limit <= 0:
                pair.filter_trace[self.name] = {"escalated": False}
                continue
            if int(pair.metadata.get("escalation_count") or 0) >= 1:
                pair.action = "quarantine"
                pair.grade = "quarantine"
                pair.filter_trace[self.name] = {"escalated": False, "reason": "escalation_cap"}
                continue
            data = json_payload(ctx.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "升级验证。隐藏生成器自评，只根据证据复核主张。"
                            '只输出 JSON：{"claims":[{"text":"...","status":"supported"}]}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"问题：{pair.question}\n答案：{pair.answer}\n证据：{pair.evidence_span}",
                    },
                ],
                model=ctx.model_for("strong"),
                max_tokens=400,
            ))
            _apply_claim_payload(pair, data)
            pair.metadata["escalation_count"] = int(pair.metadata.get("escalation_count") or 0) + 1
            pair.metadata["escalation_kind"] = "second_pass" if cheap == strong else "strong_model"
            covered, reason = _claims_cover(pair)
            if not covered:
                pair.action = "quarantine"
                pair.grade = "quarantine"
            else:
                pair.action = "pass"
                pair.grade = None
            pair.filter_trace[self.name] = {
                "escalated": True,
                "action": pair.action,
                "reason": pair.metadata.get("escalation_reason"),
                "kind": pair.metadata.get("escalation_kind"),
                "recheck": reason,
            }
            pair.log("filter", self.name, escalated=True, action=pair.action)
        return pairs


@register("filter", "behavior_insufficient")
class BehaviorInsufficient:
    """从已通过样本复制一条学生不可见黄金证据的说明不足样本。"""

    name = "behavior_insufficient"

    def __init__(self, max_new: int = 1, **_: object) -> None:
        self.max_new = int(max_new)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        parents = [
            pair
            for pair in pairs
            if pair.action == "pass" and pair.evidence_state == "sufficient" and pair.evidence_span
        ]
        created: list[QAPair] = []
        kinds = ("empty", "unrelated", "wrong_entity", "missing_condition", "clarify")
        for index, parent in enumerate(parents[: max(0, self.max_new)]):
            family = parent.family_id or parent.qa_id
            parent.family_id = family
            kind = kinds[index % len(kinds)]
            visible, state, action, answer = _behavior_view(parent, kind)
            if parent.evidence_span and parent.evidence_span in visible:
                state, action = "sufficient", "answer"
                answer = parent.answer
            child = QAPair(
                question=parent.question,
                answer=answer,
                chunk_id=parent.chunk_id,
                chunk_text=parent.chunk_text,
                source_doc=parent.source_doc,
                q_type=parent.q_type,
                goal=parent.goal,
                intent_primary=parent.intent_primary,
                generation_route=parent.generation_route,
                family_id=family,
                parent_sample_id=parent.qa_id,
                evidence_state=state,
                expected_action=action,
                selection_role="behavior",
                evidence_span="",
                answer_points=[],
                visible_support_refs=[],
                student_context_refs=[],
                construction_evidence_refs=[],
                action="pass",
                teacher_model=parent.teacher_model,
                metadata={
                    "student_context": visible,
                    "behavior_condition": kind,
                    "audit_gold_evidence": parent.evidence_span,
                    "source_doc": parent.source_doc,
                },
            )
            child.claims = [{"text": answer, "status": "supported"}]
            child.filter_trace[self.name] = {"parent": parent.qa_id, "condition": kind, "inherited_pass": False}
            child.log("filter", self.name, parent=parent.qa_id, condition=kind)
            created.append(child)
        return pairs + created


@register("filter", "student_diagnostic")
class StudentDiagnostic:
    """抽样记录有上下文与无上下文的内容判断，不因无上下文答对而删除。"""

    name = "student_diagnostic"

    def __init__(self, sample_size: int = 2, **_: object) -> None:
        self.sample_size = int(sample_size)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        probe = ctx.extras.get("student_probe") or (getattr(ctx.recipe, "extra", None) or {}).get("student_probe")
        if probe is None or probe is ctx.llm or not hasattr(probe, "chat"):
            for pair in pairs:
                pair.filter_trace[self.name] = {
                    "sampled": False,
                    "status": "not_executed",
                    "reason": "missing_student_probe",
                }
            ctx.note_fallback("student_diagnostic:not_executed")
            return pairs
        eligible = [pair for pair in pairs if pair.action == "pass"]
        targets = balanced_take(
            eligible,
            self.sample_size,
            lambda pair: f"{pair.q_type}:{'risk' if any(ch.isdigit() for ch in pair.answer) else 'plain'}",
            seed=getattr(ctx.recipe, "seed", 0),
        )
        chosen = {id(pair) for pair in targets}
        model_name = getattr(probe, "default_model", "") or "student"
        for pair in pairs:
            if id(pair) not in chosen:
                pair.filter_trace[self.name] = {"sampled": False, "status": "not_sampled"}
                continue
            messages = [
                {"role": "system", "content": "诊断：给定参考文本，用简短句子回答问题。"},
                {"role": "user", "content": f"参考文本：{pair.chunk_text}\n问题：{pair.question}"},
            ]
            with_text = probe.chat(messages, model=model_name, max_tokens=120)
            closed_text = probe.chat(
                [
                    {"role": "system", "content": "可以运用已有知识回答。不确定就说明不确定。"},
                    {"role": "user", "content": pair.question},
                ],
                model=model_name,
                max_tokens=120,
            )
            pair.student_context_score = round(token_f1(with_text, pair.answer), 4)
            pair.student_closed_score = round(token_f1(closed_text, pair.answer), 4)
            pair.metadata["probe_count"] = 1
            pair.metadata["diagnostic_status"] = "unknown"
            pair.metadata["student_probe_model"] = model_name
            if pair.student_context_score >= 0.8 and pair.student_closed_score >= 0.8:
                second = probe.chat(
                    [
                        {"role": "system", "content": "诊断：给定参考文本，换一种说法回答。"},
                        {"role": "user", "content": f"参考文本：{pair.chunk_text}\n问题：{pair.question}"},
                    ],
                    model=model_name,
                    max_tokens=120,
                )
                pair.metadata["probe_count"] = 2
                if token_f1(second, pair.answer) >= 0.8:
                    pair.metadata["diagnostic_status"] = "stable_known"
            pair.filter_trace[self.name] = {
                "sampled": True,
                "status": pair.metadata["diagnostic_status"],
                "student_context_score": pair.student_context_score,
                "student_closed_score": pair.student_closed_score,
                "model": model_name,
                "probe_count": pair.metadata["probe_count"],
            }
            pair.log("filter", self.name, sampled=True, model=model_name)
        return pairs


def _behavior_view(parent: QAPair, kind: str) -> tuple[str, str, str, str]:
    if kind == "empty":
        return "", "missing", "state_insufficient", "当前资料不足，无法根据给定材料回答。"
    if kind == "unrelated":
        return "这是一段与问题药品无关的仓储温湿度说明。", "missing", "state_insufficient", "当前资料不足，无法根据给定材料回答。"
    if kind == "wrong_entity":
        return "另一药品的用法与本题对象不同。", "missing", "state_insufficient", "当前资料不足，无法根据给定材料回答。"
    if kind == "missing_condition":
        return "资料只保留了一般说明，没有本题的关键条件。", "partial", "partial_answer", "资料只支持一般说明，关键条件缺失。"
    return parent.chunk_text, "ambiguous", "clarify", "请补充具体药品、规格或适用人群后再回答。"


@register("filter", "stable_known_downsample")
class StableKnownDownsample:
    """只在诊断分数都高时降低训练权重。未抽样或分数缺失的样本不删除。"""

    name = "stable_known_downsample"

    def __init__(self, min_score: float = 0.8, **_: object) -> None:
        self.min_score = float(min_score)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        for pair in pairs:
            ctx_score = pair.student_context_score
            closed = pair.student_closed_score
            if ctx_score is None or closed is None:
                pair.filter_trace[self.name] = {"applied": False, "reason": "no_diagnostic"}
                continue
            stable = (
                pair.metadata.get("diagnostic_status") == "stable_known"
                and int(pair.metadata.get("probe_count") or 0) >= 2
                and ctx_score >= self.min_score
                and closed >= self.min_score
            )
            if not stable:
                pair.filter_trace[self.name] = {"applied": False, "reason": "single_observation_not_stable"}
                continue
            pair.selection_weight = 0.25
            pair.loss_weight = 0.25
            pair.included_in_this_run = True
            pair.metadata["downweighted"] = True
            pair.filter_trace[self.name] = {"applied": True, "selection_weight": pair.selection_weight}
        return pairs


@register("filter", "replay_replace")
class ReplayReplace:
    """固定监督条数：用保留角色样本替换同等数量的新样本，不增加训练条数。"""

    name = "replay_replace"

    def __init__(self, fraction: float = 0.2, **_: object) -> None:
        self.fraction = float(fraction)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        extra = getattr(ctx.recipe, "extra", None) or {}
        pool = ctx.extras.get("retention_pairs")
        if pool is None and extra.get("retention_pool"):
            from ..store import load_pairs

            pool = load_pairs(extra["retention_pool"])
        if not pool:
            for pair in pairs:
                pair.filter_trace[self.name] = {"status": "not_executed", "reason": "missing_retention_pool"}
            ctx.note_fallback("replay_replace:not_executed")
            return pairs
        added = []
        for item in pool:
            kept = item.model_copy(deep=True)
            kept.selection_role = "retention"
            kept.included_in_this_run = True
            kept.metadata["retention_reason"] = kept.metadata.get("retention_reason") or "declared_retention_target"
            kept.filter_trace[self.name] = {"status": "retention", "reason": kept.metadata["retention_reason"]}
            added.append(kept)
        eligible = [pair for pair in pairs if pair.action == "pass" and pair.selection_role != "retention"]
        count = min(len(eligible), len(added))
        reason = "random_delete_control" if extra.get("retention_control") == "random_delete" else "replay_replace"
        for index, pair in enumerate(eligible[:count]):
            pair.included_in_this_run = False
            pair.exclude_reason = reason
            pair.filter_trace[self.name] = {"status": reason, "replaced_by": added[index].qa_id}
        return pairs + added


@register("filter", "conflict_behavior")
class ConflictBehavior:
    """默认关闭。没有已确认的合法冲突时不构造样本。"""

    name = "conflict_behavior"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        enabled = (getattr(ctx.recipe, "extra", None) or {}).get("conflict_behavior") == "enabled"
        if not enabled:
            return pairs
        created = []
        for pair in pairs:
            conflict = pair.metadata.get("confirmed_conflict")
            if not conflict or pair.action != "pass":
                continue
            child = pair.model_copy(deep=True)
            child.qa_id = _uid("qa_")
            child.evidence_state = "conflict"
            child.expected_action = "state_conflict"
            child.answer = "两份资料的说法不一致，当前无法确定哪一个是有效结论。"
            child.selection_role = "behavior"
            child.metadata["confirmed_conflict"] = conflict
            created.append(child)
        return pairs + created


@register("filter", "joint_dedup")
class JointDedup:
    """按目标、问题、要点、证据和证据状态去重。证据状态不同的对照样本保留。"""

    name = "joint_dedup"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        seen = set()
        kept = []
        for pair in pairs:
            visible = str(pair.metadata.get("student_context") if "student_context" in pair.metadata else pair.chunk_text)
            points = "\n".join(pair.answer_points)
            key = (
                pair.goal,
                pair.source_family_id,
                pair.metadata.get("version") or "",
                compact(pair.question),
                points,
                compact(pair.answer),
                compact(visible)[:240],
                pair.evidence_state,
                pair.expected_action,
            )
            duplicate = key in seen
            pair.filter_trace[self.name] = {"duplicate": duplicate}
            if duplicate:
                pair.action = "reject"
                pair.log("filter", self.name, action="reject")
                continue
            seen.add(key)
            kept.append(pair)
        return kept


@register("filter", "natural_distribution")
class NaturalDistribution:
    """对照臂：保留合格池的自然题型分布，不按配额删样本。"""

    name = "natural_distribution"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        counts: dict[str, int] = {}
        for pair in pairs:
            if pair.action == "reject":
                continue
            pair.included_in_this_run = True
            counts[pair.q_type] = counts.get(pair.q_type, 0) + 1
            pair.filter_trace[self.name] = {"kept": True}
        ctx.extras["natural_distribution"] = counts
        return [pair for pair in pairs if pair.action != "reject"]


@register("filter", "gap_fill")
class GapFill:
    """按目录缺口记录可补类型。没有规划结果时不假装已经回填。"""

    name = "gap_fill"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        extra = getattr(ctx.recipe, "extra", None) or {}
        catalog = ctx.extras.get("catalog")
        if catalog is None and extra.get("catalog_path"):
            from ..store import read_jsonl

            path = extra["catalog_path"]
            catalog = read_jsonl(path) if __import__("pathlib").Path(path).is_file() else None
        if not catalog:
            ctx.extras["gap_report"] = {"status": "not_executed", "reason": "missing_catalog"}
            ctx.note_fallback("gap_fill:not_executed")
            return pairs
        present: dict[str, int] = {}
        for pair in pairs:
            if pair.action == "pass":
                present[pair.q_type] = present.get(pair.q_type, 0) + 1
        wanted: dict[str, int] = {}
        for unit in catalog:
            kind = str(unit.get("kind") or "fact")
            wanted[kind] = wanted.get(kind, 0) + 1
        gaps = {kind: count for kind, count in wanted.items() if present.get(kind, 0) == 0}
        report = {
            "status": "planned",
            "target": wanted,
            "selected": present,
            "requested": gaps,
            "fills": [],
        }
        filler = ctx.extras.get("gap_filler")
        added: list[QAPair] = []
        if filler:
            added = list(filler(gaps, pairs) or [])
            report["fills"] = [pair.qa_id for pair in added]
            report["status"] = "filled"
        ctx.extras["gap_report"] = report
        for pair in added:
            pair.filter_trace[self.name] = {"filled": True}
        return pairs + added


@register("filter", "entity_subject")
class EntitySubject:
    """题干对象与标准对象冲突时隔离。证据原句不能抵消对象错误。"""

    name = "entity_subject"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        for pair in pairs:
            status = subject_status(pair.question, pair.document_identity, pair.chunk_text)
            if pair.expected_action == "clarify" and status in {"unresolved_anaphora", "unresolved_subject"}:
                status = ""
            pair.filter_trace[self.name] = {"status": status or "ok"}
            pair.log("filter", self.name, status=status or "ok")
            if not status:
                continue
            pair.action = "quarantine"
            pair.grade = "quarantine"
            pair.exclude_reason = pair.exclude_reason or status
            pair.verification_status = pair.verification_status or "failed"
        return pairs
