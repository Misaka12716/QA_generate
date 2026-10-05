"""数据质量指标（设计稿 §8.1）。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..pipeline import PipelineResult
from ..schemas import QAPair
from ..textutil import entropy, is_substring
from .scoring import ratio


def grade_share(pairs: list[QAPair]) -> dict[str, float]:
    n = max(1, len(pairs))
    c = Counter(p.grade or "none" for p in pairs)
    return {k: v / n for k, v in c.items()}


def type_distribution(pairs: list[QAPair]) -> dict[str, float]:
    n = max(1, len(pairs))
    c = Counter(p.q_type for p in pairs)
    return {k: v / n for k, v in c.items()}


def compute_metrics(result: PipelineResult) -> dict[str, Any]:
    kept = result.pairs
    rejected = result.rejected
    distilled = result.stats.produced.get("distilled") or (len(kept) + len(rejected))
    questions = result.stats.produced.get("questions") or len(result.questions)
    nli_vals = [p.nli_score for p in kept if p.nli_score is not None]
    judge_vals = [p.judge_overall for p in kept if p.judge_overall is not None]
    gain_vals = [p.kb_gain for p in kept if p.kb_gain is not None]
    ground_num = ground_den = 0
    for pair in kept:
        if pair.evidence_state != "sufficient":
            continue
        visible = pair.metadata.get("student_context") if "student_context" in pair.metadata else pair.chunk_text
        ground_den += 1
        if pair.evidence_span and visible and is_substring(pair.evidence_span, visible):
            ground_num += 1
    grounded = ratio(ground_num, ground_den)
    labeled = [pair for pair in list(kept) + list(rejected) if "answerable" in pair.metadata]
    if labeled:
        answerability = ratio(sum(1 for pair in labeled if pair.metadata.get("answerable")), len(labeled))["value"]
        answerability_reason = None
    else:
        answerability = None
        answerability_reason = "missing_answerable_label"
    type_counts = Counter(p.q_type for p in kept)
    grades = Counter(p.grade or "none" for p in kept)
    retention_ratio = ratio(len(kept), distilled)
    hallucination = 1.0 - (sum(nli_vals) / len(nli_vals) if nli_vals else 1.0)
    stages = dict(result.stats.stage_counts or {})
    return {
        "recipe": result.recipe_name,
        "chunks": len(result.chunks),
        "questions": questions,
        "distilled": distilled,
        "kept": len(kept),
        "rejected": len(rejected),
        "answerability": answerability,
        "answerability_reason": answerability_reason,
        "nli_mean": round(sum(nli_vals) / len(nli_vals), 4) if nli_vals else None,
        "hallucination_proxy": round(max(0.0, hallucination), 4) if nli_vals else None,
        "knowledge_gain_rate": round(sum(1 for g in gain_vals if g >= 0.2) / len(gain_vals), 4) if gain_vals else None,
        "judge_mean": round(sum(judge_vals) / len(judge_vals), 4) if judge_vals else None,
        "type_entropy": round(entropy(dict(type_counts)), 4),
        "type_distribution": {k: round(v / max(1, len(kept)), 3) for k, v in type_counts.items()},
        "retention": retention_ratio["value"],
        "retention_reason": retention_ratio["reason"],
        "s_ratio": round(grades.get("S", 0) / max(1, len(kept)), 4),
        "grade_distribution": dict(grades),
        "evidence_grounded": grounded["value"],
        "evidence_grounded_detail": grounded,
        "funnel_stages": stages,
        "llm_calls": result.stats.llm_calls,
        "prompt_tokens": result.stats.prompt_tokens,
        "completion_tokens": result.stats.completion_tokens,
        "estimated_cost_usd": result.stats.estimated_cost_usd,
        "fallbacks": result.stats.fallbacks,
        "stage_seconds": result.stats.stage_seconds,
        "rejected_by_filter": result.stats.rejected,
        "funnel": result.stats.funnel,
        "tokens_per_10k_kept": _per_10k(result),
    }


def _per_10k(result: PipelineResult) -> int | None:
    kept = len(result.pairs)
    if not kept:
        return None
    tokens = result.stats.prompt_tokens + result.stats.completion_tokens
    return int(round(tokens / kept * 10000))


def quota_deviation(pairs: list[QAPair], target: dict[str, float] | None = None) -> float:
    target = target or {
        "factual": 0.4,
        "procedural": 0.2,
        "conditional": 0.15,
        "comparative": 0.15,
        "multihop": 0.1,
    }
    dist = type_distribution(pairs)
    return float(sum(abs(dist.get(k, 0) - t) for k, t in target.items()) / 2)
