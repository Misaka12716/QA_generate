"""数据质量指标（设计稿 §8.1）。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..pipeline import PipelineResult
from ..schemas import QAPair
from ..textutil import entropy, is_substring


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
    evidence_ok = sum(1 for p in kept if is_substring(p.evidence_span, p.chunk_text))
    answerable = questions - result.stats.rejected.get("evolution", 0)
    type_counts = Counter(p.q_type for p in kept)
    grades = Counter(p.grade or "none" for p in kept)
    retention = (len(kept) / distilled) if distilled else 0.0
    hallucination = 1.0 - (sum(nli_vals) / len(nli_vals) if nli_vals else 1.0)
    return {
        "recipe": result.recipe_name,
        "chunks": len(result.chunks),
        "questions": questions,
        "distilled": distilled,
        "kept": len(kept),
        "rejected": len(rejected),
        "answerability": round((len(kept) / questions) if questions else 0.0, 4),
        "nli_mean": round(sum(nli_vals) / len(nli_vals), 4) if nli_vals else None,
        "hallucination_proxy": round(max(0.0, hallucination), 4) if nli_vals else None,
        "knowledge_gain_rate": round(sum(1 for g in gain_vals if g >= 0.2) / len(gain_vals), 4) if gain_vals else None,
        "judge_mean": round(sum(judge_vals) / len(judge_vals), 4) if judge_vals else None,
        "type_entropy": round(entropy(dict(type_counts)), 4),
        "type_distribution": {k: round(v / max(1, len(kept)), 3) for k, v in type_counts.items()},
        "retention": round(retention, 4),
        "s_ratio": round(grades.get("S", 0) / max(1, len(kept)), 4),
        "grade_distribution": dict(grades),
        "evidence_grounded": round(evidence_ok / max(1, len(kept)), 4),
        "llm_calls": result.stats.llm_calls,
        "prompt_tokens": result.stats.prompt_tokens,
        "completion_tokens": result.stats.completion_tokens,
        "estimated_cost_usd": result.stats.estimated_cost_usd,
        "fallbacks": result.stats.fallbacks,
        "stage_seconds": result.stats.stage_seconds,
        "rejected_by_filter": result.stats.rejected,
    }


def quota_deviation(pairs: list[QAPair], target: dict[str, float] | None = None) -> float:
    target = target or {"factual": 0.5, "explanatory": 0.3, "reasoning": 0.2}
    dist = type_distribution(pairs)
    return float(sum(abs(dist.get(k, 0) - t) for k, t in target.items()) / 2)
