"""要点、行为与配对评分。词面 F1 只作为锁定分词下的辅助诊断。"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any

from ..schemas import EXPECTED_ACTIONS
from ..textutil import SCORING_TOKENIZER_ID, exact_match, token_f1

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CONCRETE = re.compile(r"\d|(?:为|是|包括|应|禁用|剂量|规格)")


class AlignmentError(Exception):
    """预测、金标或干扰上下文无法按 ID 对齐。"""


def classify_action(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "answer"
    echo = "资料不足时应说明" in raw or raw.startswith("仅依据提供的资料")
    asks = any(token in raw for token in ("请补充", "哪一种", "哪种规格", "请说明具体"))
    gap = any(token in raw for token in ("资料不足", "无法确定", "缺少", "不能确定", "未提供"))
    conflict = any(token in raw for token in ("不一致", "互相矛盾", "存在冲突"))
    concrete = bool(_CONCRETE.search(raw)) and len(raw) > 8
    if conflict and not concrete:
        return "state_conflict"
    if asks and not (concrete and not gap):
        return "clarify"
    if echo and concrete:
        return "answer"
    if gap and concrete:
        return "answer"
    if gap:
        return "state_insufficient"
    if asks:
        return "clarify"
    return "answer"


def critical_errors(prediction: str, points: list[str]) -> list[str]:
    errors = []
    pred = prediction or ""
    for point in points:
        gold_nums = set(_NUMBER.findall(point or ""))
        pred_nums = set(_NUMBER.findall(pred))
        if gold_nums and pred_nums and not gold_nums.issubset(pred_nums):
            errors.append("numeric")
        units = re.findall(r"(mg|ml|mmol|μg|ug|kg|℃|%)", point or "", flags=re.I)
        if units and not any(unit.lower() in pred.lower() for unit in units):
            errors.append("unit")
        if any(token in (point or "") for token in ("不得", "禁用", "尚未", "未进行", "不是")) and not any(
            token in pred for token in ("不", "未", "无", "非", "禁用")
        ):
            errors.append("negation")
    return list(dict.fromkeys(errors))


def point_coverage(prediction: str, points: list[str], weights: list[float] | None = None) -> dict[str, Any]:
    usable = [point for point in points if point]
    if not usable:
        return {"value": None, "reason": "no_points", "hits": []}
    weights = weights or [1.0] * len(usable)
    hits = []
    gained = 0.0
    total = 0.0
    for point, weight in zip(usable, weights):
        total += weight
        ok = point in prediction or token_f1(prediction, point) >= 0.55
        hits.append(ok)
        if ok:
            gained += weight
    return {"value": round(gained / total, 4) if total else None, "reason": None, "hits": hits}


def score_task(prediction: str, case: dict[str, Any]) -> dict[str, Any]:
    points = list(case.get("answer_points") or [])
    if not points and case.get("answer"):
        points = [str(case["answer"])]
    coverage = point_coverage(prediction, points)
    errors = critical_errors(prediction, points)
    expected = str(case.get("expected_action") or "answer")
    actual = classify_action(prediction)
    action_ok = actual == expected or (expected == "partial_answer" and actual == "answer" and coverage["value"] not in {None, 0})
    if expected == "state_insufficient" and actual == "answer":
        action_ok = False
    relevant = coverage["value"] not in {None, 0} if expected in {"answer", "partial_answer"} else True
    if expected == "answer" and points:
        relevant = bool(coverage["value"])
    passed = bool(relevant and not errors and action_ok and (expected != "answer" or coverage["value"] == 1))
    return {
        "passed": passed,
        "point_coverage": coverage["value"],
        "critical_errors": errors,
        "expected_action": expected,
        "actual_action": actual,
        "action_ok": action_ok,
        "relevant": relevant,
        "tokenizer_id": SCORING_TOKENIZER_ID,
        "token_f1": token_f1(prediction, "\n".join(points)) if points else None,
        "em": exact_match(prediction, str(case.get("answer") or "")) if _single_value(case) else None,
    }


def _single_value(case: dict[str, Any]) -> bool:
    points = case.get("answer_points") or []
    return len(points) <= 1 and bool(_NUMBER.findall(str(case.get("answer") or "")))


def transition_label(before: dict[str, Any], after: dict[str, Any]) -> str:
    b_pass, a_pass = bool(before.get("passed")), bool(after.get("passed"))
    b_cov = before.get("point_coverage") or 0
    a_cov = after.get("point_coverage") or 0
    if (not b_pass) and a_pass:
        return "前错后对"
    if b_pass and not a_pass:
        return "前对后错"
    if b_pass and a_pass:
        return "均对"
    if (not b_pass) and (not a_pass) and a_cov > b_cov:
        return "部分改善仍不合格"
    if (not b_pass) and (not a_pass):
        return "均错"
    return "待裁定"


def align_by_id(rows: list[dict[str, Any]], predictions: dict[str, str], id_key: str = "case_id") -> list[tuple[dict, str]]:
    missing = [str(row.get(id_key) or row.get("id")) for row in rows if str(row.get(id_key) or row.get("id")) not in predictions]
    if missing:
        raise AlignmentError("缺失预测: " + ",".join(missing))
    extra_context = [
        str(row.get(id_key) or row.get("id"))
        for row in rows
        if row.get("require_distractor") and not row.get("distractor")
    ]
    if extra_context:
        raise AlignmentError("缺失干扰上下文: " + ",".join(extra_context))
    return [(row, predictions[str(row.get(id_key) or row.get("id"))]) for row in rows]


def confusion(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    if any(not row.get("expected_action") for row in rows):
        raise AlignmentError("缺少 expected_action，拒绝计算")
    if len(rows) != len(predictions):
        raise AlignmentError(f"预测 {len(predictions)} 与样本 {len(rows)} 数量不一致")
    matrix = {label: {pred: 0 for pred in EXPECTED_ACTIONS} for label in EXPECTED_ACTIONS}
    for row, pred in zip(rows, predictions):
        expected = row["expected_action"]
        actual = classify_action(pred)
        matrix[expected][actual] += 1
    return {"matrix": matrix, "n": len(rows)}


def binary_refusal_f1(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """该拒且拒为 TP，该答却拒为 FP，该拒却答为 FN。"""
    if any("expected_action" not in row for row in rows):
        raise AlignmentError("缺少 expected_action，拒绝计算")
    tp = fp = fn = 0
    for row, pred in zip(rows, predictions):
        should = row["expected_action"] in {"state_insufficient", "state_conflict"}
        did = classify_action(pred) in {"state_insufficient", "state_conflict"}
        if should and did:
            tp += 1
        elif (not should) and did:
            fp += 1
        elif should and not did:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "reason": None if f1 is not None else "zero_denominator",
    }


def aggregate_families(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        family = str(case.get("family_id") or case.get("case_id") or case.get("id"))
        source = str(case.get("source_family_id") or family)
        by_family[family].append(1.0 if case.get("passed") else 0.0)
        if family not in by_source[source]:
            by_source[source].append(family)
    family_mean = {key: sum(values) / len(values) for key, values in by_family.items()}
    source_mean = []
    for source, families in by_source.items():
        vals = [family_mean[family] for family in families]
        source_mean.append(sum(vals) / len(vals))
    overall = sum(source_mean) / len(source_mean) if source_mean else None
    return {
        "family_count": len(by_family),
        "source_family_count": len(by_source),
        "task_accuracy": None if overall is None else round(overall, 4),
        "reason": None if source_mean else "zero_denominator",
        "weighting": "equal_source_family",
    }


def clustered_interval(values_by_source: dict[str, float], n_boot: int = 200, seed: int = 0) -> dict[str, Any]:
    keys = list(values_by_source)
    if not keys:
        return {"low": None, "high": None, "mean": None, "reason": "zero_denominator"}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [values_by_source[rng.choice(keys)] for _ in keys]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    mean = sum(values_by_source.values()) / len(values_by_source)
    low = stats[int(0.025 * (n_boot - 1))]
    high = stats[int(0.975 * (n_boot - 1))]
    return {"low": round(low, 4), "high": round(high, 4), "mean": round(mean, 4), "reason": None}


def build_supervised_batch(
    prompt_ids: list[int],
    answer_ids: list[int],
    max_length: int,
    eos_id: int | None = None,
) -> dict[str, Any]:
    """不为凑监督追加 EOS。目标放不下就跳过。"""
    if not answer_ids or answer_ids == [eos_id]:
        return {"skipped": True, "reason": "no_effective_target"}
    room = max_length - len(answer_ids)
    if room < 1 or len(prompt_ids) > room:
        return {"skipped": True, "reason": "prompt_exceeds_budget"}
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    supervised = [token for token in labels if token != -100]
    if not supervised or (eos_id is not None and supervised == [eos_id]):
        return {"skipped": True, "reason": "no_effective_target"}
    return {
        "skipped": False,
        "input_ids": input_ids,
        "labels": labels,
        "assistant_target_tokens": len(answer_ids),
        "sequence_tokens": len(input_ids),
    }


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        return {"value": None, "numerator": numerator, "denominator": denominator, "reason": "zero_denominator"}
    return {
        "value": round(numerator / denominator, 4),
        "numerator": numerator,
        "denominator": denominator,
        "reason": None,
    }


def cache_signature(payload: dict[str, Any]) -> str:
    import hashlib
    import json

    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
