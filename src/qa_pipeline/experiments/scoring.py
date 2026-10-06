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
_GAP_TOKENS = (
    "资料不足",
    "无法确定",
    "不能确定",
    "未提供",
    "没有给出",
    "没有提供",
    "不能据此",
    "不能支持",
    "不足以",
    "缺少",
    "无法回答",
    "不能回答",
    "未见",
    "未给出",
)
_CLARIFY_TOKENS = (
    "请补充",
    "哪一种",
    "哪种",
    "哪一个",
    "需要先说明",
    "需要明确",
    "请先明确",
    "需要确认",
    "请说明",
)
_CONFLICT_TOKENS = ("互相矛盾", "相互矛盾", "存在冲突", "前后矛盾", "资料冲突")
_SENTENCE_SPLIT = re.compile(r"[。！？\n；;]")


class AlignmentError(Exception):
    """预测、金标或干扰上下文无法按 ID 对齐。"""


def classify_action_legacy(text: str) -> str:
    """v1/v2 使用的旧分类。保留是为了让历史辅助分仍可复现。"""
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


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def _has_qty(text: str) -> bool:
    if _QTY.search(text or ""):
        return True
    return bool(re.search(r"(?:一次|一日|每天)\s*\d|\d+\s*(?:片|粒|袋|支|丸)", text or ""))


def _token_at(text: str, tokens: tuple[str, ...]) -> int | None:
    positions = [text.find(token) for token in tokens if token in text]
    return min(positions) if positions else None


def action_assessment(text: str) -> dict[str, Any]:
    """区分实质回答、部分回答、澄清、资料不足、真实冲突，以及先拒答再编造。"""
    raw = (text or "").strip()
    if not raw:
        return {"action": "needs_review", "refusal_then_fabrication": False, "reason": "empty"}
    sentences = _sentences(raw)
    fabrication = False
    seen_gap = False
    has_gap = False
    has_ask = False
    has_qty = False
    has_conflict = False
    for sentence in sentences:
        gap_at = _token_at(sentence, _GAP_TOKENS)
        ask = _token_at(sentence, _CLARIFY_TOKENS) is not None
        conflict = _token_at(sentence, _CONFLICT_TOKENS) is not None
        qty = _has_qty(sentence)
        object_gap = "不一致" in sentence and gap_at is not None
        if gap_at is not None or object_gap:
            has_gap = True
        if ask:
            has_ask = True
        if qty:
            has_qty = True
        if conflict:
            has_conflict = True
        if gap_at is not None and qty:
            match = _QTY.search(sentence) or re.search(r"\d", sentence)
            if match is not None and match.start() > gap_at:
                fabrication = True
        if seen_gap and qty and gap_at is None:
            fabrication = True
        if gap_at is not None or object_gap:
            seen_gap = True
    if fabrication:
        return {"action": "answer", "refusal_then_fabrication": True, "reason": "refusal_then_fabrication"}
    if has_conflict:
        return {"action": "state_conflict", "refusal_then_fabrication": False, "reason": "source_conflict"}
    if "不一致" in raw and not has_gap and not has_ask and not has_qty:
        return {"action": "needs_review", "refusal_then_fabrication": False, "reason": "object_mismatch_unspecified"}
    if has_ask and not has_qty:
        return {"action": "clarify", "refusal_then_fabrication": False, "reason": "clarify"}
    if has_gap and has_qty:
        return {"action": "partial_answer", "refusal_then_fabrication": False, "reason": "partial_with_gap"}
    if has_gap:
        return {"action": "state_insufficient", "refusal_then_fabrication": False, "reason": "insufficient"}
    if has_ask:
        return {"action": "clarify", "refusal_then_fabrication": False, "reason": "clarify"}
    if raw:
        return {"action": "answer", "refusal_then_fabrication": False, "reason": "substantive"}
    return {"action": "needs_review", "refusal_then_fabrication": False, "reason": "underspecified"}


def classify_action(text: str) -> str:
    return str(action_assessment(text)["action"])


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
    actual = classify_action_legacy(prediction)
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
        matrix[expected][actual] = matrix[expected].get(actual, 0) + 1
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
    unresolved = 0
    for case in cases:
        if case.get("passed") is None:
            unresolved += 1
            continue
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
        "row_count": len(cases),
        "unresolved_n": unresolved,
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


SCORER_V1 = "aux-rules-v1"
SCORER_V2 = "aux-rules-v2"
SCORER_V3 = "aux-rules-v3"
SCORER_NOTE = "规则辅助分，不是已验证的语义正确率。"
SCORER_V3_NOTE = "规则辅助分，只检查已实现的数值、条件与行为标签。不是语义正确率，也不能升为正式主指标。unsupported_numeric_assertions 只列出答案、证据、上下文和要点中都没有的额外数量，不判断非数值编造。"

_QTY = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|μg|ug|ml|mL|mmol|kg|℃|%)", re.I)
_NEG_LEAD = re.compile(r"(不得|禁用|不能|不可|不宜|不是|尚未|未进行)([^。；\n]{1,24})")
_RETRACT = re.compile(r"(不正确|并非如此|以上不对|实际上不是|不能采信|是错误的)")
_MASS = {"mg": 1.0, "g": 1000.0, "kg": 1_000_000.0, "ug": 0.001, "μg": 0.001}


EVAL_STRATA = ("seen_rephrase", "independent", "behavior")


def aggregate_layered(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """seen_rephrase、independent、behavior 分开汇总，不合成泛化总分。"""
    buckets = {name: [] for name in EVAL_STRATA}
    unspecified = []
    for case in cases:
        stratum = str(case.get("stratum") or case.get("layer") or "")
        if stratum in buckets:
            buckets[stratum].append(case)
        else:
            unspecified.append(case)
    summaries = {}
    for name, rows in buckets.items():
        summary = aggregate_families(rows)
        summary["row_count"] = len(rows)
        summaries[name] = summary
    return {
        "by_stratum": summaries,
        "seen_rephrase": summaries["seen_rephrase"],
        "independent": summaries["independent"],
        "behavior": summaries["behavior"],
        "unspecified_row_count": len(unspecified),
        "row_count": len(cases),
        "combined_generalization_score": None,
        "note": "seen_rephrase、independent、behavior 分开汇总，不合成泛化总分。",
    }


def _unit_key(unit: str) -> str:
    return unit.replace("μ", "u").lower()


def _normalize_qty(number: str, unit: str) -> tuple[float, str] | None:
    try:
        value = float(number)
    except (TypeError, ValueError):
        return None
    key = _unit_key(unit)
    if key in _MASS:
        return value * _MASS[key], "mass"
    if key == "ml":
        return value, "volume"
    if key == "mmol":
        return value, "mmol"
    if key == "%":
        return value, "percent"
    if unit == "℃":
        return value, "temp"
    return value, key


def _qty_equivalent(left: tuple[float, str], right: tuple[float, str]) -> bool:
    if left[1] != right[1]:
        return False
    scale = max(abs(left[0]), abs(right[0]), 1.0)
    return abs(left[0] - right[0]) <= 0.001 * scale


def _extract_qtys(text: str) -> list[tuple[str, str, tuple[float, str]]]:
    found = []
    for match in _QTY.finditer(text or ""):
        norm = _normalize_qty(match.group(1), match.group(2))
        if norm is not None:
            found.append((match.group(1), match.group(2), norm))
    return found


def _object_qty(text: str, obj: str) -> tuple[str, str, tuple[float, str]] | None:
    idx = (text or "").find(obj)
    if idx < 0 or not obj:
        return None
    window = text[idx : idx + 48]
    match = _QTY.search(window)
    if match is None:
        return None
    norm = _normalize_qty(match.group(1), match.group(2))
    if norm is None:
        return None
    return match.group(1), match.group(2), norm


def _binding_errors(prediction: str, case: dict[str, Any]) -> list[str]:
    bindings = list(case.get("numeric_bindings") or [])
    if not bindings:
        return []
    errors = []
    normalized = []
    for binding in bindings:
        norm = _normalize_qty(str(binding.get("number")), str(binding.get("unit") or ""))
        normalized.append(norm)
    for index, binding in enumerate(bindings):
        found = _object_qty(prediction, str(binding.get("object") or ""))
        own = normalized[index]
        if found is None or own is None:
            continue
        raw, _unit, pred_norm = found
        if _qty_equivalent(pred_norm, own):
            continue
        swapped = False
        for other_index, other in enumerate(normalized):
            if other_index == index or other is None:
                continue
            if _qty_equivalent(pred_norm, other):
                swapped = True
                break
        if swapped:
            errors.append("numeric_object_swap")
        elif raw == str(binding.get("number")):
            errors.append("wrong_unit")
        else:
            errors.append("numeric")
    return list(dict.fromkeys(errors))


def _point_hit(prediction: str, point: str, synonyms: list[list[str]]) -> bool:
    if not point:
        return False
    if point in prediction or token_f1(prediction, point) >= 0.55:
        return True
    packed_point = re.sub(r"\s+", "", point)
    packed_pred = re.sub(r"\s+", "", prediction or "")
    for group in synonyms:
        packed = [re.sub(r"\s+", "", item) for item in group if item]
        if any(item and item in packed_point for item in packed) and any(item and item in packed_pred for item in packed):
            return True
    return False


def _coverage_v2(prediction: str, points: list[str], synonyms: list[list[str]]) -> dict[str, Any]:
    usable = [point for point in points if point]
    if not usable:
        return {"value": None, "reason": "no_points", "hits": []}
    hits = [_point_hit(prediction, point, synonyms) for point in usable]
    value = round(sum(1 for hit in hits if hit) / len(hits), 4)
    return {"value": value, "reason": None, "hits": hits}


def _negation_scope(prediction: str, points: list[str]) -> bool:
    for point in points:
        match = _NEG_LEAD.search(point or "")
        if not match:
            continue
        key = match.group(2).strip()[:12]
        if len(key) < 1:
            continue
        pos = (prediction or "").find(key)
        if pos < 0:
            continue
        prefix = prediction[max(0, pos - 8) : pos]
        if not re.search(r"[不未无非]|禁用", prefix):
            return True
    return False


def _retracted(prediction: str, points: list[str], answer: str) -> bool:
    anchors = [point for point in points if point] or ([answer] if answer else [])
    for anchor in anchors:
        token = anchor.strip()[:16]
        if len(token) < 2:
            continue
        idx = (prediction or "").find(token)
        if idx >= 0 and _RETRACT.search(prediction[idx:]):
            return True
    return False


def _unsupported(prediction: str, case: dict[str, Any], points: list[str]) -> list[str]:
    allowed_text = "\n".join(
        [
            str(case.get("answer") or ""),
            str(case.get("evidence") or ""),
            str(case.get("context") or ""),
            "\n".join(points),
        ]
    )
    allowed = [_qty[2] for _qty in _extract_qtys(allowed_text)]
    extras = []
    for raw, unit, norm in _extract_qtys(prediction or ""):
        if any(_qty_equivalent(norm, item) for item in allowed):
            continue
        extras.append(f"{raw}{unit}")
    return extras


def score_task_v2(prediction: str, case: dict[str, Any]) -> dict[str, Any]:
    """维度化辅助分。不能替代人工审核，也不能当作语义正确率。"""
    points = [str(point) for point in (case.get("answer_points") or []) if str(point).strip()]
    if not points and case.get("answer"):
        points = [str(case["answer"])]
    synonyms = [list(group) for group in (case.get("synonym_groups") or [])]
    coverage = _coverage_v2(prediction or "", points, synonyms)
    expected = str(case.get("expected_action") or "")
    if expected not in EXPECTED_ACTIONS:
        expected = str(case.get("expected_action") or "answer")
    actual = classify_action_legacy(prediction or "")
    errors = _binding_errors(prediction or "", case)
    if not case.get("numeric_bindings"):
        errors.extend(err for err in critical_errors(prediction or "", points) if err not in errors)
    if _negation_scope(prediction or "", points):
        errors.append("negation_scope")
    if _retracted(prediction or "", points, str(case.get("answer") or "")):
        errors.append("retracted")
    missing = []
    for condition in case.get("required_conditions") or []:
        text = str(condition).strip()
        if text and text not in (prediction or "") and token_f1(prediction or "", text) < 0.55:
            missing.append(text)
    if missing:
        errors.append("missing_condition")
    unsupported = _unsupported(prediction or "", case, points)
    evidence_state = str(case.get("evidence_state") or "")
    over_refusal = expected == "answer" and evidence_state == "sufficient" and actual == "state_insufficient"
    if over_refusal:
        errors.append("over_refusal")
    errors = list(dict.fromkeys(errors))
    if expected in {"answer", "partial_answer"}:
        relevant = coverage["value"] not in {None, 0}
    else:
        relevant = True
    action_ok = actual == expected or (
        expected == "partial_answer" and actual == "answer" and coverage["value"] not in {None, 0}
    )
    if expected == "state_insufficient" and actual == "answer":
        action_ok = False
    if over_refusal:
        action_ok = False
    passed = bool(relevant and not errors and action_ok and not unsupported)
    if expected == "answer":
        passed = bool(passed and coverage["value"] == 1)
    return {
        "scorer_version": SCORER_V2,
        "task_relevance": relevant,
        "required_point_coverage": coverage["value"],
        "critical_errors": [err for err in errors if err != "unsupported"],
        "unsupported_assertions": unsupported,
        "missing_conditions": missing,
        "expected_behavior": {"expected": expected, "actual": actual, "ok": action_ok},
        "auxiliary_f1": token_f1(prediction or "", "\n".join(points)) if points else None,
        "auxiliary_em": exact_match(prediction or "", str(case.get("answer") or "")) if _single_value(case) else None,
        "passed": passed,
        "semantic_accuracy_verified": False,
        "note": SCORER_NOTE,
    }


def _scoring_points(case: dict[str, Any]) -> list[str]:
    required = [str(point) for point in (case.get("required_points") or []) if str(point).strip()]
    if required:
        return required
    points = [str(point) for point in (case.get("answer_points") or []) if str(point).strip()]
    if not points and case.get("answer"):
        points = [str(case["answer"])]
    return points


def _point_hit_v3(prediction: str, point: str, synonyms: list[list[str]]) -> bool:
    """同义组只替换与整个要点等价的说法，不能用组里的一个片段算复合要点已满足。"""
    if not point:
        return False
    if point in prediction or token_f1(prediction, point) >= 0.55:
        return True
    packed_point = re.sub(r"\s+", "", point)
    packed_pred = re.sub(r"\s+", "", prediction or "")
    if packed_point and packed_point in packed_pred:
        return True
    for group in synonyms:
        members = [str(item).strip() for item in group if str(item).strip()]
        if point not in members and packed_point not in {re.sub(r"\s+", "", item) for item in members}:
            continue
        if any(item in prediction or re.sub(r"\s+", "", item) in packed_pred for item in members):
            return True
    return False


def _unavailable_errors(prediction: str, case: dict[str, Any]) -> list[str]:
    errors = []
    text = prediction or ""
    for point in case.get("unavailable_points") or []:
        raw = str(point).strip()
        if raw and raw in text:
            errors.append("asserted_unavailable")
            continue
        for number, unit, _norm in _extract_qtys(raw):
            if f"{number}{unit}" in re.sub(r"\s+", "", text) or f"{number} {unit}" in text:
                errors.append("asserted_unavailable")
    return list(dict.fromkeys(errors))


def score_task_v3(prediction: str, case: dict[str, Any]) -> dict[str, Any]:
    """维度化辅助分。不能替代人工审核，也不能当作语义主指标。"""
    points = _scoring_points(case)
    synonyms = [list(group) for group in (case.get("synonym_groups") or [])]
    coverage = _coverage_v2(prediction or "", points, synonyms)
    coverage["hits"] = [_point_hit_v3(prediction or "", point, synonyms) for point in points] if points else []
    if points:
        coverage["value"] = round(sum(1 for hit in coverage["hits"] if hit) / len(points), 4)
        coverage["reason"] = None
    expected = str(case.get("expected_action") or "")
    if expected not in EXPECTED_ACTIONS:
        expected = str(case.get("expected_action") or "answer")
    assessment = action_assessment(prediction or "")
    actual = str(assessment["action"])
    if actual == "needs_review":
        return {
            "scorer_version": SCORER_V3,
            "task_relevance": None,
            "required_point_coverage": coverage["value"],
            "critical_errors": [],
            "unsupported_numeric_assertions": [],
            "missing_conditions": [],
            "expected_behavior": {"expected": expected, "actual": actual, "ok": None},
            "auxiliary_f1": token_f1(prediction or "", "\n".join(points)) if points else None,
            "auxiliary_em": None,
            "passed": None,
            "judgment": "pending_review",
            "semantic_accuracy_verified": False,
            "metric_role": "auxiliary",
            "note": SCORER_V3_NOTE,
        }
    errors = _binding_errors(prediction or "", case)
    if not case.get("numeric_bindings"):
        errors.extend(err for err in critical_errors(prediction or "", points) if err not in errors)
    errors.extend(_unavailable_errors(prediction or "", case))
    if _negation_scope(prediction or "", points):
        errors.append("negation_scope")
    if _retracted(prediction or "", points, str(case.get("answer") or "")):
        errors.append("retracted")
    missing = []
    for condition in case.get("required_conditions") or []:
        text = str(condition).strip()
        if text and text not in (prediction or "") and token_f1(prediction or "", text) < 0.55:
            missing.append(text)
    if missing:
        errors.append("missing_condition")
    unsupported = _unsupported(prediction or "", case, points)
    if assessment["refusal_then_fabrication"]:
        errors.append("refusal_then_fabrication")
    evidence_state = str(case.get("evidence_state") or "")
    over_refusal = expected == "answer" and evidence_state == "sufficient" and actual == "state_insufficient"
    if over_refusal:
        errors.append("over_refusal")
    errors = list(dict.fromkeys(errors))
    if expected in {"answer", "partial_answer"}:
        relevant = coverage["value"] not in {None, 0}
    else:
        relevant = True
    action_ok = actual == expected or (
        expected == "partial_answer" and actual == "answer" and coverage["value"] not in {None, 0}
    )
    if expected == "state_insufficient" and actual == "answer":
        action_ok = False
    if over_refusal or assessment["refusal_then_fabrication"]:
        action_ok = False
    passed = bool(relevant and not errors and action_ok and not unsupported)
    if expected == "answer":
        passed = bool(passed and coverage["value"] == 1)
    return {
        "scorer_version": SCORER_V3,
        "task_relevance": relevant,
        "required_point_coverage": coverage["value"],
        "critical_errors": [err for err in errors if err != "unsupported"],
        "unsupported_numeric_assertions": unsupported,
        "missing_conditions": missing,
        "expected_behavior": {"expected": expected, "actual": actual, "ok": action_ok},
        "auxiliary_f1": token_f1(prediction or "", "\n".join(points)) if points else None,
        "auxiliary_em": exact_match(prediction or "", str(case.get("answer") or "")) if _single_value(case) else None,
        "passed": passed,
        "judgment": "auxiliary_rule",
        "semantic_accuracy_verified": False,
        "metric_role": "auxiliary",
        "note": SCORER_V3_NOTE,
    }


def visible_support(case: dict[str, Any]) -> dict[str, Any]:
    """必答要点应出现在学生可见上下文中；不可见要点不应被当成必须回答。"""
    context = str(case.get("context") or "")
    required = [str(point) for point in (case.get("required_points") or []) if str(point).strip()]
    if not required:
        required = [str(point) for point in (case.get("answer_points") or []) if str(point).strip()]
    unavailable = [str(point) for point in (case.get("unavailable_points") or []) if str(point).strip()]
    return {
        "required_not_in_context": [point for point in required if point not in context],
        "unavailable_still_visible": [point for point in unavailable if point in context],
    }


def reference_self_check(case: dict[str, Any]) -> dict[str, Any]:
    """用参考答案给自己评分。失败原因分开记录，不改金标措辞。"""
    scored = score_task_v3(str(case.get("answer") or ""), case)
    behavior = scored["expected_behavior"]
    reasons: list[str] = []
    if scored.get("judgment") == "pending_review":
        reasons.append("classifier_pending_review")
    elif behavior.get("ok") is not True:
        reasons.append(f"action:{behavior.get('expected')}->{behavior.get('actual')}")
    if scored.get("critical_errors"):
        reasons.append("critical_errors:" + ",".join(scored["critical_errors"]))
    if scored.get("unsupported_numeric_assertions"):
        reasons.append("unsupported_numeric")
    if scored.get("passed") is not True and not reasons:
        reasons.append("not_passed")
    return {
        "case_id": str(case.get("case_id") or case.get("id") or ""),
        "expected_behavior": behavior.get("expected"),
        "actual_action": behavior.get("actual"),
        "passed": scored.get("passed"),
        "reasons": reasons,
        "judgment": scored.get("judgment"),
    }


def prediction_item_signature(
    case: dict[str, Any],
    model_id: str,
    adapter_id: str,
    template_id: str,
    infer_config: dict[str, Any],
    model_role: str,
    model_input: Any | None = None,
) -> str:
    if model_input is None:
        messages = case.get("messages")
        if messages:
            model_input = {"messages": messages}
        else:
            model_input = {
                "question": case.get("question") or "",
                "context": case.get("context") or "",
                "task_mode": case.get("task_mode") or "",
            }
    return cache_signature(
        {
            "input": model_input,
            "model_id": model_id,
            "adapter_id": adapter_id if model_role == "adapter" else "",
            "model_role": model_role,
            "template_id": template_id,
            "infer_config": infer_config,
        }
    )


def score_item_signature(prediction_signature: str, case: dict[str, Any], scorer_version: str) -> str:
    gold = {
        "answer": case.get("answer") or "",
        "answer_points": case.get("answer_points") or [],
        "expected_action": case.get("expected_action"),
        "evidence_state": case.get("evidence_state"),
        "required_conditions": case.get("required_conditions") or [],
        "numeric_bindings": case.get("numeric_bindings") or [],
    }
    return cache_signature({"prediction": prediction_signature, "gold": gold, "scorer": scorer_version})


def _index_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("case_id") or "") or "text" not in row:
            failures.append({"case_id": None, "reason": "invalid_prediction", "kind": "technical"})
            continue
        case_id = str(row["case_id"])
        if case_id in indexed:
            failures.append({"case_id": case_id, "reason": "duplicate_prediction", "kind": "technical"})
            continue
        indexed[case_id] = row
    return indexed, failures


def message_digest(messages: Any) -> str:
    normalized = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        normalized.append({"role": str(message.get("role") or ""), "content": str(message.get("content") or "")})
    return cache_signature({"messages": normalized})


def align_prediction_groups(
    cases: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    require_input_identity: bool = False,
    expected_messages: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """按 case_id 对齐。缺失、重复或多余预测不缩小计划分母，也不能当成评分通过。

    require_input_identity 为真时，同 ID 但 messages 不同、缺签名、带 error 或模型身份不一致都会失败。
    """
    failures: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    planned_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            failures.append({"case_id": index, "reason": "invalid_case", "kind": "technical"})
            planned_ids.append(f"invalid-{index}")
            continue
        case_id = str(case.get("case_id") or case.get("id") or "")
        if not case_id:
            failures.append({"case_id": index, "reason": "missing_id", "kind": "technical"})
            planned_ids.append(f"missing-{index}")
            continue
        if case_id in seen_cases:
            failures.append({"case_id": case_id, "reason": "duplicate_case_id", "kind": "technical"})
        else:
            seen_cases.add(case_id)
        planned_ids.append(case_id)
    indexed = {}
    id_sets = []
    for name, rows in groups.items():
        if name not in {"base", "adapter"}:
            failures.append({"case_id": None, "reason": "unknown_model_role", "kind": "technical", "model": name})
        mapping, group_failures = _index_predictions(rows)
        for item in group_failures:
            item["model"] = name
            failures.append(item)
        identities = {str(row.get("model_id") or "") for row in mapping.values() if row.get("model_id")}
        adapters = {str(row.get("adapter_id") or "") for row in mapping.values() if "adapter_id" in row}
        if len(identities) > 1:
            failures.append({"case_id": None, "reason": "model_identity_mismatch", "kind": "technical", "model": name})
        if name == "adapter" and len(adapters) > 1:
            failures.append({"case_id": None, "reason": "adapter_identity_mismatch", "kind": "technical", "model": name})
        indexed[name] = mapping
        id_sets.append(set(mapping))
    planned_set = {item for item in planned_ids if not str(item).startswith(("invalid-", "missing-"))}
    if id_sets:
        common = set.intersection(*id_sets) if len(id_sets) > 1 else id_sets[0]
        if len(id_sets) > 1 and any(item != id_sets[0] for item in id_sets[1:]):
            failures.append({"case_id": None, "reason": "model_prediction_set_mismatch", "kind": "technical"})
        for case_id in sorted(planned_set - common):
            failures.append({"case_id": case_id, "reason": "missing_prediction", "kind": "technical"})
        extras = set.union(*id_sets) - planned_set if id_sets else set()
        for case_id in sorted(extras):
            failures.append({"case_id": case_id, "reason": "extra_prediction", "kind": "technical"})
    ok = not failures
    aligned = []
    if ok:
        for case in cases:
            case_id = str(case.get("case_id") or case.get("id"))
            predictions = {name: indexed[name][case_id] for name in groups}
            if require_input_identity:
                expected = None
                if expected_messages is not None:
                    expected = expected_messages.get(case_id)
                elif case.get("messages"):
                    expected = case.get("messages")
                expected_hash = message_digest(expected) if expected is not None else ""
                for name, pred in predictions.items():
                    if pred.get("error"):
                        failures.append({"case_id": case_id, "reason": "prediction_error", "kind": "technical", "model": name, "detail": pred.get("error")})
                    if not pred.get("messages"):
                        failures.append({"case_id": case_id, "reason": "missing_input_identity", "kind": "technical", "model": name})
                    elif expected is not None and message_digest(pred.get("messages")) != expected_hash:
                        failures.append({"case_id": case_id, "reason": "input_mismatch", "kind": "technical", "model": name})
                    if not pred.get("signature"):
                        failures.append({"case_id": case_id, "reason": "missing_signature", "kind": "technical", "model": name})
                    if not pred.get("model_id"):
                        failures.append({"case_id": case_id, "reason": "missing_model_identity", "kind": "technical", "model": name})
                    if str(pred.get("model_role") or name) != name:
                        failures.append({"case_id": case_id, "reason": "model_role_mismatch", "kind": "technical", "model": name})
            if not failures:
                aligned.append({"case": case, "predictions": predictions})
        if failures:
            ok = False
            aligned = []
    planned_n = len(cases)
    scored_n = len(aligned) if ok else 0
    affected = sorted({str(item.get("case_id")) for item in failures if item.get("case_id") not in {None, ""}})
    return {
        "ok": ok,
        "planned_n": planned_n,
        "generated_n": {name: len(rows) for name, rows in groups.items()},
        "scored_n": scored_n,
        "failed_n": planned_n - scored_n if not ok else 0,
        "failures": failures,
        "affected_ids": affected,
        "aligned": aligned,
        "denominator": planned_n,
        "silent_shrink": False,
        "count_units": {
            "planned_n": "questions",
            "generated_n": "model_answers",
            "scored_n": "questions",
            "failed_n": "questions",
        },
        "system_rule": "技术失败计为任务未完成。分母是 planned_n，不把失败样本去掉后再计算准确率。",
    }
