"""结果对比。选例规则固定，不按看到的分数临时改口径。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

EXAMPLE_RULE = (
    "先按 case_id 排序。优先各取一条改善、一条退化、一条共同失败、一条规则与教师分歧；"
    "某一类没有就记缺失，不另找替代含义。再用 case_id 顺序补满 6 条。"
)


def _empty_judgment(state: str, reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "state": state,
        "rank": None,
        "legacy_rank": None,
        "rubric_version": None,
        "task_complete": None,
        "behavior_appropriate": None,
        "factual_consistency": None,
        "required_points_complete": None,
        "evidence_adequate": None,
        "no_unsupported_claims": None,
        "task_success": None,
        "reason_codes": list(reasons or []),
    }


def answer_judgment(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return _empty_judgment("not_reviewed")
    reasons = list(record.get("reason_codes") or [])
    rubric = str(record.get("rubric_version") or "rubric-v1")
    if record.get("execution_status") != "succeeded":
        judgment = _empty_judgment("technical_failed", reasons)
        judgment["rubric_version"] = rubric
        return judgment
    if record.get("decision") == "abstain":
        judgment = _empty_judgment("abstain", reasons)
        judgment["rubric_version"] = rubric
        return judgment
    dimensions = record.get("dimensions") or {}
    task = dimensions.get("required_points_complete") is True
    behavior = dimensions.get("behavior_appropriate") is True
    rank = int(task) + int(behavior) + int(record.get("decision") == "accept")
    factual = dimensions.get("factual_consistency")
    points = dimensions.get("required_points_complete")
    evidence = dimensions.get("evidence_adequate")
    unsupported = dimensions.get("no_unsupported_claims")
    task_success = None
    if rubric != "rubric-v1":
        needed = [factual, points, dimensions.get("behavior_appropriate"), evidence, unsupported]
        if all(item is not None for item in needed):
            task_success = all(item is True for item in needed) and record.get("decision") == "accept"
    return {
        "state": record.get("decision"),
        "rank": rank,
        "legacy_rank": rank,
        "rubric_version": rubric,
        "task_complete": task,
        "behavior_appropriate": behavior,
        "factual_consistency": factual,
        "required_points_complete": points,
        "evidence_adequate": evidence,
        "no_unsupported_claims": unsupported,
        "task_success": task_success,
        "reason_codes": reasons,
    }


def compare_pair(base: dict[str, Any], adapter: dict[str, Any]) -> str:
    """旧 rank 对照。持平只表示旧分相同，不是两者正确。"""
    if base.get("rank") is None or adapter.get("rank") is None:
        return "unresolved"
    if adapter["rank"] > base["rank"]:
        return "improve"
    if adapter["rank"] < base["rank"]:
        return "regress"
    return "tie"


def compare_outcomes(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    """新维度对照。旧口径无法推出的关系保持 unavailable。"""
    legacy = compare_pair(base, adapter)
    if base.get("task_success") is None or adapter.get("task_success") is None:
        return {"legacy_relation": legacy, "relation": "unavailable", "reason": "rubric_dimension_unavailable"}
    adapter_ok = adapter.get("task_success") is True
    base_ok = base.get("task_success") is True
    if adapter_ok and not base_ok:
        relation = "wrong_to_right"
    elif base_ok and not adapter_ok:
        relation = "right_to_wrong"
    elif adapter_ok and base_ok:
        relation = "both_correct"
    else:
        relation = "both_wrong"
    return {"legacy_relation": legacy, "relation": relation, "reason": None}


def rule_disagrees(rule_passed: bool | None, judgment: dict[str, Any]) -> bool:
    if rule_passed is None or judgment.get("state") in {None, "not_reviewed", "technical_failed", "abstain"}:
        return False
    teacher_ok = judgment.get("state") == "accept" and judgment.get("behavior_appropriate") is True and judgment.get("task_complete") is True
    if rule_passed and not teacher_ok:
        return True
    if rule_passed is False and teacher_ok:
        return True
    return False


def label_case(case: dict[str, Any], base: dict[str, Any], adapter: dict[str, Any]) -> list[str]:
    labels = []
    relation = compare_pair(base, adapter)
    if relation in {"improve", "regress"}:
        labels.append(relation)
    both_scored = base.get("rank") is not None and adapter.get("rank") is not None
    if both_scored and base.get("behavior_appropriate") is not True and adapter.get("behavior_appropriate") is not True:
        labels.append("both_fail")
    for role, judgment in (("base", base), ("adapter", adapter)):
        if rule_disagrees((case.get("answers") or {}).get(role, {}).get("rule_passed"), judgment):
            labels.append("rule_disagreement")
            break
    return labels


def latest_by_subject(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """同一 subject_id 下哈希不一致时不取最后一条。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("subject_id") or ""), []).append(row)
    latest: dict[str, dict[str, Any]] = {}
    for subject_id, group in grouped.items():
        hashes = {str(item.get("subject_hash") or "") for item in group}
        if len(hashes) == 1:
            latest[subject_id] = group[-1]
    return latest


def gold_bucket(status: str) -> str:
    if status == "teacher_single_accepted":
        return "accepted"
    if status == "needs_revision":
        return "revise"
    if status == "rejected":
        return "rejected"
    return "unresolved"


def annotate_cases(cases: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for case in cases:
        item = json_copy(case)
        judgments = {}
        for role in ("base", "adapter"):
            subject_id = ((item.get("answers") or {}).get(role) or {}).get("prediction_subject_id")
            judgments[role] = answer_judgment(records.get(str(subject_id or "")))
        item["judgments"] = judgments
        item["relation"] = compare_pair(judgments["base"], judgments["adapter"])
        item["labels"] = label_case(item, judgments["base"], judgments["adapter"])
        annotated.append(item)
    return annotated


def json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _model_counts(cases: list[dict[str, Any]], role: str) -> dict[str, int]:
    task = behavior = scored = 0
    for case in cases:
        judgment = case["judgments"][role]
        if judgment.get("state") in {None, "not_reviewed", "technical_failed", "abstain"}:
            continue
        scored += 1
        task += int(judgment.get("task_complete") is True)
        behavior += int(judgment.get("behavior_appropriate") is True)
    return {"scored": scored, "task_complete": task, "behavior_appropriate": behavior}


def _reason_counts(cases: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for case in cases:
        for judgment in case["judgments"].values():
            for code in judgment.get("reason_codes") or []:
                counts[code] += 1
            if judgment.get("no_unsupported_claims") is False:
                counts["no_unsupported_claims_false"] += 1
    return counts


def render_case(case: dict[str, Any]) -> str:
    lines = [
        f"### {case['case_id']}（{case.get('selected_as')}）",
        "",
        f"- 区块：{case.get('block')}",
        f"- 行为条件：{case.get('pattern') or case.get('expected_action')} / 期望 {case.get('expected_action')}",
        f"- 对照：{case.get('relation')}",
        f"- 来源族：{case.get('source_family_id')}",
        "",
        "题目：",
        "",
        case.get("question") or "",
        "",
        "学生可见上下文：",
        "",
        case.get("student_context") or "",
        "",
        f"候选答案：{case.get('candidate_answer') or ''}",
        "",
    ]
    for role in ("base", "adapter"):
        slot = (case.get("answers") or {}).get(role) or {}
        judgment = case["judgments"][role]
        lines.extend(
            [
                f"{role}：规则 {'通过' if slot.get('rule_passed') is True else '失败' if slot.get('rule_passed') is False else '未判定'}（辅助）；"
                f"教师 {judgment.get('state')}；任务完成 {judgment.get('task_complete')}；行为适当 {judgment.get('behavior_appropriate')}；"
                f"原因 {judgment.get('reason_codes') or []}",
                "",
                slot.get("text") or "（未匹配，不并入对照）",
                "",
            ]
        )
    return "\n".join(lines)


def select_examples(cases: list[dict[str, Any]], limit: int = 6) -> tuple[list[dict[str, Any]], list[str]]:
    ordered = sorted(cases, key=lambda item: str(item.get("case_id") or ""))
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    missing: list[str] = []
    for kind in ("improve", "regress", "both_fail", "rule_disagreement"):
        found = next((item for item in ordered if kind in item.get("labels", []) and item["case_id"] not in used), None)
        if found is None:
            missing.append(kind)
            continue
        chosen.append({**found, "selected_as": kind})
        used.add(found["case_id"])
    for item in ordered:
        if len(chosen) >= limit:
            break
        if item["case_id"] in used:
            continue
        chosen.append({**item, "selected_as": "case_id_fill"})
        used.add(item["case_id"])
    return chosen[:limit], missing


def _gold_summary(case_ids: list[str], aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets: Counter = Counter()
    formal = 0
    for case_id in case_ids:
        aggregate = aggregates.get(case_id) or {}
        buckets[gold_bucket(str(aggregate.get("teacher_review_status") or "unreviewed"))] += 1
        formal += int(bool(aggregate.get("ready_for_formal_human_eval")))
    return {
        "denominator": len(case_ids),
        "accepted": buckets["accepted"],
        "revise": buckets["revise"],
        "rejected": buckets["rejected"],
        "unresolved": buckets["unresolved"],
        "formal_ready": formal,
    }


def _comparison_summary(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(case.get("relation") for case in cases)
    return {
        "denominator": len(cases),
        "improve": counts["improve"],
        "regress": counts["regress"],
        "tie": counts["tie"],
        "unresolved": counts["unresolved"],
    }


def _disagreements(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for role in ("base", "adapter"):
            slot = (case.get("answers") or {}).get(role) or {}
            judgment = case["judgments"][role]
            if not rule_disagrees(slot.get("rule_passed"), judgment):
                continue
            rows.append(
                {
                    "case_id": case["case_id"],
                    "model_role": role,
                    "rule_passed": slot.get("rule_passed"),
                    "teacher_state": judgment.get("state"),
                    "behavior_appropriate": judgment.get("behavior_appropriate"),
                    "task_complete": judgment.get("task_complete"),
                    "reason_codes": judgment.get("reason_codes") or [],
                    "expected_action": case.get("expected_action"),
                }
            )
    return rows


def write_delivery(
    *,
    view_dir: str | Path,
    pilot_dir: str | Path,
    historical_cases: list[dict[str, Any]],
    pilot_cases: list[dict[str, Any]],
    urls: dict[str, str],
) -> dict[str, Any]:
    view = Path(view_dir)
    pilot = Path(pilot_dir)
    historical_records = latest_by_subject(_read_jsonl(view / "reviews.jsonl"))
    pilot_records = latest_by_subject(_read_jsonl(pilot / "reviews.jsonl"))
    historical_aggs = latest_by_subject(_read_jsonl(view / "review_aggregates.jsonl"))
    pilot_aggs = latest_by_subject(_read_jsonl(pilot / "review_aggregates.jsonl"))
    historical = annotate_cases(historical_cases, historical_records)
    pilot_annotated = annotate_cases(pilot_cases, pilot_records)
    examples, missing = select_examples(historical)
    usage = json.loads((pilot / "summary.json").read_text(encoding="utf-8"))["usage"]
    inference_report = {}
    report_path = pilot / "inference" / "eval_report.json"
    if report_path.is_file():
        inference_report = json.loads(report_path.read_text(encoding="utf-8"))
    historical_reasons = _reason_counts(historical)
    pilot_reasons = _reason_counts(pilot_annotated)
    metrics = {
        "budget": {
            "calls": usage["calls"],
            "call_denominator": usage["max_calls"],
            "tokens": usage["tokens"],
            "token_denominator": usage["max_tokens"],
            "estimated_cost": None,
            "currency": "unknown",
        },
        "historical": {
            "planned_cases": len(historical),
            "matched_predictions": sum(
                1 for case in historical for role in ("base", "adapter") if (case["answers"][role].get("match_status") == "matched")
            ),
            "prediction_denominator": len(historical) * 2,
            "gold": _gold_summary([case["case_id"] for case in historical], historical_aggs),
            "answers_reviewed": sum(1 for case in historical for role in ("base", "adapter") if case["judgments"][role]["state"] != "not_reviewed"),
            "per_model": {"base": _model_counts(historical, "base"), "adapter": _model_counts(historical, "adapter")},
            "comparison": _comparison_summary(historical),
            "rule_disagreements": _disagreements(historical),
            "reason_codes": dict(historical_reasons),
            "single_teacher_is_not_human_accuracy": True,
        },
        "b_pilot": {
            "parent_planned_n": 48,
            "selected_n": len(pilot_annotated),
            "experiment_complete": False,
            "gold": _gold_summary([case["case_id"] for case in pilot_annotated], pilot_aggs),
            "generated_n": inference_report.get("generated_n"),
            "failed_n": inference_report.get("failed_n"),
            "generation_status": inference_report.get("status"),
            "generation_executed": inference_report.get("executed"),
            "per_model": {"base": _model_counts(pilot_annotated, "base"), "adapter": _model_counts(pilot_annotated, "adapter")},
            "comparison": _comparison_summary(pilot_annotated),
            "rule_disagreements": _disagreements(pilot_annotated),
            "reason_codes": dict(pilot_reasons),
        },
        "example_rule": EXAMPLE_RULE,
        "example_missing": missing,
    }
    combined_cases = historical_cases + pilot_cases
    with (view / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in combined_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    review_lines = []
    for path in (view / "reviews.jsonl", pilot / "reviews.jsonl"):
        if path.is_file():
            review_lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    (view / "review_records.jsonl").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    agg_lines = []
    for path in (view / "review_aggregates.jsonl", pilot / "review_aggregates.jsonl"):
        if path.is_file():
            agg_lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    (view / "review_aggregates.jsonl").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")
    comparisons = []
    for case in historical + pilot_annotated:
        comparisons.append(
            {
                "case_id": case["case_id"],
                "block": case.get("block"),
                "relation": case.get("relation"),
                "labels": case.get("labels"),
                "expected_action": case.get("expected_action"),
                "pattern": case.get("pattern"),
                "base": case["judgments"]["base"],
                "adapter": case["judgments"]["adapter"],
                "base_rule_passed": (case.get("answers") or {}).get("base", {}).get("rule_passed"),
                "adapter_rule_passed": (case.get("answers") or {}).get("adapter", {}).get("rule_passed"),
            }
        )
    with (view / "case_comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for row in comparisons:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (view / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = view / "view_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update(
        {
            "banner": "历史真实预测与实验 B 小试",
            "limitation": "历史对照里的旧 adapter 来自脏工作区，tokenizer 与基座配置不一致。两侧差异不能直接归因于微调。实验 B 小试是本轮新推理，父协议分母仍是 48，不把 8 题写成实验完成。",
            "protocol_n": len(historical),
            "b_pilot_selected_n": len(pilot_annotated),
            "b_parent_planned_n": 48,
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    gold = metrics["historical"]["gold"]
    hist_cmp = metrics["historical"]["comparison"]
    b_cmp = metrics["b_pilot"]["comparison"]
    lines = [
        "# batch1 与实验 B 小试",
        "",
        "选例规则（在查看本页案例之前固定）：" + EXAMPLE_RULE,
        "",
        "单模型分数是探索性自动评估，不是人工准确率。8 题不能外推到 48 题父协议。工作区不是干净复现。",
        "",
        "## 访问",
        "",
        f"- 本机：{urls['local']}",
        f"- 局域网：{urls['lan']}",
        f"- SSH 转发：`{urls['ssh']}`，然后打开 {urls['local']}",
        "",
        "## 用量",
        "",
        f"教师调用 {usage['calls']} / {usage['max_calls']}，token {usage['tokens']} / {usage['max_tokens']}。费用 unknown，不记为 0。正式人工门禁打开数 {gold['formal_ready']}。",
        "",
        "## 历史 batch1",
        "",
        f"金标 {gold['accepted']} 接受、{gold['revise']} 修订、{gold['rejected']} 拒绝、{gold['unresolved']} 未决，分母 {gold['denominator']}。",
        f"预测对齐 {metrics['historical']['matched_predictions']} / {metrics['historical']['prediction_denominator']}。已评回答 {metrics['historical']['answers_reviewed']} / {metrics['historical']['prediction_denominator']}；金标未接受的题没有进入匿名回答评估。",
        f"base 任务完成 {metrics['historical']['per_model']['base']['task_complete']} / {metrics['historical']['per_model']['base']['scored']}，行为适当 {metrics['historical']['per_model']['base']['behavior_appropriate']} / {metrics['historical']['per_model']['base']['scored']}。",
        f"adapter 任务完成 {metrics['historical']['per_model']['adapter']['task_complete']} / {metrics['historical']['per_model']['adapter']['scored']}，行为适当 {metrics['historical']['per_model']['adapter']['behavior_appropriate']} / {metrics['historical']['per_model']['adapter']['scored']}。",
        f"同一题对照：改善 {hist_cmp['improve']}、退化 {hist_cmp['regress']}、持平 {hist_cmp['tie']}、未决 {hist_cmp['unresolved']}，分母 {hist_cmp['denominator']}。",
        f"规则与教师分歧 {len(metrics['historical']['rule_disagreements'])} 条。历史原因码 {metrics['historical']['reason_codes'] or '无'}。",
        "",
        "## 实验 B 小试",
        "",
        f"父协议计划 48。按核心问题 ID 排序取前 2 个完整条件组，选中 {len(pilot_annotated)}。生成状态 {inference_report.get('status')}，失败 {inference_report.get('failed_n')}，生成数 {inference_report.get('generated_n')}。",
        f"金标 {metrics['b_pilot']['gold']['accepted']} / {metrics['b_pilot']['gold']['denominator']} 探索性接受。",
        f"对照：改善 {b_cmp['improve']}、退化 {b_cmp['regress']}、持平 {b_cmp['tie']}、未决 {b_cmp['unresolved']}，分母 {b_cmp['denominator']}。这不是 48 题实验完成。",
        f"规则与教师分歧 {len(metrics['b_pilot']['rule_disagreements'])} 条。原因码 {metrics['b_pilot']['reason_codes'] or '无'}。",
        "",
        "## 案例",
        "",
        "缺失类型：" + ("、".join(missing) if missing else "无") + "。",
        "",
    ]
    lines.extend(render_case(case) for case in examples)
    lines.append("## 下一步")
    lines.append("")
    if hist_cmp["improve"] == 0 and b_cmp["improve"] == 0:
        lines.append("教师对照里没有出现 adapter 优于 base。下一步先核对评分口径，不把规则差额或单模型接受当成训练信号。")
    if not historical_reasons.get("over_refusal") and not pilot_reasons.get("over_refusal"):
        lines.append("教师没有标出过度拒答。")
    if historical_reasons.get("answered_without_evidence") or pilot_reasons.get("answered_without_evidence"):
        lines.append("教师标出了证据不足仍作答。")
    else:
        lines.append("教师没有标出证据不足仍作答。")
    (view / "result_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (pilot / "case_comparisons.jsonl").write_text((view / "case_comparisons.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (pilot / "metrics.json").write_text((view / "metrics.json").read_text(encoding="utf-8"), encoding="utf-8")
    (pilot / "result_report.md").write_text((view / "result_report.md").read_text(encoding="utf-8"), encoding="utf-8")
    return metrics
