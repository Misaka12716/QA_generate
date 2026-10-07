"""结果页只读适配。不重算历史 rubric，也不在读取时调用模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..task_mode import CLOSED_BOOK, RAG_GROUNDED, TaskModeError, canonical_task_mode

HISTORICAL_BATCH = "historical"
PILOT_BATCH = "b_pilot"
CLOSED_BATCH = "cb1"
JOINT_SNAPSHOT_NOTE = "batch1 与 B 小试的 metrics.json 是同一份联合快照，不是两次独立实验。"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _metric(
    name: str,
    numerator: int | None,
    denominator: int | None,
    *,
    unit: str,
    task_mode: str,
    eval_subset: str | None,
    review_source: str,
    rubric_version: str,
    status: str,
    limitations: list[str],
) -> dict[str, Any]:
    value = None
    if numerator is not None and denominator not in (None, 0):
        value = round(numerator / denominator, 4)
    return {
        "name": name,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "unit": unit,
        "task_mode": task_mode,
        "eval_subset": eval_subset,
        "review_source": review_source,
        "rubric_version": rubric_version,
        "status": status,
        "limitations": limitations,
    }


def discover_roots(root: Path) -> list[Path]:
    found = [root]
    parent = root.parent
    for name in ("batch1_view_20261007", "b_pilot_20261007", "cb1_20261007"):
        candidate = parent / name
        if candidate.is_dir() and candidate.resolve() not in {item.resolve() for item in found}:
            found.append(candidate)
    if root.name in {"runs", "batch1_view_20261007", "b_pilot_20261007", "cb1_20261007"}:
        for name in ("batch1_view_20261007", "b_pilot_20261007", "cb1_20261007"):
            candidate = root / name
            if candidate.is_dir() and candidate.resolve() not in {item.resolve() for item in found}:
                found.append(candidate)
    return found


def _joint_metrics(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    for candidate in discover_roots(root):
        payload = _read_json(candidate / "metrics.json")
        if payload and "historical" in payload and "b_pilot" in payload:
            return payload, candidate.name
    return None, None


def _closed_dir(root: Path) -> Path | None:
    for candidate in discover_roots(root):
        protocol = _read_json(candidate / "protocol.json")
        metrics = _read_json(candidate / "metrics.json")
        mode = ""
        if protocol:
            mode = str(protocol.get("task_mode") or "")
        if metrics and not mode:
            mode = str(metrics.get("task_mode") or "")
        if mode == CLOSED_BOOK:
            return candidate
    return None


def list_batches(root: Path) -> list[dict[str, Any]]:
    batches = []
    metrics, origin = _joint_metrics(root)
    if metrics:
        batches.append(
            {
                "batch_id": HISTORICAL_BATCH,
                "label": "历史 batch1",
                "task_mode": RAG_GROUNDED,
                "origin": origin,
                "combined_with": None,
            }
        )
        batches.append(
            {
                "batch_id": PILOT_BATCH,
                "label": "实验 B 小试",
                "task_mode": RAG_GROUNDED,
                "origin": origin,
                "combined_with": None,
            }
        )
    closed = _closed_dir(root)
    batches.append(
        {
            "batch_id": CLOSED_BATCH,
            "label": "闭卷 CB-1",
            "task_mode": CLOSED_BOOK,
            "origin": None if closed is None else closed.name,
            "available": closed is not None,
        }
    )
    return batches


def _legacy_block(block: dict[str, Any], *, task_mode: str, subset: str, limitations: list[str]) -> dict[str, Any]:
    comparison = block.get("comparison") or {}
    planned = comparison.get("denominator")
    tie = comparison.get("tie")
    unresolved = comparison.get("unresolved")
    improve = comparison.get("improve")
    regress = comparison.get("regress")
    return {
        "legacy_relation": {
            "improve": improve,
            "regress": regress,
            "tie": tie,
            "unresolved": unresolved,
            "label": "旧口径持平" if tie else None,
            "rubric_version": "rubric-v1",
            "note": "旧口径持平不能映射为两者正确。",
        },
        "outcomes": {
            "wrong_to_right": None,
            "right_to_wrong": None,
            "both_correct": None,
            "both_wrong": None,
            "unresolved": unresolved,
            "status": "unavailable",
            "reason": "rubric-v1 的 rank 推不出新维度",
        },
        "cards": [
            _metric("计划数", planned, planned, unit="input", task_mode=task_mode, eval_subset=subset, review_source="teacher_single", rubric_version="rubric-v1", status="available", limitations=limitations),
            _metric("旧口径持平", tie, planned, unit="pair", task_mode=task_mode, eval_subset=subset, review_source="teacher_single", rubric_version="rubric-v1", status="available", limitations=limitations),
            _metric("未决", unresolved, planned, unit="pair", task_mode=task_mode, eval_subset=subset, review_source="teacher_single", rubric_version="rubric-v1", status="available", limitations=limitations),
            _metric("改善", improve, planned, unit="pair", task_mode=task_mode, eval_subset=subset, review_source="teacher_single", rubric_version="rubric-v1", status="available", limitations=limitations + ["改善按旧 rank，不是事实对错"]),
        ],
    }


def _rag_comparison(root: Path, batch: str) -> dict[str, Any]:
    metrics, origin = _joint_metrics(root)
    limitations = [
        "教师辅助评价，不是人工准确率。",
        "教师模型 revision 为 unknown。",
        "历史 adapter 来自脏工作区，tokenizer 配置与基座不一致。",
        JOINT_SNAPSHOT_NOTE,
    ]
    if metrics is None:
        return {
            "availability": "missing",
            "status": "error",
            "task_mode": RAG_GROUNDED,
            "batch_id": batch,
            "headline": "没有可读的历史指标文件。",
            "error": {"code": "missing_metrics", "message": "没有包含历史块与 B 小试块的 metrics.json"},
            "cards": [],
            "bars": [],
            "limitations": ["不能用 0 填充分母"],
        }
    if batch == PILOT_BATCH:
        block = metrics.get("b_pilot") or {}
        planned_parent = block.get("parent_planned_n")
        selected = block.get("selected_n")
        shaped = _legacy_block(block, task_mode=RAG_GROUNDED, subset="evidence_conditions", limitations=limitations)
        headline = f"实验 B 小试完成 {selected}/{planned_parent} 个条件输入，旧口径持平 {block.get('comparison', {}).get('tie')}。这不是完整协议，也不能与历史题合成一次实验。"
        coverage = _metric(
            "父协议覆盖",
            selected,
            planned_parent,
            unit="input",
            task_mode=RAG_GROUNDED,
            eval_subset="evidence_conditions",
            review_source="teacher_single",
            rubric_version="rubric-v1",
            status="partial",
            limitations=limitations + ["8 个条件来自两个核心问题，不是 8 个独立来源"],
        )
        shaped["cards"][0] = coverage
    else:
        block = metrics.get("historical") or {}
        shaped = _legacy_block(block, task_mode=RAG_GROUNDED, subset="historical_batch1", limitations=limitations)
        comparison = block.get("comparison") or {}
        headline = (
            f"历史 batch1 计划 {comparison.get('denominator')} 题：旧口径持平 {comparison.get('tie')}、"
            f"未决 {comparison.get('unresolved')}、改善 {comparison.get('improve')}、退化 {comparison.get('regress')}。"
            "教师辅助评价未观察到改善。"
        )
        coverage = _metric(
            "成对覆盖",
            comparison.get("tie"),
            comparison.get("denominator"),
            unit="pair",
            task_mode=RAG_GROUNDED,
            eval_subset="historical_batch1",
            review_source="teacher_single",
            rubric_version="rubric-v1",
            status="partial",
            limitations=limitations,
        )
    per_model = block.get("per_model") or {}
    paired_denominator = (block.get("comparison") or {}).get("denominator")
    planned = block.get("parent_planned_n") if batch == PILOT_BATCH else paired_denominator
    bars = []
    for role, label in (("base", "微调前"), ("adapter", "微调后")):
        slot = per_model.get(role) or {}
        bars.append(
            {
                "id": f"{role}_task_complete",
                "label": f"{label} 任务完成",
                "numerator": slot.get("task_complete"),
                "denominator": planned,
                "scored": slot.get("scored"),
                "unit": "题 / 计划题",
                "note": "分母是计划题数。已评分题上的条件通过率另计，不与另一模型的已评分分母画成同一百分比。",
            }
        )
    bars.append(
        {
            "id": "legacy_tie",
            "label": "旧口径持平",
            "numerator": (block.get("comparison") or {}).get("tie"),
            "denominator": planned,
            "unit": "对 / 计划题",
            "filter": "tie",
        }
    )
    bars.append(
        {
            "id": "unresolved",
            "label": "未决",
            "numerator": (block.get("comparison") or {}).get("unresolved"),
            "denominator": planned,
            "unit": "对 / 计划题",
            "filter": "unresolved",
        }
    )
    return {
        "availability": "summary_only",
        "status": "partial",
        "task_mode": RAG_GROUNDED,
        "batch_id": HISTORICAL_BATCH if batch != PILOT_BATCH else PILOT_BATCH,
        "batch_label": "历史 batch1" if batch != PILOT_BATCH else "实验 B 小试",
        "origin": origin,
        "headline": headline,
        "tags": ["教师辅助评价", "探索性", "历史配置受限", "rubric-v1"],
        "cards": [coverage, *shaped["cards"][1:4]] if batch == PILOT_BATCH else [*shaped["cards"][:3], coverage],
        "legacy_relation": shaped["legacy_relation"],
        "outcomes": shaped["outcomes"],
        "bars": bars,
        "scale_denominator": planned,
        "limitations": limitations,
        "formal_ready": (block.get("gold") or {}).get("formal_ready"),
        "budget": metrics.get("budget"),
        "disagreement_ids": sorted({row.get("case_id") for row in block.get("rule_disagreements") or [] if row.get("case_id")}),
        "cases_status": "summary_only",
        "cases_missing_reason": "仅有汇总，案例产物缺失",
    }


def _closed_empty(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    limitations = ["闭卷对照尚未执行。不能显示 0% 准确率，也不能借用有资料问答的分数。"]
    if protocol:
        limitations.append(str(protocol.get("status_note") or "协议已准备，预测或审核尚未完成。"))
    return {
        "availability": "empty",
        "status": "not_executed",
        "task_mode": CLOSED_BOOK,
        "batch_id": CLOSED_BATCH,
        "batch_label": "闭卷 CB-1",
        "headline": "尚无闭卷训练对照",
        "tags": ["未执行"],
        "cards": [],
        "bars": [],
        "legacy_relation": None,
        "outcomes": {
            "wrong_to_right": None,
            "right_to_wrong": None,
            "both_correct": None,
            "both_wrong": None,
            "unresolved": None,
            "status": "not_executed",
            "reason": "尚未产生闭卷预测审核",
        },
        "scale_denominator": None,
        "limitations": limitations,
        "protocol": None
        if protocol is None
        else {
            "status": protocol.get("status"),
            "train_planned": protocol.get("train_planned"),
            "eval_planned": protocol.get("eval_planned"),
            "teacher_budget": protocol.get("teacher_budget"),
            "blockers": protocol.get("blockers") or [],
        },
        "default_subset": "CB-paraphrase",
        "cases_status": "empty",
        "cases_missing_reason": "尚无闭卷案例",
    }


def _closed_comparison(root: Path) -> dict[str, Any]:
    directory = _closed_dir(root)
    if directory is None:
        return _closed_empty()
    protocol = _read_json(directory / "protocol.json") or {}
    metrics = _read_json(directory / "metrics.json") or {}
    if metrics.get("task_mode") != CLOSED_BOOK or metrics.get("status") in {None, "not_executed", "prepared"}:
        payload = _closed_empty(protocol)
        payload["availability"] = "empty" if not protocol else "partial"
        payload["protocol_dir"] = directory.name
        if protocol and metrics.get("status") != "predicted":
            payload["headline"] = "尚无闭卷训练对照"
            payload["status"] = protocol.get("status") or "prepared"
        return payload
    return {
        "availability": metrics.get("availability") or "partial",
        "status": metrics.get("status"),
        "task_mode": CLOSED_BOOK,
        "batch_id": CLOSED_BATCH,
        "batch_label": "闭卷 CB-1",
        "origin": directory.name,
        "headline": metrics.get("headline"),
        "tags": metrics.get("tags") or [],
        "cards": metrics.get("cards") or [],
        "bars": metrics.get("bars") or [],
        "legacy_relation": None,
        "outcomes": metrics.get("outcomes"),
        "scale_denominator": metrics.get("scale_denominator"),
        "limitations": metrics.get("limitations") or [],
        "default_subset": metrics.get("default_subset") or "CB-paraphrase",
        "subsets": metrics.get("subsets") or {},
        "formal_ready": 0,
        "cases_status": "present" if (directory / "cases.jsonl").is_file() else "summary_only",
        "cases_missing_reason": None if (directory / "cases.jsonl").is_file() else "仅有汇总，案例产物缺失",
        "protocol_dir": directory.name,
    }


def comparison_view(root: str | Path, *, task_mode: str, batch: str) -> dict[str, Any]:
    try:
        mode = canonical_task_mode(task_mode)
    except TaskModeError as exc:
        return {"availability": "error", "status": "error", "error": {"code": "unknown_task_mode", "message": str(exc)}}
    directory = Path(root)
    if mode == CLOSED_BOOK:
        payload = _closed_comparison(directory)
    else:
        chosen = batch if batch in {HISTORICAL_BATCH, PILOT_BATCH} else HISTORICAL_BATCH
        payload = _rag_comparison(directory, chosen)
    payload["batches"] = [item for item in list_batches(directory) if item.get("task_mode") == mode]
    payload["active_batch"] = payload.get("batch_id")
    payload["active_mode"] = mode
    return payload


def comparison_cases(
    root: str | Path,
    *,
    task_mode: str,
    batch: str,
    relation: str = "",
    subset: str = "",
) -> dict[str, Any]:
    try:
        mode = canonical_task_mode(task_mode)
    except TaskModeError as exc:
        return {"availability": "error", "data": [], "error": {"code": "unknown_task_mode", "message": str(exc)}}
    directory = Path(root)
    if mode == CLOSED_BOOK:
        closed = _closed_dir(directory)
        rows = _read_jsonl(closed / "cases.jsonl") if closed else []
        if not rows:
            view = _closed_comparison(directory)
            return {
                "availability": view.get("cases_status") or "empty",
                "missing_reason": view.get("cases_missing_reason") or "尚无闭卷案例",
                "data": [],
                "task_mode": mode,
                "batch_id": CLOSED_BATCH,
            }
        selected = rows
        if subset:
            selected = [row for row in selected if row.get("eval_subset") == subset]
        if relation:
            selected = [row for row in selected if row.get("relation") == relation or relation in (row.get("labels") or [])]
        return {
            "availability": "present",
            "missing_reason": None,
            "data": selected,
            "task_mode": mode,
            "batch_id": CLOSED_BATCH,
            "total": len(selected),
        }
    view = _rag_comparison(directory, batch if batch in {HISTORICAL_BATCH, PILOT_BATCH} else HISTORICAL_BATCH)
    return {
        "availability": "summary_only",
        "missing_reason": "仅有汇总，案例产物缺失",
        "data": [],
        "identifiers_only": view.get("disagreement_ids") or [],
        "identifier_note": "这些标识来自规则分歧汇总，没有题目和回答原文。",
        "task_mode": mode,
        "batch_id": view.get("batch_id"),
        "filter": relation or None,
        "matched_filter_count": view.get("legacy_relation", {}).get(relation) if relation else None,
    }
