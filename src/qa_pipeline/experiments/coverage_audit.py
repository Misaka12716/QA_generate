"""只读复算题型、长度、身份和停止证据。不改历史文件。"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..qtypes import Q_TYPES
from ..schemas import Q_TYPES as SCHEMA_TYPES

STAGES = ("generated", "qualified", "selected", "exported", "consumed")
CHAR_UNIT = "python_len_including_whitespace_and_punctuation"
QFAM_PREFIX = "qfam"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return str(row.get("answer") or row.get("text") or "")


def _user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(row.get("question") or "")


def _question_text(user: str) -> str:
    if "问题：" in user:
        return user.split("问题：", 1)[-1].strip()
    return user.strip()


def char_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "mean": None, "unit": CHAR_UNIT}
    ordered = sorted(lengths)

    def pick(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return ordered[index]

    middle = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    return {
        "n": len(ordered),
        "p10": pick(0.10),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "median_pair_average": middle,
        "mean": round(sum(ordered) / len(ordered), 2),
        "unit": CHAR_UNIT,
        "percentile_method": "nearest_rank_round_half_even",
    }


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def normalize_record(row: dict[str, Any], *, stage: str, label_index: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """把训练行、问答或预测收成同一审计记录。不改入参。"""
    meta = _meta(row)
    qa_id = str(row.get("id") or row.get("qa_id") or row.get("case_id") or "")
    parent = qa_id.split("__", 1)[0]
    backfill = (label_index or {}).get(qa_id) or (label_index or {}).get(parent) or {}
    source_family = str(meta.get("source_family_id") or row.get("source_family_id") or "")
    sample_family = str(meta.get("family_id") or row.get("family_id") or "")
    fallback = False
    if source_family.startswith(QFAM_PREFIX):
        fallback = True
        sample_family = sample_family or source_family
        source_family = ""
    stored_type = str(meta.get("q_type") or row.get("q_type") or "")
    origin = "source" if stored_type in SCHEMA_TYPES else ""
    if not stored_type and backfill.get("q_type"):
        stored_type = str(backfill["q_type"])
        origin = "backfilled"
    if stored_type not in SCHEMA_TYPES:
        stored_type = "unknown"
        origin = origin or "unknown"
    answer = _assistant_text(row)
    question = _question_text(_user_text(row))
    points = meta.get("answer_points") or row.get("answer_points") or backfill.get("answer_points") or []
    if isinstance(points, str):
        points = [points]
    quotes = meta.get("evidence_quotes") or row.get("evidence_quotes") or []
    contract = meta.get("response_contract") or row.get("response_contract") or {}
    if not isinstance(contract, dict):
        contract = {}
    identity = meta.get("document_identity") or row.get("document_identity") or {}
    if not isinstance(identity, dict):
        identity = {}
    selected = row.get("selection_status") or ""
    if selected == "" and row.get("included_in_this_run") is False:
        selected = "not_selected"
    return {
        "stage": stage,
        "qa_id": qa_id,
        "parent_qa_id": parent,
        "knowledge_id": str(meta.get("knowledge_id") or row.get("knowledge_id") or ""),
        "source_family_id": source_family,
        "sample_family_id": sample_family,
        "source_family_fallback": fallback,
        "q_type": stored_type,
        "type_label_origin": str(meta.get("type_label_origin") or row.get("type_label_origin") or origin),
        "intent_primary": str(meta.get("intent_primary") or row.get("intent_primary") or backfill.get("intent_primary") or ""),
        "expected_action": str(meta.get("expected_action") or row.get("expected_action") or "answer"),
        "task_mode": str(meta.get("task_mode") or row.get("task_mode") or row.get("goal") or ""),
        "response_style": str((contract or {}).get("response_style") or meta.get("response_style") or ""),
        "question": question,
        "answer": answer,
        "char_len": len(answer),
        "point_count": len([item for item in points if str(item).strip()]),
        "evidence_count": len(quotes) if isinstance(quotes, list) and quotes else (1 if meta.get("evidence") or row.get("evidence_span") else 0),
        "condition_count": len(contract.get("critical_conditions") or []),
        "exception_count": len(contract.get("exceptions") or []),
        "step_count": len(contract.get("required_steps") or []),
        "comparison_dimension_count": len(contract.get("comparison_dimensions") or []),
        "source_doc": str(meta.get("source") or row.get("source_doc") or ""),
        "document_title": str(identity.get("document_title") or ""),
        "canonical_subject": str(identity.get("canonical_subject") or ""),
        "selection_status": selected,
        "exclude_reason": str(row.get("exclude_reason") or ""),
        "action": str(row.get("action") or ""),
        "grade": row.get("grade"),
        "deictic": bool(any(token in question for token in ("资料指出", "根据资料", "该药", "本品", "该药物"))),
    }


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key) or "")] += 1
    return dict(sorted(counts.items()))


def _cross(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"{row.get('q_type')}|{row.get('intent_primary')}|{row.get('expected_action')}|{row.get('task_mode')}"] += 1
    return dict(sorted(counts.items()))


def summarize_stage(rows: list[dict[str, Any]], *, available: bool, reason: str = "") -> dict[str, Any]:
    if not available:
        return {"available": False, "reason": reason or "missing", "rows": 0}
    families = {row["source_family_id"] for row in rows if row.get("source_family_id")}
    samples = {row["sample_family_id"] for row in rows if row.get("sample_family_id")}
    knowledge = {row["knowledge_id"] for row in rows if row.get("knowledge_id")}
    ids = {row["qa_id"] for row in rows if row.get("qa_id")}
    by_source: Counter[str] = Counter(row.get("source_doc") or "" for row in rows)
    by_knowledge: Counter[str] = Counter(row.get("knowledge_id") or row.get("qa_id") or "" for row in rows)
    return {
        "available": True,
        "reason": "",
        "rows": len(rows),
        "unique_qa_id": len(ids),
        "unique_knowledge_id": len(knowledge),
        "source_family_count": len(families),
        "sample_family_count": len(samples),
        "q_type": _distribution(rows, "q_type"),
        "intent_primary": _distribution(rows, "intent_primary"),
        "expected_action": _distribution(rows, "expected_action"),
        "task_mode": _distribution(rows, "task_mode"),
        "cross": _cross(rows),
        "chars": char_summary([int(row.get("char_len") or 0) for row in rows]),
        "token_stats": {"status": "not_executed", "reason": "tokenizer_not_supplied"},
        "mean_points": round(statistics.fmean([row.get("point_count") or 0 for row in rows]), 2) if rows else None,
        "mean_evidence": round(statistics.fmean([row.get("evidence_count") or 0 for row in rows]), 2) if rows else None,
        "empty_source_family": sum(1 for row in rows if not row.get("source_family_id")),
        "source_family_fallback": sum(1 for row in rows if row.get("source_family_fallback")),
        "unknown_q_type": sum(1 for row in rows if row.get("q_type") == "unknown"),
        "deictic_questions": sum(1 for row in rows if row.get("deictic")),
        "not_selected": sum(1 for row in rows if row.get("selection_status") == "not_selected"),
        "unresolved_subject": sum(1 for row in rows if not row.get("canonical_subject")),
        "top_source_doc": by_source.most_common(5),
        "top_knowledge": by_knowledge.most_common(5),
    }


def _label_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = _meta(row)
        qa_id = str(row.get("id") or row.get("qa_id") or "")
        q_type = str(meta.get("q_type") or row.get("q_type") or "")
        if qa_id and q_type in SCHEMA_TYPES:
            index[qa_id] = {
                "q_type": q_type,
                "intent_primary": meta.get("intent_primary") or row.get("intent_primary") or "",
                "answer_points": meta.get("answer_points") or row.get("answer_points") or [],
            }
    return index


def audit_predictions(path: Path, *, max_new_tokens: int | None) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "missing"}
    lengths = [len(str(row.get("text") or "")) for row in rows]
    reasons = Counter(str(row.get("finish_reason") or "") for row in rows)
    truncated = sum(1 for row in rows if row.get("truncated") is True)
    touched = 0
    if max_new_tokens:
        touched = sum(1 for row in rows if int(row.get("completion_tokens") or 0) >= max_new_tokens)
    tokens = [int(row.get("completion_tokens") or 0) for row in rows]
    return {
        "available": True,
        "path": str(path),
        "rows": len(rows),
        "chars": char_summary(lengths),
        "finish_reason_stored": dict(reasons),
        "finish_reason_evidence": "inferred_from_length",
        "last_token": "not_recorded",
        "truncated_flag": truncated,
        "hit_max_new_tokens": touched,
        "max_new_tokens": max_new_tokens,
        "completion_tokens": {
            "median": statistics.median(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
        "note": "历史预测没有末尾 token。stop/length 只说明是否达到当时的生成上限，不能证明 EOS。",
    }


def _subset_counts(closed_dir: Path) -> dict[str, int]:
    splits = closed_dir / "splits"
    counts = {}
    if not splits.is_dir():
        return counts
    for path in sorted(splits.glob("*.jsonl")):
        counts[path.stem] = len(_load_jsonl(path))
    return counts


def _object_review(consumed: list[dict[str, Any]]) -> dict[str, Any]:
    target = "qa_8deb52518a53"
    hit = next((row for row in consumed if row.get("qa_id") == target or row.get("parent_qa_id") == target), None)
    source = str((hit or {}).get("source_doc") or "")
    related = [row["qa_id"] for row in consumed if source and row.get("source_doc") == source]
    return {
        "qa_id": target,
        "found": hit is not None,
        "source_doc": source,
        "same_source_doc_ids": related,
        "note": "对象错配的判定在实体过滤器回归测试中固定。这里只列出历史训练行里的传播范围，不改等级。",
    }


def _migration_rows(consumed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    migrated = []
    for row in consumed:
        migrated.append(
            {
                "qa_id": row.get("qa_id"),
                "sample_family_id": row.get("sample_family_id"),
                "source_family_id": row.get("source_family_id") or "",
                "migration_status": "missing_source_family" if not row.get("source_family_id") else "present",
                "rewritten_in_place": False,
                "note": "样本族不能填到来源族。本文件是新版本审计，不是原训练文件。",
            }
        )
    return migrated


def collect_stage_files(run_dir: Path | None) -> dict[str, list[Path]]:
    if run_dir is None or not run_dir.is_dir():
        return {stage: [] for stage in STAGES}
    mapping = {
        "generated": ["questions.jsonl", "qa.raw.jsonl"],
        "qualified": ["qa.kept.jsonl"],
        "selected": ["qa.selected.jsonl"],
        "exported": ["zhixun.jsonl"],
        "consumed": ["sft/train.jsonl", "data/train.jsonl"],
    }
    found: dict[str, list[Path]] = {}
    for stage, names in mapping.items():
        found[stage] = [run_dir / name for name in names if (run_dir / name).is_file()]
    return found


def load_coverage_view(run_dir: str | Path, *, stage: str = "consumed", limit: int = 40) -> dict[str, Any]:
    """结果页读取审计产物。缺文件时保持未执行，不把空缺当成 0 分。"""
    root = Path(run_dir)
    report_path = root / "audit_report.json"
    if not report_path.is_file():
        report_path = root / "audit_baseline" / "audit_report.json"
    if not report_path.is_file():
        return {
            "status": "not_executed",
            "reason": "missing_audit_report",
            "stages": {},
            "samples": [],
            "note": "没有审计产物。缺字段不是 0 分。",
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stage_name = stage if stage in STAGES else "consumed"
    rows_path = report_path.parent / "rows" / f"{stage_name}.jsonl"
    samples = _load_jsonl(rows_path)[:limit] if rows_path.is_file() else []
    return {
        "status": "present",
        "stage": stage_name,
        "stages": list(STAGES),
        "summary": (report.get("stages") or {}).get(stage_name) or {},
        "closed_book_consumed": report.get("closed_book_consumed"),
        "predictions": report.get("predictions"),
        "differences": report.get("differences") or [],
        "samples": samples,
        "judge_notice": "闭卷证据仅供审核，学生未看到。",
        "missing_is_not_zero": True,
    }


def write_coverage_audit(
    out_dir: str | Path,
    *,
    train_path: str | Path | None = None,
    closed_book_dir: str | Path | None = None,
    candidate_run: str | Path | None = None,
    question_paths: list[str | Path] | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train = Path(train_path) if train_path else Path("runs/drug_v22/E3_g0/sft/train.jsonl")
    closed = Path(closed_book_dir) if closed_book_dir else Path("runs/cb1_20261007")
    gaps: list[dict[str, str]] = []

    question_rows: list[dict[str, Any]] = []
    for item in question_paths or []:
        path = Path(item)
        if path.is_file():
            question_rows.extend(_load_jsonl(path))
        else:
            gaps.append({"path": str(path), "reason": "missing_questions"})
    label_index = _label_index(question_rows)
    train_rows = _load_jsonl(train)
    if train.is_file():
        label_index.update(_label_index(train_rows))
    else:
        gaps.append({"path": str(train), "reason": "missing_historical_train"})

    consumed_raw = train_rows
    consumed = [normalize_record(row, stage="consumed", label_index=label_index) for row in consumed_raw]
    cb_train_path = closed / "data" / "train.jsonl"
    cb_rows_raw = _load_jsonl(cb_train_path)
    if not cb_train_path.is_file():
        gaps.append({"path": str(cb_train_path), "reason": "missing_closed_book_train"})
    cb_rows = [normalize_record(row, stage="consumed", label_index=label_index) for row in cb_rows_raw]

    protocol = {}
    protocol_path = closed / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    else:
        gaps.append({"path": str(protocol_path), "reason": "missing_protocol"})
    max_new = None
    infer = protocol.get("infer") if isinstance(protocol, dict) else None
    if isinstance(infer, dict) and infer.get("max_new_tokens"):
        max_new = int(infer["max_new_tokens"])

    candidate = Path(candidate_run) if candidate_run else None
    stage_files = collect_stage_files(candidate)
    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    stage_rows["consumed"] = consumed
    if candidate and not candidate.is_dir():
        gaps.append({"path": str(candidate), "reason": "missing_candidate_run"})
    for stage, paths in stage_files.items():
        if stage == "consumed" and consumed:
            continue
        loaded = []
        for path in paths:
            loaded.extend(_load_jsonl(path))
        stage_rows[stage] = [normalize_record(row, stage=stage, label_index=label_index) for row in loaded]

    stages = {}
    for stage in STAGES:
        rows = stage_rows[stage]
        if stage == "consumed":
            stages[stage] = summarize_stage(rows, available=train.is_file(), reason="" if train.is_file() else "missing_historical_train")
        elif candidate is None:
            stages[stage] = summarize_stage([], available=False, reason="candidate_run_not_requested")
        elif not stage_files.get(stage):
            stages[stage] = summarize_stage([], available=False, reason="stage_file_missing")
        else:
            stages[stage] = summarize_stage(rows, available=True)
    if tokenizer_path and Path(tokenizer_path).exists():
        for summary in stages.values():
            if summary.get("available"):
                summary["token_stats"] = {"status": "not_executed", "reason": "tokenizer_present_but_student_encoding_not_run_in_audit"}
    else:
        for summary in stages.values():
            if summary.get("available"):
                summary["token_stats"] = {"status": "not_executed", "reason": "tokenizer_not_local"}

    predictions = {
        "base": audit_predictions(closed / "predictions" / "base.jsonl", max_new_tokens=max_new),
        "adapter": audit_predictions(closed / "predictions" / "adapter.jsonl", max_new_tokens=max_new),
        "saved_inputs": _subset_counts(closed),
        "main_eval_note": "主评测是 CB-paraphrase 与 CB-retention 之和。CB-memory 只作记忆诊断，不计入主评测分母。",
    }
    expected = {
        "historical_train_rows": 74,
        "closed_book_train_rows": 22,
        "saved_prediction_rows_each": 50,
        "main_eval_inputs": 40,
    }
    observed = {
        "historical_train_rows": len(consumed) if train.is_file() else None,
        "closed_book_train_rows": len(cb_rows) if cb_train_path.is_file() else None,
        "base_prediction_rows": predictions["base"].get("rows"),
        "adapter_prediction_rows": predictions["adapter"].get("rows"),
        "cb_paraphrase": predictions["saved_inputs"].get("cb_paraphrase"),
        "cb_retention": predictions["saved_inputs"].get("cb_retention"),
        "cb_memory": predictions["saved_inputs"].get("cb_memory"),
    }
    differences = []
    for key, want in (
        ("historical_train_rows", expected["historical_train_rows"]),
        ("closed_book_train_rows", expected["closed_book_train_rows"]),
        ("base_prediction_rows", expected["saved_prediction_rows_each"]),
        ("adapter_prediction_rows", expected["saved_prediction_rows_each"]),
    ):
        got = observed.get(key)
        if got is None:
            differences.append({"metric": key, "expected": want, "observed": None, "source": "file_missing"})
        elif got != want:
            differences.append({"metric": key, "expected": want, "observed": got, "source": "recount"})
    main = None
    if observed.get("cb_paraphrase") is not None and observed.get("cb_retention") is not None:
        main = int(observed["cb_paraphrase"]) + int(observed["cb_retention"])
        if main != expected["main_eval_inputs"]:
            differences.append({"metric": "main_eval_inputs", "expected": 40, "observed": main, "source": "cb_paraphrase+cb_retention"})

    qtype_hist = stages["consumed"].get("q_type") or {}
    report = {
        "status": "audited",
        "rewrote_historical_files": False,
        "char_unit": CHAR_UNIT,
        "expected_reference": expected,
        "observed": observed,
        "main_eval_inputs": main,
        "differences": differences,
        "historical_q_type": {name: qtype_hist.get(name, 0) for name in (*Q_TYPES, "unknown")},
        "stages": stages,
        "closed_book_consumed": summarize_stage(cb_rows, available=cb_train_path.is_file(), reason="" if cb_train_path.is_file() else "missing"),
        "predictions": predictions,
        "object_review": _object_review(consumed),
        "gaps": gaps,
        "quota_note": "本审计不把 qa.kept 行数或 filter in/out 当成配额达成。",
    }
    (dest / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(dest / "rows" / "consumed.jsonl", consumed)
    _write_jsonl(dest / "rows" / "closed_book_consumed.jsonl", cb_rows)
    for stage, rows in stage_rows.items():
        if stage == "consumed":
            continue
        if rows:
            _write_jsonl(dest / "rows" / f"{stage}.jsonl", rows)
    migration = _migration_rows(consumed)
    _write_jsonl(dest / "migration" / "source_family_v2.jsonl", migration)
    (dest / "migration" / "note.json").write_text(
        json.dumps(
            {
                "in_place": False,
                "rows": len(migration),
                "missing_source_family": sum(1 for row in migration if row["migration_status"] == "missing_source_family"),
                "rule": "source_family_id 为空就保持为空，不用 family_id 或 qfam 填补。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
