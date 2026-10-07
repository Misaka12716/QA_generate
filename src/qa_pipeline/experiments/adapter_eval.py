"""评测已经训好的 adapter。不导出训练集，也不调用 train_lora。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .devices import select_trainable_gpus
from .scoring import (
    SCORER_V1,
    SCORER_V2,
    SCORER_V3,
    aggregate_layered,
    align_prediction_groups,
    cache_signature,
    message_digest,
    prediction_item_signature,
    score_task,
    score_task_v2,
    score_task_v3,
)
from ..task_mode import TaskModeError, declared_task_mode, messages_conflict
from .sft import eval_messages

EXPLORATORY_DISCLAIMER = "未完成人工审核，仅供探索"
PROTECTED_PROTOCOL_NAMES = frozenset({"heldout_protocol.jsonl", "heldout.jsonl"})
REVIEWED_STATUS = frozenset({"agreed", "dual_agreed"})
KNOWN_ROLES = frozenset({"base", "adapter"})
SUPPORTED_INFER = frozenset({"max_new_tokens", "do_sample", "temperature", "top_p", "dtype"})
DEFAULT_INFER = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "dtype": "bfloat16",
}
CARRY_FIELDS = (
    "family_id",
    "source_family_id",
    "source_hash",
    "dataset_version",
    "split",
    "task_mode",
    "evidence_state",
    "expected_action",
    "stratum",
    "source",
    "generic_family_id",
)
CONTENT_FIELDS = (
    "signature",
    "text",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "truncated",
    "device",
    "elapsed_sec",
    "error",
    "model_id",
    "adapter_id",
    "model_role",
    "template_id",
    "infer_config",
)


class ProtocolError(Exception):
    """协议不合法。应在加载模型之前抛出。"""

    def __init__(self, message: str, *, sha256: str | None = None, universe_sha256: str | None = None):
        super().__init__(message)
        self.sha256 = sha256
        self.universe_sha256 = universe_sha256


def case_id_of(case: dict[str, Any]) -> str:
    return str(case.get("case_id") or case.get("id") or "")


def case_question(case: dict[str, Any]) -> str:
    if str(case.get("question") or "").strip():
        return str(case["question"]).strip()
    for message in case.get("messages") or []:
        if message.get("role") == "user" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def canonical_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """已有 messages 不能绕过声明模式。冲突时拒绝，不静默改成另一种任务。"""
    mode = declared_task_mode(case)
    explicit = case.get("messages")
    if explicit:
        normalized = [{"role": str(item.get("role") or ""), "content": str(item.get("content") or "")} for item in explicit]
        if mode:
            reason = messages_conflict(normalized, mode, case)
            if reason:
                raise TaskModeError(reason)
        return normalized
    return eval_messages(case)


def review_content_hash(case: dict[str, Any]) -> str:
    payload = {
        "case_id": case_id_of(case),
        "question": case.get("question"),
        "context": case.get("context"),
        "messages": canonical_messages(case),
        "answer": case.get("answer"),
        "answer_points": case.get("answer_points"),
        "required_points": case.get("required_points") or [],
        "optional_points": case.get("optional_points") or [],
        "unavailable_points": case.get("unavailable_points") or [],
        "evidence": case.get("evidence"),
        "numeric_bindings": case.get("numeric_bindings") or [],
        "required_conditions": case.get("required_conditions") or [],
        "expected_action": case.get("expected_action"),
        "evidence_state": case.get("evidence_state"),
        "source_family_id": case.get("source_family_id"),
        "family_id": case.get("family_id"),
        "dataset_version": case.get("dataset_version"),
        "source_hash": case.get("source_hash"),
    }
    return cache_signature(payload)


def _review_reasons(case: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(case.get("review_source") or "") == "teacher":
        return ["teacher_review_not_formal"]
    status = case.get("review_status")
    if status not in REVIEWED_STATUS:
        reasons.append("unreviewed")
        return reasons
    digest = str(case.get("review_content_hash") or "")
    if not digest:
        reasons.append("missing_review_hash")
    elif digest != review_content_hash(case):
        reasons.append("review_hash_mismatch")
    if status == "dual_agreed":
        required = ("reviewer_a", "reviewer_b", "opinion_a", "opinion_b", "adjudication")
        if not all(str(case.get(key) or "").strip() for key in required):
            reasons.append("incomplete_dual_review")
    elif not str(case.get("reviewer_a") or "").strip() or not str(case.get("opinion_a") or "").strip():
        reasons.append("incomplete_review")
    return reasons


def screen_candidates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """候选筛选只用于发布前。已发布协议不得再靠它缩小分母。"""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            excluded.append({"id": index, "reasons": ["invalid_object"]})
            continue
        reasons: list[str] = []
        case_id = case_id_of(case)
        if not case_id:
            reasons.append("missing_id")
        elif case_id in seen:
            reasons.append("duplicate_id")
        else:
            seen.add(case_id)
        if not case_question(case):
            reasons.append("empty_question")
        if reasons:
            excluded.append({"id": case_id or index, "reasons": reasons})
        else:
            kept.append(case)
    return {
        "kept": kept,
        "excluded": excluded,
        "note": "筛选只发生在发布前。已发布协议不再按这份清单缩小分母。",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_train_families(path: str | Path) -> dict[str, Any]:
    """读取非空、带出处的冻结训练来源清单。

    JSON 对象至少包含 source_family_ids。每条 id 还要有 record，
    并且 record 带 stem 或 path，便于对照 inventory。空清单、缺文件和
    无法追溯的裸 id 列表都拒绝，不把它们当成“没有来源重叠”。
    """
    file = Path(path)
    if not file.is_file():
        raise ProtocolError("invalid_source_list:file_missing")
    digest = _sha256_file(file)

    def fail(reason: str) -> None:
        raise ProtocolError(reason, sha256=digest)

    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_source_list:json", sha256=digest) from exc
    if not isinstance(payload, dict):
        fail("invalid_source_list:schema")
    raw_ids = payload.get("source_family_ids")
    if not isinstance(raw_ids, list):
        fail("invalid_source_list:schema")
    if len(raw_ids) == 0:
        fail("invalid_empty_source_list")
    ids: list[str] = []
    for item in raw_ids:
        text = str(item or "").strip()
        if not text:
            fail("invalid_source_list:blank_id")
        if text not in ids:
            ids.append(text)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        fail("invalid_source_list:untraceable")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            fail("invalid_source_list:schema")
        family = str(record.get("source_family_id") or "").strip()
        stem = str(record.get("stem") or "").strip()
        record_path = str(record.get("path") or "").strip()
        if not family or not (stem or record_path):
            fail("invalid_source_list:untraceable")
        by_id[family] = {"source_family_id": family, "stem": stem, "path": record_path}
    missing = [family for family in ids if family not in by_id]
    if missing:
        fail("invalid_source_list:untraceable")
    return {
        "ids": set(ids),
        "sha256": digest,
        "records": [by_id[family] for family in ids],
        "path": str(file),
    }


def load_source_universe(path: str | Path) -> dict[str, Any]:
    """读取来源宇宙。JSON 对象或 JSONL 均可。

    家族 id、stem、path 分开保存。只有 stem 的 inventory 不能单独证明
    协议里的 source_family_id 可追溯。
    """
    file = Path(path)
    if not file.is_file():
        raise ProtocolError("invalid_source_list:universe_missing")
    digest = _sha256_file(file)
    families: set[str] = set()
    stems: set[str] = set()
    paths: set[str] = set()
    text = file.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ProtocolError("invalid_source_list:universe_empty")
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid_source_list:json") from exc
        rows: list[Any]
        if isinstance(payload, dict):
            raw_ids = payload.get("source_family_ids") or []
            if not isinstance(raw_ids, list):
                raise ProtocolError("invalid_source_list:schema")
            families.update(str(item).strip() for item in raw_ids if str(item).strip())
            rows = list(payload.get("records") or [])
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ProtocolError("invalid_source_list:schema")
        for record in rows:
            if not isinstance(record, dict):
                raise ProtocolError("invalid_source_list:schema")
            family = str(record.get("source_family_id") or "").strip()
            stem = str(record.get("stem") or "").strip()
            record_path = str(record.get("path") or "").strip()
            if family:
                families.add(family)
            if stem:
                stems.add(stem)
            if record_path:
                paths.add(record_path)
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError("invalid_source_list:json") from exc
            if not isinstance(record, dict):
                raise ProtocolError("invalid_source_list:schema")
            family = str(record.get("source_family_id") or "").strip()
            stem = str(record.get("stem") or "").strip()
            record_path = str(record.get("path") or "").strip()
            if family:
                families.add(family)
            if stem:
                stems.add(stem)
            if record_path:
                paths.add(record_path)
    if not families and not stems and not paths:
        raise ProtocolError("invalid_source_list:universe_empty")
    return {"families": families, "stems": stems, "paths": paths, "sha256": digest, "path": str(file)}


def trace_source_list(manifest: dict[str, Any], universe: dict[str, Any]) -> dict[str, Any]:
    """训练清单里的每个家族都要能在宇宙中找到。找不到则清单无效。"""
    families = set(universe.get("families") or [])
    stems = set(universe.get("stems") or [])
    paths = set(universe.get("paths") or [])
    if not families and not stems and not paths:
        return {"ok": False, "reason": "invalid_source_list:universe_empty", "untraced": []}
    untraced = []
    for record in manifest.get("records") or []:
        family = str(record.get("source_family_id") or "")
        stem = str(record.get("stem") or "")
        record_path = str(record.get("path") or "")
        located = False
        if families and family in families:
            located = True
        if stem and stem in stems:
            located = True
        if record_path and record_path in paths:
            located = True
        if families and family not in families:
            located = False
        if not located:
            untraced.append(family)
    if untraced or not families:
        reason = "invalid_source_list:untraceable" if untraced or not families else None
        if not families:
            reason = "invalid_source_list:universe_families_missing"
        if untraced:
            reason = "invalid_source_list:untraceable"
        return {"ok": False, "reason": reason, "untraced": untraced}
    return {"ok": True, "reason": None, "untraced": []}


def refuse_source_list(
    out_dir: str | Path,
    reason: str,
    *,
    mode: str,
    protocol_path: str | Path | None = None,
    frozen_train_families_sha256: str | None = None,
    source_universe_sha256: str | None = None,
) -> dict[str, Any]:
    """来源清单无效时写报告并停止，不加载模型。"""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    planned = 0
    if protocol_path and Path(protocol_path).is_file():
        planned = len(read_jsonl(protocol_path))
    _dump_report.source_meta = source_list_report_fields(
        frozen_sha256=frozen_train_families_sha256,
        universe_sha256=source_universe_sha256,
    )
    report = _not_executed(
        reason,
        mode,
        {
            "planned_n": planned,
            "failed_n": planned,
            "scored_n": 0,
            "protocol_failures": [{"id": None, "reasons": [reason.split(":")[0]]}],
        },
    )
    _dump_report(destination, report, "source_list")
    _dump_report.source_meta = None
    return report


def source_list_report_fields(
    *,
    frozen_train_families: set[str] | frozenset[str] | None = None,
    frozen_sha256: str | None = None,
    source_universe: set[str] | frozenset[str] | None = None,
    universe_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "frozen_train_families_sha256": frozen_sha256,
        "frozen_train_families_n": None if frozen_train_families is None else len(set(frozen_train_families)),
        "source_universe_sha256": universe_sha256,
        "source_universe_n": None if source_universe is None else len(set(source_universe)),
    }


def validate_protocol(
    cases: Any,
    mode: str,
    frozen_train_families: set[str] | frozenset[str] | None = None,
    source_universe: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"formal", "exploratory"}:
        raise ProtocolError(f"unknown_mode:{mode}")
    if not isinstance(cases, list):
        return {
            "mode": mode,
            "status": "invalid_structure",
            "eligible": [],
            "failures": [{"id": None, "reasons": ["invalid_object"]}],
            "planned_n": 0,
            "executable": False,
        }
    if len(cases) == 0:
        return {
            "mode": mode,
            "status": "empty_protocol",
            "eligible": [],
            "failures": [],
            "planned_n": 0,
            "executable": False,
        }
    failures = []
    seen: set[str] = set()
    review_only = True
    for index, case in enumerate(cases):
        reasons: list[str] = []
        if not isinstance(case, dict):
            failures.append({"id": index, "reasons": ["invalid_object"]})
            review_only = False
            continue
        case_id = case_id_of(case)
        if not case_id:
            reasons.append("missing_id")
        elif case_id in seen:
            reasons.append("duplicate_id")
        else:
            seen.add(case_id)
        if not case_question(case):
            reasons.append("empty_question")
        if mode == "formal":
            train_ids = None if frozen_train_families is None else set(frozen_train_families)
            universe_ids = None if source_universe is None else set(source_universe)
            list_reasons: list[str] = []
            if train_ids is None:
                list_reasons.append("source_list_missing")
            elif len(train_ids) == 0:
                list_reasons.append("invalid_empty_source_list")
            elif universe_ids is not None and len(universe_ids) == 0:
                list_reasons.append("invalid_source_list")
            elif universe_ids is not None and not train_ids <= universe_ids:
                list_reasons.append("invalid_source_list")
            reasons.extend(list_reasons)
            if not list_reasons and train_ids is not None:
                family = str(case.get("source_family_id") or "")
                if universe_ids is not None and family and family not in universe_ids:
                    reasons.append("source_untraceable")
                elif family and family in train_ids:
                    reasons.append("source_overlap")
                elif case.get("source_overlap_train"):
                    reasons.append("source_overlap_unverified")
            if not str(case.get("answer") or "").strip() and not (case.get("answer_points") or case.get("required_points")):
                reasons.append("missing_gold")
            reasons.extend(_review_reasons(case))
            if not (case.get("source_family_id") or case.get("source")):
                reasons.append("missing_source")
            if not case.get("expected_action"):
                reasons.append("missing_expected_action")
        if reasons:
            if any(reason not in {"unreviewed", "incomplete_dual_review", "incomplete_review", "missing_review_hash"} for reason in reasons):
                review_only = False
            failures.append({"id": case_id or index, "reasons": reasons})
    if mode == "formal" and failures:
        status = "pending_review" if review_only else "batch_stopped"
        return {
            "mode": mode,
            "status": status,
            "eligible": [],
            "failures": failures,
            "planned_n": len(cases),
            "executable": False,
        }
    if mode == "exploratory" and failures:
        return {
            "mode": mode,
            "status": "invalid_protocol",
            "eligible": [],
            "failures": failures,
            "planned_n": len(cases),
            "executable": False,
        }
    return {
        "mode": mode,
        "status": "executable",
        "eligible": list(cases),
        "failures": [],
        "planned_n": len(cases),
        "executable": True,
    }


def assert_candidate_destination(path: str | Path, protected: list[str | Path] | None = None) -> None:
    dest = Path(path)
    if dest.name in PROTECTED_PROTOCOL_NAMES:
        raise ProtocolError("refuse_overwrite_formal_protocol")
    resolved = dest.resolve() if dest.exists() else dest.absolute()
    for item in protected or []:
        guard = Path(item)
        if guard.exists() and guard.resolve() == resolved:
            raise ProtocolError("refuse_overwrite_formal_protocol")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    file = Path(path)
    if not file.is_file():
        return rows
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _carry(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case.get(key) for key in CARRY_FIELDS if case.get(key) not in {None, ""}}


def resolve_infer(infer_config: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    supplied = dict(infer_config or {})
    unknown = sorted(set(supplied) - SUPPORTED_INFER)
    if unknown:
        return None, unknown
    declared = dict(DEFAULT_INFER)
    declared.update(supplied)
    if declared.get("dtype") != "bfloat16":
        return None, ["dtype"]
    effective = {
        "max_new_tokens": int(declared["max_new_tokens"]),
        "do_sample": bool(declared["do_sample"]),
        "dtype": "bfloat16",
    }
    inactive: list[str] = []
    if effective["do_sample"]:
        effective["temperature"] = declared["temperature"]
        effective["top_p"] = declared["top_p"]
    else:
        inactive = ["temperature", "top_p"]
    return {"declared": declared, "effective": effective, "inactive_when_greedy": inactive}, []


def prediction_run_id(
    cases: list[dict[str, Any]],
    base_id: str,
    adapter_id: str,
    template_id: str,
    infer_effective: dict[str, Any],
) -> str:
    payload = {
        "base_id": base_id,
        "adapter_id": adapter_id,
        "template_id": template_id,
        "infer_config": infer_effective,
        "inputs": [{"case_id": case_id_of(case), "messages": canonical_messages(case)} for case in cases],
    }
    return cache_signature(payload)[:16]


def score_run_id(pred_id: str, scorer_version: str, cases: list[dict[str, Any]]) -> str:
    gold = []
    for case in cases:
        gold.append(
            {
                "case_id": case_id_of(case),
                "answer": case.get("answer"),
                "answer_points": case.get("answer_points"),
                "required_points": case.get("required_points"),
                "unavailable_points": case.get("unavailable_points"),
                "expected_action": case.get("expected_action"),
                "numeric_bindings": case.get("numeric_bindings"),
                "required_conditions": case.get("required_conditions"),
                "evidence_state": case.get("evidence_state"),
            }
        )
    return cache_signature({"prediction_run_id": pred_id, "scorer": scorer_version, "gold": gold})[:16]


def _archive_if_different(path: Path, run_id: str) -> None:
    """同一路径上的另一运行先改名保留，不覆盖旧结果。"""
    meta_path = Path(str(path) + ".run.json")
    if not path.exists():
        return
    old_id = None
    if meta_path.is_file():
        try:
            old_id = str(json.loads(meta_path.read_text(encoding="utf-8")).get("run_id") or "")
        except json.JSONDecodeError:
            old_id = ""
    if old_id == run_id:
        return
    stamp = old_id or "legacy"
    archived = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    if archived.exists():
        raise ProtocolError(f"refuse_overwrite:{archived.name}")
    path.replace(archived)
    if meta_path.is_file():
        meta_path.replace(Path(str(archived) + ".run.json"))


def _write_versioned_json(path: Path, run_id: str, payload: dict[str, Any]) -> None:
    _archive_if_different(path, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(path) + ".run.json").write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")


def _write_versioned_jsonl(path: Path, run_id: str, rows: list[dict[str, Any]]) -> None:
    _archive_if_different(path, run_id)
    write_jsonl(path, rows)
    Path(str(path) + ".run.json").write_text(json.dumps({"run_id": run_id}, ensure_ascii=False), encoding="utf-8")


def _load_content_cache(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    found: dict[str, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []
    blocked: set[str] = set()
    if not path.is_file():
        return found, problems
    for index, row in enumerate(read_jsonl(path)):
        if not isinstance(row, dict):
            problems.append({"index": index, "reason": "corrupt"})
            continue
        signature = str(row.get("signature") or "")
        if not signature:
            problems.append({"index": index, "reason": "missing_signature"})
            continue
        if row.get("case_id"):
            problems.append({"index": index, "reason": "content_cache_has_case_id", "signature": signature})
        if row.get("error"):
            problems.append({"index": index, "reason": "failure_in_success_cache", "signature": signature})
            continue
        if signature in blocked:
            continue
        if signature in found and found[signature].get("text") != row.get("text"):
            problems.append({"signature": signature, "reason": "conflict"})
            found.pop(signature, None)
            blocked.add(signature)
            continue
        if signature in found:
            problems.append({"signature": signature, "reason": "duplicate"})
            continue
        found[signature] = {key: row.get(key) for key in CONTENT_FIELDS}
    return found, problems


def _content_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in CONTENT_FIELDS}


def _bind_content(content: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    bound = {
        "case_id": spec["case_id"],
        "model_role": spec["model_role"],
        "messages": spec["messages"],
        "signature": content.get("signature") or spec["signature"],
        "model_id": content.get("model_id") or spec.get("model_id"),
        "adapter_id": content.get("adapter_id") if spec["model_role"] == "adapter" else "",
        "template_id": content.get("template_id") or spec.get("template_id"),
        "infer_config": content.get("infer_config") or spec.get("infer_config"),
        "text": str(content.get("text") or ""),
        "prompt_tokens": content.get("prompt_tokens"),
        "completion_tokens": content.get("completion_tokens"),
        "finish_reason": content.get("finish_reason"),
        "truncated": bool(content.get("truncated")),
        "device": content.get("device"),
        "elapsed_sec": content.get("elapsed_sec"),
        "error": content.get("error"),
    }
    for key, value in spec.items():
        if key in CARRY_FIELDS and value not in {None, ""}:
            bound[key] = value
    return bound


def _score_one(prediction: str, case: dict[str, Any], scorer_version: str) -> dict[str, Any]:
    if scorer_version == SCORER_V1:
        scored = score_task(prediction, case)
        scored["scorer_version"] = SCORER_V1
        scored["semantic_accuracy_verified"] = False
        scored["metric_role"] = "auxiliary"
        return scored
    if scorer_version == SCORER_V2:
        scored = score_task_v2(prediction, case)
        scored["metric_role"] = "auxiliary"
        return scored
    if scorer_version == SCORER_V3:
        return score_task_v3(prediction, case)
    raise ProtocolError(f"unknown_scorer:{scorer_version}")


def _not_executed(reason: str, mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {
        "status": "not_executed",
        "executed": False,
        "reason": reason,
        "mode": mode,
        "result_class": "not_executed",
        "enters_formal_metric": False,
        "formal_main_metric": "not_executed",
        "formal_gold": False,
        "semantic_accuracy_verified": False,
        "metric_role": "auxiliary",
        "reason_main_metric": "uncalibrated_rules" if reason != "protocol_missing" else reason,
        "delta_f1": None,
        "planned_n": 0,
        "scored_n": 0,
        "failed_n": 0,
        "count_units": {
            "planned_n": "questions",
            "generated_n": "model_answers",
            "scored_n": "questions",
            "failed_n": "questions",
        },
    }
    if mode == "exploratory":
        report["disclaimer"] = EXPLORATORY_DISCLAIMER
    if extra:
        report.update(extra)
    return report


def _apply_scores(
    alignment: dict[str, Any],
    scorer_version: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for item in alignment["aligned"]:
        case = item["case"]
        base = item["predictions"]["base"]
        tuned = item["predictions"]["adapter"]
        before = _score_one(str(base.get("text") or ""), case, scorer_version)
        after = _score_one(str(tuned.get("text") or ""), case, scorer_version)
        row = {
            "case_id": case_id_of(case),
            **_carry(case),
            "before": before,
            "after": after,
            "content_error": before.get("passed") is False or after.get("passed") is False,
            "unresolved": before.get("passed") is None or after.get("passed") is None,
            "technical_failure": False,
            "metric_role": "auxiliary",
        }
        if mode == "exploratory":
            row["disclaimer"] = EXPLORATORY_DISCLAIMER
            row["review_status"] = case.get("review_status") or "unreviewed"
            row["enters_formal_metric"] = False
        else:
            row["review_status"] = case.get("review_status")
            row["enters_formal_metric"] = False
        rows.append(row)
    return rows


def _spec_for(
    case: dict[str, Any],
    role: str,
    messages: list[dict[str, str]],
    signature: str,
    base_id: str,
    adapter_id: str,
    template_id: str,
    infer_effective: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id_of(case),
        "model_role": role,
        "messages": messages,
        "signature": signature,
        "model_id": base_id,
        "adapter_id": adapter_id if role == "adapter" else "",
        "template_id": template_id,
        "infer_config": infer_effective,
        **_carry(case),
    }


def eval_adapter(
    *,
    base_model: str,
    base_id: str,
    adapter: str,
    adapter_id: str,
    protocol_path: str | Path,
    out_dir: str | Path,
    infer_config: dict[str, Any] | None = None,
    scorer_version: str = SCORER_V2,
    mode: str = "exploratory",
    template_id: str = "unspecified-template",
    generate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
    device: int | None = None,
    reuse_predictions: bool = True,
    frozen_train_families: set[str] | frozenset[str] | None = None,
    source_universe: set[str] | frozenset[str] | None = None,
    frozen_train_families_sha256: str | None = None,
    source_universe_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """推理与评分分开。正式协议有无效项时停止，不缩小计划分母。"""
    destination = Path(out_dir)
    _dump_report.source_meta = source_list_report_fields(
        frozen_train_families=frozen_train_families,
        frozen_sha256=frozen_train_families_sha256,
        source_universe=source_universe,
        universe_sha256=source_universe_sha256,
    )
    destination.mkdir(parents=True, exist_ok=True)
    protocol_file = Path(protocol_path)
    if not protocol_file.is_file():
        report = _not_executed("protocol_missing", mode)
        _dump_report(destination, report, "missing")
        return report
    resolved, unknown = resolve_infer(infer_config)
    if resolved is None:
        report = _not_executed(f"unsupported_infer:{','.join(unknown)}", mode)
        _dump_report(destination, report, "unsupported")
        return report
    cases = read_jsonl(protocol_file)
    checked = validate_protocol(
        cases,
        mode,
        frozen_train_families=frozen_train_families,
        source_universe=source_universe,
    )
    (destination / "protocol_check.json").write_text(
        json.dumps(
            {
                "status": checked["status"],
                "failures": checked["failures"],
                "eligible_n": len(checked["eligible"]),
                "planned_n": checked["planned_n"],
                "executable": checked["executable"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not checked["executable"]:
        report = _not_executed(
            checked["status"],
            mode,
            {
                "protocol_failures": checked["failures"],
                "planned_n": checked["planned_n"],
                "scored_n": 0,
                "failed_n": checked["planned_n"],
                "main_metric": "not_executed",
            },
        )
        _dump_report(destination, report, checked["status"])
        if mode == "formal" or checked["status"] in {"invalid_protocol", "invalid_structure"}:
            raise ProtocolError(checked["status"])
        return report
    target = checked["eligible"]
    pred_id = prediction_run_id(target, base_id, adapter_id, template_id, resolved["effective"])
    gold_id = score_run_id(pred_id, scorer_version, target)
    if resume:
        existing = destination / "predictions.jsonl.run.json"
        if existing.is_file():
            previous = str(json.loads(existing.read_text(encoding="utf-8")).get("run_id") or "")
            if previous != pred_id:
                report = _not_executed("resume_mismatch", mode, {"prediction_run_id": pred_id, "found_run_id": previous})
                _dump_report(destination, report, gold_id)
                return report
    cache_path = destination / "prediction_cache.jsonl"
    cache, cache_problems = _load_content_cache(cache_path) if reuse_predictions else ({}, [])
    planned_specs: list[dict[str, Any]] = []
    pending_by_sig: dict[str, dict[str, Any]] = {}
    for case in target:
        messages = canonical_messages(case)
        for role in ("base", "adapter"):
            signature = prediction_item_signature(
                case, base_id, adapter_id, template_id, resolved["effective"], role, model_input=messages
            )
            spec = _spec_for(case, role, messages, signature, base_id, adapter_id, template_id, resolved["effective"])
            planned_specs.append(spec)
            hit = cache.get(signature)
            if (
                hit is not None
                and not hit.get("error")
                and hit.get("model_id") == base_id
                and hit.get("model_role") == role
                and (role != "adapter" or hit.get("adapter_id") == adapter_id)
            ):
                continue
            pending_by_sig.setdefault(signature, spec)
    pending = list(pending_by_sig.values())
    cache_hits = len(planned_specs) - len(pending)
    fresh: dict[str, dict[str, Any]] = {}
    if pending:
        if generate_fn is None:
            produced = generate_with_model(
                pending,
                base_model=base_model,
                adapter=adapter,
                infer_config=resolved["effective"],
                device=device,
                persist_dir=destination,
            )
        else:
            produced = generate_fn(pending)
        if isinstance(produced, dict):
            report = _not_executed(
                str(produced.get("reason") or "generation_failed"),
                mode,
                {"cache_hits": cache_hits, "cache_problems": cache_problems, "prediction_run_id": pred_id},
            )
            _dump_report(destination, report, gold_id)
            return report
        if len(produced) != len(pending):
            report = _not_executed(
                "prediction_count_mismatch",
                mode,
                {"expected": len(pending), "got": len(produced), "cache_hits": cache_hits, "prediction_run_id": pred_id},
            )
            _dump_report(destination, report, gold_id)
            return report
        for item, spec in zip(produced, pending):
            if not isinstance(item, dict) or "text" not in item:
                report = _not_executed("invalid_generation_row", mode, {"prediction_run_id": pred_id})
                _dump_report(destination, report, gold_id)
                return report
            row = {
                **spec,
                "text": str(item.get("text") or ""),
                "prompt_tokens": item.get("prompt_tokens"),
                "completion_tokens": item.get("completion_tokens"),
                "finish_reason": item.get("finish_reason"),
                "truncated": bool(item.get("truncated")),
                "device": item.get("device"),
                "elapsed_sec": item.get("elapsed_sec"),
                "error": item.get("error"),
            }
            content = _content_record(row)
            fresh[spec["signature"]] = content
            if row.get("error"):
                _append_jsonl(destination / "prediction_failures.jsonl", content)
            else:
                previous = cache.get(spec["signature"])
                if previous is not None and previous.get("text") != content.get("text"):
                    cache_problems.append({"signature": spec["signature"], "reason": "conflict"})
                else:
                    cache[spec["signature"]] = content
                    _append_jsonl(cache_path, content)
    selected = []
    for spec in planned_specs:
        content = fresh.get(spec["signature"]) or cache.get(spec["signature"])
        if content is None:
            report = _not_executed("missing_generated_content", mode, {"prediction_run_id": pred_id, "case_id": spec["case_id"]})
            _dump_report(destination, report, gold_id)
            return report
        selected.append(_bind_content(content, spec))
    _write_versioned_jsonl(destination / "predictions.jsonl", pred_id, selected)
    errored = [row for row in selected if row.get("error")]
    if errored:
        failed_questions = len({row.get("case_id") for row in errored})
        report = _not_executed(
            "generation_item_failed",
            mode,
            {
                "planned_n": len(target),
                "generated_n": {"base": sum(1 for row in selected if row.get("model_role") == "base"), "adapter": sum(1 for row in selected if row.get("model_role") == "adapter")},
                "cache_hits": cache_hits,
                "scored_n": 0,
                "failed_n": failed_questions,
                "technical_failure_n": failed_questions,
                "content_error_n": 0,
                "prediction_run_id": pred_id,
                "score_run_id": None,
                "cache_problems": cache_problems,
                "failures": [
                    {"case_id": row.get("case_id"), "model_role": row.get("model_role"), "reason": row.get("error"), "kind": "technical"}
                    for row in errored
                ],
            },
        )
        _dump_report(destination, report, gold_id)
        return report
    by_role: dict[str, list[dict[str, Any]]] = {"base": [], "adapter": []}
    for row in selected:
        role = str(row.get("model_role") or "")
        if role in by_role:
            by_role[role].append(row)
    expected = {case_id_of(case): canonical_messages(case) for case in target}
    alignment = align_prediction_groups(target, by_role, require_input_identity=True, expected_messages=expected)
    alignment_public = {key: value for key, value in alignment.items() if key != "aligned"}
    (destination / "alignment.json").write_text(json.dumps(alignment_public, ensure_ascii=False, indent=2), encoding="utf-8")
    if not alignment["ok"]:
        report = _not_executed(
            "alignment_failed",
            mode,
            {
                "planned_n": alignment["planned_n"],
                "generated_n": alignment["generated_n"],
                "scored_n": alignment["scored_n"],
                "failed_n": alignment["failed_n"],
                "failures": alignment["failures"],
                "affected_ids": alignment["affected_ids"],
                "cache_hits": cache_hits,
                "cache_problems": cache_problems,
                "task_accuracy": None,
                "technical_failure_n": alignment["failed_n"],
                "content_error_n": 0,
                "prediction_run_id": pred_id,
            },
        )
        _dump_report(destination, report, gold_id)
        return report
    return _finalize_scores(
        destination,
        alignment,
        target,
        scorer_version,
        mode,
        pred_id,
        gold_id,
        base_id,
        adapter_id,
        template_id,
        resolved,
        cache_hits,
        cache_problems,
        checked,
        model_loaded=generate_fn is None and bool(pending),
    )


def _finalize_scores(
    destination: Path,
    alignment: dict[str, Any],
    target: list[dict[str, Any]],
    scorer_version: str,
    mode: str,
    pred_id: str,
    gold_id: str,
    base_id: str,
    adapter_id: str,
    template_id: str,
    resolved: dict[str, Any],
    cache_hits: int,
    cache_problems: list[dict[str, Any]],
    checked: dict[str, Any],
    model_loaded: bool,
) -> dict[str, Any]:
    scored = _apply_scores(alignment, scorer_version, mode)
    score_path = destination / f"scores.{scorer_version}.jsonl"
    _write_versioned_jsonl(score_path, gold_id, scored)
    passed_rows = [{**row, "passed": row["after"].get("passed")} for row in scored]
    layered = aggregate_layered(passed_rows)
    exploratory = mode == "exploratory"
    report = {
        "status": "exploratory_executed" if exploratory else "auxiliary_executed",
        "executed": True,
        "reason": None,
        "mode": mode,
        "result_class": "exploratory_auxiliary" if exploratory else "formal_auxiliary",
        "disclaimer": EXPLORATORY_DISCLAIMER if exploratory else "规则分未校准，不能作为正式语义主指标。",
        "enters_formal_metric": False,
        "formal_main_metric": "not_executed",
        "formal_gold": False,
        "formal_protocol_executable": not exploratory,
        "semantic_accuracy_verified": False,
        "metric_role": "auxiliary",
        "reason_main_metric": "uncalibrated_rules",
        "scorer_version": scorer_version,
        "prediction_run_id": pred_id,
        "score_run_id": gold_id,
        "base_id": base_id,
        "adapter_id": adapter_id,
        "template_id": template_id,
        "infer_config": resolved["declared"],
        "infer_config_effective": resolved["effective"],
        "infer_inactive_when_greedy": resolved["inactive_when_greedy"],
        "planned_n": alignment["planned_n"],
        "generated_n": alignment["generated_n"],
        "scored_n": alignment["scored_n"],
        "failed_n": alignment["failed_n"],
        "count_units": alignment["count_units"],
        "cache_hits": cache_hits,
        "cache_problems": cache_problems,
        "protocol_failures": checked["failures"],
        "content_error_n": sum(1 for row in scored if row["content_error"]),
        "unresolved_n": sum(1 for row in scored if row["unresolved"]),
        "technical_failure_n": 0,
        "layered": layered,
        "combined_generalization_score": None,
        "delta_f1": None,
        "model_loaded": model_loaded,
        "judge": {"mean": None, "reason": "not_called", "note": "本轮不为填空值调用模型评委，评委结果也不是人工金标。"},
    }
    _dump_report(destination, report, gold_id)
    (destination / "run_ids.json").write_text(
        json.dumps(
            {
                "prediction_run_id": pred_id,
                "score_run_id": gold_id,
                "base_id": base_id,
                "adapter_id": adapter_id,
                "template_id": template_id,
                "scorer_version": scorer_version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _dump_report(destination: Path, report: dict[str, Any], score_id: str) -> None:
    meta = getattr(_dump_report, "source_meta", None)
    if meta:
        report.update(meta)
    _write_versioned_json(destination / "eval_report.json", score_id, report)


def _group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"base": [], "adapter": []}
    unknown: list[dict[str, Any]] = []
    for row in rows:
        role = str(row.get("model_role") or "")
        if role in groups:
            groups[role].append(row)
        else:
            unknown.append(row)
    if unknown:
        groups["unknown"] = unknown
    return groups


def rescore_saved(
    *,
    protocol_path: str | Path,
    predictions_path: str | Path,
    out_dir: str | Path,
    scorer_version: str,
    mode: str,
    frozen_train_families: set[str] | frozenset[str] | None = None,
    base_id: str = "",
    adapter_id: str = "",
    template_id: str = "",
    infer_config: dict[str, Any] | None = None,
    source_universe: set[str] | frozenset[str] | None = None,
    frozen_train_families_sha256: str | None = None,
    source_universe_sha256: str | None = None,
) -> dict[str, Any]:
    """只读取已经落盘的预测。输入 messages 不一致时拒绝复用，不加载模型。"""
    destination = Path(out_dir)
    _dump_report.source_meta = source_list_report_fields(
        frozen_train_families=frozen_train_families,
        frozen_sha256=frozen_train_families_sha256,
        source_universe=source_universe,
        universe_sha256=source_universe_sha256,
    )
    destination.mkdir(parents=True, exist_ok=True)
    protocol_file = Path(protocol_path)
    if not protocol_file.is_file():
        report = _not_executed("protocol_missing", mode)
        _dump_report(destination, report, "missing")
        return report
    cases = read_jsonl(protocol_file)
    checked = validate_protocol(
        cases,
        mode,
        frozen_train_families=frozen_train_families,
        source_universe=source_universe,
    )
    if not checked["executable"]:
        report = _not_executed(
            checked["status"],
            mode,
            {"protocol_failures": checked["failures"], "planned_n": checked["planned_n"], "scored_n": 0, "failed_n": checked["planned_n"]},
        )
        _dump_report(destination, report, checked["status"])
        if mode == "formal" or checked["status"] in {"invalid_protocol", "invalid_structure"}:
            raise ProtocolError(checked["status"])
        return report
    resolved, unknown = resolve_infer(infer_config)
    if resolved is None:
        report = _not_executed(f"unsupported_infer:{','.join(unknown)}", mode)
        _dump_report(destination, report, "unsupported")
        return report
    groups = _group_predictions(read_jsonl(predictions_path))
    expected = {case_id_of(case): canonical_messages(case) for case in checked["eligible"]}
    alignment = align_prediction_groups(
        checked["eligible"],
        groups,
        require_input_identity=True,
        expected_messages=expected,
    )
    pred_id = prediction_run_id(checked["eligible"], base_id, adapter_id, template_id, resolved["effective"]) if base_id else "saved"
    gold_id = score_run_id(pred_id, scorer_version, checked["eligible"])
    if not alignment["ok"]:
        report = _not_executed(
            "alignment_failed",
            mode,
            {
                "failures": alignment["failures"],
                "affected_ids": alignment["affected_ids"],
                "planned_n": alignment["planned_n"],
                "generated_n": alignment["generated_n"],
                "failed_n": alignment["failed_n"],
                "scored_n": 0,
                "technical_failure_n": alignment["failed_n"],
                "content_error_n": 0,
                "task_accuracy": None,
                "model_loaded": False,
                "prediction_run_id": pred_id,
            },
        )
        _dump_report(destination, report, gold_id)
        return report
    report = _finalize_scores(
        destination,
        alignment,
        checked["eligible"],
        scorer_version,
        mode,
        pred_id,
        gold_id,
        base_id,
        adapter_id,
        template_id,
        resolved,
        0,
        [],
        checked,
        model_loaded=False,
    )
    report["status"] = "rescored" if mode == "exploratory" else "auxiliary_rescored"
    report["model_loaded"] = False
    _dump_report(destination, report, gold_id)
    return report


def generate_with_model(
    pending: list[dict[str, Any]],
    *,
    base_model: str,
    adapter: str,
    infer_config: dict[str, Any],
    device: int | None = None,
    persist_dir: str | Path | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """在一张空闲 GPU 上串行跑基座和 adapter。没有足够显存时不回退到 CPU。"""
    unknown = set(infer_config) - SUPPORTED_INFER - {"inactive_when_greedy"}
    if infer_config.get("dtype") not in {None, "bfloat16"}:
        return {"skipped": True, "reason": "unsupported_infer:dtype"}
    if unknown:
        return {"skipped": True, "reason": "unsupported_infer:" + ",".join(sorted(unknown))}
    requested = None if device is None else [int(device)]
    chosen, error = select_trainable_gpus(requested)
    if not chosen:
        return {"skipped": True, "reason": error or "no_gpu"}
    index = chosen[0]
    try:
        import time

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"missing_infer_deps: {exc}"}
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "cuda_unavailable"}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": index},
    )
    model.eval()
    max_new = int(infer_config.get("max_new_tokens") or 512)
    do_sample = bool(infer_config.get("do_sample"))
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    partial = None if persist_dir is None else Path(persist_dir) / "generation_partial.jsonl"

    def _run(spec: dict[str, Any], current) -> dict[str, Any]:
        started = time.perf_counter()
        messages = spec["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(current.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            generate_kwargs["temperature"] = float(infer_config["temperature"])
            generate_kwargs["top_p"] = float(infer_config["top_p"])
        with torch.no_grad():
            generated = current.generate(**inputs, **generate_kwargs)
        new_tokens = generated[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(new_tokens.shape[0])
        truncated = completion_tokens >= max_new
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "length" if truncated else "stop",
            "truncated": truncated,
            "device": index,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }

    try:
        base_pending = [item for item in pending if item.get("model_role") == "base"]
        adapter_pending = [item for item in pending if item.get("model_role") == "adapter"]

        def _safe(spec, current):
            try:
                result = _run(spec, current)
            except Exception as exc:
                result = {
                    "text": "",
                    "error": str(exc),
                    "finish_reason": "error",
                    "truncated": False,
                    "device": index,
                }
            if partial is not None:
                _append_jsonl(partial, {"case_id": spec.get("case_id"), "model_role": spec.get("model_role"), **result})
            return result

        for spec in base_pending:
            outputs[(spec["case_id"], "base")] = _safe(spec, model)
        if adapter_pending:
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()
            for spec in adapter_pending:
                outputs[(spec["case_id"], "adapter")] = _safe(spec, model)
    except Exception as exc:
        return {"skipped": True, "reason": f"generation_exception: {exc}"}
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = []
    for spec in pending:
        key = (spec["case_id"], spec["model_role"])
        if key not in outputs:
            return {"skipped": True, "reason": "missing_generated_row"}
        rows.append(outputs[key])
    return rows
