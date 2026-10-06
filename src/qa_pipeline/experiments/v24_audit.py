"""drug_v24 只读审计、离线重评分和空白审核包。不改写 v22/v23 历史文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..textutil import compact, jaccard
from .adapter_eval import (
    canonical_messages,
    case_id_of,
    eval_adapter,
    rescore_saved,
)
from .dev_candidates import _section
from .devices import select_trainable_gpus
from .drug_corpus import _family_key, extract_identity, raw_dir, split_for_family
from .frozen_loader import FrozenLoadError, load_frozen_subset
from .scoring import (
    SCORER_V1,
    SCORER_V2,
    SCORER_V3,
    aggregate_families,
    message_digest,
    prediction_item_signature,
    reference_self_check,
    score_task,
    score_task_v2,
    score_task_v3,
    visible_support,
)

REPO = Path(__file__).resolve().parents[3]
BASE_MODEL = Path("/data1/pjw/models/Qwen2.5-7B-Instruct")
ADAPTER = REPO / "runs/drug_v22/E3_g0/sft/lora/adapter"
BLIND_SEED = 20261006
INFER = {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "dtype": "bfloat16"}


def assert_writable_audit_dest(dest: Path, repo: Path | None = None) -> None:
    root = repo or REPO
    protected = (root / "runs/drug_v23_audit").resolve()
    resolved = dest.resolve()
    if resolved == protected or protected in resolved.parents or resolved in protected.parents and resolved != root:
        raise RuntimeError("refuse_overwrite_protection_baseline")
    if resolved.name == "drug_v23_audit" or "drug_v23_eval" in resolved.parts or resolved.name == "drug_v22":
        raise RuntimeError("refuse_overwrite_protection_baseline")


def _ngram_set(text: str, n: int = 5) -> set[str]:
    packed = compact(text)
    if len(packed) < n:
        return {packed} if packed else set()
    return {packed[i : i + n] for i in range(len(packed) - n + 1)}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0, "sha256": _sha256(path)}


def resolve_output_dir(name: str, repo: Path | None = None) -> Path:
    root = repo or REPO
    dest = root / "runs" / name
    assert_writable_audit_dest(dest, root)
    commit = _git_commit()
    if dest.exists():
        stamp_path = dest / "version.json"
        if stamp_path.is_file():
            old = str(_read_json(stamp_path).get("git_commit") or "")
            if old and old != commit:
                dest = root / "runs" / f"{name}_{commit[:8]}"
                assert_writable_audit_dest(dest, root)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def verify_saved_input_identity(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    template_id: str,
    infer_config: dict[str, Any],
    base_id: str,
    adapter_id: str,
) -> dict[str, Any]:
    by_id = {case_id_of(case): case for case in cases}
    message_mismatch = []
    signature_mismatch = []
    missing_case = []
    for row in predictions:
        case_id = str(row.get("case_id") or "")
        case = by_id.get(case_id)
        if case is None:
            missing_case.append(case_id)
            continue
        if message_digest(canonical_messages(case)) != message_digest(row.get("messages")):
            message_mismatch.append({"case_id": case_id, "model_role": row.get("model_role")})
        role = str(row.get("model_role") or "")
        expected = prediction_item_signature(
            case,
            str(row.get("model_id") or base_id),
            adapter_id if role == "adapter" else "",
            template_id,
            infer_config,
            role,
            model_input=row.get("messages"),
        )
        if row.get("signature") != expected:
            signature_mismatch.append({"case_id": case_id, "model_role": role})
    return {
        "prediction_n": len(predictions),
        "message_mismatch_n": len(message_mismatch),
        "signature_mismatch_n": len(signature_mismatch),
        "missing_case_n": len(missing_case),
        "message_mismatch": message_mismatch,
        "signature_mismatch": signature_mismatch,
        "input_match": not message_mismatch and not missing_case,
        "signature_match": not signature_mismatch,
    }


def apply_same_input_gold_patch(case: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """只拆可见必答要点和不可见要点，不改问题、上下文或 system。"""
    if case.get("case_id") != "v23_behavior_partial_usage":
        return case, None
    context = str(case.get("context") or "")
    points = [str(point) for point in (case.get("answer_points") or []) if str(point).strip()]
    required = [point for point in points if point in context]
    unavailable = [point for point in points if point not in context]
    if not required or not unavailable:
        return case, "could_not_separate_visible_points"
    patched = dict(case)
    patched["required_points"] = required
    patched["optional_points"] = []
    patched["unavailable_points"] = unavailable
    patched["prohibited_conclusions"] = list(unavailable)
    patched["gold_revision"] = "visible_required_points"
    return patched, "gold_structure_only"


def _quote_line(text: str, needle: str) -> str | None:
    if not needle:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if needle in stripped and 0 < len(stripped) <= 80:
            return stripped
    return None


def _heading_quote(text: str, body: str) -> str | None:
    raw = (body or "").strip()
    if not raw:
        return None
    index = text.find(raw)
    if index < 0:
        return None
    prefix = text[max(0, index - 40) : index]
    match = re.search(r"【[^】]{1,24}】\s*$", prefix)
    if match is None:
        return None
    start = max(0, index - 40) + match.start()
    snippet = text[start : index + len(raw)].strip()
    if snippet and snippet in text:
        return snippet
    return None


def propose_input_revision(case: dict[str, Any], full_text: str) -> dict[str, Any] | None:
    """只有冻结原文里能逐字引用、且学生当前看不见关键身份时才生成新输入。"""
    pattern = str(case.get("pattern") or "")
    if pattern in {"wrong_object", "replaced_context"}:
        shown = str(case.get("shown_generic") or "")
        line = _quote_line(full_text, shown)
        if line is None or line in str(case.get("context") or ""):
            return None
        revised = dict(case)
        revised["case_id"] = f"v24_behavior_{pattern}"
        revised["parent_case_id"] = case.get("case_id")
        revised["context"] = line + "\n" + str(case.get("context") or "")
        revised["dataset_version"] = "drug_v24_input"
        revised["review_status"] = "unreviewed"
        revised["input_change"] = "added_visible_object_identity"
        return revised
    if pattern in {"missing_reaction", "contra_not_visible"}:
        snippet = _heading_quote(full_text, str(case.get("context") or ""))
        if snippet is None or snippet == str(case.get("context") or "").strip():
            return None
        revised = dict(case)
        revised["case_id"] = f"v24_behavior_{pattern}"
        revised["parent_case_id"] = case.get("case_id")
        revised["context"] = snippet
        revised["dataset_version"] = "drug_v24_input"
        revised["review_status"] = "unreviewed"
        revised["input_change"] = "added_visible_section_heading"
        return revised
    return None


def audit_behavior_case(case: dict[str, Any], full_text: str) -> dict[str, Any]:
    context = str(case.get("context") or "")
    question = str(case.get("question") or "")
    pattern = str(case.get("pattern") or "")
    issues = []
    notes = []
    if pattern == "clarify_strength":
        strengths = case.get("strengths") or []
        usage = _section(full_text, "用法用量") if full_text else ""
        spec = _section(full_text, "规格") if full_text else ""
        in_usage = [item for item in strengths if item.replace(" ", "") in re.sub(r"\s+", "", usage)]
        in_spec = [item for item in strengths if item.replace(" ", "") in re.sub(r"\s+", "", spec)]
        if strengths and in_usage and not in_spec:
            issues.append("抽出的数字位于用法用量，不是多个规格。金标前提不能当成已核实。")
    if pattern == "replaced_context" and len(context) < 20:
        issues.append("上下文过短，看不到被替换资料属于哪一个对象。")
    if pattern in {"wrong_object", "replaced_context", "retained_distractor"}:
        shown = str(case.get("shown_generic") or "")
        asked = str(case.get("asked_generic") or "")
        if shown and shown not in context:
            issues.append("学生可见上下文没有另一药品的身份。")
        if asked and asked not in question and asked not in context:
            issues.append("问题对象名称没有出现在题干里。")
        if pattern == "retained_distractor":
            extra = [line.strip() for line in context.splitlines() if line.strip() and line.strip() not in str(case.get("answer") or "")]
            if shown and shown not in context or not extra or all(len(line) <= 8 for line in extra):
                issues.append("干扰材料没有展示另一对象，不能当成干扰条件下的充分证据。")
    if "该条件" in question and "该条件" not in context:
        issues.append("题干使用“该条件”，可见上下文没有给出这个指代。被删原句没有保存在候选里，无法确认是否仍有等效表述。")
    if pattern == "partial_usage":
        support = visible_support({**case, "required_points": [], "unavailable_points": []})
        hidden = [point for point in (case.get("answer_points") or []) if point and point not in context]
        if hidden:
            issues.append("answer_points 含有学生看不见的要点。评分若把这些要点当必答，会把合法部分回答判错。")
            support = {"hidden_points": hidden}
        else:
            support = support
    else:
        support = visible_support(case)
    if pattern in {"missing_reaction", "contra_not_visible"} and "【" not in context:
        issues.append("上下文没有章节标题，学生看不到金标所声称的章节身份。")
    if "中文名：" in question or "汉语拼音" in question:
        notes.append("题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。")
    return {
        "case_id": case.get("case_id"),
        "pattern": pattern,
        "expected_action": case.get("expected_action"),
        "self_check": reference_self_check(case),
        "visible_support": support,
        "issues": issues,
        "notes": notes,
        "review_status": "pending_review",
        "adjudication": "",
    }


def _field_coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def qty(text: str) -> bool:
        return bool(re.search(r"\d+(?:\.\d+)?\s*(?:mg|g|μg|ug|ml|mL|片|粒)", text or "", flags=re.I))

    def conditional(case: dict[str, Any]) -> bool:
        blob = "\n".join([str(case.get("question") or ""), str(case.get("answer") or "")])
        return bool(re.search(r"如果|若|当|饭后|条件下", blob))

    rows = []
    specs = [
        (
            "numeric_bindings",
            "把数字绑到对象。历史评分没有这个字段时不会执行。",
            lambda case: qty("\n".join([str(case.get("answer") or ""), str(case.get("context") or "")])),
            lambda case: bool(case.get("numeric_bindings")),
        ),
        (
            "required_conditions",
            "条件要点必须出现在回答中。字段为空时不会执行。",
            conditional,
            lambda case: bool(case.get("required_conditions")),
        ),
        (
            "synonym_groups",
            "同义说法。v3 只把与整个要点等价的同义组算命中，字段为空时不会执行。",
            lambda case: True,
            lambda case: bool(case.get("synonym_groups")),
        ),
        (
            "atomic_answer_points",
            "评分器按整条 answer_points 做覆盖，不会自动拆开复合句。",
            lambda case: bool(case.get("answer_points")),
            lambda case: bool(case.get("answer_points")) and all("。" not in str(point) and "；" not in str(point) for point in case.get("answer_points") or []),
        ),
        (
            "visible_evidence_map",
            "证据位于学生可见 excerpt。历史 v2 评分没有单独执行这项。",
            lambda case: True,
            lambda case: bool(case.get("evidence")) and str(case.get("evidence")) in str(case.get("context") or ""),
        ),
    ]
    for name, note, applicable, present in specs:
        applicable_ids = [case_id_of(case) for case in cases if applicable(case)]
        present_ids = [case_id_of(case) for case in cases if case_id_of(case) in set(applicable_ids) and present(case)]
        rows.append(
            {
                "check": name,
                "applicable_n": len(applicable_ids),
                "with_required_fields_n": len(present_ids),
                "executed_n": 0,
                "not_executed_reason": "本批候选没有执行该检查所需的字段，或评分器不会仅因支持该字段就自动运行。" if len(present_ids) < len(applicable_ids) else note,
                "note": note,
            }
        )
    visible_executed = 0
    for case in cases:
        if case.get("stratum") == "behavior" or case.get("evidence"):
            visible_executed += 1
    for row in rows:
        if row["check"] == "visible_evidence_map":
            row["executed_n"] = visible_executed
            row["not_executed_reason"] = "v24 审计对行为题和带 evidence 字段的题做了可见性检查。其余题没有证据映射字段。"
            row["historical_scorer_executed_n"] = 0
    return {"checks": rows, "candidate_n": len(cases)}


def _score_text(text: str, case: dict[str, Any], scorer: str) -> dict[str, Any]:
    if scorer == SCORER_V1:
        scored = score_task(text, case)
        scored["scorer_version"] = SCORER_V1
        return scored
    if scorer == SCORER_V2:
        return score_task_v2(text, case)
    if scorer == SCORER_V3:
        return score_task_v3(text, case)
    raise ValueError(scorer)


def _blank_review() -> dict[str, str]:
    return {
        "reviewer_a": "",
        "reviewer_b": "",
        "opinion_a": "",
        "opinion_b": "",
        "adjudication": "",
        "review_status": "pending_review",
    }


def _stratum_counts(rows: list[dict[str, Any]], key: str = "passed") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stratum") or "unspecified")].append(row)
    summary = {}
    for name, items in grouped.items():
        packed = [{**item, "passed": item.get(key)} for item in items]
        stats = aggregate_families(packed)
        summary[name] = {
            "row_count": len(items),
            "family_count": stats["family_count"],
            "source_family_count": stats["source_family_count"],
            "auxiliary_pass_rate": stats["task_accuracy"],
            "unresolved_n": stats.get("unresolved_n"),
            "pass_n": sum(1 for item in items if item.get(key) is True),
            "fail_n": sum(1 for item in items if item.get(key) is False),
            "weighting": "equal_source_family",
            "result_class": "exploratory_auxiliary",
            "review_status": "pending_review",
        }
    return summary


def _model_ids() -> dict[str, str]:
    config = _sha256(BASE_MODEL / "config.json") or ""
    adapter = _sha256(ADAPTER / "adapter_model.safetensors") or ""
    template = _sha256(BASE_MODEL / "tokenizer_config.json") or ""
    adapter_template = _sha256(ADAPTER / "tokenizer_config.json") or ""
    return {
        "base_model": str(BASE_MODEL),
        "base_id": f"Qwen2.5-7B-Instruct@{config[:16]}",
        "adapter": str(ADAPTER),
        "adapter_id": adapter[:16],
        "template_id": template[:16],
        "adapter_tokenizer_config_sha256": adapter_template,
        "base_tokenizer_config_sha256": template,
        "tokenizer_configs_differ": template != adapter_template,
    }


def _load_docs() -> list[dict[str, Any]]:
    frozen = REPO / "data/campus_hospital_drug_instructions/frozen"
    try:
        return load_frozen_subset(frozen / "subset", frozen / "subset_manifest.json")
    except FrozenLoadError as exc:
        return [{"error": str(exc)}]


def _source_audit(train: list[dict[str, Any]], candidates: list[dict[str, Any]], docs: list[dict[str, Any]], source_index: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {}
    for doc in docs:
        if doc.get("error"):
            continue
        by_name[doc.get("stem")] = doc
        by_name[Path(str(doc.get("path") or "")).name] = doc
        by_name[Path(str(doc.get("path") or "")).stem] = doc
    index_by = {}
    for row in source_index:
        index_by[row.get("stem")] = row
        index_by[row.get("filename")] = row
        index_by[Path(str(row.get("filename") or "")).stem] = row
    mapped = 0
    unmapped = []
    split_changed = []
    family_changed = []
    empty_family = []
    for row in train:
        meta = row.get("metadata") or {}
        if not meta.get("family_id"):
            empty_family.append(row.get("id"))
        source = str(meta.get("source") or "")
        indexed = index_by.get(source)
        if indexed is None:
            unmapped.append(row.get("id"))
            continue
        mapped += 1
        doc = by_name.get(source) or by_name.get(Path(source).stem)
        if not doc or not doc.get("text"):
            continue
        identity = extract_identity(doc["text"])
        family = _family_key(identity, "source")
        if family != indexed.get("source_family_id"):
            family_changed.append({"source": source, "stored": indexed.get("source_family_id"), "recomputed": family})
        stored_split = indexed.get("split")
        recomputed_split = split_for_family(str(indexed.get("source_family_id") or family))
        if stored_split != recomputed_split:
            split_changed.append({"source": source, "stored": stored_split, "recomputed": recomputed_split})
    independent = [case for case in candidates if case.get("stratum") == "independent"]
    ind_families = {case.get("source_family_id") for case in independent}
    ind_files = {case.get("source") for case in independent}
    train_sets = []
    for doc in docs:
        text = doc.get("text") or ""
        if not text:
            continue
        train_sets.append((_ngram_set(text[:4000]), _ngram_set(text)))
    missed = []
    max_full = 0.0
    raw_root = raw_dir()
    seen_sources: set[str] = set()
    for case in independent:
        name = str(case.get("source") or "")
        if name in seen_sources:
            continue
        seen_sources.add(name)
        path = raw_root / name
        if path.suffix != ".txt":
            path = raw_root / f"{name}.txt"
        if not path.is_file():
            missed.append({"case_id": case.get("case_id"), "reason": "independent_source_file_missing", "source": name})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        full_set = _ngram_set(text)
        prefix_set = _ngram_set(text[:4000])
        # 前 4000 字之外再切窗，避免只看开头时漏掉后文重合。
        tail_sets = [_ngram_set(text[start : start + 4000]) for start in range(4000, len(text), 4000)]
        for train_prefix, train_full in train_sets:
            prefix_score = jaccard(prefix_set, train_prefix)
            full_score = jaccard(full_set, train_full)
            tail_score = max((jaccard(item, train_full) for item in tail_sets), default=0.0)
            max_full = max(max_full, full_score, tail_score)
            if prefix_score < 0.92 and (full_score >= 0.92 or tail_score >= 0.92):
                missed.append(
                    {
                        "source": name,
                        "prefix_jaccard": round(prefix_score, 4),
                        "full_jaccard": round(full_score, 4),
                        "tail_window_jaccard": round(tail_score, 4),
                    }
                )
                break
    evidence_rows = []
    for case in candidates:
        evidence = str(case.get("evidence") or "")
        context = str(case.get("context") or "")
        if not evidence:
            continue
        evidence_rows.append({"case_id": case.get("case_id"), "evidence_in_excerpt": evidence in context, "stratum": case.get("stratum")})
    raw = _jsonl(REPO / "runs/drug_v22/E1_direct/qa.raw.jsonl")
    kept = _jsonl(REPO / "runs/drug_v22/E1_direct/qa.kept.jsonl")
    rejected = _jsonl(REPO / "runs/drug_v22/E1_direct/qa.rejected.jsonl")
    dropped = sorted({row.get("qa_id") for row in raw} - {row.get("qa_id") for row in kept} - {row.get("qa_id") for row in rejected})
    return {
        "source_index_n": len(source_index),
        "train_rows_mapped_by_source_index": mapped,
        "train_rows_unmapped": unmapped,
        "empty_family_id_n": len(empty_family),
        "empty_family_id": empty_family,
        "empty_family_status": "待补映射。本次没有填写 family_id。",
        "recomputed_family_mismatch_n": len(family_changed),
        "recomputed_family_mismatch": family_changed[:20],
        "recomputed_split_mismatch_n": len(split_changed),
        "recomputed_split_mismatch": split_changed[:20],
        "frozen_partition_rewritten": False,
        "independent_case_n": len(independent),
        "independent_source_family_n": len(ind_families),
        "independent_file_n": len(ind_files),
        "independent_is_ten_families": len(ind_families) == 10 and len(ind_files) == 10,
        "prefix_screen_missed_full_text_near_duplicate": missed,
        "max_full_jaccard_seen_during_audit": round(max_full, 4),
        "evidence_in_excerpt": {
            "with_evidence_field_n": len(evidence_rows),
            "evidence_inside_context_n": sum(1 for row in evidence_rows if row["evidence_in_excerpt"]),
            "evidence_outside_context_n": sum(1 for row in evidence_rows if not row["evidence_in_excerpt"]),
        },
        "rule_clean_dropped": [
            {"id": item, "reason": "unknown", "provenance": "reconstructed_from_id_set_difference"} for item in dropped
        ],
        "note": "重算结果只作审计，没有回写冻结分区或 source_index。",
    }


def _behavior_findings(cases: list[dict[str, Any]], docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_name = {}
    for doc in docs:
        if not doc.get("text"):
            continue
        by_name[Path(str(doc.get("path") or "")).stem] = doc["text"]
        by_name[doc.get("stem")] = doc["text"]
    findings = []
    revisions = []
    for case in cases:
        if case.get("stratum") != "behavior":
            continue
        text = by_name.get(str(case.get("source") or ""), "")
        if not text:
            text = by_name.get(Path(str(case.get("source") or "")).stem, "")
        finding = audit_behavior_case(case, text)
        patched, gold_change = apply_same_input_gold_patch(case)
        finding["gold_patch"] = gold_change
        finding["patched_self_check"] = reference_self_check(patched) if gold_change else None
        proposal_case = case
        pattern = str(case.get("pattern") or "")
        if pattern == "replaced_context" and not case.get("shown_generic"):
            sibling = next((item.get("shown_generic") for item in cases if item.get("source") == case.get("source") and item.get("shown_generic")), "")
            if sibling:
                proposal_case = {**case, "shown_generic": sibling}
        revision = propose_input_revision(proposal_case, text) if text else None
        if revision is not None:
            finding["new_input_case_id"] = revision["case_id"]
            finding["input_change"] = revision["input_change"]
            revisions.append(revision)
        else:
            finding["new_input_case_id"] = None
            finding["input_change"] = None
        patched_ok = bool(finding["patched_self_check"] and finding["patched_self_check"].get("passed") is True)
        if gold_change and patched_ok:
            finding["disposition"] = "gold_structure_only_reuse_predictions"
        elif finding["issues"] or finding["self_check"]["passed"] is not True:
            finding["disposition"] = "pending_review"
        else:
            finding["disposition"] = "reuse_predictions_pending_human_review"
        findings.append(finding)
    return findings, revisions


def _comparison_rows(cases: list[dict[str, Any]], predictions: list[dict[str, Any]], old_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_role: dict[tuple[str, str], str] = {}
    for row in predictions:
        by_role[(str(row.get("case_id")), str(row.get("model_role")))] = str(row.get("text") or "")
    old_by_id = {str(row.get("case_id")): row for row in old_scores}
    compared = []
    for case in cases:
        patched, gold_change = apply_same_input_gold_patch(case)
        case_id = case_id_of(case)
        base = by_role.get((case_id, "base"), "")
        tuned = by_role.get((case_id, "adapter"), "")
        v1_before = _score_text(base, case, SCORER_V1)
        v1_after = _score_text(tuned, case, SCORER_V1)
        v2_before = _score_text(base, case, SCORER_V2)
        v2_after = _score_text(tuned, case, SCORER_V2)
        v3_before = _score_text(base, case, SCORER_V3)
        v3_after = _score_text(tuned, case, SCORER_V3)
        v3_patched_before = _score_text(base, patched, SCORER_V3)
        v3_patched_after = _score_text(tuned, patched, SCORER_V3)
        historical = old_by_id.get(case_id) or {}
        historical_before = (historical.get("before") or {}).get("passed")
        historical_after = (historical.get("after") or {}).get("passed")
        reasons = []
        if historical_before is not None and bool(historical_before) != bool(v2_before.get("passed")):
            reasons.append("v2_recompute_mismatch")
        if historical_after is not None and bool(historical_after) != bool(v2_after.get("passed")):
            reasons.append("v2_recompute_mismatch")
        if bool(v2_after.get("passed")) != (v3_after.get("passed") is True) or bool(v2_before.get("passed")) != (v3_before.get("passed") is True):
            reasons.append("scoring_fix")
        if gold_change and (
            v3_after.get("passed") != v3_patched_after.get("passed")
            or v3_before.get("passed") != v3_patched_before.get("passed")
            or v3_after.get("critical_errors") != v3_patched_after.get("critical_errors")
            or v3_before.get("critical_errors") != v3_patched_before.get("critical_errors")
            or v3_after.get("required_point_coverage") != v3_patched_after.get("required_point_coverage")
            or v3_before.get("required_point_coverage") != v3_patched_before.get("required_point_coverage")
        ):
            reasons.append("gold_structure_fix")
        if not reasons:
            reasons.append("no_auxiliary_change")
        compared.append(
            {
                "case_id": case_id,
                "stratum": case.get("stratum"),
                "family_id": case.get("family_id"),
                "source_family_id": case.get("source_family_id"),
                "historical_v2_before_passed": historical_before,
                "historical_v2_after_passed": historical_after,
                "v1_before_passed": v1_before.get("passed"),
                "v1_after_passed": v1_after.get("passed"),
                "v2_before_passed": v2_before.get("passed"),
                "v2_after_passed": v2_after.get("passed"),
                "v3_before_passed": v3_before.get("passed"),
                "v3_after_passed": v3_after.get("passed"),
                "v3_patched_before_passed": v3_patched_before.get("passed"),
                "v3_patched_after_passed": v3_patched_after.get("passed"),
                "difference_sources": reasons,
                "human_disagreement": "pending_review",
                "input_changed": False,
                "result_class": "exploratory_auxiliary",
            }
        )
    return compared


def _review_materials(
    dest: Path,
    train: list[dict[str, Any]],
    seen: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    finding_by_id = {item["case_id"]: item for item in findings}
    gold_rows = []
    for row in train:
        meta = row.get("metadata") or {}
        messages = row.get("messages") or []
        user = next((item.get("content") for item in messages if item.get("role") == "user"), "")
        target = next((item.get("content") for item in messages if item.get("role") == "assistant"), "")
        gold_rows.append(
            {
                "packet": "gold",
                "case_id": row.get("id"),
                "stratum": "train_target",
                "question": user,
                "actual_input": messages[:-1],
                "source": meta.get("source"),
                "source_file_id": meta.get("source_file_id"),
                "visible_evidence": meta.get("evidence"),
                "required_points": [target] if target else [],
                "optional_points": [],
                "unavailable_points": [],
                "expected_action": meta.get("expected_action") or "answer",
                "constraints": [],
                "open_questions": ["训练目标是否只包含资料中可见的事实。"] if not meta.get("family_id") else [],
                "family_id_status": "missing_unmapped" if not meta.get("family_id") else "present",
                **_blank_review(),
            }
        )
    for case in candidates:
        finding = finding_by_id.get(case.get("case_id")) or {}
        patched, _gold_change = apply_same_input_gold_patch(case)
        gold_rows.append(
            {
                "packet": "gold",
                "case_id": case.get("case_id"),
                "stratum": case.get("stratum"),
                "question": case.get("question"),
                "actual_input": canonical_messages(case),
                "source": case.get("source"),
                "source_family_id": case.get("source_family_id"),
                "source_hash": case.get("source_hash"),
                "visible_context": case.get("context"),
                "visible_evidence": case.get("evidence"),
                "required_points": patched.get("required_points") or case.get("answer_points"),
                "optional_points": patched.get("optional_points") or [],
                "unavailable_points": patched.get("unavailable_points") or [],
                "prohibited_conclusions": patched.get("prohibited_conclusions") or [],
                "expected_action": case.get("expected_action"),
                "reference_answer": case.get("answer"),
                "open_questions": finding.get("issues") or [],
                "gold_revision": patched.get("gold_revision"),
                "source_in_training_subset": case.get("stratum") == "seen_rephrase",
                "eval_partition": case.get("split"),
                **_blank_review(),
            }
        )
    answers = []
    for row in seen:
        for role, key in (("base", "base_answer"), ("adapter", "tuned_answer")):
            answers.append({"case_id": row.get("case_id"), "stratum": "train_seen", "model_role": role, "text": row.get(key) or ""})
    for row in predictions:
        answers.append(
            {
                "case_id": row.get("case_id"),
                "stratum": row.get("stratum"),
                "model_role": row.get("model_role"),
                "text": row.get("text") or "",
            }
        )
    rng = random.Random(BLIND_SEED)
    order = list(range(len(answers)))
    rng.shuffle(order)
    blind = []
    mapping = []
    for slot, index in enumerate(order, start=1):
        item = answers[index]
        blind_id = f"blind_{slot:04d}"
        blind.append(
            {
                "blind_id": blind_id,
                "case_id": item["case_id"],
                "text": item["text"],
                "task_completed": "",
                "key_errors": "",
                "unsupported_content": "",
                "behavior_appropriate": "",
                "reviewer_a": "",
                "reviewer_b": "",
                "adjudication": "",
                "review_status": "pending_review",
            }
        )
        mapping.append({"blind_id": blind_id, "case_id": item["case_id"], "model_role": item["model_role"], "stratum": item["stratum"]})
    _write_jsonl(dest / "gold_review.jsonl", gold_rows)
    _write_jsonl(dest / "blind_answers.jsonl", blind)
    _write_json(dest / "blind_identity_map.json", {"seed": BLIND_SEED, "n": len(mapping), "rows": mapping, "note": "身份映射与盲评包分开。审核界面不应读取此文件。"})
    return {
        "gold_items": len(gold_rows),
        "blind_answers": len(blind),
        "filled_adjudications": 0,
        "review_status": "pending_review",
        "reviewers_recorded": False,
    }


def _attach_auxiliary_counts(reinfer_result: dict[str, Any], rescore_dir: Path) -> None:
    paths = {
        "train_seen": rescore_dir / "train_seen_confirmed/scores.aux-rules-v3.jsonl",
        "new_behavior_inputs": rescore_dir / "new_inputs/scores.aux-rules-v3.jsonl",
    }
    for item in reinfer_result.get("runs") or []:
        path = paths.get(str(item.get("name")))
        if path is None or not path.is_file():
            continue
        rows = _jsonl(path)
        item["auxiliary_n"] = len(rows)
        item["auxiliary_before_pass_n"] = sum(1 for row in rows if (row.get("before") or {}).get("passed") is True)
        item["auxiliary_after_pass_n"] = sum(1 for row in rows if (row.get("after") or {}).get("passed") is True)
        item["enters_formal_metric"] = False


def write_new_input_reviews(dest: Path, protocol: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """新输入单独成包，不并进原来的 174 条金标和 348 份盲评。"""
    if not protocol or not predictions:
        return {"new_input_gold_items": 0, "new_input_blind_answers": 0, "new_input_filled_adjudications": 0}
    gold_rows = []
    for case in protocol:
        messages = canonical_messages(case)
        gold_rows.append(
            {
                "packet": "gold",
                "case_id": case.get("case_id"),
                "parent_case_id": case.get("parent_case_id"),
                "stratum": "behavior_new_input",
                "question": case.get("question"),
                "actual_input": messages,
                "visible_context": case.get("context"),
                "input_change": case.get("input_change"),
                "required_points": case.get("required_points") or case.get("answer_points") or [],
                "optional_points": case.get("optional_points") or [],
                "unavailable_points": case.get("unavailable_points") or [],
                "expected_action": case.get("expected_action"),
                "reference_answer": case.get("answer"),
                "reviewer_a": "",
                "reviewer_b": "",
                "opinion_a": "",
                "opinion_b": "",
                "adjudication": "",
                "review_status": "pending_review",
            }
        )
    answers = [
        {"case_id": row.get("case_id"), "model_role": row.get("model_role"), "text": row.get("text") or ""}
        for row in predictions
        if not row.get("error")
    ]
    rng = random.Random(BLIND_SEED)
    order = list(range(len(answers)))
    rng.shuffle(order)
    blind = []
    mapping = []
    for slot, index in enumerate(order, start=1):
        item = answers[index]
        blind_id = f"new_blind_{slot:04d}"
        blind.append(
            {
                "blind_id": blind_id,
                "case_id": item["case_id"],
                "text": item["text"],
                "task_completed": "",
                "key_errors": "",
                "unsupported_content": "",
                "behavior_appropriate": "",
                "reviewer_a": "",
                "reviewer_b": "",
                "adjudication": "",
                "review_status": "pending_review",
            }
        )
        mapping.append({"blind_id": blind_id, "case_id": item["case_id"], "model_role": item["model_role"], "stratum": "behavior_new_input"})
    dest.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dest / "new_input_gold_review.jsonl", gold_rows)
    _write_jsonl(dest / "new_input_blind_answers.jsonl", blind)
    _write_json(
        dest / "new_input_blind_identity_map.json",
        {"seed": BLIND_SEED, "n": len(mapping), "rows": mapping, "note": "与 348 份盲评分开。审核界面不应读取此文件。"},
    )
    return {
        "new_input_gold_items": len(gold_rows),
        "new_input_blind_answers": len(blind),
        "new_input_filled_adjudications": 0,
    }


def _next_experiments() -> str:
    return """# 下一阶段实验协议

这些实验本次没有启动。下面只写启动条件。

## A 固定上下文的新问法对照

- 使用已有 30 个核心问题。
- 保持该题训练时的完整上下文和 system 不变，只替换问题文字。
- 原题、两种改写使用同一评分器和同一金标版本。
- 原题预测只有 messages、基座、adapter、tokenizer/模板和生效推理参数都匹配时才复用。
- 最多新增 60 个输入、基座和 G0 共 120 份回答。
- 现有整篇说明书上下文的 v23 改写结果保留为另一条件，不并进这一条件的分数。

启动条件：v24 的输入身份审计和辅助重评分已经落盘，金标审核至少完成这 30 个核心问题，并且当前 GPU 有一张空闲卡。不能按基座答错来决定改写是否保留。

## B 成对行为边界测试

- 预先选定 12 个核心问题、至少 8 个来源家族。
- 每题 4 个条件：正确充分、移除关键支持、明确错误对象、保留正确证据并加入干扰。
- 共 48 个输入，最多 96 份回答。
- 每个条件单独审核。删掉一个片段之后，还要检查全文其他位置没有等效支持，才能把该条件标成证据不足。

启动条件：行为分类的参考答案自评通过，并且可见证据规则已经用于这 48 个输入的人工审核。不能用 v23 的 10 道行为题直接充当这组成对题。

## C 更有区分能力的独立来源开发题

- 保留原 30 道独立来源简单题，不删除。
- 按预先写好的题型增加条件、否定、数值绑定和多证据组合题。
- 筛选规则在看模型回答之前固定。

启动条件：独立来源层的来源家族审计没有发现全文近重复，并且新题的字段覆盖计划已经写明。不能根据基座答错来筛题。

## D 训练实验

多 seed、G1–G3、行为训练或其他对照，只在评测协议可靠、而且缺陷或收益经过人工审核确认之后选择一条路线。不要为了补表同时启动全部路线。

启动条件：正式主指标仍然未执行也可以先做方法设计，但不能把当前辅助分当作已经确认的收益。
"""


def _render_report(summary: dict[str, Any]) -> str:
    strata = summary["exploratory_auxiliary"]["by_stratum"]
    lines = [
        "# drug_v24 评测可靠性审计",
        "",
        "本报告使用与 JSON 相同的分层：seen_rephrase、independent、behavior 分开，不合成泛化总分。",
        "全部新题和原题都还没有真实人工裁定。规则分是探索性辅助结果。",
        "",
        "## 已完成",
        "",
    ]
    for item in summary["completed"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 探索性辅助结果", ""])
    for name in ("seen_rephrase", "independent", "behavior"):
        row = strata.get(name) or {}
        lines.append(
            f"- {name}：题数 {row.get('row_count')}，来源家族 {row.get('source_family_count')}，核心问题 {row.get('family_count')}。"
            f"已判定通过 {row.get('pass_n')}，未通过 {row.get('fail_n')}，待复核 {row.get('unresolved_n')}。"
            f"等权家族通过率 {row.get('auxiliary_pass_rate')} 只统计已经判定的题，待复核不进入分子或分母。"
            "这不是语义正确率。"
        )
    lines.extend(
        [
            "",
            "## 待人工审核",
            "",
            f"- 金标包 {summary['review']['gold_items']} 条，盲评回答 {summary['review']['blind_answers']} 份。",
            "- 审核者、意见和裁定均为空白。filled_adjudications = 0。",
            f"- 新输入单独成包：金标 {summary['review'].get('new_input_gold_items', 0)} 条，盲评 {summary['review'].get('new_input_blind_answers', 0)} 份，不并进上面的 174 和 348。",
        ]
    )
    for note in summary["review"].get("audit_notes") or []:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 技术失败或资源受阻",
            "",
        ]
    )
    if summary["blocked"]:
        for item in summary["blocked"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 没有把技术失败写成内容错误。")
    lines.extend(["", "## 本次未授权启动的后续实验", ""])
    for item in summary["not_authorized"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 旧辅助分与 v3", ""])
    diff = summary["score_differences"]
    lines.append(f"- 与历史 v2 重新计算不一致：{diff['v2_recompute_mismatch_n']}。")
    lines.append(f"- 仅因评分规则变化而改变通过与否：{diff['scoring_fix_n']}。")
    lines.append(f"- 因可见要点金标结构调整而变化：{diff['gold_structure_fix_n']}。")
    lines.append("- 输入变化的旧 100 题：0。新输入如果存在，写在补推理清单里，不混进这 100 题。")
    lines.append("- 人工分歧：全部 pending_review，没有已确认分歧。")
    identity = summary.get("identity") or {}
    lines.extend(
        [
            "",
            "## 历史保护与输入身份",
            "",
            f"- 受保护历史文件哈希前后一致：{summary.get('historical_files_unchanged')}。",
            f"- v23 的 200 份回答输入匹配：{identity.get('input_match')}，签名匹配：{identity.get('signature_match')}，消息不一致 {identity.get('message_mismatch_n')}，签名不一致 {identity.get('signature_mismatch_n')}。",
            "- v22 的 148 份 train_seen 回答保存了 messages，但没有逐条签名、模板哈希和生效推理参数。adapter 目录的 tokenizer_config 与基座不同，且当时 git_dirty。本次不把它们写成输入匹配证明。",
            "- 新算出的当前哈希不能证明 v23 基线之前文件没有变化。v23 manifest 没有被重写。",
            "",
            "## 行为题",
            "",
        ]
    )
    for item in summary.get("behavior_items") or []:
        issues = "；".join(item.get("issues") or []) or "没有自动发现的结构问题"
        lines.append(
            f"- {item.get('case_id')}：自评通过 {item.get('self_check_passed')}，处置 {item.get('disposition')}，新输入 {item.get('new_input_case_id') or '无'}。{issues}"
            + (f" 备注：{'；'.join(item.get('notes') or [])}" if item.get("notes") else "")
        )
    source = summary.get("source_audit") or {}
    evidence = source.get("evidence_in_excerpt") or {}
    lines.extend(
        [
            "",
            "## 来源与核心问题",
            "",
            f"- 独立来源：{source.get('independent_case_n')} 题，{source.get('independent_source_family_n')} 个来源家族，{source.get('independent_file_n')} 个文件。十家族且十文件：{source.get('independent_is_ten_families')}。",
            f"- 空 family_id：{source.get('empty_family_id_n')}，保持未映射。训练行未能对照 source_index：{source.get('train_rows_unmapped_n')}。",
            f"- rule_clean 丢弃 {source.get('rule_clean_dropped_n')} 条，原因未知，来源是 id 集合差。",
            f"- 前缀筛漏掉的全文近重复：{source.get('prefix_missed_n')}。审计中见到的最大全文 Jaccard：{source.get('max_full_jaccard')}。",
            f"- 有 evidence 字段 {evidence.get('with_evidence_field_n')}，其中落在学生可见摘录内 {evidence.get('evidence_inside_context_n')}，摘录外 {evidence.get('evidence_outside_context_n')}。",
            "- source_in_training_subset、eval_partition、frozen_family_split 分开记录。冻结分区没有回写。",
            "",
            "## 字段覆盖",
            "",
        ]
    )
    for check in summary.get("field_coverage") or []:
        lines.append(
            f"- {check.get('check')}：适用 {check.get('applicable_n')}，有字段 {check.get('with_required_fields_n')}，已执行 {check.get('executed_n')}。{check.get('not_executed_reason') or ''}"
        )
    reinfer = summary.get("reinfer") or {}
    lines.extend(["", "## 补推理", ""])
    lines.append(f"- 状态：{reinfer.get('status')}。使用的 GPU 候选：{reinfer.get('gpu_candidates')}。")
    if reinfer.get("reason") or reinfer.get("gpu_error"):
        lines.append(f"- 未执行原因：{reinfer.get('reason') or reinfer.get('gpu_error')}。")
    for item in reinfer.get("runs") or []:
        lines.append(
            f"- {item.get('name')}：状态 {item.get('status')}，计划 {item.get('planned_n')}，已评分 {item.get('scored_n')}，失败 {item.get('failed_n')}，"
            f"设备 {item.get('device')}，prompt tokens {item.get('prompt_tokens')}，completion tokens {item.get('completion_tokens')}，耗时 {item.get('elapsed_sec')} 秒。"
            f" prediction_run_id {item.get('prediction_run_id')}。"
            + (
                f" 基座辅助通过 {item.get('auxiliary_before_pass_n')}，adapter 辅助通过 {item.get('auxiliary_after_pass_n')}。"
                if item.get("auxiliary_after_pass_n") is not None
                else ""
            )
            + " 这不是泛化分，也不进入正式主指标。"
        )
    lines.append("- train_seen 是新的确认推理，不并入 seen_rephrase、independent 或 behavior，也不计算泛化总分。")
    lines.extend(["", "## 测试", "", "```", summary.get("test_log_excerpt") or "测试日志尚未附上。", "```", ""])
    return "\n".join(lines)


def _reinfer(repo: Path, queue: dict[str, Any], dest: Path, ids: dict[str, str]) -> dict[str, Any]:
    chosen, error = select_trainable_gpus(None)
    result: dict[str, Any] = {"gpu_candidates": chosen, "gpu_error": error, "runs": []}
    if not chosen:
        result["status"] = "not_executed"
        result["reason"] = error or "no_gpu"
        return result
    device = chosen[0]
    jobs = []
    if queue["train_seen_case_ids"]:
        jobs.append(("train_seen", dest / "train_seen_protocol.jsonl", repo / "runs/drug_v24_rescore/train_seen_confirmed"))
    if queue["new_input_cases"]:
        jobs.append(("new_behavior_inputs", dest / "new_input_protocol.jsonl", repo / "runs/drug_v24_rescore/new_inputs"))
    for name, protocol, out in jobs:
        if not protocol.is_file():
            result["runs"].append({"name": name, "status": "not_executed", "reason": "protocol_missing"})
            continue
        try:
            report = eval_adapter(
                base_model=str(BASE_MODEL),
                base_id=ids["base_id"],
                adapter=str(ADAPTER),
                adapter_id=ids["adapter_id"],
                protocol_path=protocol,
                out_dir=out,
                infer_config=INFER,
                scorer_version=SCORER_V3,
                mode="exploratory",
                template_id=ids["template_id"],
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            result["runs"].append({"name": name, "status": "not_executed", "reason": str(exc)})
            continue
        preds = _jsonl(out / "predictions.jsonl")
        result["runs"].append(
            {
                "name": name,
                "status": report.get("status"),
                "executed": report.get("executed"),
                "reason": report.get("reason"),
                "planned_n": report.get("planned_n"),
                "scored_n": report.get("scored_n"),
                "failed_n": report.get("failed_n"),
                "device": sorted({row.get("device") for row in preds}),
                "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in preds),
                "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in preds),
                "elapsed_sec": round(sum(float(row.get("elapsed_sec") or 0) for row in preds), 3),
                "infer_config_effective": report.get("infer_config_effective"),
                "prediction_run_id": report.get("prediction_run_id"),
                "out": str(out),
            }
        )
    result["status"] = "executed" if any(item.get("executed") for item in result["runs"]) else "not_executed"
    return result


def run_audit(repo: Path | None = None, test_log: str | None = None, reinfer: bool = True) -> dict[str, Any]:
    root = repo or REPO
    audit_dir = resolve_output_dir("drug_v24_audit", root)
    review_dir = resolve_output_dir("drug_v24_review", root)
    rescore_dir = resolve_output_dir("drug_v24_rescore", root)
    protected = [
        root / "runs/drug_v23_audit/manifest.json",
        root / "runs/drug_v23_audit/audit.json",
        root / "runs/drug_v22/E3_g0/sft/train.jsonl",
        root / "runs/drug_v22/E3_g0/sft/train_seen.jsonl",
        root / "runs/drug_v23_eval/candidates/dev_candidates.jsonl",
        root / "runs/drug_v23_eval/exploratory/predictions.jsonl",
        root / "runs/drug_v23_eval/exploratory/scores.jsonl",
        root / "runs/drug_v23_eval/exploratory/eval_report.json",
        root / "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl",
    ]
    before = {str(path): _sha256(path) for path in protected}
    ids = _model_ids()
    manifest = _read_json(root / "runs/drug_v23_audit/manifest.json")
    manifest_compare = []
    for item in manifest.get("files") or []:
        path = Path(item["path"])
        current = _sha256(path) if path.is_file() else None
        manifest_compare.append({"path": item["path"], "matches_v23_baseline": current == item.get("sha256"), "current_sha256": current})
    _write_json(
        audit_dir / "current_baseline.json",
        {
            "note": "这些哈希只说明本次审计开始时的文件状态，不能证明它们在 v23 基线之前没有变化。v23 manifest 没有被重写。",
            "git_commit": _git_commit(),
            "v23_manifest_comparison": manifest_compare,
            "files": [_file_record(path) for path in protected],
            "models": ids,
        },
    )
    candidates = _jsonl(root / "runs/drug_v23_eval/candidates/dev_candidates.jsonl")
    predictions = _jsonl(root / "runs/drug_v23_eval/exploratory/predictions.jsonl")
    old_scores = _jsonl(root / "runs/drug_v23_eval/exploratory/scores.jsonl")
    old_report = _read_json(root / "runs/drug_v23_eval/exploratory/eval_report.json")
    train = _jsonl(root / "runs/drug_v22/E3_g0/sft/train.jsonl")
    seen = _jsonl(root / "runs/drug_v22/E3_g0/sft/train_seen.jsonl")
    source_index = _read_json(root / "runs/drug_v23_eval/source_index.json")
    docs = _load_docs()
    identity = verify_saved_input_identity(
        candidates,
        predictions,
        template_id=str(old_report.get("template_id") or ""),
        infer_config=dict(old_report.get("infer_config") or INFER),
        base_id=str(old_report.get("base_id") or ids["base_id"]),
        adapter_id=str(old_report.get("adapter_id") or ids["adapter_id"]),
    )
    self_checks = [reference_self_check(case) for case in candidates]
    failures = [item for item in self_checks if item["passed"] is not True]
    patched_cases = []
    for case in candidates:
        patched, _change = apply_same_input_gold_patch(case)
        patched_cases.append(patched)
    patched_failures = [reference_self_check(case) for case in patched_cases if reference_self_check(case)["passed"] is not True]
    _write_json(
        audit_dir / "gold_self_check.json",
        {
            "scorer": SCORER_V3,
            "candidate_n": len(candidates),
            "failed_n": len(failures),
            "failures": failures,
            "failed_after_visible_point_patch_n": len(patched_failures),
            "failed_after_visible_point_patch": patched_failures,
        },
    )
    findings, revisions = _behavior_findings(candidates, docs)
    _write_json(audit_dir / "behavior_items.json", {"items": findings, "new_input_n": len(revisions), "review_status": "pending_review"})
    coverage = _field_coverage(candidates)
    for check in coverage["checks"]:
        if check["check"] == "visible_evidence_map":
            check["executed_n"] = sum(1 for item in findings)
            check["not_executed_reason"] = "本次只对 10 道行为题执行了可见证据审计。普通题没有证据映射字段，历史评分器没有执行该检查。"
    _write_json(audit_dir / "field_coverage.json", coverage)
    source = _source_audit(train, candidates, docs, source_index)
    _write_json(audit_dir / "source_audit.json", source)
    compared = _comparison_rows(candidates, predictions, old_scores)
    _write_jsonl(rescore_dir / "score_comparison.jsonl", compared)
    _write_jsonl(rescore_dir / "protocol_same_input.jsonl", patched_cases)
    offline = rescore_saved(
        protocol_path=rescore_dir / "protocol_same_input.jsonl",
        predictions_path=root / "runs/drug_v23_eval/exploratory/predictions.jsonl",
        out_dir=rescore_dir / "v3",
        scorer_version=SCORER_V3,
        mode="exploratory",
        base_id=str(old_report.get("base_id") or ""),
        adapter_id=str(old_report.get("adapter_id") or ""),
        template_id=str(old_report.get("template_id") or ""),
        infer_config=dict(old_report.get("infer_config") or INFER),
    )
    v2_offline = rescore_saved(
        protocol_path=root / "runs/drug_v23_eval/candidates/dev_candidates.jsonl",
        predictions_path=root / "runs/drug_v23_eval/exploratory/predictions.jsonl",
        out_dir=rescore_dir / "v2",
        scorer_version=SCORER_V2,
        mode="exploratory",
        base_id=str(old_report.get("base_id") or ""),
        adapter_id=str(old_report.get("adapter_id") or ""),
        template_id=str(old_report.get("template_id") or ""),
        infer_config=dict(old_report.get("infer_config") or INFER),
    )
    exploratory_rows = []
    for row in compared:
        exploratory_rows.append({"case_id": row["case_id"], "stratum": row["stratum"], "family_id": row["family_id"], "source_family_id": row["source_family_id"], "passed": row["v3_patched_after_passed"]})
    by_stratum = _stratum_counts(exploratory_rows)
    family_rows = []
    for row in compared:
        family_rows.append(
            {
                "stratum": row["stratum"],
                "family_id": row["family_id"],
                "source_family_id": row["source_family_id"],
                "v3_before": row["v3_patched_before_passed"],
                "v3_after": row["v3_patched_after_passed"],
            }
        )
    _write_json(rescore_dir / "strata.json", {"by_stratum": by_stratum, "combined_generalization_score": None, "rows": family_rows})
    run_meta = _read_json(root / "runs/drug_v22/run_meta.json")
    train_queue = [row.get("case_id") for row in seen]
    queue = {
        "v23_reused_n": 200 if identity["input_match"] and identity["signature_match"] else 200 - len(identity["message_mismatch"]),
        "v23_input_match": identity["input_match"],
        "v23_signature_match": identity["signature_match"],
        "v23_affected_ids": sorted({item["case_id"] for item in identity["message_mismatch"]}),
        "train_seen_case_ids": train_queue,
        "train_seen_reason": "保存了 messages 和回答，但没有逐条签名、模板哈希和生效推理参数。当时代码在 G0 路径从 adapter 目录加载 tokenizer，而 adapter 与基座的 tokenizer_config 哈希不同。运行记录 git_dirty 不能补成输入匹配证明。",
        "train_seen_input_n": len(train_queue),
        "train_seen_answer_n": len(train_queue) * 2,
        "new_input_cases": [case.get("case_id") for case in revisions],
        "new_input_n": len(revisions),
        "new_input_answer_n": len(revisions) * 2,
        "git_dirty_at_v22": run_meta.get("git_dirty"),
        "v22_commit_recorded": run_meta.get("git_commit"),
    }
    seen_protocol = []
    for row in seen:
        seen_protocol.append(
            {
                "case_id": row.get("case_id"),
                "messages": row.get("messages"),
                "question": next((item.get("content") for item in row.get("messages") or [] if item.get("role") == "user"), ""),
                "answer": row.get("answer"),
                "answer_points": [row.get("answer")] if row.get("answer") else [],
                "expected_action": "answer",
                "evidence_state": "sufficient",
                "stratum": "train_seen",
                "review_status": "unreviewed",
                "family_id": row.get("case_id"),
            }
        )
    _write_jsonl(audit_dir / "train_seen_protocol.jsonl", seen_protocol)
    _write_jsonl(audit_dir / "new_input_protocol.jsonl", revisions)
    _write_json(audit_dir / "reinfer_queue.json", queue)
    review = _review_materials(review_dir, train, seen, candidates, predictions, findings)
    reinfer_result = {"status": "not_requested"}
    if reinfer:
        reinfer_result = _reinfer(root, queue, audit_dir, ids)
        _write_json(audit_dir / "reinfer_result.json", reinfer_result)
    review.update(write_new_input_reviews(review_dir, revisions, _jsonl(rescore_dir / "new_inputs/predictions.jsonl")))
    review["audit_notes"] = [
        "新输入的辅助通过不能当成行为已经修好。v24_behavior_missing_reaction 的 adapter 回答称“未提及该药品是否为孕妇禁用”，但新上下文里有“孕妇禁用”。规则把拒答判成通过，这句话要人工看。",
    ]
    diff_counter = Counter()
    for row in compared:
        for reason in row["difference_sources"]:
            diff_counter[reason] += 1
    test_excerpt = ""
    if test_log and Path(test_log).is_file():
        text = Path(test_log).read_text(encoding="utf-8", errors="replace")
        test_excerpt = "\n".join(text.splitlines()[-30:])
    after = {str(path): _sha256(path) for path in protected}
    unchanged = before == after
    _attach_auxiliary_counts(reinfer_result, rescore_dir)
    summary = {
        "git_commit": _git_commit(),
        "completed": [
            "修复了行为分类、离线重评分身份核对、缓存分层和正式协议门禁。",
            "用已有 200 份新题回答做了 v1/v2/v3 离线对照，没有重跑 drug_v23。",
            "生成了空白金标审核包和盲评包。",
            "写了来源审计和下一阶段协议，没有启动 A–D。",
        ],
        "exploratory_auxiliary": {"by_stratum": by_stratum, "combined_generalization_score": None, "offline_v3": {"executed": offline.get("executed"), "scored_n": offline.get("scored_n"), "failed_n": offline.get("failed_n")}, "offline_v2": {"executed": v2_offline.get("executed"), "scored_n": v2_offline.get("scored_n")}},
        "review": review,
        "blocked": [],
        "not_authorized": ["固定上下文新问法对照", "成对行为边界测试", "新增独立来源难题", "多 seed / G1–G3 / 行为训练"],
        "score_differences": {
            "v2_recompute_mismatch_n": diff_counter["v2_recompute_mismatch"],
            "scoring_fix_n": diff_counter["scoring_fix"],
            "gold_structure_fix_n": diff_counter["gold_structure_fix"],
            "no_auxiliary_change_n": diff_counter["no_auxiliary_change"],
        },
        "identity": identity,
        "behavior_items": [
            {
                "case_id": item.get("case_id"),
                "disposition": item.get("disposition"),
                "issues": item.get("issues"),
                "notes": item.get("notes"),
                "new_input_case_id": item.get("new_input_case_id"),
                "self_check_passed": (item.get("self_check") or {}).get("passed"),
                "patched_self_check_passed": None if item.get("patched_self_check") is None else item["patched_self_check"].get("passed"),
            }
            for item in findings
        ],
        "source_audit": {
            "empty_family_id_n": source.get("empty_family_id_n"),
            "train_rows_unmapped_n": len(source.get("train_rows_unmapped") or []),
            "independent_case_n": source.get("independent_case_n"),
            "independent_source_family_n": source.get("independent_source_family_n"),
            "independent_file_n": source.get("independent_file_n"),
            "independent_is_ten_families": source.get("independent_is_ten_families"),
            "prefix_missed_n": len(source.get("prefix_screen_missed_full_text_near_duplicate") or []),
            "max_full_jaccard": source.get("max_full_jaccard_seen_during_audit"),
            "rule_clean_dropped_n": len(source.get("rule_clean_dropped") or []),
            "evidence_in_excerpt": source.get("evidence_in_excerpt"),
            "frozen_partition_rewritten": source.get("frozen_partition_rewritten"),
        },
        "field_coverage": coverage.get("checks"),
        "reinfer": reinfer_result,
        "historical_files_unchanged": unchanged,
        "test_log_excerpt": test_excerpt,
    }
    if not unchanged:
        summary["blocked"].append("审计结束后历史文件哈希发生变化，需要停下来检查。")
    if reinfer_result.get("status") == "not_executed":
        summary["blocked"].append(f"补推理未执行：{reinfer_result.get('reason') or reinfer_result.get('gpu_error')}")
    for item in reinfer_result.get("runs") or []:
        if not item.get("executed"):
            summary["blocked"].append(f"{item.get('name')} 未完成：{item.get('reason')}")
    _write_json(audit_dir / "summary.json", summary)
    (audit_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    (audit_dir / "next_experiments.md").write_text(_next_experiments(), encoding="utf-8")
    _write_json(audit_dir / "version.json", {"git_commit": _git_commit(), "kind": "drug_v24_audit"})
    _write_json(review_dir / "version.json", {"git_commit": _git_commit(), "kind": "drug_v24_review"})
    _write_json(rescore_dir / "version.json", {"git_commit": _git_commit(), "kind": "drug_v24_rescore"})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 drug_v24 只读审计")
    parser.add_argument("--test-log", default="")
    parser.add_argument("--skip-reinfer", action="store_true")
    args = parser.parse_args(argv)
    summary = run_audit(test_log=args.test_log or None, reinfer=not args.skip_reinfer)
    print(json.dumps({"historical_files_unchanged": summary["historical_files_unchanged"], "reinfer": summary["reinfer"].get("status"), "strata": summary["exploratory_auxiliary"]["by_stratum"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
