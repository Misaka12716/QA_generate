"""金标与盲评表的导出、导入和校验。

空白单元格保持 pending_review。这里不代填审核者姓名、意见或裁定。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .adapter_eval import canonical_messages, case_id_of, review_content_hash
from .scoring import cache_signature

GOLD_COLUMNS = (
    "batch_id",
    "case_id",
    "condition_id",
    "core_question_id",
    "content_hash",
    "question_clear",
    "semantic_preserved",
    "visible_evidence_adequate",
    "required_points_ok",
    "unavailable_points_ok",
    "expected_behavior_ok",
    "answerable_part",
    "insufficient_part",
    "reviewer_a",
    "opinion_a",
    "reviewer_b",
    "opinion_b",
    "adjudication",
    "ai_note",
)
BLIND_COLUMNS = (
    "batch_id",
    "blind_id",
    "content_hash",
    "answer_text",
    "task_completed",
    "key_factual_errors",
    "unsupported_content",
    "behavior_appropriate",
    "reviewer_a",
    "opinion_a",
    "reviewer_b",
    "opinion_b",
    "adjudication",
    "ai_note",
)
READONLY_GOLD = ("batch_id", "case_id", "condition_id", "core_question_id", "content_hash")
READONLY_BLIND = ("batch_id", "blind_id", "content_hash", "answer_text")
YES_NO = {"yes", "no"}
YES_NO_NA = {"yes", "no", "na"}
ADJUDICATIONS = {"accept", "revise", "reject"}
PENDING_TOKENS = {"", "pending_review"}
MACHINE_REVIEWERS = {"ai", "llm", "model", "assistant", "模型", "助手"}


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _cell(value) for key, value in row.items() if key} for row in reader]


def csv_has_human_input(path: Path) -> bool:
    if not path.is_file():
        return False
    for row in read_csv(path):
        for key in ("reviewer_a", "reviewer_b", "opinion_a", "opinion_b", "adjudication"):
            if _cell(row.get(key)) not in PENDING_TOKENS:
                return True
    return False


def blind_content_hash(blind_id: str, answer_text: str) -> str:
    return cache_signature({"blind_id": blind_id, "answer_text": answer_text})


def gold_row_from_case(case: dict[str, Any], *, batch_id: str, ai_note: str = "") -> dict[str, str]:
    return {
        "batch_id": batch_id,
        "case_id": case_id_of(case),
        "condition_id": str(case.get("condition_id") or ""),
        "core_question_id": str(case.get("core_question_id") or case.get("family_id") or ""),
        "content_hash": review_content_hash(case),
        "question_clear": "",
        "semantic_preserved": "",
        "visible_evidence_adequate": "",
        "required_points_ok": "",
        "unavailable_points_ok": "",
        "expected_behavior_ok": "",
        "answerable_part": "",
        "insufficient_part": "",
        "reviewer_a": "",
        "opinion_a": "",
        "reviewer_b": "",
        "opinion_b": "",
        "adjudication": "",
        "ai_note": ai_note,
    }


def render_gold_markdown(cases: list[dict[str, Any]], *, batch_id: str) -> str:
    lines = [
        f"# {batch_id} 金标材料",
        "",
        "每题给出完整 system 与 user，供对照 CSV。模型身份不在这里出现。",
        "ai_note 只是准备阶段的提示，不是审核意见。",
        "",
    ]
    for case in cases:
        messages = canonical_messages(case)
        lines.append(f"## {case_id_of(case)} / {case.get('condition_id') or '-'}")
        lines.append("")
        lines.append(f"- content_hash: `{review_content_hash(case)}`")
        lines.append(f"- core_question_id: `{case.get('core_question_id') or case.get('family_id') or ''}`")
        lines.append(f"- expected_action: `{case.get('expected_action') or ''}`")
        lines.append(f"- evidence_state: `{case.get('evidence_state') or ''}`")
        lines.append("")
        lines.append("### 实际输入")
        lines.append("")
        for message in messages:
            lines.append(f"**{message.get('role')}**")
            lines.append("")
            lines.append("```text")
            lines.append(str(message.get("content") or ""))
            lines.append("```")
            lines.append("")
        lines.append("### 当前金标")
        lines.append("")
        lines.append(f"- answer: {case.get('answer') or ''}")
        lines.append(f"- required_points: {json.dumps(case.get('required_points') or [], ensure_ascii=False)}")
        lines.append(f"- unavailable_points: {json.dumps(case.get('unavailable_points') or [], ensure_ascii=False)}")
        lines.append(f"- answerable_part: {case.get('answerable_part') or ''}")
        lines.append(f"- insufficient_part: {case.get('insufficient_part') or ''}")
        lines.append("")
    return "\n".join(lines)


def render_blind_markdown(rows: list[dict[str, str]], *, batch_id: str) -> str:
    lines = [
        f"# {batch_id} 盲评材料",
        "",
        "只看到盲 ID 和完整回答。不要打开 identity map，以免看到模型身份。",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['blind_id']}")
        lines.append("")
        lines.append(f"- content_hash: `{row['content_hash']}`")
        lines.append("")
        lines.append("```text")
        lines.append(row.get("answer_text") or "")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _machine_name(name: str) -> bool:
    return name.casefold() in MACHINE_REVIEWERS


def _review_decision(row: dict[str, str], *, kind: str) -> dict[str, Any]:
    """把一行表格收成审核状态。空值和机器名都保持 pending_review。"""
    reasons: list[str] = []
    adjudication = _cell(row.get("adjudication"))
    if adjudication in PENDING_TOKENS:
        adjudication = ""
    elif adjudication not in ADJUDICATIONS:
        reasons.append("invalid_adjudication")
        adjudication = ""
    reviewer_a = _cell(row.get("reviewer_a"))
    reviewer_b = _cell(row.get("reviewer_b"))
    opinion_a = _cell(row.get("opinion_a"))
    opinion_b = _cell(row.get("opinion_b"))
    if _machine_name(reviewer_a):
        reasons.append("machine_reviewer")
        reviewer_a = ""
    if _machine_name(reviewer_b):
        reasons.append("machine_reviewer")
        reviewer_b = ""
    checks = ("question_clear", "visible_evidence_adequate", "required_points_ok", "expected_behavior_ok")
    blind_checks = ("task_completed", "behavior_appropriate")
    required_checks = checks if kind == "gold" else blind_checks
    allowed = {"question_clear": YES_NO, "visible_evidence_adequate": {"yes", "no", "partial"}, "required_points_ok": YES_NO, "expected_behavior_ok": YES_NO, "semantic_preserved": YES_NO_NA, "unavailable_points_ok": YES_NO_NA, "task_completed": YES_NO, "behavior_appropriate": YES_NO}
    filled_checks = True
    for key in required_checks:
        value = _cell(row.get(key))
        if not value:
            filled_checks = False
        elif value not in allowed[key]:
            reasons.append(f"invalid_{key}")
            filled_checks = False
    for key in ("semantic_preserved", "unavailable_points_ok"):
        if kind != "gold":
            continue
        value = _cell(row.get(key))
        if not value:
            filled_checks = False
        elif value not in YES_NO_NA:
            reasons.append(f"invalid_{key}")
            filled_checks = False
    dual_bits = [reviewer_b, opinion_b]
    if any(dual_bits) and not all(dual_bits):
        reasons.append("incomplete_dual_review")
    if reviewer_a and reviewer_b and reviewer_a == reviewer_b:
        reasons.append("same_reviewer")
    single_ready = bool(reviewer_a and opinion_a and adjudication and filled_checks)
    dual_ready = single_ready and bool(reviewer_b and opinion_b and reviewer_a != reviewer_b)
    if reasons or not single_ready:
        return {
            "review_status": "pending_review",
            "adjudication": adjudication or "pending_review",
            "reviewer_a": reviewer_a,
            "reviewer_b": reviewer_b,
            "opinion_a": opinion_a,
            "opinion_b": opinion_b,
            "reasons": reasons,
            "accepted": False,
        }
    return {
        "review_status": "dual_agreed" if dual_ready else "agreed",
        "adjudication": adjudication,
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b if dual_ready else "",
        "opinion_a": opinion_a,
        "opinion_b": opinion_b if dual_ready else "",
        "reasons": [],
        "accepted": adjudication == "accept",
    }


def import_gold_csv(csv_path: Path, cases: list[dict[str, Any]], *, batch_id: str) -> dict[str, Any]:
    rows = read_csv(csv_path) if csv_path.is_file() else []
    by_id = {case_id_of(case): case for case in cases}
    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    imported: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    for row in rows:
        case_id = _cell(row.get("case_id"))
        if row.get("batch_id") and row.get("batch_id") != batch_id:
            errors.append({"case_id": case_id, "reasons": ["batch_mismatch"]})
            continue
        case = by_id.get(case_id)
        if case is None:
            errors.append({"case_id": case_id, "reasons": ["unknown_id"]})
            continue
        if case_id in seen:
            errors.append({"case_id": case_id, "reasons": ["duplicate_id"]})
            continue
        seen.add(case_id)
        expected = {
            "batch_id": batch_id,
            "case_id": case_id,
            "condition_id": str(case.get("condition_id") or ""),
            "core_question_id": str(case.get("core_question_id") or case.get("family_id") or ""),
            "content_hash": review_content_hash(case),
        }
        mismatched = [key for key in READONLY_GOLD if _cell(row.get(key)) != expected[key]]
        if mismatched:
            errors.append({"case_id": case_id, "reasons": ["content_mismatch"], "fields": mismatched})
            cloned = dict(case)
            cloned["review_status"] = "pending_review"
            cloned["adjudication"] = "pending_review"
            updated.append(cloned)
            continue
        decision = _review_decision(row, kind="gold")
        record = {**expected, **decision, "answerable_part": _cell(row.get("answerable_part")), "insufficient_part": _cell(row.get("insufficient_part"))}
        imported.append(record)
        cloned = dict(case)
        cloned["review_status"] = decision["review_status"]
        cloned["reviewer_a"] = decision["reviewer_a"]
        cloned["reviewer_b"] = decision["reviewer_b"]
        cloned["opinion_a"] = decision["opinion_a"]
        cloned["opinion_b"] = decision["opinion_b"]
        cloned["adjudication"] = decision["adjudication"]
        cloned["answerable_part"] = record["answerable_part"]
        cloned["insufficient_part"] = record["insufficient_part"]
        if decision["review_status"] in {"agreed", "dual_agreed"}:
            cloned["review_content_hash"] = review_content_hash(cloned)
        updated.append(cloned)
    missing = [case_id for case_id in by_id if case_id not in seen]
    for case_id in missing:
        errors.append({"case_id": case_id, "reasons": ["missing_row"]})
        cloned = dict(by_id[case_id])
        cloned["review_status"] = "pending_review"
        cloned["adjudication"] = "pending_review"
        updated.append(cloned)
    structural = [item for item in errors if "missing_row" not in item["reasons"]]
    adjudicated = sum(1 for row in imported if row["review_status"] in {"agreed", "dual_agreed"})
    accepted = sum(1 for row in imported if row["accepted"])
    pending = len(cases) - adjudicated
    ready = not structural and accepted == len(cases) and pending == 0 and len(cases) > 0
    return {
        "batch_id": batch_id,
        "kind": "gold",
        "planned_n": len(cases),
        "pending_review_n": pending,
        "adjudicated_n": adjudicated,
        "accepted_n": accepted,
        "error_n": len(structural),
        "ready_for_inference": ready,
        "rows": imported,
        "errors": errors,
        "cases": updated,
    }


def import_blind_csv(csv_path: Path, identity_rows: list[dict[str, Any]], *, batch_id: str) -> dict[str, Any]:
    rows = read_csv(csv_path) if csv_path.is_file() else []
    by_id = {str(item["blind_id"]): item for item in identity_rows}
    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    imported: list[dict[str, Any]] = []
    for row in rows:
        blind_id = _cell(row.get("blind_id"))
        if any(key in row and _cell(row.get(key)) for key in ("model_role", "model_id", "adapter_id")):
            errors.append({"blind_id": blind_id, "reasons": ["identity_leaked"]})
            continue
        item = by_id.get(blind_id)
        if item is None:
            errors.append({"blind_id": blind_id, "reasons": ["unknown_id"]})
            continue
        if blind_id in seen:
            errors.append({"blind_id": blind_id, "reasons": ["duplicate_id"]})
            continue
        seen.add(blind_id)
        expected_hash = blind_content_hash(blind_id, str(item.get("answer_text") or ""))
        if _cell(row.get("content_hash")) != expected_hash or _cell(row.get("answer_text")) != str(item.get("answer_text") or ""):
            errors.append({"blind_id": blind_id, "reasons": ["content_mismatch"]})
            continue
        if _cell(row.get("batch_id")) not in {"", batch_id}:
            errors.append({"blind_id": blind_id, "reasons": ["batch_mismatch"]})
            continue
        decision = _review_decision(row, kind="blind")
        imported.append({"blind_id": blind_id, "case_id": item.get("case_id"), **decision, "task_completed": _cell(row.get("task_completed")), "behavior_appropriate": _cell(row.get("behavior_appropriate")), "key_factual_errors": _cell(row.get("key_factual_errors")), "unsupported_content": _cell(row.get("unsupported_content"))})
    missing = [blind_id for blind_id in by_id if blind_id not in seen]
    for blind_id in missing:
        errors.append({"blind_id": blind_id, "reasons": ["missing_row"]})
    structural = [item for item in errors if "missing_row" not in item["reasons"]]
    adjudicated = sum(1 for row in imported if row["review_status"] in {"agreed", "dual_agreed"})
    pending = len(identity_rows) - adjudicated
    return {
        "batch_id": batch_id,
        "kind": "blind",
        "planned_n": len(identity_rows),
        "pending_review_n": pending,
        "adjudicated_n": adjudicated,
        "accepted_n": sum(1 for row in imported if row["accepted"]),
        "error_n": len(structural),
        "ready_for_inference": False,
        "rows": imported,
        "errors": errors,
    }


def write_fill_example(directory: Path) -> Path:
    """不含真实药品内容的填写示例。不要把这个文件导入正式批次。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fill_example.csv"
    write_csv(
        path,
        GOLD_COLUMNS,
        [
            {
                "batch_id": "example",
                "case_id": "ex_case_001",
                "condition_id": "original",
                "core_question_id": "ex_core_001",
                "content_hash": "examplehash",
                "question_clear": "yes",
                "semantic_preserved": "na",
                "visible_evidence_adequate": "yes",
                "required_points_ok": "yes",
                "unavailable_points_ok": "na",
                "expected_behavior_ok": "yes",
                "answerable_part": "示例物品A的颜色",
                "insufficient_part": "",
                "reviewer_a": "示例审核者甲",
                "opinion_a": "题干清楚，可见句子能支持颜色这一点。",
                "reviewer_b": "",
                "opinion_b": "",
                "adjudication": "accept",
                "ai_note": "这是示例提示，导入时忽略，不能当成裁定。",
            },
            {
                "batch_id": "example",
                "case_id": "ex_case_002",
                "condition_id": "removed_support",
                "core_question_id": "ex_core_001",
                "content_hash": "examplehash2",
                "question_clear": "",
                "semantic_preserved": "",
                "visible_evidence_adequate": "",
                "required_points_ok": "",
                "unavailable_points_ok": "",
                "expected_behavior_ok": "",
                "answerable_part": "",
                "insufficient_part": "",
                "reviewer_a": "",
                "opinion_a": "",
                "reviewer_b": "",
                "opinion_b": "",
                "adjudication": "",
                "ai_note": "空行必须保持 pending_review，不能默认通过。",
            },
        ],
    )
    guide = directory / "fill_example.md"
    guide.write_text(
        "\n".join(
            [
                "# 填写示例",
                "",
                "示例使用虚构的“示例物品A”，不要抄进正式审核表。",
                "",
                "- `question_clear`：题干是否清楚，填 yes 或 no。",
                "- `semantic_preserved`：改写是否仍问同一件事。原题可填 na。",
                "- `visible_evidence_adequate`：学生可见的完整输入是否够用，填 yes、no 或 partial。",
                "- `required_points_ok` / `unavailable_points_ok`：必答要点和不可回答部分是否合适。",
                "- `expected_behavior_ok`：预期行为是否合适。有充分证据时，拒答不能算正确。",
                "- `answerable_part` / `insufficient_part`：部分支持时，写明可以回答的部分和应说明不足的部分。",
                "- `reviewer_a`、`opinion_a`：第一位审核者的姓名和意见。留空就是待审核。",
                "- `reviewer_b`、`opinion_b`：只有第二位真人填写后才可能成为双人一致。",
                "- `adjudication`：accept、revise 或 reject。不要填 dual_agreed，程序不会把单人审核写成双人一致。",
                "- `ai_note`：准备材料时的提示。导入时忽略，不能代替姓名或裁定。",
                "",
                "盲评表另有 `task_completed`、`key_factual_errors`、`unsupported_content`、`behavior_appropriate`。盲评材料里没有模型身份。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def import_run_reviews(run_dir: str | Path, batch: str | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = root / "review" / "batches.json"
    if not manifest_path.is_file():
        return {"error_n": 1, "errors": [{"reasons": ["batches_missing"]}], "batches": [], "ready_for_inference": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    ready: dict[str, bool] = {}
    for spec in manifest.get("batches") or []:
        name = str(spec.get("batch_id") or "")
        if batch and name != batch:
            continue
        kind = spec.get("kind")
        if kind == "gold":
            protocol = root / spec["protocol"]
            cases = []
            if protocol.is_file():
                cases = [json.loads(line) for line in protocol.read_text(encoding="utf-8").splitlines() if line.strip()]
            result = import_gold_csv(root / spec["csv"], cases, batch_id=name)
            reviewed = root / spec.get("reviewed_protocol", f"review/{name}.reviewed.jsonl")
            reviewed.parent.mkdir(parents=True, exist_ok=True)
            body = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in result["cases"])
            reviewed.write_text(body, encoding="utf-8")
            ready[name] = bool(result["ready_for_inference"])
            public = {key: value for key, value in result.items() if key != "cases"}
            results.append(public)
        elif kind == "blind":
            identity_path = root / spec["identity_map"]
            identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else []
            result = import_blind_csv(root / spec["csv"], identity, batch_id=name)
            results.append(result)
    payload = {
        "batches": results,
        "error_n": sum(int(item.get("error_n") or 0) for item in results),
        "pending_review_n": sum(int(item.get("pending_review_n") or 0) for item in results),
        "adjudicated_n": sum(int(item.get("adjudicated_n") or 0) for item in results),
        "ready_for_inference": ready,
    }
    out = root / "review" / "import_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
