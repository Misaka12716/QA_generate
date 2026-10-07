"""闭卷数据冻结、训练与预测。不把缺上下文的 RAG 题直接当成闭卷题。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ..adapters.zhixun import CLOSED_POLICY
from ..task_mode import CLOSED_BOOK, messages_conflict

DEICTIC = re.compile(r"根据资料|资料中|资料里|文中|上述|该药|该药物|本品|本药|此药|该复方|该制剂|本制剂|这种药|这个药")
REWRITES = (
    ("化学名称是什么？", "化学名是什么？"),
    ("化学名称是什么？", "化学名称怎么写？"),
    ("是什么？", "具体是什么？"),
    ("是什么？", "是哪一个？"),
    ("有哪些", "包括哪些"),
    ("有哪些", "都有什么"),
    ("应采取什么措施", "应当如何处理"),
    ("应采取什么措施", "需要怎样处理"),
    ("是什么时候", "在什么时间"),
    ("来源于哪种", "来自哪一种"),
    ("包括哪些", "包含哪些"),
    ("包括哪些", "有哪几项"),
    ("可能出现的", "会出现的"),
    ("可能出现的", "常见的"),
)
TRAIN_LIMIT = 100
KNOWLEDGE_LIMIT = 10
PARAPHRASES_PER_KNOWLEDGE = 2
INFER = {"max_new_tokens": 256, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "dtype": "bfloat16"}
TRAIN_HYPER = {"epochs": 3, "lr": 2e-4, "rank": 16, "max_length": 1024, "seed": 42}


def question_issues(question: str) -> list[str]:
    text = str(question or "").strip()
    issues = []
    if not text:
        issues.append("empty_question")
    if DEICTIC.search(text):
        issues.append("needs_passage_or_deictic")
    if text.startswith("根据"):
        issues.append("passage_dependent_prefix")
    return issues


def extract_question(user: str) -> str:
    if "问题：" in user:
        return user.split("问题：", 1)[-1].strip()
    return str(user or "").strip()


def surface_paraphrases(question: str, gold: str = "") -> list[str]:
    found: list[str] = []
    for source, target in REWRITES:
        if source in question:
            variant = question.replace(source, target, 1)
            if variant != question and variant not in found:
                found.append(variant)
        if len(found) >= PARAPHRASES_PER_KNOWLEDGE:
            break
    if len(found) < PARAPHRASES_PER_KNOWLEDGE and question.endswith("？"):
        wrapped = "请直接回答：" + question
        if wrapped not in found:
            found.append(wrapped)
    if len(found) < PARAPHRASES_PER_KNOWLEDGE and "？" in question:
        swapped = question.replace("？", "呢？", 1)
        if swapped != question and swapped not in found:
            found.append(swapped)
    kept = []
    for variant in found:
        if question_issues(variant):
            continue
        if gold and len(gold) >= 8 and gold in variant:
            continue
        if variant == question:
            continue
        kept.append(variant)
    return kept[:PARAPHRASES_PER_KNOWLEDGE]


def student_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": CLOSED_POLICY},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def leak_reason(messages: list[dict[str, str]], *, question: str, gold: str = "", evidence: str = "") -> str | None:
    visible = [item for item in messages if item.get("role") != "assistant"]
    conflict = messages_conflict(visible, CLOSED_BOOK, {"question": question, "answer": gold, "context": evidence})
    if conflict:
        return conflict
    user = next((item["content"] for item in messages if item["role"] == "user"), "")
    if user != question:
        return "user_is_not_question"
    if evidence and len(evidence) >= 12 and evidence not in question and evidence in user:
        return "evidence_in_user"
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def audit_training_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    user = next((item.get("content") or "" for item in messages if item.get("role") == "user"), "")
    answer = next((item.get("content") or "" for item in messages if item.get("role") == "assistant"), "")
    question = extract_question(user)
    meta = row.get("metadata") or {}
    family = str(meta.get("source_family_id") or meta.get("family_id") or "")
    evidence = str(meta.get("evidence") or "")
    issues = question_issues(question)
    if not answer.strip():
        issues.append("empty_answer")
    if not family:
        issues.append("missing_source_family")
    closed = student_messages(question, answer)
    leaked = leak_reason(closed, question=question, gold=answer, evidence=evidence)
    if leaked:
        issues.append(leaked)
    return {
        "qa_id": row.get("id"),
        "source_family_id": family,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "source": meta.get("source"),
        "eligible": not issues,
        "issues": issues,
        "paraphrases": surface_paraphrases(question, answer) if not issues else [],
    }


def select_knowledge(eligible: list[dict[str, Any]], limit: int = KNOWLEDGE_LIMIT) -> list[dict[str, Any]]:
    usable = [row for row in eligible if len(row.get("paraphrases") or []) >= PARAPHRASES_PER_KNOWLEDGE]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(usable, key=lambda item: (item["source_family_id"], str(item["qa_id"]))):
        by_family.setdefault(row["source_family_id"], []).append(row)
    picked: list[dict[str, Any]] = []
    while len(picked) < limit:
        progressed = False
        for family in sorted(by_family):
            bucket = by_family[family]
            if not bucket:
                continue
            picked.append(bucket.pop(0))
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def _budget(calls_used: int | None, call_cap: int | None, tokens_used: int | None, token_cap: int | None, needed_calls: int) -> dict[str, Any]:
    remaining_calls = None if calls_used is None or call_cap is None else call_cap - calls_used
    remaining_tokens = None if tokens_used is None or token_cap is None else token_cap - tokens_used
    blocked = remaining_calls is None or remaining_calls < needed_calls
    return {
        "ledger_calls_used": calls_used,
        "ledger_call_cap": call_cap,
        "remaining_calls": remaining_calls,
        "remaining_tokens": remaining_tokens,
        "ideal_review_calls": needed_calls,
        "status": "blocked_teacher_budget" if blocked else "within_ledger",
        "note": "沿用已记录共享额度，不自动扩容。未知用量不按零计算。",
    }


def prepare_closed_book(
    *,
    source: str | Path,
    out_dir: str | Path,
    retention_path: str | Path,
    base_model: str,
    ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = Path(source)
    dest = Path(out_dir)
    if (dest / "adapter").exists() or (dest / "predictions").exists():
        raise FileExistsError("refuse_overwrite_closed_book_run")
    dest.mkdir(parents=True, exist_ok=True)
    audited = [audit_training_row(row) for row in _load_jsonl(source_path)]
    eligible = [row for row in audited if row["eligible"]]
    knowledge = select_knowledge(eligible)
    train_rows = eligible[:TRAIN_LIMIT]
    train_questions = {row["question"] for row in train_rows}
    paraphrase_cases = []
    for row in knowledge:
        for index, question in enumerate(row["paraphrases"], start=1):
            if question in train_questions:
                continue
            paraphrase_cases.append(
                {
                    "case_id": f"{row['qa_id']}__p{index}",
                    "eval_subset": "CB-paraphrase",
                    "knowledge_id": f"{row['source_family_id']}::{row['qa_id']}",
                    "source_family_id": row["source_family_id"],
                    "question": question,
                    "reference_answer": row["answer"],
                    "required_points": [row["answer"]],
                    "source_evidence": row["evidence"],
                    "train_seen_fact": True,
                    "train_seen_question": False,
                    "parent_qa_id": row["qa_id"],
                }
            )
    retention = []
    for row in _load_jsonl(Path(retention_path)):
        question = str(row.get("question") or "")
        if question_issues(question) or question in train_questions:
            continue
        retention.append(
            {
                "case_id": row["case_id"],
                "eval_subset": "CB-retention",
                "knowledge_id": None,
                "source_family_id": row.get("source_family_id") or "retention_general_v1",
                "question": question,
                "reference_answer": row.get("answer") or "",
                "required_points": [row["answer"]] if row.get("answer") else [],
                "source_evidence": "",
                "train_seen_fact": False,
                "train_seen_question": False,
            }
        )
    memory = [
        {
            "case_id": f"{row['qa_id']}__memory",
            "eval_subset": "CB-memory",
            "knowledge_id": f"{row['source_family_id']}::{row['qa_id']}",
            "source_family_id": row["source_family_id"],
            "question": row["question"],
            "reference_answer": row["answer"],
            "required_points": [row["answer"]],
            "source_evidence": row["evidence"],
            "train_seen_fact": True,
            "train_seen_question": True,
            "parent_qa_id": row["qa_id"],
        }
        for row in knowledge
    ]
    exported = []
    for row in train_rows:
        messages = student_messages(row["question"], row["answer"])
        reason = leak_reason(messages, question=row["question"], gold=row["answer"], evidence=row["evidence"])
        if reason:
            raise RuntimeError(reason)
        exported.append(
            {
                "id": row["qa_id"],
                "messages": messages,
                "metadata": {
                    "task_mode": CLOSED_BOOK,
                    "knowledge_id": f"{row['source_family_id']}::{row['qa_id']}",
                    "source_family_id": row["source_family_id"],
                    "train_seen_question": True,
                },
            }
        )
    _write_jsonl(dest / "data" / "train.jsonl", exported)
    _write_jsonl(dest / "data" / "audit.jsonl", audited)
    _write_jsonl(dest / "splits" / "cb_paraphrase.jsonl", paraphrase_cases)
    _write_jsonl(dest / "splits" / "cb_retention.jsonl", retention)
    _write_jsonl(dest / "splits" / "cb_memory.jsonl", memory)
    main_inputs = paraphrase_cases + retention
    needed_calls = len(main_inputs) + len(main_inputs) * 2
    ledger = ledger or {}
    budget = _budget(
        ledger.get("calls"),
        ledger.get("call_denominator"),
        ledger.get("tokens"),
        ledger.get("token_denominator"),
        needed_calls,
    )
    shortfall = []
    if len(knowledge) < KNOWLEDGE_LIMIT:
        shortfall.append(f"可改写知识单元 {len(knowledge)}，少于计划 {KNOWLEDGE_LIMIT}")
    if len(paraphrase_cases) < KNOWLEDGE_LIMIT * PARAPHRASES_PER_KNOWLEDGE:
        shortfall.append(f"CB-paraphrase {len(paraphrase_cases)}，少于计划 {KNOWLEDGE_LIMIT * PARAPHRASES_PER_KNOWLEDGE}")
    if len(retention) < 20:
        shortfall.append(f"CB-retention {len(retention)}，少于计划 20")
    protocol = {
        "task_mode": CLOSED_BOOK,
        "status": "frozen",
        "hypothesis": "同一基座上，闭卷 adapter 是否改善已训练知识的新问法，并保持冻结保留题。",
        "seed": TRAIN_HYPER["seed"],
        "base_model": base_model,
        "hyperparameters": TRAIN_HYPER,
        "infer": INFER,
        "system_policy": CLOSED_POLICY,
        "paraphrase_method": "预先固定的表面改写，不是教师生成的新问法。",
        "train_planned": len(exported),
        "train_limit": TRAIN_LIMIT,
        "knowledge_planned": KNOWLEDGE_LIMIT,
        "knowledge_selected": len(knowledge),
        "eval_planned": {
            "CB-paraphrase": len(paraphrase_cases),
            "CB-retention": len(retention),
            "CB-memory": len(memory),
            "main_inputs": len(main_inputs),
            "main_predictions": len(main_inputs) * 2,
        },
        "teacher_budget": budget,
        "blockers": ["teacher_review_not_authorized_for_full_protocol"] if budget["status"] != "within_ledger" else [],
        "shortfalls": shortfall,
        "limitations": [
            "历史样本原为 rag_grounded。只有通过脱离上下文明确性检查的题目进入闭卷训练集。",
            "旧教师接受不是闭卷 rubric 下的重新审核。",
            "CB-paraphrase 允许事实重叠，测试问法不进入训练。",
            "保留题是冻结的通用题，不是药品语料，也不是训练样本。",
            "全局来源隔离仍然适用于独立来源评测；本协议显式标记为已训练知识诊断。",
        ],
        "source_sha256": _sha256_file(source_path),
        "retention_sha256": _sha256_file(Path(retention_path)),
        "status_note": "协议已冻结。教师审核调用超出剩余额度时不执行审核。",
    }
    (dest / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    exposure = {
        "task_mode": CLOSED_BOOK,
        "knowledge": [
            {
                "knowledge_id": f"{row['source_family_id']}::{row['qa_id']}",
                "source_family_id": row["source_family_id"],
                "train_qa_id": row["qa_id"],
                "train_question_sha256": _sha256_text(row["question"]),
                "paraphrase_sha256": [_sha256_text(item) for item in row["paraphrases"]],
                "train_seen_fact": True,
            }
            for row in knowledge
        ],
        "audited_n": len(audited),
        "eligible_n": len(eligible),
        "rejected_n": len(audited) - len(eligible),
    }
    (dest / "splits" / "knowledge_exposure.json").write_text(json.dumps(exposure, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "data" / "input_audit.json").write_text(
        json.dumps(
            {
                "train_messages_checked": len(exported),
                "leak_failures": 0,
                "policy": CLOSED_POLICY,
                "input_mode_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return protocol


def train_closed_book(out_dir: str | Path, *, device: int | None = None) -> dict[str, Any]:
    dest = Path(out_dir)
    protocol = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from .sft import train_lora

    hyper = protocol["hyperparameters"]
    metrics = train_lora(
        dest / "data" / "train.jsonl",
        dest / "sft",
        base_model=protocol["base_model"],
        epochs=hyper["epochs"],
        lr=hyper["lr"],
        rank=hyper["rank"],
        max_length=hyper["max_length"],
        seed=hyper["seed"],
    )
    return metrics


def _load_eval_rows(dest: Path) -> list[dict[str, Any]]:
    rows = []
    for name in ("cb_paraphrase.jsonl", "cb_retention.jsonl", "cb_memory.jsonl"):
        rows.extend(_load_jsonl(dest / "splits" / name))
    return rows


def predict_closed_book(out_dir: str | Path, *, device: int | None = None) -> dict[str, Any]:
    dest = Path(out_dir)
    protocol = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    adapter = dest / "sft" / "adapter"
    if not adapter.is_dir():
        return {"status": "not_executed", "reason": "closed_book_adapter_missing", "fallback": "refused"}
    rows = _load_eval_rows(dest)
    pending = []
    for row in rows:
        messages = student_messages(row["question"])
        reason = leak_reason(
            messages,
            question=row["question"],
            gold=str(row.get("reference_answer") or ""),
            evidence=str(row.get("source_evidence") or ""),
        )
        if reason:
            return {"status": "blocked", "reason": reason, "case_id": row.get("case_id")}
        for role in ("base", "adapter"):
            pending.append({"case_id": row["case_id"], "model_role": role, "messages": messages})
    from .adapter_eval import generate_with_model

    generated = generate_with_model(
        pending,
        base_model=protocol["base_model"],
        adapter=str(adapter),
        infer_config=protocol["infer"],
        device=device,
        persist_dir=dest / "predictions",
    )
    if isinstance(generated, dict):
        (dest / "predictions" / "error.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "not_executed", **generated}
    prediction_dir = dest / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for role in ("base", "adapter"):
        target = prediction_dir / f"{role}.jsonl"
        if target.exists():
            target.unlink()
    by_key = {}
    for spec, result in zip(pending, generated):
        by_key[(spec["case_id"], spec["model_role"])] = result
        target = dest / "predictions" / f"{spec['model_role']}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"case_id": spec["case_id"], "model_role": spec["model_role"], "messages": spec["messages"], **result}, ensure_ascii=False) + "\n")
    cases = []
    for row in rows:
        base = by_key.get((row["case_id"], "base")) or {}
        adapter_row = by_key.get((row["case_id"], "adapter")) or {}
        cases.append(
            {
                "case_id": row["case_id"],
                "question": row["question"],
                "task_mode": CLOSED_BOOK,
                "eval_subset": row["eval_subset"],
                "knowledge_id": row.get("knowledge_id"),
                "source_family_id": row.get("source_family_id"),
                "train_seen_fact": row.get("train_seen_fact"),
                "train_seen_question": row.get("train_seen_question"),
                "input_mode_verified": True,
                "relation": "unresolved",
                "labels": ["unresolved"],
                "base": {"text": base.get("text") or "", "status": "generated" if base.get("text") else "missing", "label": "教师未审核"},
                "adapter": {"text": adapter_row.get("text") or "", "status": "generated" if adapter_row.get("text") else "missing", "label": "教师未审核"},
                "judge_source_context": row.get("source_evidence") or "",
                "reference_answer": row.get("reference_answer") or "",
                "review_reason": "教师审核未执行。剩余共享调用不足以覆盖本协议，未决不是内容失败，也不是没有改善。",
                "rubric_version": None,
                "review_source": "not_executed",
            }
        )
    _write_jsonl(dest / "cases.jsonl", cases)
    main = [row for row in cases if row["eval_subset"] in {"CB-paraphrase", "CB-retention"}]
    complete = [row for row in main if row["base"]["text"] and row["adapter"]["text"]]
    planned_inputs = len(main)
    metrics = _prediction_metrics(protocol, cases, planned_inputs, len(complete))
    (dest / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _prediction_metrics(protocol: dict[str, Any], cases: list[dict[str, Any]], planned_inputs: int, complete_pairs: int) -> dict[str, Any]:
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        by_subset.setdefault(row["eval_subset"], []).append(row)
    subsets = {}
    for name, rows in by_subset.items():
        done = sum(1 for row in rows if row["base"]["text"] and row["adapter"]["text"])
        subsets[name] = {
            "planned_inputs": len(rows),
            "complete_pairs": done,
            "wrong_to_right": None,
            "right_to_wrong": None,
            "both_correct": None,
            "both_wrong": None,
            "unresolved": done,
            "status": "teacher_not_executed",
        }
    limitations = list(protocol.get("limitations") or [])
    limitations.append("教师审核未执行。下面的未决是缺少裁判，不是模型没有变化。")
    limitations.append("词面差异不代表事实对错。")
    return {
        "task_mode": CLOSED_BOOK,
        "status": "predicted",
        "availability": "partial",
        "default_subset": "CB-paraphrase",
        "headline": (
            f"闭卷主评测计划 {planned_inputs} 个输入，已生成完整配对 {complete_pairs}。"
            "教师审核未执行，不能判断改善或没有改善。"
        ),
        "tags": ["教师审核未执行", "探索性", "闭卷", "单种子"],
        "cards": [
            {"name": "计划输入", "numerator": planned_inputs, "denominator": planned_inputs, "unit": "input", "status": "available"},
            {"name": "完整预测配对", "numerator": complete_pairs, "denominator": planned_inputs, "unit": "pair", "status": "partial"},
            {"name": "教师有效配对", "numerator": None, "denominator": planned_inputs, "unit": "pair", "status": "not_executed"},
            {"name": "未决", "numerator": complete_pairs, "denominator": planned_inputs, "unit": "pair", "status": "teacher_not_executed"},
        ],
        "bars": [
            {
                "id": name,
                "label": name,
                "numerator": len(rows),
                "denominator": planned_inputs or len(cases),
                "unit": "计划输入",
                "filter": name,
            }
            for name, rows in by_subset.items()
            if name != "CB-memory"
        ],
        "scale_denominator": planned_inputs,
        "outcomes": {
            "wrong_to_right": None,
            "right_to_wrong": None,
            "both_correct": None,
            "both_wrong": None,
            "unresolved": complete_pairs,
            "status": "teacher_not_executed",
            "reason": "没有教师审核，不能把未决写成两者错误或没有改善",
        },
        "subsets": subsets,
        "limitations": limitations,
        "review_source": "not_executed",
        "rubric_version": None,
        "formal_ready": 0,
    }
