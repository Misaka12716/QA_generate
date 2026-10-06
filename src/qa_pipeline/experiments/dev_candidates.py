"""drug_v23 审核材料和未审核开发候选。不覆盖正式协议，不填写人工裁定。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..textutil import char_ngram_jaccard, heading_sections, rouge_l
from .adapter_eval import EXPLORATORY_DISCLAIMER, assert_candidate_destination, write_jsonl
from .drug_corpus import _family_key, extract_identity, file_sha256, split_for_family
from .frozen_loader import load_frozen_subset

_QTY = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|μg|ug|ml|mmol|kg|%)", re.I)
TEACHER_BUDGET = {
    "temperature": 0.2,
    "max_tokens": 700,
    "max_calls": 48,
    "max_retries": 1,
    "max_consecutive_transport_failures": 2,
    "model": "qwen3.8-27b",
}
_SHEET_KEYS = (
    "case_id",
    "stratum",
    "family_id",
    "source_family_id",
    "source",
    "source_hash",
    "question",
    "context",
    "answer",
    "answer_points",
    "evidence",
    "expected_action",
    "evidence_state",
    "review_status",
    "authoring",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _user_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def index_subset_sources(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for doc in docs:
        identity = extract_identity(doc["text"])
        family = _family_key(identity, "source")
        filename = Path(doc["path"]).stem
        indexed[filename] = {
            **doc,
            "identity": identity,
            "source_family_id": family,
            "generic_family_id": _family_key(identity, "generic"),
            "generic_name": identity.get("generic_name") or "",
            "split": split_for_family(family),
            "filename": filename,
        }
    return indexed


def select_core_questions(train_rows: list[dict[str, Any]], limit: int = 30) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_family = []
    for row in train_rows:
        family = str((row.get("metadata") or {}).get("family_id") or "")
        if not family:
            missing_family.append(row.get("id"))
            continue
        grouped[family].append(row)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for family, rows in grouped.items():
        meta = rows[0].get("metadata") or {}
        answer = _assistant_text(rows[0])
        if _QTY.search(answer):
            risk = "numeric"
        elif re.search(r"不得|禁用|尚未|未进行|不是", answer):
            risk = "negation"
        else:
            risk = "plain"
        buckets[(str(meta.get("q_type") or "factual"), risk)].append(family)
    for families in buckets.values():
        families.sort()
    picked: list[str] = []
    order = sorted(buckets)
    while len(picked) < limit and any(buckets.values()):
        for key in order:
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].pop(0))
    return {
        "family_ids": picked,
        "available_core_questions": len(grouped),
        "requested": limit,
        "selected": len(picked),
        "shortfall": max(0, limit - len(picked)),
        "missing_family_ids": missing_family,
        "reason": None if len(picked) == limit else "distinct_core_questions_below_request",
        "seeds": {family: grouped[family][0] for family in picked},
    }


def _quantities(text: str) -> set[str]:
    return {re.sub(r"\s+", "", item) for item in _QTY.findall(text or "")}


def _accept_paraphrases(original: str, proposal: Any) -> tuple[list[str], str | None]:
    items = proposal.get("paraphrases") if isinstance(proposal, dict) else None
    if not isinstance(items, list):
        return [], "schema"
    cleaned = []
    original_qty = _quantities(original)
    for item in items:
        text = str(item or "").strip()
        if not text or text == original or rouge_l(text, original) >= 0.95:
            continue
        if _quantities(text) - original_qty:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) == 2:
            break
    if len(cleaned) < 2:
        return [], "not_enough_distinct_paraphrases"
    return cleaned[:2], None


def _section(text: str, title: str) -> str:
    for path, body in heading_sections(text):
        if path and path[-1] == title and body.strip():
            return body.strip()
    return ""


def _behavior(pattern: str, doc: dict[str, Any], **fields: Any) -> dict[str, Any]:
    extra = fields.pop("extra", {})
    points = fields.pop("points", None)
    return {
        "case_id": f"v23_behavior_{pattern}",
        "stratum": "behavior",
        "family_id": f"bfam_{pattern}",
        "source_family_id": doc.get("source_family_id") or "",
        "generic_family_id": doc.get("generic_family_id") or "",
        "source_hash": doc.get("content_sha256") or "",
        "source": doc.get("filename") or doc.get("stem") or "",
        "dataset_version": "drug_v23_dev_unreviewed",
        "split": "dev",
        "task_mode": "grounded",
        "review_status": "unreviewed",
        "authoring": "evidence_transform",
        "pattern": pattern,
        "answer_points": points or [fields.get("answer") or ""],
        "semantic_gold_verified": False,
        **fields,
        **extra,
    }


def build_behavior_cases(docs_by_file: dict[str, dict[str, Any]], limit: int = 10) -> dict[str, Any]:
    """只用冻结正文构造行为题。凑不齐的模式记为缺失，不编造药学事实。"""
    docs = [doc for doc in docs_by_file.values() if doc.get("text")]
    cases: list[dict[str, Any]] = []
    missing: list[str] = []

    def add(case: dict[str, Any]) -> None:
        if len(cases) < limit:
            cases.append(case)

    usage_doc = next((doc for doc in docs if _section(doc["text"], "用法用量") and _section(doc["text"], "贮藏")), None)
    if usage_doc is None:
        missing.append("insufficient_section")
    else:
        add(
            _behavior(
                "insufficient_section",
                usage_doc,
                question=f"{usage_doc.get('generic_name') or '该药品'}的用法用量是什么？",
                context=_section(usage_doc["text"], "贮藏"),
                answer="当前可见资料只有贮藏信息，不能确定用法用量。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    named = [doc for doc in docs if doc.get("generic_name")]
    pair = None
    for left in named:
        for right in named:
            if left["generic_name"] != right["generic_name"]:
                pair = (left, right)
                break
        if pair:
            break
    if pair is None:
        missing.extend(["wrong_object", "replaced_context", "retained_distractor"])
    else:
        left, right = pair
        left_usage = _section(left["text"], "用法用量") or left["text"][:400]
        right_body = _section(right["text"], "成分") or _section(right["text"], "贮藏") or right["text"][:400]
        fact = left_usage.split("。")[0].strip()
        add(
            _behavior(
                "wrong_object",
                right,
                question=f"{left['generic_name']}的用法用量是什么？",
                context=right_body,
                answer="所给资料的药品与问题对象不一致，不能据此回答。",
                expected_action="state_insufficient",
                evidence_state="missing",
                extra={"asked_generic": left["generic_name"], "shown_generic": right["generic_name"]},
            )
        )
        add(
            _behavior(
                "retained_distractor",
                left,
                question=f"根据资料，{left['generic_name']}的用法用量要点是什么？",
                context=left_usage + "\n" + right_body,
                answer=fact,
                expected_action="answer",
                evidence_state="sufficient",
                points=[fact],
            )
        )
        add(
            _behavior(
                "replaced_context",
                right,
                question=f"根据资料，{left['generic_name']}的用法用量要点是什么？",
                context=right_body,
                answer="所给资料不能支持该问题，不能改用另一份药品资料里的说法。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    clarify_doc = None
    for doc in docs:
        strengths = sorted(set(re.findall(r"\d+(?:\.\d+)?\s*(?:mg|g|ml|μg)", doc["text"], flags=re.I)))
        if len(strengths) >= 2:
            clarify_doc = (doc, strengths)
            break
    if clarify_doc is None:
        missing.append("clarify_strength")
    else:
        doc, strengths = clarify_doc
        add(
            _behavior(
                "clarify_strength",
                doc,
                question="这个规格应该怎么使用？",
                context=doc["text"][:800],
                answer="资料里出现多个规格，需要先说明是哪一个规格。",
                expected_action="clarify",
                evidence_state="ambiguous",
                extra={"strengths": strengths[:6]},
            )
        )
    condition_doc = None
    condition_sentence = ""
    for doc in docs:
        for sentence in re.split(r"(?<=。)", doc["text"]):
            if re.search(r"(如果|若|当)", sentence) and len(sentence.strip()) > 12:
                condition_doc = doc
                condition_sentence = sentence.strip()
                break
        if condition_doc:
            break
    if condition_doc is None:
        missing.append("missing_condition")
    else:
        visible = condition_doc["text"].replace(condition_sentence, "").strip()[:800]
        add(
            _behavior(
                "missing_condition",
                condition_doc,
                question="在该条件成立时应当如何处理？",
                context=visible or "（未提供包含该条件的资料）",
                answer="当前资料没有给出该条件对应的处理，不能确定。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    reaction_doc = next((doc for doc in docs if _section(doc["text"], "禁忌")), None)
    if reaction_doc is None:
        missing.append("missing_reaction")
    else:
        add(
            _behavior(
                "missing_reaction",
                reaction_doc,
                question=f"{reaction_doc.get('generic_name') or '该药品'}有哪些不良反应？",
                context=_section(reaction_doc["text"], "禁忌"),
                answer="当前资料只给出禁忌，没有把不良反应作为可见资料，不能确定。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    child_doc = next(
        (
            doc
            for doc in docs
            if "成人" in _section(doc["text"], "用法用量") and "儿童" not in _section(doc["text"], "用法用量")
        ),
        None,
    )
    if child_doc is None:
        missing.append("child_dose_absent")
    else:
        add(
            _behavior(
                "child_dose_absent",
                child_doc,
                question=f"{child_doc.get('generic_name') or '该药品'}用于儿童时的剂量是多少？",
                context=_section(child_doc["text"], "用法用量"),
                answer="当前用法用量资料没有给出儿童剂量，不能确定。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    contra_doc = next((doc for doc in docs if _section(doc["text"], "不良反应") and _section(doc["text"], "禁忌")), None)
    if contra_doc is None:
        missing.append("contra_not_visible")
    else:
        add(
            _behavior(
                "contra_not_visible",
                contra_doc,
                question=f"{contra_doc.get('generic_name') or '该药品'}有哪些禁忌？",
                context=_section(contra_doc["text"], "不良反应"),
                answer="当前可见资料是不良反应，没有给出禁忌，不能确定。",
                expected_action="state_insufficient",
                evidence_state="missing",
            )
        )
    partial_doc = None
    for doc in docs:
        usage = _section(doc["text"], "用法用量")
        parts = [part.strip() for part in usage.split("。") if len(part.strip()) > 8]
        if len(parts) >= 2:
            partial_doc = (doc, parts[0], parts[1])
            break
    if partial_doc is None:
        missing.append("partial_usage")
    else:
        doc, first, second = partial_doc
        add(
            _behavior(
                "partial_usage",
                doc,
                question=f"{doc.get('generic_name') or '该药品'}的用法用量包含哪些要点？",
                context=first + "。",
                answer=first,
                expected_action="partial_answer",
                evidence_state="partial",
                points=[first, second],
            )
        )
    return {"cases": cases[:limit], "missing_patterns": missing, "constructed_n": len(cases[:limit]), "requested": limit}


def select_independent_sources(
    raw_dir: Path,
    train_docs: dict[str, dict[str, Any]],
    limit_families: int = 10,
    max_scanned: int = 400,
) -> dict[str, Any]:
    """开发分区里未进入训练、且与训练文本不构成近重复的来源。通用名是否未见单独计数。"""
    train_families = {doc["source_family_id"] for doc in train_docs.values()}
    train_generics = {doc["generic_name"] for doc in train_docs.values() if doc.get("generic_name")}
    train_hashes = {
        hashlib.sha256(re.sub(r"\s+", "", doc["text"]).encode("utf-8")).hexdigest() for doc in train_docs.values()
    }
    train_prefixes = [doc["text"][:4000] for doc in train_docs.values()]
    train_stems = {doc["stem"] for doc in train_docs.values()}
    chosen = []
    scanned = 0
    seen_generic_skipped = 0
    near_rejected = 0
    for path in sorted(raw_dir.glob("*.txt")):
        if len(chosen) >= limit_families or scanned >= max_scanned:
            break
        if path.stem in train_stems:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        identity = extract_identity(text)
        family = _family_key(identity, "source")
        if split_for_family(family) != "dev":
            continue
        scanned += 1
        generic_name = identity.get("generic_name") or ""
        if family in train_families:
            continue
        exact = hashlib.sha256(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()
        if exact in train_hashes or any(char_ngram_jaccard(text[:4000], prefix) >= 0.92 for prefix in train_prefixes):
            near_rejected += 1
            continue
        if not generic_name or generic_name in train_generics:
            seen_generic_skipped += 1
            continue
        chosen.append(
            {
                "stem": path.stem,
                "path": str(path),
                "text": text,
                "source_family_id": family,
                "generic_family_id": _family_key(identity, "generic"),
                "generic_name": generic_name,
                "source_hash": file_sha256(path),
                "split": "dev",
                "unseen_generic_name": True,
                "identity": identity,
            }
        )
    return {
        "sources": chosen,
        "scanned_dev_files": scanned,
        "selected_source_families": len(chosen),
        "requested_source_families": limit_families,
        "shortfall": max(0, limit_families - len(chosen)),
        "near_or_exact_rejected": near_rejected,
        "seen_generic_skipped": seen_generic_skipped,
        "unseen_generic_name_count": len(chosen),
        "independent_source_count": len(chosen),
        "near_duplicate_screen": "前 4000 字 5-gram Jaccard ≥ 0.92，或全文去空白哈希相同。",
        "scan_cap": max_scanned,
        "definition": {
            "independent_source": "source_family_id 不在训练子集，split 为 dev，且未与训练文本精确重复或近重复。",
            "unseen_generic_name": "通用名不在训练子集抽出的通用名中。二者分开计数。",
        },
    }


def _teacher_json(llm, messages: list[dict[str, str]], ledger: dict[str, Any]) -> tuple[dict | None, str | None]:
    if llm.calls >= TEACHER_BUDGET["max_calls"]:
        ledger["stopped_reason"] = "call_cap"
        return None, "call_cap"
    last_error = "empty"
    for _ in range(TEACHER_BUDGET["max_retries"] + 1):
        if llm.calls >= TEACHER_BUDGET["max_calls"]:
            ledger["stopped_reason"] = "call_cap"
            return None, "call_cap"
        response = llm.chat_json(
            messages,
            temperature=TEACHER_BUDGET["temperature"],
            max_tokens=TEACHER_BUDGET["max_tokens"],
        )
        ledger["calls"] = llm.calls
        ledger["prompt_tokens"] = llm.prompt_tokens
        ledger["completion_tokens"] = llm.completion_tokens
        if getattr(response, "status", "") == "ok" and isinstance(response.data, dict):
            ledger["consecutive_transport_failures"] = 0
            return response.data, None
        last_error = getattr(response, "status", "") or "failed"
        ledger["retries"] = int(ledger.get("retries") or 0) + 1
        if last_error == "transport_failed":
            ledger["consecutive_transport_failures"] = int(ledger.get("consecutive_transport_failures") or 0) + 1
            if ledger["consecutive_transport_failures"] >= TEACHER_BUDGET["max_consecutive_transport_failures"]:
                ledger["stopped_reason"] = "teacher_unreachable"
                return None, "teacher_unreachable"
        else:
            ledger["consecutive_transport_failures"] = 0
    return None, last_error


def _blank_ledger(llm) -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retries": 0,
        "consecutive_transport_failures": 0,
        "stopped_reason": None,
        "temperature": TEACHER_BUDGET["temperature"],
        "max_tokens": TEACHER_BUDGET["max_tokens"],
        "max_calls": TEACHER_BUDGET["max_calls"],
        "model": getattr(llm, "default_model", TEACHER_BUDGET["model"]),
    }


def generate_rephrase_cases(seeds: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]], llm) -> dict[str, Any]:
    ledger = _blank_ledger(llm)
    cases = []
    rejected = []
    for family, row in seeds.items():
        if ledger.get("stopped_reason"):
            break
        meta = row.get("metadata") or {}
        original = _user_text(row)
        question = original.split("问题：")[-1].strip() if "问题：" in original else original
        answer = _assistant_text(row)
        source_key = str(meta.get("source") or "")
        source = sources.get(source_key) or {}
        data, error = _teacher_json(
            llm,
            [
                {
                    "role": "system",
                    "content": (
                        "只改写问法。保持任务、必要条件、数字、单位和正确答案不变。"
                        "不要新增事实，不要输出答案。"
                        '输出 JSON：{"paraphrases":["...", "..."]}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"原问题：{question}\n标准答案：{answer}\n请给两条自然的新问法，不要只替换一两个词。",
                },
            ],
            ledger,
        )
        if error == "teacher_unreachable":
            break
        paraphrases, reason = _accept_paraphrases(question, data or {})
        if reason:
            rejected.append({"family_id": family, "reason": reason or error})
            continue
        for index, text in enumerate(paraphrases, start=1):
            cases.append(
                {
                    "case_id": f"v23_rephrase_{family}_{index}",
                    "stratum": "seen_rephrase",
                    "family_id": family,
                    "source_family_id": source.get("source_family_id") or "",
                    "generic_family_id": source.get("generic_family_id") or "",
                    "source_hash": source.get("content_sha256") or "",
                    "source": source_key,
                    "dataset_version": "drug_v23_dev_unreviewed",
                    "split": "dev",
                    "task_mode": "grounded",
                    "evidence_state": meta.get("evidence_state") or "sufficient",
                    "expected_action": meta.get("expected_action") or "answer",
                    "question": text,
                    "context": source.get("text") or "",
                    "answer": answer,
                    "answer_points": [answer],
                    "parent_id": row.get("id"),
                    "review_status": "unreviewed",
                    "authoring": "teacher_paraphrase",
                    "semantic_gold_verified": False,
                }
            )
    return {"cases": cases, "rejected": rejected, "ledger": ledger}


def generate_independent_cases(sources: list[dict[str, Any]], llm, ledger: dict[str, Any]) -> dict[str, Any]:
    cases = []
    rejected = []
    for source in sources:
        if ledger.get("stopped_reason"):
            break
        excerpt = source["text"][:1800]
        data, error = _teacher_json(
            llm,
            [
                {
                    "role": "system",
                    "content": (
                        "只根据给定说明书出题。答案和证据引文必须是摘录中的连续原文，不要编造。"
                        "最多三个问题，摘录不支持就少出。"
                        '输出 JSON：{"items":[{"question":"","answer":"","evidence_quote":"","focus":"factual"}]}'
                    ),
                },
                {"role": "user", "content": f"说明书摘录：\n{excerpt}"},
            ],
            ledger,
        )
        if error == "teacher_unreachable":
            break
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            rejected.append({"source": source["stem"], "reason": error or "schema"})
            continue
        kept = 0
        for item in items:
            if not isinstance(item, dict) or kept >= 3:
                continue
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            quote = str(item.get("evidence_quote") or "").strip()
            if not question or not answer or not quote:
                continue
            if quote not in source["text"] or answer not in quote:
                continue
            kept += 1
            cases.append(
                {
                    "case_id": f"v23_indep_{source['stem']}_{kept}",
                    "stratum": "independent",
                    "family_id": f"ifam_{source['stem']}_{kept}",
                    "source_family_id": source["source_family_id"],
                    "generic_family_id": source["generic_family_id"],
                    "source_hash": source["source_hash"],
                    "source": source["stem"],
                    "dataset_version": "drug_v23_dev_unreviewed",
                    "split": "dev",
                    "task_mode": "grounded",
                    "evidence_state": "sufficient",
                    "expected_action": "answer",
                    "question": question,
                    "context": excerpt,
                    "answer": answer,
                    "answer_points": [answer],
                    "evidence": quote,
                    "focus": item.get("focus") or "",
                    "review_status": "unreviewed",
                    "authoring": "teacher_from_evidence",
                    "unseen_generic_name": True,
                    "semantic_gold_verified": False,
                }
            )
        if kept < 3:
            rejected.append({"source": source["stem"], "reason": "verified_items_below_3", "kept": kept})
    return {"cases": cases, "rejected": rejected}


def write_candidate_file(path: str | Path, cases: list[dict[str, Any]], protected: list[str | Path] | None = None) -> None:
    assert_candidate_destination(path, protected)
    write_jsonl(Path(path), cases)


def write_review_pack(
    train_rows: list[dict[str, Any]],
    seen_rows: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    out_dir: Path,
    seed: int = 20261006,
) -> dict[str, Any]:
    """审核者材料不暴露模型身份。映射单独保存。不填写人工意见。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_by_id = {row["case_id"]: row for row in seen_rows}
    kept_by_id = {row.get("qa_id"): row for row in kept_rows}
    rng = random.Random(seed)
    packet = []
    mapping = []
    reproduction = []
    for row in train_rows:
        case_id = row["id"]
        seen = seen_by_id.get(case_id) or {}
        kept = kept_by_id.get(case_id) or {}
        meta = row.get("metadata") or {}
        source_key = str(meta.get("source") or "")
        source = sources.get(source_key) or {}
        base = str(seen.get("base_answer") or "")
        tuned = str(seen.get("tuned_answer") or "")
        flip = rng.random() < 0.5
        packet.append(
            {
                "case_id": case_id,
                "source_file": source.get("path"),
                "source_hash": source.get("content_sha256"),
                "source_family_id": source.get("source_family_id"),
                "generic_name": source.get("generic_name"),
                "evidence": kept.get("evidence_span") or meta.get("evidence"),
                "evidence_location": meta.get("location"),
                "original_question": kept.get("question"),
                "training_input": _user_text(row),
                "training_target": _assistant_text(row),
                "required_points": kept.get("answer_points") or [_assistant_text(row)],
                "answers": {"回答甲": tuned if flip else base, "回答乙": base if flip else tuned},
                "reviewer_id": None,
                "gold_correct": None,
                "answer_judgement": {"回答甲": None, "回答乙": None},
                "adjudication": None,
            }
        )
        mapping.append({"case_id": case_id, "回答甲": "g0" if flip else "base", "回答乙": "base" if flip else "g0"})
        reproduction.append(
            {
                "case_id": case_id,
                "training_input": _user_text(row),
                "training_target": _assistant_text(row),
                "note": "这是训练原题复现，不是人工纠正后的语义金标。",
            }
        )
    write_jsonl(out_dir / "packet.jsonl", packet)
    write_jsonl(out_dir / "train_reproduction.jsonl", reproduction)
    (out_dir / "analysis_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 训练原题审核包",
        "",
        "请只根据来源、问题和训练目标判断。两份回答已匿名并随机排列。",
        "人工意见栏留空，不能由程序代填。",
        "",
    ]
    for item in packet:
        lines.extend(
            [
                f"## {item['case_id']}",
                "",
                f"- 来源文件：{item.get('source_file')}",
                f"- 来源哈希：{item.get('source_hash')}",
                "",
                "### 证据",
                str(item.get("evidence") or ""),
                "",
                "### 原问题",
                str(item.get("original_question") or ""),
                "",
                "### 实际训练输入",
                item["training_input"],
                "",
                "### 实际训练目标",
                item["training_target"],
                "",
                "### 回答甲",
                item["answers"]["回答甲"],
                "",
                "### 回答乙",
                item["answers"]["回答乙"],
                "",
                "### 审核",
                "- 审核者：",
                "- 训练目标是否正确：",
                "- 回答甲判断：",
                "- 回答乙判断：",
                "- 裁定：",
                "",
            ]
        )
    (out_dir / "packet.md").write_text("\n".join(lines), encoding="utf-8")
    return {"questions": len(packet), "answers": len(packet) * 2, "review_status": "unreviewed", "filled_adjudications": 0}


def prepare_materials(
    *,
    repo: Path,
    train_path: Path,
    seen_path: Path,
    kept_path: Path,
    subset_dir: Path,
    manifest_path: Path,
    raw_dir: Path,
    out_dir: Path,
    llm=None,
    protected: list[Path] | None = None,
) -> dict[str, Any]:
    docs = load_frozen_subset(subset_dir, manifest_path)
    sources = index_subset_sources(docs)
    train_rows = _read_jsonl(train_path)
    review = write_review_pack(train_rows, _read_jsonl(seen_path), _read_jsonl(kept_path), sources, out_dir / "review")
    core = select_core_questions(train_rows, 30)
    behavior = build_behavior_cases(sources, 10)
    independent = select_independent_sources(raw_dir, sources, 10)
    if llm is None:
        rephrase = {"cases": [], "rejected": [], "ledger": {"stopped_reason": "teacher_not_configured", "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}
        indep_cases = {"cases": [], "rejected": []}
    else:
        rephrase = generate_rephrase_cases(core["seeds"], sources, llm)
        indep_cases = generate_independent_cases(independent["sources"], llm, rephrase["ledger"])
    cases = rephrase["cases"] + indep_cases["cases"] + behavior["cases"]
    dest = out_dir / "candidates" / "dev_candidates.jsonl"
    guards = [
        repo / "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl",
        repo / "data/campus_hospital_drug_instructions/frozen/heldout.jsonl",
        *(protected or []),
    ]
    write_candidate_file(dest, cases, protected=guards)
    sheet = []
    for case in cases:
        item = {key: case.get(key) for key in _SHEET_KEYS}
        item["reviewer_id"] = None
        item["gold_accept"] = None
        item["model_output_hidden"] = True
        sheet.append(item)
    write_jsonl(out_dir / "review" / "new_items_for_review.jsonl", sheet)
    summary = {
        "disclaimer": EXPLORATORY_DISCLAIMER,
        "review_pack": review,
        "core_questions": {key: value for key, value in core.items() if key != "seeds"},
        "rephrase_n": len(rephrase["cases"]),
        "rephrase_rejected": rephrase["rejected"],
        "independent_n": len(indep_cases["cases"]),
        "independent_rejected": indep_cases["rejected"],
        "independent_sources": {key: value for key, value in independent.items() if key != "sources"},
        "behavior_n": len(behavior["cases"]),
        "behavior_missing_patterns": behavior["missing_patterns"],
        "candidate_n": len(cases),
        "requested_n": 100,
        "shortfall": 100 - len(cases),
        "review_status": "unreviewed",
        "teacher_ledger": rephrase["ledger"],
        "candidate_path": str(dest),
        "formal_protocol_written": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "teacher_ledger.json").write_text(json.dumps(rephrase["ledger"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "source_index.json").write_text(
        json.dumps(
            [
                {
                    "filename": doc["filename"],
                    "stem": doc["stem"],
                    "source_family_id": doc["source_family_id"],
                    "generic_family_id": doc["generic_family_id"],
                    "generic_name": doc["generic_name"],
                    "split": doc["split"],
                    "content_sha256": doc["content_sha256"],
                }
                for doc in sources.values()
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
