"""准备 drug_v25 的审核包、实验 A/B 候选和进度报告。

不训练、不调用教师、不覆盖 v22–v24 产物。金标未全部接受前不加载模型。
"""

from __future__ import annotations

import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..textutil import heading_sections, rouge_l
from .adapter_eval import canonical_messages, case_id_of, review_content_hash
from .drug_corpus import _family_key, extract_identity, split_for_family
from .review_io import (
    BLIND_COLUMNS,
    GOLD_COLUMNS,
    blind_content_hash,
    csv_has_human_input,
    gold_row_from_case,
    import_run_reviews,
    render_blind_markdown,
    render_gold_markdown,
    write_csv,
    write_fill_example,
)
from .scoring import SCORER_V3, prediction_item_signature, reference_self_check, score_task_v3

REPO = Path(__file__).resolve().parents[3]
PROTECTED_RUNS = frozenset(
    {
        "drug_v22",
        "drug_v23_eval",
        "drug_v23_audit",
        "drug_v24_audit",
        "drug_v24_review",
        "drug_v24_rescore",
    }
)
BLIND_SEED = 20261007
SELECTION_SEED = 20261007
EFFECTIVE_INFER = {"max_new_tokens": 512, "do_sample": False, "dtype": "bfloat16"}
QUESTION_MARKER = "\n\n问题："
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9*+\-]{1,}")
_DIGIT = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = ("不得", "禁用", "禁止", "尚未", "不是")
PREFERRED_SECTIONS = ("用法用量", "功能主治", "适应症", "禁忌", "成分", "规格", "贮藏", "性状", "注意事项")

PARAPHRASES: dict[str, dict[str, str]] = {
    "qfam_4c3584eb34de5a4fd850": {
        "original_question": "资料指出，HLA-B*1502等位基因阳性患者使用卡马西平治疗的限制条件是什么？",
        "rephrase_1": "HLA-B*1502等位基因阳性的患者，使用卡马西平治疗时有什么限制？",
        "rephrase_2": "对HLA-B*1502等位基因阳性患者，卡马西平治疗受到哪些限制？",
    },
    "qfam_af536f77618a221e6c5c": {
        "original_question": "根据资料，克林霉素磷酸酯禁止用于哪些人群的肌肉注射？",
        "rephrase_1": "克林霉素磷酸酯的肌肉注射禁止用于哪些人群？",
        "rephrase_2": "哪些人群被禁止进行克林霉素磷酸酯肌肉注射？",
    },
    "qfam_06f1d57c827febba8713": {
        "original_question": "资料指出克林霉素磷酸酯与哪些药物存在交叉耐药性，且对哪些药物有过敏史者禁用？",
        "rephrase_1": "克林霉素磷酸酯会与哪些药物发生交叉耐药？曾经对哪些药物过敏的人应当禁用？",
        "rephrase_2": "若某人有过敏史，涉及哪些药物时要禁用克林霉素磷酸酯，这些药物又与它如何交叉耐药？",
    },
    "qfam_4a07d41cc70725c94d35": {
        "original_question": "根据资料，该药物的起始剂量和一般推荐的最大常规剂量分别是多少？",
        "rephrase_1": "该药物的起始剂量是多少，一般推荐的最大常规剂量又是多少？",
        "rephrase_2": "请分别说明该药物开始时的剂量，以及一般推荐的最大常规剂量。",
    },
    "qfam_05d2b860495eb4b2498b": {
        "original_question": "地鳖虫药材在烫死后的干燥方式有哪些？",
        "rephrase_1": "地鳖虫药材烫死以后，可以用哪些方式干燥？",
        "rephrase_2": "烫死处理之后，地鳖虫药材有哪些干燥方式？",
    },
    "qfam_170c384f47a15a5dab24": {
        "original_question": "资料中描述的麦粒发芽干燥工艺包括哪些主要步骤？",
        "rephrase_1": "麦粒发芽并干燥的工艺，主要包括哪些步骤？",
        "rephrase_2": "把麦粒加工成发芽干燥品时，主要经过哪些步骤？",
    },
    "qfam_51588c86e590ebd3a5c8": {
        "original_question": "资料中小儿用药的剂量计算标准是什么？",
        "rephrase_1": "小儿用药时，剂量按什么标准计算？",
        "rephrase_2": "给小儿计算用药剂量时，采用的标准是什么？",
    },
    "qfam_06413022d6593d5ec41b": {
        "original_question": "根据资料，该药物目前广泛用于哪类感染的治疗？",
        "rephrase_1": "该药物目前广泛治疗的是哪一类感染？",
        "rephrase_2": "从资料看，这种药物被广泛用于治疗哪类感染？",
    },
    "qfam_389c8722eafd45152f67": {
        "original_question": "密蒙花采收后需要进行哪些处理步骤？",
        "rephrase_1": "密蒙花采下来之后，还要进行哪些处理？",
        "rephrase_2": "采收密蒙花以后，后续处理包括哪些步骤？",
    },
    "qfam_8b212f7f510fce598a4a": {
        "original_question": "该复方制剂每片含有哪些成分及其具体含量？",
        "rephrase_1": "这种复方制剂每一片含哪些成分，各自的含量是多少？",
        "rephrase_2": "请列出该复方制剂单片的成分和对应含量。",
    },
    "qfam_0940f383386d5a774f22": {
        "original_question": "该药品的辅料中包含哪些成分？",
        "rephrase_1": "这种药品的辅料由哪些成分组成？",
        "rephrase_2": "该药品的辅料里包含什么成分？",
    },
    "qfam_3eb5cf6ccc7447e5c06b": {
        "original_question": "在给予维生素K1注射液期间，若患者出现过敏症状，应采取什么措施？",
        "rephrase_1": "使用维生素K1注射液时，患者一旦出现过敏症状，应当怎么处理？",
        "rephrase_2": "给予维生素K1注射液期间出现过敏，需要采取什么措施？",
    },
    "qfam_a33ae357ea7e4b7b878a": {
        "original_question": "资料中提到的该药物平均剂量是多少？",
        "rephrase_1": "该药物的平均剂量是多少？",
        "rephrase_2": "资料给出的这种药物，平均剂量为多少？",
    },
    "qfam_10ebcff3daa3436ee830": {
        "original_question": "资料中列出的该药品辅料包括哪些物质？",
        "rephrase_1": "该药品的辅料包括哪些物质？",
        "rephrase_2": "资料里列出的辅料物质有哪些？",
    },
    "qfam_4ba1a7f4306537c2e009": {
        "original_question": "密蒙花应在什么季节及植物生长阶段进行采收？",
        "rephrase_1": "采收密蒙花应选在什么季节，以及植物的哪个生长阶段？",
        "rephrase_2": "密蒙花的采收季节和对应植物生长阶段是什么？",
    },
    "qfam_a583925f36f3e6dbab19": {
        "original_question": "根据资料，口服给药时若出现便秘，可合并服用什么药物？",
        "rephrase_1": "口服给药后如果出现便秘，可以合并服用什么药物？",
        "rephrase_2": "口服用药期间发生便秘时，允许合并使用哪些药物？",
    },
    "qfam_11f639fe67ebfcac5d0c": {
        "original_question": "资料指出磷酸奥司他韦治疗乙型流感时存在什么局限性？",
        "rephrase_1": "用磷酸奥司他韦治疗乙型流感时，存在什么局限？",
        "rephrase_2": "磷酸奥司他韦用于乙型流感，其应用上的局限性是什么？",
    },
    "qfam_73264e36deffd34615cb": {
        "original_question": "若怀疑或已确诊为艰难梭菌相关性腹泻（CDAD），资料建议采取哪些处理措施？",
        "rephrase_1": "怀疑或已经确诊艰难梭菌相关性腹泻（CDAD）时，建议如何处理？",
        "rephrase_2": "对疑似或确诊的艰难梭菌相关性腹泻（CDAD），应采取哪些处理措施？",
    },
    "qfam_cd8487cdc4abee1282ab": {
        "original_question": "二甲双胍相关乳酸酸中毒的实验室异常表现包括哪些指标？",
        "rephrase_1": "发生二甲双胍相关乳酸酸中毒时，实验室有哪些异常指标？",
        "rephrase_2": "二甲双胍相关乳酸酸中毒在化验结果上有哪些异常表现？",
    },
    "qfam_172279245ea13b6a7d29": {
        "original_question": "地鳖虫药材的基原昆虫包括哪两种？",
        "rephrase_1": "地鳖虫药材来自哪两种基原昆虫？",
        "rephrase_2": "作为地鳖虫药材来源的基原昆虫有哪两种？",
    },
    "qfam_83532e640135f10909bc": {
        "original_question": "根据资料，莎草Cyperus rotundus L.的干燥根茎在采挖后首先需要进行什么处理？",
        "rephrase_1": "莎草Cyperus rotundus L.的干燥根茎采挖之后，首先要做什么处理？",
        "rephrase_2": "采挖莎草Cyperus rotundus L.的干燥根茎后，最先进行的处理是什么？",
    },
    "qfam_db6eb43eea683e665072": {
        "original_question": "资料中规定直肠给药时，高位保留灌肠的混匀介质是什么？",
        "rephrase_1": "直肠给药做高位保留灌肠时，用什么介质混匀？",
        "rephrase_2": "高位保留灌肠这种直肠给药方式，规定的混匀介质是什么？",
    },
    "qfam_1960f19815ff6c491677": {
        "original_question": "匹伐他汀钙的化学名称是什么？",
        "rephrase_1": "匹伐他汀钙对应的化学名称如何书写？",
        "rephrase_2": "请写出匹伐他汀钙的化学名称。",
    },
    "qfam_8adea35e80915295cf6f": {
        "original_question": "如果怀疑发生二甲双胍相关乳酸酸中毒，应采取什么措施？",
        "rephrase_1": "怀疑出现二甲双胍相关乳酸酸中毒时，应当怎样处理？",
        "rephrase_2": "一旦怀疑已经发生二甲双胍相关乳酸酸中毒，需要采取什么措施？",
    },
    "qfam_1979019076df32c41cd7": {
        "original_question": "资料中提到的本品是由哪种植物的成熟果实经发芽干燥制成的？",
        "rephrase_1": "本品使用的成熟果实来自哪种植物，并经过发芽干燥制成？",
        "rephrase_2": "这种经发芽干燥制成的成熟果实，来源于哪一种植物？",
    },
    "qfam_8d68c98425dc06adfb2e": {
        "original_question": "莎草Cyperus rotundus L.的干燥根茎在秋季采挖后，经过燎去毛须处理，后续有哪些具体的加工方式？",
        "rephrase_1": "秋季采挖莎草Cyperus rotundus L.的干燥根茎并燎去毛须之后，后续怎样加工？",
        "rephrase_2": "莎草Cyperus rotundus L.的干燥根茎在秋季采挖、燎去毛须之后，还有哪些加工方式？",
    },
    "qfam_1d348122735cf5b827fd": {
        "original_question": "资料中描述的狭叶番泻表面颜色、毛被情况及质地特征有哪些？",
        "rephrase_1": "狭叶番泻的表面是什么颜色，毛被情况和质地有什么特征？",
        "rephrase_2": "请说明狭叶番泻在表面颜色、毛被和质地方面的特征。",
    },
    "qfam_d1cb1bd3eb9d5a2f83b5": {
        "original_question": "根据资料，在开始卡马西平治疗前，针对遗传风险人群患者建议进行什么筛查？",
        "rephrase_1": "对遗传风险人群，开始卡马西平治疗之前建议做哪项筛查？",
        "rephrase_2": "遗传风险人群在启用卡马西平之前，建议先进行什么筛查？",
    },
    "qfam_343d9bfa5fe239f56bb9": {
        "original_question": "资料指出本品对大鼠胆汁分泌有什么影响？",
        "rephrase_1": "本品会怎样影响大鼠的胆汁分泌？",
        "rephrase_2": "大鼠的胆汁分泌在使用本品后有什么变化？",
    },
    "qfam_df01ffd7ec0032507b93": {
        "original_question": "根据资料，磷酸奥司他韦用于甲型和乙型流感治疗时，患者应在何时开始使用？",
        "rephrase_1": "磷酸奥司他韦治疗甲型流感和乙型流感时，患者应在什么时候开始使用？",
        "rephrase_2": "治疗甲型或乙型流感时，开始使用磷酸奥司他韦的时间应当是何时？",
    },
}


def assert_v25_destination(dest: Path) -> None:
    if dest.name in PROTECTED_RUNS or any(part in PROTECTED_RUNS for part in dest.parts):
        raise RuntimeError("refuse_overwrite_historical_run")


def paraphrase_problems(original: str, rewritten: str) -> list[str]:
    reasons = []
    if not rewritten or rewritten == original:
        reasons.append("unchanged")
    if rouge_l(rewritten, original) >= 0.95:
        reasons.append("too_similar")
    if set(_DIGIT.findall(rewritten)) != set(_DIGIT.findall(original)):
        reasons.append("digit_changed")
    for token in _NEGATION:
        if (token in original) != (token in rewritten):
            reasons.append(f"negation:{token}")
    for token in set(_LATIN.findall(original)):
        if token not in rewritten:
            reasons.append(f"missing_token:{token}")
    return reasons


def split_train_user(user: str) -> tuple[str, str]:
    if QUESTION_MARKER not in user:
        raise ValueError("train_user_missing_question_marker")
    context, question = user.rsplit(QUESTION_MARKER, 1)
    return context, question


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(text)
        return [item for item in payload if isinstance(item, dict)]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixed_context_cases(
    train_rows: list[dict[str, Any]],
    family_ids: list[str],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        family = str((row.get("metadata") or {}).get("family_id") or "")
        if family and family not in by_family:
            by_family[family] = row
    cases: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for family in family_ids:
        row = by_family.get(family)
        draft = PARAPHRASES.get(family)
        if row is None or draft is None:
            blocked.append({"family_id": family, "reason": "missing_seed"})
            continue
        messages = list(row.get("messages") or [])
        system = next((item.get("content") for item in messages if item.get("role") == "system"), "")
        user = next((item.get("content") for item in messages if item.get("role") == "user"), "")
        answer = next((item.get("content") for item in messages if item.get("role") == "assistant"), "")
        try:
            context_prefix, question = split_train_user(str(user))
        except ValueError:
            blocked.append({"family_id": family, "reason": "question_marker_missing"})
            continue
        if question != draft["original_question"]:
            blocked.append({"family_id": family, "reason": "question_drift"})
            continue
        meta = row.get("metadata") or {}
        source = sources.get(str(meta.get("source") or ""), {})
        base = {
            "core_question_id": family,
            "family_id": family,
            "parent_case_id": row.get("id"),
            "source": meta.get("source"),
            "source_family_id": source.get("source_family_id") or "",
            "generic_family_id": source.get("generic_family_id") or "",
            "source_hash": source.get("content_sha256") or "",
            "split": source.get("split") or "train",
            "stratum": "fixed_context_rephrase",
            "dataset_version": "drug_v25_experiment_a",
            "gold_version": "train_assistant_v22",
            "task_mode": "grounded",
            "expected_action": meta.get("expected_action") or "answer",
            "evidence_state": meta.get("evidence_state") or "sufficient",
            "answer": answer,
            "answer_points": [answer] if answer else [],
            "required_points": [answer] if answer else [],
            "unavailable_points": [],
            "review_status": "pending_review",
            "source_overlap_train": True,
            "authoring": "fixed_context_draft",
            "system": system,
        }
        variants = [("original", question, row.get("id"))]
        for label in ("rephrase_1", "rephrase_2"):
            problems = paraphrase_problems(question, draft[label])
            if problems:
                blocked.append({"family_id": family, "condition_id": label, "reason": ",".join(problems)})
                continue
            variants.append((label, draft[label], f"{row.get('id')}__{label}"))
        for label, text, case_id in variants:
            prompt = [
                _message("system", str(system)),
                _message("user", f"{context_prefix}{QUESTION_MARKER}{text}"),
            ]
            context = context_prefix.split("资料：\n", 1)[-1]
            cases.append(
                {
                    **base,
                    "case_id": case_id,
                    "condition_id": label,
                    "question": text,
                    "context": context,
                    "messages": prompt,
                }
            )
    return {"cases": cases, "blocked": blocked, "planned_n": len(family_ids) * 3, "built_n": len(cases)}


def match_saved_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    base_id: str,
    adapter_id: str,
    template_id: str,
    infer_effective: dict[str, Any],
) -> dict[str, Any]:
    """只接受输入、基座、adapter、模板和生效推理参数都一致的预测。"""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions:
        indexed[(str(row.get("case_id") or ""), str(row.get("model_role") or ""))] = row
    decisions = []
    reused = 0
    for case in cases:
        if case.get("condition_id") != "original":
            continue
        messages = canonical_messages(case)
        for role, expected_adapter in (("base", ""), ("adapter", adapter_id)):
            saved = indexed.get((case_id_of(case), role))
            reasons = []
            if saved is None:
                reasons.append("missing_prediction")
            else:
                if saved.get("messages") != messages:
                    reasons.append("message_mismatch")
                if saved.get("model_id") != base_id:
                    reasons.append("model_id_mismatch")
                if role == "adapter" and saved.get("adapter_id") != expected_adapter:
                    reasons.append("adapter_id_mismatch")
                if saved.get("template_id") != template_id:
                    reasons.append("template_id_mismatch")
                if saved.get("infer_config") != infer_effective:
                    reasons.append("infer_mismatch")
                signature = prediction_item_signature(
                    case, base_id, adapter_id, template_id, infer_effective, role, model_input=messages
                )
                if saved.get("signature") != signature:
                    reasons.append("signature_mismatch")
            ok = not reasons
            if ok:
                reused += 1
            decisions.append(
                {
                    "case_id": case_id_of(case),
                    "core_question_id": case.get("core_question_id"),
                    "model_role": role,
                    "reused": ok,
                    "reasons": reasons or ["reused"],
                    "source": "drug_v24_rescore/train_seen_confirmed",
                }
            )
    return {
        "checked_n": len(decisions),
        "reused_n": reused,
        "rejected_n": len(decisions) - reused,
        "v22_train_seen_answers": "not_reused_missing_prediction_identity",
        "decisions": decisions,
    }


def _clean_name(text: str) -> str:
    for line in text.splitlines():
        if "通用名称" in line and ("：" in line or ":" in line):
            raw = line.split("：", 1)[-1].split(":", 1)[-1]
            name = raw.split("英文名称")[0].split("汉语拼音")[0].strip()
            if 2 <= len(name) <= 40 and name in text:
                return name
    identity = extract_identity(text)
    name = str(identity.get("generic_name") or "")
    name = name.split("英文名称")[0].split("汉语拼音")[0].strip("：: ")
    if 2 <= len(name) <= 40 and name in text:
        return name
    return ""


def _support_sentence(text: str, name: str) -> dict[str, str] | None:
    sections = [(path[-1], body) for path, body in heading_sections(text) if path]
    ordered = sorted(sections, key=lambda item: (0 if item[0] in PREFERRED_SECTIONS else 1, PREFERRED_SECTIONS.index(item[0]) if item[0] in PREFERRED_SECTIONS else 99))
    for title, body in ordered:
        for piece in re.split(r"(?<=。)", body):
            sentence = piece.strip()
            if not (8 <= len(sentence) <= 160) or sentence.count("。") != 1:
                continue
            body_sentences = [part.strip() for part in re.split(r"(?<=。)", body) if part.strip()]
            if body_sentences != [sentence]:
                continue
            if name in sentence or text.count(sentence) != 1:
                continue
            if any(token in sentence for token in ("资料不足", "无法确定", "URL", "http")):
                continue
            others = []
            for other_title, other_body in sections:
                if other_body == body:
                    continue
                if sentence in other_body:
                    others.append(sentence)
                for part in re.split(r"(?<=。)", other_body):
                    part = part.strip()
                    if part and part != sentence:
                        others.append(part)
            if any(rouge_l(sentence, other) >= 0.85 for other in others[:80]):
                continue
            return {"section": title, "sentence": sentence}
    return None


def _removal_case(text: str, sentence: str) -> dict[str, Any]:
    remaining = text.replace(sentence, "", 1)
    clauses = [item.strip() for item in re.split(r"[，,；;]", sentence) if len(item.strip()) >= 6]
    still = [item for item in clauses if item in remaining]
    gone = [item for item in clauses if item not in remaining]
    if sentence in remaining or not gone:
        return {"ok": False, "reason": "equivalent_support_remains"}
    if still:
        return {
            "ok": True,
            "context": remaining,
            "expected_action": "partial_answer",
            "evidence_state": "partial",
            "required_points": still,
            "unavailable_points": gone,
            "answer": "。".join(still) + "。其余关键内容在当前资料中不足，无法确定。",
            "answerable_part": "。".join(still),
            "insufficient_part": "。".join(gone),
        }
    return {
        "ok": True,
        "context": remaining,
        "expected_action": "state_insufficient",
        "evidence_state": "missing",
        "required_points": [],
        "unavailable_points": [sentence],
        "answer": "当前资料不足，无法确定该问题。",
        "answerable_part": "",
        "insufficient_part": sentence,
    }


def build_behavior_boundary_cases(
    docs: list[dict[str, Any]],
    *,
    train_families: set[str] | None = None,
    train_generic_names: set[str] | None = None,
    limit: int = 12,
    min_families: int = 8,
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    """用 dev 正文构造 4 个条件。凑不齐就记缺口，不编造事实，也不改用 v23 行为题。"""
    train_families = train_families or set()
    train_generic_names = train_generic_names or set()
    prepared = []
    rejected = []
    for doc in docs:
        if doc.get("split") == "locked_test" or split_for_family(str(doc.get("source_family_id") or "skip")) == "locked_test":
            rejected.append({"stem": doc.get("stem"), "reason": "locked_test"})
            continue
        text = str(doc.get("text") or "")
        if not text or len(text) > 80000:
            rejected.append({"stem": doc.get("stem"), "reason": "empty_or_too_long"})
            continue
        name = _clean_name(text)
        support = _support_sentence(text, name) if name else None
        if not name or support is None:
            rejected.append({"stem": doc.get("stem"), "reason": "no_unique_support"})
            continue
        prepared.append({**doc, "generic_name": name, "support": support, "text": text})
    rng = random.Random(seed)
    prepared.sort(key=lambda item: (str(item.get("source_family_id")), str(item.get("stem"))))
    rng.shuffle(prepared)
    ordered: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in prepared:
        family = str(item.get("source_family_id") or "")
        if family in seen_families:
            continue
        ordered.append(item)
        seen_families.add(family)
    chosen = {id(item) for item in ordered}
    for item in prepared:
        if id(item) not in chosen:
            ordered.append(item)
    cases: list[dict[str, Any]] = []
    skipped_pairs = []
    for item in ordered:
        if len({case["core_question_id"] for case in cases}) >= limit:
            break
        partner = next((other for other in prepared if other["generic_name"] != item["generic_name"] and other["source_family_id"] != item["source_family_id"]), None)
        if partner is None:
            skipped_pairs.append({"stem": item.get("stem"), "reason": "no_partner"})
            continue
        partner_sentence = _support_sentence(partner["text"], partner["generic_name"])
        if partner_sentence is None or partner["generic_name"] not in partner_sentence["sentence"] and partner["generic_name"] not in partner["text"]:
            identity = next((line.strip() for line in partner["text"].splitlines() if partner["generic_name"] in line), "")
            if not identity:
                skipped_pairs.append({"stem": item.get("stem"), "reason": "partner_name_invisible"})
                continue
            partner_visible = identity
        else:
            partner_visible = partner_sentence["sentence"] if partner["generic_name"] in partner_sentence["sentence"] else next(line.strip() for line in partner["text"].splitlines() if partner["generic_name"] in line)
        removal = _removal_case(item["text"], item["support"]["sentence"])
        if not removal["ok"]:
            skipped_pairs.append({"stem": item.get("stem"), "reason": removal["reason"]})
            continue
        name = item["generic_name"]
        question = f"根据资料，{name}在{item['support']['section']}中的要点是什么？"
        sufficient_context = f"【药品名称】\n{name}\n\n【{item['support']['section']}】\n{item['support']['sentence']}"
        if name not in sufficient_context or item["support"]["sentence"] not in sufficient_context:
            skipped_pairs.append({"stem": item.get("stem"), "reason": "sufficient_context_incomplete"})
            continue
        overlap = item.get("source_family_id") in train_families or name in train_generic_names
        common = {
            "core_question_id": f"v25b_{item.get('stem')}",
            "family_id": f"v25b_{item.get('stem')}",
            "source_family_id": item.get("source_family_id") or "",
            "generic_family_id": item.get("generic_family_id") or "",
            "generic_name": name,
            "source": item.get("stem"),
            "source_hash": item.get("content_sha256") or "",
            "split": "dev",
            "stratum": "paired_behavior_boundary",
            "dataset_version": "drug_v25_experiment_b",
            "gold_version": "evidence_transform_v25",
            "task_mode": "grounded",
            "review_status": "pending_review",
            "authoring": "evidence_transform",
            "source_overlap_train": overlap,
            "system": "仅依据提供的资料回答；资料不足时说明缺少的信息。",
        }
        specs = [
            ("sufficient", question, sufficient_context, "answer", "sufficient", [item["support"]["sentence"]], [], item["support"]["sentence"], item["support"]["sentence"], ""),
            ("removed_support", question, removal["context"], removal["expected_action"], removal["evidence_state"], removal["required_points"], removal["unavailable_points"], removal["answer"], removal["answerable_part"], removal["insufficient_part"]),
            ("wrong_object", question, partner_visible, "state_insufficient", "wrong_object", [], [item["support"]["sentence"]], "所给资料的对象与问题不一致，资料不足，无法据此回答。", "", item["support"]["sentence"]),
            ("distractor", question, sufficient_context + "\n" + partner_visible, "answer", "sufficient", [item["support"]["sentence"]], [], item["support"]["sentence"], item["support"]["sentence"], ""),
        ]
        group = []
        failed_reason = None
        for condition, asked, context, action, state, required, unavailable, answer, answerable, insufficient in specs:
            if condition == "wrong_object" and partner["generic_name"] not in context:
                failed_reason = "wrong_object_name_hidden"
                break
            if condition == "distractor" and (item["support"]["sentence"] not in context or partner["generic_name"] not in context):
                failed_reason = "distractor_incomplete"
                break
            if str(case_id := f"v25b_{item.get('stem')}_{condition}").startswith("v23_behavior"):
                failed_reason = "renamed_v23_behavior"
                break
            case = {
                **common,
                "case_id": case_id,
                "condition_id": condition,
                "question": asked,
                "context": context,
                "messages": [
                    _message("system", common["system"]),
                    _message("user", f"资料：\n{context}{QUESTION_MARKER}{asked}"),
                ],
                "expected_action": action,
                "evidence_state": state,
                "answer": answer,
                "answer_points": [answer] if answer else [],
                "required_points": required,
                "unavailable_points": unavailable,
                "answerable_part": answerable,
                "insufficient_part": insufficient,
                "partner_generic_name": partner["generic_name"],
                "partner_source_family_id": partner.get("source_family_id"),
            }
            case["self_check"] = reference_self_check(case)
            if case["self_check"].get("passed") is not True:
                failed_reason = "self_check_failed"
                break
            group.append(case)
        if failed_reason or len(group) != 4:
            skipped_pairs.append({"stem": item.get("stem"), "reason": failed_reason or "incomplete_group"})
            continue
        cases.extend(group)
    cores = []
    for case in cases:
        if case["core_question_id"] not in cores:
            cores.append(case["core_question_id"])
    complete = []
    for core in cores:
        group = [case for case in cases if case["core_question_id"] == core]
        if len(group) == 4:
            complete.extend(group)
    families = {case.get("source_family_id") for case in complete}
    return {
        "cases": complete,
        "planned_n": limit * 4,
        "built_n": len(complete),
        "core_n": len({case["core_question_id"] for case in complete}),
        "source_family_n": len(families),
        "min_families": min_families,
        "shortfall_cases": max(0, limit * 4 - len(complete)),
        "shortfall_cores": max(0, limit - len({case["core_question_id"] for case in complete})),
        "rejected_n": len(rejected),
        "skipped_pairs": skipped_pairs,
        "selection_seed": seed,
        "locked_test_used": False,
    }


def _load_docs_for_boundary(inventory_path: Path, train_families: set[str]) -> list[dict[str, Any]]:
    docs = []
    if not inventory_path.is_file():
        return docs
    for row in _read_jsonl(inventory_path):
        if row.get("split") != "dev" or row.get("status") != "parsed_text":
            continue
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        identity = extract_identity(text)
        family = _family_key(identity, "source")
        if split_for_family(family) == "locked_test":
            continue
        docs.append(
            {
                "stem": row.get("stem"),
                "text": text,
                "source_family_id": family,
                "generic_family_id": _family_key(identity, "generic"),
                "split": "dev",
                "content_sha256": _sha256(path),
                "source_overlap_train": family in train_families,
            }
        )
    return docs


def _blind_pack(predictions: list[dict[str, Any]], *, batch_id: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = random.Random(BLIND_SEED)
    order = list(range(len(predictions)))
    rng.shuffle(order)
    rows = []
    mapping = []
    for slot, index in enumerate(order, start=1):
        item = predictions[index]
        label = batch_id[:-6] if batch_id.endswith("_blind") else batch_id
        blind_id = f"{label}_blind_{slot:04d}"
        text = str(item.get("text") or "")
        rows.append(
            {
                "batch_id": batch_id,
                "blind_id": blind_id,
                "content_hash": blind_content_hash(blind_id, text),
                "answer_text": text,
                "task_completed": "",
                "key_factual_errors": "",
                "unsupported_content": "",
                "behavior_appropriate": "",
                "reviewer_a": "",
                "opinion_a": "",
                "reviewer_b": "",
                "opinion_b": "",
                "adjudication": "",
                "ai_note": "只根据回答本身判断。不要猜测模型身份。",
            }
        )
        mapping.append(
            {
                "blind_id": blind_id,
                "case_id": item.get("case_id"),
                "model_role": item.get("model_role"),
                "model_id": item.get("model_id"),
                "adapter_id": item.get("adapter_id"),
                "signature": item.get("signature"),
                "answer_text": text,
            }
        )
    return rows, mapping


def _with_prediction_messages(case: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in predictions if str(row.get("case_id")) == case_id_of(case)]
    messages = [row.get("messages") for row in matched if row.get("messages")]
    cloned = dict(case)
    if messages and all(item == messages[0] for item in messages):
        cloned["messages"] = messages[0]
    cloned.setdefault("condition_id", case.get("pattern") or case.get("condition_id") or "behavior")
    cloned.setdefault("core_question_id", case.get("family_id") or case_id_of(case))
    return cloned


def maybe_execute_batch(
    *,
    ready: bool,
    allow_inference: bool,
    self_check_blocked: bool = False,
    generate_fn: Any = None,
) -> dict[str, Any]:
    if self_check_blocked:
        return {"status": "blocked", "reason": "self_check_failed", "model_loaded": False}
    if not ready:
        return {"status": "blocked", "reason": "pending_review", "model_loaded": False}
    if not allow_inference:
        return {"status": "not_started", "reason": "inference_not_requested", "model_loaded": False}
    if generate_fn is not None:
        return {"status": "ready", "reason": "generate_fn_supplied", "model_loaded": False}
    return {"status": "not_started", "reason": "inference_not_requested", "model_loaded": False}


def _auxiliary_behavior_counts(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in predictions:
        by_case.setdefault(str(row.get("case_id")), {})[str(row.get("model_role"))] = row
    over = answered = scored = 0
    examples = []
    transitions = []
    for case in cases:
        pair = by_case.get(case_id_of(case)) or {}
        if "base" not in pair or "adapter" not in pair:
            continue
        base_score = score_task_v3(str(pair["base"].get("text") or ""), case)
        tuned_score = score_task_v3(str(pair["adapter"].get("text") or ""), case)
        scored += 1
        for role, scored_row in (("base", base_score), ("adapter", tuned_score)):
            actual = (scored_row.get("expected_behavior") or {}).get("actual")
            expected = case.get("expected_action")
            if expected == "answer" and case.get("evidence_state") == "sufficient" and actual == "state_insufficient":
                over += 1
                examples.append({"case_id": case_id_of(case), "model_role": role, "kind": "over_refusal", "metric_role": "auxiliary"})
            if expected == "state_insufficient" and actual == "answer":
                answered += 1
                examples.append({"case_id": case_id_of(case), "model_role": role, "kind": "answered_despite_insufficient", "metric_role": "auxiliary"})
        transitions.append(
            {
                "case_id": case_id_of(case),
                "core_question_id": case.get("core_question_id"),
                "source_family_id": case.get("source_family_id"),
                "condition_id": case.get("condition_id"),
                "base_passed": base_score.get("passed"),
                "adapter_passed": tuned_score.get("passed"),
                "metric_role": "auxiliary",
                "human_adjudication": "pending_review",
            }
        )
    return {
        "scored_prediction_pairs": scored,
        "auxiliary_over_refusal_n": over,
        "auxiliary_answered_despite_insufficient_n": answered,
        "examples": examples[:12],
        "transitions": transitions,
        "metric_role": "auxiliary",
        "scorer_version": SCORER_V3,
        "semantic_accuracy_verified": False,
        "human_disagreement": "pending_review",
    }


def _core_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    cores = {str(case.get("core_question_id") or case_id_of(case)) for case in cases}
    families = {str(case.get("source_family_id") or "") for case in cases if case.get("source_family_id")}
    return {
        "condition_n": len(cases),
        "core_question_n": len(cores),
        "source_family_n": len(families),
        "weighting": "core_question_and_source_family",
        "note": "同一核心问题的多个条件或改写不按独立样本相加。",
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# drug_v25 评测审核与实验材料",
        "",
        "本轮准备了审核闭环和实验 A/B 候选。没有重新训练，没有调用教师，也没有把规则分写成语义正确率。",
        "",
        "## 已完成",
        "",
    ]
    for item in summary["status"]["completed"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 待人工审核", ""])
    for item in summary["status"]["pending_review"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 阻塞", ""])
    for item in summary["status"]["blocked"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 未开始", ""])
    for item in summary["status"]["not_started"]:
        lines.append(f"- {item}")
    audit = summary["batches"]
    lines.extend(["", "## 分母", ""])
    for name, row in audit.items():
        lines.append(
            f"- {name}：计划 {row.get('planned_n')}，已构造 {row.get('built_n')}，有效预测 {row.get('valid_prediction_n')}，"
            f"失败 {row.get('failed_n')}，待审核 {row.get('pending_review_n')}，已裁定 {row.get('adjudicated_n')}。"
        )
    lines.extend(
        [
            "",
            "审核覆盖率只描述已经裁定的部分，不外推到未审核题目。",
            "",
            "## 人工裁定",
            "",
            f"- 导入校验错误 {summary.get('import', {}).get('error_n', '未知')}，待审核 {summary.get('import', {}).get('pending_review_n', '未知')}，已裁定 {summary['human']['adjudicated_n']}。",
            f"- 任务通过与行为适当：已裁定 {summary['human']['adjudicated_n']} 条，因此这两项都还不能汇总。",
            f"- 规则与人工不一致：{summary['human']['rule_disagreement']}。",
            "",
            "## 实验 A",
            "",
            f"- 沿用冻结的 {summary['experiment_a']['core_n']} 个核心问题，构造 {summary['experiment_a']['built_n']} 个输入，计划分母 {summary['experiment_a']['planned_n']}。",
            f"- 原题预测复用 {summary['experiment_a']['reused_n']} / {summary['experiment_a']['reuse_checked_n']}。v22 的旧回答没有预测身份，没有复用。",
            "- 聚合单位是核心问题和来源家族。三条问法不按三道独立题相加。",
            "- v23 的整篇说明书改写仍是另一条件，不并进本实验。",
            "- 这些输入来自训练中见过的来源，只作为学习诊断，使用 exploratory，不用 formal 取消来源重叠检查。",
            "",
            "## 实验 B",
            "",
            f"- 计划 48 个条件输入，实际构造 {summary['experiment_b']['built_n']}，核心问题 {summary['experiment_b']['core_n']}，来源家族 {summary['experiment_b']['source_family_n']}。",
            f"- 缺口 {summary['experiment_b']['shortfall_cases']}。没有用 v23 的 10 道行为题补位，也没有使用 locked_test。",
            f"- 构造时跳过 {summary['experiment_b'].get('skipped_n', 0)} 个候选文档，原因见 experiment_b/selection.json。这些文档没有改写成合格条件。",
            f"- 与训练来源重叠的条件 {summary['experiment_b']['overlap_n']} 个，已单独记录。",
            f"- 已写入协议的条件里，参考答案自评未通过 {summary['experiment_b']['self_check_failed_n']} 个。未通过的组不会进入协议；凑不齐时记入缺口，计划分母仍是 48。",
            f"- 有证据却拒答、证据不足仍作答的人工计数尚未产生。已有预测上的规则计数见辅助结果，且只覆盖有预测的批次。",
            "",
            "## 辅助规则",
            "",
            f"- 评分器 {SCORER_V3} 是辅助分。formal_main_metric 仍是 not_executed。",
            f"- 已有预测上的规则“有证据却拒答” {summary['auxiliary']['auxiliary_over_refusal_n']}，规则“证据不足仍作答” {summary['auxiliary']['auxiliary_answered_despite_insufficient_n']}。",
            "- 开发集用于构造实验 B；实验 A 是见过来源上的诊断；正式测试仍要求独立来源、非空可追溯清单和人工金标。三者不能混称。",
            "",
            "## 真实失败与限制",
            "",
        ]
    )
    for item in summary["failures"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "历史 G0 是 seed=42 的单次训练。当时工作区为脏状态，adapter 与基座的 tokenizer 配置不一致。以后若做训练对照，应另建可复现的新基线，不能把实现变化算进处理效应。",
            "",
            "## 下一步",
            "",
            "先完成下面的人工审核。审核没有全部接受之前，不运行实验 A 或 B 的推理，也不训练。",
            "",
            "- 若改写收益经人工确认，再考虑至少三个训练 seed 的稳定性。",
            "- 若行为错误明确，再预先固定普通 SFT 与加入行为样本的单因素对照及训练预算。",
            "- 若独立来源简单题出现天花板，再从 dev 增加条件、否定、数值绑定和多证据题；选题规则要在看模型输出前冻结，并保留旧简单题。",
            "- 只有真实学生探测器、有效选样干预和冻结回放池都具备后，才考虑 G1–G3。",
            "- 不使用 locked_test 调题、改评分器或选择训练方案。",
            "",
            "## 审核后如何继续",
            "",
            "填完 CSV 后，在仓库根目录执行：",
            "",
            "```sh",
            "qa-pipeline import-review --run runs/drug_v25_eval",
            "```",
            "",
            "导入结果写在 `runs/drug_v25_eval/review/import_result.json`。只有对应批次的 `ready_for_inference` 为 true，才可以对该批运行有上限的 exploratory 推理。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_v25(repo: Path | None = None, dest_name: str = "drug_v25_eval", *, allow_inference: bool = False) -> dict[str, Any]:
    root = repo or REPO
    dest = root / "runs" / dest_name
    assert_v25_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    review = dest / "review"
    review.mkdir(exist_ok=True)
    write_fill_example(review / "examples")

    train_path = root / "runs/drug_v22/E3_g0/sft/train.jsonl"
    status_path = root / "runs/drug_v23_eval/candidate_status.json"
    source_path = root / "runs/drug_v23_eval/source_index.json"
    candidates_path = root / "runs/drug_v23_eval/candidates/dev_candidates.jsonl"
    predictions_path = root / "runs/drug_v23_eval/exploratory/predictions.jsonl"
    new_protocol = root / "runs/drug_v24_audit/new_input_protocol.jsonl"
    new_predictions = root / "runs/drug_v24_rescore/new_inputs/predictions.jsonl"
    seen_predictions = root / "runs/drug_v24_rescore/train_seen_confirmed/predictions.jsonl"
    seen_report = root / "runs/drug_v24_rescore/train_seen_confirmed/eval_report.json"
    inventory = root / "data/campus_hospital_drug_instructions/frozen/inventory.jsonl"

    family_ids = []
    if status_path.is_file():
        family_ids = list(json.loads(status_path.read_text(encoding="utf-8"))["core_questions"]["family_ids"])
    sources = {str(item.get("filename")): item for item in _read_jsonl(source_path)}
    experiment_a = build_fixed_context_cases(_read_jsonl(train_path), family_ids, sources)
    seen_meta = json.loads(seen_report.read_text(encoding="utf-8")) if seen_report.is_file() else {}
    reuse = match_saved_predictions(
        [case for case in experiment_a["cases"] if case.get("condition_id") == "original"],
        _read_jsonl(seen_predictions),
        base_id=str(seen_meta.get("base_id") or ""),
        adapter_id=str(seen_meta.get("adapter_id") or ""),
        template_id=str(seen_meta.get("template_id") or ""),
        infer_effective=dict(seen_meta.get("infer_config_effective") or EFFECTIVE_INFER),
    )
    train_families = {str(item.get("source_family_id") or "") for item in sources.values() if item.get("source_family_id")}
    train_names = {str(item.get("generic_name") or "") for item in sources.values() if item.get("generic_name")}
    docs = _load_docs_for_boundary(inventory, train_families)
    experiment_b = build_behavior_boundary_cases(docs, train_families=train_families, train_generic_names=train_names)

    behavior_cases = [row for row in _read_jsonl(candidates_path) if row.get("stratum") == "behavior"]
    behavior_predictions = [row for row in _read_jsonl(predictions_path) if row.get("stratum") == "behavior" or row.get("case_id") in {case_id_of(case) for case in behavior_cases}]
    behavior_cases = [_with_prediction_messages(case, behavior_predictions) for case in behavior_cases]
    new_cases = [_with_prediction_messages(case, _read_jsonl(new_predictions)) for case in _read_jsonl(new_protocol)]
    for case in new_cases:
        case["stratum"] = case.get("stratum") or "behavior_new_input"
        case.setdefault("condition_id", "new_input")
        case.setdefault("core_question_id", case.get("parent_case_id") or case_id_of(case))
    batch1_cases = behavior_cases + new_cases
    batch1_predictions = behavior_predictions + [row for row in _read_jsonl(new_predictions)]

    _write_jsonl(dest / "experiment_a/protocol.jsonl", experiment_a["cases"])
    _write_jsonl(dest / "experiment_b/protocol.jsonl", experiment_b["cases"])
    _write_jsonl(dest / "batch1/protocol.jsonl", batch1_cases)
    _write_json(dest / "experiment_a/reuse_decision.json", reuse)
    _write_json(dest / "experiment_a/blocked.json", experiment_a["blocked"])
    _write_json(dest / "experiment_b/selection.json", {key: value for key, value in experiment_b.items() if key != "cases"})

    batches = [
        ("batch1_behavior", batch1_cases, "优先审核：v2.3 的 10 道行为题和 v2.4 的 4 道新输入。请看可读材料中的完整输入。"),
        ("batch_a_gold", experiment_a["cases"], "实验 A。原题 semantic_preserved 填 na；两条改写必须确认语义和答案要求没变。"),
        ("batch_b_gold", experiment_b["cases"], "实验 B。每个条件单独看。移除支持时确认全文没有等效信息；错误对象的名称必须在可见内容里。"),
    ]
    batch_specs = []
    for batch_id, cases, note in batches:
        csv_path = review / f"{batch_id}.csv"
        md_path = review / f"{batch_id}.md"
        if not csv_has_human_input(csv_path):
            write_csv(csv_path, GOLD_COLUMNS, [gold_row_from_case(case, batch_id=batch_id, ai_note=note) for case in cases])
            md_path.write_text(render_gold_markdown(cases, batch_id=batch_id), encoding="utf-8")
        batch_specs.append(
            {
                "batch_id": batch_id,
                "kind": "gold",
                "csv": str(csv_path.relative_to(dest)),
                "protocol": str((dest / {"batch1_behavior": "batch1/protocol.jsonl", "batch_a_gold": "experiment_a/protocol.jsonl", "batch_b_gold": "experiment_b/protocol.jsonl"}[batch_id]).relative_to(dest)),
                "reviewed_protocol": f"review/{batch_id}.reviewed.jsonl",
            }
        )
    blind_rows, blind_map = _blind_pack(batch1_predictions, batch_id="batch1_behavior_blind")
    blind_csv = review / "batch1_behavior_blind.csv"
    if not csv_has_human_input(blind_csv):
        write_csv(blind_csv, BLIND_COLUMNS, blind_rows)
        (review / "batch1_behavior_blind.md").write_text(render_blind_markdown(blind_rows, batch_id="batch1_behavior_blind"), encoding="utf-8")
    _write_json(review / "batch1_behavior_identity_map.json", blind_map)
    batch_specs.append(
        {
            "batch_id": "batch1_behavior_blind",
            "kind": "blind",
            "csv": "review/batch1_behavior_blind.csv",
            "identity_map": "review/batch1_behavior_identity_map.json",
        }
    )
    _write_json(review / "batches.json", {"batches": batch_specs})
    (review / "review_guide.md").write_text(_review_guide(), encoding="utf-8")
    imported = import_run_reviews(dest)

    auxiliary = _auxiliary_behavior_counts(batch1_cases, batch1_predictions)
    reused_predictions = []
    saved_by_key = {(str(row.get("case_id")), str(row.get("model_role"))): row for row in _read_jsonl(seen_predictions)}
    for decision in reuse["decisions"]:
        if decision["reused"]:
            reused_predictions.append(saved_by_key[(decision["case_id"], decision["model_role"])])
    original_cases = [case for case in experiment_a["cases"] if case.get("condition_id") == "original"]
    auxiliary_a = _auxiliary_behavior_counts(original_cases, reused_predictions)
    transitions = auxiliary["transitions"] + auxiliary_a["transitions"]
    _write_jsonl(dest / "analysis/item_transitions.jsonl", transitions)
    self_failed = [case for case in experiment_b["cases"] if case.get("self_check", {}).get("passed") is not True]
    _write_json(dest / "experiment_b/self_check_failures.json", [{"case_id": case_id_of(case), "reasons": case.get("self_check", {}).get("reasons")} for case in self_failed])

    a_ready = bool(imported["ready_for_inference"].get("batch_a_gold")) and not experiment_a["blocked"]
    b_ready = bool(imported["ready_for_inference"].get("batch_b_gold")) and not self_failed and experiment_b["built_n"] > 0
    execution = {
        "batch_a": maybe_execute_batch(ready=a_ready, allow_inference=allow_inference, self_check_blocked=bool(experiment_a["blocked"])),
        "batch_b": maybe_execute_batch(ready=b_ready, allow_inference=allow_inference, self_check_blocked=bool(self_failed)),
    }
    failures = []
    if experiment_a["blocked"]:
        failures.append(f"实验 A 有 {len(experiment_a['blocked'])} 个构造问题，见 experiment_a/blocked.json。")
    if reuse["rejected_n"]:
        reasons = Counter(reason for item in reuse["decisions"] if not item["reused"] for reason in item["reasons"])
        failures.append("实验 A 原题未复用预测：" + "，".join(f"{key} {value}" for key, value in reasons.items()))
    if experiment_b["shortfall_cases"]:
        failures.append(f"实验 B 距离 48 个条件还差 {experiment_b['shortfall_cases']}，没有用编造事实或旧行为题填满。")
    if self_failed:
        failures.append(f"实验 B 有 {len(self_failed)} 个条件的参考答案自评未通过，见 experiment_b/self_check_failures.json。")
    if not inventory.is_file():
        failures.append("缺少冻结 inventory.jsonl，实验 B 无法从 dev 文档构造。")
    failures.append("人工裁定仍为 0。规则分不能代替盲评。")

    identity = {
        "train_jsonl_sha256": _sha256(train_path),
        "adapter_sha256": _sha256(root / "runs/drug_v22/E3_g0/sft/lora/adapter/adapter_model.safetensors"),
        "base_config_sha256": _sha256(Path("/data1/pjw/models/Qwen2.5-7B-Instruct/config.json")),
        "tokenizer_config_sha256": _sha256(Path("/data1/pjw/models/Qwen2.5-7B-Instruct/tokenizer_config.json")),
        "source_index_sha256": _sha256(source_path),
        "candidate_status_sha256": _sha256(status_path),
        "experiment_a_protocol_sha256": _sha256(dest / "experiment_a/protocol.jsonl"),
        "experiment_b_protocol_sha256": _sha256(dest / "experiment_b/protocol.jsonl"),
        "base_id": seen_meta.get("base_id"),
        "adapter_id": seen_meta.get("adapter_id"),
        "template_id": seen_meta.get("template_id"),
        "infer_config_effective": seen_meta.get("infer_config_effective") or EFFECTIVE_INFER,
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inference": "not_executed",
        "gpu_seconds": 0,
    }
    _write_json(dest / "identity.json", identity)
    status = {
        "completed": [
            "核对 v2.2 训练 JSONL、consumed_ids 与 G0 adapter 仍在原路径。",
            "核对 v2.3 的 100 道候选和 200 份预测，以及 v2.4 的重评分、补推理和空白审核。",
            "为 eval-adapter 增加非空、可追溯的冻结来源清单参数。",
            "写出实验 A、实验 B 和优先行为批次的候选协议与审核包。",
            "离线核对实验 A 原题能否复用 v2.4 预测身份。",
        ],
        "pending_review": [
            "batch1_behavior：10 道 v2.3 行为题和 4 道 v2.4 新输入的金标与盲评。",
            "batch_a_gold：实验 A 的原题和两条改写。",
            "batch_b_gold：实验 B 的每个条件。",
            "其余 v2.4 历史审核包留作后续批次，本轮不重导。",
        ],
        "blocked": [
            "实验 A 推理：金标尚未全部接受。",
            "实验 B 推理：金标尚未全部接受。协议内参考答案自评未通过时同样不能启动。",
        ],
        "not_started": [
            "多 seed 稳定性训练。",
            "普通 SFT 与行为样本的单因素对照。",
            "G1–G3。",
            "独立来源正式测试。",
        ],
    }
    if execution["batch_a"]["status"] != "blocked":
        status["blocked"] = [item for item in status["blocked"] if not item.startswith("实验 A")]
    batches_view = {
        "batch1_behavior": {
            "planned_n": 14,
            "built_n": len(batch1_cases),
            "valid_prediction_n": len(batch1_predictions),
            "failed_n": 14 - len(batch1_cases) if len(batch1_cases) < 14 else 0,
            "pending_review_n": 14,
            "adjudicated_n": 0,
        },
        "experiment_a": {
            "planned_n": experiment_a["planned_n"],
            "built_n": experiment_a["built_n"],
            "valid_prediction_n": reuse["reused_n"],
            "failed_n": experiment_a["planned_n"] - experiment_a["built_n"],
            "pending_review_n": experiment_a["built_n"],
            "adjudicated_n": 0,
        },
        "experiment_b": {
            "planned_n": experiment_b["planned_n"],
            "built_n": experiment_b["built_n"],
            "valid_prediction_n": 0,
            "failed_n": experiment_b["shortfall_cases"],
            "pending_review_n": experiment_b["built_n"],
            "adjudicated_n": 0,
        },
    }
    summary = {
        "status": status,
        "batches": batches_view,
        "human": {"adjudicated_n": imported.get("adjudicated_n") or 0, "rule_disagreement": "pending_review"},
        "experiment_a": {
            "core_n": len(family_ids),
            "planned_n": experiment_a["planned_n"],
            "built_n": experiment_a["built_n"],
            "reused_n": reuse["reused_n"],
            "reuse_checked_n": reuse["checked_n"],
            "aggregate": _core_aggregate(experiment_a["cases"]),
        },
        "experiment_b": {
            "planned_n": experiment_b["planned_n"],
            "built_n": experiment_b["built_n"],
            "core_n": experiment_b["core_n"],
            "source_family_n": experiment_b["source_family_n"],
            "shortfall_cases": experiment_b["shortfall_cases"],
            "skipped_n": len(experiment_b.get("skipped_pairs") or []),
            "overlap_n": sum(1 for case in experiment_b["cases"] if case.get("source_overlap_train")),
            "self_check_failed_n": len(self_failed),
            "aggregate": _core_aggregate(experiment_b["cases"]),
        },
        "auxiliary": {
            "auxiliary_over_refusal_n": auxiliary["auxiliary_over_refusal_n"] + auxiliary_a["auxiliary_over_refusal_n"],
            "auxiliary_answered_despite_insufficient_n": auxiliary["auxiliary_answered_despite_insufficient_n"] + auxiliary_a["auxiliary_answered_despite_insufficient_n"],
            "examples": auxiliary["examples"],
        },
        "failures": failures,
        "execution": execution,
        "import": {key: value for key, value in imported.items() if key != "batches"},
    }
    _write_json(dest / "status.json", status)
    _write_json(dest / "summary.json", summary)
    (dest / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def _review_guide() -> str:
    return """# 审核填写说明

先看 `review/examples/fill_example.md`。示例是虚构的“示例物品A”，不要把示例行抄进正式表。

## 先审哪一批

1. 打开 `review/batch1_behavior.md`，填写 `review/batch1_behavior.csv`。这是 v2.3 的 10 道行为题和 v2.4 的 4 道新输入。
2. 同一批的回答盲评：打开 `review/batch1_behavior_blind.md`，填写 `review/batch1_behavior_blind.csv`。不要打开 `review/batch1_behavior_identity_map.json`。
3. 然后打开 `review/batch_a_gold.md`，填写 `review/batch_a_gold.csv`。每道核心题有原题、改写一、改写二，都要看完整输入。
4. 最后打开 `review/batch_b_gold.md`，填写 `review/batch_b_gold.csv`。48 个条件各自审核；若实际构造少于 48，只审已经列出的条件，不要自行补题。

历史的 174 条金标和 348 份盲评留在 `runs/drug_v24_review/`，本轮不重导。

## 金标列

不要改 `batch_id`、`case_id`、`condition_id`、`core_question_id`、`content_hash`。这些用来确认你审的是这份输入。

- `question_clear`：题干是否清楚，填 yes 或 no。
- `semantic_preserved`：改写是否仍问同一件事并保持答案要求。原题填 na，改写填 yes 或 no。
- `visible_evidence_adequate`：完整可见输入是否够用，填 yes、no 或 partial。
- `required_points_ok`：必答要点是否合适，填 yes 或 no。
- `unavailable_points_ok`：不可回答的部分是否合适，填 yes、no 或 na。
- `expected_behavior_ok`：预期行为是否合适，填 yes 或 no。有充分证据时不能把拒答当成正确。
- `answerable_part`：部分支持时，写下可以回答的部分。
- `insufficient_part`：部分支持或证据不足时，写下应说明不足的部分。
- `reviewer_a`、`opinion_a`：第一位审核者的真实姓名和意见。
- `reviewer_b`、`opinion_b`：第二位审核者。只填了一人时，结果是单人 `agreed`，不会变成双人一致。
- `adjudication`：accept、revise 或 reject。留空则保持 pending_review，不会默认通过。
- `ai_note`：程序提示。导入时忽略，不能充当姓名、意见或裁定。

## 盲评列

`task_completed`、`behavior_appropriate` 填 yes 或 no。`key_factual_errors` 和 `unsupported_content` 写具体问题；没有就写“无”。审核者、意见和裁定的规则与金标相同。

## 填完后

在仓库根目录执行：

```sh
qa-pipeline import-review --run runs/drug_v25_eval
```

查看 `review/import_result.json`。只有该批 `ready_for_inference` 为 true，才继续对该批做有上限推理。revise 或 reject 的题目会留在分母中并阻塞该批，不会被删掉。
"""


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="准备 drug_v25 审核与实验材料")
    parser.add_argument("--run-name", default="drug_v25_eval")
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare_v25(dest_name=args.run_name, allow_inference=args.allow_inference)
    print(json.dumps({"execution": summary["execution"], "batches": summary["batches"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
