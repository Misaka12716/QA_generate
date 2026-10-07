"""生成前的证据能力目录与题型配额。证据不够时只报告缺口。"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from ..qtypes import QUOTA_RATIOS, largest_remainder
from ..registry import register
from ..schemas import QUOTA_RATIOS as SCHEMA_RATIOS
from ..schemas import Chunk, EvidenceQuote

_STEP = re.compile(r"先.{0,16}再|步骤|然后|最后")
_STEP_VERB = re.compile(r"送服|配制|煎|操作|记录|观察|使用")
_COND = re.compile(r"除非|禁用|不宜|例外|如果|不适用于")
_LINK = re.compile(r"因此|从而|先确认.{0,20}再确认")
_OBJECT = re.compile(r"示例制剂[甲乙丙丁]|[\u4e00-\u9fff]{2,8}(?:片|胶囊|颗粒)")


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[。！？]", text or "") if part.strip()]
    return [part for part in parts if len(part) >= 6]


def chunk_capabilities(chunk: Chunk) -> dict[str, Any]:
    text = chunk.text or ""
    caps: set[str] = set()
    reasons: dict[str, str] = {}
    if _sentences(text):
        caps.add("factual")
        reasons["factual"] = "chunk_has_attribute_sentence"
    if _STEP.search(text) and _STEP_VERB.search(text):
        caps.add("procedural")
        reasons["procedural"] = "ordered_procedure"
    else:
        reasons["procedural"] = "no_procedure_evidence"
    if _COND.search(text):
        caps.add("conditional")
        reasons["conditional"] = "condition_or_exception"
    else:
        reasons["conditional"] = "no_condition_evidence"
    objects = []
    for match in _OBJECT.finditer(text):
        name = match.group(0)
        if name not in objects:
            objects.append(name)
    if len(objects) >= 2 and re.search(r"含量|剂量|规格|维度", text):
        caps.add("comparative")
        reasons["comparative"] = "two_objects_one_dimension"
    else:
        reasons["comparative"] = "need_two_objects_and_one_dimension"
    facts = _sentences(text)
    if len(facts) >= 2 and _LINK.search(text):
        caps.add("multihop")
        reasons["multihop"] = "two_linked_facts"
    else:
        reasons["multihop"] = "need_two_non_omittable_facts"
    return {"capabilities": sorted(caps), "reasons": reasons, "objects": objects, "facts": facts[:3]}


def _quote(chunk: Chunk, text: str) -> EvidenceQuote:
    snippet = text.strip()
    start = chunk.text.find(snippet) if snippet else -1
    if start < 0:
        start = 0
        snippet = ""
    digest = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
    return EvidenceQuote(
        quote_id=f"q_{digest}",
        source_id=chunk.document_identity.source_id or chunk.doc_id,
        source_version=chunk.document_identity.version,
        source_family_id=chunk.source_family_id,
        chunk_id=chunk.chunk_id,
        char_start=chunk.char_start + start,
        char_end=chunk.char_start + start + len(snippet),
        quote=snippet,
        quote_hash=digest,
    )


def _bundle_for(chunks: list[Chunk], q_type: str) -> list[EvidenceQuote] | None:
    """同一文档内最多取 2–3 个片段。单片段也可以承载真正的多跳。"""
    if q_type in {"comparative", "multihop"}:
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.doc_id or chunk.source_doc].append(chunk)
        for group in grouped.values():
            if len(group) < 2:
                info = chunk_capabilities(group[0])
                if q_type in info["capabilities"] and info["facts"]:
                    return [_quote(group[0], info["facts"][0]), _quote(group[0], info["facts"][1] if len(info["facts"]) > 1 else info["facts"][0])]
                continue
            picked = group[:3]
            quotes = []
            for chunk in picked:
                facts = _sentences(chunk.text)
                if facts:
                    quotes.append(_quote(chunk, facts[0]))
            if len(quotes) >= 2:
                return quotes[:3]
        return None
    for chunk in chunks:
        info = chunk_capabilities(chunk)
        if q_type in info["capabilities"] and info["facts"]:
            return [_quote(chunk, info["facts"][0])]
    return None


def build_catalog(chunks: list[Chunk]) -> dict[str, Any]:
    per_chunk = []
    eligible = {name: 0 for name in SCHEMA_RATIOS}
    for chunk in chunks:
        info = chunk_capabilities(chunk)
        per_chunk.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, **info})
        for name in info["capabilities"]:
            eligible[name] = eligible.get(name, 0) + 1
    for name in ("comparative", "multihop"):
        if _bundle_for(chunks, name):
            eligible[name] = max(eligible[name], 1)
    return {"chunks": per_chunk, "eligible": eligible}


def plan_tasks(
    chunks: list[Chunk],
    *,
    answerable_target: int,
    ratios: dict[str, float] | None = None,
    behavior_samples: int = 0,
) -> dict[str, Any]:
    catalog = build_catalog(chunks)
    targets = largest_remainder(answerable_target, ratios or QUOTA_RATIOS)
    tasks: list[dict[str, Any]] = []
    shortage: dict[str, Any] = {}
    for q_type, target in targets.items():
        bundle = _bundle_for(chunks, q_type)
        eligible = 0 if bundle is None else int(catalog["eligible"].get(q_type, 0))
        if bundle is None or eligible <= 0:
            shortage[q_type] = {
                "requested": target,
                "eligible": 0,
                "generated": 0,
                "shortage_reason": "insufficient_evidence",
            }
            continue
        producible = min(target, eligible)
        for index in range(producible):
            tasks.append(
                {
                    "requested_q_type": q_type,
                    "task_index": index,
                    "quotes": [item.model_dump() for item in bundle],
                    "chunk_id": bundle[0].chunk_id,
                    "expected_action": "answer",
                    "selection_role": "learning",
                }
            )
        if producible < target:
            shortage[q_type] = {
                "requested": target,
                "eligible": eligible,
                "generated_tasks": producible,
                "shortage_reason": "insufficient_evidence",
            }
    behavior = []
    if behavior_samples > 0 and chunks:
        behavior = _behavior_tasks(chunks[0], behavior_samples)
    return {
        "catalog": catalog,
        "targets": targets,
        "tasks": tasks,
        "behavior_tasks": behavior,
        "shortage": shortage,
        "answerable_target": answerable_target,
        "ratios": dict(ratios or QUOTA_RATIOS),
    }


def _behavior_tasks(chunk: Chunk, limit: int) -> list[dict[str, Any]]:
    """行为题不进入五类配额。资料不足必须真的没有支持句。"""
    fact = _sentences(chunk.text)
    quote = _quote(chunk, fact[0]).model_dump() if fact else None
    specs = [
        {
            "requested_q_type": "factual",
            "expected_action": "state_insufficient",
            "selection_role": "behavior",
            "prompt": "询问资料中不存在的患者体重换算，不要编造。",
            "quotes": [],
        },
        {
            "requested_q_type": "conditional",
            "expected_action": "partial_answer",
            "selection_role": "behavior",
            "prompt": "只回答资料支持的部分，并说明其余缺口。",
            "quotes": [quote] if quote else [],
        },
        {
            "requested_q_type": "factual",
            "expected_action": "clarify",
            "selection_role": "behavior",
            "prompt": "对象有歧义时先澄清，不要猜标准对象。",
            "quotes": [quote] if quote else [],
        },
        {
            "requested_q_type": "factual",
            "expected_action": "correct_premise",
            "selection_role": "behavior",
            "prompt": "前提与证据冲突时先纠正，再回答可回答部分。",
            "quotes": [quote] if quote else [],
        },
        {
            "requested_q_type": "factual",
            "expected_action": "state_scope",
            "selection_role": "behavior",
            "prompt": "超出任务范围时说明边界，并给出资料内的有限帮助。",
            "quotes": [quote] if quote else [],
        },
    ]
    tasks = []
    for index, spec in enumerate(specs[:limit]):
        tasks.append({**spec, "task_index": index, "chunk_id": chunk.chunk_id, "behavior": True})
    return tasks


def gap_from_counts(targets: dict[str, int], actual: dict[str, int], eligible: dict[str, int]) -> dict[str, Any]:
    """过滤后的缺口。不下调目标，也不把事实题改成缺失类型。"""
    report = {}
    tasks = []
    for name, target in targets.items():
        got = int(actual.get(name, 0))
        can = int(eligible.get(name, 0))
        missing = max(0, target - got)
        reason = ""
        if missing and can <= got:
            reason = "insufficient_evidence"
        elif missing:
            reason = "remaining_quota"
        report[name] = {
            "requested": target,
            "eligible": can,
            "qualified": got,
            "shortage": missing,
            "shortage_reason": reason,
        }
        if missing and can > got:
            tasks.append({"requested_q_type": name, "need": missing, "reason": reason})
    return {"by_type": report, "actionable_tasks": tasks}


@register("planner", "none")
class NoPlanner:
    name = "none"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        return chunks


@register("planner", "capability_catalog")
class CapabilityCatalog:
    name = "capability_catalog"

    def __init__(
        self,
        answerable_target: int = 10,
        quota: dict | None = None,
        behavior_samples: int = 0,
        **_: object,
    ) -> None:
        self.answerable_target = int(answerable_target)
        self.quota = quota or dict(QUOTA_RATIOS)
        self.behavior_samples = int(behavior_samples)

    def run(self, chunks: list[Chunk], ctx) -> list[Chunk]:
        plan = plan_tasks(
            chunks,
            answerable_target=self.answerable_target,
            ratios=self.quota,
            behavior_samples=self.behavior_samples,
        )
        ctx.extras["generation_plan"] = plan
        ctx.extras["generation_tasks"] = list(plan["tasks"]) + list(plan["behavior_tasks"])
        ctx.extras["quota_targets"] = plan["targets"]
        ctx.extras["coverage_report"] = {
            "targets": plan["targets"],
            "eligible": plan["catalog"]["eligible"],
            "shortage": plan["shortage"],
            "behavior_tasks": len(plan["behavior_tasks"]),
        }
        return chunks
