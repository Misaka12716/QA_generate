"""题型判定与配额整数分配。

主类型优先级：multihop、comparative、conditional、procedural、factual。
只对 expected_action=answer 的样本计入五类配额。
"""

from __future__ import annotations

import re
from typing import Any

from .schemas import Q_TYPES, QUOTA_RATIOS

_MULTIHOP = re.compile(r"结合|综合|同时依据|先确认.+再确认|哪些事实")
_COMPARE = re.compile(r"相比|比较|区别|不同|哪一个更|和.+有什么")
_CONDITIONAL = re.compile(r"除非|例外|禁用|不宜|如果|何种条件下|什么条件下|不适用于|不适用")
_PROCEDURE = re.compile(r"步骤|如何操作|怎样操作|操作流程|按什么顺序|依次|先.+再.+最后")
_TIME_QUERY = re.compile(r"什么季节|何时|什么时候|哪个阶段|生长阶段|什么时间|什么时期")


def classify_q_type(question: str, *, requested: str = "") -> str:
    """按问题任务判定主类型。采收时间不是完整流程题。"""
    text = str(question or "").strip()
    if not text:
        return "factual"
    if _MULTIHOP.search(text):
        return "multihop"
    if _COMPARE.search(text):
        return "comparative"
    if _CONDITIONAL.search(text):
        return "conditional"
    if _TIME_QUERY.search(text) and not _PROCEDURE.search(text):
        return "factual"
    if _PROCEDURE.search(text):
        return "procedural"
    if requested in Q_TYPES and requested == "factual":
        return "factual"
    return "factual"


def intent_projection_note(intent: str, actual: str) -> str:
    """旧意图到题型的投影只作记录，不证明任务已经验证。"""
    if intent == "synthesize" and actual != "multihop":
        return "synthesize_is_not_verified_multihop"
    if intent == "diagnose_explain" and actual != "procedural":
        return "diagnose_explain_is_not_verified_procedure"
    if intent == "diagnose_explain" and actual == "procedural":
        return "diagnose_explain_still_needs_step_check"
    return ""


def largest_remainder(total: int, ratios: dict[str, float] | None = None) -> dict[str, int]:
    """最大余数法。比例之和为 1 时，各目标整数之和等于 total。"""
    weights = dict(ratios or QUOTA_RATIOS)
    names = [name for name in Q_TYPES if name in weights]
    if total <= 0:
        return {name: 0 for name in names}
    raw = {name: total * float(weights[name]) for name in names}
    floors = {name: int(raw[name]) for name in names}
    left = total - sum(floors.values())
    order = sorted(names, key=lambda name: (raw[name] - floors[name], -names.index(name)), reverse=True)
    for name in order[: max(0, left)]:
        floors[name] += 1
    return floors


def partial_quota_caps(total: int, quota: dict[str, float]) -> dict[str, int]:
    """未给出完整五类比例时，按各自上限取整，不把缺口补进其他类型。"""
    total_ratio = sum(float(value) for value in quota.values())
    if abs(total_ratio - 1.0) < 1e-6:
        return largest_remainder(total, quota)
    return {name: min(total, int(total * float(ratio))) for name, ratio in quota.items()}


def answerable(item: Any) -> bool:
    action = str(getattr(item, "expected_action", None) or (item.get("expected_action") if isinstance(item, dict) else "") or "answer")
    role = str(getattr(item, "selection_role", None) or (item.get("selection_role") if isinstance(item, dict) else "") or "learning")
    return action == "answer" and role != "behavior"
