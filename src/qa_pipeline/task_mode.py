"""任务模式。规范值只有 rag_grounded 与 closed_book_domain。"""

from __future__ import annotations

from typing import Any

from .adapters.zhixun import CLOSED_POLICY, GROUNDED_POLICY

RAG_GROUNDED = "rag_grounded"
CLOSED_BOOK = "closed_book_domain"

_ALIASES = {
    "rag_grounded": RAG_GROUNDED,
    "grounded": RAG_GROUNDED,
    "rag": RAG_GROUNDED,
    "closed_book_domain": CLOSED_BOOK,
    "closed_book": CLOSED_BOOK,
    "closed-book": CLOSED_BOOK,
}


class TaskModeError(ValueError):
    """显式消息与声明模式冲突，或模式名无法规范化。"""


def canonical_task_mode(value: str | None) -> str:
    key = str(value or "").strip()
    if key not in _ALIASES:
        raise TaskModeError(f"unknown_task_mode:{key}")
    return _ALIASES[key]


def declared_task_mode(row: dict[str, Any] | None, *, context_key: str = "") -> str | None:
    """读取 task_mode 或旧 goal。未声明时返回 None，不按目录名猜测。"""
    if context_key == "closed_book":
        return CLOSED_BOOK
    if not isinstance(row, dict):
        return None
    raw = row.get("task_mode") or row.get("goal")
    if raw in (None, ""):
        return None
    return canonical_task_mode(str(raw))


def _blob(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(item.get("content") or "") for item in messages)


def messages_conflict(messages: list[dict[str, str]], mode: str, row: dict[str, Any] | None = None) -> str | None:
    """显式 messages 与声明模式不一致时返回原因。"""
    row = row or {}
    text = _blob(messages)
    context = str(row.get("context") or row.get("student_context") or "")
    gold = str(row.get("answer") or row.get("candidate_answer") or "")
    question = str(row.get("question") or "")
    if mode == CLOSED_BOOK:
        if GROUNDED_POLICY in text:
            return "closed_book_contains_grounded_policy"
        if "资料：" in text or "（当前未提供可回答该问题的资料）" in text:
            return "closed_book_contains_context_block"
        if context.strip() and len(context.strip()) >= 8 and context.strip() in text and context.strip() not in question:
            return "closed_book_leaks_context"
        if gold.strip() and len(gold.strip()) >= 8 and gold.strip() in text and gold.strip() not in question:
            return "closed_book_leaks_reference"
        for point in list(row.get("required_points") or []) + list(row.get("answer_points") or []):
            snippet = str(point or "").strip()
            if len(snippet) >= 8 and snippet in text and snippet not in question:
                return "closed_book_leaks_points"
        return None
    if mode == RAG_GROUNDED:
        system = next((str(item.get("content") or "") for item in messages if item.get("role") == "system"), "")
        user = next((str(item.get("content") or "") for item in messages if item.get("role") == "user"), "")
        if system.strip() == CLOSED_POLICY and context.strip() and context.strip() not in user and "资料：" not in user:
            return "rag_messages_are_closed_book"
    return None
