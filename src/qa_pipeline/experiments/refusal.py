"""干扰上下文拒答评测：黄金文档应作答，无依据上下文应拒答。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..textutil import token_f1

_REFUSAL = ("无法回答", "无法确定", "不足以", "没有足够", "未提及", "无法从")


def load_refusal_set(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def is_refusal(text: str) -> bool:
    return any(flag in (text or "") for flag in _REFUSAL)


def _prompt(row: dict[str, Any]) -> str:
    context = row.get("context") or "（未检索到相关文档）"
    return (
        "下面是检索到的文档。若文档足以回答问题，请直接作答；"
        "若文档不足以回答，只回复「无法回答」。\n"
        f"文档：{context}\n问题：{row['question']}"
    )


def score_refusal(preds: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_f1 = []
    gold_refuse = 0
    gold_n = 0
    neg_refuse = 0
    neg_n = 0
    for pred, row in zip(preds, rows):
        refused = is_refusal(pred)
        if row.get("has_gold"):
            gold_n += 1
            gold_f1.append(token_f1(pred, str(row.get("answer") or "")))
            gold_refuse += int(refused)
        else:
            neg_n += 1
            neg_refuse += int(refused)
    refusal_rate = round(neg_refuse / neg_n, 4) if neg_n else None
    false_refuse = round(gold_refuse / gold_n, 4) if gold_n else None
    precision = refusal_rate or 0.0
    recall = refusal_rate or 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n": len(rows),
        "refusal_rate": refusal_rate,
        "gold_f1": round(sum(gold_f1) / gold_n, 4) if gold_n else None,
        "false_refusal_rate": false_refuse,
        "refusal_f1": round(f1, 4),
    }


def evaluate_refusal(
    rows: list[dict[str, Any]],
    base_model: str,
    adapter: str | None,
) -> dict[str, Any]:
    from .sft import generate_answers

    questions = [_prompt(row) for row in rows]
    preds = generate_answers(questions, base_model, adapter=adapter, max_new_tokens=128)
    if isinstance(preds, dict):
        return {"skipped": True, "reason": preds.get("reason", "generate_failed")}
    return score_refusal(preds, rows)
