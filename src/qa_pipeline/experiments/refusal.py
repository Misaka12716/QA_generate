"""按 expected_action 评估行为，不再把负例拒答率当成 F1。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scoring import AlignmentError, binary_refusal_f1, classify_action, confusion
from .sft import eval_messages, generate_answers


def load_refusal_set(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def is_refusal(text: str) -> bool:
    return classify_action(text) in {"state_insufficient", "state_conflict"}


def score_refusal(preds: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(preds) != len(rows):
        raise AlignmentError(f"预测 {len(preds)} 与样本 {len(rows)} 数量不一致")
    matrix = confusion(rows, preds)
    binary = binary_refusal_f1(rows, preds)
    return {
        "n": len(rows),
        "confusion": matrix["matrix"],
        "refusal_f1": binary["f1"],
        "refusal_precision": binary["precision"],
        "refusal_recall": binary["recall"],
        "refusal_detail": binary,
        "reason": binary["reason"],
    }


def evaluate_refusal(
    rows: list[dict[str, Any]],
    base_model: str,
    adapter: str | None,
) -> dict[str, Any]:
    prompts = [eval_messages(row) for row in rows]
    preds = generate_answers(prompts, base_model, adapter=adapter, max_new_tokens=512)
    if isinstance(preds, dict):
        return {"skipped": True, "reason": preds.get("reason", "generate_failed")}
    return score_refusal(preds, rows)
