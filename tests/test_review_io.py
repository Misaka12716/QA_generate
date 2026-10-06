"""审核导入不能把空表或单人意见写成通过或双人一致。"""

from __future__ import annotations

import json

from qa_pipeline.experiments.adapter_eval import review_content_hash
from qa_pipeline.experiments.review_io import (
    BLIND_COLUMNS,
    GOLD_COLUMNS,
    gold_row_from_case,
    import_blind_csv,
    import_gold_csv,
    render_blind_markdown,
    write_csv,
    write_fill_example,
)


def _case() -> dict:
    case = {
        "case_id": "ex_001",
        "condition_id": "original",
        "core_question_id": "core_001",
        "family_id": "core_001",
        "question": "示例物品A是什么颜色？",
        "context": "示例物品A为蓝色。",
        "messages": [
            {"role": "system", "content": "只根据可见资料回答。"},
            {"role": "user", "content": "资料：\n示例物品A为蓝色。\n\n问题：示例物品A是什么颜色？"},
        ],
        "answer": "蓝色。",
        "required_points": ["蓝色。"],
        "unavailable_points": [],
        "expected_action": "answer",
        "evidence_state": "sufficient",
        "review_status": "pending_review",
    }
    return case


def _filled(row: dict, **kwargs) -> dict:
    row = dict(row)
    row.update(
        {
            "question_clear": "yes",
            "semantic_preserved": "na",
            "visible_evidence_adequate": "yes",
            "required_points_ok": "yes",
            "unavailable_points_ok": "na",
            "expected_behavior_ok": "yes",
            "reviewer_a": "审核者甲",
            "opinion_a": "题干和可见资料一致。",
            "adjudication": "accept",
        }
    )
    row.update(kwargs)
    return row


def test_blank_gold_stays_pending_and_example_has_no_real_drug(tmp_path):
    case = _case()
    row = gold_row_from_case(case, batch_id="batch", ai_note="模型建议通过")
    path = tmp_path / "gold.csv"
    write_csv(path, GOLD_COLUMNS, [row])
    result = import_gold_csv(path, [case], batch_id="batch")
    assert result["ready_for_inference"] is False
    assert result["pending_review_n"] == 1
    assert result["adjudicated_n"] == 0
    assert result["cases"][0]["review_status"] == "pending_review"
    example = write_fill_example(tmp_path / "examples")
    text = example.read_text(encoding="utf-8")
    assert "示例物品A" in text
    assert "卡马西平" not in text


def test_single_reviewer_is_not_dual_agreed(tmp_path):
    case = _case()
    row = _filled(gold_row_from_case(case, batch_id="batch"), adjudication="accept")
    path = tmp_path / "gold.csv"
    write_csv(path, GOLD_COLUMNS, [row])
    result = import_gold_csv(path, [case], batch_id="batch")
    assert result["ready_for_inference"] is True
    assert result["cases"][0]["review_status"] == "agreed"
    assert result["cases"][0]["review_status"] != "dual_agreed"
    assert result["cases"][0]["review_content_hash"] == review_content_hash(result["cases"][0])


def test_two_reviewers_can_be_dual_and_hash_mismatch_blocks(tmp_path):
    case = _case()
    row = _filled(
        gold_row_from_case(case, batch_id="batch"),
        reviewer_b="审核者乙",
        opinion_b="同意。",
    )
    path = tmp_path / "gold.csv"
    write_csv(path, GOLD_COLUMNS, [row])
    agreed = import_gold_csv(path, [case], batch_id="batch")
    assert agreed["cases"][0]["review_status"] == "dual_agreed"
    broken = _filled(gold_row_from_case(case, batch_id="batch"), content_hash="changed")
    write_csv(path, GOLD_COLUMNS, [broken])
    rejected = import_gold_csv(path, [case], batch_id="batch")
    assert rejected["ready_for_inference"] is False
    assert rejected["error_n"] == 1
    assert rejected["pending_review_n"] == 1
    assert rejected["cases"][0]["review_status"] == "pending_review"


def test_blind_sheet_hides_model_identity(tmp_path):
    identity = [{"blind_id": "b1", "case_id": "ex_001", "model_role": "adapter", "model_id": "secret-model", "answer_text": "蓝色。"}]
    from qa_pipeline.experiments.review_io import blind_content_hash

    row = {
        "batch_id": "blind",
        "blind_id": "b1",
        "content_hash": blind_content_hash("b1", "蓝色。"),
        "answer_text": "蓝色。",
        "task_completed": "",
        "key_factual_errors": "",
        "unsupported_content": "",
        "behavior_appropriate": "",
        "reviewer_a": "",
        "opinion_a": "",
        "reviewer_b": "",
        "opinion_b": "",
        "adjudication": "",
        "ai_note": "",
    }
    path = tmp_path / "blind.csv"
    write_csv(path, BLIND_COLUMNS, [row])
    text = path.read_text(encoding="utf-8")
    markdown = render_blind_markdown([row], batch_id="blind")
    assert "adapter" not in text
    assert "secret-model" not in text
    assert "model_role" not in markdown
    result = import_blind_csv(path, identity, batch_id="blind")
    assert result["pending_review_n"] == 1
    assert result["ready_for_inference"] is False
    leaked = dict(row, model_role="adapter")
    write_csv(path, BLIND_COLUMNS + ("model_role",), [leaked])
    leaked_result = import_blind_csv(path, identity, batch_id="blind")
    assert leaked_result["error_n"] == 1
    assert leaked_result["pending_review_n"] == 1
