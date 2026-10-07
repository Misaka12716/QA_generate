"""闭卷模式、审核身份和结果页口径。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa_pipeline.experiments.adapter_eval import canonical_messages
from qa_pipeline.experiments.closed_book import audit_training_row, prepare_closed_book, surface_paraphrases
from qa_pipeline.experiments.exploratory_review import prediction_subject
from qa_pipeline.experiments.result_report import compare_outcomes
from qa_pipeline.experiments.sft import eval_messages
from qa_pipeline.experiments.workbench import comparison_cases, comparison_view
from qa_pipeline.reviewing.identity import gold_hash, model_input_hash, review_cache_key, subject_hash
from qa_pipeline.reviewing.policy import ReviewPolicy, build_aggregate, locate_evidence_ref
from qa_pipeline.reviewing.teacher import build_messages, make_record
from qa_pipeline.task_mode import CLOSED_BOOK, TaskModeError


def _subject(**extra):
    row = {
        "subject_type": "prediction",
        "subject_id": "case::base",
        "subject_version": "1",
        "question": "剂量是多少？",
        "student_context": "",
        "answer_text": "一次一片",
        "task_mode": "closed_book",
        "expected_action": "answer",
    }
    row.update(extra)
    return row


def test_answer_text_changes_review_identity_not_student_input():
    before = _subject()
    after = _subject(answer_text="一次两片")
    assert subject_hash(before) != subject_hash(after)
    assert model_input_hash(before) == model_input_hash(after)
    reference = _subject(reference_answer="一次一片", required_points=["一次一片"])
    changed_reference = _subject(reference_answer="一次两片", required_points=["一次两片"])
    assert model_input_hash(reference) == model_input_hash(changed_reference)
    assert subject_hash(reference) != subject_hash(changed_reference)
    assert gold_hash(reference) != gold_hash(changed_reference)
    left = review_cache_key(
        subject_hash_value=subject_hash(reference),
        review_kind="prediction",
        model_id="teacher",
        model_revision="unknown",
        prompt_digest="p",
        rubric_version="rubric-v1",
        policy_version="1",
        evidence_hash="e",
        judge_role="gold_reviewer",
        reference_hash=gold_hash(reference),
    )
    right = review_cache_key(
        subject_hash_value=subject_hash(changed_reference),
        review_kind="prediction",
        model_id="teacher",
        model_revision="unknown",
        prompt_digest="p",
        rubric_version="rubric-v1",
        policy_version="1",
        evidence_hash="e",
        judge_role="gold_reviewer",
        reference_hash=gold_hash(changed_reference),
    )
    assert left != right


def test_task_mode_changes_cache_and_closed_book_alias_matches():
    closed = _subject(task_mode="closed_book")
    domain = _subject(task_mode="closed_book_domain")
    rag = _subject(task_mode="rag_grounded", student_context="资料")
    assert subject_hash(closed) == subject_hash(domain)
    assert subject_hash(closed) != subject_hash(rag)


def test_prediction_request_carries_mode_and_points_without_hiding_answer_text():
    case = {
        "case_id": "c1",
        "question": "剂量是多少？",
        "student_context": "",
        "candidate_answer": "一次一片",
        "expected_action": "answer",
        "task_mode": "closed_book_domain",
        "condition_id": "full",
        "required_points": ["一次一片"],
        "unavailable_points": ["儿童剂量"],
        "answers": {"base": {"text": "不清楚"}},
    }
    payload = json.loads(build_messages(prediction_subject(case, "base", "batch"), prompt_text="审核")[1]["content"].split("<subject>", 1)[1].split("</subject>", 1)[0])
    assert payload["task_mode"] == CLOSED_BOOK
    assert payload["condition_id"] == "full"
    assert payload["required_points"] == ["一次一片"]
    assert payload["unavailable_points"] == ["儿童剂量"]
    assert payload["review_mode"] == "reference_based"
    assert payload["answer_text"] == "不清楚"
    assert not payload["answer"]
    assert payload["reference_answer"] == "一次一片"


def test_closed_book_messages_drop_context_and_reject_conflicts():
    row = {"question": "二甲双胍的主要风险是什么？", "context": "这是一段不应该进入学生输入的说明书正文。", "task_mode": "closed_book_domain", "answer": "乳酸酸中毒"}
    messages = eval_messages(row)
    assert messages[0]["content"].startswith("可以运用已有领域知识")
    assert messages[1]["content"] == row["question"]
    assert "说明书" not in json.dumps(messages, ensure_ascii=False)
    assert eval_messages({**row, "task_mode": "closed_book"}) == messages
    conflicted = {**row, "messages": [{"role": "system", "content": "仅依据提供的资料回答；资料不足时说明缺少的信息。"}, {"role": "user", "content": "资料：\n" + row["context"]}]}
    with pytest.raises(TaskModeError):
        canonical_messages(conflicted)


def test_non_locating_citation_is_not_evidence():
    assert locate_evidence_ref("随便写的编号", "物品为红色。") is None
    assert locate_evidence_ref("0:4", "物品为红色。") == "物品为红"
    policy = ReviewPolicy(review_policy_id="p", policy_version="1", review_mode="teacher_only", allow_single=True, require_dual=False)
    subject = {"subject_type": "protocol_case", "subject_id": "s", "question": "颜色？", "student_context": "物品为红色。", "answer": "红色", "task_mode": "rag_grounded"}
    parsed = {
        "dimensions": {
            "evidence_adequate": True,
            "factual_consistency": True,
            "behavior_appropriate": True,
            "no_unsupported_claims": True,
            "required_points_complete": True,
        },
        "claims": [{"claim_id": "c", "claim_text": "物品为红色", "status": "supported", "visible_evidence_refs": ["e1"], "source_evidence_refs": []}],
        "contradictions": [],
        "reason_codes": [],
    }
    record = make_record(
        subject=subject,
        policy=policy,
        judge={"provider_id": "fake", "model_id": "m", "model_revision": "r", "judge_role": "gold_reviewer", "independence_level": "distinct_family", "model_identity_verified": True},
        review_set_id="set",
        prompt_digest="p",
        inference_digest="i",
        decision="accept",
        execution_status="succeeded",
        reason_codes=[],
        parsed=parsed,
    )
    aggregate = build_aggregate([record], subject_id="s", expected_hash=subject_hash(subject), policy=policy, task_mode="rag_grounded", student_text="物品为红色。")
    assert aggregate.accepted_by_policy is False
    parsed["claims"][0]["visible_evidence_refs"] = ["物品为红色"]
    located = make_record(
        subject=subject,
        policy=policy,
        judge={"provider_id": "fake", "model_id": "m", "model_revision": "r", "judge_role": "gold_reviewer", "independence_level": "distinct_family", "model_identity_verified": True},
        review_set_id="set",
        prompt_digest="p",
        inference_digest="i",
        decision="accept",
        execution_status="succeeded",
        reason_codes=[],
        parsed=parsed,
    )
    accepted = build_aggregate([located], subject_id="s", expected_hash=subject_hash(subject), policy=policy, task_mode="rag_grounded", student_text="物品为红色。")
    assert accepted.teacher_review_status == "teacher_single_accepted"


def test_v2_outcomes_use_task_success_not_legacy_tie():
    def judgment(success: bool):
        return {"rank": 3, "legacy_rank": 3, "task_success": success}

    assert compare_outcomes(judgment(False), judgment(True))["relation"] == "wrong_to_right"
    assert compare_outcomes(judgment(True), judgment(False))["relation"] == "right_to_wrong"
    assert compare_outcomes(judgment(True), judgment(True))["relation"] == "both_correct"


def test_closed_book_page_does_not_reuse_rag_scores(tmp_path: Path):
    metrics = {
        "budget": {"calls": 97, "call_denominator": 120, "tokens": 107092, "token_denominator": 200000},
        "historical": {"comparison": {"denominator": 14, "improve": 0, "regress": 0, "tie": 10, "unresolved": 4}, "per_model": {"base": {"scored": 11, "task_complete": 11}, "adapter": {"scored": 10, "task_complete": 10}}, "gold": {"formal_ready": 0}, "rule_disagreements": [{"case_id": "only_id"}]},
        "b_pilot": {"parent_planned_n": 48, "selected_n": 8, "comparison": {"denominator": 8, "improve": 0, "regress": 0, "tie": 8, "unresolved": 0}, "per_model": {"base": {"scored": 8, "task_complete": 8}, "adapter": {"scored": 8, "task_complete": 8}}, "gold": {"formal_ready": 0}, "rule_disagreements": []},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    historical = comparison_view(tmp_path, task_mode="rag_grounded", batch="historical")
    assert historical["legacy_relation"]["tie"] == 10
    assert historical["legacy_relation"]["unresolved"] == 4
    assert historical["outcomes"]["both_correct"] is None
    pilot = comparison_view(tmp_path, task_mode="rag_grounded", batch="b_pilot")
    assert pilot["cards"][0]["numerator"] == 8
    assert pilot["cards"][0]["denominator"] == 48
    closed = comparison_view(tmp_path, task_mode="closed_book_domain", batch="cb1")
    assert closed["headline"] == "尚无闭卷训练对照"
    assert closed["cards"] == []
    assert closed["outcomes"]["wrong_to_right"] is None
    missing = comparison_cases(tmp_path, task_mode="rag_grounded", batch="historical", relation="unresolved")
    assert missing["missing_reason"] == "仅有汇总，案例产物缺失"
    assert missing["data"] == []


def test_prepare_drops_deictic_questions_and_keeps_paraphrases_out_of_train(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    rows = [
        {
            "id": "qa_named",
            "messages": [
                {"role": "user", "content": "资料：\n长文\n\n问题：二甲双胍的化学名称是什么？"},
                {"role": "assistant", "content": "一种双胍类化合物的完整化学名"},
            ],
            "metadata": {"source_family_id": "fam", "evidence": "化学名称为一种双胍类化合物的完整化学名。"},
        },
        {
            "id": "qa_deictic",
            "messages": [
                {"role": "user", "content": "问题：根据资料，该药物有哪些功效？"},
                {"role": "assistant", "content": "功效"},
            ],
            "metadata": {"source_family_id": "fam", "evidence": "功效正文"},
        },
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert audit_training_row(rows[1])["eligible"] is False
    assert len(surface_paraphrases("二甲双胍的化学名称是什么？", "一种双胍类化合物的完整化学名")) == 2
    retention = tmp_path / "retention.jsonl"
    retention.write_text(json.dumps({"case_id": "ret", "question": "一天有多少小时？", "answer": "24"}, ensure_ascii=False) + "\n", encoding="utf-8")
    protocol = prepare_closed_book(source=source, out_dir=tmp_path / "cb", retention_path=retention, base_model="/models/example", ledger={"calls": 97, "call_denominator": 100, "tokens": 1, "token_denominator": 2})
    train = [json.loads(line) for line in (tmp_path / "cb" / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 1
    assert train[0]["messages"][1]["content"] == "二甲双胍的化学名称是什么？"
    assert "资料" not in train[0]["messages"][1]["content"]
    paraphrases = [json.loads(line) for line in (tmp_path / "cb" / "splits" / "cb_paraphrase.jsonl").read_text(encoding="utf-8").splitlines()]
    assert paraphrases
    assert all(item["question"] not in {row["messages"][1]["content"] for row in train} for item in paraphrases)
    assert protocol["teacher_budget"]["status"] == "blocked_teacher_budget"
    assert protocol["eval_planned"]["CB-memory"] == 1
