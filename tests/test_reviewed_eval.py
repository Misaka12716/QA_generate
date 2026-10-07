"""已审核评测编排：未授权不调用生成，金标变化不误用旧评分。"""

from __future__ import annotations

from qa_pipeline.experiments.reviewed_eval import execute_reviewed_batch
from qa_pipeline.experiments.scoring import prediction_item_signature, score_item_signature
from qa_pipeline.experiments.v25_prepare import maybe_execute_batch
from qa_pipeline.reviewing.identity import subject_hash
from qa_pipeline.reviewing.policy import ReviewPolicy, build_aggregate
from qa_pipeline.reviewing.schemas import ReviewAggregate
from qa_pipeline.reviewing.service import normalize_subject
from qa_pipeline.reviewing.teacher import make_record


def _policy() -> ReviewPolicy:
    return ReviewPolicy(review_policy_id="teacher_only_v1", policy_version="1", review_mode="teacher_only")


def _case(**overrides) -> dict:
    row = {
        "case_id": "case-1",
        "question": "颜色是什么？",
        "context": "物品为红色。",
        "answer": "红色",
        "expected_action": "answer",
        "messages": [
            {"role": "system", "content": "只依据资料回答"},
            {"role": "user", "content": "资料：物品为红色。\n\n问题：颜色是什么？"},
        ],
    }
    row.update(overrides)
    return row


def _accepted_aggregate(case: dict) -> ReviewAggregate:
    subject = normalize_subject(case, "reviewed_eval")
    digest = subject_hash(subject)
    records = []
    for model_id, revision in (("model-a", "rev-a"), ("model-b", "rev-b")):
        records.append(
            make_record(
                subject=subject,
                policy=_policy(),
                judge={
                    "provider_id": "fake",
                    "model_id": model_id,
                    "model_revision": revision,
                    "judge_role": "gold_reviewer",
                    "independence_level": "distinct_family",
                    "model_identity_verified": True,
                },
                review_set_id="set_test",
                prompt_digest="prompt",
                inference_digest="infer",
                decision="accept",
                execution_status="succeeded",
                reason_codes=[],
                parsed={
                    "dimensions": {
                        "question_clear": True,
                        "semantic_preserved": True,
                        "evidence_adequate": True,
                        "factual_consistency": True,
                        "required_points_complete": True,
                        "behavior_appropriate": True,
                        "no_unsupported_claims": True,
                    },
                    "claims": [
                        {
                            "claim_id": "c1",
                            "claim_text": "物品为红色",
                            "status": "supported",
                            "visible_evidence_refs": ["0:4"],
                            "source_evidence_refs": [],
                        }
                    ],
                    "contradictions": [],
                    "evidence_summary": "可见",
                },
            )
        )
    return build_aggregate(records, subject_id=case["case_id"], expected_hash=digest, policy=_policy(), risk="high")


def test_unauthorized_inference_does_not_call_generate():
    calls = {"n": 0}

    def generate(_pending):
        calls["n"] += 1
        return {"executed": True, "model_loaded": True}

    blocked = maybe_execute_batch(ready=True, allow_inference=True, generate_fn=generate, inference_authorized=False)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "inference_budget_missing"
    assert blocked["model_loaded"] is False
    assert calls["n"] == 0
    case = _case()
    report = execute_reviewed_batch(
        cases=[case],
        aggregates={case["case_id"]: _accepted_aggregate(case)},
        policy=_policy(),
        mode="exploratory",
        allow_inference=True,
        inference_authorized=False,
        generate_fn=generate,
    )
    assert report["executed"] is False
    assert report["model_loaded"] is False
    assert calls["n"] == 0


def test_formal_mode_never_loads_model():
    calls = {"n": 0}

    def generate(_pending):
        calls["n"] += 1
        return {"executed": True, "model_loaded": True}

    case = _case()
    report = execute_reviewed_batch(
        cases=[case],
        aggregates={case["case_id"]: _accepted_aggregate(case)},
        policy=_policy(),
        mode="formal",
        allow_inference=True,
        inference_authorized=True,
        generate_fn=generate,
    )
    assert report["status"] == "blocked"
    assert report["reason"] == "formal_requires_human_gold"
    assert calls["n"] == 0


def test_gold_change_keeps_prediction_identity_and_changes_score_identity():
    case = _case()
    infer = {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0, "top_p": 1.0}
    before = prediction_item_signature(case, "base-id", "", "template", infer, "base")
    score_before = score_item_signature(before, case, "aux-rules-v3")
    changed = _case(answer="蓝色")
    after = prediction_item_signature(changed, "base-id", "", "template", infer, "base")
    score_after = score_item_signature(after, changed, "aux-rules-v3")
    assert before == after
    assert score_before != score_after
    bare = {key: value for key, value in case.items() if key != "messages"}
    bare_before = prediction_item_signature(bare, "base-id", "", "template", infer, "base")
    bare_question = dict(bare)
    bare_question["question"] = "重量是多少？"
    assert prediction_item_signature(bare_question, "base-id", "", "template", infer, "base") != bare_before
    message_changed = _case()
    message_changed["messages"] = [
        {"role": "system", "content": "只依据资料回答"},
        {"role": "user", "content": "资料：物品为红色。\n\n问题：重量是多少？"},
    ]
    assert prediction_item_signature(message_changed, "base-id", "", "template", infer, "base") != before


def test_incomplete_group_does_not_drop_cases():
    calls = {"n": 0}

    def generate(_pending):
        calls["n"] += 1
        return {"executed": True, "model_loaded": True}

    case = _case()
    report = execute_reviewed_batch(
        cases=[case],
        aggregates={case["case_id"]: _accepted_aggregate(case)},
        policy=_policy(),
        mode="exploratory",
        allow_inference=True,
        inference_authorized=True,
        generate_fn=generate,
        planned_n=2,
    )
    assert report["status"] == "blocked"
    assert report["reason"] == "incomplete_group"
    assert report["planned_n"] == 2
    assert report["dropped_case_ids"] == []
    assert calls["n"] == 0
