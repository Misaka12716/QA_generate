"""正式实验前的确定性反例。只使用 FakeLLM，不调用真实模型。"""

import random

import pytest

from qa_pipeline.adapters.zhixun import ReleaseRejected, bind_validation_hash, export_zhixun
from qa_pipeline.config import load_recipe
from qa_pipeline.experiments.drug_corpus import (
    admits_main_metric,
    apply_evidence_condition,
    cluster_documents,
)
from qa_pipeline.experiments.scoring import (
    AlignmentError,
    aggregate_families,
    align_by_id,
    build_supervised_batch,
    cache_signature,
    classify_action,
    critical_errors,
    score_task,
)
from qa_pipeline.textutil import token_f1
from qa_pipeline.llm import FakeLLM
from qa_pipeline.pipeline import Pipeline, PipelineContext
from qa_pipeline.plugins.filters import ClaimEvidence, DiversitySample, NaturalDistribution, ReplayReplace, RiskEscalate
from qa_pipeline.plugins.question_gen import review_gold
from qa_pipeline.schemas import QAPair

ROOT_RECIPE = "configs/recipes/smoke.yaml"


def _ctx(llm=None):
    recipe = load_recipe(ROOT_RECIPE)
    return PipelineContext(recipe=recipe, llm=llm or FakeLLM(), rng=random.Random(0))


def _pair(**kwargs) -> QAPair:
    base = dict(
        question="阿莫西林胶囊的成人剂量是多少？",
        answer="一次 0.5 g",
        answer_points=["一次 0.5 g"],
        evidence_span="一次 0.5 g",
        chunk_text="阿莫西林胶囊成人一次 0.5 g。",
        evidence_state="sufficient",
        expected_action="answer",
        action="pass",
        grade="A",
        claims=[{"text": "一次 0.5 g", "status": "supported"}],
    )
    base.update(kwargs)
    return QAPair(**base)


def test_dangling_question_stays_out_of_main_metric():
    item = {
        "question": "根据给定资料，说明这一句表述的具体内容。",
        "context": "一次 0.5 g。肾功能不全者减量。",
        "validity_status": "self_contained",
        "review_status": "agreed",
    }
    assert admits_main_metric(item) is False


def test_mismatched_gold_fails_review():
    reasons = review_gold(
        "阿莫西林胶囊的成人常用剂量是多少",
        "布洛芬缓释胶囊应整片吞服",
        ["布洛芬缓释胶囊应整片吞服"],
    )
    assert reasons == ["gold_mismatch"]
    assert admits_main_metric(
        {
            "question": "阿莫西林胶囊的成人常用剂量是多少",
            "validity_status": "self_contained",
            "review_status": "agreed",
            "gold_mismatch": True,
        }
    ) is False


def test_same_insert_different_filenames_share_source_family():
    text = "【药品名称】\n通用名称：阿莫西林胶囊\n【规格】\n0.25g\n"
    rows = [
        {"stem": "copy_a.pdf", "text": text, "parse_ok": True},
        {"stem": "copy_b.pdf", "text": text, "parse_ok": True},
    ]
    cluster_documents(rows)
    assert rows[0]["source_family_id"] == rows[1]["source_family_id"]
    assert rows[0]["split"] == rows[1]["split"]
    assert rows[1]["dup_kind"] == "exact"


def test_other_drug_without_equivalent_support_is_missing():
    item = {
        "id": "dose",
        "question": "阿莫西林胶囊的成人剂量是多少？",
        "answer_points": ["一次0.5g"],
        "support_spans": ["一次0.5g"],
        "validity_status": "self_contained",
        "review_status": "agreed",
    }
    out = apply_evidence_condition(item, "missing", "布洛芬缓释胶囊一次0.3g。")
    assert out["evidence_state"] == "missing"
    assert out["expected_action"] == "state_insufficient"


def test_equivalent_span_keeps_support():
    item = {
        "id": "dose",
        "question": "阿莫西林胶囊的成人剂量是多少？",
        "answer_points": ["一次0.5g"],
        "support_spans": ["已被删除的另一处", "成人一次0.5g"],
        "validity_status": "self_contained",
        "review_status": "agreed",
    }
    visible = "饭后服用，成人一次0.5g。"
    out = apply_evidence_condition(item, "missing", visible, ["成人一次0.5g"])
    assert out["evidence_state"] == "sufficient"
    assert out["expected_action"] == "answer"


def test_prompt_that_fills_the_window_is_not_eos_only():
    packed = build_supervised_batch(list(range(20)), [7, 8, 9], max_length=10, eos_id=1)
    assert packed["skipped"] is True
    assert packed["reason"] == "prompt_exceeds_budget"
    assert "labels" not in packed
    fitted = build_supervised_batch([1, 2], [7, 8], max_length=8, eos_id=1)
    assert fitted["labels"] == [-100, -100, 7, 8]
    assert fitted["labels"] != [1]


def test_numeric_risk_escalates_and_failed_recheck_quarantines():
    llm = FakeLLM()
    llm.fail_mode = "parse_failed"
    pair = _pair()
    claimed = ClaimEvidence().run([pair], _ctx(llm))[0]
    assert claimed.action == "needs_escalation"
    checked = RiskEscalate().run(claimed and [claimed], _ctx(llm))[0]
    assert checked.metadata["escalation_count"] == 1
    assert checked.metadata["escalation_kind"] == "second_pass"
    assert checked.grade == "quarantine"


def test_true_but_off_question_fails_task():
    scored = score_task("本品应避光保存。", {"answer": "一次 0.5 g", "answer_points": ["一次 0.5 g"], "expected_action": "answer"})
    assert scored["passed"] is False


def test_changed_number_is_critical_despite_overlap():
    gold = "成人一次 0.5 g"
    pred = "成人一次 5 g"
    assert "numeric" in critical_errors(pred, [gold])
    assert token_f1(pred, gold) > 0.5
    assert score_task(pred, {"answer": gold, "answer_points": [gold], "expected_action": "answer"})["passed"] is False


def test_instruction_echo_is_not_a_refusal():
    text = "资料不足时应说明，成人一次 0.5 g。"
    assert classify_action(text) == "answer"


def test_ten_paraphrases_count_as_one_family():
    cases = [
        {"family_id": "fam", "source_family_id": "src", "passed": True, "id": str(i)}
        for i in range(10)
    ]
    summary = aggregate_families(cases)
    assert summary["family_count"] == 1


def test_content_change_invalidates_validation_and_cache():
    pair = _pair(grade="S")
    first = bind_validation_hash(pair)
    signature = cache_signature({"id": pair.qa_id, "answer": pair.answer, "hash": first})
    pair.answer = "一次 1 g"
    with pytest.raises(ReleaseRejected):
        bind_validation_hash(pair)
    assert cache_signature({"id": pair.qa_id, "answer": pair.answer, "hash": first}) != signature


def test_quarantine_export_is_rejected(tmp_path):
    pair = _pair(grade="quarantine", action="quarantine")
    with pytest.raises(ReleaseRejected):
        export_zhixun([pair], tmp_path / "out.jsonl")


def test_missing_prediction_or_distractor_raises():
    rows = [{"case_id": "a", "require_distractor": True}, {"case_id": "b"}]
    with pytest.raises(AlignmentError, match="缺失预测"):
        align_by_id(rows, {"a": "回答"})
    with pytest.raises(AlignmentError, match="缺失干扰上下文"):
        align_by_id(rows, {"a": "回答", "b": "回答"})


def test_bad_json_is_technical_failure_not_unanswerable():
    llm = FakeLLM()
    llm.fail_mode = "parse_failed"
    response = llm.chat_json([{"role": "user", "content": "问题"}], model="fake")
    assert response.status == "parse_failed"
    llm.fail_mode = "transport_failed"
    timed_out = llm.chat_json([{"role": "user", "content": "问题"}], model="fake")
    assert timed_out.status == "transport_failed"
    assert timed_out.status != "semantic_rejected"


def test_quota_and_replay_change_supervision():
    pairs = [_pair(q_type="factual", source_doc=f"doc{i}", qa_id=f"q{i}") for i in range(6)]
    quota = DiversitySample(quota={"factual": 0.4}, per_doc_cap=1).run([p.model_copy(deep=True) for p in pairs], _ctx())
    natural = NaturalDistribution().run([p.model_copy(deep=True) for p in pairs], _ctx())
    assert sum(p.included_in_this_run is True for p in quota) < sum(p.included_in_this_run is True for p in natural)
    assert any(p.exclude_reason == "quota" for p in quota)

    pool = [_pair(qa_id="old", selection_role="retention")]
    replayed = ReplayReplace(fraction=0.2).run([p.model_copy(deep=True) for p in pairs], _ctx())
    assert all(item.get("status") == "not_executed" for item in (p.filter_trace["replay_replace"] for p in replayed))
    ctx = _ctx()
    ctx.extras["retention_pairs"] = pool
    ctx.recipe.extra["retention_control"] = "random_delete"
    changed = ReplayReplace(fraction=0.2).run([p.model_copy(deep=True) for p in pairs], ctx)
    assert any(p.selection_role == "retention" for p in changed)
    assert any(p.exclude_reason == "random_delete_control" for p in changed)


def test_select_only_keeps_intervention_on_frozen_pool():
    recipe = load_recipe("configs/recipes/drug/e3_g0.yaml")
    parent = _pair(grade="S", data_stage="accepted")
    result = Pipeline(recipe, llm=FakeLLM()).select_only([parent])
    assert result.pairs
    assert result.pairs[0].data_stage == "selected"
    assert result.pairs[0].qa_id == parent.qa_id
