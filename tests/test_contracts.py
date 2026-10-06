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


def test_write_heldout_keeps_legacy_and_excludes_unreviewed(tmp_path):
    import hashlib
    import json

    from qa_pipeline.experiments.drug_corpus import write_heldout

    legacy = tmp_path / "heldout.jsonl"
    legacy.write_text('{"question":"旧题"}\n', encoding="utf-8")
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    rows = [
        {
            "split": "locked_test",
            "parse_ok": True,
            "source_family_id": "fam-1",
            "stem": "demo",
            "text": "【药品名称】\n通用名称：阿莫西林胶囊\n【用法用量】\n成人一次 0.5 g。\n",
        }
    ]
    path = write_heldout(rows, dest=tmp_path)
    assert path.name == "heldout_protocol.jsonl"
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == digest
    protocol = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [
        json.loads(line)
        for line in (tmp_path / "heldout_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert candidates
    assert protocol == []
    assert all(admits_main_metric(item) is False for item in candidates)


def test_unreviewed_row_stays_out_of_main_partition():
    from qa_pipeline.experiments.sft import partition_heldout

    rows = [
        {
            "id": "u1",
            "question": "阿莫西林胶囊的成人剂量是多少？",
            "answer": "一次 0.5 g",
            "answer_points": ["一次 0.5 g"],
            "validity_status": "self_contained",
            "review_status": "unreviewed",
            "source_family_id": "fam",
        },
        {
            "id": "ok",
            "question": "阿莫西林胶囊的成人剂量是多少？",
            "answer": "一次 0.5 g",
            "validity_status": "self_contained",
            "review_status": "dual_agreed",
            "source_family_id": "fam",
        },
    ]
    eligible, rejected = partition_heldout(rows)
    assert [row["id"] for row in eligible] == ["ok"]
    assert rejected[0]["id"] == "u1"
    assert "not_admitted" in rejected[0]["reasons"]


def test_chat_template_keeps_end_marker_and_rejects_empty_body():
    from qa_pipeline.experiments.sft import encode_supervised_messages

    class Tok:
        eos_token_id = 9

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            ids = []
            for message in messages:
                ids.append(1)
                ids.extend((ord(ch) % 20) + 3 for ch in (message.get("content") or ""))
                ids.append(2)
            if add_generation_prompt:
                ids.append(1)
            return ids

    messages = [
        {"role": "system", "content": "政策"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "成人一次0.5g"},
    ]
    packed = encode_supervised_messages(Tok(), messages, max_length=64)
    assert packed["skipped"] is False
    assert packed["labels"][-1] == 2
    assert packed["labels"].count(-100) < len(packed["labels"])
    empty = [*messages[:-1], {"role": "assistant", "content": ""}]
    rejected = encode_supervised_messages(Tok(), empty, max_length=64)
    assert rejected["skipped"] is True
    assert rejected["reason"] == "no_effective_target"
    too_long = encode_supervised_messages(Tok(), messages, max_length=4)
    assert too_long["skipped"] is True
    assert too_long["reason"] == "prompt_exceeds_budget"
    assert "labels" not in too_long


def test_empty_main_still_runs_train_seen_probe(tmp_path, monkeypatch):
    from qa_pipeline.experiments import sft as sft_mod
    from qa_pipeline.schemas import QAPair

    pair = QAPair(
        qa_id="seen-1",
        question="阿莫西林胶囊的成人剂量是多少？",
        answer="不应被探针使用的答案",
        grade="S",
        chunk_text="成人一次 0.5 g。",
        evidence_span="成人一次 0.5 g。",
        metadata={"assistant_target": "成人一次 0.5 g", "student_context": "错误上下文"},
    )
    heldout = tmp_path / "heldout_protocol.jsonl"
    heldout.write_text("", encoding="utf-8")
    calls = []

    def fake_train(train_jsonl, out_dir, base_model="x", **kwargs):
        return {
            "skipped": False,
            "consumed_ids": ["seen-1"],
            "epochs": 3,
            "global_step": 6,
            "adapter": "adapter",
        }

    def fake_generate(questions, base_model, adapter=None, max_new_tokens=512):
        calls.append(adapter)
        text = "成人一次 0.5 g" if adapter else "不知道"
        return [text for _ in questions]

    monkeypatch.setattr(sft_mod, "train_lora", fake_train)
    monkeypatch.setattr(sft_mod, "generate_answers", fake_generate)
    report = sft_mod.evaluate_sft([pair], heldout, tmp_path / "sft", skip_sft=False, base_model="base")
    assert report["main_metric"] == "not_executed"
    assert report["delta_f1"] is None
    assert report["main_eval"] == "model_not_loaded"
    seen = report["train_seen"]
    assert seen["export_status"] == "exported"
    assert seen["eval_status"] == "executed"
    assert seen["enters_generalization_score"] is False
    assert seen["epochs"] == 3
    assert seen["trainer_global_step"] == 6
    assert calls == [None, "adapter"]
    row = __import__("json").loads((tmp_path / "sft" / "train_seen.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["answer"] == "成人一次 0.5 g"
    assert "错误上下文" in row["messages"][1]["content"]
    assert row["base_answer"] == "不知道"
    assert row["tuned_answer"] == "成人一次 0.5 g"


def test_missing_consumed_ids_do_not_run_probe(tmp_path, monkeypatch):
    from qa_pipeline.experiments import sft as sft_mod
    from qa_pipeline.schemas import QAPair

    pair = QAPair(qa_id="seen-1", question="剂量？", answer="一次 0.5 g", grade="S", chunk_text="一次 0.5 g")
    heldout = tmp_path / "heldout_protocol.jsonl"
    heldout.write_text("", encoding="utf-8")

    def fake_train(*args, **kwargs):
        return {"skipped": False, "consumed_ids": [], "epochs": 3, "global_step": 0, "adapter": "adapter"}

    def fake_generate(*args, **kwargs):
        raise AssertionError("没有消费清单时不应加载模型做探针")

    monkeypatch.setattr(sft_mod, "train_lora", fake_train)
    monkeypatch.setattr(sft_mod, "generate_answers", fake_generate)
    report = sft_mod.evaluate_sft([pair], heldout, tmp_path / "sft", skip_sft=False)
    assert report["train_seen"]["export_status"] == "not_executed"
    assert report["train_seen"]["eval_status"] == "not_executed"
    assert report["train_seen"]["reason"] == "missing_actually_trained"
    assert report["main_metric"] == "not_executed"


def test_serial_and_parallel_heldout_gate(tmp_path):
    from pathlib import Path

    import yaml

    from qa_pipeline.experiments.parallel import run_parallel
    from qa_pipeline.experiments.runner import gate_heldout, run_suite

    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert gate_heldout(missing)["ok"] is False
    assert gate_heldout(missing)["reason"] == "missing_heldout"
    assert gate_heldout(empty) == gate_heldout(empty)
    assert gate_heldout(empty)["ok"] is True
    assert gate_heldout(empty)["heldout_status"] == "empty"

    def suite_file(heldout: Path, name: str) -> Path:
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "run_id": name,
                    "heldout": str(heldout),
                    "input": "fixtures/sample_manual.md",
                    "experiments": [{"id": "E_skip", "kind": "skipped", "reason": "未执行"}],
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return path

    missing_out = tmp_path / "missing_out"
    serial_missing = run_suite(suite_file(missing, "missing"), out_dir=missing_out, fake=True, skip_sft=True)
    parallel_missing = run_parallel(
        suite={"name": "missing", "experiments": [{"id": "E_skip", "kind": "skipped", "reason": "未执行"}]},
        out=tmp_path / "parallel_missing",
        recipes_dir=tmp_path,
        input_path=tmp_path,
        heldout=missing,
        refusal_path=tmp_path / "refusal.jsonl",
        model_path="none",
        student="none",
        devices=[0],
        skip_sft=True,
        notes=[],
    )
    assert serial_missing["aborted"] is True
    assert parallel_missing["aborted"] is True
    assert serial_missing["reason"] == parallel_missing["reason"] == "missing_heldout"
    assert not missing_out.exists()
    assert not (tmp_path / "parallel_missing").exists()

    serial_empty = run_suite(suite_file(empty, "empty"), out_dir=tmp_path / "empty_out", fake=True, skip_sft=True)
    parallel_empty = run_parallel(
        suite={"name": "empty", "run_id": "empty", "experiments": [{"id": "E_skip", "kind": "skipped", "reason": "未执行"}]},
        out=tmp_path / "parallel_empty",
        recipes_dir=tmp_path,
        input_path=tmp_path,
        heldout=empty,
        refusal_path=tmp_path / "refusal.jsonl",
        model_path="none",
        student="none",
        devices=[0],
        skip_sft=True,
        notes=[],
    )
    assert serial_empty.get("aborted") is not True
    assert parallel_empty.get("aborted") is not True
    assert any("主评测不加载模型" in item for item in serial_empty["limitations"])
    assert any("主评测不加载模型" in item for item in parallel_empty["limitations"])


def test_validation_suite_trains_only_representative_arm():
    from qa_pipeline.experiments.runner import load_suite

    suite = load_suite("configs/experiments/suite_drug_v22_validate.yaml")
    assert suite["run_id"] == "drug_v22"
    trained = [item["id"] for item in suite["experiments"] if item.get("sft")]
    assert trained == ["E3_g0"]
    g0 = load_recipe("configs/recipes/drug/e3_g0.yaml")
    direct = load_recipe("configs/recipes/drug/e1_direct.yaml")
    assert g0.filters == []
    assert "student_probe" not in g0.extra
    assert "retention_pool" not in g0.extra
    assert "catalog_path" not in direct.extra
    notes = "\n".join(suite["notes"])
    assert "学生诊断未执行" in notes
    assert "第一种回放干预未执行" in notes
    assert "补题未执行" in notes
    assert "第二种回放预算未检验" in notes


def test_trainable_gpu_requires_free_memory():
    from qa_pipeline.experiments.devices import parse_trainable_gpus

    text = "0, 18, 8000\n1, 20, 20000\n2, 9000, 20000\n"
    assert parse_trainable_gpus(text) == [1]


def test_select_only_keeps_intervention_on_frozen_pool():
    recipe = load_recipe("configs/recipes/drug/e3_g0.yaml")
    parent = _pair(grade="S", data_stage="accepted")
    result = Pipeline(recipe, llm=FakeLLM()).select_only([parent])
    assert result.pairs
    assert result.pairs[0].data_stage == "selected"
    assert result.pairs[0].qa_id == parent.qa_id
