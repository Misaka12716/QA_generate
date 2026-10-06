"""评测可靠性契约。只用临时样例，不读取 runs/ 或服务器数据。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa_pipeline.experiments.adapter_eval import (
    ProtocolError,
    canonical_messages,
    eval_adapter,
    rescore_saved,
    review_content_hash,
    screen_candidates,
)
from qa_pipeline.experiments.scoring import (
    SCORER_V2,
    SCORER_V3,
    action_assessment,
    classify_action,
    reference_self_check,
    score_task_v3,
    visible_support,
)
from qa_pipeline.experiments.v24_audit import assert_writable_audit_dest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _case(**kwargs) -> dict:
    row = {
        "case_id": "a",
        "question": "成人一次剂量是多少？",
        "context": "成人一次 0.5 g",
        "answer": "成人一次 0.5 g",
        "answer_points": ["成人一次 0.5 g"],
        "expected_action": "answer",
        "evidence_state": "sufficient",
        "source_family_id": "src_a",
        "family_id": "fam_a",
        "stratum": "seen_rephrase",
        "review_status": "unreviewed",
    }
    row.update(kwargs)
    return row


def _reviewed(**kwargs) -> dict:
    row = _case(
        review_status="dual_agreed",
        reviewer_a="reviewer_a",
        reviewer_b="reviewer_b",
        opinion_a="同意金标",
        opinion_b="同意金标",
        adjudication="同意",
        **kwargs,
    )
    row["review_content_hash"] = review_content_hash(row)
    return row


def _gen(text: str = "成人一次 0.5 g"):
    calls = {"n": 0}

    def generate(pending):
        calls["n"] += 1
        calls["last"] = pending
        return [{"text": text} for _ in pending]

    return generate, calls


def _identified(case: dict, role: str, text: str, model_id: str = "base-1") -> dict:
    return {
        "case_id": case["case_id"],
        "model_role": role,
        "text": text,
        "messages": canonical_messages(case),
        "signature": f"sig-{case['case_id']}-{role}",
        "model_id": model_id,
        "adapter_id": "g0" if role == "adapter" else "",
    }


def test_reference_answer_and_equivalent_behavior_are_self_consistent():
    refusal = _case(
        case_id="child",
        question="儿童剂量是多少？",
        context="口服，成人，一次1片，一日3次。",
        answer="当前用法用量资料没有给出儿童剂量，不能确定。",
        answer_points=["当前用法用量资料没有给出儿童剂量，不能确定。"],
        expected_action="state_insufficient",
        evidence_state="missing",
    )
    assert classify_action("不能确定儿童剂量，资料没有给出。") == "state_insufficient"
    assert reference_self_check(refusal)["passed"] is True
    clarify = _case(
        answer="需要先明确是哪一个规格。",
        answer_points=["需要先明确是哪一个规格。"],
        expected_action="clarify",
        evidence_state="ambiguous",
    )
    assert classify_action("请补充具体规格后再回答。") == "clarify"
    assert reference_self_check(clarify)["passed"] is True
    wrong = _case(
        answer="药品与问题对象不一致，不能据此回答。",
        answer_points=["药品与问题对象不一致，不能据此回答。"],
        expected_action="state_insufficient",
        evidence_state="missing",
    )
    assert classify_action(wrong["answer"]) == "state_insufficient"
    assert classify_action("两份资料互相矛盾，不能同时采信。") == "state_conflict"
    assert reference_self_check(wrong)["passed"] is True
    partial = _case(
        answer="成人一次 0.5 g。其余没有给出，不能确定。",
        answer_points=["成人一次 0.5 g"],
        required_points=["成人一次 0.5 g"],
        unavailable_points=["儿童一次 0.25 g"],
        expected_action="partial_answer",
        evidence_state="partial",
        context="成人一次 0.5 g",
    )
    assert classify_action(partial["answer"]) == "partial_answer"
    assert reference_self_check(partial)["passed"] is True
    fabricated = score_task_v3(
        "资料不足。儿童一次 2 片。",
        {"answer": "不能确定", "answer_points": ["不能确定"], "expected_action": "state_insufficient", "evidence_state": "missing"},
    )
    assert action_assessment("资料不足。儿童一次 2 片。")["refusal_then_fabrication"] is True
    assert fabricated["passed"] is False
    assert "refusal_then_fabrication" in fabricated["critical_errors"]
    over = score_task_v3(
        "资料不足，不能确定。",
        {
            "answer": "成人一次 0.5 g",
            "answer_points": ["成人一次 0.5 g"],
            "expected_action": "answer",
            "evidence_state": "sufficient",
        },
    )
    assert "over_refusal" in over["critical_errors"]
    assert over["passed"] is False
    assert over["semantic_accuracy_verified"] is False


def test_synonym_fragment_does_not_satisfy_compound_point():
    scored = score_task_v3(
        "按照说明每天服用三次就可以。",
        {
            "answer": "一日三次，并且必须饭后服用",
            "answer_points": ["一日三次，并且必须饭后服用"],
            "synonym_groups": [["一日三次", "每天服用三次"]],
            "expected_action": "answer",
            "evidence_state": "sufficient",
        },
    )
    assert scored["required_point_coverage"] != 1
    assert scored["passed"] is False


def test_visible_required_point_and_hidden_point_are_separated():
    case = _case(
        context="成人一次 0.5 g。",
        answer="成人一次 0.5 g",
        required_points=["成人一次 0.5 g"],
        unavailable_points=["儿童一次 0.25 g"],
        answer_points=["成人一次 0.5 g", "儿童一次 0.25 g"],
        expected_action="partial_answer",
        evidence_state="partial",
    )
    support = visible_support(case)
    assert support["required_not_in_context"] == []
    assert support["unavailable_still_visible"] == []
    hidden = dict(case)
    hidden["required_points"] = ["儿童一次 0.25 g"]
    assert visible_support(hidden)["required_not_in_context"] == ["儿童一次 0.25 g"]


def test_same_id_with_changed_context_or_system_is_rejected(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    original = _case()
    _write_jsonl(protocol, [original])
    generate, calls = _gen()
    eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "pred",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    changed = _case(context="另一份上下文")
    _write_jsonl(tmp_path / "changed.jsonl", [changed])
    context_report = rescore_saved(
        protocol_path=tmp_path / "changed.jsonl",
        predictions_path=tmp_path / "pred" / "predictions.jsonl",
        out_dir=tmp_path / "context",
        scorer_version=SCORER_V3,
        mode="exploratory",
    )
    assert context_report["executed"] is False
    assert context_report["scored_n"] == 0
    assert "a" in context_report["affected_ids"]
    assert any(item["reason"] == "input_mismatch" for item in context_report["failures"])
    system_case = _case()
    messages = canonical_messages(system_case)
    messages[0] = {"role": "system", "content": "改写后的系统提示"}
    system_case["messages"] = messages
    _write_jsonl(tmp_path / "system.jsonl", [system_case])
    system_report = rescore_saved(
        protocol_path=tmp_path / "system.jsonl",
        predictions_path=tmp_path / "pred" / "predictions.jsonl",
        out_dir=tmp_path / "system",
        scorer_version=SCORER_V3,
        mode="exploratory",
    )
    assert system_report["executed"] is False
    assert any(item["reason"] == "input_mismatch" for item in system_report["failures"])
    assert calls["n"] == 1


def test_gold_only_change_can_be_rescored_offline(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, calls = _gen()
    eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "pred",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
        scorer_version=SCORER_V2,
    )
    revised = _case(answer="成人一次 0.5 g。补充说明。", answer_points=["成人一次 0.5 g", "补充说明"])
    _write_jsonl(tmp_path / "gold.jsonl", [revised])
    report = rescore_saved(
        protocol_path=tmp_path / "gold.jsonl",
        predictions_path=tmp_path / "pred" / "predictions.jsonl",
        out_dir=tmp_path / "rescored",
        scorer_version=SCORER_V3,
        mode="exploratory",
    )
    assert report["executed"] is True
    assert report["model_loaded"] is False
    assert report["scored_n"] == 1
    assert calls["n"] == 1
    assert (tmp_path / "pred" / "scores.aux-rules-v2.jsonl").is_file()
    assert (tmp_path / "rescored" / "scores.aux-rules-v3.jsonl").is_file()


def test_error_prediction_is_not_a_successful_result(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    case = _case()
    _write_jsonl(protocol, [case])
    _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {**_identified(case, "base", ""), "error": "boom"},
            _identified(case, "adapter", "成人一次 0.5 g"),
        ],
    )
    report = rescore_saved(
        protocol_path=protocol,
        predictions_path=tmp_path / "predictions.jsonl",
        out_dir=tmp_path / "out",
        scorer_version=SCORER_V3,
        mode="exploratory",
    )
    assert report["executed"] is False
    assert report["scored_n"] == 0
    assert report["content_error_n"] == 0
    assert report["technical_failure_n"] == 1
    assert any(item["reason"] == "prediction_error" for item in report["failures"])


def test_same_input_different_id_reuses_content_but_binds_current_id(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case(case_id="a"), _case(case_id="b")])
    generate, calls = _gen()
    first = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    assert first["executed"] is True
    assert len(calls["last"]) == 2
    rows = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert {row["case_id"] for row in rows} == {"a", "b"}
    cache_rows = [json.loads(line) for line in (tmp_path / "out" / "prediction_cache.jsonl").read_text().splitlines()]
    assert cache_rows and all("case_id" not in row for row in cache_rows)
    second = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    assert calls["n"] == 1
    assert second["cache_hits"] == 4


def test_failed_cache_retries_only_failed_items(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case(case_id="ok"), _case(case_id="bad", question="另一道题的剂量？")])
    calls = {"n": 0, "ids": []}

    def generate(pending):
        calls["n"] += 1
        calls["ids"].append([item["case_id"] for item in pending])
        rows = []
        for item in pending:
            if item["case_id"] == "bad" and calls["n"] == 1:
                rows.append({"text": "", "error": "temporary"})
            else:
                rows.append({"text": "成人一次 0.5 g"})
        return rows

    first = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    assert first["executed"] is False
    assert first["reason"] == "generation_item_failed"
    second = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    assert calls["n"] == 2
    assert calls["ids"][1] == ["bad", "bad"]
    assert second["executed"] is True
    assert second["scored_n"] == 2


def test_empty_or_missing_protocol_is_not_success(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    report = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=empty,
        out_dir=tmp_path / "empty",
        mode="exploratory",
        template_id="tpl",
        generate_fn=_gen()[0],
    )
    assert report["executed"] is False
    assert report["status"] == "not_executed"
    assert report["reason"] == "empty_protocol"
    missing = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=tmp_path / "missing.jsonl",
        out_dir=tmp_path / "missing",
        mode="exploratory",
        template_id="tpl",
        generate_fn=_gen()[0],
    )
    assert missing["reason"] == "protocol_missing"
    assert missing["executed"] is False


def test_formal_protocol_does_not_shrink_denominator(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_reviewed(case_id="a"), _reviewed(case_id="a")])
    called = {"n": 0}

    def generate(pending):
        called["n"] += 1
        return [{"text": "成人一次 0.5 g"} for _ in pending]

    with pytest.raises(ProtocolError):
        eval_adapter(
            base_model="unused",
            base_id="base",
            adapter="unused",
            adapter_id="g0",
            protocol_path=protocol,
            out_dir=tmp_path / "out",
            mode="formal",
            template_id="tpl",
            generate_fn=generate,
            frozen_train_families=set(),
        )
    assert called["n"] == 0
    report = json.loads((tmp_path / "out" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["planned_n"] == 2
    assert report["scored_n"] == 0
    assert report["reason"] == "batch_stopped"
    screened = screen_candidates([_case(), _case(question="")])
    assert len(screened["kept"]) == 1
    assert screened["excluded"]


def test_review_content_change_invalidates_review(tmp_path):
    case = _reviewed()
    case["answer"] = "被改过的答案"
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [case])
    with pytest.raises(ProtocolError):
        eval_adapter(
            base_model="unused",
            base_id="base",
            adapter="unused",
            adapter_id="g0",
            protocol_path=protocol,
            out_dir=tmp_path / "out",
            mode="formal",
            template_id="tpl",
            generate_fn=_gen()[0],
            frozen_train_families=set(),
        )
    report = json.loads((tmp_path / "out" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["executed"] is False
    assert any("review_hash_mismatch" in item["reasons"] for item in report["protocol_failures"])
    fresh = _reviewed()
    assert review_content_hash(fresh) == fresh["review_content_hash"]


def test_uncalibrated_rules_do_not_become_main_metric(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_reviewed()])
    report = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="formal",
        template_id="tpl",
        generate_fn=_gen()[0],
        frozen_train_families=set(),
        scorer_version=SCORER_V3,
    )
    assert report["executed"] is True
    assert report["formal_protocol_executable"] is True
    assert report["enters_formal_metric"] is False
    assert report["formal_main_metric"] == "not_executed"
    assert report["semantic_accuracy_verified"] is False
    assert report["reason_main_metric"] == "uncalibrated_rules"
    assert report["combined_generalization_score"] is None


def test_new_score_version_does_not_overwrite_old_scores(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, calls = _gen()
    common = dict(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    eval_adapter(scorer_version=SCORER_V2, **common)
    v2_text = (tmp_path / "out" / "scores.aux-rules-v2.jsonl").read_text(encoding="utf-8")
    eval_adapter(scorer_version=SCORER_V3, **common)
    assert calls["n"] == 1
    assert (tmp_path / "out" / "scores.aux-rules-v2.jsonl").read_text(encoding="utf-8") == v2_text
    assert (tmp_path / "out" / "scores.aux-rules-v3.jsonl").is_file()


def test_unknown_role_and_mixed_model_identity_are_rejected(tmp_path):
    case = _case()
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [case])
    rows = [
        _identified(case, "base", "成人一次 0.5 g", model_id="base-a"),
        _identified(case, "adapter", "成人一次 0.5 g"),
        {"case_id": "a", "model_role": "other", "text": "成人一次 0.5 g", "messages": canonical_messages(case), "signature": "x", "model_id": "base-a"},
    ]
    _write_jsonl(tmp_path / "predictions.jsonl", rows)
    report = rescore_saved(
        protocol_path=protocol,
        predictions_path=tmp_path / "predictions.jsonl",
        out_dir=tmp_path / "out",
        scorer_version=SCORER_V3,
        mode="exploratory",
    )
    assert report["executed"] is False
    assert any(item["reason"] == "unknown_model_role" for item in report["failures"])


def test_protection_baseline_is_not_rewritten(tmp_path):
    repo = tmp_path / "repo"
    protected = repo / "runs" / "drug_v23_audit"
    protected.mkdir(parents=True)
    manifest = protected / "manifest.json"
    manifest.write_text('{"note":"baseline"}\n', encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="refuse_overwrite_protection_baseline"):
        assert_writable_audit_dest(protected / "manifest.json", repo)
    assert manifest.read_bytes() == before
    dest = repo / "runs" / "drug_v24_audit"
    assert_writable_audit_dest(dest, repo)
    assert manifest.read_bytes() == before


def test_source_overlap_uses_frozen_list_not_boolean(tmp_path):
    case = _reviewed(source_family_id="train_family", source_overlap_train=False)
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [case])
    with pytest.raises(ProtocolError):
        eval_adapter(
            base_model="unused",
            base_id="base",
            adapter="unused",
            adapter_id="g0",
            protocol_path=protocol,
            out_dir=tmp_path / "out",
            mode="formal",
            template_id="tpl",
            generate_fn=_gen()[0],
            frozen_train_families={"train_family"},
        )
    report = json.loads((tmp_path / "out" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["scored_n"] == 0
    assert report["planned_n"] == 1
    assert any("source_overlap" in item["reasons"] for item in report["protocol_failures"])
