"""已有 adapter 评测的契约。不训练 LoRA。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qa_pipeline.experiments.adapter_eval import (
    EXPLORATORY_DISCLAIMER,
    ProtocolError,
    eval_adapter,
    rescore_saved,
    review_content_hash,
)
from qa_pipeline.experiments.dev_candidates import write_candidate_file
from qa_pipeline.experiments.frozen_loader import FrozenLoadError, load_frozen_subset
from qa_pipeline.experiments.scoring import (
    SCORER_V1,
    SCORER_V2,
    aggregate_families,
    aggregate_layered,
    align_prediction_groups,
    score_task_v2,
)
from qa_pipeline.pipeline import load_documents

ROOT = Path(__file__).resolve().parents[1]


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
    row = _case(review_status="dual_agreed", **kwargs)
    row["review_content_hash"] = review_content_hash(row)
    return row


def _gen(text: str = "成人一次 0.5 g"):
    calls = {"n": 0}

    def generate(pending):
        calls["n"] += 1
        calls["last"] = pending
        return [{"text": text} for _ in pending]

    return generate, calls


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_eval_does_not_train_or_export(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("不应调用训练或训练导出")

    monkeypatch.setattr("qa_pipeline.experiments.sft.train_lora", boom)
    monkeypatch.setattr("qa_pipeline.experiments.sft.write_sft_jsonl", boom)
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, calls = _gen()
    report = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        scorer_version=SCORER_V2,
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
    )
    assert calls["n"] == 1
    assert report["executed"] is True
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_saved_predictions_can_be_rescored_offline(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
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
    second = rescore_saved(
        protocol_path=protocol,
        predictions_path=tmp_path / "out" / "predictions.jsonl",
        out_dir=tmp_path / "rescored",
        scorer_version=SCORER_V2,
        mode="exploratory",
    )
    assert first["scored_n"] == second["scored_n"] == 1
    assert second["model_loaded"] is False
    assert calls["n"] == 1


def test_shuffled_predictions_keep_the_same_score(tmp_path):
    cases = [
        _case(case_id="a"),
        _case(case_id="b", question="儿童剂量？", answer="儿童一次 0.25 g", answer_points=["儿童一次 0.25 g"]),
    ]
    groups = {
        "base": [{"case_id": "b", "text": "儿童一次 0.25 g"}, {"case_id": "a", "text": "成人一次 0.5 g"}],
        "adapter": [{"case_id": "a", "text": "成人一次 0.5 g"}, {"case_id": "b", "text": "儿童一次 0.25 g"}],
    }
    ordered = align_prediction_groups(cases, groups)
    flipped = align_prediction_groups(list(reversed(cases)), groups)
    assert [item["case"]["case_id"] for item in ordered["aligned"]] == ["a", "b"]
    assert [item["predictions"]["adapter"]["text"] for item in ordered["aligned"]] == ["成人一次 0.5 g", "儿童一次 0.25 g"]
    assert [item["predictions"]["adapter"]["text"] for item in flipped["aligned"]] == ["儿童一次 0.25 g", "成人一次 0.5 g"]
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, cases)
    _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"case_id": "b", "model_role": "base", "text": "儿童一次 0.25 g"},
            {"case_id": "a", "model_role": "adapter", "text": "成人一次 0.5 g"},
            {"case_id": "a", "model_role": "base", "text": "成人一次 0.5 g"},
            {"case_id": "b", "model_role": "adapter", "text": "儿童一次 0.25 g"},
        ],
    )
    report = rescore_saved(
        protocol_path=protocol,
        predictions_path=tmp_path / "predictions.jsonl",
        out_dir=tmp_path / "out",
        scorer_version=SCORER_V2,
        mode="exploratory",
    )
    assert report["executed"] is True
    assert report["scored_n"] == 2


def test_missing_duplicate_or_extra_predictions_do_not_pass(tmp_path):
    cases = [_case()]
    missing = align_prediction_groups(cases, {"base": [], "adapter": []})
    duplicate = align_prediction_groups(
        cases,
        {
            "base": [{"case_id": "a", "text": "甲"}, {"case_id": "a", "text": "乙"}],
            "adapter": [{"case_id": "a", "text": "甲"}],
        },
    )
    extra = align_prediction_groups(
        cases,
        {
            "base": [{"case_id": "a", "text": "甲"}, {"case_id": "extra", "text": "乙"}],
            "adapter": [{"case_id": "a", "text": "甲"}, {"case_id": "extra", "text": "乙"}],
        },
    )
    assert missing["ok"] is False and missing["scored_n"] == 0 and missing["planned_n"] == 1
    assert duplicate["ok"] is False and duplicate["scored_n"] == 0
    assert extra["ok"] is False and extra["failed_n"] == 1
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, cases)
    _write_jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"case_id": "a", "model_role": "base", "text": "甲"},
            {"case_id": "extra", "model_role": "base", "text": "乙"},
            {"case_id": "a", "model_role": "adapter", "text": "甲"},
            {"case_id": "extra", "model_role": "adapter", "text": "乙"},
        ],
    )
    report = rescore_saved(
        protocol_path=protocol,
        predictions_path=tmp_path / "predictions.jsonl",
        out_dir=tmp_path / "out",
        scorer_version=SCORER_V2,
        mode="exploratory",
    )
    assert report["executed"] is False
    assert report["status"] == "not_executed"
    assert report.get("task_accuracy") is None


def test_invalid_protocol_stops_before_generation(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [{"case_id": "a", "question": "", "review_status": "unreviewed"}])
    called = {"n": 0}

    def generate(_pending):
        called["n"] += 1
        return []

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
        )
    assert called["n"] == 0


def test_unreviewed_items_stay_out_of_formal_metric(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_reviewed(case_id="ok"), _case(case_id="raw", review_status="unreviewed")])
    seen = []

    def generate(pending):
        seen.extend(item["case_id"] for item in pending)
        return [{"text": "成人一次 0.5 g"} for _ in pending]

    report = eval_adapter(
        base_model="unused",
        base_id="base",
        adapter="unused",
        adapter_id="g0",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="formal",
        template_id="tpl",
        generate_fn=generate,
    )
    assert seen == ["ok", "ok"]
    assert report["enters_formal_metric"] is True
    scores = [json.loads(line)["case_id"] for line in (tmp_path / "out" / "scores.jsonl").read_text().splitlines()]
    assert scores == ["ok"]


def test_exploratory_is_not_formal_gold(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, _calls = _gen()
    report = eval_adapter(
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
    assert report["result_class"] == "exploratory_auxiliary"
    assert report["formal_gold"] is False
    assert report["formal_main_metric"] == "not_executed"
    assert report["enters_formal_metric"] is False
    assert report["disclaimer"] == EXPLORATORY_DISCLAIMER
    score = json.loads((tmp_path / "out" / "scores.jsonl").read_text().splitlines()[0])
    assert score["disclaimer"] == EXPLORATORY_DISCLAIMER
    assert score["enters_formal_metric"] is False


def test_candidate_rewrite_does_not_overwrite_formal_protocol(tmp_path):
    protocol = tmp_path / "heldout_protocol.jsonl"
    protocol.write_text('{"id":"locked"}\n', encoding="utf-8")
    original = protocol.read_text(encoding="utf-8")
    with pytest.raises(ProtocolError, match="refuse_overwrite_formal_protocol"):
        write_candidate_file(protocol, [_case()], protected=[protocol])
    assert protocol.read_text(encoding="utf-8") == original
    dest = tmp_path / "candidates" / "dev_candidates.jsonl"
    write_candidate_file(dest, [_case()], protected=[protocol])
    assert dest.is_file()
    assert protocol.read_text(encoding="utf-8") == original


def test_metadata_json_is_not_loaded_as_body(tmp_path):
    (tmp_path / "note.txt").write_text("正文药品说明", encoding="utf-8")
    (tmp_path / "snapshot_manifest.json").write_text('{"files":["不是正文"]}', encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "不是正文" not in docs[0].text
    manifest = tmp_path / "subset_manifest.json"
    manifest.write_text(json.dumps([{"order": 0, "stem": "1082", "source": "raw/1082.txt"}]), encoding="utf-8")
    (tmp_path / "000_1082.txt").write_text("说明书正文", encoding="utf-8")
    loaded = load_frozen_subset(tmp_path, manifest)
    assert loaded[0]["text"] == "说明书正文"
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps([{"order": 9, "stem": "absent"}]), encoding="utf-8")
    with pytest.raises(FrozenLoadError, match="missing_file"):
        load_frozen_subset(tmp_path, bad)


def test_source_id_reaches_predictions(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case(source_family_id="src_keep")])
    generate, _calls = _gen()
    eval_adapter(
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
    rows = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert {row["source_family_id"] for row in rows} == {"src_keep"}


def test_repeated_phrasings_do_not_increase_family_weight():
    cases = [{"family_id": "a", "source_family_id": "src", "passed": True, "stratum": "seen_rephrase"} for _ in range(10)]
    cases.append({"family_id": "b", "source_family_id": "src", "passed": False, "stratum": "seen_rephrase"})
    cases.append({"family_id": "beh", "source_family_id": "src", "passed": False, "stratum": "behavior"})
    summary = aggregate_families([row for row in cases if row["stratum"] != "behavior"])
    assert summary["task_accuracy"] == 0.5
    assert summary["family_count"] == 2
    layered = aggregate_layered(cases)
    assert layered["ordinary"]["task_accuracy"] == 0.5
    assert layered["behavior"]["row_count"] == 1
    assert layered["combined_generalization_score"] is None
    assert layered["row_count"] == 12


def test_failed_generation_is_not_executed(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])

    def generate(_pending):
        return {"skipped": True, "reason": "no_gpu"}

    report = eval_adapter(
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
    assert report["executed"] is False
    assert report["status"] == "not_executed"
    assert report["reason"] == "no_gpu"


def test_prediction_cache_invalidates_on_input_model_or_config(tmp_path):
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, calls = _gen()
    common = dict(
        base_model="unused",
        adapter="unused",
        protocol_path=protocol,
        out_dir=tmp_path / "out",
        mode="exploratory",
        template_id="tpl",
        generate_fn=generate,
        scorer_version=SCORER_V2,
    )
    eval_adapter(base_id="base", adapter_id="g0", **common)
    eval_adapter(base_id="base", adapter_id="g0", **common)
    assert calls["n"] == 1
    eval_adapter(base_id="base-b", adapter_id="g0", **common)
    eval_adapter(base_id="base-b", adapter_id="g0-b", **common)
    eval_adapter(base_id="base-b", adapter_id="g0-b", infer_config={"max_new_tokens": 128}, **common)
    assert calls["n"] == 4
    _write_jsonl(protocol, [_case(question="改写后的剂量问题？")])
    eval_adapter(base_id="base-b", adapter_id="g0-b", infer_config={"max_new_tokens": 128}, **common)
    assert calls["n"] == 5


def test_scorer_change_reuses_predictions(tmp_path):
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
    report = eval_adapter(scorer_version=SCORER_V1, **common)
    assert calls["n"] == 1
    assert report["scorer_version"] == SCORER_V1
    assert report["cache_hits"] == 2


def test_historical_files_stay_unmodified(tmp_path):
    watched = [
        ROOT / "runs/drug_v22/run_meta.json",
        ROOT / "runs/drug_v22/E3_g0/sft/train.jsonl",
        ROOT / "runs/drug_v22/E3_g0/sft/train_seen.jsonl",
        ROOT / "data/campus_hospital_drug_instructions/frozen/heldout.jsonl",
        ROOT / "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl",
    ]
    v21 = ROOT / "runs/drug_v21"
    if v21.is_dir():
        found = next((path for path in v21.rglob("*") if path.is_file() and path.stat().st_size < 2_000_000), None)
        if found is not None:
            watched.append(found)
    before = {str(path): _sha(path) for path in watched}
    protocol = tmp_path / "protocol.jsonl"
    _write_jsonl(protocol, [_case()])
    generate, _calls = _gen()
    eval_adapter(
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
    assert {str(path): _sha(path) for path in watched} == before


def test_scoring_rules_cover_required_cases():
    swap = score_task_v2(
        "成人一次 0.25 g，儿童一次 0.5 g",
        {
            "answer": "成人一次 0.5 g，儿童一次 0.25 g",
            "answer_points": ["成人一次 0.5 g", "儿童一次 0.25 g"],
            "numeric_bindings": [
                {"object": "成人", "number": "0.5", "unit": "g"},
                {"object": "儿童", "number": "0.25", "unit": "g"},
            ],
            "expected_action": "answer",
        },
    )
    assert "numeric_object_swap" in swap["critical_errors"]
    equivalent = score_task_v2(
        "成人一次 500 mg",
        {
            "answer": "成人一次 0.5 g",
            "answer_points": ["成人一次 0.5 g"],
            "numeric_bindings": [{"object": "成人", "number": "0.5", "unit": "g"}],
            "expected_action": "answer",
        },
    )
    assert "wrong_unit" not in equivalent["critical_errors"]
    assert "numeric_object_swap" not in equivalent["critical_errors"]
    wrong = score_task_v2(
        "成人一次 0.5 mg",
        {
            "answer": "成人一次 0.5 g",
            "answer_points": ["成人一次 0.5 g"],
            "numeric_bindings": [{"object": "成人", "number": "0.5", "unit": "g"}],
            "expected_action": "answer",
        },
    )
    assert "wrong_unit" in wrong["critical_errors"]
    missing = score_task_v2(
        "一次 0.5 g",
        {
            "answer": "饭后一次 0.5 g",
            "answer_points": ["饭后一次 0.5 g"],
            "required_conditions": ["饭后"],
            "expected_action": "answer",
        },
    )
    assert "missing_condition" in missing["critical_errors"]
    negation = score_task_v2(
        "孕妇可以使用",
        {"answer": "孕妇不得使用", "answer_points": ["孕妇不得使用"], "expected_action": "answer"},
    )
    assert "negation_scope" in negation["critical_errors"]
    retract = score_task_v2(
        "成人一次 0.5 g。上述说法不正确。",
        {"answer": "成人一次 0.5 g", "answer_points": ["成人一次 0.5 g"], "expected_action": "answer"},
    )
    assert "retracted" in retract["critical_errors"]
    extra = score_task_v2(
        "成人一次 0.5 g，另给 12 mg。",
        {
            "answer": "成人一次 0.5 g",
            "answer_points": ["成人一次 0.5 g"],
            "expected_action": "answer",
            "evidence_state": "sufficient",
        },
    )
    assert extra["unsupported_assertions"]
    synonym = score_task_v2(
        "每天服用三次",
        {
            "answer": "一日三次",
            "answer_points": ["一日三次"],
            "synonym_groups": [["一日三次", "每天服用三次"]],
            "expected_action": "answer",
            "evidence_state": "sufficient",
        },
    )
    assert synonym["required_point_coverage"] == 1
    assert synonym["passed"] is True
    partial = score_task_v2(
        "饭后服用",
        {
            "answer": "饭后服用，睡前加服",
            "answer_points": ["饭后服用", "睡前加服"],
            "expected_action": "partial_answer",
            "evidence_state": "partial",
        },
    )
    assert partial["required_point_coverage"] == 0.5
    assert partial["passed"] is True
    insufficient = score_task_v2(
        "资料不足，无法确定。",
        {
            "answer": "资料不足，无法确定。",
            "answer_points": ["资料不足，无法确定。"],
            "expected_action": "state_insufficient",
            "evidence_state": "missing",
        },
    )
    assert insufficient["expected_behavior"]["ok"] is True
    refused = score_task_v2(
        "资料不足，无法确定。",
        {
            "answer": "成人一次 0.5 g",
            "answer_points": ["成人一次 0.5 g"],
            "expected_action": "answer",
            "evidence_state": "sufficient",
        },
    )
    assert "over_refusal" in refused["critical_errors"]
    assert refused["semantic_accuracy_verified"] is False
