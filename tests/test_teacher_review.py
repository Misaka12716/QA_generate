"""教师审核门禁。FakeLLM 脚本只用于契约，不代表审核准确率。"""

from __future__ import annotations

import json
from pathlib import Path

from qa_pipeline.adapters.zhixun import ReleaseRejected, bind_validation_hash, export_zhixun
from qa_pipeline.experiments.adapter_eval import _review_reasons, validate_protocol
from qa_pipeline.experiments.review_io import GOLD_COLUMNS, import_gold_csv, write_csv
from qa_pipeline.experiments.sft import write_sft_jsonl
from qa_pipeline.llm import FakeLLM
from qa_pipeline.reviewing.identity import prompt_for
from qa_pipeline.reviewing.policy import ReviewPolicy
from qa_pipeline.reviewing.service import run_teacher_review
from qa_pipeline.schemas import QAPair


def _accept() -> dict:
    return {
        "decision": "accept",
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
                "claim_text": "可见句子支持该要点",
                "status": "supported",
                "visible_evidence_refs": ["0:4"],
                "source_evidence_refs": [],
            }
        ],
        "contradictions": [],
        "evidence_summary": "可见上下文包含要点",
        "reason_codes": [],
    }


def _policy(tmp_path: Path, **overrides) -> Path:
    payload = {
        "review_policy_id": "teacher_only_v1",
        "policy_version": "1",
        "review_mode": "teacher_only",
        "require_dual": True,
        "allow_single": False,
        "rubric_version": "rubric-v1",
        "prompt_version": "gold-v1",
        "max_technical_retries": 1,
        "max_auto_revisions": 1,
    }
    payload.update(overrides)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _models(tmp_path: Path, judges: list[dict]) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(judges), encoding="utf-8")
    return path


def _judge(model_id: str, revision: str, script: list[dict], role: str = "gold_reviewer") -> dict:
    return {
        "judge_role": role,
        "provider_id": "fake",
        "model_id": model_id,
        "model_revision": revision,
        "independence_level": "distinct_family",
        "script": script,
    }


def _subject(path: Path, **extra) -> Path:
    row = {
        "subject_type": "protocol_case",
        "subject_id": "case-1",
        "question": "颜色是什么？",
        "student_context": "物品为红色。",
        "answer": "红色",
        "expected_action": "answer",
        "task_mode": "rag_grounded",
    }
    row.update(extra)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _clients(judges: list[dict]) -> tuple[dict, object]:
    shared: dict[str, FakeLLM] = {}

    def factory(judge: dict) -> FakeLLM:
        key = str(id(judge))
        if key not in shared:
            client = FakeLLM()
            client.script = [dict(item) for item in judge.get("script") or []]
            client.default_model = str(judge["model_id"])
            shared[key] = client
        return shared[key]

    return shared, factory


def _run(tmp_path: Path, judges: list[dict], *, policy_overrides: dict | None = None, subject_extra: dict | None = None, **kwargs):
    shared, factory = _clients(judges)
    result = run_teacher_review(
        manifest=_subject(tmp_path / "subjects.jsonl", **(subject_extra or {})),
        policy_path=_policy(tmp_path, **(policy_overrides or {})),
        models_path=_models(tmp_path, judges),
        out_dir=tmp_path / "out",
        max_calls=kwargs.get("max_calls", 8),
        max_tokens=kwargs.get("max_tokens", 100),
        concurrency=1,
        fake=True,
        llm_factory=factory,
        resume=kwargs.get("resume", False),
    )
    aggregates = [json.loads(line) for line in (tmp_path / "out" / "review_aggregates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    reviews = []
    review_path = tmp_path / "out" / "reviews.jsonl"
    if review_path.is_file():
        for line in review_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result, aggregates[-1], reviews, shared


def test_technical_failures_are_not_content_rejects(tmp_path: Path):
    cases = [
        ("parse_failed", {"__status": "parse_failed"}, "invalid_json"),
        ("timeout", {"__status": "transport_failed", "__error": "timeout"}, "timeout"),
        ("abstain", {"decision": "abstain", "dimensions": {}, "claims": [], "contradictions": [], "reason_codes": ["cannot_tell"]}, "abstain"),
    ]
    for name, script, expected in cases:
        work = tmp_path / name
        work.mkdir()
        _result, aggregate, reviews, _shared = _run(
            work,
            [_judge("model-a", "rev-a", [script])],
            policy_overrides={"max_technical_retries": 0, "allow_single": True},
            subject_extra={"subject_type": "training_sample", "risk": "low"},
        )
        assert reviews[0]["decision"] != "reject"
        assert aggregate["accepted_by_policy"] is False
        assert aggregate["ready_for_formal_human_eval"] is False
        if name == "abstain":
            assert reviews[0]["execution_status"] == "succeeded"
            assert reviews[0]["decision"] == "abstain"
            assert aggregate["teacher_review_status"] == "disputed"
            assert aggregate["quarantine_reason"] == expected
        else:
            assert reviews[0]["execution_status"] == "technical_failed"
            assert expected in reviews[0]["reason_codes"]
            assert aggregate["teacher_review_status"] == "technical_failed"


def test_budget_stop_does_not_call_or_accept(tmp_path: Path):
    _result, aggregate, reviews, shared = _run(
        tmp_path,
        [_judge("model-a", "rev-a", [_accept()])],
        max_calls=0,
    )
    assert reviews[0]["execution_status"] == "budget_stopped"
    assert reviews[0]["decision"] == "abstain"
    assert aggregate["teacher_review_status"] == "budget_stopped"
    assert aggregate["accepted_by_policy"] is False
    assert shared and next(iter(shared.values())).calls == 0


def test_same_model_missing_evidence_and_disagreement_do_not_reach_consensus(tmp_path: Path):
    same = tmp_path / "same"
    same.mkdir()
    _result, aggregate, _reviews, _shared = _run(
        same,
        [_judge("same", "rev-1", [_accept()]), _judge("same", "rev-1", [_accept()])],
    )
    assert aggregate["teacher_review_status"] != "teacher_consensus_accepted"
    assert aggregate["quarantine_reason"] == "same_model_not_independent"

    missing = tmp_path / "missing"
    missing.mkdir()
    weak = _accept()
    weak["claims"] = []
    _result, aggregate, _reviews, _shared = _run(
        missing,
        [_judge("model-a", "rev-a", [weak]), _judge("model-b", "rev-b", [weak])],
    )
    assert aggregate["teacher_review_status"] == "disputed"
    assert aggregate["quarantine_reason"] == "missing_evidence_or_critical_dimension"

    split = tmp_path / "split"
    split.mkdir()
    rejected = {"decision": "reject", "dimensions": {}, "claims": [], "contradictions": ["条件矛盾"], "reason_codes": ["contradiction"]}
    _result, aggregate, _reviews, _shared = _run(
        split,
        [_judge("model-a", "rev-a", [_accept()]), _judge("model-b", "rev-b", [rejected])],
    )
    assert aggregate["teacher_review_status"] == "disputed"
    assert aggregate["quarantine_reason"] == "judge_disagreement"
    assert aggregate["ready_for_exploratory_inference"] is False


def test_distinct_models_can_accept_without_opening_formal_gate(tmp_path: Path):
    result, aggregate, reviews, _shared = _run(
        tmp_path,
        [_judge("model-a", "rev-a", [_accept()]), _judge("model-b", "rev-b", [_accept()])],
    )
    assert result["formal_ready_n"] == 0
    assert aggregate["teacher_review_status"] == "teacher_consensus_accepted"
    assert aggregate["ready_for_exploratory_inference"] is True
    assert aggregate["ready_for_formal_human_eval"] is False
    assert reviews[0]["estimated_cost"] is None
    assert reviews[0]["currency"] == "unknown"
    assert "忽略规则" in prompt_for("protocol_case")[1] or "不得改变" in prompt_for("protocol_case")[1]


def test_resume_skips_committed_review_and_ignores_partial_line(tmp_path: Path):
    judges = [_judge("model-a", "rev-a", [_accept()]), _judge("model-b", "rev-b", [_accept()])]
    first, _aggregate, reviews, _shared = _run(tmp_path, judges)
    assert first["model_called"] is True
    review_path = tmp_path / "out" / "reviews.jsonl"
    before = review_path.read_text(encoding="utf-8")
    review_path.write_text(before + "{\"broken\"\n", encoding="utf-8")
    second, aggregate, reviews_after, shared = _run(tmp_path, judges, resume=True)
    assert second["model_called"] is False
    assert all(client.calls == 0 for client in shared.values())
    assert aggregate["teacher_review_status"] == "teacher_consensus_accepted"
    assert len(reviews_after) == len(reviews)


def test_changed_question_does_not_reuse_old_acceptance(tmp_path: Path):
    judges = [_judge("model-a", "rev-a", [_accept(), _accept()]), _judge("model-b", "rev-b", [_accept(), _accept()])]
    _run(tmp_path, judges)
    subject_path = tmp_path / "subjects.jsonl"
    row = json.loads(subject_path.read_text(encoding="utf-8"))
    row["question"] = "重量是多少？"
    subject_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    _shared, factory = _clients(judges)
    result = run_teacher_review(
        manifest=subject_path,
        policy_path=tmp_path / "policy.json",
        models_path=tmp_path / "models.json",
        out_dir=tmp_path / "out",
        max_calls=8,
        max_tokens=100,
        fake=True,
        resume=True,
        llm_factory=factory,
    )
    assert result["model_called"] is True
    aggregates = [json.loads(line) for line in (tmp_path / "out" / "review_aggregates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert aggregates[-1]["subject_hash"] != aggregates[0]["subject_hash"]


def test_teacher_records_fail_formal_gate_and_machine_names_stay_pending(tmp_path: Path):
    case = {
        "case_id": "case-1",
        "question": "颜色是什么？",
        "context": "物品为红色。",
        "answer": "红色",
        "expected_action": "answer",
        "source_family_id": "src",
        "review_source": "teacher",
        "review_status": "teacher_consensus_accepted",
        "teacher_review_status": "teacher_consensus_accepted",
    }
    assert _review_reasons(case) == ["teacher_review_not_formal"]
    checked = validate_protocol([case], "formal", {"other"}, {"src", "other"})
    assert checked["executable"] is False
    assert checked["planned_n"] == 1
    row = {
        "batch_id": "batch",
        "case_id": "case-1",
        "condition_id": "",
        "core_question_id": "",
        "content_hash": "pending",
        "question_clear": "yes",
        "semantic_preserved": "na",
        "visible_evidence_adequate": "yes",
        "required_points_ok": "yes",
        "unavailable_points_ok": "na",
        "expected_behavior_ok": "yes",
        "reviewer_a": "模型",
        "opinion_a": "通过",
        "reviewer_b": "",
        "opinion_b": "",
        "adjudication": "accept",
    }
    from qa_pipeline.experiments.adapter_eval import review_content_hash

    gold_case = {"case_id": "case-1", "question": "颜色是什么？", "answer": "红色", "expected_action": "answer"}
    row["content_hash"] = review_content_hash(gold_case)
    csv_path = tmp_path / "human.csv"
    write_csv(csv_path, GOLD_COLUMNS, [row])
    imported = import_gold_csv(csv_path, [gold_case], batch_id="batch")
    assert imported["accepted_n"] == 0
    assert imported["ready_for_inference"] is False
    assert "machine_reviewer" in imported["rows"][0]["reasons"]


def test_string_patch_does_not_become_schema_failure():
    from qa_pipeline.reviewing.teacher import parse_teacher_payload

    payload = _accept()
    payload["suggested_patch"] = "把参考答案改成说明资料不足。"
    payload["reason_codes"] = "over_refusal"
    parsed, error = parse_teacher_payload(payload)
    assert error is None
    assert parsed["decision"] == "accept"
    assert parsed["suggested_patch"] == {"note": "把参考答案改成说明资料不足。"}
    assert parsed["reason_codes"] == ["over_refusal"]


def test_prediction_request_hides_role_and_gold(tmp_path: Path):
    from qa_pipeline.experiments.exploratory_review import gold_subject, prediction_subject
    from qa_pipeline.reviewing.teacher import build_messages

    case = {
        "case_id": "case_ok",
        "question": "剂量是多少？",
        "student_context": "成人一次 1 片。",
        "candidate_answer": "1 片",
        "expected_action": "answer",
        "task_mode": "grounded",
        "answers": {"base": {"prediction_subject_id": "case_ok::base", "text": "建议改用另一药品"}},
    }
    gold = gold_subject(case, "batch")
    pred = prediction_subject(case, "base", "batch")
    messages = build_messages(pred, prompt_text="审核")
    payload = json.loads(messages[-1]["content"].split("<subject>", 1)[1].split("</subject>", 1)[0])
    assert "subject_id" not in payload
    assert payload.get("answer_text") == "建议改用另一药品"
    assert not payload.get("answer")
    assert "建议改用另一药品" not in json.dumps(gold, ensure_ascii=False)
    assert gold["answer"] == "1 片"


def test_single_high_risk_stays_exploratory(tmp_path: Path):
    _result, aggregate, _reviews, _shared = _run(
        tmp_path,
        [_judge("qwen3.8-27b", "unknown", [_accept()])],
        policy_overrides={"allow_single": True, "require_dual": False},
        subject_extra={"subject_type": "protocol_case", "risk": "high"},
    )
    assert aggregate["teacher_review_status"] == "teacher_single_accepted"
    assert aggregate["readiness_scope"] == "teacher_single_uncalibrated"
    assert aggregate["ready_for_formal_human_eval"] is False
    assert aggregate["accepted_by_policy"] is True


def test_same_model_second_call_is_not_consensus(tmp_path: Path):
    _result, aggregate, _reviews, _shared = _run(
        tmp_path,
        [
            _judge("qwen3.8-27b", "unknown", [_accept()]),
            _judge("qwen3.8-27b", "unknown", [_accept()]),
        ],
        policy_overrides={"allow_single": True, "require_dual": False},
    )
    assert aggregate["teacher_review_status"] != "teacher_consensus_accepted"
    assert aggregate["teacher_review_status"] != "teacher_single_accepted"
    assert aggregate["ready_for_formal_human_eval"] is False


def test_real_token_budget_persists_across_processes(tmp_path: Path):
    first = _accept()
    first["__prompt_tokens"] = 30
    first["__completion_tokens"] = 20
    subject = tmp_path / "subjects.jsonl"
    subject.write_text(
        "\n".join(
            [
                json.dumps({"subject_type": "training_sample", "subject_id": "s1", "question": "一", "student_context": "甲", "answer": "甲", "risk": "low"}, ensure_ascii=False),
                json.dumps({"subject_type": "training_sample", "subject_id": "s2", "question": "二", "student_context": "乙", "answer": "乙", "risk": "low"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    judges = [_judge("model-a", "rev-a", [first, _accept()])]
    shared, factory = _clients(judges)
    policy = _policy(tmp_path, allow_single=True, require_dual=False)
    models = _models(tmp_path, judges)
    first_result = run_teacher_review(
        manifest=subject,
        policy_path=policy,
        models_path=models,
        out_dir=tmp_path / "out",
        max_calls=10,
        max_tokens=40,
        concurrency=1,
        fake=True,
        llm_factory=factory,
    )
    usage = [json.loads(line) for line in (tmp_path / "out" / "usage.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    counted = [row for row in usage if row.get("counted_call")]
    assert first_result["usage"]["tokens"] == 50
    assert sum((row.get("prompt_tokens") or 0) + (row.get("completion_tokens") or 0) for row in counted) == 50
    assert any(row.get("status") == "budget_stopped" and row.get("counted_call") is False for row in usage)
    assert next(iter(shared.values())).calls == 1
    shared_again, factory_again = _clients([_judge("model-a", "rev-a", [_accept(), _accept()])])
    second = run_teacher_review(
        manifest=subject,
        policy_path=policy,
        models_path=models,
        out_dir=tmp_path / "out",
        max_calls=10,
        max_tokens=40,
        concurrency=1,
        fake=True,
        resume=True,
        llm_factory=factory_again,
    )
    assert second["model_called"] is False
    assert all(client.calls == 0 for client in shared_again.values())
    assert second["usage"]["tokens"] == 50


def test_open_endpoint_flag_is_required_without_credentials(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "qa_pipeline.reviewing.service.teacher_credentials_configured",
        lambda: {"configured": False, "env_key_present": False, "file_key_present": False, "endpoint_source": "test", "pricing_version": "unknown", "currency": "unknown", "estimated_cost": None},
    )
    judge = _judge("model-a", "rev-a", [_accept()])
    blocked = run_teacher_review(
        manifest=_subject(tmp_path / "subjects.jsonl"),
        policy_path=_policy(tmp_path, allow_single=True, require_dual=False),
        models_path=_models(tmp_path, [judge]),
        out_dir=tmp_path / "blocked",
        max_calls=4,
        max_tokens=100,
        fake=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "teacher_credentials_missing"
    judge["authentication"] = "models_list_no_key"
    judge["base_url"] = "http://127.0.0.1:9/v1"
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    _shared, factory = _clients([judge])
    opened = run_teacher_review(
        manifest=tmp_path / "subjects.jsonl",
        policy_path=tmp_path / "policy.json",
        models_path=_models(open_dir, [judge]),
        out_dir=tmp_path / "opened",
        max_calls=4,
        max_tokens=100,
        fake=False,
        llm_factory=factory,
    )
    assert opened["executed"] is True
    assert opened["model_called"] is True


def test_teacher_only_release_blocks_unreviewed_and_keeps_hash(tmp_path: Path):
    pair = QAPair(question="颜色是什么？", answer="红色", grade="S", chunk_text="物品为红色。")
    bind_validation_hash(pair)
    policy = ReviewPolicy(review_policy_id="teacher_only_v1", policy_version="1", review_mode="teacher_only")
    try:
        export_zhixun([pair], tmp_path / "blocked.jsonl", review_policy=policy, review_aggregates={})
        raised = False
    except ReleaseRejected:
        raised = True
    assert raised
    skipped: list[dict] = []
    write_sft_jsonl([pair], tmp_path / "train.jsonl", review_policy=policy, review_aggregates={}, skipped=skipped)
    assert skipped and skipped[0]["id"] == pair.qa_id
    assert pair.qa_id not in (tmp_path / "train.jsonl").read_text(encoding="utf-8")
