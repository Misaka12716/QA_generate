"""结果页按预测身份对齐，签名或 messages 不一致时不并入回答。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from qa_pipeline.demo.app import create_app
from qa_pipeline.experiments.adapter_eval import canonical_messages
from qa_pipeline.experiments.result_view import assemble_cases, public_case, write_view
from qa_pipeline.experiments.scoring import prediction_item_signature


def _case() -> dict:
    case = {
        "case_id": "case_ok",
        "question": "剂量是多少？",
        "context": "成人一次 1 片。",
        "answer": "1 片",
        "expected_action": "answer",
        "evidence_state": "sufficient",
        "pattern": "direct",
        "task_mode": "grounded",
        "review_status": "unreviewed",
        "source_family_id": "src_demo",
        "messages": [
            {"role": "system", "content": "只根据资料回答。"},
            {"role": "user", "content": "资料：成人一次 1 片。\n问题：剂量是多少？"},
        ],
    }
    case["messages"] = canonical_messages(case)
    return case


def _report() -> dict:
    return {
        "base_id": "base-model",
        "adapter_id": "adapter-1",
        "template_id": "tmpl",
        "infer_config": {"max_new_tokens": 32, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "dtype": "bfloat16"},
    }


def _row(case: dict, role: str, text: str, report: dict, *, messages=None, signature=None) -> dict:
    model_input = messages if messages is not None else case["messages"]
    signed = signature or prediction_item_signature(
        case,
        report["base_id"],
        report["adapter_id"],
        report["template_id"],
        report["infer_config"],
        role,
        model_input=model_input,
    )
    return {
        "case_id": case["case_id"],
        "model_role": role,
        "messages": model_input,
        "signature": signed,
        "model_id": report["base_id"],
        "adapter_id": report["adapter_id"] if role == "adapter" else "",
        "template_id": None,
        "text": text,
    }


def test_signature_mismatch_is_not_joined():
    case = _case()
    report = _report()
    bad = _row(case, "base", "不该并入", report, signature="0" * 64)
    good = _row(case, "adapter", "一次 1 片。", report)
    cases, summary = assemble_cases([case], [{"label": "unit", "predictions": [bad, good], "report": report}])
    assert summary["matched_predictions"] == 1
    assert summary["unmatched_predictions"] == 1
    assert cases[0]["answers"]["base"]["text"] is None
    assert cases[0]["answers"]["base"]["match_status"] == "unmatched"
    assert "signature_mismatch" in cases[0]["answers"]["base"]["unmatch_reasons"]
    assert cases[0]["answers"]["adapter"]["text"] == "一次 1 片。"
    assert cases[0]["teacher_gold"] is None
    assert public_case(cases[0], full=False)["teacher_label"] == "待自动评估"


def test_message_change_is_not_joined_by_position():
    case = _case()
    report = _report()
    changed = [dict(item) for item in case["messages"]]
    changed[-1] = {**changed[-1], "content": changed[-1]["content"] + "额外句子"}
    row = _row(case, "base", "按改写后的输入回答", report, messages=changed)
    other = _row(case, "adapter", "对齐回答", report)
    cases, _summary = assemble_cases([case], [{"label": "unit", "predictions": [row, other], "report": report}])
    assert cases[0]["answers"]["base"]["match_status"] == "unmatched"
    assert cases[0]["answers"]["base"]["text"] is None
    assert cases[0]["answers"]["adapter"]["match_status"] == "matched"


def test_results_api_returns_full_context(tmp_path: Path):
    case = _case()
    report = _report()
    built, summary = assemble_cases(
        [case],
        [{"label": "unit", "predictions": [_row(case, "base", "1 片", report), _row(case, "adapter", "一次 1 片", report)], "report": report}],
    )
    write_view(tmp_path, built, {"banner": "历史真实预测", "protocol_n": summary["protocol_n"], "block": "historical"})
    client = TestClient(create_app(tmp_path))
    listed = client.get("/api/v1/result-cases", params={"pattern": "direct", "rule": "pass"})
    assert listed.status_code == 200
    body = listed.json()
    assert body["meta"]["total"] == 1
    assert body["data"][0]["teacher_label"] == "待自动评估"
    assert "student_context" not in body["data"][0]
    detail = client.get("/api/v1/result-cases/case_ok")
    full = detail.json()["data"]
    assert full["student_context"] == case["context"]
    assert full["answers"]["base"]["text"] == "1 片"
    assert full["messages"] == case["messages"]
    missing = client.get("/api/v1/result-cases")
    page = client.get("/results")
    assert page.status_code == 200
    assert "历史真实预测" in page.text
    empty = TestClient(create_app(tmp_path / "empty"))
    assert empty.get("/api/v1/result-cases").status_code == 404
    assert missing.status_code == 200
