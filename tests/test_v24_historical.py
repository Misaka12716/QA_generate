"""读取真实历史运行的集成测试。干净 checkout 缺少这些文件时跳过。"""

from __future__ import annotations

from pathlib import Path

import pytest

from qa_pipeline.experiments.adapter_eval import read_jsonl
from qa_pipeline.experiments.v24_audit import verify_saved_input_identity

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "runs/drug_v23_eval/candidates/dev_candidates.jsonl"
PREDICTIONS = ROOT / "runs/drug_v23_eval/exploratory/predictions.jsonl"
REPORT = ROOT / "runs/drug_v23_eval/exploratory/eval_report.json"
MANIFEST = ROOT / "runs/drug_v23_audit/manifest.json"


pytestmark = pytest.mark.skipif(not CANDIDATES.is_file() or not PREDICTIONS.is_file() or not REPORT.is_file(), reason="历史运行文件不在这个 checkout 里")


def test_v23_predictions_match_saved_messages_and_signatures():
    import json

    cases = read_jsonl(CANDIDATES)
    predictions = read_jsonl(PREDICTIONS)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checked = verify_saved_input_identity(
        cases,
        predictions,
        template_id=report["template_id"],
        infer_config=report["infer_config"],
        base_id=report["base_id"],
        adapter_id=report["adapter_id"],
    )
    assert checked["input_match"] is True
    assert checked["signature_match"] is True
    assert checked["prediction_n"] == 200


def test_historical_manifest_is_readable_without_rewrite():
    import json

    before = MANIFEST.read_bytes()
    payload = json.loads(before.decode("utf-8"))
    assert payload.get("files")
    assert MANIFEST.read_bytes() == before
