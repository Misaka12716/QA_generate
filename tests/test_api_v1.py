"""只读 /api/v1 契约。缺失产物保持未知，不返回伪造的空成功。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from qa_pipeline.demo.app import create_app


def test_state_missing_is_an_error(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/v1/state")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "missing_state"
    assert "data" not in body


def test_reviews_missing_are_partial_and_runs_page(tmp_path: Path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "report.md").write_text("历史报告\n", encoding="utf-8")
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_b" / "stats.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state_snapshot.json").write_text(json.dumps({"schema_version": "state-snapshot-v1"}), encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    state = client.get("/api/v1/state")
    assert state.status_code == 200
    assert state.json()["data"]["schema_version"] == "state-snapshot-v1"
    assert state.json()["meta"]["schema_version"] == "v1"
    reviews = client.get("/api/v1/review-batches")
    assert reviews.status_code == 200
    body = reviews.json()
    assert body["data"] == []
    assert body["meta"]["availability"] == "historical_partial"
    assert "review_aggregates.jsonl" in body["meta"]["missing_fields"]
    listed = client.get("/api/v1/runs", params={"limit": 1})
    page = listed.json()
    assert page["meta"]["total"] == 2
    assert page["meta"]["next_cursor"] == "1"
    assert page["data"][0]["availability"] == "historical_partial"
    cursor = client.get("/api/v1/runs", params={"cursor": "bad"})
    assert cursor.status_code == 422
