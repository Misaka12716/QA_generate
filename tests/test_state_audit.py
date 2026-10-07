"""资产盘点不得把缺失或空文件当成空数据集。"""

from __future__ import annotations

import os
from pathlib import Path

from qa_pipeline.experiments.state_audit import inspect_repo_asset
from qa_pipeline.observability import stage_count_balanced


def test_missing_and_empty_files_are_blocked(tmp_path: Path):
    missing = inspect_repo_asset(
        tmp_path,
        {"asset_id": "missing_protocol", "role": "protocol", "relative_path": "missing.jsonl", "kind": "jsonl", "schema_version": "v1"},
    )
    assert missing["exists"] is False
    assert missing["record_count"] is None
    assert missing["integrity_status"] == "blocked"
    assert missing["missing_reason"] == "missing"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    inspected = inspect_repo_asset(
        tmp_path,
        {"asset_id": "empty_protocol", "role": "protocol", "relative_path": "empty.jsonl", "kind": "jsonl", "schema_version": "v1"},
    )
    assert inspected["exists"] is True
    assert inspected["record_count"] is None
    assert inspected["integrity_status"] == "blocked"
    assert inspected["missing_reason"] == "empty_file"
    before = empty.read_bytes()
    assert empty.read_bytes() == before


def test_symlink_escape_is_not_read(tmp_path: Path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"secret": true}\n', encoding="utf-8")
    link_parent = tmp_path / "repo"
    link_parent.mkdir()
    link = link_parent / "escape.jsonl"
    os.symlink(outside, link)
    inspected = inspect_repo_asset(
        link_parent,
        {"asset_id": "escape", "role": "protocol", "relative_path": "escape.jsonl", "kind": "jsonl", "schema_version": "v1"},
    )
    assert inspected["integrity_status"] == "blocked"
    assert inspected["missing_reason"] == "path_escape"
    assert inspected["record_count"] is None
    assert "secret" not in str(inspected)


def test_unknown_stage_count_is_not_zero():
    assert stage_count_balanced(None, 0, 1, 1) is None
    assert stage_count_balanced(2, 0, 1, 1) is True
