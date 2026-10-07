"""只读盘点已登记资产。不扫描任意磁盘，缺文件也不当成空数据集。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "state-snapshot-v1"

REPO_ASSETS: tuple[dict[str, str], ...] = (
    {"asset_id": "v22_train_jsonl", "role": "train_jsonl", "relative_path": "runs/drug_v22/E3_g0/sft/train.jsonl", "kind": "jsonl", "producing_run": "drug_v22", "schema_version": "zhixun_messages_v1"},
    {"asset_id": "v22_metrics", "role": "consumed_ids", "relative_path": "runs/drug_v22/metrics.json", "kind": "consumed_ids", "producing_run": "drug_v22", "schema_version": "metrics_json_v1"},
    {"asset_id": "v22_run_meta", "role": "run_meta", "relative_path": "runs/drug_v22/run_meta.json", "kind": "json", "producing_run": "drug_v22", "schema_version": "run_meta_v1"},
    {"asset_id": "v22_adapter_config", "role": "adapter_config", "relative_path": "runs/drug_v22/E3_g0/sft/lora/adapter/adapter_config.json", "kind": "json", "producing_run": "drug_v22", "schema_version": "peft_adapter_config"},
    {"asset_id": "v22_adapter_weights", "role": "adapter_weights", "relative_path": "runs/drug_v22/E3_g0/sft/lora/adapter/adapter_model.safetensors", "kind": "file", "producing_run": "drug_v22", "schema_version": "safetensors"},
    {"asset_id": "frozen_inventory", "role": "source_inventory", "relative_path": "data/campus_hospital_drug_instructions/frozen/inventory.jsonl", "kind": "jsonl", "producing_run": "frozen_corpus", "schema_version": "inventory_jsonl_v1"},
    {"asset_id": "frozen_subset_manifest", "role": "subset_manifest", "relative_path": "data/campus_hospital_drug_instructions/frozen/subset_manifest.json", "kind": "json", "producing_run": "frozen_corpus", "schema_version": "subset_manifest_v1"},
    {"asset_id": "frozen_chunk_freeze", "role": "chunk_freeze", "relative_path": "data/campus_hospital_drug_instructions/frozen/chunk_freeze.json", "kind": "json", "producing_run": "frozen_corpus", "schema_version": "chunk_freeze_v1"},
    {"asset_id": "frozen_subset_dir", "role": "frozen_subset", "relative_path": "data/campus_hospital_drug_instructions/frozen/subset", "kind": "directory", "producing_run": "frozen_corpus", "schema_version": "directory"},
    {"asset_id": "frozen_heldout_legacy", "role": "heldout_legacy", "relative_path": "data/campus_hospital_drug_instructions/frozen/heldout.jsonl", "kind": "jsonl", "producing_run": "frozen_corpus", "schema_version": "heldout_jsonl_v1"},
    {"asset_id": "frozen_heldout_protocol", "role": "heldout_protocol", "relative_path": "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl", "kind": "jsonl", "producing_run": "frozen_corpus", "schema_version": "heldout_protocol_v1"},
    {"asset_id": "frozen_train_families", "role": "frozen_train_families", "relative_path": "data/campus_hospital_drug_instructions/frozen/frozen_train_families.json", "kind": "json", "producing_run": "frozen_corpus", "schema_version": "frozen_train_families_v1"},
    {"asset_id": "source_universe", "role": "source_universe", "relative_path": "data/campus_hospital_drug_instructions/frozen/source_universe.json", "kind": "json", "producing_run": "frozen_corpus", "schema_version": "source_universe_v1"},
    {"asset_id": "v23_predictions", "role": "predictions", "relative_path": "runs/drug_v23_eval/exploratory/predictions.jsonl", "kind": "jsonl", "producing_run": "drug_v23_eval", "schema_version": "prediction_jsonl_v1"},
    {"asset_id": "v23_report", "role": "report", "relative_path": "runs/drug_v23_eval/report.md", "kind": "file", "producing_run": "drug_v23_eval", "schema_version": "markdown_report"},
    {"asset_id": "v24_report", "role": "report", "relative_path": "runs/drug_v24_audit/report.md", "kind": "file", "producing_run": "drug_v24_audit", "schema_version": "markdown_report"},
    {"asset_id": "v24_train_seen_predictions", "role": "predictions", "relative_path": "runs/drug_v24_rescore/train_seen_confirmed/predictions.jsonl", "kind": "jsonl", "producing_run": "drug_v24_rescore", "schema_version": "prediction_jsonl_v1"},
    {"asset_id": "v24_new_input_predictions", "role": "predictions", "relative_path": "runs/drug_v24_rescore/new_inputs/predictions.jsonl", "kind": "jsonl", "producing_run": "drug_v24_rescore", "schema_version": "prediction_jsonl_v1"},
    {"asset_id": "v25_batches", "role": "review_manifest", "relative_path": "runs/drug_v25_eval/review/batches.json", "kind": "json", "producing_run": "drug_v25_eval", "schema_version": "review_batches_v1"},
    {"asset_id": "v25_import_result", "role": "human_import_result", "relative_path": "runs/drug_v25_eval/review/import_result.json", "kind": "json", "producing_run": "drug_v25_eval", "schema_version": "import_result_v1"},
    {"asset_id": "v25_batch1_protocol", "role": "protocol", "relative_path": "runs/drug_v25_eval/batch1/protocol.jsonl", "kind": "jsonl", "producing_run": "drug_v25_eval", "schema_version": "protocol_jsonl_v1"},
    {"asset_id": "v25_experiment_a_protocol", "role": "protocol", "relative_path": "runs/drug_v25_eval/experiment_a/protocol.jsonl", "kind": "jsonl", "producing_run": "drug_v25_eval", "schema_version": "protocol_jsonl_v1"},
    {"asset_id": "v25_experiment_b_protocol", "role": "protocol", "relative_path": "runs/drug_v25_eval/experiment_b/protocol.jsonl", "kind": "jsonl", "producing_run": "drug_v25_eval", "schema_version": "protocol_jsonl_v1"},
    {"asset_id": "v25_report", "role": "report", "relative_path": "runs/drug_v25_eval/report.md", "kind": "file", "producing_run": "drug_v25_eval", "schema_version": "markdown_report"},
)

REGISTERED_MODELS: tuple[dict[str, str], ...] = (
    {"asset_id": "qwen25_7b_registered", "role": "base_model", "path": "/data1/pjw/models/Qwen2.5-7B-Instruct", "note": "v2.3 报告与 adapter 配置登记的基座"},
    {"asset_id": "sft_default_base", "role": "base_model_default", "path": "/data/pjw/data/models/Qwen2.5-7B-Instruct", "note": "sft.py DEFAULT_BASE 的环境变量回退后路径"},
)

CODE_FEATURES: tuple[dict[str, str], ...] = (
    {"feature_id": "pipeline", "relative_path": "src/qa_pipeline/pipeline.py"},
    {"feature_id": "import_review", "relative_path": "src/qa_pipeline/experiments/review_io.py"},
    {"feature_id": "eval_adapter", "relative_path": "src/qa_pipeline/experiments/adapter_eval.py"},
    {"feature_id": "v25_prepare", "relative_path": "src/qa_pipeline/experiments/v25_prepare.py"},
    {"feature_id": "teacher_review", "relative_path": "src/qa_pipeline/reviewing/service.py"},
    {"feature_id": "reviewed_eval", "relative_path": "src/qa_pipeline/experiments/reviewed_eval.py"},
    {"feature_id": "demo", "relative_path": "src/qa_pipeline/demo/app.py"},
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_snapshot(repo: Path) -> dict[str, Any]:
    def run(args: list[str], binary: bool = False) -> bytes | str:
        result = subprocess.run(args, cwd=repo, check=False, capture_output=True)
        if result.returncode != 0:
            return b"" if binary else ""
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")

    commit = str(run(["git", "rev-parse", "HEAD"])).strip()
    porcelain = str(run(["git", "status", "--porcelain"]))
    diff = run(["git", "diff", "HEAD"], binary=True)
    assert isinstance(diff, bytes)
    dirty = bool(porcelain.strip())
    untracked = []
    for line in porcelain.splitlines():
        if not line.startswith("??"):
            continue
        rel = line[3:].strip()
        candidate = repo / rel
        if candidate.is_file() and _inside(repo, candidate):
            untracked.append(_sha256_file(candidate))
    return {
        "git_commit": commit or None,
        "git_dirty": dirty,
        "git_status_sha256": _sha256_bytes(porcelain.encode("utf-8")),
        "git_diff_sha256": _sha256_bytes(diff),
        "untracked_content_sha256": _sha256_bytes("\n".join(untracked).encode("utf-8")) if untracked else None,
        "reproducible_baseline": None if dirty or not commit else commit,
        "note": "脏工作区不能当作可复现实验基线" if dirty else "工作区与 HEAD 一致",
    }


def _jsonl_count(path: Path) -> tuple[int | None, str | None]:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                return None, "invalid_jsonl"
            count += 1
    return count, None


def _consumed_ids(payload: Any) -> list[Any] | None:
    found: list[list[Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            value = node.get("consumed_ids")
            if isinstance(value, list):
                found.append(value)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                if isinstance(child, dict):
                    walk(child)

    walk(payload)
    return found[0] if found else None


def _blank(asset: dict[str, str], *, exists: bool, reason: str, integrity: str) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "role": asset["role"],
        "relative_path": asset.get("relative_path") or asset.get("path"),
        "exists": exists,
        "sha256": None,
        "schema_version": asset.get("schema_version"),
        "record_count": None,
        "producing_run": asset.get("producing_run"),
        "git_commit": None,
        "integrity_status": integrity,
        "missing_reason": reason,
    }


def inspect_repo_asset(repo: Path, asset: dict[str, str]) -> dict[str, Any]:
    relative = asset["relative_path"]
    root = repo.resolve()
    raw = root / relative
    if not _inside(root, raw):
        row = _blank(asset, exists=False, reason="path_escape", integrity="blocked")
        row["relative_path"] = relative
        return row
    path = raw.resolve()
    if not path.exists():
        return _blank(asset, exists=False, reason="missing", integrity="blocked")
    if asset["kind"] == "directory":
        if not path.is_dir():
            return _blank(asset, exists=True, reason="not_a_directory", integrity="blocked")
        children = [item for item in path.iterdir() if item.is_file()]
        row = _blank(asset, exists=True, reason=None, integrity="partial")
        row["exists"] = True
        row["child_file_n"] = len(children)
        row["missing_reason"] = None
        row["note"] = "目录存在只说明有文件，不是已验证样本数"
        return row
    if not path.is_file():
        return _blank(asset, exists=True, reason="not_a_file", integrity="blocked")
    size = path.stat().st_size
    if size == 0:
        row = _blank(asset, exists=True, reason="empty_file", integrity="blocked")
        row["sha256"] = _sha256_file(path)
        row["size_bytes"] = 0
        return row
    digest = _sha256_file(path)
    record_count: int | None = None
    reason = None
    integrity = "artifact_verified"
    extra: dict[str, Any] = {}
    if asset["kind"] == "jsonl":
        record_count, reason = _jsonl_count(path)
        if reason:
            integrity = "blocked"
            record_count = None
    elif asset["kind"] == "json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
            record_count = 1
        except json.JSONDecodeError:
            integrity = "blocked"
            reason = "invalid_json"
            record_count = None
    elif asset["kind"] == "consumed_ids":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            integrity = "blocked"
            reason = "invalid_json"
        else:
            ids = _consumed_ids(payload)
            if ids is None:
                integrity = "blocked"
                reason = "consumed_ids_missing"
                record_count = None
            else:
                record_count = len(ids)
                extra["container"] = "metrics.json"
                extra["note"] = "consumed_ids 嵌在 metrics.json，不是独立文件"
    elif asset["kind"] == "file":
        record_count = None
        extra["size_bytes"] = size
        extra["note"] = "文件哈希只证明字节未变，不证明实验已在当前代码上复现"
    row = {
        "asset_id": asset["asset_id"],
        "role": asset["role"],
        "relative_path": relative,
        "exists": True,
        "sha256": digest,
        "schema_version": asset.get("schema_version"),
        "record_count": record_count,
        "producing_run": asset.get("producing_run"),
        "git_commit": None,
        "integrity_status": integrity,
        "missing_reason": reason,
        "size_bytes": size,
    }
    row.update(extra)
    return row


def inspect_model(spec: dict[str, str]) -> dict[str, Any]:
    path = Path(spec["path"])
    base = {
        "asset_id": spec["asset_id"],
        "role": spec["role"],
        "relative_path": spec["path"],
        "exists": path.is_dir(),
        "sha256": None,
        "schema_version": "model_summary_v1",
        "record_count": None,
        "producing_run": None,
        "git_commit": None,
        "copied": False,
        "weights_hashed": False,
        "note": spec.get("note"),
    }
    if not path.exists():
        base["integrity_status"] = "blocked"
        base["missing_reason"] = "missing"
        return base
    if not path.is_dir():
        base["integrity_status"] = "blocked"
        base["missing_reason"] = "not_a_directory"
        return base
    config = path / "config.json"
    tokenizer = path / "tokenizer_config.json"
    weights = sorted([*path.glob("*.safetensors"), *path.glob("*.bin")])
    base.update(
        {
            "config_sha256": _sha256_file(config) if config.is_file() else None,
            "tokenizer_sha256": _sha256_file(tokenizer) if tokenizer.is_file() else None,
            "weight_files": [item.name for item in weights],
            "weight_bytes": sum(item.stat().st_size for item in weights),
            "integrity_status": "summary_verified" if config.is_file() and tokenizer.is_file() else "blocked",
            "missing_reason": None if config.is_file() and tokenizer.is_file() else "model_summary_incomplete",
        }
    )
    return base


def _asset_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["asset_id"]: row for row in rows}


def _stage(stage_id: str, status: str, scope: str, evidence: list[str], **extra: Any) -> dict[str, Any]:
    row = {"stage_id": stage_id, "status": status, "scope": scope, "evidence": evidence}
    row.update(extra)
    return row


def build_stages(repo: Path, assets: dict[str, dict[str, Any]], out_dir: Path | None) -> list[dict[str, Any]]:
    def ok(asset_id: str) -> bool:
        row = assets.get(asset_id) or {}
        return row.get("integrity_status") in {"artifact_verified", "summary_verified"} and row.get("exists") is True

    stages = []
    for feature in CODE_FEATURES:
        path = repo / feature["relative_path"]
        stages.append(
            _stage(
                feature["feature_id"],
                "implemented" if path.is_file() else "not_started",
                "只表示源码文件存在，不表示真实实验已执行",
                [feature["relative_path"]] if path.is_file() else [],
            )
        )
    smoke = None if out_dir is None else out_dir / "smoke" / "stats.json"
    suite = None if out_dir is None else out_dir / "suite" / "report.md"
    stages.append(
        _stage(
            "fake_smoke",
            "reproduced" if smoke and smoke.is_file() else "not_started",
            "FakeLLM 流程复现，不是教师或学生模型效果",
            [str(smoke)] if smoke and smoke.is_file() else [],
        )
    )
    stages.append(
        _stage(
            "fake_suite",
            "reproduced" if suite and suite.is_file() else "not_started",
            "FakeLLM 套件复现，不是真实训练或评测",
            [str(suite)] if suite and suite.is_file() else [],
        )
    )
    train_status = "artifact_verified" if ok("v22_train_jsonl") else "blocked"
    stages.append(
        _stage(
            "v22_g0_training",
            train_status,
            "历史单 seed 训练产物。run_meta 记载当时工作区为脏状态。本环境没有重新训练，不能标 reproduced",
            ["runs/drug_v22/E3_g0/sft/train.jsonl", "runs/drug_v22/run_meta.json"],
            report_claim="reported_executed",
        )
    )
    stages.append(
        _stage(
            "v23_predictions",
            "artifact_verified" if ok("v23_predictions") else "blocked",
            "本地预测文件可核对字节和行数。没有在本环境重跑推理",
            ["runs/drug_v23_eval/exploratory/predictions.jsonl", "runs/drug_v23_eval/report.md"],
            report_claim="reported_executed",
        )
    )
    stages.append(
        _stage(
            "v24_rescore",
            "artifact_verified" if ok("v24_train_seen_predictions") else "blocked",
            "v2.4 重评分与补推理文件在 drug_v24_rescore。人工裁定不因这些文件存在而完成",
            ["runs/drug_v24_rescore/train_seen_confirmed/predictions.jsonl", "runs/drug_v24_audit/report.md"],
            report_claim="reported_executed",
        )
    )
    stages.append(
        _stage(
            "v25_materials",
            "artifact_verified" if ok("v25_batch1_protocol") and ok("v25_experiment_a_protocol") and ok("v25_experiment_b_protocol") else "blocked",
            "协议和审核包已准备。ready_for_inference 仍取决于人工导入结果，不能把文件存在当成已审核",
            ["runs/drug_v25_eval/review/batches.json", "runs/drug_v25_eval/report.md"],
            report_claim="reported_executed",
        )
    )
    human = assets.get("v25_import_result") or {}
    stages.append(
        _stage(
            "human_gold",
            "blocked",
            "人工金标未完成。教师审核不能改写为 agreed 或 dual_agreed",
            ["runs/drug_v25_eval/review/import_result.json"] if human.get("exists") else [],
            report_claim="reported_executed",
            missing_reason="adjudicated_zero_or_unverified",
        )
    )
    heldout = assets.get("frozen_heldout_protocol") or {}
    formal_reason = heldout.get("missing_reason") or "heldout_protocol_unverified"
    if not (assets.get("frozen_train_families") or {}).get("exists") or not (assets.get("source_universe") or {}).get("exists"):
        formal_reason = "frozen_source_list_missing"
    stages.append(
        _stage(
            "formal_eval",
            "blocked",
            "正式评测仍要求人工金标、非空冻结来源清单和来源宇宙。空协议不能当成零样本成功",
            ["data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl"],
            missing_reason=formal_reason,
        )
    )
    stages.append(_stage("teacher_review_execution", "not_started", "代码可实现后，真实教师调用仍要凭据和预算。dry-run 不是审核结论", []))
    stages.append(_stage("reviewed_inference", "not_started", "A/B 执行器未获生成预算前不加载学生模型", []))
    stages.append(_stage("new_g0_training", "not_started", "没有诊断结论和训练预算，不启动新的 G0", []))
    stages.append(_stage("g1_g3", "not_started", "缺少真实学生探测器、有效选样和冻结回放池，保持未执行", []))
    return stages


def _historical_observability(repo: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in ("runs/drug_v22", "runs/drug_v23_eval", "runs/drug_v24_audit", "runs/drug_v25_eval"):
        path = repo / relative
        events = path / "events.jsonl"
        rows.append(
            {
                "run": relative,
                "exists": path.is_dir(),
                "observability": "events_present" if events.is_file() else "historical_partial",
                "missing_fields": [] if events.is_file() else ["events.jsonl"],
                "duration_sec": None,
            }
        )
    return rows


def collect_state(
    repo: Path,
    *,
    out_dir: Path | None = None,
    assets: tuple[dict[str, str], ...] | None = None,
    models: tuple[dict[str, str], ...] | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    chosen = assets if assets is not None else REPO_ASSETS
    chosen_models = models if models is not None else REGISTERED_MODELS
    asset_rows = [inspect_repo_asset(repo, item) for item in chosen]
    model_rows = [inspect_model(item) for item in chosen_models]
    command_results = None
    if out_dir is not None and (out_dir / "command_results.json").is_file():
        command_results = json.loads((out_dir / "command_results.json").read_text(encoding="utf-8"))
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "repo": str(repo),
        "git": _git_snapshot(repo),
        "assets": asset_rows + model_rows,
        "stages": build_stages(repo, _asset_map(asset_rows + model_rows), out_dir),
        "observability": _historical_observability(repo),
        "command_results": command_results,
        "blockers": _blockers(asset_rows, model_rows),
    }
    return snapshot


def _blockers(assets: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for item in assets + models:
        if item.get("integrity_status") == "blocked":
            rows.append(
                {
                    "asset_id": item["asset_id"],
                    "reason": str(item.get("missing_reason") or "blocked"),
                    "path": str(item.get("relative_path") or ""),
                }
            )
    rows.append({"asset_id": "teacher_api", "reason": "credentials_not_assumed", "path": ""})
    rows.append({"asset_id": "gpu_training", "reason": "no_training_budget", "path": ""})
    return rows


def render_report(snapshot: dict[str, Any]) -> str:
    git = snapshot["git"]
    lines = [
        "# 状态盘点",
        "",
        f"- git_commit: `{git.get('git_commit')}`",
        f"- git_dirty: `{git.get('git_dirty')}`",
        f"- reproducible_baseline: `{git.get('reproducible_baseline')}`",
        f"- {git.get('note')}",
        "",
        "阶段状态只描述本次盘点能够证明的范围。`reported_executed` 来自历史报告，`artifact_verified` 只表示文件通过了存在性、哈希或行数检查，`reproduced` 只用于本次 FakeLLM 命令。",
        "",
        "## 阶段",
        "",
    ]
    for stage in snapshot["stages"]:
        claim = stage.get("report_claim")
        suffix = f"；报告记载 {claim}" if claim else ""
        reason = stage.get("missing_reason")
        reason_text = f"；原因 {reason}" if reason else ""
        lines.append(f"- `{stage['stage_id']}`: {stage['status']}{suffix}{reason_text}。{stage['scope']}")
    lines.extend(["", "## 本次命令", ""])
    commands = snapshot.get("command_results") or {}
    if not commands:
        lines.append("- 未附带 command_results.json。")
    for key, value in commands.items():
        lines.append(f"- `{key}`: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["", "## 阻塞资产", ""])
    blocked = [item for item in snapshot["assets"] if item.get("integrity_status") == "blocked"]
    if not blocked:
        lines.append("- 登记资产没有 blocked 项。")
    for item in blocked:
        count = item.get("record_count")
        lines.append(
            f"- `{item['asset_id']}` {item.get('missing_reason')}，record_count={count}，路径 `{item.get('relative_path')}`"
        )
    lines.extend(["", "## 观测", ""])
    for row in snapshot["observability"]:
        lines.append(
            f"- `{row['run']}`: {row['observability']}，缺失 {row['missing_fields'] or '无'}，duration_sec={row['duration_sec']}"
        )
    lines.append("")
    lines.append("价格未知记为 unknown，不把历史内网教师的 0 美元估价写成已核对费用。")
    lines.append("")
    return "\n".join(lines)


def write_state(repo: Path, out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {
        item["asset_id"]: item.get("sha256")
        for item in [inspect_repo_asset(repo, asset) for asset in (kwargs.get("assets") or REPO_ASSETS)]
    }
    snapshot = collect_state(repo, out_dir=out_dir, **kwargs)
    after = {
        item["asset_id"]: item.get("sha256")
        for item in [inspect_repo_asset(repo, asset) for asset in (kwargs.get("assets") or REPO_ASSETS)]
    }
    snapshot["historical_hashes_unchanged"] = before == after
    (out_dir / "state_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "state_report.md").write_text(render_report(snapshot), encoding="utf-8")
    return snapshot
