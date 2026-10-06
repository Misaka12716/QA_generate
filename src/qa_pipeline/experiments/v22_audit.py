"""只读复核 drug_v22，并把基线写到新目录。不修改历史产物。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..textutil import entropy
from .scoring import score_task, transition_label

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs" / "drug_v22"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _user(row: dict[str, Any]) -> str:
    return next(message.get("content") or "" for message in row["messages"] if message.get("role") == "user")


def _assistant(row: dict[str, Any]) -> str:
    return next(message.get("content") or "" for message in row["messages"] if message.get("role") == "assistant")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "bytes": stat.st_size, "sha256": _sha256(path)}


def write_audit(dest: Path | None = None) -> dict[str, Any]:
    out = dest or (REPO / "runs" / "drug_v23_audit")
    out.mkdir(parents=True, exist_ok=True)
    files = [
        RUN / "run_meta.json",
        RUN / "metrics.json",
        RUN / "report.md",
        RUN / "E1_direct/stats.json",
        RUN / "E1_direct/metrics.json",
        RUN / "E1_direct/qa.kept.jsonl",
        RUN / "E1_direct/qa.raw.jsonl",
        RUN / "E1_direct/qa.rejected.jsonl",
        RUN / "E1_direct/questions.jsonl",
        RUN / "E1_direct/chunks.jsonl",
        RUN / "E3_g0/qa.kept.jsonl",
        RUN / "E3_g0/sft/train.jsonl",
        RUN / "E3_g0/sft/train_seen.jsonl",
        RUN / "E3_g0/sft/eval.json",
        RUN / "E3_g0/sft/lora/sft_metrics.json",
        RUN / "E3_g0/sft/lora/adapter/adapter_config.json",
        RUN / "E3_g0/sft/lora/adapter/adapter_model.safetensors",
        REPO / "data/campus_hospital_drug_instructions/frozen/heldout.jsonl",
        REPO / "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl",
        REPO / "data/campus_hospital_drug_instructions/frozen/subset_manifest.json",
        REPO / "data/campus_hospital_drug_instructions/frozen/chunk_freeze.json",
    ]
    manifest = {
        "note": "这些哈希是本次审计时的保护基线，不能证明文件在此之前没有被改过。",
        "files": [_file_record(path) for path in files],
    }
    base_model = Path("/data1/pjw/models/Qwen2.5-7B-Instruct")
    alt_model = Path("/data/pjw/data/models/Qwen2.5-7B-Instruct")
    manifest["base_model"] = {
        "path": str(base_model),
        "exists": base_model.is_dir(),
        "config_sha256": _sha256(base_model / "config.json") if (base_model / "config.json").is_file() else None,
    }
    manifest["alternate_base_model_exists"] = alt_model.exists()
    git = _git_state()
    manifest["git_at_audit"] = git
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    train = _jsonl(RUN / "E3_g0/sft/train.jsonl")
    kept = _jsonl(RUN / "E1_direct/qa.kept.jsonl")
    g0_kept = _jsonl(RUN / "E3_g0/qa.kept.jsonl")
    raw = _jsonl(RUN / "E1_direct/qa.raw.jsonl")
    rejected = _jsonl(RUN / "E1_direct/qa.rejected.jsonl")
    seen = _jsonl(RUN / "E3_g0/sft/train_seen.jsonl")
    questions = _jsonl(RUN / "E1_direct/questions.jsonl")
    chunks = _jsonl(RUN / "E1_direct/chunks.jsonl")
    stats = json.loads((RUN / "E1_direct/stats.json").read_text(encoding="utf-8"))
    metrics = json.loads((RUN / "E1_direct/metrics.json").read_text(encoding="utf-8"))
    sft = json.loads((RUN / "E3_g0/sft/lora/sft_metrics.json").read_text(encoding="utf-8"))
    adapter_cfg = json.loads((RUN / "E3_g0/sft/lora/adapter/adapter_config.json").read_text(encoding="utf-8"))
    run_meta = json.loads((RUN / "run_meta.json").read_text(encoding="utf-8"))

    train_ids = [row["id"] for row in train]
    kept_ids = [row.get("qa_id") for row in kept]
    kept_by_id = {row.get("qa_id"): row for row in kept}
    content_diffs = []
    for row in train:
        source = kept_by_id.get(row["id"])
        if source is None:
            content_diffs.append({"id": row["id"], "reason": "missing_in_kept"})
            continue
        answer = _assistant(row)
        user = _user(row)
        if answer != (source.get("answer") or ""):
            content_diffs.append({"id": row["id"], "reason": "assistant_ne_answer"})
        if (source.get("question") or "") not in user:
            content_diffs.append({"id": row["id"], "reason": "question_not_in_user"})
    field_diffs = _kept_field_diffs(kept, g0_kept)
    seen_ids = [row["case_id"] for row in seen]
    seen_diffs = []
    train_by_id = {row["id"]: row for row in train}
    for row in seen:
        source = train_by_id.get(row["case_id"])
        if source is None:
            seen_diffs.append({"id": row["case_id"], "reason": "missing_train"})
            continue
        if row.get("messages") != source["messages"][:-1]:
            seen_diffs.append({"id": row["case_id"], "reason": "messages_differ"})
        if row.get("answer") != source["messages"][-1].get("content"):
            seen_diffs.append({"id": row["case_id"], "reason": "gold_differ"})
    cells = Counter()
    label_mismatch = []
    rescored_mismatch = []
    for row in seen:
        before_pass = bool(row["before"]["passed"])
        after_pass = bool(row["after"]["passed"])
        if before_pass and after_pass:
            cells["pass_pass"] += 1
        elif (not before_pass) and after_pass:
            cells["fail_pass"] += 1
        elif before_pass and not after_pass:
            cells["pass_fail"] += 1
        else:
            cells["fail_fail"] += 1
        expected = transition_label(row["before"], row["after"])
        if row.get("change") != expected:
            label_mismatch.append({"id": row["case_id"], "stored": row.get("change"), "recomputed": expected})
        case = {"answer": row["answer"], "answer_points": [row["answer"]], "expected_action": "answer"}
        fresh_before = score_task(row.get("base_answer") or "", case)
        fresh_after = score_task(row.get("tuned_answer") or "", case)
        if fresh_before["passed"] != before_pass or fresh_after["passed"] != after_pass:
            rescored_mismatch.append(row["case_id"])
    consumed = [str(item) for item in sft.get("consumed_ids") or []]
    families = [(row.get("metadata") or {}).get("family_id") or "" for row in train]
    qtypes = Counter((row.get("metadata") or {}).get("q_type") for row in train)
    prompt = int(stats["prompt_tokens"])
    completion = int(stats["completion_tokens"])
    kept_n = len(train)
    raw_ids = {row.get("qa_id") for row in raw}
    kept_id_set = set(kept_ids)
    rejected_ids = {row.get("qa_id") for row in rejected}
    dropped_ids = sorted(raw_ids - kept_id_set - rejected_ids)
    chunk_sources = []
    for row in chunks:
        chunk_sources.append(str(row.get("source_doc") or row.get("doc_id") or (row.get("metadata") or {}).get("source_doc") or ""))
    subset_dir = REPO / "data/campus_hospital_drug_instructions/frozen/subset"
    subset_txt = sorted(path.name for path in subset_dir.glob("*.txt")) if subset_dir.is_dir() else []
    report = {
        "scope": "只读复核。没有重跑 drug_v22，没有改写其文件。",
        "base_model_confirmed": adapter_cfg.get("base_model_name_or_path"),
        "base_model_matches_run_meta": adapter_cfg.get("base_model_name_or_path") == run_meta.get("base_model"),
        "alternate_base_not_used": not alt_model.exists(),
        "adapter_path": str(RUN / "E3_g0/sft/lora/adapter"),
        "historical_gpu_schedule": json.loads((RUN / "metrics.json").read_text(encoding="utf-8")).get("devices"),
        "historical_gpu_note": "这是当时的调度记录，不是三个训练 seed，也不是下一轮必须使用的卡。",
        "sft_seed": sft.get("seed"),
        "alignment": {
            "train_n": len(train),
            "kept_n": len(kept),
            "g0_kept_n": len(g0_kept),
            "train_ids_unique": len(set(train_ids)) == len(train_ids),
            "id_set_equal": set(train_ids) == set(kept_ids) == set(row.get("qa_id") for row in g0_kept),
            "order_train_equals_e1_kept": train_ids == kept_ids,
            "content_diffs": content_diffs,
            "seen_n": len(seen),
            "seen_ids_equal_train_order": seen_ids == train_ids,
            "seen_content_diffs": seen_diffs,
            "consumed_ids_n": len(consumed),
            "consumed_ids_equal_train_ids": consumed == train_ids,
            "skipped_rows": sft.get("skipped_rows"),
            "e1_vs_g0_kept_field_diffs": field_diffs,
            "empty_source_family_id": sum(1 for row in train if not (row.get("metadata") or {}).get("source_family_id")),
            "empty_family_id": sum(1 for item in families if not item),
            "family_id_nonempty_unique": len({item for item in families if item}),
            "source_values_unique": len({(row.get("metadata") or {}).get("source") for row in train}),
            "expected_action": dict(Counter((row.get("metadata") or {}).get("expected_action") for row in train)),
            "evidence_state": dict(Counter((row.get("metadata") or {}).get("evidence_state") for row in train)),
        },
        "auxiliary_transition": {
            "scorer": "aux-rules-v1",
            "source": "train_seen.jsonl before.passed / after.passed",
            "pass_pass": cells["pass_pass"],
            "fail_pass": cells["fail_pass"],
            "pass_fail": cells["pass_fail"],
            "fail_fail": cells["fail_fail"],
            "label_mismatch": label_mismatch,
            "rescore_pass_mismatch_ids": rescored_mismatch,
            "enters_generalization_score": False,
            "note": "这是旧辅助通过率的复核，不是语义正确率。",
        },
        "funnel": {
            "subset_txt_files": len(subset_txt),
            "documents_read_counter": None,
            "documents_read_note": "stats 没有单独的正文字段。输入目录现有 txt 数如上。chunks.jsonl 的来源数见 chunk_source_n。",
            "chunk_rows": len(chunks),
            "chunk_source_n": len({item for item in chunk_sources if item}),
            "questions": len(questions),
            "raw_answers": len(raw),
            "stats_produced": stats.get("produced"),
            "funnel": stats.get("funnel"),
            "rejected_by_filter": stats.get("rejected"),
            "rejected_file_rows": len(rejected),
            "rejected_grades": dict(Counter(row.get("grade") for row in rejected)),
            "raw_ids_absent_from_kept_and_rejected": len(dropped_ids),
            "absent_id_sample": dropped_ids[:12],
            "stage_counts": stats.get("stage_counts"),
            "interpretation": "rule_clean 的 dropped 计数来自 stats.funnel。被该阶段移除的样本没有写入 qa.rejected.jsonl。qa.rejected.jsonl 是过滤后未发布的隔离样本。",
        },
        "metrics": {
            "retention": {
                "value": metrics.get("retention"),
                "numerator": metrics.get("kept"),
                "denominator": metrics.get("distilled"),
                "meaning": "样本保留率 kept/distilled，不是模型准确率，也不是防遗忘率。",
            },
            "evidence_grounded": {
                "value": metrics.get("evidence_grounded"),
                "detail": metrics.get("evidence_grounded_detail"),
                "meaning": "sufficient 样本的证据片段是否出现在可见上下文中，不证明答案语义正确。",
            },
            "s_ratio": {
                "value": metrics.get("s_ratio"),
                "grade_distribution": metrics.get("grade_distribution"),
                "meaning": "当前规则把保留样本判为 S 的比例，不是人工正确率。",
            },
            "type_entropy": {
                "value": metrics.get("type_entropy"),
                "counts": dict(qtypes),
                "recomputed": round(entropy({key: value for key, value in qtypes.items() if key}), 4),
                "meaning": "保留样本 q_type 的 log2 香农熵。口径是生成时写入的 q_type，不是人工题型。",
            },
            "answerability": {"value": metrics.get("answerability"), "reason": metrics.get("answerability_reason")},
            "nli_mean": {
                "value": metrics.get("nli_mean"),
                "reason": "配方 filters 没有 NLI 阶段，74 条 metadata.nli_score 均为空。空值不是 0，也不是已经通过。",
            },
            "judge_mean": {
                "value": metrics.get("judge_mean"),
                "reason": "配方没有教师裁判阶段，74 条 judge_overall 均为空。空值不是 0。",
            },
            "delta_f1": {
                "value": None,
                "reason": "heldout_protocol.jsonl 为 0 字节。eval.json 的 main_eval=model_not_loaded，主指标未执行。空值不是提升为零。",
            },
            "tokens": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "sum": prompt + completion,
                "tokens_per_10k_kept": int(round((prompt + completion) / kept_n * 10000)),
                "tokens_per_kept": round((prompt + completion) / kept_n, 4),
                "reported_tokens_per_10k_kept": 10855676,
                "g0_teacher_tokens": 0,
                "g0_teacher_token_note": "G0 从已有合格池选样，没有新的教师生成。0 不包含 LoRA 训练或原题推理成本。",
                "supervised_tokens": sft.get("supervised_tokens"),
                "llm_calls": stats.get("llm_calls"),
                "call_statuses": stats.get("call_statuses"),
                "call_status_note": "call_statuses 只统计部分 JSON 调用的终态，不是 llm_calls 的失败拆分。",
            },
            "optimizer_steps": {
                "global_step": sft.get("global_step"),
                "epochs": sft.get("epochs"),
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "expected_steps": 57,
                "formula": "ceil(74/4)*3=57。consumed_ids 是进入 Dataset 的 74 个 ID，不是逐批消费日志。",
                "exposure_count": sft.get("exposure_count"),
            },
        },
        "human_review": {
            "train_seen_review_status": "没有可追溯的双人审核记录。",
            "heldout_protocol_bytes": (REPO / "data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl").stat().st_size,
        },
        "reuse_g0": {
            "adapter_matches_recorded_base": adapter_cfg.get("base_model_name_or_path") == "/data1/pjw/models/Qwen2.5-7B-Instruct",
            "train_export_matches_kept_ids_and_targets": not content_diffs and set(train_ids) == set(kept_ids),
            "blocking_issue": None,
            "note": "可以复用这对基座和 adapter 做新输入推理。source_family_id 在训练记录中为空，来源族要靠外置映射，不影响权重本身。",
        },
    }
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _kept_field_diffs(left: list[dict], right: list[dict]) -> list[dict[str, Any]]:
    diffs = []
    by_id = {row.get("qa_id"): row for row in right}
    for row in left:
        other = by_id.get(row.get("qa_id"))
        if other is None:
            diffs.append({"id": row.get("qa_id"), "reason": "missing"})
            continue
        changed = []
        for key in sorted(set(row) | set(other)):
            if row.get(key) != other.get(key):
                changed.append(key)
        if changed:
            diffs.append({"id": row.get("qa_id"), "fields": changed})
    return diffs


def _git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        completed = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
        return (completed.stdout or completed.stderr).strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--porcelain"]),
        "note": "这是审计命令执行时的工作区，不是 drug_v22 开跑时的 git_status。",
    }


if __name__ == "__main__":
    write_audit()
