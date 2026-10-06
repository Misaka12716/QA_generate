"""根据落盘 JSON 生成 drug_v23 报告。不读取终端历史。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .scoring import aggregate_families
from .v22_audit import _sha256

REPO = Path(__file__).resolve().parents[3]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "pass_pass"
    if (not before) and after:
        return "fail_pass"
    if before and (not after):
        return "pass_fail"
    return "fail_fail"


def _stratum_summary(rows: list[dict]) -> dict:
    packed_after = []
    packed_before = []
    cells = Counter()
    for row in rows:
        before = bool((row.get("before") or {}).get("passed"))
        after = bool((row.get("after") or {}).get("passed"))
        cells[_transition(before, after)] += 1
        base = {**row, "passed": before, "family_id": row.get("family_id") or row.get("case_id")}
        tuned = {**row, "passed": after, "family_id": row.get("family_id") or row.get("case_id")}
        packed_before.append(base)
        packed_after.append(tuned)
    before_summary = aggregate_families(packed_before)
    after_summary = aggregate_families(packed_after)
    return {
        "row_count": len(rows),
        "family_count": after_summary["family_count"],
        "source_family_count": after_summary["source_family_count"],
        "auxiliary_pass_rate_before": before_summary["task_accuracy"],
        "auxiliary_pass_rate_after": after_summary["task_accuracy"],
        "weighting": after_summary["weighting"],
        "transitions": dict(cells),
        "note": "未完成人工审核，仅供探索。不是正式语义正确率。",
    }


def render_report(dest: Path | None = None) -> str:
    root = dest or (REPO / "runs/drug_v23_eval")
    audit = _read(REPO / "runs/drug_v23_audit/audit.json")
    manifest = _read(REPO / "runs/drug_v23_audit/manifest.json")
    status = _read(root / "candidate_status.json")
    ledger = _read(root / "teacher_ledger.json")
    report_path = root / "exploratory/eval_report.json"
    run_ids_path = root / "exploratory/run_ids.json"
    run_ids = _read(run_ids_path) if run_ids_path.is_file() else {}
    exploratory = _read(report_path) if report_path.is_file() else None
    scores = _jsonl(root / "exploratory/scores.jsonl")
    predictions = _jsonl(root / "exploratory/predictions.jsonl")
    hash_checks = []
    for item in manifest["files"]:
        path = Path(item["path"])
        if not item.get("exists"):
            hash_checks.append({"path": item["path"], "unchanged": False, "reason": "missing_at_audit"})
            continue
        current = _sha256(path) if path.is_file() else None
        hash_checks.append({"path": item["path"], "unchanged": current == item.get("sha256")})
    by_stratum: dict[str, list] = {}
    for row in scores:
        by_stratum.setdefault(row.get("stratum") or "unspecified", []).append(row)
    strata = {name: _stratum_summary(rows) for name, rows in sorted(by_stratum.items())}
    truncated = sum(1 for row in predictions if row.get("truncated"))
    errors = [row for row in predictions if row.get("error")]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in predictions)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in predictions)
    devices = sorted({row.get("device") for row in predictions if row.get("device") is not None})
    lines = [
        "# drug_v23 评测报告",
        "",
        "本报告只根据 `runs/drug_v23_audit` 与 `runs/drug_v23_eval` 的落盘文件汇总。",
        "探索性辅助分标记为：未完成人工审核，仅供探索。它不是正式主指标，也不是已确认的语义正确率。",
        "",
        "## A 代码与测试",
        "",
        "- 新增 `eval-adapter`：加载已有基座和 adapter，写原始预测，再离线评分。不调用 `train_lora`，不重写 `train.jsonl`。",
        "- 预测缓存包含实际消息、模型与 adapter 标识、tokenizer 配置哈希和完整推理配置。评分另用金标与评分器版本。",
        "- `aux-rules-v1` 仍是原来的 `score_task`。`aux-rules-v2` 只写新字段。",
        "- 冻结加载按 `subset_manifest.json` 读 txt，跳过 `snapshot_manifest.json` 等元数据；路径缺失即失败。",
        "- 测试：`python -m pytest tests/test_adapter_eval.py tests/test_contracts.py tests/test_pipeline.py`。",
        "- 真实推理使用已安装依赖的解释器 `/opt/miniconda/envs/opsd/bin/python`。默认解释器因 `huggingface-hub==1.8.0` 无法导入当前 transformers，没有改它的包版本。",
        "",
        "本次探索性运行使用的标识：",
        "",
        f"- base_id：`{run_ids.get('base_id')}`",
        f"- adapter_id：`{run_ids.get('adapter_id')}`",
        f"- template_id：`{run_ids.get('template_id')}`",
        f"- 当时空闲且显存足够的 GPU：{run_ids.get('gpu_candidates')}。推理实际使用其中一张，见预测记录里的 device。",
        "",
        "```bash",
        "PYTHONPATH=src /opt/miniconda/envs/opsd/bin/python -m qa_pipeline eval-adapter \\",
        "  --base-model /data1/pjw/models/Qwen2.5-7B-Instruct \\",
        "  --base-id " + str(run_ids.get("base_id") or "") + " \\",
        "  --adapter runs/drug_v22/E3_g0/sft/lora/adapter \\",
        "  --adapter-id " + str(run_ids.get("adapter_id") or "") + " \\",
        "  --protocol runs/drug_v23_eval/candidates/dev_candidates.jsonl \\",
        "  --out runs/drug_v23_eval/exploratory \\",
        "  --mode exploratory \\",
        "  --scorer aux-rules-v2 \\",
        "  --max-new-tokens 512",
        "```",
        "",
        "正式模式 `--mode formal` 会在加载模型前拒绝未审核题。本次没有发布正式协议。",
        "",
        "已有能力：独立评测入口、ID 对齐、预测/评分缓存、协议门禁、家族等权聚合、冻结正文加载、v2 辅助规则。",
        "后续待办：多 seed、学生诊断、回放、补题、G1–G3、人工双审后的正式协议。",
        "",
        "## B drug_v22 审计",
        "",
        f"- 基座：`{audit['base_model_confirmed']}`。adapter 配置与 `run_meta.json` 一致。另一路径 `/data/pjw/data/models/Qwen2.5-7B-Instruct` 不存在，没有替换。",
        f"- adapter：`{audit['adapter_path']}`。训练 seed={audit['sft_seed']}，不是三个 seed。",
        f"- 历史 GPU 记录 {audit['historical_gpu_schedule']} 只是当时调度。",
        f"- 训练导出、E1 合格池、consumed_ids、train_seen 的 ID 与顺序一致，问题在 user 消息中，assistant 目标与合格池答案一致。内容差异条数：{len(audit['alignment']['content_diffs'])}。",
        "- E1 `qa.kept.jsonl` 与 G0 `qa.kept.jsonl` 的题干和答案相同，差异字段是 `audit`、`data_stage`、`included_in_this_run`。",
        f"- 非空 family_id 的核心问题 {audit['alignment']['family_id_nonempty_unique']} 个；family_id 为空的训练行 {audit['alignment']['empty_family_id']} 条。`source_family_id` 在训练记录中为空，来源族写在外置 `source_index.json`。",
        f"- 旧辅助四格：通过→通过 {audit['auxiliary_transition']['pass_pass']}，未通过→通过 {audit['auxiliary_transition']['fail_pass']}，通过→未通过 {audit['auxiliary_transition']['pass_fail']}，未通过→未通过 {audit['auxiliary_transition']['fail_fail']}。离线重算通过标记不一致的 ID 数：{len(audit['auxiliary_transition']['rescore_pass_mismatch_ids'])}。该数字不进入泛化总分。",
        f"- 漏斗：子集 txt {audit['funnel']['subset_txt_files']}；落盘 chunk {audit['funnel']['chunk_rows']}，其中来源数 {audit['funnel']['chunk_source_n']}。stats 没有单独的“已读正文”计数。问题与原始答案各 {audit['funnel']['questions']}。`rule_clean` 89→80，后续过滤器未再丢弃。`qa.rejected.jsonl` 有 {audit['funnel']['rejected_file_rows']} 条隔离样本。原始 ID 中有 {audit['funnel']['raw_ids_absent_from_kept_and_rejected']} 条既不在合格池也不在 rejected 文件，与 rule_clean 丢弃数一致；这 9 条没有逐条原因文件。",
        f"- 教师 token：prompt {audit['metrics']['tokens']['prompt_tokens']}，completion {audit['metrics']['tokens']['completion_tokens']}，合计 {audit['metrics']['tokens']['sum']}。每条保留样本 {audit['metrics']['tokens']['tokens_per_kept']} token。`tokens_per_10k_kept` 复算为 {audit['metrics']['tokens']['tokens_per_10k_kept']}。",
        f"- G0 教师 token 为 0，因为没有新的教师生成。监督 token {audit['metrics']['tokens']['supervised_tokens']} 属于训练目标，不在该折算指标里。优化步 {audit['metrics']['optimizer_steps']['global_step']}，与 ceil(74/4)*3 一致。`consumed_ids` 是进入 Dataset 的 ID，不是逐批消费日志。",
        "- retention 是 74/89 的样本保留率。evidence_grounded=1.0 只表示证据片段能在上下文中找到。s_ratio=1.0 只表示规则分级全是 S。type_entropy=0.8074 对应 factual 61、procedural 10、conditional 3。",
        "- answerability 为空是因为没有 answerable 标签。nli_mean 与 judge_mean 为空是因为该配方没有跑 NLI 和裁判。delta_f1 为空是因为主测试协议是空文件，主评测没有加载模型。这些空值都不是 0。",
        "- 没有可追溯的人工审核。现有 G0 与基座路径匹配，可以复用做新输入推理；训练记录缺 source_family_id 不影响权重文件本身。",
        "",
        "审计后复核历史文件哈希：",
        "",
    ]
    changed = [item["path"] for item in hash_checks if not item["unchanged"]]
    lines.append(f"- 与审计基线相比，被改动的历史文件数：{len(changed)}。")
    if changed:
        lines.extend(f"- 变化：`{path}`" for path in changed)
    lines.extend(
        [
            "",
            "## C 新测试与审核材料",
            "",
            f"- 原题审核包：{status['review_pack']['questions']} 题、{status['review_pack']['answers']} 份回答。审核者版本匿名随机排列，映射在 `review/analysis_map.json`。已填写的人工裁定：{status['review_pack']['filled_adjudications']}。",
            f"- 候选 {status['candidate_n']} 条，请求 100。已见知识新问法 {status['rephrase_n']}，独立来源 {status['independent_n']}，行为题 {status['behavior_n']}。状态：{status['review_status']}。正式协议写入：{status['formal_protocol_written']}。",
            f"- 核心问题：可用非空 family {status['core_questions']['available_core_questions']}，选取 {status['core_questions']['selected']}。family_id 为空的训练 ID 没有拿来复制凑数。",
            f"- 独立来源家族 {status['independent_sources']['independent_source_count']}，其中未见通用名 {status['independent_sources']['unseen_generic_name_count']}。近重复筛除 {status['independent_sources']['near_or_exact_rejected']}。扫描上限 {status['independent_sources']['scan_cap']}，实际扫描开发文件 {status['independent_sources']['scanned_dev_files']}。",
            f"- 行为题缺失模式：{status['behavior_missing_patterns'] or '无'}。",
            "- 开发候选来自 dev 分区或证据改写，没有写入 `heldout_protocol.jsonl`，也没有用 locked_test 调评分器。",
            "- 新题审核表 `review/new_items_for_review.jsonl` 先给题干、证据和候选金标，没有附模型输出。这些金标未经人工接受。",
            "",
            "教师账本：",
            "",
            f"- 模型 {ledger.get('model')}，温度 {ledger.get('temperature')}，max_tokens {ledger.get('max_tokens')}，调用上限 {ledger.get('max_calls')}。",
            f"- 实际调用 {ledger.get('calls')}，重试 {ledger.get('retries')}，prompt token {ledger.get('prompt_tokens')}，completion token {ledger.get('completion_tokens')}。停止原因：{ledger.get('stopped_reason')}。",
            "",
            "需要人工完成、程序不能代填的审核：",
            "",
            "- 74 道原题的训练目标，以及 148 份匿名回答。",
            f"- {status['candidate_n']} 道新题的题干、证据和候选金标。审核通过之前不能发布正式协议。",
            "",
            "## D 新评测结果",
            "",
        ]
    )
    if not exploratory or not exploratory.get("executed"):
        reason = None if not exploratory else exploratory.get("reason")
        lines.extend(
            [
                f"- 探索性推理未完成。原因：{reason or '没有 eval_report.json'}。",
                "- 没有用模拟回答填充分数。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- 结果类别：{exploratory.get('result_class')}。正式主指标：{exploratory.get('formal_main_metric')}。formal_gold={exploratory.get('formal_gold')}。",
                f"- 声明：{exploratory.get('disclaimer')}",
                f"- planned_n={exploratory.get('planned_n')}，scored_n={exploratory.get('scored_n')}，failed_n={exploratory.get('failed_n')}，cache_hits={exploratory.get('cache_hits')}。",
                f"- 技术错误条数：{len(errors)}。截断条数：{truncated}。推理 prompt token {prompt_tokens}，completion token {completion_tokens}。设备：{devices or '记录为空'}。",
                "- 三个层次分开汇总，没有合成泛化总分。行级转移按题目计数。来源等权通过率先在核心问题内平均，再对来源等权；同一核心问题的两条改写不会把该问题算两次。",
                "",
            ]
        )
        for name, summary in strata.items():
            cells = summary["transitions"]
            before_rows = cells.get("pass_pass", 0) + cells.get("pass_fail", 0)
            after_rows = cells.get("pass_pass", 0) + cells.get("fail_pass", 0)
            lines.append(
                f"- {name}：行 {summary['row_count']}，核心问题 {summary['family_count']}，来源 {summary['source_family_count']}。"
                f"行级辅助通过 {before_rows}/{summary['row_count']} → {after_rows}/{summary['row_count']}。"
                f"来源等权辅助通过率 {summary['auxiliary_pass_rate_before']} → {summary['auxiliary_pass_rate_after']}。"
                f"转移 {summary['transitions']}。"
            )
        lines.extend(["", "历史 T0 的 148 份原题回答没有重新推理，仍以 drug_v22 的 train_seen.jsonl 为准。", ""])
    lines.extend(["## E 下一步", ""])
    lines.extend(_next_steps(strata if exploratory and exploratory.get("executed") else None))
    lines.extend(
        [
            "",
            "本次没有做：重跑 drug_v22、新训 G0、G1–G3、回放、补题、其他生成臂、全库出题、多 seed、提交或部署。",
            "",
        ]
    )
    text = "\n".join(lines)
    (root / "report.md").write_text(text, encoding="utf-8")
    (root / "strata.json").write_text(json.dumps(strata, ensure_ascii=False, indent=2), encoding="utf-8")
    return text


def _next_steps(strata: dict | None) -> list[str]:
    if not strata:
        return ["- 推理没有完成，先解决运行环境后再做探索性评测。不要启动 G1–G3。"]
    lines = []
    rephrase = strata.get("seen_rephrase") or {}
    indep = strata.get("independent") or {}
    behavior = strata.get("behavior") or {}
    re_cells = rephrase.get("transitions") or {}
    in_cells = indep.get("transitions") or {}
    be_cells = behavior.get("transitions") or {}
    if re_cells.get("fail_pass", 0) > re_cells.get("pass_fail", 0) and re_cells.get("fail_fail", 0):
        lines.append(
            "- 已见知识的新问法有辅助分改善，同时仍有前后都未通过的题。先人工核对这批评分是不是规则误判；确认是真实错答后，再考虑表达覆盖。"
        )
    if in_cells and (in_cells.get("pass_pass", 0) >= in_cells.get("fail_pass", 0)):
        lines.append(
            "- 独立来源题在基座上的辅助通过已经很高，G0 几乎没有新增通过。这些题是教师按证据原文写出的、尚未人工审核的抽取题，不能据此宣称独立泛化。"
        )
    if be_cells.get("fail_fail", 0) > be_cells.get("fail_pass", 0):
        lines.append(
            "- 行为题里多数辅助失败没有被 G0 纠正，而且不少预期是说明资料不足或澄清。下一步先核对行为边界和可见证据，不把这一层并进总分。"
        )
    lines.append("- 全部新题仍是 unreviewed。审核若改了题干或上下文，对应项要重新推理；只改判定、不改输入时可以复用已有回答。")
    lines.append("- 原题 52/22 仍是旧辅助分。若人工审核发现主要是评分误判，先改评分，再决定是否做训练 seed 稳定性实验。")
    lines.append("- 学生诊断、有效降权、合格回放池和预算控制都还没有。现在不计划 G1–G3，也不重新调用教师生成训练数据。")
    return lines


if __name__ == "__main__":
    render_report()
