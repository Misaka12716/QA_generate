# QA 数据生成、蒸馏与知识过滤插件

可拔插的 QA 管线。默认配方按现行设计稿一期路径组装：结构切分 → 直接证据生成（或可选知识单元）→ 复用候选答案 → 主张验证 → 联合去重 → 门槛分层。`rag_grounded` 导出把可见资料放进 messages。产物可导出为智训平台 JSONL，本阶段**不修改** `data_governance` 业务流程。

设计依据见仓库内《微调平台 QA 数据侧 — QA 数据生成、蒸馏及知识过滤技术设计稿.md》。

## 安装

```bash
cd /home/mmc/workspace/qa_generate
# 使用智训同一 conda 环境亦可
pip install -e ".[dev]"
# 可选：本地 NLI / 向量 / LoRA
# pip install -e ".[nli,embed,sft]"
```

教师默认是 `192.168.4.110:4000` 上的 `qwen3.8-27b`（OpenAI 兼容，`default` / `cheap` / `strong` 同一模型）。可用环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 覆盖。

可选重模型（不设则自动降级，并在 `stats.fallbacks` 记录）：

| 环境变量 | 作用 |
| --- | --- |
| `QA_PIPELINE_NLI_MODEL` | 如 `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| `QA_PIPELINE_EMBED_MODEL` | 如 `BAAI/bge-small-zh-v1.5` |

## CLI

```bash
# 查看已注册策略
qa-pipeline list-strategies

# 按配方跑管线（示例文档）
qa-pipeline run --recipe configs/recipes/recommended.yaml --input fixtures/sample_manual.md --out runs/demo

# 不打真实 API
qa-pipeline run --recipe configs/recipes/smoke.yaml --input fixtures/sample_manual.md --out runs/smoke --fake

# 导出智训 JSONL
qa-pipeline export --run runs/demo --out runs/demo/zhixun.jsonl

# 设计稿 E1–E8（默认跳过 SFT）
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/design_suite --fake

# 教师为 192.168.4.110:4000 的 qwen3.8-27b，本地 Qwen2.5-7B 只作 LoRA 学生基座
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/design_suite \
  --local-model /data1/pjw/models/Qwen2.5-7B-Instruct --sft
```

`run` 目录会写出 `qa.kept.jsonl`、`qa.raw.jsonl`、`zhixun.jsonl`、`stats.json`。智训 JSONL 字段为 `messages` + `metadata.evidence`，证据尽量保持为 chunk 子串，便于日后手工导入数据集发布。

## 可拔插策略

| 阶段 | 策略名 |
| --- | --- |
| chunking | `fixed_overlap` `heading_window` `semantic_boundary` |
| anchor | `none` `tfidf_keyword` `ner_rake_textrank` `llm_extract` |
| question_gen | `direct_grounded` `knowledge_unit` `direct_qa` `anchor_reverse` `self_instruct` `answer_aware` |
| evolution | `none` `evol_depth` `evol_breadth` `evol_both` `tag_evol` `evol_unconstrained` |
| question_filter | `none` `difficulty_sample` |
| distillation | `reuse_candidate` `concise_response` `cot_mixed` `multi_teacher_judge` `routed_teacher` |
| teacher_router | `single` `grade_by_qtype` `always_strong` |
| filter | `rule_clean` `evidence_substring` `claim_evidence` `risk_escalate` `behavior_insufficient` `student_diagnostic` `joint_dedup` `exact_hash_dedup` `minhash_dedup` `semdedup` `nli_fact` `roundtrip` `selfcheck` `llm_judge` `knowledge_ablation` `llm_supported` `diversity_sample` |
| grading | `validity_tier` `sab` `binary` `none` |

Recipe 只写策略名和参数，见 `configs/recipes/`。新增算法：在对应 `plugins/` 模块用 `@register("stage", "name")` 注册即可。

## 实验套件

`configs/experiments/suite.yaml`：

| ID | 目的 |
| --- | --- |
| E1_baseline / E1_direct / E1_anchor / E1_ku | 强基线、直接生成、传统锚点、知识单元 |
| E2_fixed / E2_gap | 固定比例对照 / 默认路径 |
| E3_diagnostic | 抽样诊断，不因无上下文答对而删除 |
| E4_multi | 固定多次复验对照 |
| E5_annotation | 导出 S/A 抽检表 |
| E6_cost / E6_behavior | 成本漏斗 / 说明不足行为样本 |
| E7_rationale / E7_refusal | 解释蒸馏 / 干扰上下文行为 |
| E8_scale | 通过门槛的 S/A 子集 |

输出 `runs/<suite>/report.md` 与 `metrics.json`（保留率、NLI、知识增益、Judge、类型熵、S 级占比、估算成本、可选 SFT ΔF1）。

SFT 对齐智训 `training_worker`：本地 Qwen2.5-7B-Instruct + LoRA r=16、3 epoch、chat template、只监督 assistant。评测用 `fixtures/heldout.jsonl`（不进入训练）。拒答用 `fixtures/refusal.jsonl`。

## 测试

```bash
pytest -q
```

单测全程 FakeLLM，不依赖网络或 GPU。
