# QA 数据生成、蒸馏与知识过滤插件

可拔插的 QA 管线，按 YAML Recipe 组装策略：Chunking → 锚点 → 提问 → Evol → 教师蒸馏 → 过滤链 → S/A/B 分层。产物可导出为智训平台 JSONL，本阶段**不修改** `data_governance` 业务流程。

设计依据见仓库内《微调平台 QA 数据侧 — QA 数据生成、蒸馏及知识过滤技术设计稿.md》。

## 安装

```bash
cd /home/mmc/workspace/qa_generate
# 使用智训同一 conda 环境亦可
pip install -e ".[dev]"
# 可选：本地 NLI / 向量 / LoRA
# pip install -e ".[nli,embed,sft]"
```

教师 API 默认读取环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`，或智训仓库根目录的 `gpt_api` 文件（DeepSeek 兼容）。

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

# 五组方案对比实验（默认跳过 SFT）
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/ablation --fake

# 含 LoRA SFT（Qwen2.5-0.5B，需 GPU 与 extras）
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/ablation --sft
```

`run` 目录会写出 `qa.kept.jsonl`、`qa.raw.jsonl`、`zhixun.jsonl`、`stats.json`。智训 JSONL 字段为 `messages` + `metadata.evidence`，证据尽量保持为 chunk 子串，便于日后手工导入数据集发布。

## 可拔插策略

| 阶段 | 策略名 |
| --- | --- |
| chunking | `fixed_overlap` `heading_window` `semantic_boundary` |
| anchor | `none` `tfidf_keyword` `ner_rake_textrank` `llm_extract` |
| question_gen | `direct_qa` `anchor_reverse` `self_instruct` `answer_aware` |
| evolution | `none` `evol_depth` `evol_breadth` `evol_both` |
| distillation | `concise_response` `cot_mixed` `multi_teacher_judge` |
| teacher_router | `single` `grade_by_qtype` `always_strong` |
| filter | `rule_clean` `exact_hash_dedup` `minhash_dedup` `semdedup` `nli_fact` `llm_judge` `knowledge_ablation` `evidence_substring` `llm_supported` `diversity_sample` |
| grading | `sab` `binary` `none` |

Recipe 只写策略名和参数，见 `configs/recipes/`。新增算法：在对应 `plugins/` 模块用 `@register("stage", "name")` 注册即可。

## 实验套件

`configs/experiments/suite.yaml`：

| ID | 目的 |
| --- | --- |
| E1_baseline | 复现智训：字符窗 + 一次生成 Q/A + 证据子串 + LLM supported |
| E2_recommended | 锚点反向提问 + CoT 混合 + 规则/MinHash/NLI/Judge/消融 + SAB |
| E3_evol | E2 + Evol-Instruct 深度 25% |
| E4_* | 同一批蒸馏结果上消融 NLI / 消融 / Judge |
| E5_cost_min / E5_quality_max | 成本帕累托两端 |

输出 `runs/<suite>/report.md` 与 `metrics.json`（保留率、NLI、知识增益、Judge、类型熵、S 级占比、估算成本、可选 SFT ΔF1）。

SFT 对齐智训 `training_worker`：Qwen2.5-0.5B-Instruct + LoRA r=16、3 epoch、chat template、只监督 assistant。评测用 `fixtures/heldout.jsonl`（不进入训练）。

## 测试

```bash
pytest -q
```

单测全程 FakeLLM，不依赖网络或 GPU。
