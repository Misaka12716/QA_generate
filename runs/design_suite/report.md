# design_e1_e8 实验报告

| id | kept | retention | answerability | evidence_grounded | nli_mean | judge_mean | type_entropy | s_ratio | tokens_per_10k_kept | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline | 6 | 0.6667 | 0.6667 | 1.0 | None | None | 0.0 | 0.0 | 11068333 | 0.1319 |
| E1_ours_full | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E1_ours_s_only | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E2_a1 | 10 | 0.8333 | 0.8333 | 1.0 | None | None | 1.6855 | 0.0 | 21419000 | None |
| E2_a2 | 4 | 0.8 | 0.8 | 1.0 | None | None | 1.5 | 0.0 | 25147500 | None |
| E2_a3 | 5 | 1.0 | 1.0 | 1.0 | None | None | 2.3219 | 0.0 | 21936000 | None |
| E2_a4 | 7 | 0.7778 | 0.7778 | 1.0 | None | None | 1.9502 | 0.0 | 22758571 | None |
| E2_a5 | 4 | 1.0 | 1.0 | 1.0 | None | None | 2.0 | 0.0 | 19890000 | None |
| E3_b1 | 4 | 0.8 | 0.8 | 1.0 | None | 5.0 | 1.5 | 0.0 | 9312500 | None |
| E3_b2 | 4 | 0.8 | 0.8 | 1.0 | None | 5.0 | 1.5 | 0.0 | 11865000 | None |
| E3_b3 | 4 | 0.8 | 0.8 | 1.0 | None | 5.0 | 1.5 | 0.0 | 11112500 | None |
| E3_b4 | 4 | 0.8 | 0.8 | 1.0 | None | 5.0 | 1.5 | 0.0 | 11140000 | None |
| E4_c1 | 1 | 0.0833 | 0.0833 | 1.0 | 1.0 | None | 0.0 | 0.0 | 137510000 | None |
| E4_c2 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E4_c3 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E4_c4 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E4_c5 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E5_annotation | 0 | None | None | None | None | None | None | None | None | None |
| E6_cost | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E7_refusal | None | None | None | None | None | None | None | None | None | None |
| E8_d1 | 6 | 1.0 | 1.0 | 1.0 | None | None | 0.0 | 0.0 | 0 | 0.1319 |
| E8_d2 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E8_d3 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |
| E8_d4 | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.0 | 0.0 | None | None |

## 说明

- **E1_baseline**：智训现状方案，一次生成 Q+A+证据并做规则过滤
- **E1_ours_full**：推荐路线全量，训练时使用 S+A
- **E1_ours_s_only**：推荐路线仅 S 级
- **E2_a1**：纯 Self-Instruct，无锚点无进化
- **E2_a2**：锚点反向提问，不进化
- **E2_a3**：锚点 + Tag-Evol 受控进化
- **E2_a4**：锚点 + 无约束 Evol-Instruct
- **E2_a5**：锚点 + Tag-Evol + 难度感知采样
- **E3_b1**：全量低成本单教师
- **E3_b2**：全量强教师
- **E3_b3**：按题型分级路由
- **E3_b4**：仅难题子集多教师选优
- **E4_c1**：无锚点，纯 LLM 自由生成
- **E4_c2**：仅 TF-IDF 关键词
- **E4_c3**：NER + TF-IDF
- **E4_c4**：NER + TF-IDF + 关键句
- **E4_c5**：纯 LLM 抽取锚点
- **E5_annotation**：导出 S/A 抽检表；无人工标签则不计算 Kappa 未提供人工标签，不计算 Cohen's Kappa
- **E6_cost**：汇总推荐路线的过滤漏斗、token 与费用 教师 qwen3.8-27b 为内网接口，API 费用按 0 计，tokens_per_10k_kept 为等价 token。
- **E7_refusal**：对比基线与推荐路线在干扰上下文上的拒答 {"E1_baseline": {"n": 10, "refusal_rate": 1.0, "gold_f1": 0.6649, "false_refusal_rate": 0.0, "refusal_f1": 1.0}, "E1_ours_full": {"skipped": true, "reason": "no_adapter"}}
- **E8_d1**：基线全量训练数据
- **E8_d2**：仅 S 级
- **E8_d3**：S+A
- **E8_d4**：S+A+B

## 运行约束

- 语料仅为 fixtures/sample_manual.md，规模小于设计稿的 500 chunk / 1 万条。
- 教师为 192.168.4.110:4000 的 qwen3.8-27b，API 费用记 0 美元，并报告等价 token。学生基座为本地 Qwen2.5-7B-Instruct。
- E5 无人工标签，不计算 Cohen's Kappa。事实遵循使用 NLI 与证据子串，未接入 RAGAS。
- 管线并行于 GPU [3, 4, 6]。教师为 qwen3.8-27b，实验卡只加载 NLI、向量和 LoRA。
