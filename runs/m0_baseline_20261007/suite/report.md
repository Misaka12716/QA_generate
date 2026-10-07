# design_v2 实验报告

| id | kept | retention | answerability | evidence_grounded | nli_mean | judge_mean | type_entropy | s_ratio | tokens_per_10k_kept | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E1_direct | 3 | 1.0 | None | 1.0 | None | None | 0.0 | 1.0 | 8676667 | None |
| E1_anchor | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E1_ku | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E2_fixed | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E2_gap | 4 | 1.3333 | None | 1.0 | None | None | 0.0 | 0.75 | 6507500 | None |
| E3_diagnostic | 1 | 0.3333 | None | 1.0 | None | None | 0.0 | 1.0 | 8980000 | None |
| E4_multi | 3 | 1.0 | None | 1.0 | None | None | 0.0 | 0.0 | 6100000 | None |
| E5_annotation | 4 | None | None | None | None | None | None | None | None | None |
| E6_cost | 4 | 1.3333 | None | 1.0 | None | None | 0.0 | 0.75 | 6507500 | None |
| E6_behavior | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E7_rationale | 0 | 0.0 | None | None | None | None | 0.0 | 0.0 | None | None |
| E7_refusal | None | None | None | None | None | None | None | None | None | None |
| E8_scale | 4 | 1.0 | None | 1.0 | None | None | 0.0 | 0.75 | 0 | None |

## 说明

- **E1_baseline**：强基线。同样给定材料，一次生成问题、答案和证据，并做基本验证
- **E1_direct**：路线 D，直接证据约束生成
- **E1_anchor**：传统锚点路径，后续验证与直接路径相同
- **E1_ku**：知识单元路径，单元经原文核验后才出题
- **E2_fixed**：固定题型比例对照
- **E2_gap**：默认路径。允许零题，并做行为样本与抽样诊断
- **E3_diagnostic**：抽样诊断。无上下文答对不删除样本
- **E4_multi**：固定多次复验对照
- **E5_annotation**：导出通过门槛的 S/A 抽检表 未提供人工标签，不计算 Cohen's Kappa
- **E6_cost**：汇总默认路径的过滤漏斗、token 与费用 教师 qwen3.8-27b 为内网接口，API 费用按 0 计，tokens_per_10k_kept 为等价 token。
- **E6_behavior**：充分证据之外的说明不足行为样本
- **E7_rationale**：解释文本作为单独蒸馏因素
- **E7_refusal**：对比基线与默认路径在干扰上下文上的行为 "skip_sft"
- **E8_scale**：发布子集。只保留通过门槛的 S/A，隔离样本不进入该子集

## 运行约束

- 语料仅为 fixtures/sample_manual.md，规模小于设计稿的 500 chunk / 1 万条。
- 教师为 192.168.4.110:4000 的 qwen3.8-27b，API 费用记 0 美元，并报告等价 token。学生基座为本地 Qwen2.5-7B-Instruct。
- E5 无人工标签，不计算 Cohen's Kappa。事实遵循使用 NLI 与证据子串，未接入 RAGAS。
