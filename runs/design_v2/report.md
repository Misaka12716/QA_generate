# design_v2 实验报告

| id | kept | retention | answerability | evidence_grounded | nli_mean | judge_mean | type_entropy | s_ratio | tokens_per_10k_kept | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline | 77 | 0.9625 | 0.9625 | 1.0 | None | None | 0.0 | 0.0 | 4687792 | None |
| E1_direct | 72 | 0.96 | 0.96 | 1.0 | None | None | 0.8451 | 1.0 | 5450000 | None |
| E1_anchor | 63 | 0.9 | 0.9 | 1.0 | None | None | 1.0423 | 1.0 | 9989841 | None |
| E1_ku | 58 | 1.0 | 1.0 | 1.0 | None | None | 0.0 | 1.0 | 7789828 | None |
| E2_fixed | 76 | 0.962 | 0.962 | 1.0 | None | None | 0.7879 | 1.0 | 5366053 | None |
| E2_gap | 71 | 0.9861 | 0.9861 | 1.0 | None | None | 0.6741 | 0.9859 | 5551831 | None |
| E3_diagnostic | 72 | 0.96 | 0.96 | 1.0 | None | None | 0.8451 | 1.0 | 2392361 | None |
| E4_multi | 68 | 1.0 | 0.0 | 1.0 | None | None | 0.8262 | 0.0 | 0 | None |
| E5_annotation | 71 | None | None | None | None | None | None | None | None | None |
| E6_cost | 71 | 0.9861 | 0.9861 | 1.0 | None | None | 0.6741 | 0.9859 | 5551831 | None |
| E6_behavior | 69 | 0.9857 | 0.9857 | 1.0 | None | None | 0.9203 | 0.9855 | 5381449 | None |
| E7_rationale | 69 | 0.9079 | 0.9079 | 1.0 | None | None | 0.8413 | 1.0 | 10138261 | None |
| E7_refusal | None | None | None | None | None | None | None | None | None | None |
| E8_scale | 71 | 1.0 | 1.0 | 1.0 | None | None | 0.6741 | 0.9859 | 0 | None |

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
- **E7_refusal**：对比基线与默认路径在干扰上下文上的行为 {"E1_baseline": {"skipped": true, "reason": "no_adapter"}, "E2_gap": {"skipped": true, "reason": "no_adapter"}}
- **E8_scale**：发布子集。只保留通过门槛的 S/A，隔离样本不进入该子集

## 运行约束

- 语料仅为 fixtures/sample_manual.md，规模小于设计稿的 500 chunk / 1 万条。
- 教师为 192.168.4.110:4000 的 qwen3.8-27b，API 费用记 0 美元，并报告等价 token。学生基座为本地 Qwen2.5-7B-Instruct。
- E5 无人工标签，不计算 Cohen's Kappa。事实遵循使用 NLI 与证据子串，未接入 RAGAS。
- 单卡运行，CUDA_VISIBLE_DEVICES=3。
