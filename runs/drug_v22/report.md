# drug_v22 实验报告

| id | kept | retention | answerability | evidence_grounded | nli_mean | judge_mean | type_entropy | s_ratio | tokens_per_10k_kept | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_direct | 74 | 0.8315 | None | 1.0 | None | None | 0.8074 | 1.0 | 10855676 | None |
| E3_g0 | 74 | 1.0 | None | 1.0 | None | None | 0.8074 | 1.0 | 0 | None |
| E1_anchor | None | None | None | None | None | None | None | None | None | None |
| E1_kseq | None | None | None | None | None | None | None | None | None | None |
| E1_kjoint | None | None | None | None | None | None | None | None | None | None |
| E1_mix | None | None | None | None | None | None | None | None | None | None |
| E2_gap | None | None | None | None | None | None | None | None | None | None |
| E3_g1 | None | None | None | None | None | None | None | None | None | None |
| E3_g2 | None | None | None | None | None | None | None | None | None | None |
| E3_g3 | None | None | None | None | None | None | None | None | None | None |
| E6_conflict | None | None | None | None | None | None | None | None | None | None |
| E_replay_budget_2 | None | None | None | None | None | None | None | None | None | None |
| E_full_corpus | None | None | None | None | None | None | None | None | None | None |

## 说明

- **E1_direct**：直接证据生成，作为 G0 的合格池
- **E3_g0**：代表臂。无诊断、无回放、无补题，只验证训练链路和训练原题前后对照
- **E1_anchor**：锚点路线 本轮流程验证不跑，不构成路线对照。
- **E1_kseq**：顺序知识单元 本轮流程验证不跑，不构成路线对照。
- **E1_kjoint**：联合知识单元 本轮流程验证不跑，不构成路线对照。
- **E1_mix**：混合路由 本轮流程验证不跑，不构成路线对照。
- **E2_gap**：覆盖缺口补题 补题未执行。目录仍是示例文件，没有在冻结训练文档上走完补题闭环。
- **E3_g1**：诊断选样 学生诊断未执行。没有与学生基座一致的探针。
- **E3_g2**：回放替换 第一种回放干预未执行。没有冻结保留池。
- **E3_g3**：诊断加回放 学生诊断未执行，第一种回放干预未执行。
- **E6_conflict**：冲突行为通道 冲突臂未执行。未确认可合法使用的冲突说明书，不编造样本。
- **E_replay_budget_2**：第二种回放预算 第二种回放预算未检验。新数据条数固定后再追加回放，本轮未跑。
- **E_full_corpus**：全库出题 全库出题未执行。本轮仍只用 60 篇冻结子集。

## 运行约束

- 本轮是修复后的流程验证与训练集学习诊断，run_id=drug_v22。不是五组 LoRA 效果对照。
- 主测试没有双人金标时主指标未执行，delta_f1 为空。不排列组别，不写独立泛化提升。
- train_seen 只说明已见样本是否学到，不进入泛化总分。task_accuracy 只作辅助，不是已验证的语义正确率。
- 学生诊断未执行。第一种回放干预未执行。补题未执行。第二种回放预算未检验。
- 冲突臂与全库出题未执行。锚点、K 路线、混合路由、E4–E8 本轮不调用教师。
- 多维题型和同义改写未做。后续才区分训练原题、已见知识点的新问法和未见来源的多维题。
- 旧 heldout.jsonl 题干歧义，不可用于主效果结论。
- 教师为 192.168.4.110:4000 的 qwen3.8-27b。学生是本地 Qwen2.5-7B-Instruct。输入是 60 篇冻结子集，最多 100 个 chunk、150 条样本。
- 主测试清单存在但没有可评分题目。允许训练和训练原题探针，主评测不加载模型，delta_f1 为空。
- 管线并行于 GPU [3, 4, 6]。教师为 qwen3.8-27b，实验卡只加载 NLI、向量和 LoRA。
