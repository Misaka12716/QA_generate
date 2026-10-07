# CB-1 闭卷首轮

日期：2026-10-07。这是单种子探索运行，不是正式人工评测。

## 结论

教师审核没有执行。主评测 40 个输入都生成了 base 与新闭卷 adapter 的回答，50 个案例（含 10 个 CB-memory）的对照关系全部是未决。不能把文字差异写成改善，也不能写成没有改善。

训练 loss 从约 3.95 降到末步约 1.54，最终 `train_loss` 为 1.629。这只说明监督损失下降，不说明闭卷问答变好。

## 数据

历史 `runs/drug_v22/E3_g0/sft/train.jsonl` 有 74 条，全部是 `rag_grounded`。逐条去掉“根据资料 / 该药物”等指代后，22 条问题在脱离原文后仍然点名对象。其余 52 条没有进入训练集。

实际训练 22 条，少于计划上限 100。按来源家族轮转选出 10 个知识单元，各 2 个表面改写，得到 20 个 CB-paraphrase。改写规则在看模型输出之前写进协议，不是教师生成的新问法。另有 20 个冻结通用保留题。CB-memory 10 题是训练原题，单独列出，不计入 40 个主评测输入。

学生消息只有闭卷 system 和问题。抽查训练样本的 loss mask：prompt 位置为 -100，只有 assistant 目标参与监督。预测 JSONL 中没有“资料：”或“参考答案”。

## 模型

基座 `/data1/pjw/models/Qwen2.5-7B-Instruct`。新 adapter 在 `sft/adapter`，种子 42，3 个 epoch，18 步，监督 token 2151。训练进程设置 `CUDA_VISIBLE_DEVICES=3`。预测使用物理 GPU 3，`max_new_tokens=256`，贪心解码。历史 G0 adapter 没有换名复用。

50 对回答的文字都不相同。保留题上能看到 adapter 回答更短，例如“12乘以8等于多少？”两侧都能写出 96，但这不是教师评分。平均回答长度大约是 base 135 字、adapter 13 字。长度变化不能当成事实正确。

## 预算与未做的审核

共享账本仍是 97/120 次调用、107092/200000 token，剩余 23 次调用和 92908 token。主评测若每题一次参考审核、每份预测一次审核，需要 120 次。缺口约 97 次。因此没有调用教师。CB-memory 若另审，还会再增加调用，没有计入这 120 次。

`wrong_to_right`、`right_to_wrong`、`both_correct`、`both_wrong` 都是空值。未决 40/40 表示缺少裁判。`formal_ready` 为 0。教师 revision 仍是 unknown。

## 产物

`protocol.json`、`splits/knowledge_exposure.json`、`data/train.jsonl`、`sft/sft_metrics.json`、`predictions/base.jsonl`、`predictions/adapter.jsonl`、`cases.jsonl`、`metrics.json`、`manifest.json`。
