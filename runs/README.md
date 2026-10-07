# 实验运行目录索引

本目录存放管线与实验产物。Git 入库规则见根目录 [`.gitignore`](../.gitignore)：**问答对、账本 jsonl、报告与小体积 JSON 可提交**；LoRA 权重、`_cache`、日志与 HuggingFace 大 tokenizer 文件不进仓库。本地可保留完整树（含 adapter），克隆仓库后需自行训练或从外部取权重。

## 当前应以何为准

| 目录 | 性质 | 请优先参考 |
| --- | --- | --- |
| [`batch1_view_20261007`](batch1_view_20261007/) | 历史 batch1 + 实验 B 小试联合账本 | 本目录 `metrics.json`、`cases.jsonl`、`result_report.md` |
| [`b_pilot_20261007`](b_pilot_20261007/) | 与上表 `metrics.json` **字节相同**的别名快照 | 只看 `batch1_view_20261007`，勿计为两次独立实验 |
| [`drug_v22`](drug_v22/) | 药品域流程验证与 G0 训练链 | `report.md`、`E3_g0/sft/train.jsonl`（闭卷导出 `--source`） |
| [`drug_v25_eval`](drug_v25_eval/) | 当前评测与人工审核叙事 | `report.md`、`review/` |
| [`cb1_20261007`](cb1_20261007/) | 2026-10-07 闭卷单种子探索 | `result_report.md`、`metrics.json`；adapter 仅本地 |

## 历史 / 非主线（勿当作当前契约）

| 目录 | 说明 |
| --- | --- |
| [`drug_v21`](drug_v21/) | 早期全臂探索；小样本 token F1 不稳定。流程以 **drug_v22** 为准。 |
| [`design_suite`](design_suite/) / [`design_v2`](design_v2/) | 精神病学试点与设计套件历史跑；结论见 [v2 试点实验报告](../docs/archive/v2试点实验报告.md)。 |
| [`drug_v24_audit`](drug_v24_audit/) / [`drug_v24_rescore`](drug_v24_rescore/) / [`drug_v24_review`](drug_v24_review/) | v25 前置审计与重打分；叙事合并到 **drug_v25_eval**。 |
| [`drug_v23_eval`](drug_v23_eval/) / [`drug_v23_audit`](drug_v23_audit/) | 探索性预测与输入身份校验；集成测试读取部分文件。 |
| [`m0_baseline_20261007`](m0_baseline_20261007/) | 状态审计与 smoke 快照，非模型评测结论。 |

## 根目录脚本

`launch_*.sh`、`qwen_download.*` 为环境运维用，默认不提交。新实验请使用新的 `runs/<名称>/` 子目录，勿覆盖已入库的报告与账本。
