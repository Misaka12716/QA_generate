# 质量实验执行状态

记录时间：2026-10-07。数值来自 `runs/quality_20261007/`。历史运行、冻结协议和 CB-1 预测没有回写。

## 检查

| 检查 | 结果 |
| --- | --- |
| `python -X utf8 -m pytest -q` | 141 passed，0 failed，0 skipped |
| `qa-pipeline list-strategies` | 含 `planner/capability_catalog`、`question_gen/planned_grounded`、`filters/entity_subject` |
| `audit-coverage` | 已写入 `runs/quality_20261007/audit_baseline/` |
| Fake 规划配方 | `runs/quality_20261007/fake_planned`，kept=5，rejected=0 |
| Fake 行为探针 | `runs/quality_20261007/fake_behavior`，5 条行为任务，不进入五类配额 |
| 真实教师调用、模型下载、GPU 训练 | 未执行 |

`test_v24_historical.py` 在本次完整测试中被收集并计入通过，没有出现 skip。

## 基线复算

命令：

```sh
qa-pipeline audit-coverage --out runs/quality_20261007/audit_baseline
```

| 分母 | 复算 | 计划规模 |
| --- | --- | --- |
| 历史训练行 | 74 | 74 |
| 闭卷训练行 | 22 | 22 |
| base / adapter 预测 | 50 / 50 | 50 / 50 |
| CB-paraphrase / retention / memory | 20 / 20 / 10 | 20 / 20 / 10 |
| 主评测输入 | 40 | 40 |

`differences` 为空。历史训练题型为事实 61、流程 10、条件 3，比较和多跳为 0。字符长度使用 Python `len`，含标点和空白。74 条答案均值 25.61，最近秩 P50 为 19，两端中点平均为 19.5。闭卷 22 条均值 38.55，最近秩 P50 为 25，两端中点平均为 28.5。后一个 28.5 与原报告的偶数中位数一致。

74 条来源族为空，审计没有用 `qfam` 回填。闭卷 22 条把样本族误写入来源族，迁移清单在 `migration/source_family_v2.jsonl`，`in_place` 为 false。对象错配样例 `qa_8deb52518a53` 的同文档历史 ID 已列出，等级未改。

已有预测的 `finish_reason` 保留原值，证据标为 `inferred_from_length`，`last_token` 为 `not_recorded`。tokenizer 计数为 `not_executed`。adapter 50 条存储停止原因均为 stop，没有触达长度上限；base 为 stop 45、length 5。

## 阶段状态

| 阶段 | 状态 | 原因 |
| --- | --- | --- |
| A 诊断协议 | 已冻结 20 条，标记 `dev_diagnostic` | 预测未启动。GPU 查询为 available，但没有当前教师余额，不能把旧账本当成余额 |
| B 试产 100+5 | `not_executed` | 药品原文目录存在，本次未授权真实生成 |
| C 三臂训练 | `not_executed` | 没有新的合格池。共享超参 `proposed_not_frozen`，seed 42 只是提案 |
| D | `not_executed` | 等待阶段 C 的可复验胜出者 |
| 正式锁定测试 | `not_executed` | 120 个输入 × 4 个模型 = 480 次预测审核，另加最多 120 次 gold。旧账本是 `snapshot_not_current_balance`，额度不够不启动，分母不缩小 |

方向筛选门槛仍是：复杂题完整且有依据完成率相对 C0 至少高 10 个百分点，事实题和通用保留退化不超过 5 个百分点。本次没有 C0/C1/C2 预测，不能写改善或无改善。

Fake 运行只验证策略接线。它的 token 和费用不是教师或学生的真实用量。
