# QA Generate

从来源文档生成带证据的问答数据，经过蒸馏、验证、过滤和分级，导出智训平台可用的 JSONL。项目提供可配置管线、方案对照实验、结果浏览界面和已有 LoRA adapter 的评测入口。

当前推荐配方采用：标题切块 → 直接证据生成 → 复用候选答案 → 主张验证与风险复验 → 去重 → 分级发布。默认任务是 `rag_grounded`，导出的用户消息包含学生可见资料。项目通过文件对接智训平台，不直接修改其 `data_governance` 业务流程。

## 快速开始

需要 Python **3.10 或更高版本**。以下命令从仓库根目录执行；建议先创建并激活虚拟环境。

```sh
git clone https://github.com/Misaka12716/QA_generate.git
cd QA_generate
python -m pip install -e ".[dev]"
qa-pipeline run --recipe configs/recipes/smoke.yaml --input fixtures/sample_manual.md --out runs/local_smoke --fake
```

`--fake` 使用内置 FakeLLM，不调用真实教师 API。这个例子用于验证流程，不代表生成质量或模型效果。

成功后在 `runs/local_smoke/` 查看 `qa.kept.jsonl`、`qa.rejected.jsonl`、`zhixun.jsonl` 和 `stats.json`。完整产物说明见[使用指南](docs/usage.md#运行产物)。

```sh
# 运行示例实验套件（默认不训练）
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/local_suite --fake
# 浏览上述实验结果；启动后访问 http://127.0.0.1:8765
qa-pipeline demo --run runs/local_suite
# 运行测试
python -X utf8 -m pytest -q
```

## 文档入口

| 你要做什么 | 阅读文档 |
| --- | --- |
| 配置教师、输入文档、选择配方、导出数据 | [使用指南](docs/usage.md) |
| 运行实验、训练学生、评测已有 adapter、理解结果限制 | [实验与评测](docs/experiments.md) |
| 理解模块边界、数据契约、增加策略、运行开发检查 | [架构与开发](docs/development.md) |
| 让编码代理修改这个仓库 | [AGENTS.md](AGENTS.md) |
| 查阅旧设计稿、试点报告和演示推演 | [历史资料](docs/archive/README.md) |

## 使用边界

- 首次使用真实教师前，显式配置服务地址及配方中的模型名。代码保留了原实验内网默认地址；`--model` 不会覆盖配方的 `teacher_models`，详见[配置规则](docs/usage.md#教师配置)。
- `recommended.yaml` 是小规模配方，当前最多使用 3 个 chunk、8 个候选问题，不是全量生产配置。
- 配方中出现策略名，不等于对应实验已执行。学生探测器、回放池或覆盖目录缺失时，相关策略可能记录 `not_executed`。
- 仓库保留部分历史报告，但不包含完整领域语料、模型和运行中间文件。历史报告不等于当前版本已复现的效果结论。

当前文档以源码、配置和测试为依据；旧设计稿集中归档，保留其原始结论与时间背景。
