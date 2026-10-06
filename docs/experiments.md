# 实验与评测

[返回 README](../README.md) · [使用指南](usage.md) · [架构与开发](development.md)

## 套件与数据前置条件

| 套件 | 输入来源 | 干净 checkout |
| --- | --- | --- |
| [suite.yaml](../configs/experiments/suite.yaml) | `fixtures/` 示例文档、heldout、拒答集 | 可运行 fake 流程 |
| [suite_psychiatry.yaml](../configs/experiments/suite_psychiatry.yaml) | `data/psychiatry/` | 需另行准备领域数据 |
| [suite_drug.yaml](../configs/experiments/suite_drug.yaml) | `data/campus_hospital_drug_instructions/frozen/` | 需冻结语料与 heldout 协议 |
| [suite_drug_v22_validate.yaml](../configs/experiments/suite_drug_v22_validate.yaml) | 领域验证套件所声明的数据和配置 | 逐项核对其路径与冻结产物 |

`data/` 默认不提交。提供历史报告并不意味着报告所需的输入、预测和模型都已随仓库分发。套件内相对的 `input / recipes_dir / heldout / refusal` 路径按仓库根目录解析，不按套件文件所在目录解析。

运行前检查主测试清单。清单缺失时套件在生成前中止，不回退旧题库；存在但为空时允许流程验证及适用的训练原题探针，主评测不能据此报告有效增益。

## 运行与浏览

```sh
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/local_suite --fake
qa-pipeline demo --run runs/local_suite --host 127.0.0.1 --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。对照台读取已有实验产物；不是触发生成和训练的管理服务。

示例套件包含 E1 生成路线比较、E2 固定比例与默认路径、E3 诊断、E4 复验、E5 抽检表、E6 成本与行为样本、E7 解释与拒答、E8 发布子集。具体分组始终以 YAML 为准，领域套件中的同名实验可能具有不同干预，不能仅凭编号类比。

主要输出为 `report.md`、`metrics.json`、`run_meta.json` 及各实验子目录。元数据记录代码提交、heldout 哈希、学生基座信息和可用的训练文件哈希。不同实验类型的产物不同；抽检、成本汇总和子集导出并非重新生成一轮 QA。

## 可选学生训练

先安装 `python -m pip install -e ".[sft]"`，准备可用的本地 Hugging Face 学生基座、GPU 和领域数据。下面的路径是占位符，需替换：

```sh
qa-pipeline experiment --suite configs/experiments/suite.yaml --out runs/local_sft --local-model /path/to/student-model --devices 0 --sft
```

- 教师仍走配置的 API；`--local-model` 指学生基座。
- 无 `--sft` 默认跳过训练；套件条目还会决定哪些实验臂参与 SFT。
- 当前 GPU 选择要求已用显存低于 2048 MiB、空闲显存至少 18432 MiB。即使显卡无人使用，也不代表显存足够。
- 训练与评测逻辑位于 [sft.py](../src/qa_pipeline/experiments/sft.py)；并行编排位于 [parallel.py](../src/qa_pipeline/experiments/parallel.py)。
- heldout 不进入训练；训练原题探针与独立主测试分别报告，不能互相替代。

真实 API、训练及推理运行应记录配方、输入版本和运行环境。不要将 fake 的 token、成本或质量指标解读为真实模型测量。

## 评测已有 adapter

`eval-adapter` 只评测已有 adapter，不启动训练、不重写训练 JSONL。需提供本地基座（包含 `tokenizer_config.json`）、adapter、评测协议和可复用的模型标识。先安装 SFT 可选依赖。以下是参数模板，不是仓库自带模型示例：

```sh
qa-pipeline eval-adapter --base-model /path/to/base-model --base-id base-v1 --adapter /path/to/adapter --adapter-id adapter-v1 --protocol /path/to/protocol.jsonl --out runs/local_adapter_eval --mode exploratory --device 0
```

`--mode` 必填，可取 `formal` 或 `exploratory`。默认评分器为 `aux-rules-v2`，默认最大新 token 数为 512。

**当前 CLI 的正式模式存在接口缺口：**正式协议校验需要 `frozen_train_families`，但 CLI 没有对应参数，也未传入冻结训练来源列表。因此仅添加 `--mode formal` 不能完成正式评测；非空正式协议会因缺少来源列表停止。需要由 Python 调用 [adapter_eval.py](../src/qa_pipeline/experiments/adapter_eval.py) 的 `eval_adapter(..., frozen_train_families=...)`，传入真实冻结清单及其他必需参数。不要为了通过校验伪造空清单或改成探索模式后宣称正式结果。

### 协议与复核

协议是 JSONL，每行一个用例。探索模式至少需唯一的 `case_id`（或 `id`）及非空问题（或用户消息）。正式模式还检查：

- 参考答案或答案要点、来源和 `expected_action`；
- 已复核状态、复核人员与意见；双人一致状态还要求双人记录与裁决信息；
- 与当前内容一致的 `review_content_hash`；
- 与冻结训练来源集合的隔离。

字段和条件以 `validate_protocol`、`review_content_hash` 和测试为准。内容变化后必须重新复核。正式协议有无效项时整批停止，不通过删除题目缩小分母。

### 产物、缓存与结果解释

查看 `protocol_check.json`、`eval_report.json` 和适用的预测、评分产物。协议失败或未执行时不一定存在预测文件。

预测缓存绑定实际消息、基座、adapter、模板与推理配置；评分版本和 gold 改变时可用 `rescore_saved` 复用匹配的已存预测。它是 Python 接口，不是当前 CLI 子命令。保存新版本应保留旧评分和历史身份信息。

`eval-adapter` 成功执行返回 0，捕获的协议错误或缺少 tokenizer 配置返回 1，返回未执行报告时返回 2。进一步判断应查看报告中的 `executed`、失败项和统计分母，不能只检查输出文件是否存在。

## 如何阅读实验结论

- 接受条数、导出条数、训练条数和主评测条数是不同分母。
- `student_diagnostic` 没有真实学生探测器时记录未执行；不能用教师回答代替学生诊断结论。
- `gap_fill` 可记录缺口规划；不能由策略名称推断已完成缺口回填。
- 回放池缺失时不能声称抗遗忘实验有效。
- 辅助规则分数不自动成为经过校准的主指标；缺失预测、错误预测、未复核题目和探索结果必须明确区分。
- 历史设计稿中的规模、预算账本和效果目标属于设计背景。当前实现的账本包含占位字段，不能视为完整的并发费用预留系统。
- `runs/` 的既有报告和[归档资料](archive/README.md) 保留当时条件，不作为当前版本自动验收凭证。
