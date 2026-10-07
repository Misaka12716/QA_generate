# 实验与评测

[返回 README](../README.md) · [使用指南](usage.md) · [架构与开发](development.md)

## 套件与数据前置条件

| 套件 | 输入来源 | 干净 checkout |
| --- | --- | --- |
| [suite.yaml](../configs/experiments/suite.yaml) | `fixtures/` 示例文档、heldout、拒答集 | 可运行 fake 流程 |
| [suite_psychiatry.yaml](../configs/experiments/suite_psychiatry.yaml) | `data/psychiatry/` | 仓库含 `eval/` 协议；`train/`、`heldout/` 正文需本地准备 |
| [suite_drug.yaml](../configs/experiments/suite_drug.yaml) | `data/campus_hospital_drug_instructions/frozen/` | 仓库含冻结清单与 heldout 协议；`raw/` 与 zip 不进 Git |
| [suite_drug_v22_validate.yaml](../configs/experiments/suite_drug_v22_validate.yaml) | 领域验证套件所声明的数据和配置 | 逐项核对其路径与冻结产物 |

语料**原文**（`data/**/raw/`、`train/`、`heldout/`、zip、`_cache/`）默认不提交；**冻结协议与元数据**（如 `frozen/*.jsonl`、`psychiatry/eval/`、`SOURCES.md`）随仓库分发。`runs/` 内问答对、账本 jsonl 与报告可提交，LoRA 权重与 `_cache` 仍忽略。目录 lineage 见 [runs/README.md](../runs/README.md)。提供历史报告并不意味着 adapter 权重已随仓库分发。套件内相对的 `input / recipes_dir / heldout / refusal` 路径按仓库根目录解析，不按套件文件所在目录解析。

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

`--mode` 必填，可取 `formal` 或 `exploratory`。默认评分器为 `aux-rules-v2`，默认最大新 token 数为 512。`aux-rules-v1`、`aux-rules-v2`、`aux-rules-v3` 都是辅助规则分。规则通过不能改称为语义正确率，`formal_main_metric` 在没有人工盲评主指标时保持 `not_executed`。人工盲评的任务完成和行为适当单独汇总。

探索模式不要求来源清单，也不能把探索结果改称为正式结果。见过来源上的问法诊断、开发集上的证据条件，都留在探索或诊断报告里。正式测试要求协议来源不在冻结训练清单中，并且来源可追溯。

正式模式必须同时给出两份文件：

```sh
qa-pipeline eval-adapter --base-model /path/to/base-model --base-id base-v1 --adapter /path/to/adapter --adapter-id adapter-v1 --protocol /path/to/protocol.jsonl --out runs/local_formal_eval --mode formal --frozen-train-families /path/to/frozen_train_families.json --source-inventory /path/to/source_universe.json --device 0
```

`--frozen-train-families` 是 JSON 对象，至少包含非空的 `source_family_ids`，以及 `records`。每条 record 含 `source_family_id`，并含 `stem` 或 `path`。`--source-inventory` 是来源宇宙，可以是带 `source_family_id` 的 JSON 或 JSONL。只有 stem、没有家族 id 的 inventory 不能证明协议来源可追溯。两份文件的字节 SHA-256 写入 `eval_report.json` 的 `frozen_train_families_sha256` 与 `source_universe_sha256`。

正式模式在下列情况整批停止，不加载模型，也不缩小 `planned_n`：

- 未提供 `--frozen-train-families`：`source_list_missing`。
- 清单存在但 `source_family_ids` 为空：`invalid_empty_source_list`。空集合不能绕过重叠检查。
- 文件缺失、JSON 无效、只有裸 id、或清单中的家族在宇宙里找不到：`invalid_source_list`。
- 协议里的 `source_family_id` 不在来源宇宙中：`source_untraceable`。
- 协议来源出现在训练清单中：`source_overlap`。

### 协议与复核

协议是 JSONL，每行一个用例。探索模式至少需唯一的 `case_id`（或 `id`）及非空问题（或用户消息）。正式模式还检查：

- 参考答案或答案要点、来源和 `expected_action`；
- 已复核状态、复核人员与意见；双人一致状态还要求双人记录与裁决信息；
- 与当前内容一致的 `review_content_hash`；
- 与冻结训练来源集合的隔离，以及来源宇宙中的可追溯性。

字段和条件以 `validate_protocol`、`review_content_hash` 和测试为准。题干、上下文或 system 变化后，对应预测失效并必须重新复核。只改 gold 或评分器时，可在预测身份匹配的前提下复用预测并写入新评分版本。正式协议有无效项时整批停止，不通过删除题目缩小分母。空白审核或 `pending_review` 保持待审核，不会默认通过。只填写一名审核者时状态为 `agreed`，不能写成 `dual_agreed`。

人工表用 `qa-pipeline import-review --run <运行目录>` 导入。该命令按 `review/batches.json` 校验 ID、内容哈希和审核完整性，并写出 `review/import_result.json`。`runs/drug_v25_eval/review/review_guide.md` 是这一轮审核包的填写说明，包含不含真实药品内容的示例。开发集用来构造新条件；诊断集描述见过来源或固定行为边界；正式测试仍是独立来源上的冻结协议。三者的分母和结论不能合并。

`teacher-review` 把教师裁定写到新的运行目录，不回写人工 CSV。`review_source=teacher` 的记录不能通过正式模式的人工门禁；正式主指标仍是 `not_executed`。教师接受只允许配置范围内的探索推理，并且技术失败、预算停止、弃权和分歧都不会被当成内容通过。历史报告里的规则分和文件存在性都不是这次教师审核的结论。

### 产物、缓存与结果解释

查看 `protocol_check.json`、`eval_report.json` 和适用的预测、评分产物。协议失败或未执行时不一定存在预测文件。来源清单无效时报告仍保留计划分母，并记录清单哈希。

预测缓存绑定实际消息、基座、adapter、模板与推理配置；评分版本和 gold 改变时可用 `rescore_saved` 复用匹配的已存预测。它是 Python 接口，不是当前 CLI 子命令。保存新版本应保留旧评分和历史身份信息。

`eval-adapter` 成功执行返回 0，捕获的协议错误、来源清单错误或缺少 tokenizer 配置返回 1，返回未执行报告时返回 2。`import-review` 在校验错误时返回 1。进一步判断应查看报告中的 `executed`、失败项和统计分母，不能只检查输出文件是否存在。

## 闭卷首轮 CB-1

`runs/cb1_20261007` 是 2026-10-07 的单种子探索，不是正式结果。它从历史 74 条 RAG 训练题里保留 22 条脱离原文后仍然明确的问题，训练新的闭卷 adapter；10 个知识单元各 2 个表面改写构成 20 个 CB-paraphrase，另加 20 个冻结保留题。base 与 adapter 的 40 个主评测输入都已生成回答。教师共享额度剩余 23 次调用，整轮审核需要 120 次，所以审核未执行，页面上的未决表示缺少裁判。训练损失下降没有被写成问答改善。细节见该目录的 `result_report.md`。

```sh
qa-pipeline prepare-closed-book \
  --source runs/drug_v22/E3_g0/sft/train.jsonl \
  --retention configs/protocols/cb_retention_v1.jsonl \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --out runs/cb1_YYYYMMDD \
  --ledger runs/batch1_view_20261007/metrics.json
```

训练和预测分开追加 `--train --device <空闲GPU>` 与 `--predict --device <空闲GPU>`。不要覆盖 `runs/cb1_20261007`。

## 如何阅读实验结论

- 接受条数、导出条数、训练条数和主评测条数是不同分母。
- `student_diagnostic` 没有真实学生探测器时记录未执行；不能用教师回答代替学生诊断结论。
- `gap_fill` 可记录缺口规划；不能由策略名称推断已完成缺口回填。
- 回放池缺失时不能声称抗遗忘实验有效。
- 辅助规则分数不自动成为经过校准的主指标；缺失预测、错误预测、未复核题目和探索结果必须明确区分。
- 历史设计稿中的规模、预算账本和效果目标属于设计背景。当前实现的账本包含占位字段，不能视为完整的并发费用预留系统。
- `runs/` 的既有报告和[归档资料](archive/README.md) 保留当时条件，不作为当前版本自动验收凭证。
