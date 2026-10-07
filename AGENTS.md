# 仓库协作约定

本文件适用于整个仓库，供在此修改代码、配置、测试和文档的编码代理使用。

## 先了解任务

- 阅读 [README.md](README.md)，再按任务阅读 [使用指南](docs/usage.md)、[架构与开发](docs/development.md) 或 [实验与评测](docs/experiments.md)。
- 以当前源码、配置和测试确认实现。`docs/archive/` 是历史设计与分析，不把其中的计划当作已经实现的功能。
- 修改前检查工作区状态，保留用户未提交改动。限定修改范围，不顺带重构或刷新历史实验。
- 默认用中文解释变更和维护文档；代码标识符与既有项目风格一致。

## 开发环境与常用验证

Python >= 3.10，项目采用 src 布局。从仓库根目录执行：

```sh
python -m pip install -e ".[dev]"
python -X utf8 -m pytest -q
qa-pipeline list-strategies
qa-pipeline run --recipe configs/recipes/smoke.yaml --input fixtures/sample_manual.md --out runs/agent_smoke --fake
```

使用独立虚拟环境；若 `runs/agent_smoke` 已有数据，改用新的输出目录。普通修改使用 FakeLLM 和临时数据验证。真实服务调用、模型下载、GPU 训练不属于默认验证步骤；只有任务需要且已获授权时才执行。

测试按改动选择：管线用 `test_pipeline.py` 与 `test_contracts.py`；评测用 `test_adapter_eval.py` 与 `test_eval_reliability.py`；对照台用 `test_demo.py`。行为变更完成后运行完整测试。纯文档改动核对链接、参数及可复制示例即可，不增加只复述文档的测试。

`test_v24_historical.py` 依赖未随干净 checkout 提供的历史运行文件，可能跳过。报告通过、失败和跳过情况，不把跳过说成验证成功。仓库未配置专门的 lint、formatter 或文档构建命令。

## 代码边界

- CLI 参数放在 `src/qa_pipeline/cli.py`，配方契约放在 `config.py`，数据结构放在 `schemas.py`。
- 算法放在 `plugins/`，通过 `@register(stage, name)` 注册；新增模块还需加入 `plugins/__init__.py` 的加载列表。
- `pipeline.py` 负责编排和阶段状态，不按配方名称硬编码算法。
- YAML 只描述策略和参数。策略参数与 `name` 同级，不嵌套 `params`。
- 持久化与智训导出分别由 `store.py` 和 `adapters/zhixun.py` 管理。改变字段时检查所有消费者。
- 保持可选重依赖延迟加载和回退记录，不能让基础流程无故要求 GPU。

## 不得破坏的数据与评测契约

- 保留来源、证据、学生可见上下文的区别。隐藏证据不等于学生可见支持。
- 不允许待核验、B 级、拒绝、隔离或待升级样本绕过发布检查。
- 最终问题、上下文或答案变化后重新验证；不要删除哈希、改等级或篡改复核状态来“修好”测试。
- 保留来源族、样本族、任务变体与父样本关系，避免改写样本重复计权或训练/测试泄漏。
- 技术失败、空响应、预算停止不是成功，也不是语义上的资料不足；缺少依赖时明确记录回退或 `not_executed`。
- 正式评测使用冻结的协议和训练来源清单，无效协议整批停止，不缩小分母；探索性评测不能改名为正式结果。
- 缓存必须绑定实际输入、模型、模板及推理配置。修改 gold 或评分器时保留预测身份和评分版本。

## 文件与结果保护

- 新运行使用新的 `runs/` 子目录或测试临时目录，不覆盖已提交报告、冻结语料、协议、manifest、训练 JSONL、adapter 和历史预测。
- 尊重 `.gitignore`；不要提交 API key、`gpt_api` 凭据、语料原文（`data/**/raw/` 等）、模型权重（含 `runs/**/sft/lora` 与 `*.safetensors`）或 `runs/**/_cache`。可提交冻结协议、问答 jsonl、评测账本与报告；见 [runs/README.md](runs/README.md)。
- 教师默认地址来自原实验内网。检查真实配置时不打印密钥；切换教师需核对 `teacher_models`，不能假设 `--model` 覆盖它。
- 不为跑通领域套件伪造缺失数据或替换正式 heldout。缺失条件应如实记录。

## 文档与交付

- README 保持简短入口；操作放 `docs/usage.md`，实验放 `docs/experiments.md`，开发说明放 `docs/development.md`。
- 优先更新现有主题页，不新增平行版本说明。历史资料归入 `docs/archive/` 并更新索引；保留历史结论的时间背景。
- 文档命令从仓库根目录可执行，私有路径用明确占位符，不把占位模板称为开箱即用示例。
- 最终说明改了什么、做过哪些检查及结果、哪些验证未执行以及原因。不得把静态核对、fake 流程或未执行的 GPU 任务描述为真实效果验证。
