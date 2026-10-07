# 架构与开发

[返回 README](../README.md) · [使用指南](usage.md) · [实验与评测](experiments.md)

## 代码地图

| 路径 | 职责 |
| --- | --- |
| [cli.py](../src/qa_pipeline/cli.py) | 参数解析与命令入口 |
| [config.py](../src/qa_pipeline/config.py) | YAML 配方与 Pydantic 模型 |
| [schemas.py](../src/qa_pipeline/schemas.py) | Document、Chunk、Question、QAPair、UsageStats |
| [pipeline.py](../src/qa_pipeline/pipeline.py) | 加载文档、执行阶段、统计和结果分流 |
| [registry.py](../src/qa_pipeline/registry.py) / [plugins](../src/qa_pipeline/plugins) | 策略注册、发现与各阶段实现 |
| [llm.py](../src/qa_pipeline/llm.py) / [embeddings.py](../src/qa_pipeline/embeddings.py) | 教师客户端、FakeLLM、可选本地模型和向量后端 |
| [store.py](../src/qa_pipeline/store.py) / [adapters/zhixun.py](../src/qa_pipeline/adapters/zhixun.py) | 结果落盘、最终消息构造和发布检查 |
| [experiments](../src/qa_pipeline/experiments) | 套件编排、SFT、评测、评分及历史审计 |
| [reviewing](../src/qa_pipeline/reviewing) | 教师审核契约、政策、缓存和运行器；人工 CSV 仍由 `review_io` 负责 |
| [demo](../src/qa_pipeline/demo) | 已有实验结果的 API 与静态界面；另有只读 `/api/v1/state`、`/runs`、`/review-batches` |
| [configs](../configs) / [fixtures](../fixtures) / [tests](../tests) | 配方和套件、示例数据、回归测试 |

## 主流程与数据边界

```mermaid
flowchart LR
  D[Document] --> C[Chunk]
  C --> A[锚点与出题]
  A --> Q[Question]
  Q --> E[演化与问题筛选]
  E --> T[教师路由与蒸馏]
  T --> R[raw QA 快照]
  R --> F[顺序过滤与分级]
  F --> K[接受或拒绝]
  K --> X[导出资格检查]
  X --> J[智训 JSONL]
```

`Pipeline.run` 是完整入口。实验复用还使用以下入口：

- `distill_from_questions`：从问题快照重新蒸馏、过滤、分级。
- `refilter`：对已有 QA 重新过滤和分级。
- `select_only`：对冻结的接受池执行选择，保留选择状态及排除原因。

这些入口的阶段和统计口径不同；不要把其输出数量直接当成同一阶段的“保留率”。

`QAPair` 同时承载内容、来源、证据、学生可见资料、验证状态、分级、选择状态与审计信息。扩展数据结构时同步检查存储、导出、训练和评测消费者。

必须保持的约束：

1. 来源证据与学生可见上下文分开；隐藏证据不能证明一个可见资料不足的回答正确。
2. `pending`、隔离、待升级、拒绝及 B 级样本不能进入发布数据。
3. 最终问题、上下文和 assistant 目标受 `validation_subject_hash` 约束；改变内容会使旧验证失效。
4. `source_family_id`、`family_id`、`task_variant_id` 和父样本关系用于隔离、追溯及统计；改写同一来源不能变成独立样本收益。
5. 技术失败、预算停止和解析失败不等于语义上的“资料不足”，也不等于成功执行。

## 添加策略

在对应 `plugins/` 模块实现类，并使用注册装饰器：

```python
from qa_pipeline.registry import register

@register("filter", "example_filter")
class ExampleFilter:
    def __init__(self, **params):
        self.params = params

    def run(self, items, ctx):
        return items
```

这是接口示意，不是质量过滤算法。随后：

1. 若增加新模块，将模块路径加入 [plugins/__init__.py](../src/qa_pipeline/plugins/__init__.py) 的 `_MODULES`；只写装饰器但未导入不会注册。
2. 在独立配方中使用 `{name: example_filter, ...}`，保持参数与 `name` 同级。
3. 在策略内使用 `ctx.recipe / ctx.llm / ctx.stats / ctx.extras`；可选后端缺失时记录真实回退或未执行原因。
4. 为新行为或修复增加针对性的测试，覆盖成功路径和相关失败边界。
5. 运行 `qa-pipeline list-strategies` 与最相关测试，再验证组合流程。

算法放在插件中；编排器负责顺序和状态，配方负责选择与参数。不要在编排器里按配方名称硬编码算法。

## 测试与验证

```sh
python -m pip install -e ".[dev]"
python -X utf8 -m pytest -q
```

| 测试文件 | 关注点 |
| --- | --- |
| `test_pipeline.py` | 策略注册、切块、生成、发布与套件流程 |
| `test_contracts.py` | 内容契约、来源隔离、训练监督、预算和 GPU 门槛 |
| `test_adapter_eval.py` | adapter 独立评测、缓存、协议和历史文件保护 |
| `test_eval_reliability.py` | 评分可靠性、复核失效、固定分母和错误预测 |
| `test_demo.py` | 对照台接口与样本对齐 |
| `test_review_io.py` | 人工审核 CSV 的哈希、机器名和待审核状态 |
| `test_teacher_review.py` | 教师审核的失败类型、双审、预算、缓存和正式门禁 |
| `test_reviewed_eval.py` | 已审核评测编排、预测身份和未授权不加载模型 |
| `test_state_audit.py` | 资产缺失、空文件和路径越界 |
| `test_api_v1.py` | 只读状态、运行列表和审核批次的缺失状态 |
| `test_v24_historical.py` | 依赖历史运行文件的集成检查；干净 checkout 通常跳过 |

常规测试使用 FakeLLM、测试替身和临时目录；无需为文档检查调用真实服务或启动训练。历史文件缺失造成的 skip 需单独报告，不能声称历史实验已验证。

文档改动至少核对：相对链接、CLI 参数、配方字段和可复制的 fake 示例。仓库目前未配置专门的文档构建器、formatter 或 linter，不应虚构相应必跑命令。

## 文档与实验文件维护

- README 只放项目定位、最短可运行例子和导航。
- 操作、配置放在 `docs/usage.md`；实验协议放在 `docs/experiments.md`；代码边界与扩展方式放在本页。
- 代理行为约定放在根目录 [AGENTS.md](../AGENTS.md)，不要复制一份完整使用指南。
- 旧方案和一次性分析放在 `docs/archive/`，并在归档索引标明性质。既有原文保留历史语境；它们不是当前实现契约。
- 新行为优先更新现有主题页；避免再新增按版本命名的“最新方案”“修订说明”作为并行入口。
- 已提交的 `runs/` 报告是历史记录。新验证使用新的输出目录，不能覆盖它们。依照 [.gitignore](../.gitignore) 区分临时产物和有意保留的报告。
