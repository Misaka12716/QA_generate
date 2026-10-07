# 使用指南

[返回 README](../README.md) · [实验与评测](experiments.md) · [架构与开发](development.md)

## 安装与命令入口

项目采用 `src/` 布局，Python >= 3.10。仅运行管线可执行 `python -m pip install -e .`；开发和浏览对照台使用 `python -m pip install -e ".[dev]"`。

| 可选依赖 | 用途 |
| --- | --- |
| `.[demo]` | FastAPI / Uvicorn 结果对照台 |
| `.[dev]` | pytest，以及对照台依赖 |
| `.[nli]` | Transformers / Torch 本地 NLI |
| `.[embed]` | Sentence Transformers 向量模型 |
| `.[sft]` | Transformers / Torch / PEFT / Accelerate 学生训练与推理 |

依赖以 [pyproject.toml](../pyproject.toml) 为准。重模型依赖按需安装；默认流程检查不需要 GPU。

```sh
qa-pipeline --help
qa-pipeline list-strategies
qa-pipeline audit-state --out runs/local_state
```

`audit-state` 只读取代码里登记的仓库资产和基座路径，并在 `--out` 写出 `state_snapshot.json` 与 `state_report.md`。它不扫描任意磁盘，也不会把缺失或空文件当成空数据集。`git_dirty` 为真时，快照不会把当前工作区标成可复现基线。

如果终端找不到 `qa-pipeline`，可用同一 Python 环境的 `python -m qa_pipeline` 替代。CLI 定义位于 [cli.py](../src/qa_pipeline/cli.py)。

## 教师审核

教师审核与人工 CSV 分开。`import-review` 仍只接受人工表；模型名不能填进 `reviewer_a` / `reviewer_b`，也不会被写成 `agreed` 或 `dual_agreed`。

```sh
qa-pipeline teacher-review \
  --manifest path/to/subjects.jsonl \
  --policy configs/review/teacher_only_v1.json \
  --models path/to/judges.json \
  --out runs/local_teacher_review \
  --max-calls 120 \
  --max-tokens 200000 \
  --concurrency 2 \
  --dry-run
```

没有可用的 API key 时，真实模式会停在 `teacher_credentials_missing`，不会发起调用。裁判文件如果显式写 `authentication: models_list_no_key` 并且自带服务地址，才允许在没有密钥时调用；聊天若返回 401，该次记为 `unauthorized` 并不再重试。`--fake` 只用于流程检查。费用未知时记录为 `unknown`，不要把内置估价 0 当成已核对账单。调用账本按运行目录里已落盘的真实 prompt/completion token 累计，达到 `--max-calls` 或 `--max-tokens` 就停止。

`configs/review/teacher_single_exploratory_v1.json` 允许单模型给出 `teacher_single_accepted`。这只表示探索性自动评估，`ready_for_formal_human_eval` 仍为 false。同一模型再评一次不能变成共识。

结果页和方案对照台分开。`qa-pipeline demo --run <含 cases.jsonl 的目录> --host 0.0.0.0 --port 8775` 后打开 `/results`。列表接口是 `GET /api/v1/result-cases`，详情返回完整上下文。教师分还没有时显示「待自动评估」，不填 0。

`run-reviewed-eval` 读取已有协议和 `review_aggregates.jsonl`。未同时给出 `--allow-inference`、`--authorize-inference`、`--device` 和生成预算时，它不加载学生模型。当前命令本身也不附带权重推理实现；缓存未命中时保持未执行。

```sh
qa-pipeline run-reviewed-eval \
  --protocol path/to/protocol.jsonl \
  --reviews path/to/review_aggregates.jsonl \
  --policy configs/review/teacher_only_v1.json \
  --mode exploratory \
  --out runs/local_reviewed_eval
```

## 输入与单次运行

```sh
qa-pipeline run --recipe configs/recipes/recommended.yaml --input fixtures/sample_manual.md --out runs/local_recommended --fake
```

`--recipe` 和 `--input` 必填；建议始终显式设置 `--out`，避免覆盖已有实验。

- 输入目录递归读取 `.md`、`.txt`、`.json`，排除代码定义的语料元数据文件。
- 文件按 UTF-8 文本读取。`.json` 文件在这里按正文处理，不是通用 JSON 数据集解析器；没有 PDF、Word 或 JSONL 批量文档解析入口。
- 不存在的路径会被当成内联正文字符串，而不是报“文件不存在”。运行前检查输入路径，避免对路径文本生成 QA。

读取行为见 [pipeline.py](../src/qa_pipeline/pipeline.py) 的 `load_documents` 和 `document_from_file`。

## 教师配置

真实运行需去掉 `--fake`，并配置可访问的兼容服务。不要把 API key 写入配方或提交到仓库。

| 设置 | 优先顺序 |
| --- | --- |
| API key | `--api-key` → `OPENAI_API_KEY` → `QA_PIPELINE_API_KEY` → 旧 `gpt_api` 文件 → `EMPTY` |
| 服务地址 | `--base-url` → `OPENAI_BASE_URL` → `QA_PIPELINE_BASE_URL` → 内置实验地址 |
| 客户端默认模型 | `--model` → `OPENAI_MODEL` → `QA_PIPELINE_MODEL` → `qwen3.8-27b` |
| 管线阶段实际模型 | 配方 `teacher_models[档位]` → 配方 `teacher_models.default` → 客户端默认模型 |

内置服务地址为 `http://192.168.4.110:4000/v1`，只代表原实验环境。当前客户端不会自动加载 `.env`。旧 `gpt_api` 读取逻辑仅作兼容，具体搜索路径见 [llm.py](../src/qa_pipeline/llm.py)。

**切换模型时，应一并修改配方的模型映射。** `Recipe` 默认将三个档位都设为 `qwen3.8-27b`，因此只设置 `--model` 不足以切换管线模型。复制推荐配方为自己的配置，并添加：

```yaml
teacher_models:
  default: your-model-id
  cheap: your-model-id
  strong: your-model-id
```

以下为 PowerShell 环境变量示例，替换占位值后运行自己的配方：

```powershell
$env:OPENAI_BASE_URL = "https://your-service.example/v1"
$env:OPENAI_API_KEY = "your-api-key"
qa-pipeline run --recipe configs/recipes/local.yaml --input fixtures/sample_manual.md --out runs/local_live
```

`configs/recipes/local.yaml` 需自行从推荐配方复制并修改。Bash 使用 `export OPENAI_BASE_URL=...`、`export OPENAI_API_KEY=...` 设置同名变量。

`QA_PIPELINE_NLI_MODEL` 和 `QA_PIPELINE_EMBED_MODEL` 分别指定可选本地 NLI 和向量模型。不具备对应模型时，相关策略可退回 LLM 或 TF-IDF；查看 `stats.fallbacks`，不要把回退结果当作重模型结果。

## 配方规则

配方是策略组合，不是算法实现。字段定义见 [config.py](../src/qa_pipeline/config.py)，可用策略通过 `list-strategies` 查询，参数以对应 [plugins](../src/qa_pipeline/plugins) 类的构造函数为准。

```yaml
# 参数与 name 同级，不要嵌套 params:
chunking: {name: heading_window, max_tokens: 512, overlap: 0.1}
anchor: {name: none}
question_gen: {name: direct_grounded}
filters:
  - {name: rule_clean, min_answer_tokens: 4}
  - {name: evidence_substring, on_fail: quarantine}
```

上面是格式片段，完整运行请使用仓库内配方。

| 配置项 | 含义与注意事项 |
| --- | --- |
| `goal` | `rag_grounded` 或 `closed_book_domain`，影响最终消息中的学生可见上下文 |
| `chunking / anchor / question_gen` | 必需的切块、锚点和出题策略 |
| `evolution / question_filter` | 出题后的演化与问题筛选，默认 `none` |
| `teacher_router / distillation` | 教师路由与答案蒸馏 |
| `filters` | 按列表顺序执行；顺序会影响后续样本与风险处理 |
| `grading` | `validity_tier / sab / binary / none`；不保证任何配置都能产出可发布样本 |
| `max_chunks / max_samples` | 限制块数和蒸馏前候选问题数；派生行为样本可改变最终条数 |
| `questions_per_chunk` | 由出题策略解释；推荐配方为 0，不能理解成所有策略都不出题 |
| `budget_cap_usd` | 默认 `null`；0 阻止启动有效生成，成本计算使用代码内置估价 |
| `extra.max_llm_calls / extra.max_prompt_tokens` | 客户端调用数和提示 token 限制 |
| `split / seed` | 默认 `train / 42`；固定 seed 不保证真实远程模型完全确定 |

成本表是实现中的估值，不是实时供应商账单；当前默认教师按 0 美元计价。检查调用数和 token，不要只依赖美元预算控制规模。

## 运行产物

`run` 先保存管线结果，再导出智训 JSONL：

| 文件 | 内容 |
| --- | --- |
| `chunks.jsonl` | 文档切块和来源信息 |
| `questions.jsonl` | 路由、蒸馏前的问题快照 |
| `qa.raw.jsonl` | 蒸馏后、过滤前的 QA 快照 |
| `qa.kept.jsonl` | 最终接受的 QA |
| `qa.rejected.jsonl` | 经过过滤链后仍保留在列表、但最终未获接受的 QA |
| `stats.json` | 阶段计数、过滤漏斗、耗时、用量、预算与回退记录 |
| `meta.json` | 配方名称和数量摘要 |
| `zhixun.jsonl` | 带 `id / split / messages / metadata` 的发布数据 |

过滤器可能直接移除样本，因此 `qa.rejected.jsonl` **不是所有被丢弃样本的完整日志**；排查时同时查看 raw、kept 和过滤漏斗。单次 `run` 的 `meta.json` 也不保存完整配方，请自行保留配置和代码版本。

## 导出与发布资格

```sh
qa-pipeline export --run runs/local_smoke --out runs/local_smoke/export.jsonl --split train
```

`--run` 也可传 `qa.kept.jsonl` 路径。`--split` 只改变导出标签，不执行数据集划分。

导出器检查 S/A 等级、拒绝/隔离/待升级状态、待核验状态和最终训练对象哈希。不合格样本会使导出失败，不会静默丢掉后继续发布。问题、答案或可见资料改变后应重新验证，不能仅改等级或重填哈希。

`rag_grounded` 使用学生可见资料构建消息；`closed_book_domain` 不把资料放入用户消息。证据字段属于追溯信息，不代表学生实际看到了它。详见 [zhixun.py](../src/qa_pipeline/adapters/zhixun.py)。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 请求原内网地址或模型未切换 | 环境变量、CLI 参数和 `teacher_models` 的优先顺序 |
| 样本数为零 | 输入路径、候选数、预算停止、过滤漏斗、发布状态 |
| 有 kept 文件但导出失败 | 最终哈希、等级和待核验状态；CLI 已先保存中间结果 |
| 对照台没有数据 | 指向 `experiment` 产物目录，而非普通单次 `run` 目录 |
| 某项策略没有执行 | `stats.fallbacks` 与样本 `filter_trace` 中的原因 |
| 领域实验缺数据 | 参见[实验前置条件](experiments.md#套件与数据前置条件) |
