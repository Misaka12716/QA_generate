# drug_v23 评测报告

本报告只根据 `runs/drug_v23_audit` 与 `runs/drug_v23_eval` 的落盘文件汇总。
探索性辅助分标记为：未完成人工审核，仅供探索。它不是正式主指标，也不是已确认的语义正确率。

## A 代码与测试

- 新增 `eval-adapter`：加载已有基座和 adapter，写原始预测，再离线评分。不调用 `train_lora`，不重写 `train.jsonl`。
- 预测缓存包含实际消息、模型与 adapter 标识、tokenizer 配置哈希和完整推理配置。评分另用金标与评分器版本。
- `aux-rules-v1` 仍是原来的 `score_task`。`aux-rules-v2` 只写新字段。
- 冻结加载按 `subset_manifest.json` 读 txt，跳过 `snapshot_manifest.json` 等元数据；路径缺失即失败。
- 测试：`python -m pytest tests/test_adapter_eval.py tests/test_contracts.py tests/test_pipeline.py`。
- 真实推理使用已安装依赖的解释器 `/opt/miniconda/envs/opsd/bin/python`。默认解释器因 `huggingface-hub==1.8.0` 无法导入当前 transformers，没有改它的包版本。

本次探索性运行使用的标识：

- base_id：`Qwen2.5-7B-Instruct@7463bb0ea7831536`
- adapter_id：`4f4bdec7b50dab9e`
- template_id：`5b5d4f65d0acd3b2`
- 当时空闲且显存足够的 GPU：[3, 4, 6]。推理实际使用其中一张，见预测记录里的 device。

```bash
PYTHONPATH=src /opt/miniconda/envs/opsd/bin/python -m qa_pipeline eval-adapter \
  --base-model /data1/pjw/models/Qwen2.5-7B-Instruct \
  --base-id Qwen2.5-7B-Instruct@7463bb0ea7831536 \
  --adapter runs/drug_v22/E3_g0/sft/lora/adapter \
  --adapter-id 4f4bdec7b50dab9e \
  --protocol runs/drug_v23_eval/candidates/dev_candidates.jsonl \
  --out runs/drug_v23_eval/exploratory \
  --mode exploratory \
  --scorer aux-rules-v2 \
  --max-new-tokens 512
```

正式模式 `--mode formal` 会在加载模型前拒绝未审核题。本次没有发布正式协议。

已有能力：独立评测入口、ID 对齐、预测/评分缓存、协议门禁、家族等权聚合、冻结正文加载、v2 辅助规则。
后续待办：多 seed、学生诊断、回放、补题、G1–G3、人工双审后的正式协议。

## B drug_v22 审计

- 基座：`/data1/pjw/models/Qwen2.5-7B-Instruct`。adapter 配置与 `run_meta.json` 一致。另一路径 `/data/pjw/data/models/Qwen2.5-7B-Instruct` 不存在，没有替换。
- adapter：`/data1/pjw/QA_generate-main/runs/drug_v22/E3_g0/sft/lora/adapter`。训练 seed=42，不是三个 seed。
- 历史 GPU 记录 [3, 4, 6] 只是当时调度。
- 训练导出、E1 合格池、consumed_ids、train_seen 的 ID 与顺序一致，问题在 user 消息中，assistant 目标与合格池答案一致。内容差异条数：0。
- E1 `qa.kept.jsonl` 与 G0 `qa.kept.jsonl` 的题干和答案相同，差异字段是 `audit`、`data_stage`、`included_in_this_run`。
- 非空 family_id 的核心问题 59 个；family_id 为空的训练行 15 条。`source_family_id` 在训练记录中为空，来源族写在外置 `source_index.json`。
- 旧辅助四格：通过→通过 52，未通过→通过 22，通过→未通过 0，未通过→未通过 0。离线重算通过标记不一致的 ID 数：0。该数字不进入泛化总分。
- 漏斗：子集 txt 60；落盘 chunk 100，其中来源数 59。stats 没有单独的“已读正文”计数。问题与原始答案各 89。`rule_clean` 89→80，后续过滤器未再丢弃。`qa.rejected.jsonl` 有 6 条隔离样本。原始 ID 中有 9 条既不在合格池也不在 rejected 文件，与 rule_clean 丢弃数一致；这 9 条没有逐条原因文件。
- 教师 token：prompt 59308，completion 21024，合计 80332。每条保留样本 1085.5676 token。`tokens_per_10k_kept` 复算为 10855676。
- G0 教师 token 为 0，因为没有新的教师生成。监督 token 5100 属于训练目标，不在该折算指标里。优化步 57，与 ceil(74/4)*3 一致。`consumed_ids` 是进入 Dataset 的 ID，不是逐批消费日志。
- retention 是 74/89 的样本保留率。evidence_grounded=1.0 只表示证据片段能在上下文中找到。s_ratio=1.0 只表示规则分级全是 S。type_entropy=0.8074 对应 factual 61、procedural 10、conditional 3。
- answerability 为空是因为没有 answerable 标签。nli_mean 与 judge_mean 为空是因为该配方没有跑 NLI 和裁判。delta_f1 为空是因为主测试协议是空文件，主评测没有加载模型。这些空值都不是 0。
- 没有可追溯的人工审核。现有 G0 与基座路径匹配，可以复用做新输入推理；训练记录缺 source_family_id 不影响权重文件本身。

审计后复核历史文件哈希：

- 与审计基线相比，被改动的历史文件数：0。

## C 新测试与审核材料

- 原题审核包：74 题、148 份回答。审核者版本匿名随机排列，映射在 `review/analysis_map.json`。已填写的人工裁定：0。
- 候选 100 条，请求 100。已见知识新问法 60，独立来源 30，行为题 10。状态：unreviewed。正式协议写入：False。
- 核心问题：可用非空 family 59，选取 30。family_id 为空的训练 ID 没有拿来复制凑数。
- 独立来源家族 10，其中未见通用名 10。近重复筛除 0。扫描上限 400，实际扫描开发文件 10。
- 行为题缺失模式：无。
- 开发候选来自 dev 分区或证据改写，没有写入 `heldout_protocol.jsonl`，也没有用 locked_test 调评分器。
- 新题审核表 `review/new_items_for_review.jsonl` 先给题干、证据和候选金标，没有附模型输出。这些金标未经人工接受。

教师账本：

- 模型 qwen3.8-27b，温度 0.2，max_tokens 700，调用上限 48。
- 实际调用 40，重试 0，prompt token 12702，completion token 4026。停止原因：None。

需要人工完成、程序不能代填的审核：

- 74 道原题的训练目标，以及 148 份匿名回答。
- 100 道新题的题干、证据和候选金标。审核通过之前不能发布正式协议。

## D 新评测结果

- 结果类别：exploratory_auxiliary。正式主指标：not_executed。formal_gold=False。
- 声明：未完成人工审核，仅供探索
- planned_n=100，scored_n=100，failed_n=0，cache_hits=4。
- 技术错误条数：0。截断条数：0。推理 prompt token 492636，completion token 9539。设备：[3]。
- 三个层次分开汇总，没有合成泛化总分。行级转移按题目计数。来源等权通过率先在核心问题内平均，再对来源等权；同一核心问题的两条改写不会把该问题算两次。

- behavior：行 10，核心问题 10，来源 7。行级辅助通过 2/10 → 4/10。来源等权辅助通过率 0.1429 → 0.2857。转移 {'fail_fail': 6, 'fail_pass': 2, 'pass_pass': 2}。
- independent：行 30，核心问题 30，来源 10。行级辅助通过 29/30 → 30/30。来源等权辅助通过率 0.9667 → 1.0。转移 {'pass_pass': 29, 'fail_pass': 1}。
- seen_rephrase：行 60，核心问题 30，来源 18。行级辅助通过 29/60 → 53/60。来源等权辅助通过率 0.4861 → 0.875。转移 {'fail_pass': 24, 'pass_pass': 29, 'fail_fail': 7}。

历史 T0 的 148 份原题回答没有重新推理，仍以 drug_v22 的 train_seen.jsonl 为准。

## E 下一步

- 已见知识的新问法有辅助分改善，同时仍有前后都未通过的题。先人工核对这批评分是不是规则误判；确认是真实错答后，再考虑表达覆盖。
- 独立来源题在基座上的辅助通过已经很高，G0 几乎没有新增通过。这些题是教师按证据原文写出的、尚未人工审核的抽取题，不能据此宣称独立泛化。
- 行为题里多数辅助失败没有被 G0 纠正，而且不少预期是说明资料不足或澄清。下一步先核对行为边界和可见证据，不把这一层并进总分。
- 全部新题仍是 unreviewed。审核若改了题干或上下文，对应项要重新推理；只改判定、不改输入时可以复用已有回答。
- 原题 52/22 仍是旧辅助分。若人工审核发现主要是评分误判，先改评分，再决定是否做训练 seed 稳定性实验。
- 学生诊断、有效降权、合格回放池和预算控制都还没有。现在不计划 G1–G3，也不重新调用教师生成训练数据。

本次没有做：重跑 drug_v22、新训 G0、G1–G3、回放、补题、其他生成臂、全库出题、多 seed、提交或部署。
