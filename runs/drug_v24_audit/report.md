# drug_v24 评测可靠性审计

本报告使用与 JSON 相同的分层：seen_rephrase、independent、behavior 分开，不合成泛化总分。
全部新题和原题都还没有真实人工裁定。规则分是探索性辅助结果。

## 已完成

- 修复了行为分类、离线重评分身份核对、缓存分层和正式协议门禁。
- 用已有 200 份新题回答做了 v1/v2/v3 离线对照，没有重跑 drug_v23。
- 生成了空白金标审核包和盲评包。
- 写了来源审计和下一阶段协议，没有启动 A–D。

## 探索性辅助结果

- seen_rephrase：题数 60，来源家族 18，核心问题 30。已判定通过 53，未通过 7，待复核 0。等权家族通过率 0.875 只统计已经判定的题，待复核不进入分子或分母。这不是语义正确率。
- independent：题数 30，来源家族 10，核心问题 30。已判定通过 30，未通过 0，待复核 0。等权家族通过率 1.0 只统计已经判定的题，待复核不进入分子或分母。这不是语义正确率。
- behavior：题数 10，来源家族 7，核心问题 10。已判定通过 6，未通过 4，待复核 0。等权家族通过率 0.5 只统计已经判定的题，待复核不进入分子或分母。这不是语义正确率。

## 待人工审核

- 金标包 174 条，盲评回答 348 份。
- 审核者、意见和裁定均为空白。filled_adjudications = 0。
- 新输入单独成包：金标 4 条，盲评 8 份，不并进上面的 174 和 348。
- 新输入的辅助通过不能当成行为已经修好。v24_behavior_missing_reaction 的 adapter 回答称“未提及该药品是否为孕妇禁用”，但新上下文里有“孕妇禁用”。规则把拒答判成通过，这句话要人工看。

## 技术失败或资源受阻

- 没有把技术失败写成内容错误。

## 本次未授权启动的后续实验

- 固定上下文新问法对照
- 成对行为边界测试
- 新增独立来源难题
- 多 seed / G1–G3 / 行为训练

## 旧辅助分与 v3

- 与历史 v2 重新计算不一致：0。
- 仅因评分规则变化而改变通过与否：4。
- 因可见要点金标结构调整而变化：1。
- 输入变化的旧 100 题：0。新输入如果存在，写在补推理清单里，不混进这 100 题。
- 人工分歧：全部 pending_review，没有已确认分歧。

## 历史保护与输入身份

- 受保护历史文件哈希前后一致：True。
- v23 的 200 份回答输入匹配：True，签名匹配：True，消息不一致 0，签名不一致 0。
- v22 的 148 份 train_seen 回答保存了 messages，但没有逐条签名、模板哈希和生效推理参数。adapter 目录的 tokenizer_config 与基座不同，且当时 git_dirty。本次不把它们写成输入匹配证明。
- 新算出的当前哈希不能证明 v23 基线之前文件没有变化。v23 manifest 没有被重写。

## 行为题

- v23_behavior_insufficient_section：自评通过 True，处置 reuse_predictions_pending_human_review，新输入 无。没有自动发现的结构问题 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_wrong_object：自评通过 True，处置 pending_review，新输入 v24_behavior_wrong_object。学生可见上下文没有另一药品的身份。 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_retained_distractor：自评通过 True，处置 pending_review，新输入 无。干扰材料没有展示另一对象，不能当成干扰条件下的充分证据。 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_replaced_context：自评通过 True，处置 pending_review，新输入 v24_behavior_replaced_context。上下文过短，看不到被替换资料属于哪一个对象。 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_clarify_strength：自评通过 True，处置 pending_review，新输入 无。抽出的数字位于用法用量，不是多个规格。金标前提不能当成已核实。
- v23_behavior_missing_condition：自评通过 True，处置 pending_review，新输入 无。题干使用“该条件”，可见上下文没有给出这个指代。被删原句没有保存在候选里，无法确认是否仍有等效表述。
- v23_behavior_missing_reaction：自评通过 True，处置 pending_review，新输入 v24_behavior_missing_reaction。上下文没有章节标题，学生看不到金标所声称的章节身份。 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_child_dose_absent：自评通过 True，处置 reuse_predictions_pending_human_review，新输入 无。没有自动发现的结构问题 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_contra_not_visible：自评通过 True，处置 pending_review，新输入 v24_behavior_contra_not_visible。上下文没有章节标题，学生看不到金标所声称的章节身份。 备注：题干中的药品名称夹带了抽取残留。这是原输入的一部分，本次不改旧题。
- v23_behavior_partial_usage：自评通过 False，处置 gold_structure_only_reuse_predictions，新输入 无。answer_points 含有学生看不见的要点。评分若把这些要点当必答，会把合法部分回答判错。

## 来源与核心问题

- 独立来源：30 题，10 个来源家族，10 个文件。十家族且十文件：True。
- 空 family_id：15，保持未映射。训练行未能对照 source_index：0。
- rule_clean 丢弃 9 条，原因未知，来源是 id 集合差。
- 前缀筛漏掉的全文近重复：0。审计中见到的最大全文 Jaccard：0.3143。
- 有 evidence 字段 30，其中落在学生可见摘录内 30，摘录外 0。
- source_in_training_subset、eval_partition、frozen_family_split 分开记录。冻结分区没有回写。

## 字段覆盖

- numeric_bindings：适用 89，有字段 0，已执行 0。本批候选没有执行该检查所需的字段，或评分器不会仅因支持该字段就自动运行。
- required_conditions：适用 16，有字段 0，已执行 0。本批候选没有执行该检查所需的字段，或评分器不会仅因支持该字段就自动运行。
- synonym_groups：适用 100，有字段 0，已执行 0。本批候选没有执行该检查所需的字段，或评分器不会仅因支持该字段就自动运行。
- atomic_answer_points：适用 100，有字段 39，已执行 0。本批候选没有执行该检查所需的字段，或评分器不会仅因支持该字段就自动运行。
- visible_evidence_map：适用 100，有字段 30，已执行 10。本次只对 10 道行为题执行了可见证据审计。普通题没有证据映射字段，历史评分器没有执行该检查。

## 补推理

- 状态：executed。使用的 GPU 候选：[3, 4, 6]。
- train_seen：状态 exploratory_executed，计划 74，已评分 74，失败 0，设备 [3]，prompt tokens 25302，completion tokens 5219，耗时 190.821 秒。 prediction_run_id d13a5e20ec4ed907。 基座辅助通过 52，adapter 辅助通过 74。 这不是泛化分，也不进入正式主指标。
- new_behavior_inputs：状态 exploratory_executed，计划 4，已评分 4，失败 0，设备 [3]，prompt tokens 504，completion tokens 354，耗时 9.045 秒。 prediction_run_id 4e07f24b6e37bcf8。 基座辅助通过 0，adapter 辅助通过 4。 这不是泛化分，也不进入正式主指标。
- train_seen 是新的确认推理，不并入 seen_rephrase、independent 或 behavior，也不计算泛化总分。

## 测试

```
command: PYTHONPATH=src python -m pytest tests/test_eval_reliability.py tests/test_adapter_eval.py tests/test_contracts.py tests/test_v24_historical.py tests/test_pipeline.py tests/test_demo.py -q --tb=line
python: /opt/miniconda/bin/python
note: /opt/miniconda/envs/opsd/bin/python has transformers for inference and has no pytest module. This log uses the default interpreter.
git_commit: abbca92d9aaf9eff98ef80849224c529e06b4a88
---
........................................................................ [ 90%]
........                                                                 [100%]
80 passed in 1.99s
exit_code: 0
```
