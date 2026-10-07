# 审核填写说明

先看 `review/examples/fill_example.md`。示例是虚构的“示例物品A”，不要把示例行抄进正式表。

## 先审哪一批

1. 打开 `review/batch1_behavior.md`，填写 `review/batch1_behavior.csv`。这是 v2.3 的 10 道行为题和 v2.4 的 4 道新输入。
2. 同一批的回答盲评：打开 `review/batch1_behavior_blind.md`，填写 `review/batch1_behavior_blind.csv`。不要打开 `review/batch1_behavior_identity_map.json`。
3. 然后打开 `review/batch_a_gold.md`，填写 `review/batch_a_gold.csv`。每道核心题有原题、改写一、改写二，都要看完整输入。
4. 最后打开 `review/batch_b_gold.md`，填写 `review/batch_b_gold.csv`。48 个条件各自审核；若实际构造少于 48，只审已经列出的条件，不要自行补题。

历史的 174 条金标和 348 份盲评留在 `runs/drug_v24_review/`，本轮不重导。

## 金标列

不要改 `batch_id`、`case_id`、`condition_id`、`core_question_id`、`content_hash`。这些用来确认你审的是这份输入。

- `question_clear`：题干是否清楚，填 yes 或 no。
- `semantic_preserved`：改写是否仍问同一件事并保持答案要求。原题填 na，改写填 yes 或 no。
- `visible_evidence_adequate`：完整可见输入是否够用，填 yes、no 或 partial。
- `required_points_ok`：必答要点是否合适，填 yes 或 no。
- `unavailable_points_ok`：不可回答的部分是否合适，填 yes、no 或 na。
- `expected_behavior_ok`：预期行为是否合适，填 yes 或 no。有充分证据时不能把拒答当成正确。
- `answerable_part`：部分支持时，写下可以回答的部分。
- `insufficient_part`：部分支持或证据不足时，写下应说明不足的部分。
- `reviewer_a`、`opinion_a`：第一位审核者的真实姓名和意见。
- `reviewer_b`、`opinion_b`：第二位审核者。只填了一人时，结果是单人 `agreed`，不会变成双人一致。
- `adjudication`：accept、revise 或 reject。留空则保持 pending_review，不会默认通过。
- `ai_note`：程序提示。导入时忽略，不能充当姓名、意见或裁定。

## 盲评列

`task_completed`、`behavior_appropriate` 填 yes 或 no。`key_factual_errors` 和 `unsupported_content` 写具体问题；没有就写“无”。审核者、意见和裁定的规则与金标相同。

## 填完后

在仓库根目录执行：

```sh
qa-pipeline import-review --run runs/drug_v25_eval
```

查看 `review/import_result.json`。只有该批 `ready_for_inference` 为 true，才继续对该批做有上限推理。revise 或 reject 的题目会留在分母中并阻塞该批，不会被删掉。
