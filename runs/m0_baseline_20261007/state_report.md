# 状态盘点

- git_commit: `cf06d453fe2314f1931889335849f1b2f09d124f`
- git_dirty: `True`
- reproducible_baseline: `None`
- 脏工作区不能当作可复现实验基线

阶段状态只描述本次盘点能够证明的范围。`reported_executed` 来自历史报告，`artifact_verified` 只表示文件通过了存在性、哈希或行数检查，`reproduced` 只用于本次 FakeLLM 命令。

## 阶段

- `pipeline`: implemented。只表示源码文件存在，不表示真实实验已执行
- `import_review`: implemented。只表示源码文件存在，不表示真实实验已执行
- `eval_adapter`: implemented。只表示源码文件存在，不表示真实实验已执行
- `v25_prepare`: implemented。只表示源码文件存在，不表示真实实验已执行
- `teacher_review`: implemented。只表示源码文件存在，不表示真实实验已执行
- `reviewed_eval`: implemented。只表示源码文件存在，不表示真实实验已执行
- `demo`: implemented。只表示源码文件存在，不表示真实实验已执行
- `fake_smoke`: reproduced。FakeLLM 流程复现，不是教师或学生模型效果
- `fake_suite`: reproduced。FakeLLM 套件复现，不是真实训练或评测
- `v22_g0_training`: artifact_verified；报告记载 reported_executed。历史单 seed 训练产物。run_meta 记载当时工作区为脏状态。本环境没有重新训练，不能标 reproduced
- `v23_predictions`: artifact_verified；报告记载 reported_executed。本地预测文件可核对字节和行数。没有在本环境重跑推理
- `v24_rescore`: artifact_verified；报告记载 reported_executed。v2.4 重评分与补推理文件在 drug_v24_rescore。人工裁定不因这些文件存在而完成
- `v25_materials`: artifact_verified；报告记载 reported_executed。协议和审核包已准备。ready_for_inference 仍取决于人工导入结果，不能把文件存在当成已审核
- `human_gold`: blocked；报告记载 reported_executed；原因 adjudicated_zero_or_unverified。人工金标未完成。教师审核不能改写为 agreed 或 dual_agreed
- `formal_eval`: blocked；原因 frozen_source_list_missing。正式评测仍要求人工金标、非空冻结来源清单和来源宇宙。空协议不能当成零样本成功
- `teacher_review_execution`: not_started。代码可实现后，真实教师调用仍要凭据和预算。dry-run 不是审核结论
- `reviewed_inference`: not_started。A/B 执行器未获生成预算前不加载学生模型
- `new_g0_training`: not_started。没有诊断结论和训练预算，不启动新的 G0
- `g1_g3`: not_started。缺少真实学生探测器、有效选样和冻结回放池，保持未执行

## 本次命令

- `pytest_before_code_changes`: {"passed": 92, "failed": 0, "skipped": 0, "command": "python -X utf8 -m pytest -q -ra", "note": "改动前的基线。历史文件存在，test_v24_historical 没有跳过。"}
- `pytest_after_code_changes`: {"passed": 109, "failed": 0, "skipped": 0, "command": "python -X utf8 -m pytest -q -ra", "note": "包含新增审核、盘点和 API 测试。0 skipped 不是把跳过当成通过。"}
- `list_strategies`: {"out": "runs/m0_baseline_20261007/list_strategies.json", "fake": false}
- `smoke`: {"out": "runs/m0_baseline_20261007/smoke", "kept": 2, "rejected": 0, "fake": true, "note": "FakeLLM，不是教师效果"}
- `suite`: {"out": "runs/m0_baseline_20261007/suite", "n": 14, "exit_code": 0, "fake": true, "note": "FakeLLM 套件，不是训练或评测效果"}
- `batch1_teacher_dry_run`: {"out": "runs/m0_baseline_20261007/batch1_teacher_dry_run", "status": "blocked", "reason": "teacher_credentials_missing", "planned_subjects": 14, "planned_calls": 28, "model_called": false, "estimated_cost": null, "currency": "unknown"}
- `batch1_eval`: {"out": "runs/m0_baseline_20261007/batch1_eval_blocked", "status": "blocked", "reason": "pending_review", "planned_n": 14, "executed": false, "model_loaded": false}
- `smoke_events`: {"out": "runs/m0_baseline_20261007/smoke_events", "fake": true, "events": true, "note": "新运行写出 events.jsonl；历史运行仍是 historical_partial"}

## 阻塞资产

- `frozen_heldout_protocol` empty_file，record_count=None，路径 `data/campus_hospital_drug_instructions/frozen/heldout_protocol.jsonl`
- `frozen_train_families` missing，record_count=None，路径 `data/campus_hospital_drug_instructions/frozen/frozen_train_families.json`
- `source_universe` missing，record_count=None，路径 `data/campus_hospital_drug_instructions/frozen/source_universe.json`
- `sft_default_base` missing，record_count=None，路径 `/data/pjw/data/models/Qwen2.5-7B-Instruct`

## 观测

- `runs/drug_v22`: historical_partial，缺失 ['events.jsonl']，duration_sec=None
- `runs/drug_v23_eval`: historical_partial，缺失 ['events.jsonl']，duration_sec=None
- `runs/drug_v24_audit`: historical_partial，缺失 ['events.jsonl']，duration_sec=None
- `runs/drug_v25_eval`: historical_partial，缺失 ['events.jsonl']，duration_sec=None

价格未知记为 unknown，不把历史内网教师的 0 美元估价写成已核对费用。
