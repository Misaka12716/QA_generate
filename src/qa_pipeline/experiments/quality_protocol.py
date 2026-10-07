"""冻结质量实验的分组、诊断协议和预算门。资源不足时写 not_executed。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DIAGNOSTIC_ITEMS = [
    {"case_id": "dev_short_01", "bucket": "short_answer", "question": "12乘以8等于多少？", "reference_answer": "96", "required_points": ["96"], "expected_action": "answer"},
    {"case_id": "dev_short_02", "bucket": "short_answer", "question": "一天有多少小时？", "reference_answer": "24", "required_points": ["24"], "expected_action": "answer"},
    {"case_id": "dev_short_03", "bucket": "short_answer", "question": "水的化学式是什么？", "reference_answer": "H2O", "required_points": ["H2O"], "expected_action": "answer"},
    {"case_id": "dev_short_04", "bucket": "short_answer", "question": "三角形内角和是多少度？", "reference_answer": "180度", "required_points": ["180"], "expected_action": "answer"},
    {"case_id": "dev_list_01", "bucket": "list_condition", "question": "列出三种常见标点，并用顿号分隔。", "reference_answer": "逗号、句号、问号", "required_points": ["逗号", "句号", "问号"], "expected_action": "answer"},
    {"case_id": "dev_list_02", "bucket": "list_condition", "question": "如果今天下雨就带伞，否则不带。下雨时应该怎么做？", "reference_answer": "带伞", "required_points": ["带伞"], "exceptions": [], "expected_action": "answer"},
    {"case_id": "dev_list_03", "bucket": "list_condition", "question": "除非体温超过38度，否则不使用退热说明。体温37度时是否使用？", "reference_answer": "不使用", "required_points": ["不使用"], "exceptions": ["体温超过38度"], "expected_action": "answer"},
    {"case_id": "dev_list_04", "bucket": "list_condition", "question": "把“红、蓝”两项都写出来。", "reference_answer": "红、蓝", "required_points": ["红", "蓝"], "expected_action": "answer"},
    {"case_id": "dev_step_01", "bucket": "steps", "question": "泡茶的顺序是什么？请按先烧水、再投茶、最后出汤说明。", "reference_answer": "先烧水，再投茶，最后出汤。", "required_steps": ["烧水", "投茶", "出汤"], "expected_action": "answer"},
    {"case_id": "dev_step_02", "bucket": "steps", "question": "洗手要按什么顺序完成？", "reference_answer": "湿手、涂皂、搓洗、冲洗、擦干。", "required_steps": ["湿手", "涂皂", "搓洗", "冲洗", "擦干"], "expected_action": "answer"},
    {"case_id": "dev_step_03", "bucket": "steps", "question": "比较步行和骑车在速度上的差别，不要推荐其中一种作为医疗建议。", "reference_answer": "骑车通常比步行更快。", "comparison_dimensions": ["速度"], "expected_action": "answer"},
    {"case_id": "dev_step_04", "bucket": "steps", "question": "结合“水会结冰”和“冰比水冷”两件事实，说明冬天路面可能怎样。", "reference_answer": "水结冰后路面可能更冷、更滑。", "required_points": ["结冰", "冷"], "expected_action": "answer"},
    {"case_id": "dev_explain_01", "bucket": "explanation", "question": "用两句话解释为什么影子在晴天会出现。", "reference_answer": "光沿直线传播，物体挡住光线就形成影子。", "required_points": ["光线", "挡住"], "expected_action": "answer"},
    {"case_id": "dev_explain_02", "bucket": "explanation", "question": "解释闰年大约每四年出现一次的原因，不要展开历法史。", "reference_answer": "地球公转一周约365.25天，每四年补一天。", "required_points": ["四年", "一天"], "expected_action": "answer"},
    {"case_id": "dev_explain_03", "bucket": "explanation", "question": "为什么不能把“禁用，除非收益大于风险”写成无条件禁用？", "reference_answer": "因为存在收益大于风险的例外。", "required_points": ["例外"], "exceptions": ["收益大于风险"], "expected_action": "answer"},
    {"case_id": "dev_explain_04", "bucket": "explanation", "question": "资料没给出身高时，能不能回答体重剂量？", "reference_answer": "不能。缺少身高或体重时要说明信息不足。", "required_points": ["信息不足"], "expected_action": "state_insufficient"},
    {"case_id": "dev_style_01", "bucket": "style", "question": "用不超过十个字回答：太阳从哪边升起？", "reference_answer": "东方。", "required_points": ["东方"], "style_instruction": "short", "expected_action": "answer"},
    {"case_id": "dev_style_02", "bucket": "style", "question": "请分点说明如何泡茶，保留顺序。", "reference_answer": "1. 烧水 2. 投茶 3. 出汤", "required_steps": ["烧水", "投茶", "出汤"], "style_instruction": "steps", "expected_action": "answer"},
    {"case_id": "dev_style_03", "bucket": "style", "question": "请只回答一个词：冰的常见形态。", "reference_answer": "固体", "required_points": ["固体"], "style_instruction": "one_word", "expected_action": "answer"},
    {"case_id": "dev_style_04", "bucket": "style", "question": "请用完整句子说明雨天带伞的条件，不要加无关建议。", "reference_answer": "下雨时带伞。", "required_points": ["下雨", "带伞"], "style_instruction": "complete_sentence", "expected_action": "answer"},
]

ARMS = {
    "C0": "同一合格池按自然类型取可回答样本，使用已核验的简洁候选答案。",
    "C1": "与 C0 相同的问题、证据和知识，改用 response_contract 的完整回答。简单题不扩写。",
    "C2": "同一合格池按 40/20/15/15/10 选择，并使用同一完整回答规范。",
}
SHARED_HYPER = {
    "base_model_status": "not_selected_until_local_path_confirmed",
    "lora_rank": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "seed": 42,
    "epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_length": 1024,
    "save_epochs": True,
    "status": "proposed_not_frozen",
    "reason": "独立 dev 预试尚未执行，不能把这组数字写成已冻结超参。",
}


def _read_ledger(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "not_executed", "reason": "ledger_missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    budget = payload.get("budget") if isinstance(payload, dict) else None
    return {
        "status": "snapshot_not_current_balance",
        "path": str(path),
        "budget": budget,
        "note": "历史账本不能当作当前余额，也不能据此扩容。",
    }


def _gpu_status() -> dict[str, Any]:
    from .devices import select_trainable_gpus

    try:
        devices, error = select_trainable_gpus(None)
    except Exception as exc:  # pragma: no cover
        return {"status": "not_executed", "reason": f"gpu_query_failed:{exc}"}
    if error or not devices:
        return {"status": "not_executed", "reason": error or "no_trainable_gpu", "devices": devices or []}
    return {"status": "available", "devices": devices}


def write_quality_protocol(
    out_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    ledger_path: str | Path | None = None,
    corpus_dir: str | Path | None = None,
) -> dict[str, Any]:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    corpus = Path(corpus_dir) if corpus_dir else root / "data" / "campus_hospital_drug_instructions"
    ledger = _read_ledger(Path(ledger_path) if ledger_path else root / "runs" / "batch1_view_20261007" / "metrics.json")
    gpu = _gpu_status()
    formal_inputs = 120
    models = 4
    prediction_reviews = formal_inputs * models
    gold_reviews = formal_inputs
    budget = {
        "gold_review_calls": gold_reviews,
        "prediction_review_calls": prediction_reviews,
        "generation_and_revision": "stage_B_unpriced_until_source_exists",
        "reserve": "review_margin_not_estimated_from_old_120",
        "formal_locked_test": "not_executed",
        "reason": "预测审核 480 次加 gold 最多 120 次。未读取到可覆盖该数量的当前额度。",
        "ledger": ledger,
    }
    diagnostic = []
    for item in DIAGNOSTIC_ITEMS:
        diagnostic.append(
            {
                **item,
                "split": "dev_diagnostic",
                "locked_test": False,
                "task_mode": "closed_book_domain",
                "note": "开发诊断。不能转成锁定测试。",
            }
        )
    (dest / "diagnostic_protocol.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostic),
        encoding="utf-8",
    )
    arms = {
        "arms": ARMS,
        "shared_hyperparameters": SHARED_HYPER,
        "old_cb1": "historical_reference_only",
        "common_knowledge_exposure": "not_executed",
        "reason": "合格池尚未生成，不能冻结三份 100 条清单。",
    }
    (dest / "arms.json").write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "budget.json").write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_ok = corpus.is_dir()
    status = {
        "stage_A_protocol": "frozen_dev_diagnostic",
        "stage_A_prediction": {
            "status": "not_executed",
            "reason": gpu.get("reason") or "prediction_not_started",
            "gpu": gpu,
            "conditions": ["A0 max_new_tokens=256", "A1 max_new_tokens=1024", "A2 completeness prompt max_new_tokens=1024"],
        },
        "stage_B": {
            "status": "not_executed" if not corpus_ok else "not_executed",
            "reason": "missing_source_corpus" if not corpus_ok else "real_generation_not_authorized_in_this_run",
            "corpus": str(corpus),
            "corpus_present": corpus_ok,
            "answerable_target": 100,
            "behavior_probe": 5,
        },
        "stage_C": {"status": "not_executed", "reason": "qualified_pool_missing"},
        "stage_D": {"status": "not_executed", "reason": "waiting_for_stage_c_winner"},
        "formal_locked_test": {"status": "not_executed", "planned_inputs": formal_inputs, "reason": budget["reason"]},
        "thresholds": {
            "complex_complete_gain_vs_c0": 0.10,
            "factual_and_retention_regression_limit": 0.05,
            "note": "小样本只写有或无改善信号，待复验。",
        },
    }
    if gpu.get("status") == "available" and not corpus_ok:
        status["stage_A_prediction"]["reason"] = "diagnostic_protocol_frozen_prediction_not_started"
    (dest / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status
