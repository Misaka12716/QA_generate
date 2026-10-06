"""实验 A/B 候选保持输入边界，审核未完成时不推理。"""

from __future__ import annotations

from qa_pipeline.experiments.drug_corpus import split_for_family
from qa_pipeline.experiments.v25_prepare import (
    PARAPHRASES,
    build_behavior_boundary_cases,
    build_fixed_context_cases,
    match_saved_predictions,
    maybe_execute_batch,
    paraphrase_problems,
)


def _dev_families(count: int) -> list[str]:
    found = []
    cursor = 0
    while len(found) < count:
        family = f"src_v25_test_{cursor}"
        if split_for_family(family) == "dev":
            found.append(family)
        cursor += 1
    return found


def _train_row(family: str, question: str) -> dict:
    system = "仅依据提供的资料回答；资料不足时说明缺少的信息。"
    context = "示例物品A为蓝色。"
    return {
        "id": "qa_example",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
            {"role": "assistant", "content": "蓝色。"},
        ],
        "metadata": {"family_id": family, "source": "000_example", "expected_action": "answer", "evidence_state": "sufficient"},
    }


def test_all_paraphrase_drafts_keep_digits_and_names():
    assert len(PARAPHRASES) == 30
    for family, draft in PARAPHRASES.items():
        for label in ("rephrase_1", "rephrase_2"):
            assert paraphrase_problems(draft["original_question"], draft[label]) == [], family


def test_fixed_context_replaces_only_the_question():
    family, draft = next(iter(PARAPHRASES.items()))
    built = build_fixed_context_cases(
        [_train_row(family, draft["original_question"])],
        [family],
        {"000_example": {"source_family_id": "src_train", "content_sha256": "abc", "split": "train"}},
    )
    assert built["blocked"] == []
    assert built["built_n"] == 3
    original = built["cases"][0]
    rewritten = built["cases"][1]
    assert original["messages"][0] == rewritten["messages"][0]
    assert original["context"] == rewritten["context"]
    assert original["question"] != rewritten["question"]
    assert rewritten["question"] in rewritten["messages"][1]["content"]
    assert original["context"] in rewritten["messages"][1]["content"]
    assert "v23" not in rewritten["stratum"]


def test_prediction_reuse_requires_full_identity():
    family, draft = next(iter(PARAPHRASES.items()))
    cases = build_fixed_context_cases([_train_row(family, draft["original_question"])], [family], {})["cases"]
    original = cases[0]
    messages = original["messages"]
    infer = {"max_new_tokens": 512, "do_sample": False, "dtype": "bfloat16"}
    from qa_pipeline.experiments.scoring import prediction_item_signature

    signature = prediction_item_signature(original, "base-id", "adapter-id", "template", infer, "base", model_input=messages)
    saved = {
        "case_id": original["case_id"],
        "model_role": "base",
        "messages": messages,
        "model_id": "base-id",
        "adapter_id": "",
        "template_id": "template",
        "infer_config": infer,
        "signature": signature,
        "text": "蓝色。",
    }
    matched = match_saved_predictions(cases, [saved], base_id="base-id", adapter_id="adapter-id", template_id="template", infer_effective=infer)
    assert matched["reused_n"] == 1
    changed = dict(saved, infer_config={**infer, "max_new_tokens": 32})
    rejected = match_saved_predictions(cases, [changed], base_id="base-id", adapter_id="adapter-id", template_id="template", infer_effective=infer)
    assert rejected["reused_n"] == 0
    assert "infer_mismatch" in rejected["decisions"][0]["reasons"]


def test_behavior_pairs_use_real_text_and_skip_locked_test():
    left, right = _dev_families(2)
    assert left != right
    docs = [
        {
            "stem": "left",
            "split": "dev",
            "source_family_id": left,
            "generic_family_id": "gen_left",
            "text": "【药品名称】\n通用名称：示例甲\n\n【贮藏】\n置于阴凉干燥处。\n\n【性状】\n本品为蓝色块。\n",
        },
        {
            "stem": "right",
            "split": "dev",
            "source_family_id": right,
            "generic_family_id": "gen_right",
            "text": "【药品名称】\n通用名称：示例乙\n\n【成分】\n辅料含有示例淀粉与纯化水。\n",
        },
        {
            "stem": "locked",
            "split": "locked_test",
            "source_family_id": "src_locked",
            "text": "【药品名称】\n通用名称：示例丙\n\n【贮藏】\n密封保存。\n",
        },
    ]
    built = build_behavior_boundary_cases(docs, train_families={left, right}, limit=1, min_families=1, seed=3)
    assert built["locked_test_used"] is False
    assert built["built_n"] == 4
    assert all(not case["case_id"].startswith("v23_behavior") for case in built["cases"])
    by_condition = {case["condition_id"]: case for case in built["cases"]}
    wrong = by_condition["wrong_object"]
    assert wrong["partner_generic_name"] in wrong["context"]
    assert wrong["generic_name"] in wrong["question"]
    assert wrong["generic_name"] != wrong["partner_generic_name"]
    support = by_condition["sufficient"]["answer"]
    assert support not in by_condition["removed_support"]["context"]
    assert support in by_condition["distractor"]["context"]
    assert by_condition["distractor"]["partner_generic_name"] in by_condition["distractor"]["context"]
    assert by_condition["sufficient"]["expected_action"] == "answer"
    assert all(case["source_overlap_train"] for case in built["cases"])
    assert built["planned_n"] == 4


def test_inference_stays_blocked_until_gold_is_accepted():
    calls = {"n": 0}

    def generate(_pending):
        calls["n"] += 1
        return []

    blocked = maybe_execute_batch(ready=False, allow_inference=True, generate_fn=generate)
    assert blocked["status"] == "blocked"
    assert blocked["model_loaded"] is False
    assert calls["n"] == 0
    waiting = maybe_execute_batch(ready=True, allow_inference=False, generate_fn=generate)
    assert waiting["status"] == "not_started"
    assert calls["n"] == 0
