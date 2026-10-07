"""选例规则在看分数之前固定：缺类就记缺失，再用 case_id 补满。"""

from qa_pipeline.experiments.result_report import answer_judgment, compare_pair, select_examples


def _case(case_id: str, labels: list[str]) -> dict:
    return {"case_id": case_id, "labels": labels, "question": case_id}


def test_missing_class_is_not_replaced_by_another_meaning():
    cases = [
        _case("b", ["improve"]),
        _case("a", ["tie"]),
        _case("c", ["rule_disagreement"]),
        _case("d", []),
        _case("e", []),
        _case("f", []),
        _case("g", []),
    ]
    chosen, missing = select_examples(cases)
    assert missing == ["regress", "both_fail"]
    assert [item["selected_as"] for item in chosen] == [
        "improve",
        "rule_disagreement",
        "case_id_fill",
        "case_id_fill",
        "case_id_fill",
        "case_id_fill",
    ]
    assert [item["case_id"] for item in chosen if item["selected_as"] == "case_id_fill"] == ["a", "d", "e", "f"]


def test_unresolved_pair_is_not_an_improvement():
    failed = answer_judgment({"execution_status": "technical_failed", "decision": "abstain", "reason_codes": ["schema_invalid"], "dimensions": {}})
    accepted = answer_judgment(
        {
            "execution_status": "succeeded",
            "decision": "accept",
            "reason_codes": [],
            "dimensions": {"required_points_complete": True, "behavior_appropriate": True},
        }
    )
    assert compare_pair(failed, accepted) == "unresolved"
    assert accepted["rank"] == 3
