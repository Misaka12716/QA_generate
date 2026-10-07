"""选例规则在看分数之前固定：缺类就记缺失，再用 case_id 补满。"""

from qa_pipeline.experiments.result_report import answer_judgment, compare_outcomes, compare_pair, latest_by_subject, select_examples


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
    assert accepted["legacy_rank"] == 3
    assert accepted["task_success"] is None


def test_legacy_tie_is_not_both_correct():
    record = {
        "execution_status": "succeeded",
        "decision": "accept",
        "rubric_version": "rubric-v1",
        "dimensions": {
            "required_points_complete": True,
            "behavior_appropriate": True,
            "factual_consistency": False,
        },
    }
    judgment = answer_judgment(record)
    outcome = compare_outcomes(judgment, judgment)
    assert compare_pair(judgment, judgment) == "tie"
    assert outcome["relation"] == "unavailable"
    assert outcome["relation"] != "both_correct"
    assert outcome["legacy_relation"] == "tie"


def test_same_subject_id_with_two_hashes_is_not_collapsed():
    rows = [
        {"subject_id": "s", "subject_hash": "a", "decision": "accept"},
        {"subject_id": "s", "subject_hash": "b", "decision": "reject"},
    ]
    assert "s" not in latest_by_subject(rows)
    assert latest_by_subject([rows[0], {"subject_id": "s", "subject_hash": "a", "decision": "revise"}])["s"]["decision"] == "revise"
