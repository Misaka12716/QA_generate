"""对照台 API：用合成实验目录检查套件与样本字段。"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from qa_pipeline.demo.app import create_app

IDS = [
    "E1_baseline",
    "E2_recommended",
    "E3_evol",
    "E4_full",
    "E4_no_nli",
    "E4_no_ablation",
    "E4_no_judge",
    "E5_cost_min",
    "E5_quality_max",
]


def _write_run(tmp_path: Path) -> Path:
    experiments = []
    for exp_id in IDS:
        experiments.append(
            {
                "id": exp_id,
                "purpose": f"目的 {exp_id}",
                "recipe": exp_id,
                "recipe_snapshot": {
                    "name": exp_id,
                    "chunking": {"name": "fixed_overlap"},
                    "anchor": {"name": "none"},
                    "question_gen": {"name": "direct_qa"},
                    "evolution": {"name": "none"},
                    "distillation": {"name": "concise_response"},
                    "teacher_router": {"name": "single"},
                    "filters": [{"name": "rule_clean"}],
                    "grading": "binary",
                },
                "metrics": {
                    "kept": 1,
                    "retention": 0.5,
                    "nli_mean": 0.8,
                    "rejected_by_filter": {"rule_clean": 0},
                },
                "sft": {},
            }
        )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {"suite": "ablation_suite", "input": "fixtures/sample_manual.md", "llm": "fake", "experiments": experiments},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    kept = {
        "qa_id": "qa_1",
        "question": "阈值是多少？",
        "answer": "答" * 600,
        "evidence_span": "过温保护阈值设置为 65℃",
        "chunk_text": "段" * 600,
        "grade": "A",
        "q_type": "factual",
        "nli_score": 0.91,
        "judge_overall": 4.2,
        "kb_gain": 0.7,
        "action": "pass",
        "filter_trace": {"rule_clean": {"action": "pass"}},
        "evolution_type": None,
    }
    rejected = {
        "qa_id": "qa_drop",
        "question": "无关问题",
        "answer": "编造",
        "evidence_span": "",
        "grade": "reject",
        "q_type": "reasoning",
        "action": "reject",
        "filter_trace": {"nli_fact": {"action": "reject"}},
    }
    exp = tmp_path / "E1_baseline"
    exp.mkdir()
    (exp / "qa.kept.jsonl").write_text(json.dumps(kept, ensure_ascii=False) + "\n", encoding="utf-8")
    (exp / "qa.rejected.jsonl").write_text(json.dumps(rejected, ensure_ascii=False) + "\n", encoding="utf-8")
    return tmp_path


def test_suite_and_samples(tmp_path: Path):
    client = TestClient(create_app(_write_run(tmp_path)))
    suite = client.get("/api/suite")
    assert suite.status_code == 200
    body = suite.json()
    assert body["llm"] == "fake"
    assert [row["id"] for row in body["experiments"]] == IDS
    snap = body["experiments"][0]["recipe_snapshot"]
    assert snap["question_gen"]["name"] == "direct_qa"
    assert snap["grading"] == "binary"

    page = client.get("/api/experiments/E1_baseline", params={"status": "kept", "clip": 80})
    assert page.status_code == 200
    sample = page.json()["samples"][0]
    for key in ("qa_id", "question", "answer", "evidence_span", "grade", "q_type", "filter_trace", "nli_score"):
        assert key in sample
    assert sample["question"] == "阈值是多少？"
    assert sample["answer"].endswith("…")
    assert "answer" in sample["clipped"]
    assert len(sample["evidence_span"]) < 80

    dropped = client.get("/api/experiments/E1_baseline", params={"status": "rejected"})
    assert dropped.json()["samples"][0]["qa_id"] == "qa_drop"
    assert client.get("/api/experiments/not-a-run").status_code == 404
    assert client.get("/api/experiments/bad id").status_code == 400
    home = client.get("/")
    assert home.status_code == 200
    assert "方案对照台账" in home.text
    assert "同源文本对照" in home.text


def test_compare_aligns_same_chunk(tmp_path: Path):
    root = _write_run(tmp_path)
    chunk = "过温保护阈值设置为 65℃，恢复后可继续运行。"
    for exp_id, question in (("E2_a1", "阈值是多少？"), ("E2_a2", "保护如何恢复？")):
        exp = root / exp_id
        exp.mkdir(exist_ok=True)
        row = {
            "question": question,
            "answer": "65℃",
            "evidence_span": "过温保护阈值设置为 65℃",
            "chunk_text": chunk,
            "grade": "A",
            "q_type": "factual",
            "filter_trace": {"rule_clean": {"action": "pass"}},
        }
        (exp / "qa.kept.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    client = TestClient(create_app(root))
    page = client.get("/api/compare", params={"group": "e2", "status": "kept"})
    assert page.status_code == 200
    body = page.json()
    assert body["align"] == "chunk"
    assert body["total"] == 1
    assert body["rows"][0]["cells"]["E2_a1"][0]["question"] == "阈值是多少？"
    assert body["rows"][0]["cells"]["E2_a2"][0]["question"] == "保护如何恢复？"
    assert client.get("/api/compare", params={"group": "e9"}).status_code == 422


def test_compare_question_and_overlap(tmp_path: Path):
    root = _write_run(tmp_path)
    shared_q = "阈值是多少？"
    for exp_id, answer in (("E3_b1", "便宜答案"), ("E3_b2", "强教师答案")):
        exp = root / exp_id
        exp.mkdir()
        (exp / "qa.kept.jsonl").write_text(
            json.dumps({"question": shared_q, "answer": answer, "chunk_text": "材料甲"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    client = TestClient(create_app(root))
    body = client.get("/api/compare", params={"group": "e3"}).json()
    assert body["align"] == "question"
    assert body["total"] == 1
    assert body["rows"][0]["cells"]["E3_b1"][0]["answer"] == "便宜答案"
    assert body["rows"][0]["cells"]["E3_b2"][0]["answer"] == "强教师答案"

    evidence = "过温保护阈值设置为 65℃"
    (root / "E1_ours_full").mkdir()
    (root / "E1_ours_full" / "qa.kept.jsonl").write_text(
        json.dumps(
            {"question": "推荐问法", "answer": "65℃", "evidence_span": evidence, "chunk_text": "标题窗 " + evidence},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    overlap = client.get("/api/compare", params={"group": "e1"}).json()
    assert overlap["align"] == "overlap"
    matched = [row for row in overlap["rows"] if row["cells"]["E1_baseline"] and row["cells"]["E1_ours_full"]]
    assert matched
