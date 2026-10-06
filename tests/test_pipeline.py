"""可拔插 QA 管线单测：统一使用 FakeLLM，不打真实 API。"""

from pathlib import Path

import pytest

from qa_pipeline.config import load_recipe
from qa_pipeline.llm import FakeLLM
from qa_pipeline.pipeline import Pipeline, load_documents
from qa_pipeline.registry import list_strategies
from qa_pipeline.textutil import is_substring, rouge_l, token_f1

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "fixtures" / "sample_manual.md"
SMOKE = ROOT / "configs" / "recipes" / "smoke.yaml"
BASELINE = ROOT / "configs" / "recipes" / "baseline_platform.yaml"


def test_strategy_catalog():
    catalog = list_strategies()
    assert "heading_window" in catalog["chunking"]
    assert "anchor_reverse" in catalog["question_gen"]
    assert "tag_evol" in catalog["evolution"]
    assert "ner_tfidf" in catalog["anchor"]
    assert "nli_fact" in catalog["filter"]
    assert "roundtrip" in catalog["filter"]
    assert "selfcheck" in catalog["filter"]
    assert "difficulty_sample" in catalog["question_filter"]
    assert "sab" in catalog["grading"]
    assert "grade_by_qtype" in catalog["teacher_router"]
    assert "routed_teacher" in catalog["distillation"]


def test_heading_chunking():
    from qa_pipeline.plugins.chunking import HeadingWindowChunking

    docs = load_documents(MANUAL)
    chunks = HeadingWindowChunking(max_tokens=400, min_tokens=40).run(docs, None)
    assert len(chunks) >= 3
    assert any("过温告警" in c.text for c in chunks)
    assert all(c.title_path for c in chunks)


def test_fixed_overlap_matches_window():
    from qa_pipeline.plugins.chunking import FixedOverlapChunking

    docs = load_documents(MANUAL)
    chunks = FixedOverlapChunking(split=400, overlap=40).run(docs, None)
    assert len(chunks) >= 2


def test_tfidf_anchors():
    from qa_pipeline.pipeline import PipelineContext
    from qa_pipeline.plugins.anchors import TfidfKeywordAnchor
    from qa_pipeline.plugins.chunking import HeadingWindowChunking
    import random

    recipe = load_recipe(SMOKE)
    docs = load_documents(MANUAL)
    chunks = HeadingWindowChunking(max_tokens=500, min_tokens=40).run(docs, None)
    ctx = PipelineContext(recipe=recipe, llm=FakeLLM(), rng=random.Random(0))
    out = TfidfKeywordAnchor(top_k=3).run(chunks[:2], ctx)
    assert out[0].metadata["anchors"]


def test_rouge_and_f1():
    assert rouge_l("过温告警如何处理", "过温告警应如何处理") > 0.7
    assert token_f1("额定冷却能力 9.0 kW", "额定冷却能力为 9.0 kW") > 0.5


def test_smoke_pipeline_keeps_grounded_pairs():
    recipe = load_recipe(SMOKE)
    result = Pipeline(recipe, llm=FakeLLM()).run(MANUAL)
    assert result.chunks
    assert result.questions
    assert result.pairs, "smoke 配方在 FakeLLM 下应保留至少一条"
    for p in result.pairs:
        assert is_substring(p.evidence_span, p.chunk_text)
        assert p.answer


def test_baseline_and_recommended_differ():
    rec = load_recipe(ROOT / "configs/recipes/recommended.yaml")
    rec.max_chunks = 3
    rec.max_samples = 8
    rec.filters = [f for f in rec.filters if f.name in {"rule_clean", "minhash_dedup", "diversity_sample"}]
    base = load_recipe(BASELINE)
    base.max_chunks = 3
    base.max_samples = 8
    r1 = Pipeline(base, llm=FakeLLM()).run(MANUAL)
    r2 = Pipeline(rec, llm=FakeLLM()).run(MANUAL)
    assert r1.recipe_name != r2.recipe_name
    assert r1.stats.produced["chunks"] >= 1
    assert r2.questions


def test_zhixun_export_evidence():
    from qa_pipeline.adapters.zhixun import to_zhixun_row
    from qa_pipeline.schemas import QAPair

    pair = QAPair(
        question="阈值是多少？",
        answer="65℃",
        evidence_span="过温保护阈值设置为 65℃",
        chunk_text="出厂默认将过温保护阈值设置为 65℃，用户可调整。",
        source_doc="手册",
        q_type="factual",
        grade="A",
        goal="rag_grounded",
        evidence_state="sufficient",
        expected_action="answer",
    )
    row = to_zhixun_row(pair)
    assert row["messages"][1]["role"] == "user"
    assert "资料" in row["messages"][1]["content"]
    assert "阈值是多少？" in row["messages"][1]["content"]
    assert row["messages"][2]["content"] == "65℃"
    assert row["metadata"]["goal"] == "rag_grounded"
    assert row["metadata"]["evidence_state"] == "sufficient"
    assert "65℃" in row["metadata"]["evidence"]
    compact_ev = "".join(row["metadata"]["evidence"].split())
    compact_chunk = "".join(pair.chunk_text.split())
    assert compact_ev in compact_chunk

    hidden = pair.model_copy(deep=True)
    hidden.evidence_state = "missing"
    hidden.expected_action = "state_insufficient"
    hidden.metadata["student_context"] = ""
    hidden.answer = "当前资料不足，无法根据给定材料回答。"
    withheld = to_zhixun_row(hidden)
    assert "过温保护阈值" not in withheld["messages"][1]["content"]
    assert withheld["metadata"]["expected_action"] == "state_insufficient"


def test_metrics_retention():
    from qa_pipeline.experiments.metrics import compute_metrics
    from qa_pipeline.schemas import QAPair, UsageStats
    from qa_pipeline.pipeline import PipelineResult

    kept = [
        QAPair(question="q1", answer="a" * 20, chunk_text="a" * 20, evidence_span="a" * 20, nli_score=0.9, judge_overall=4.0, kb_gain=0.6, q_type="factual", grade="A"),
        QAPair(question="q2", answer="b" * 20, chunk_text="b" * 20, evidence_span="b" * 20, nli_score=0.7, judge_overall=3.2, kb_gain=0.1, q_type="reasoning", grade="B"),
    ]
    result = PipelineResult(
        recipe_name="t",
        documents=[],
        chunks=[],
        questions=[],
        pairs=kept,
        rejected=[],
        stats=UsageStats(produced={"questions": 4, "distilled": 3}),
    )
    m = compute_metrics(result)
    assert m["kept"] == 2
    assert m["nli_mean"] == pytest.approx(0.8)
    assert 0 < m["type_entropy"]


def test_free_gpu_parse_and_waves():
    from qa_pipeline.experiments.devices import parse_gpu_table
    from qa_pipeline.experiments.parallel import split_waves
    from qa_pipeline.experiments.runner import load_suite

    assert parse_gpu_table("0, 16561\n3, 18\n4, 18 MiB\n7, 47099\n") == [3, 4]
    from qa_pipeline.experiments.devices import parse_trainable_gpus

    assert parse_trainable_gpus("3, 18, 4000\n4, 18, 22000\n") == [4]
    suite = load_suite(ROOT / "configs/experiments/suite.yaml")
    independent, dependent, light, refusal = split_waves(suite["experiments"])
    assert {item["id"] for item in independent} >= {"E1_baseline", "E1_direct", "E1_ku", "E2_gap"}
    assert {item["id"] for item in dependent} == {"E3_diagnostic", "E4_multi"}
    assert "E8_scale" in {item["id"] for item in light}
    assert "E7_refusal" in {item["id"] for item in refusal}
    assert "E1_baseline" not in {item["id"] for item in dependent}
    drug = load_suite(ROOT / "configs/experiments/suite_drug.yaml")
    _, _, drug_light, _ = split_waves(drug["experiments"])
    assert "E6_conflict" in {item["id"] for item in drug_light}


def test_suite_fake(tmp_path):
    from qa_pipeline.experiments.runner import run_suite

    suite = ROOT / "configs/experiments/suite.yaml"
    payload = run_suite(suite, out_dir=tmp_path, fake=True, skip_sft=True)
    ids = [e["id"] for e in payload["experiments"]]
    assert "E1_baseline" in ids
    assert "E1_direct" in ids
    assert "E1_ku" in ids
    assert "E2_gap" in ids
    assert "E6_cost" in ids
    assert "E6_behavior" in ids
    assert "E7_refusal" in ids
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "E1_baseline" / "zhixun.jsonl").is_file()
    assert (tmp_path / "E5_annotation" / "annotation.jsonl").is_file()
    baseline = next(row for row in payload["experiments"] if row["id"] == "E1_baseline")
    snap = baseline["recipe_snapshot"]
    assert snap["question_gen"]["name"] == "direct_qa"
    assert snap["chunking"]["name"] == "fixed_overlap"
    assert snap["grading"] == "binary"
    assert [item["name"] for item in snap["filters"]] == [
        "rule_clean",
        "exact_hash_dedup",
        "evidence_substring",
        "llm_supported",
    ]
    ours = next(row for row in payload["experiments"] if row["id"] == "E2_gap")
    assert ours["recipe_snapshot"]["question_gen"]["name"] == "direct_grounded"
    assert ours["recipe_snapshot"]["evolution"]["name"] == "none"
    assert "knowledge_ablation" not in [item["name"] for item in ours["recipe_snapshot"]["filters"]]
    assert "roundtrip" not in [item["name"] for item in ours["recipe_snapshot"]["filters"]]
    assert "tag_evol" != ours["recipe_snapshot"]["evolution"]["name"]
    assert payload["llm"] == "fake"
    cost = next(row for row in payload["experiments"] if row["id"] == "E6_cost")
    assert "funnel" in (cost.get("metrics") or {})


def test_recommended_path_and_budget():
    rec = load_recipe(ROOT / "configs/recipes/recommended.yaml")
    names = [item.name for item in rec.filters]
    assert rec.question_gen.name == "direct_grounded"
    assert rec.evolution.name == "none"
    assert rec.grading == "validity_tier"
    assert "knowledge_ablation" not in names
    assert "roundtrip" not in names
    assert "selfcheck" not in names
    result = Pipeline(rec, llm=FakeLLM()).run(MANUAL)
    assert result.pairs
    assert all(pair.grade in {"S", "A"} for pair in result.pairs)
    assert all(pair.action not in {"reject", "quarantine"} for pair in result.pairs)
    from qa_pipeline.adapters.zhixun import to_zhixun_row

    row = to_zhixun_row(result.pairs[0])
    assert "资料" in row["messages"][1]["content"]

    blocked = load_recipe(ROOT / "configs/recipes/e1_direct.yaml")
    blocked.budget_cap_usd = 0
    stopped = Pipeline(blocked, llm=FakeLLM()).run(MANUAL)
    assert stopped.pairs == []
    assert "budget_cap" in stopped.stats.fallbacks


def test_quarantine_not_released():
    recipe = load_recipe(SMOKE)
    recipe.filters = []
    recipe.grading = "validity_tier"
    from qa_pipeline.schemas import QAPair

    sample = QAPair(
        question="阈值是多少？",
        answer="资料里没有写这个数值。",
        evidence_span="这句话不在原文中",
        chunk_text="出厂默认将过温保护阈值设置为 65℃，用户可调整。",
        evidence_state="sufficient",
        action="pass",
    )
    result = Pipeline(recipe, llm=FakeLLM()).refilter([sample])
    assert result.pairs == []
    assert result.rejected
    assert result.rejected[0].grade == "quarantine"


def test_k_sequential_ignores_same_call_samples():
    from qa_pipeline.pipeline import PipelineContext
    from qa_pipeline.plugins.question_gen import KnowledgeUnitQG, KnowledgeUnitJointQG
    from qa_pipeline.schemas import Chunk
    import random

    recipe = load_recipe(ROOT / "configs/recipes/e1_ku.yaml")
    recipe.max_chunks = 1
    recipe.max_samples = 2
    ctx = PipelineContext(recipe=recipe, llm=FakeLLM(), rng=random.Random(0))
    chunk = Chunk(text="过温保护阈值设置为 65℃。用户可调整。", source_doc="手册")
    sequential = KnowledgeUnitQG(per_chunk=1).run([chunk], ctx)
    assert sequential
    assert sequential[0].generation_route == "k_sequential"
    assert sequential[0].metadata["verification_status"] == "semantically_verified"
    joint = KnowledgeUnitJointQG(per_chunk=1).run([chunk], ctx)
    assert joint
    assert joint[0].generation_route == "k_joint"
    assert joint[0].metadata["verification_status"] == "pending"


def test_pending_joint_not_accepted():
    recipe = load_recipe(ROOT / "configs/recipes/e1_direct.yaml")
    recipe.question_gen.name = "k_joint"
    recipe.filters = [item for item in recipe.filters if item.name != "claim_evidence"]
    result = Pipeline(recipe, llm=FakeLLM()).run(MANUAL)
    assert result.pairs == []
    assert any(pair.metadata.get("verification_status") == "pending" for pair in result.rejected)


def test_hidden_evidence_does_not_pass():
    from qa_pipeline.plugins.filters import ClaimEvidence
    from qa_pipeline.pipeline import PipelineContext
    from qa_pipeline.schemas import QAPair
    import random

    recipe = load_recipe(SMOKE)
    pair = QAPair(
        question="阈值是多少？",
        answer="65℃",
        evidence_span="过温保护阈值设置为 65℃",
        chunk_text="过温保护阈值设置为 65℃，用户可调整。",
        evidence_state="sufficient",
        goal="rag_grounded",
        metadata={"student_context": "这里没有阈值。"},
        student_context_refs=["这里没有阈值。"],
    )
    ctx = PipelineContext(recipe=recipe, llm=FakeLLM(), rng=random.Random(0))
    out = ClaimEvidence().run([pair], ctx)
    assert out[0].grade == "quarantine"


def test_sft_skip(tmp_path):
    from qa_pipeline.experiments.sft import evaluate_sft
    from qa_pipeline.schemas import QAPair

    pairs = [QAPair(question="q", answer="a", split="train")]
    report = evaluate_sft(pairs, ROOT / "fixtures/heldout.jsonl", tmp_path, skip_sft=True)
    assert report["skipped"] is True
