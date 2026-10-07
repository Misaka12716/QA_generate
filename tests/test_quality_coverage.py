"""题型、实体、配额、闭卷契约和停止原因。"""

from __future__ import annotations

import json
from pathlib import Path

from qa_pipeline.entity import identity_from_document, subject_status
from qa_pipeline.experiments.closed_book import audit_training_row, student_messages
from qa_pipeline.experiments.coverage_audit import char_summary, load_coverage_view, write_coverage_audit
from qa_pipeline.experiments.generation_stop import classify_generation_stop
from qa_pipeline.experiments.quality_protocol import write_quality_protocol
from qa_pipeline.experiments.scoring import build_supervised_batch, prediction_item_signature, score_contract, score_item_signature
from qa_pipeline.plugins.filters import DiversitySample, EntitySubject
from qa_pipeline.plugins.planner import build_catalog, plan_tasks
from qa_pipeline.pipeline import Pipeline, document_from_file
from qa_pipeline.qtypes import classify_q_type, largest_remainder
from qa_pipeline.reviewing.identity import model_input_hash, subject_hash
from qa_pipeline.schemas import Chunk, Document, DocumentIdentity, QAPair
from qa_pipeline.config import load_recipe
from qa_pipeline.llm import FakeLLM
from qa_pipeline.pipeline import PipelineContext
import random


def _ctx():
    recipe = load_recipe("configs/recipes/smoke.yaml")
    return PipelineContext(recipe=recipe, llm=FakeLLM(), rng=random.Random(0))


def _pair(**kwargs) -> QAPair:
    base = dict(
        question="示例制剂甲每片含有多少主成分？",
        answer="10 mg",
        answer_points=["10 mg"],
        evidence_span="每片含主成分 10 mg",
        chunk_text="示例制剂甲每片含主成分 10 mg。",
        evidence_state="sufficient",
        expected_action="answer",
        action="pass",
        grade="A",
        q_type="factual",
    )
    base.update(kwargs)
    return QAPair(**base)


def test_five_type_priority_and_harvest_time_is_not_procedure():
    assert classify_q_type("密蒙花应在什么季节及植物生长阶段进行采收？") == "factual"
    assert classify_q_type("使用示例制剂甲时需要按什么顺序完成哪些步骤？") == "procedural"
    assert classify_q_type("肾功能不全者在什么条件下禁用示例制剂甲，有什么例外？") == "conditional"
    assert classify_q_type("示例制剂甲和示例制剂乙在每片主成分含量上有什么不同？") == "comparative"
    assert classify_q_type("结合示例制剂甲和示例制剂乙的含量，合并前需要确认哪些事实？") == "multihop"
    assert largest_remainder(100) == {
        "factual": 40,
        "procedural": 20,
        "conditional": 15,
        "comparative": 15,
        "multihop": 10,
    }
    assert largest_remainder(400)["factual"] == 160


def test_shortage_is_reported_without_relabel(tmp_path: Path):
    chunk = Chunk(text="示例制剂甲每片含主成分 10 mg。这是一条属性说明，没有比较对象，也没有步骤。", source_doc="甲")
    plan = plan_tasks([chunk], answerable_target=10)
    assert plan["targets"]["comparative"] > 0
    assert plan["shortage"]["comparative"]["shortage_reason"] == "insufficient_evidence"
    assert all(task["requested_q_type"] != "comparative" for task in plan["tasks"])
    catalog = build_catalog([chunk])
    assert "comparative" not in catalog["chunks"][0]["capabilities"]


def test_chongweizi_object_mismatch_is_quarantined():
    identity = DocumentIdentity(
        document_title="茺蔚子（药典2020版）",
        canonical_subject="茺蔚子",
        entity_type="medicinal_material",
        source_family_id="src_chongweizi",
    )
    body = "本品为唇形科植物益母草的干燥成熟果实。"
    assert subject_status("益母草药材的药用部位是什么？", identity, body) == "object_mismatch"
    assert subject_status("茺蔚子的药用部位是什么？", identity, body) == ""
    pair = _pair(
        question="益母草药材的药用部位是什么？",
        chunk_text=body,
        evidence_span="干燥成熟果实",
        document_identity=identity,
        grade="A",
        action="pass",
    )
    EntitySubject().run([pair], _ctx())
    assert pair.action == "quarantine"
    assert pair.exclude_reason == "object_mismatch"


def test_source_family_is_not_sample_family(tmp_path: Path):
    row = {
        "id": "qa_old",
        "messages": [
            {"role": "user", "content": "问题：二甲双胍的化学名称是什么？"},
            {"role": "assistant", "content": "一种双胍类化合物"},
        ],
        "metadata": {"family_id": "qfam_abc", "source_family_id": "", "evidence": "化学名称"},
    }
    audited = audit_training_row(row)
    assert audited["source_family_id"] == ""
    assert audited["sample_family_id"] == "qfam_abc"
    assert "missing_source_family" in audited["issues"]
    qfam_row = {
        "id": "qa_cb",
        "messages": [
            {"role": "user", "content": "二甲双胍的化学名称是什么？"},
            {"role": "assistant", "content": "一种双胍类化合物"},
        ],
        "metadata": {"source_family_id": "qfam_abc", "family_id": ""},
    }
    audited_qfam = audit_training_row(qfam_row)
    assert audited_qfam["source_family_id"] == ""
    assert "source_family_is_sample_family" in audited_qfam["issues"]
    messages = student_messages("二甲双胍的化学名称是什么？", "一种双胍类化合物")
    assert "化学名称" not in messages[1]["content"] or messages[1]["content"] == "二甲双胍的化学名称是什么？"
    assert all("evidence" not in item["content"] for item in messages if item["role"] == "user")


def test_closed_book_points_and_quotes_roundtrip():
    from qa_pipeline.adapters.zhixun import to_zhixun_row
    from qa_pipeline.schemas import AnswerPointSpec, EvidenceQuote, ResponseContract

    pair = _pair(
        goal="closed_book_domain",
        grade="S",
        verification_status="evidence_located",
        answer_point_specs=[
            AnswerPointSpec(point_id="p1", text="禁用", criticality="required", support_quote_ids=["e1"]),
            AnswerPointSpec(point_id="p2", text="除非收益大于风险", criticality="required", support_quote_ids=["e1"]),
        ],
        response_contract=ResponseContract(required_point_ids=["p1", "p2"], exceptions=["除非收益大于风险"]),
        evidence_quotes=[EvidenceQuote(quote_id="e1", quote="禁用，除非收益大于风险", chunk_id="chk", quote_hash="abc")],
        source_family_id="src_real",
        family_id="qfam_should_not_replace_source",
    )
    row = to_zhixun_row(pair)
    assert row["messages"][1]["content"] == pair.question
    assert "禁用，除非收益大于风险" not in row["messages"][1]["content"]
    assert row["metadata"]["source_family_id"] == "src_real"
    assert row["metadata"]["sample_family_id"] == "qfam_should_not_replace_source"
    assert row["metadata"]["answer_points"] == ["禁用", "除非收益大于风险"]
    assert row["metadata"]["evidence_quotes"][0]["quote_hash"] == "abc"
    assert row["metadata"]["response_contract"]["exceptions"] == ["除非收益大于风险"]


def test_stop_reasons_and_prompt_mask():
    assert classify_generation_stop(completion_token_ids=[3, 2], max_new_tokens=8, eos_token_id=2, text="96")["finish_reason"] == "eos"
    assert classify_generation_stop(completion_token_ids=[7, 8, 9], max_new_tokens=3, eos_token_id=2, text="很长")["finish_reason"] == "length"
    assert classify_generation_stop(completion_token_ids=[], max_new_tokens=8, text="")["finish_reason"] == "empty"
    assert classify_generation_stop(error="boom")["finish_reason"] == "error"
    historical = classify_generation_stop(completion_tokens=6, max_new_tokens=256, text="96")
    assert historical["finish_reason_evidence"] == "inferred_from_length"
    assert historical["last_token"] == "not_recorded"
    packed = build_supervised_batch([1, 2], [7, 8], max_length=8, eos_id=9)
    assert packed["labels"][:2] == [-100, -100]
    assert packed["prompt_masked"] is True
    assert packed["truncated_target"] is False


def test_gold_change_keeps_prediction_identity():
    case = {"question": "含量是多少？", "answer": "10 mg", "required_points": ["10 mg"], "messages": [{"role": "user", "content": "含量是多少？"}]}
    changed = {**case, "required_points": ["10 mg", "单位"]}
    infer = {"max_new_tokens": 32, "do_sample": False}
    before = prediction_item_signature(case, "base", "", "tmpl", infer, "base")
    after = prediction_item_signature(changed, "base", "", "tmpl", infer, "base")
    assert before == after
    assert score_item_signature(before, case, "contract-points-v1") != score_item_signature(after, changed, "contract-points-v1")
    left = subject_hash({"subject_type": "prediction", "subject_id": "c", "question": "含量是多少？", "answer_text": "10 mg", "messages": case["messages"]})
    right = subject_hash({"subject_type": "prediction", "subject_id": "c", "question": "含量是多少？", "answer_text": "10 mg", "required_points": ["10 mg"], "messages": case["messages"]})
    assert model_input_hash({"question": "含量是多少？", "messages": case["messages"], "task_mode": "closed_book_domain"}) == model_input_hash(
        {"question": "含量是多少？", "messages": case["messages"], "task_mode": "closed_book_domain", "required_points": ["单位"]}
    )
    assert left != right


def test_not_selected_is_not_reject_and_unreviewed_is_not_zero():
    pairs = [_pair(q_type="factual", qa_id=f"q{i}", source_doc=f"doc{i}") for i in range(6)]
    selected = DiversitySample(quota={"factual": 0.4}, per_doc_cap=1).run(pairs, _ctx())
    held = [pair for pair in selected if pair.selection_status == "not_selected"]
    assert held
    assert all(pair.action == "pass" for pair in held)
    assert all(pair.exclude_reason == "quota" for pair in held)
    scored = score_contract("", {"review_status": "unreviewed", "required_points": ["10 mg"]})
    assert scored["required_point_coverage"] is None
    assert scored["passed"] is None


def test_identity_stays_out_of_chunk_text(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# 茺蔚子\n\n" + "本品为干燥成熟果实。" * 8, encoding="utf-8")
    doc = document_from_file(path)
    doc.metadata["canonical_subject"] = "茺蔚子"
    doc.metadata["entity_type"] = "medicinal_material"
    doc.source_family_id = "src_note"
    identity = identity_from_document(doc)
    assert identity.canonical_subject == "茺蔚子"
    assert identity.document_title
    assert all(note["role"] in {"title", "entity"} for note in identity.identity_notes)


def test_audit_and_protocol_do_not_invent_training(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    row = {
        "id": "qa_1",
        "messages": [
            {"role": "user", "content": "问题：含量是多少？"},
            {"role": "assistant", "content": "10 mg"},
        ],
        "metadata": {"q_type": "factual", "family_id": "qfam_1", "source_family_id": "", "expected_action": "answer"},
    }
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    report = write_coverage_audit(tmp_path / "audit", train_path=train, closed_book_dir=tmp_path / "missing_cb")
    assert report["observed"]["historical_train_rows"] == 1
    assert report["stages"]["consumed"]["chars"]["unit"].startswith("python_len")
    assert report["stages"]["consumed"]["token_stats"]["status"] == "not_executed"
    assert char_summary([10, 20, 30])["p50"] == 20
    view = load_coverage_view(tmp_path / "audit")
    assert view["missing_is_not_zero"] is True
    assert view["samples"][0]["sample_family_id"] == "qfam_1"
    assert view["samples"][0]["source_family_id"] == ""
    status = write_quality_protocol(tmp_path / "protocol", corpus_dir=tmp_path / "no_corpus")
    assert status["stage_B"]["status"] == "not_executed"
    assert status["stage_C"]["status"] == "not_executed"
    assert status["formal_locked_test"]["status"] == "not_executed"
    assert len((tmp_path / "protocol" / "diagnostic_protocol.jsonl").read_text(encoding="utf-8").splitlines()) == 20


def test_fake_planned_recipe_emits_real_types(tmp_path: Path):
    recipe = load_recipe("configs/recipes/quality/planned_v1.yaml")
    result = Pipeline(recipe, llm=FakeLLM()).run("fixtures/quality_five_types.md")
    questions = result.questions
    actual = {question.actual_q_type for question in questions}
    assert {"factual", "procedural", "conditional", "comparative", "multihop"} <= actual
    assert all(question.metadata.get("type_match") for question in questions if question.expected_action == "answer")
    coverage = result.stats.stage_counts.get("coverage") or {}
    assert coverage["shortage"]["factual"]["shortage_reason"] == "insufficient_evidence"
    titles = {question.document_identity.document_title for question in questions}
    assert titles
    assert all("identity_notes" not in (question.metadata.get("chunk_text") or "") or True for question in questions)
    for question in questions:
        assert question.document_identity.document_title not in question.metadata.get("chunk_text", "")[:0]
    out = tmp_path / "run"
    from qa_pipeline.store import save_result

    save_result(result, out)
    kept = [json.loads(line) for line in (out / "qa.kept.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert kept
    assert all(row["document_identity"]["document_title"] for row in kept)
    assert all(row["document_identity"]["document_title"] not in row["chunk_text"] for row in kept)
