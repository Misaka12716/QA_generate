"""独立 LoRA SFT 与 held-out 评测，超参对齐智训 training_worker。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..adapters.zhixun import GROUNDED_POLICY, ReleaseRejected, assert_releasable, to_zhixun_row
from ..llm import json_payload
from ..schemas import QAPair
from ..textutil import exact_match, token_f1
from .scoring import AlignmentError, build_supervised_batch, clustered_interval, score_task, transition_label

logger = logging.getLogger(__name__)

DEFAULT_BASE = os.environ.get(
    "QA_PIPELINE_SFT_BASE",
    "/data/pjw/data/models/Qwen2.5-7B-Instruct",
)


def write_sft_jsonl(pairs: list[QAPair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            if pair.included_in_this_run is False or pair.exclude_reason:
                continue
            if pair.grade not in {"S", "A"}:
                continue
            if pair.verification_status == "pending" or pair.metadata.get("verification_status") == "pending":
                continue
            try:
                assert_releasable(pair)
            except ReleaseRejected:
                continue
            if pair.data_stage in {None, "accepted", "selected"}:
                pair.data_stage = "released"
            handle.write(json.dumps(to_zhixun_row(pair, bind=False), ensure_ascii=False) + "\n")
    return path


def load_heldout(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_generations(preds: list[str], golds: list[str], case_ids: list[str] | None = None) -> dict[str, float]:
    if case_ids is not None and (len(preds) != len(case_ids) or len(golds) != len(case_ids)):
        raise AlignmentError(f"ID 对齐失败 preds={len(preds)} golds={len(golds)} ids={len(case_ids)}")
    if len(preds) != len(golds):
        raise AlignmentError(f"预测 {len(preds)} 与金标 {len(golds)} 数量不一致")
    ems, f1s = [], []
    for pred, gold in zip(preds, golds):
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))
    n = len(golds)
    return {
        "em": round(sum(ems) / n, 4) if n else None,
        "f1": round(sum(f1s) / n, 4) if n else None,
        "n": n,
        "reason": None if n else "zero_denominator",
    }


def judge_vs_reference(llm, questions: list, preds: list[str], golds: list[str], model: str | None = None) -> dict[str, Any]:
    scores = []
    failed = 0
    for question, pred, gold in zip(questions, preds, golds):
        response = llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "比较模型回答与参考答案的事实一致性，1-5 分。"
                        "解析失败不要猜测。允许平局。"
                        '输出 JSON：{"score":1}'
                    ),
                },
                {"role": "user", "content": f"问题：{question}\n参考：{gold}\n模型：{pred}"},
            ],
            model=model,
            max_tokens=80,
        )
        data = json_payload(response)
        if getattr(response, "status", "ok") != "ok" or "score" not in data:
            failed += 1
            continue
        scores.append(float(data["score"]))
    mean = round(sum(scores) / len(scores), 4) if scores else None
    return {"mean": mean, "n": len(scores), "failed": failed, "reason": None if scores else "no_valid_judge"}


def train_lora(
    train_jsonl: Path,
    out_dir: Path,
    base_model: str = DEFAULT_BASE,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 16,
    max_length: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover
        logger.warning("SFT extras missing: %s", exc)
        return {"skipped": True, "reason": f"missing_sft_deps: {exc}"}

    rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"skipped": True, "reason": "empty_train"}
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False

    prepared = []
    skipped_rows = []
    for row in rows:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        packed = build_supervised_batch(prompt_ids, answer_ids, max_length, tokenizer.eos_token_id)
        if packed.get("skipped"):
            skipped_rows.append({"id": row.get("id"), "reason": packed["reason"]})
            continue
        packed["id"] = row.get("id")
        prepared.append(packed)

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(prepared)

        def __getitem__(self, index):
            item = prepared[index]
            return {"input_ids": item["input_ids"], "labels": item["labels"]}

    def collate(batch):
        length = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (length - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (length - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (length - len(b["input_ids"])) for b in batch]
            ),
        }

    if not prepared:
        return {"skipped": True, "reason": "no_effective_target", "skipped_rows": skipped_rows}
    supervised_tokens = sum(item["assistant_target_tokens"] for item in prepared) * epochs
    use_cuda = torch.cuda.is_available()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_strategy="no",
            logging_steps=1,
            bf16=use_cuda,
            report_to=[],
            disable_tqdm=True,
            seed=seed,
        ),
        train_dataset=Rows(),
        data_collator=collate,
    )
    result = trainer.train()
    artifact = out_dir / "adapter"
    model.save_pretrained(artifact)
    tokenizer.save_pretrained(artifact)
    metrics = {
        "skipped": False,
        "train_loss": float(result.training_loss),
        "train_samples": len(prepared),
        "planned_samples": len(rows),
        "skipped_rows": skipped_rows,
        "supervised_tokens": supervised_tokens,
        "epochs": epochs,
        "seed": seed,
        "base_model": base_model,
        "adapter": str(artifact),
        "consumed_ids": [item.get("id") for item in prepared],
    }
    (out_dir / "sft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer, model
    if use_cuda:
        torch.cuda.empty_cache()
    return metrics


def generate_answers(
    questions: list,
    base_model: str,
    adapter: str | None = None,
    max_new_tokens: int = 512,
) -> list[str] | dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"missing_sft_deps: {exc}"}

    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    answers = []
    for question in questions:
        if isinstance(question, str):
            messages = [{"role": "system", "content": GROUNDED_POLICY}, {"role": "user", "content": question}]
        else:
            messages = question
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        answers.append(text.strip())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answers


def eval_messages(row: dict, context_key: str = "context") -> list[dict[str, str]]:
    question = str(row.get("question") or "")
    closed = row.get("task_mode") == "closed_book" or context_key == "closed_book"
    if closed:
        return [
            {"role": "system", "content": "可以运用已有领域知识回答。不确定时说明依据不足。"},
            {"role": "user", "content": question},
        ]
    context = str(row.get(context_key) or "")
    if context:
        user = f"资料：\n{context}\n\n问题：{question}"
    else:
        user = f"资料：\n（当前未提供可回答该问题的资料）\n\n问题：{question}"
    return [{"role": "system", "content": GROUNDED_POLICY}, {"role": "user", "content": user}]


def _prompt_for(row: dict, context_key: str = "context") -> list[dict[str, str]]:
    return eval_messages(row, context_key)


def _delta_label(base: str, tuned: str, gold: str, before: dict | None = None, after: dict | None = None) -> str:
    if before and after:
        return transition_label(before, after)
    if base.strip() == tuned.strip():
        return "实质相同"
    return "待裁定"


def write_answer_sheet(
    heldout: list[dict],
    base_preds: list[str],
    tuned_preds: list[str],
    path: Path,
    distractor_base: list[str] | None = None,
    distractor_tuned: list[str] | None = None,
) -> Path:
    """锁定测试每一题的全文对照。数字只放在文首索引。"""
    groups: dict[str, list[int]] = {}
    blocks = []
    for index, row in enumerate(heldout):
        gold = str(row.get("answer") or "")
        base = base_preds[index] if index < len(base_preds) else ""
        tuned = tuned_preds[index] if index < len(tuned_preds) else ""
        label = _delta_label(base, tuned, gold)
        groups.setdefault(label, []).append(index + 1)
        context = str(row.get("context") or "")
        blocks.append(
            "\n".join(
                [
                    f"## 题 {index + 1}",
                    "",
                    f"变化：{label}",
                    "",
                    "### 问题",
                    str(row.get("question") or ""),
                    "",
                    "### 学生看见的资料",
                    context or "（无）",
                    "",
                    "### 微调前",
                    base or "（空）",
                    "",
                    "### 微调后",
                    tuned or "（空）",
                    "",
                    "### 参考答案",
                    gold or "（空）",
                ]
            )
        )
    lines = ["# 微调前后回答对照", "", "下列是完整回答，不是摘要。", ""]
    for label, nums in groups.items():
        lines.append(f"- {label}：题 {', '.join(str(n) for n in nums)}")
    lines.append("")
    lines.extend(blocks)
    if distractor_base and distractor_tuned:
        lines.extend(["", "# 干扰资料下的回答", ""])
        for index, row in enumerate(heldout):
            lines.extend(
                [
                    f"## 干扰题 {index + 1}",
                    "",
                    "### 问题",
                    str(row.get("question") or ""),
                    "",
                    "### 干扰资料",
                    str(row.get("distractor") or ""),
                    "",
                    "### 微调前",
                    distractor_base[index] if index < len(distractor_base) else "",
                    "",
                    "### 微调后",
                    distractor_tuned[index] if index < len(distractor_tuned) else "",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def assemble_answer_book(out_dir: Path, entries: list[dict]) -> None:
    """默认管线的全文对照放在套件根目录，其余已训组别只附变化最大的几题。"""
    primary = next((entry for entry in entries if entry.get("answer_sheet")), None)
    if not primary:
        return
    source = out_dir / primary["id"] / "sft" / "answers.md"
    if not source.is_file():
        return
    parts = [source.read_text(encoding="utf-8").rstrip(), ""]
    for entry in entries:
        if entry.get("id") == primary["id"] or not entry.get("sft"):
            continue
        sheet = out_dir / entry["id"] / "sft" / "answers.md"
        if not sheet.is_file():
            continue
        picked = _changed_sections(sheet.read_text(encoding="utf-8"), limit=4)
        if not picked:
            continue
        parts.append(f"# {entry['id']} 非实质相同样例")
        parts.append("")
        parts.extend(picked)
    (out_dir / "answers.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _changed_sections(text: str, limit: int = 4) -> list[str]:
    chunks = text.split("\n## 题 ")
    ranked = []
    for chunk in chunks[1:]:
        head = chunk.split("###", 1)[0]
        if "变化：实质相同" in head:
            continue
        rank = 0 if "变化：前错后对" in head or "变化：前对后错" in head else 1
        ranked.append((rank, "## 题 " + chunk.strip()))
    ranked.sort(key=lambda item: item[0])
    return [text for _, text in ranked[:limit]]


def export_train_seen(pairs: list[QAPair], consumed_ids: list[str], path: Path) -> dict[str, Any]:
    wanted = {item for item in consumed_ids if item}
    rows = []
    for pair in pairs:
        if pair.qa_id not in wanted:
            continue
        pair.data_stage = "actually_trained"
        pair.exposure_count = max(1, pair.exposure_count)
        probe = eval_messages(
            {
                "question": pair.question,
                "context": pair.metadata.get("student_context") if "student_context" in pair.metadata else pair.chunk_text,
                "task_mode": "closed_book" if pair.goal == "closed_book_domain" else "rag_grounded",
            }
        )
        rows.append(
            {
                "case_id": pair.qa_id,
                "sample_id": pair.qa_id,
                "family_id": pair.family_id,
                "messages": probe,
                "answer": pair.answer,
                "answer_points": pair.answer_points,
                "exposure_count": pair.exposure_count,
                "layer": "T0",
            }
        )
    if not rows:
        return {"status": "not_executed", "reason": "missing_actually_trained"}
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return {"status": "executed", "n": len(rows), "path": str(path), "enters_generalization_score": False}


def evaluate_sft(
    pairs: list[QAPair],
    heldout_path: str | Path,
    out_dir: Path,
    llm=None,
    base_model: str = DEFAULT_BASE,
    skip_sft: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = write_sft_jsonl(pairs, out_dir / "train.jsonl")
    heldout = load_heldout(heldout_path)
    questions = [_prompt_for(h) for h in heldout]
    golds = [h["answer"] for h in heldout]
    report: dict[str, Any] = {"train_samples": len(pairs), "heldout": len(heldout), "train_jsonl": str(train_path)}
    report["train_seen"] = {"status": "not_executed", "reason": "missing_actually_trained"}
    ambiguous = [row for row in heldout if "这一句" in str(row.get("question") or "")]
    if ambiguous:
        report["heldout_validity"] = "ambiguous_referent"
        report["heldout_note"] = "题干歧义、不可用于主效果结论"
    if skip_sft:
        report["skipped"] = True
        report["reason"] = "skip_sft"
        return report
    trained = train_lora(train_path, out_dir / "lora", base_model=base_model)
    report["train"] = trained
    if trained.get("skipped"):
        return report
    report["train_seen"] = export_train_seen(pairs, trained.get("consumed_ids") or [], out_dir / "train_seen.jsonl")
    adapter = trained.get("adapter")
    base_preds = generate_answers(questions, base_model)
    tuned_preds = generate_answers(questions, base_model, adapter=adapter)
    if isinstance(base_preds, dict) or isinstance(tuned_preds, dict):
        report["eval_skipped"] = base_preds if isinstance(base_preds, dict) else tuned_preds
        return report
    case_ids = [str(row.get("id") or index) for index, row in enumerate(heldout)]
    report["base"] = score_generations(base_preds, golds, case_ids)
    report["tuned"] = score_generations(tuned_preds, golds, case_ids)
    if report["base"]["f1"] is None or report["tuned"]["f1"] is None:
        report["delta_f1"] = None
        report["delta_em"] = None
    else:
        report["delta_f1"] = round(report["tuned"]["f1"] - report["base"]["f1"], 4)
        report["delta_em"] = round(report["tuned"]["em"] - report["base"]["em"], 4)
    missing_distractor = [row.get("id") for row in heldout if row.get("require_distractor") and not row.get("distractor")]
    if missing_distractor:
        raise AlignmentError("缺失干扰上下文: " + ",".join(str(item) for item in missing_distractor))
    distractor_base = distractor_tuned = None
    if any(row.get("distractor") for row in heldout):
        distractor_prompts = [_prompt_for(row, "distractor") if row.get("distractor") else _prompt_for(row) for row in heldout]
        distractor_base = generate_answers(distractor_prompts, base_model)
        distractor_tuned = generate_answers(distractor_prompts, base_model, adapter=adapter)
        if isinstance(distractor_base, dict) or isinstance(distractor_tuned, dict):
            distractor_base = distractor_tuned = None
        elif len(distractor_base) != len(heldout) or len(distractor_tuned) != len(heldout):
            raise AlignmentError("干扰回答数量与题目不一致")
    sheet = write_answer_sheet(
        heldout,
        base_preds,
        tuned_preds,
        out_dir / "answers.md",
        distractor_base if isinstance(distractor_base, list) else None,
        distractor_tuned if isinstance(distractor_tuned, list) else None,
    )
    report["answer_sheet"] = str(sheet)
    by_source: dict[str, list[float]] = {}
    transitions = []
    for row, base, tuned in zip(heldout, base_preds, tuned_preds):
        case = {
            "answer": row.get("answer"),
            "answer_points": row.get("answer_points") or [row.get("answer")],
            "expected_action": row.get("expected_action") or "answer",
        }
        before = score_task(base, case)
        after = score_task(tuned, case)
        transitions.append(transition_label(before, after))
        source = str(row.get("source_family_id") or row.get("family_id") or row.get("id"))
        by_source.setdefault(source, []).append(float(after["passed"]) - float(before["passed"]))
    report["task_transitions"] = transitions
    report["clustered_delta"] = clustered_interval({key: sum(vals) / len(vals) for key, vals in by_source.items()})
    report["pass_decision"] = "delta_min_unset"
    report["retention_claim"] = "保留测试未配置，只报告效应与区间"
    report["effect_note"] = "词面 F1 只作辅助诊断，不能单独作为质量结论"
    if llm is not None:
        report["judge_base"] = judge_vs_reference(llm, questions, base_preds, golds)
        report["judge_tuned"] = judge_vs_reference(llm, questions, tuned_preds, golds)
        base_mean = report["judge_base"].get("mean")
        tuned_mean = report["judge_tuned"].get("mean")
        report["delta_judge"] = None if base_mean is None or tuned_mean is None else round(tuned_mean - base_mean, 4)
    (out_dir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
