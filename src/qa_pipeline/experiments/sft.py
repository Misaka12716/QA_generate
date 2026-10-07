"""独立 LoRA SFT 与 held-out 评测，超参对齐智训 training_worker。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..adapters.zhixun import CLOSED_POLICY, GROUNDED_POLICY, ReleaseRejected, assert_releasable, to_zhixun_row
from ..task_mode import CLOSED_BOOK, declared_task_mode
from ..llm import json_payload
from ..schemas import QAPair
from ..textutil import exact_match, token_f1
from .scoring import AlignmentError, build_supervised_batch, clustered_interval, score_task, transition_label

logger = logging.getLogger(__name__)

DEFAULT_BASE = os.environ.get(
    "QA_PIPELINE_SFT_BASE",
    "/data/pjw/data/models/Qwen2.5-7B-Instruct",
)


def write_sft_jsonl(
    pairs: list[QAPair],
    path: Path,
    *,
    review_policy: Any = None,
    review_aggregates: dict[str, Any] | None = None,
    skipped: list[dict[str, Any]] | None = None,
) -> Path:
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
            except ReleaseRejected as exc:
                if skipped is not None:
                    skipped.append({"id": pair.qa_id, "reasons": [str(exc)]})
                continue
            if review_policy is not None:
                from ..reviewing.policy import release_block_reasons

                aggregate = None if review_aggregates is None else review_aggregates.get(pair.qa_id)
                reasons = release_block_reasons(pair.model_dump(), aggregate, review_policy)
                if reasons:
                    if skipped is not None:
                        skipped.append({"id": pair.qa_id, "reasons": reasons})
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


EXPOSURE_DEFINITION = (
    "consumed_ids 是预处理通过并进入 Dataset 的样本。"
    "exposure_count 等于 epoch 数，表示按训练轮次计的计划曝光。"
    "trainer_global_step 是优化步数。这不是逐批消费日志。"
)
TASK_ACCURACY_NOTE = "task_accuracy 只作辅助，不是已验证的语义正确率。"


def _template_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def encode_supervised_messages(tokenizer, messages: list[dict], max_length: int) -> dict[str, Any]:
    """用聊天模板编码完整 assistant 回合，保留模板自己的结束标记。正文为空则拒绝。"""
    if not messages or messages[-1].get("role") != "assistant":
        return {"skipped": True, "reason": "no_effective_target"}
    body = str(messages[-1].get("content") or "").strip()
    if not body:
        return {"skipped": True, "reason": "no_effective_target"}
    try:
        prompt_ids = _template_ids(tokenizer, messages[:-1], True)
        full_ids = _template_ids(tokenizer, messages, False)
    except Exception as exc:
        return {"skipped": True, "reason": f"template_failed: {exc}"}
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return {"skipped": True, "reason": "template_mismatch"}
    answer_ids = full_ids[len(prompt_ids) :]
    closer: list[int] = []
    try:
        empty_ids = _template_ids(tokenizer, [*messages[:-1], {"role": "assistant", "content": ""}], False)
        if empty_ids[: len(prompt_ids)] == prompt_ids:
            closer = empty_ids[len(prompt_ids) :]
    except Exception:
        closer = []
    if not answer_ids or (closer and answer_ids == closer):
        return {"skipped": True, "reason": "no_effective_target"}
    return build_supervised_batch(prompt_ids, answer_ids, max_length, getattr(tokenizer, "eos_token_id", None))


def train_lora(
    train_jsonl: Path,
    out_dir: Path,
    base_model: str = DEFAULT_BASE,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 16,
    max_length: int = 1024,
    seed: int = 42,
    save_epochs: bool = False,
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
        packed = encode_supervised_messages(tokenizer, messages, max_length)
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
    per_epoch_tokens = sum(item["assistant_target_tokens"] for item in prepared)
    supervised_tokens = per_epoch_tokens * epochs
    prompt_masked = all(item.get("prompt_masked") for item in prepared)
    use_cuda = torch.cuda.is_available()
    train_args: dict[str, Any] = {
        "output_dir": str(out_dir / "trainer"),
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_strategy": "no",
        "logging_steps": 1,
        "bf16": use_cuda,
        "report_to": [],
        "disable_tqdm": True,
        "seed": seed,
    }
    if save_epochs:
        train_args["save_strategy"] = "epoch"
        train_args["save_total_limit"] = 3
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**train_args),
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
        "supervised_tokens_per_epoch": per_epoch_tokens,
        "supervised_tokens_note": "supervised_tokens 是每条答案 token 之和再乘 epoch，不是单条答案长度。",
        "per_sample_supervised_tokens": [item.get("assistant_target_tokens") for item in prepared],
        "prompt_masked": prompt_masked,
        "chat_template_sha256": hashlib.sha256(str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")).hexdigest(),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "epochs": epochs,
        "save_epochs": save_epochs,
        "seed": seed,
        "base_model": base_model,
        "adapter": str(artifact),
        "consumed_ids": [item.get("id") for item in prepared],
        "global_step": int(getattr(result, "global_step", 0) or 0),
        "optimizer_steps": int(getattr(result, "global_step", 0) or 0),
        "exposure_count": epochs,
        "exposure_definition": EXPOSURE_DEFINITION,
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
    if not questions:
        return []
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
    mode = declared_task_mode(row, context_key=context_key)
    if mode == CLOSED_BOOK:
        return [
            {"role": "system", "content": CLOSED_POLICY},
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


def _gold_text(row: dict) -> str:
    answer = str(row.get("answer") or "").strip()
    if answer:
        return answer
    points = [str(point).strip() for point in (row.get("answer_points") or []) if str(point).strip()]
    return "\n".join(points)


def partition_heldout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """主指标只收审核通过、金标完整、ID 唯一且来源合格的题。未审核题显式拒绝。"""
    from .drug_corpus import admits_main_metric

    eligible: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        case_id = str(row.get("id") or "")
        reasons: list[str] = []
        if not case_id:
            reasons.append("missing_id")
        elif case_id in seen:
            reasons.append("duplicate_id")
        else:
            seen.add(case_id)
        if not admits_main_metric(row):
            reasons.append("not_admitted")
        if not _gold_text(row):
            reasons.append("missing_gold")
        if not (row.get("source_family_id") or row.get("source")):
            reasons.append("missing_source")
        if row.get("historical"):
            reasons.append("historical")
        if row.get("main_metric") is False:
            reasons.append("not_main")
        if reasons:
            rejected.append({"id": case_id or index, "reasons": reasons})
        else:
            eligible.append(row)
    return eligible, rejected


def _not_executed_probe(reason: str, epochs: int | None = None, global_step: int | None = None, **extra: Any) -> dict[str, Any]:
    payload = {
        "export_status": "not_executed",
        "eval_status": "not_executed",
        "reason": reason,
        "enters_generalization_score": False,
        "exposure_definition": EXPOSURE_DEFINITION,
        "epochs": epochs,
        "trainer_global_step": global_step,
        "task_accuracy_note": TASK_ACCURACY_NOTE,
    }
    payload.update(extra)
    return payload


def build_train_seen_rows(
    train_path: Path,
    consumed_ids: list[str],
    epochs: int | None,
    global_step: int | None,
) -> dict[str, Any]:
    """按消费 ID 回查 train.jsonl。输入是 messages[:-1]，目标是最终 assistant 消息。"""
    ids = [str(item) for item in consumed_ids if item]
    if not ids:
        return _not_executed_probe("missing_actually_trained", epochs, global_step)
    by_id: dict[str, dict] = {}
    if train_path.is_file():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_id[str(row.get("id"))] = row
    missing = [item for item in ids if item not in by_id]
    if missing:
        return _not_executed_probe("train_row_missing", epochs, global_step, missing_ids=missing)
    probes = []
    for case_id in ids:
        messages = list(by_id[case_id].get("messages") or [])
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            return _not_executed_probe("train_message_missing", epochs, global_step, missing_ids=[case_id])
        probes.append(
            {
                "case_id": case_id,
                "sample_id": case_id,
                "messages": messages[:-1],
                "answer": messages[-1].get("content") or "",
                "exposure_count": epochs,
                "layer": "T0",
            }
        )
    return {
        "export_status": "ready",
        "eval_status": "not_executed",
        "reason": None,
        "probes": probes,
        "n": len(probes),
        "enters_generalization_score": False,
        "exposure_definition": EXPOSURE_DEFINITION,
        "epochs": epochs,
        "trainer_global_step": global_step,
        "task_accuracy_note": TASK_ACCURACY_NOTE,
    }


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _score_main(report: dict[str, Any], eligible: list[dict], base_preds: list[str], tuned_preds: list[str], llm, base_model: str, adapter: str | None) -> None:
    questions = [_prompt_for(row) for row in eligible]
    golds = [_gold_text(row) for row in eligible]
    case_ids = [str(row.get("id")) for row in eligible]
    report["base"] = score_generations(base_preds, golds, case_ids)
    report["tuned"] = score_generations(tuned_preds, golds, case_ids)
    if report["base"]["f1"] is None or report["tuned"]["f1"] is None:
        report["delta_f1"] = None
        report["delta_em"] = None
    else:
        report["delta_f1"] = round(report["tuned"]["f1"] - report["base"]["f1"], 4)
        report["delta_em"] = round(report["tuned"]["em"] - report["base"]["em"], 4)
    missing_distractor = [row.get("id") for row in eligible if row.get("require_distractor") and not row.get("distractor")]
    if missing_distractor:
        raise AlignmentError("缺失干扰上下文: " + ",".join(str(item) for item in missing_distractor))
    distractor_base = distractor_tuned = None
    if any(row.get("distractor") for row in eligible):
        distractor_prompts = [_prompt_for(row, "distractor") if row.get("distractor") else _prompt_for(row) for row in eligible]
        distractor_base = generate_answers(distractor_prompts, base_model)
        distractor_tuned = generate_answers(distractor_prompts, base_model, adapter=adapter)
        if isinstance(distractor_base, dict) or isinstance(distractor_tuned, dict):
            distractor_base = distractor_tuned = None
        elif len(distractor_base) != len(eligible) or len(distractor_tuned) != len(eligible):
            raise AlignmentError("干扰回答数量与题目不一致")
    by_source: dict[str, list[float]] = {}
    transitions = []
    for row, base, tuned in zip(eligible, base_preds, tuned_preds):
        case = {
            "answer": _gold_text(row),
            "answer_points": row.get("answer_points") or [_gold_text(row)],
            "expected_action": row.get("expected_action") or "answer",
        }
        before = score_task(base, case)
        after = score_task(tuned, case)
        transitions.append(transition_label(before, after))
        source = str(row.get("source_family_id") or row.get("family_id") or row.get("id"))
        by_source.setdefault(source, []).append(float(after["passed"]) - float(before["passed"]))
    report["task_transitions"] = transitions
    report["clustered_delta"] = clustered_interval({key: sum(vals) / len(vals) for key, vals in by_source.items()})
    report["task_accuracy_note"] = TASK_ACCURACY_NOTE
    if llm is not None:
        report["judge_base"] = judge_vs_reference(llm, questions, base_preds, golds)
        report["judge_tuned"] = judge_vs_reference(llm, questions, tuned_preds, golds)
        base_mean = report["judge_base"].get("mean")
        tuned_mean = report["judge_tuned"].get("mean")
        report["delta_judge"] = None if base_mean is None or tuned_mean is None else round(tuned_mean - base_mean, 4)
    sheet = write_answer_sheet(
        [{**row, "answer": _gold_text(row)} for row in eligible],
        base_preds,
        tuned_preds,
        Path(report["answer_sheet_path"]),
        distractor_base if isinstance(distractor_base, list) else None,
        distractor_tuned if isinstance(distractor_tuned, list) else None,
    )
    report["answer_sheet"] = str(sheet)


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
    heldout_file = Path(heldout_path)
    parse_errors: list[dict] = []
    heldout: list[dict] = []
    if heldout_file.is_file():
        for index, line in enumerate(heldout_file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                heldout.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors.append({"id": index, "reasons": ["invalid_json"]})
    eligible, rejected = partition_heldout(heldout)
    rejected = parse_errors + rejected
    report: dict[str, Any] = {
        "train_samples": len(pairs),
        "heldout": len(heldout),
        "heldout_main": len(eligible),
        "heldout_rejections": rejected,
        "train_jsonl": str(train_path),
        "delta_f1": None,
        "delta_em": None,
        "main_metric": "not_executed" if not eligible else "pending",
        "main_eval": "model_not_loaded" if not eligible else "pending",
        "task_accuracy_note": TASK_ACCURACY_NOTE,
        "effect_note": "词面 F1 只作辅助诊断，不能单独作为质量结论",
        "pass_decision": "delta_min_unset",
        "retention_claim": "保留测试未配置，只报告效应与区间",
        "answer_sheet_path": str(out_dir / "answers.md"),
    }
    report["train_seen"] = _not_executed_probe("missing_actually_trained")
    if any("这一句" in str(row.get("question") or "") for row in heldout):
        report["heldout_validity"] = "ambiguous_referent"
        report["heldout_note"] = "题干歧义、不可用于主效果结论"

    def finish() -> dict[str, Any]:
        public = {key: value for key, value in report.items() if key != "answer_sheet_path"}
        (out_dir / "eval.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
        return public

    if skip_sft:
        report["skipped"] = True
        report["reason"] = "skip_sft"
        return finish()
    trained = train_lora(train_path, out_dir / "lora", base_model=base_model)
    report["train"] = trained
    if trained.get("skipped"):
        return finish()
    epochs = trained.get("epochs")
    global_step = trained.get("global_step")
    probe = build_train_seen_rows(train_path, trained.get("consumed_ids") or [], epochs, global_step)
    seen_path = out_dir / "train_seen.jsonl"
    if probe.get("export_status") != "ready":
        report["train_seen"] = probe
    else:
        rows = probe.pop("probes")
        _write_jsonl_rows(seen_path, rows)
        probe["export_status"] = "exported"
        probe["path"] = str(seen_path)
        base_seen = generate_answers([row["messages"] for row in rows], base_model)
        tuned_seen = generate_answers([row["messages"] for row in rows], base_model, adapter=trained.get("adapter"))
        if isinstance(base_seen, dict) or isinstance(tuned_seen, dict):
            probe["eval_status"] = "not_executed"
            probe["reason"] = (base_seen if isinstance(base_seen, dict) else tuned_seen).get("reason")
            report["train_seen"] = probe
        else:
            scored = []
            passed = 0
            for row, base, tuned in zip(rows, base_seen, tuned_seen):
                case = {"answer": row["answer"], "answer_points": [row["answer"]], "expected_action": "answer"}
                before = score_task(base, case)
                after = score_task(tuned, case)
                if after["passed"]:
                    passed += 1
                scored.append(
                    {
                        **row,
                        "base_answer": base,
                        "tuned_answer": tuned,
                        "before": before,
                        "after": after,
                        "change": transition_label(before, after),
                    }
                )
            _write_jsonl_rows(seen_path, scored)
            probe["eval_status"] = "executed"
            probe["n"] = len(scored)
            probe["auxiliary_pass_rate"] = round(passed / len(scored), 4) if scored else None
            probe["enters_generalization_score"] = False
            report["train_seen"] = probe
    if not eligible:
        report["main_metric"] = "not_executed"
        report["main_eval"] = "model_not_loaded"
        report["delta_f1"] = None
        return finish()
    adapter = trained.get("adapter")
    questions = [_prompt_for(row) for row in eligible]
    base_preds = generate_answers(questions, base_model)
    tuned_preds = generate_answers(questions, base_model, adapter=adapter)
    if isinstance(base_preds, dict) or isinstance(tuned_preds, dict):
        report["main_metric"] = "not_executed"
        report["main_eval"] = "generation_failed"
        report["eval_skipped"] = base_preds if isinstance(base_preds, dict) else tuned_preds
        report["delta_f1"] = None
        return finish()
    _score_main(report, eligible, base_preds, tuned_preds, llm, base_model, adapter)
    report["main_metric"] = "executed"
    report["main_eval"] = "executed"
    return finish()
