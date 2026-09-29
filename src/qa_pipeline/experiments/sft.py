"""独立 LoRA SFT 与 held-out 评测，超参对齐智训 training_worker。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..adapters.zhixun import to_zhixun_row
from ..schemas import QAPair
from ..textutil import exact_match, token_f1

logger = logging.getLogger(__name__)

DEFAULT_BASE = os.environ.get(
    "QA_PIPELINE_SFT_BASE",
    "/data/pjw/data/models/Qwen2.5-7B-Instruct",
)


def write_sft_jsonl(pairs: list[QAPair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(to_zhixun_row(p), ensure_ascii=False) + "\n")
    return path


def load_heldout(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_generations(preds: list[str], golds: list[str]) -> dict[str, float]:
    ems, f1s = [], []
    for p, g in zip(preds, golds):
        ems.append(exact_match(p, g))
        f1s.append(token_f1(p, g))
    n = max(1, len(golds))
    return {"em": round(sum(ems) / n, 4), "f1": round(sum(f1s) / n, 4), "n": len(golds)}


def judge_vs_reference(llm, questions: list[str], preds: list[str], golds: list[str], model: str | None = None) -> float:
    wins = 0
    for q, pred, gold in zip(questions, preds, golds):
        data = llm.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "比较模型回答与参考答案的事实一致性，1-5 分。"
                        '输出 JSON：{"score":1}'
                    ),
                },
                {"role": "user", "content": f"问题：{q}\n参考：{gold}\n模型：{pred}"},
            ],
            model=model,
            max_tokens=80,
        ) or {}
        wins += float(data.get("score") or 0)
    return round(wins / max(1, len(golds)), 4)


def train_lora(
    train_jsonl: Path,
    out_dir: Path,
    base_model: str = DEFAULT_BASE,
    epochs: int = 3,
    lr: float = 2e-4,
    rank: int = 16,
    max_length: int = 1024,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover
        logger.warning("SFT extras missing: %s", exc)
        return {"skipped": True, "reason": f"missing_sft_deps: {exc}"}

    rows = [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"skipped": True, "reason": "empty_train"}
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

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(
                prompt + messages[-1]["content"] + (tokenizer.eos_token or ""),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(pids) >= len(ids):
                ids = ids + tokenizer(tokenizer.eos_token or "", add_special_tokens=False)["input_ids"]
            labels = [-100] * min(len(pids), len(ids)) + ids[len(pids) :]
            return {"input_ids": ids, "labels": labels[: len(ids)]}

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
            seed=42,
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
        "train_samples": len(rows),
        "epochs": epochs,
        "base_model": base_model,
        "adapter": str(artifact),
    }
    (out_dir / "sft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer, model
    if use_cuda:
        torch.cuda.empty_cache()
    return metrics


def generate_answers(
    questions: list[str],
    base_model: str,
    adapter: str | None = None,
    max_new_tokens: int = 128,
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
    for q in questions:
        messages = [{"role": "user", "content": q}]
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
    questions = [h["question"] for h in heldout]
    golds = [h["answer"] for h in heldout]
    report: dict[str, Any] = {"train_samples": len(pairs), "heldout": len(heldout), "train_jsonl": str(train_path)}
    if skip_sft:
        report["skipped"] = True
        report["reason"] = "skip_sft"
        return report
    trained = train_lora(train_path, out_dir / "lora", base_model=base_model)
    report["train"] = trained
    if trained.get("skipped"):
        return report
    adapter = trained.get("adapter")
    base_preds = generate_answers(questions, base_model)
    tuned_preds = generate_answers(questions, base_model, adapter=adapter)
    if isinstance(base_preds, dict) or isinstance(tuned_preds, dict):
        report["eval_skipped"] = base_preds if isinstance(base_preds, dict) else tuned_preds
        return report
    report["base"] = score_generations(base_preds, golds)
    report["tuned"] = score_generations(tuned_preds, golds)
    report["delta_f1"] = round(report["tuned"]["f1"] - report["base"]["f1"], 4)
    report["delta_em"] = round(report["tuned"]["em"] - report["base"]["em"], 4)
    if llm is not None:
        report["judge_base"] = judge_vs_reference(llm, questions, base_preds, golds)
        report["judge_tuned"] = judge_vs_reference(llm, questions, tuned_preds, golds)
        report["delta_judge"] = round(report["judge_tuned"] - report["judge_base"], 4)
    (out_dir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
