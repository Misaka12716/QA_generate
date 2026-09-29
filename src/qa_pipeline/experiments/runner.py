"""实验套件：按 YAML 跑多组 recipe，产出 metrics.json 与 report.md。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..config import Recipe, StrategySpec, load_recipe
from ..llm import FakeLLM, LLMClient
from ..pipeline import Pipeline, PipelineResult
from ..schemas import QAPair
from ..store import save_result
from .metrics import compute_metrics
from .sft import evaluate_sft


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_suite(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def recipe_snapshot(recipe: Recipe) -> dict[str, Any]:
    """记录一组实验实际使用的各阶段策略，供对照台解释分数差异。"""

    def spec(item: StrategySpec) -> dict[str, Any]:
        return {"name": item.name, **item.params}

    return {
        "name": recipe.name,
        "description": recipe.description,
        "chunking": spec(recipe.chunking),
        "anchor": spec(recipe.anchor),
        "question_gen": spec(recipe.question_gen),
        "evolution": spec(recipe.evolution),
        "question_filter": spec(recipe.question_filter),
        "distillation": spec(recipe.distillation),
        "teacher_router": spec(recipe.teacher_router),
        "filters": [spec(item) for item in recipe.filters],
        "grading": recipe.grading,
    }


def _recipe_from_entry(entry: dict[str, Any], recipes_dir: Path) -> Recipe:
    if "recipe" in entry:
        return load_recipe(recipes_dir / entry["recipe"] if not Path(entry["recipe"]).is_file() else entry["recipe"])
    if "from" in entry:
        base = load_recipe(recipes_dir / entry["from"] if not Path(entry["from"]).is_file() else entry["from"])
        data = base.dump()
        overlay = deepcopy(entry.get("override") or {})
        data.update({k: v for k, v in overlay.items() if k != "filters"})
        if "filters" in overlay:
            data["filters"] = overlay["filters"]
        if entry.get("name"):
            data["name"] = entry["name"]
        return Recipe.from_dict(data)
    return Recipe.from_dict(entry)


def _render_report(suite_name: str, rows: list[dict[str, Any]], notes: list[str]) -> str:
    keys = [
        "id",
        "kept",
        "retention",
        "answerability",
        "evidence_grounded",
        "nli_mean",
        "judge_mean",
        "type_entropy",
        "s_ratio",
        "tokens_per_10k_kept",
        "delta_f1",
    ]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [f"# {suite_name} 实验报告", "", header, sep]
    for row in rows:
        m = row.get("metrics") or {}
        sft = row.get("sft") or {}
        values = {
            "id": row.get("id"),
            "kept": m.get("kept"),
            "retention": m.get("retention"),
            "answerability": m.get("answerability"),
            "evidence_grounded": m.get("evidence_grounded"),
            "nli_mean": m.get("nli_mean"),
            "judge_mean": m.get("judge_mean"),
            "type_entropy": m.get("type_entropy"),
            "s_ratio": m.get("s_ratio"),
            "tokens_per_10k_kept": m.get("tokens_per_10k_kept"),
            "delta_f1": sft.get("delta_f1"),
        }
        lines.append("| " + " | ".join(str(values[k]) for k in keys) + " |")
    lines += ["", "## 说明", ""]
    for row in rows:
        extra = row.get("note") or ""
        lines.append(f"- **{row.get('id')}**：{row.get('purpose') or ''} {extra}".rstrip())
    if notes:
        lines += ["", "## 运行约束", ""]
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines) + "\n"


def _empty_result(name: str, pairs: list[QAPair]) -> PipelineResult:
    from ..schemas import UsageStats

    return PipelineResult(
        recipe_name=name,
        documents=[],
        chunks=[],
        questions=[],
        pairs=pairs,
        rejected=[],
        stats=UsageStats(produced={"distilled": len(pairs), "kept": len(pairs), "questions": len(pairs)}),
        raw_pairs=list(pairs),
    )


def _apply_grades(pairs: list[QAPair], grades: list[str] | None) -> list[QAPair]:
    if not grades:
        return list(pairs)
    allow = set(grades)
    return [p.model_copy(deep=True) for p in pairs if p.grade in allow]


def _export_annotation(pairs: list[QAPair], path: Path, grades: list[str] | None) -> dict[str, Any]:
    chosen = _apply_grades(pairs, grades or ["S", "A"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in chosen:
            handle.write(
                json.dumps(
                    {
                        "qa_id": pair.qa_id,
                        "grade": pair.grade,
                        "question": pair.question,
                        "answer": pair.answer,
                        "evidence": pair.evidence_span,
                        "q_type": pair.q_type,
                        "labels": {
                            "factuality": None,
                            "answerability": None,
                            "completeness": None,
                            "language": None,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    labeled = [p for p in chosen if (p.metadata or {}).get("human_labels")]
    return {
        "exported": len(chosen),
        "path": str(path),
        "annotated": bool(labeled),
        "reason": None if labeled else "未提供人工标签，不计算 Cohen's Kappa",
        "kappa": None,
    }


def _reset_pairs(pairs: list[QAPair]) -> list[QAPair]:
    copies = []
    for p in pairs:
        c = p.model_copy(deep=True)
        c.action = "pass"
        c.grade = None
        c.filter_trace = {}
        copies.append(c)
    return copies


def run_suite(
    suite_path: str | Path,
    out_dir: str | Path | None = None,
    fake: bool = False,
    skip_sft: bool = True,
    llm: LLMClient | None = None,
    base_model: str | None = None,
) -> dict[str, Any]:
    from ..adapters.zhixun import export_zhixun
    from .sft import DEFAULT_BASE

    suite = load_suite(suite_path)
    root = _repo_root()
    recipes_dir = Path(suite.get("recipes_dir") or root / "configs" / "recipes")
    if not recipes_dir.is_absolute():
        recipes_dir = (root / recipes_dir).resolve()
    input_path = Path(suite.get("input") or root / "fixtures" / "sample_manual.md")
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()
    heldout = Path(suite.get("heldout") or root / "fixtures" / "heldout.jsonl")
    if not heldout.is_absolute():
        heldout = (root / heldout).resolve()
    refusal_path = Path(suite.get("refusal") or root / "fixtures" / "refusal.jsonl")
    if not refusal_path.is_absolute():
        refusal_path = (root / refusal_path).resolve()
    out = Path(out_dir or root / "runs" / suite.get("name", "suite"))
    out.mkdir(parents=True, exist_ok=True)
    client = llm or (FakeLLM() if fake else LLMClient())
    student = base_model or DEFAULT_BASE
    llm_label = "fake" if isinstance(client, FakeLLM) else ("local" if type(client).__name__ == "LocalLLM" else "live")

    cache_pairs: dict[str, list[QAPair]] = {}
    cache_kept: dict[str, list[QAPair]] = {}
    cache_questions: dict[str, list] = {}
    sft_done: dict[tuple[str, ...], dict[str, Any]] = {}
    adapters: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    notes = [
        "语料仅为 fixtures/sample_manual.md，规模小于设计稿的 500 chunk / 1 万条。",
        "教师与学生均为本地 Qwen2.5-7B-Instruct 时，API 费用记 0 美元，并报告等价 token。",
        "E5 无人工标签，不计算 Cohen's Kappa。事实遵循使用 NLI 与证据子串，未接入 RAGAS。",
    ]

    def remember(entry: dict[str, Any], result: PipelineResult) -> None:
        key = entry.get("cache_as")
        if key:
            cache_pairs[key] = list(result.raw_pairs or (list(result.pairs) + list(result.rejected)))
            cache_kept[key] = list(result.pairs)
        qkey = entry.get("cache_questions")
        if qkey:
            cache_questions[qkey] = list(result.questions)

    def maybe_sft(entry: dict[str, Any], pairs: list[QAPair], exp_dir: Path) -> dict[str, Any]:
        if skip_sft or not entry.get("sft"):
            return {"skipped": True, "reason": "skip_sft" if skip_sft else "not_requested"}
        if hasattr(client, "release"):
            client.release()
        sig = tuple(sorted(p.qa_id for p in pairs))
        if not sig:
            return {"skipped": True, "reason": "empty_train"}
        if sig in sft_done:
            return {**sft_done[sig], "reused": True}
        report = evaluate_sft(pairs, heldout, exp_dir / "sft", llm=client, base_model=student, skip_sft=False)
        sft_done[sig] = report
        adapter = (report.get("train") or {}).get("adapter")
        if adapter:
            adapters[entry["id"]] = adapter
        return report

    def finalize(entry: dict[str, Any], result: PipelineResult, recipe: Recipe | None, exp_dir: Path, note: str = "") -> None:
        grades = entry.get("grades")
        full_kept = list(result.pairs)
        remember(entry, result)
        if grades:
            result.pairs = _apply_grades(full_kept, grades)
        save_result(result, exp_dir)
        export_zhixun(result.pairs, exp_dir / "zhixun.jsonl")
        metrics = compute_metrics(result)
        row = {
            "id": entry["id"],
            "purpose": entry.get("purpose", ""),
            "recipe": recipe.name if recipe else entry.get("recipe", ""),
            "recipe_snapshot": recipe_snapshot(recipe) if recipe else {},
            "metrics": metrics,
            "sft": maybe_sft(entry, result.pairs, exp_dir),
            "note": note,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)

    for entry in suite.get("experiments") or []:
        exp_id = entry["id"]
        kind = entry.get("kind") or "pipeline"
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(client, "reset_usage"):
            client.reset_usage()

        if kind == "annotate":
            source = cache_kept.get(entry.get("from_kept") or "") or []
            ann = _export_annotation(source, exp_dir / "annotation.jsonl", entry.get("grades"))
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {"kept": ann["exported"]},
                    "sft": {},
                    "note": ann.get("reason") or "",
                    "annotation": ann,
                }
            )
            continue

        if kind == "cost":
            source_id = entry.get("source")
            found = next((row for row in rows if row["id"] == source_id), None)
            metrics = dict((found or {}).get("metrics") or {})
            metrics["api_cost_usd"] = metrics.get("estimated_cost_usd")
            metrics["cost_note"] = "本地教师 API 费用按 0 计，tokens_per_10k_kept 为等价 token。"
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": metrics,
                    "sft": {},
                    "note": metrics["cost_note"],
                }
            )
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        if kind == "refusal":
            report: dict[str, Any] = {"skipped": skip_sft, "arms": {}}
            if skip_sft or not refusal_path.is_file():
                report["reason"] = "skip_sft" if skip_sft else f"missing {refusal_path}"
            else:
                if hasattr(client, "release"):
                    client.release()
                from .refusal import evaluate_refusal, load_refusal_set

                refusal_rows = load_refusal_set(refusal_path)
                for arm in entry.get("compare") or []:
                    adapter = adapters.get(arm)
                    if not adapter:
                        report["arms"][arm] = {"skipped": True, "reason": "no_adapter"}
                        continue
                    report["arms"][arm] = evaluate_refusal(refusal_rows, student, adapter)
            (exp_dir / "refusal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(
                {
                    "id": exp_id,
                    "purpose": entry.get("purpose", ""),
                    "recipe": "",
                    "recipe_snapshot": {},
                    "metrics": {},
                    "sft": {},
                    "note": json.dumps(report.get("arms") or report.get("reason"), ensure_ascii=False),
                    "refusal": report,
                }
            )
            continue

        if kind == "subset":
            source = cache_kept.get(entry.get("from_kept") or "") or []
            pairs = _apply_grades(source, entry.get("grades"))
            result = _empty_result(entry.get("name") or exp_id, pairs)
            finalize(entry, result, None, exp_dir)
            continue

        recipe = _recipe_from_entry(entry, recipes_dir)
        pipe = Pipeline(recipe, llm=client)
        q_source = entry.get("from_questions")
        kept_source = entry.get("from_kept")
        raw_source = entry.get("from_cache")
        if q_source and q_source in cache_questions:
            result = pipe.distill_from_questions(cache_questions[q_source])
            result.recipe_name = recipe.name
        elif kept_source and kept_source in cache_kept:
            pairs = _apply_grades(cache_kept[kept_source], entry.get("grades"))
            result = _empty_result(recipe.name, pairs)
            entry = {**entry, "grades": None}
        elif raw_source and raw_source in cache_pairs:
            result = pipe.refilter(_reset_pairs(cache_pairs[raw_source]), llm=client)
            result.recipe_name = recipe.name
        else:
            result = pipe.run(input_path)
        finalize(entry, result, recipe, exp_dir)

    payload = {
        "suite": suite.get("name"),
        "input": str(input_path),
        "llm": llm_label,
        "base_model": student,
        "limitations": notes,
        "experiments": rows,
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _render_report(suite.get("name", "suite"), rows, notes)
    (out / "report.md").write_text(report, encoding="utf-8")
    return payload
