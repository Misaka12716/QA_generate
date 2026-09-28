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


def _render_report(suite_name: str, rows: list[dict[str, Any]]) -> str:
    keys = [
        "id",
        "kept",
        "retention",
        "nli_mean",
        "judge_mean",
        "knowledge_gain_rate",
        "type_entropy",
        "s_ratio",
        "estimated_cost_usd",
        "delta_f1",
    ]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [
        f"# {suite_name} 实验报告",
        "",
        header,
        sep,
    ]
    for row in rows:
        m = row.get("metrics") or {}
        sft = row.get("sft") or {}
        values = {
            "id": row.get("id"),
            "kept": m.get("kept"),
            "retention": m.get("retention"),
            "nli_mean": m.get("nli_mean"),
            "judge_mean": m.get("judge_mean"),
            "knowledge_gain_rate": m.get("knowledge_gain_rate"),
            "type_entropy": m.get("type_entropy"),
            "s_ratio": m.get("s_ratio"),
            "estimated_cost_usd": m.get("estimated_cost_usd"),
            "delta_f1": sft.get("delta_f1"),
        }
        lines.append("| " + " | ".join(str(values[k]) for k in keys) + " |")
    lines += ["", "## 说明", ""]
    for row in rows:
        lines.append(f"- **{row.get('id')}**：{row.get('purpose') or ''}")
    return "\n".join(lines) + "\n"


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
) -> dict[str, Any]:
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
    out = Path(out_dir or root / "runs" / suite.get("name", "suite"))
    out.mkdir(parents=True, exist_ok=True)
    client = llm or (FakeLLM() if fake else LLMClient())

    cache_pairs: dict[str, list[QAPair]] = {}
    rows = []
    for entry in suite.get("experiments") or []:
        exp_id = entry["id"]
        recipe = _recipe_from_entry(entry, recipes_dir)
        exp_dir = out / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        source = entry.get("from_cache")
        if source and source in cache_pairs:
            pipe = Pipeline(recipe, llm=client)
            result = pipe.refilter(_reset_pairs(cache_pairs[source]), llm=client)
            result.recipe_name = recipe.name
        else:
            pipe = Pipeline(recipe, llm=client)
            result = pipe.run(input_path)
            cache_key = entry.get("cache_as")
            if cache_key:
                cache_pairs[cache_key] = list(result.raw_pairs or (list(result.pairs) + list(result.rejected)))
        save_result(result, exp_dir)
        from ..adapters.zhixun import export_zhixun

        export_zhixun(result.pairs, exp_dir / "zhixun.jsonl")
        metrics = compute_metrics(result)
        sft_report: dict[str, Any] = {}
        if not skip_sft:
            sft_report = evaluate_sft(
                result.pairs,
                heldout,
                exp_dir / "sft",
                llm=client,
                skip_sft=False,
            )
        row = {
            "id": exp_id,
            "purpose": entry.get("purpose", ""),
            "recipe": recipe.name,
            "recipe_snapshot": recipe_snapshot(recipe),
            "metrics": metrics,
            "sft": sft_report,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)

    payload = {
        "suite": suite.get("name"),
        "input": str(input_path),
        "llm": "fake" if isinstance(client, FakeLLM) else "live",
        "experiments": rows,
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _render_report(suite.get("name", "suite"), rows)
    (out / "report.md").write_text(report, encoding="utf-8")
    return payload
