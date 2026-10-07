"""YAML Recipe：只声明策略名与参数，不把算法写进编排器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class StrategySpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_obj(cls, value: Any) -> "StrategySpec":
        if value is None:
            raise ValueError("策略不能为空")
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, StrategySpec):
            return value
        data = dict(value)
        name = data.pop("name")
        return cls(name=name, params=data)


class Recipe(BaseModel):
    name: str = "default"
    description: str = ""
    chunking: StrategySpec
    anchor: StrategySpec
    question_gen: StrategySpec
    planner: StrategySpec = Field(default_factory=lambda: StrategySpec(name="none"))
    evolution: StrategySpec = Field(default_factory=lambda: StrategySpec(name="none"))
    question_filter: StrategySpec = Field(default_factory=lambda: StrategySpec(name="none"))
    distillation: StrategySpec = Field(default_factory=lambda: StrategySpec(name="concise_response"))
    teacher_router: StrategySpec = Field(default_factory=lambda: StrategySpec(name="single"))
    filters: list[StrategySpec] = Field(default_factory=list)
    grading: Literal["sab", "binary", "none", "validity_tier"] = "sab"
    questions_per_chunk: int = 0
    max_chunks: int | None = None
    max_samples: int | None = 200
    retries: int = 1
    goal: Literal["rag_grounded", "closed_book_domain"] = "rag_grounded"
    budget_cap_usd: float | None = None
    content_repair_max: int = 1
    escalation_max: int = 1
    split: Literal["train", "validation", "test"] = "train"
    seed: int = 42
    gap_fill_rounds: int = 0
    gap_fill_max_calls: int = 0
    teacher_models: dict[str, str] = Field(
        default_factory=lambda: {
            "default": "qwen3.8-27b",
            "cheap": "qwen3.8-27b",
            "strong": "qwen3.8-27b",
        }
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        raw = dict(data)
        for key in (
            "chunking",
            "anchor",
            "question_gen",
            "planner",
            "evolution",
            "question_filter",
            "distillation",
            "teacher_router",
        ):
            if key in raw:
                raw[key] = StrategySpec.from_obj(raw[key])
        raw["filters"] = [StrategySpec.from_obj(f) for f in raw.get("filters") or []]
        return cls.model_validate(raw)

    def dump(self) -> dict[str, Any]:
        def spec(s: StrategySpec) -> dict[str, Any]:
            return {"name": s.name, **s.params}

        return {
            "name": self.name,
            "description": self.description,
            "chunking": spec(self.chunking),
            "anchor": spec(self.anchor),
            "question_gen": spec(self.question_gen),
            "planner": spec(self.planner),
            "evolution": spec(self.evolution),
            "question_filter": spec(self.question_filter),
            "distillation": spec(self.distillation),
            "teacher_router": spec(self.teacher_router),
            "filters": [spec(f) for f in self.filters],
            "grading": self.grading,
            "questions_per_chunk": self.questions_per_chunk,
            "max_chunks": self.max_chunks,
            "max_samples": self.max_samples,
            "retries": self.retries,
            "goal": self.goal,
            "budget_cap_usd": self.budget_cap_usd,
            "content_repair_max": self.content_repair_max,
            "escalation_max": self.escalation_max,
            "split": self.split,
            "seed": self.seed,
            "gap_fill_rounds": self.gap_fill_rounds,
            "gap_fill_max_calls": self.gap_fill_max_calls,
            "teacher_models": self.teacher_models,
            "extra": self.extra,
        }


def load_recipe(path: str | Path) -> Recipe:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Recipe.from_dict(data)


def dump_recipe(recipe: Recipe, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(recipe.dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
