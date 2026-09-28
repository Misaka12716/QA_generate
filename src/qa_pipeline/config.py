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
    evolution: StrategySpec = Field(default_factory=lambda: StrategySpec(name="none"))
    distillation: StrategySpec = Field(default_factory=lambda: StrategySpec(name="concise_response"))
    teacher_router: StrategySpec = Field(default_factory=lambda: StrategySpec(name="single"))
    filters: list[StrategySpec] = Field(default_factory=list)
    grading: Literal["sab", "binary", "none"] = "sab"
    questions_per_chunk: int = 4
    max_chunks: int | None = None
    max_samples: int | None = 200
    retries: int = 1
    split: Literal["train", "validation", "test"] = "train"
    seed: int = 42
    teacher_models: dict[str, str] = Field(
        default_factory=lambda: {
            "default": "deepseek-chat",
            "cheap": "deepseek-chat",
            "strong": "deepseek-chat",
        }
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        raw = dict(data)
        for key in ("chunking", "anchor", "question_gen", "evolution", "distillation", "teacher_router"):
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
            "evolution": spec(self.evolution),
            "distillation": spec(self.distillation),
            "teacher_router": spec(self.teacher_router),
            "filters": [spec(f) for f in self.filters],
            "grading": self.grading,
            "questions_per_chunk": self.questions_per_chunk,
            "max_chunks": self.max_chunks,
            "max_samples": self.max_samples,
            "retries": self.retries,
            "split": self.split,
            "seed": self.seed,
            "teacher_models": self.teacher_models,
        }


def load_recipe(path: str | Path) -> Recipe:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Recipe.from_dict(data)


def dump_recipe(recipe: Recipe, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(recipe.dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
