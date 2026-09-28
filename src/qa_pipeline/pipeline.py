"""管线编排：按 Recipe 组装策略，串联 Chunk → Q → Evol → Distill → Filter → Grade。"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import Recipe, load_recipe
from .llm import FakeLLM, LLMClient
from .registry import build_strategy, ensure_plugins
from .schemas import Chunk, Document, QAPair, UsageStats
from .textutil import approx_tokens, heading_sections, normalize

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    recipe: Recipe
    llm: LLMClient
    rng: random.Random
    stats: UsageStats = field(default_factory=UsageStats)
    extras: dict[str, Any] = field(default_factory=dict)

    def model_for(self, tier: str = "default") -> str:
        models = self.recipe.teacher_models or {}
        return models.get(tier) or models.get("default") or self.llm.default_model

    def note_fallback(self, message: str) -> None:
        if message not in self.stats.fallbacks:
            self.stats.fallbacks.append(message)


@dataclass
class PipelineResult:
    recipe_name: str
    documents: list[Document]
    chunks: list[Chunk]
    questions: list[Question]
    pairs: list[QAPair]
    rejected: list[QAPair]
    stats: UsageStats
    raw_pairs: list[QAPair] = field(default_factory=list)

    @property
    def kept(self) -> list[QAPair]:
        return [p for p in self.pairs if p.action != "reject" and p.grade != "reject"]


class Pipeline:
    def __init__(self, recipe: Recipe, llm: LLMClient | None = None) -> None:
        ensure_plugins()
        self.recipe = recipe
        self.llm = llm or LLMClient()
        self.chunker = build_strategy("chunking", _spec(recipe.chunking))
        self.anchorer = build_strategy("anchor", _spec(recipe.anchor))
        self.qgen = build_strategy("question_gen", _spec(recipe.question_gen))
        self.evolver = build_strategy("evolution", _spec(recipe.evolution))
        self.router = build_strategy("teacher_router", _spec(recipe.teacher_router))
        self.distiller = build_strategy("distillation", _spec(recipe.distillation))
        self.filters = [build_strategy("filter", _spec(f)) for f in recipe.filters]
        self.grader = build_strategy("grading", {"name": recipe.grading})

    def run(self, documents: Iterable[Document] | Iterable[str] | str | Path) -> PipelineResult:
        docs = load_documents(documents)
        ctx = PipelineContext(
            recipe=self.recipe,
            llm=self.llm,
            rng=random.Random(self.recipe.seed),
        )
        t0 = time.perf_counter()

        chunks = self._timed("chunking", lambda: self.chunker.run(docs, ctx), ctx)
        if self.recipe.max_chunks:
            chunks = chunks[: self.recipe.max_chunks]
        ctx.stats.produced["chunks"] = len(chunks)

        anchored = self._timed("anchor", lambda: self.anchorer.run(chunks, ctx), ctx)
        questions = self._timed("question_gen", lambda: self.qgen.run(anchored, ctx), ctx)
        questions = self._timed("evolution", lambda: self.evolver.run(questions, ctx), ctx)
        if self.recipe.max_samples:
            questions = questions[: self.recipe.max_samples]
        ctx.stats.produced["questions"] = len(questions)

        routed = self._timed("teacher_router", lambda: self.router.run(questions, ctx), ctx)
        pairs = self._timed("distillation", lambda: self.distiller.run(routed, ctx), ctx)
        ctx.stats.produced["distilled"] = len(pairs)
        raw_pairs = [p.model_copy(deep=True) for p in pairs]

        pairs, ctx = self._apply_filters(pairs, ctx)
        pairs = self._timed("grading", lambda: self.grader.run(pairs, ctx), ctx)

        kept, rejected = [], []
        for p in pairs:
            p.split = self.recipe.split
            if p.action == "reject" or p.grade == "reject":
                rejected.append(p)
            else:
                kept.append(p)
        ctx.stats.produced["kept"] = len(kept)
        ctx.stats.produced["rejected"] = len(rejected)
        snap = self.llm.usage_snapshot()
        ctx.stats.llm_calls = snap["llm_calls"]
        ctx.stats.prompt_tokens = snap["prompt_tokens"]
        ctx.stats.completion_tokens = snap["completion_tokens"]
        ctx.stats.estimated_cost_usd = snap["estimated_cost_usd"]
        ctx.stats.stage_seconds["total"] = round(time.perf_counter() - t0, 3)
        return PipelineResult(
            recipe_name=self.recipe.name,
            documents=docs,
            chunks=chunks,
            questions=questions,
            pairs=kept,
            rejected=rejected,
            stats=ctx.stats,
            raw_pairs=raw_pairs,
        )

    def _apply_filters(self, pairs: list[QAPair], ctx: PipelineContext) -> tuple[list[QAPair], PipelineContext]:
        for filt in self.filters:
            before = len(pairs)
            name = getattr(filt, "name", type(filt).__name__)
            pairs = self._timed(f"filter:{name}", lambda f=filt, current=pairs: f.run(current, ctx), ctx)
            dropped = before - len(pairs)
            if dropped:
                ctx.stats.rejected[name] = ctx.stats.rejected.get(name, 0) + dropped
        return pairs, ctx

    def refilter(self, pairs: list[QAPair], llm: LLMClient | None = None) -> PipelineResult:
        ctx = PipelineContext(recipe=self.recipe, llm=llm or self.llm, rng=random.Random(self.recipe.seed))
        pairs, ctx = self._apply_filters([p.model_copy(deep=True) for p in pairs], ctx)
        pairs = self.grader.run(pairs, ctx)
        kept = [p for p in pairs if p.action != "reject" and p.grade != "reject"]
        rejected = [p for p in pairs if p not in kept]
        return PipelineResult(
            recipe_name=self.recipe.name,
            documents=[],
            chunks=[],
            questions=[],
            pairs=kept,
            rejected=rejected,
            stats=ctx.stats,
            raw_pairs=pairs,
        )

    def _timed(self, stage: str, fn, ctx: PipelineContext):
        start = time.perf_counter()
        out = fn()
        ctx.stats.stage_seconds[stage] = round(time.perf_counter() - start, 3)
        logger.info("stage %s -> %s items in %.2ss", stage, len(out) if hasattr(out, "__len__") else "?", ctx.stats.stage_seconds[stage])
        return out


def _spec(s) -> dict[str, Any]:
    return {"name": s.name, **s.params}


def load_documents(source: Iterable[Document] | Iterable[str] | str | Path) -> list[Document]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in {".md", ".txt", ".json"} and p.is_file()
            )
            return [document_from_file(p) for p in files]
        if path.is_file():
            return [document_from_file(path)]
        return [Document(title="inline", text=normalize(str(source)))]
    items = list(source)
    docs: list[Document] = []
    for item in items:
        if isinstance(item, Document):
            docs.append(item)
        elif isinstance(item, dict):
            docs.append(Document.model_validate(item))
        else:
            p = Path(str(item))
            docs.append(document_from_file(p) if p.exists() else Document(title="inline", text=normalize(str(item))))
    return docs


def document_from_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem
    sections = heading_sections(text)
    if sections and sections[0][0]:
        title = sections[0][0][0]
    return Document(
        path=str(path),
        title=title,
        text=normalize(text),
        metadata={"suffix": path.suffix, "bytes": path.stat().st_size},
    )


def run_pipeline(
    recipe: Recipe | str | Path,
    documents: Iterable[Document] | Iterable[str] | str | Path,
    llm: LLMClient | None = None,
    fake: bool = False,
) -> PipelineResult:
    if not isinstance(recipe, Recipe):
        recipe = load_recipe(recipe)
    client = llm or (FakeLLM() if fake else LLMClient())
    return Pipeline(recipe, llm=client).run(documents)


def attach_chunk_index(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def annotate_token_count(chunk: Chunk) -> Chunk:
    chunk.token_count = approx_tokens(chunk.text)
    return chunk
