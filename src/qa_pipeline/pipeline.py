"""管线编排：切分 → 生成 → 复用候选答案 → 验证 → 分层。未过门槛的样本不发布。"""

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
from .schemas import Chunk, Document, QAPair, Question, UsageStats
from .textutil import approx_tokens, balanced_take, heading_sections, normalize

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
        return [p for p in self.pairs if _publishable(p)]


class Pipeline:
    def __init__(self, recipe: Recipe, llm: LLMClient | None = None) -> None:
        ensure_plugins()
        self.recipe = recipe
        self.llm = llm or LLMClient()
        self.chunker = build_strategy("chunking", _spec(recipe.chunking))
        self.anchorer = build_strategy("anchor", _spec(recipe.anchor))
        self.qgen = build_strategy("question_gen", _spec(recipe.question_gen))
        self.evolver = build_strategy("evolution", _spec(recipe.evolution))
        self.question_filter = build_strategy("question_filter", _spec(recipe.question_filter))
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
        self._arm_budget()

        chunks = self._timed("chunking", lambda: self.chunker.run(docs, ctx), ctx)
        _stamp_source_group(docs, chunks)
        if self.recipe.max_chunks:
            chunks = balanced_take(
                chunks,
                self.recipe.max_chunks,
                lambda chunk: chunk.source_family_id or chunk.doc_id or chunk.source_doc,
                self.recipe.seed,
            )
        chunks = self._fit_budget(chunks, ctx)
        ctx.stats.produced["chunks"] = len(chunks)

        anchored = self._timed("anchor", lambda: self.anchorer.run(chunks, ctx), ctx)
        questions = self._timed("question_gen", lambda: self.qgen.run(anchored, ctx), ctx)
        questions = self._timed("evolution", lambda: self.evolver.run(questions, ctx), ctx)
        questions = self._timed("question_filter", lambda: self.question_filter.run(questions, ctx), ctx)
        if self.recipe.max_samples:
            questions = balanced_take(
                questions,
                self.recipe.max_samples,
                lambda question: question.metadata.get("source_doc") or question.chunk_id,
                self.recipe.seed,
            )
        ctx.stats.produced["questions"] = len(questions)
        return self._finish(docs, chunks, questions, ctx, t0)

    def distill_from_questions(self, questions: list[Question]) -> PipelineResult:
        """跳过切分与提问，只重跑路由、蒸馏、过滤和分层。"""
        ctx = PipelineContext(
            recipe=self.recipe,
            llm=self.llm,
            rng=random.Random(self.recipe.seed),
        )
        t0 = time.perf_counter()
        self._arm_budget()
        cloned = [q.model_copy(deep=True) for q in questions]
        if self.recipe.max_samples:
            cloned = balanced_take(
                cloned,
                self.recipe.max_samples,
                lambda question: question.metadata.get("source_doc") or question.chunk_id,
                self.recipe.seed,
            )
        ctx.stats.produced["questions"] = len(cloned)
        return self._finish([], [], cloned, ctx, t0)

    def _finish(
        self,
        docs: list[Document],
        chunks: list[Chunk],
        questions: list[Question],
        ctx: PipelineContext,
        t0: float,
    ) -> PipelineResult:
        snapshot = [q.model_copy(deep=True) for q in questions]
        routed = self._timed("teacher_router", lambda: self.router.run(questions, ctx), ctx)
        pairs = self._timed("distillation", lambda: self.distiller.run(routed, ctx), ctx)
        ctx.stats.produced["distilled"] = len(pairs)
        raw_pairs = [p.model_copy(deep=True) for p in pairs]

        pairs, ctx = self._apply_filters(pairs, ctx)
        pairs = self._timed("grading", lambda: self.grader.run(pairs, ctx), ctx)

        kept, rejected = [], []
        for p in pairs:
            p.split = self.recipe.split
            p.goal = self.recipe.goal
            if p.goal == "closed_book_domain":
                p.student_context_refs = []
            if _publishable(p) and p.grade in {"S", "A"} and p.grade != "B":
                p.data_stage = "accepted"
                kept.append(p)
            else:
                rejected.append(p)
        ctx.stats.produced["kept"] = len(kept)
        ctx.stats.produced["rejected"] = len(rejected)
        ctx.stats.stage_counts = _stage_counts(questions, pairs, kept)
        snap = self.llm.usage_snapshot()
        ctx.stats.llm_calls = snap["llm_calls"]
        ctx.stats.prompt_tokens = snap["prompt_tokens"]
        ctx.stats.completion_tokens = snap["completion_tokens"]
        ctx.stats.estimated_cost_usd = snap["estimated_cost_usd"]
        for reason in getattr(self.llm, "budget_stops", []):
            ctx.note_fallback(reason)
        ctx.stats.budget_stops = list(getattr(self.llm, "budget_stops", []))
        ctx.stats.ledger = _ledger(ctx.stats.estimated_cost_usd, self.recipe.budget_cap_usd, self.llm)
        ctx.stats.stage_seconds["total"] = round(time.perf_counter() - t0, 3)
        return PipelineResult(
            recipe_name=self.recipe.name,
            documents=docs,
            chunks=chunks,
            questions=snapshot,
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
            ctx.stats.funnel[name] = {"in": before, "out": len(pairs), "dropped": dropped}
            if dropped:
                ctx.stats.rejected[name] = ctx.stats.rejected.get(name, 0) + dropped
        return pairs, ctx

    def refilter(self, pairs: list[QAPair], llm: LLMClient | None = None) -> PipelineResult:
        self._arm_budget()
        ctx = PipelineContext(recipe=self.recipe, llm=llm or self.llm, rng=random.Random(self.recipe.seed))
        pairs, ctx = self._apply_filters([p.model_copy(deep=True) for p in pairs], ctx)
        pairs = self.grader.run(pairs, ctx)
        kept = []
        rejected = []
        for pair in pairs:
            if _publishable(pair):
                if pair.data_stage is None:
                    pair.data_stage = "accepted"
                kept.append(pair)
            else:
                rejected.append(pair)
        snap = self.llm.usage_snapshot()
        ctx.stats.llm_calls = snap["llm_calls"]
        ctx.stats.prompt_tokens = snap["prompt_tokens"]
        ctx.stats.completion_tokens = snap["completion_tokens"]
        ctx.stats.estimated_cost_usd = snap["estimated_cost_usd"]
        ctx.stats.stage_counts = _stage_counts([], pairs, kept)
        ctx.stats.ledger = _ledger(ctx.stats.estimated_cost_usd, self.recipe.budget_cap_usd, self.llm)
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

    def select_only(self, pairs: list[QAPair]) -> PipelineResult:
        """在已冻结的 accepted 池上只跑本配方的选择过滤器。"""
        self._arm_budget()
        ctx = PipelineContext(recipe=self.recipe, llm=self.llm, rng=random.Random(self.recipe.seed))
        current = [pair.model_copy(deep=True) for pair in pairs]
        for filt in self.filters:
            before = len(current)
            name = getattr(filt, "name", type(filt).__name__)
            current = filt.run(current, ctx)
            ctx.stats.funnel[name] = {"in": before, "out": len(current), "dropped": before - len(current)}
        current = self.grader.run(current, ctx)
        kept, rejected = [], []
        for pair in current:
            if (
                pair.included_in_this_run is False
                or pair.exclude_reason
                or pair.action in {"reject", "quarantine"}
                or not _publishable(pair)
            ):
                rejected.append(pair)
            else:
                pair.data_stage = "selected"
                pair.included_in_this_run = True
                kept.append(pair)
        ctx.stats.stage_counts = {"accepted": len(pairs), "selected": len(kept)}
        return PipelineResult(
            recipe_name=self.recipe.name,
            documents=[],
            chunks=[],
            questions=[],
            pairs=kept,
            rejected=rejected,
            stats=ctx.stats,
            raw_pairs=current,
        )

    def _arm_budget(self) -> None:
        self.llm.budget_cap_usd = self.recipe.budget_cap_usd
        extra = self.recipe.extra or {}
        self.llm.max_calls = extra.get("max_llm_calls")
        self.llm.max_prompt_tokens = extra.get("max_prompt_tokens")
        self.llm.budget_stops = []

    def _fit_budget(self, chunks: list, ctx: PipelineContext) -> list:
        """批次准入：额度不够覆盖生成和一次验证时缩小批次。cap 为 0 时不启动。"""
        cap = self.recipe.budget_cap_usd
        if cap is None:
            return chunks
        if float(cap) <= 0:
            self.llm.budget_stops.append("insufficient_for_required_steps")
            self.llm.budget_stops.append("budget_cap")
            return []
        return chunks

    def _timed(self, stage: str, fn, ctx: PipelineContext):
        start = time.perf_counter()
        out = fn()
        ctx.stats.stage_seconds[stage] = round(time.perf_counter() - start, 3)
        logger.info("stage %s -> %s items in %.2ss", stage, len(out) if hasattr(out, "__len__") else "?", ctx.stats.stage_seconds[stage])
        return out


def _ledger(spent: float, cap: float | None, llm: LLMClient | None = None) -> dict:
    calls = getattr(llm, "calls", 0) if llm else 0
    prompt = getattr(llm, "prompt_tokens", 0) if llm else 0
    completion = getattr(llm, "completion_tokens", 0) if llm else 0
    return {
        "spent_settled": round(float(spent), 6),
        "reserved_inflight": 0.0,
        "unbilled_reserve": 0.0,
        "required_process_reserve": 0.0,
        "safety_margin": 0.0,
        "authorized_cap": cap,
        "llm_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "sequence_tokens": prompt + completion,
        "supervised_tokens": None,
        "note": "美元、调用数、序列 token 与监督 token 分开记账",
    }


def _stage_counts(questions: list, pairs: list[QAPair], kept: list[QAPair]) -> dict:
    return {
        "candidate": len(questions) if questions else len(pairs),
        "format_valid": len(questions) if questions else None,
        "semantic_valid": len([item for item in questions if getattr(item, "verification_status", "") != "failed"]) if questions else None,
        "accepted": len(kept),
        "derived": len([pair for pair in pairs if pair.parent_sample_id]),
    }


def _publishable(pair: QAPair) -> bool:
    if pair.included_in_this_run is False:
        return False
    if pair.grade == "B":
        return False
    if pair.metadata.get("verification_status") == "pending" or pair.verification_status == "pending":
        return False
    return pair.action not in {"reject", "quarantine", "needs_escalation"} and pair.grade not in {"reject", "quarantine"}


def _stamp_source_group(docs: list[Document], chunks: list[Chunk]) -> None:
    groups = {doc.doc_id: doc.source_group or doc.doc_id for doc in docs}
    for chunk in chunks:
        chunk.metadata.setdefault("source_group", groups.get(chunk.doc_id, chunk.doc_id))


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
        source_group=path.stem,
        metadata={"suffix": path.suffix, "bytes": path.stat().st_size, "source_group": path.stem},
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
