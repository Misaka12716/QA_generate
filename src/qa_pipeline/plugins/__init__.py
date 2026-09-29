"""自动导入并注册全部策略模块。"""

from __future__ import annotations

from importlib import import_module

_MODULES = [
    "qa_pipeline.plugins.chunking",
    "qa_pipeline.plugins.anchors",
    "qa_pipeline.plugins.question_gen",
    "qa_pipeline.plugins.evolution",
    "qa_pipeline.plugins.question_filter",
    "qa_pipeline.plugins.routing",
    "qa_pipeline.plugins.distillation",
    "qa_pipeline.plugins.filters",
    "qa_pipeline.plugins.grading",
]


def load_all() -> None:
    for name in _MODULES:
        import_module(name)
