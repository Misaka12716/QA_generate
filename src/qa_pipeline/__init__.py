"""可拔插 QA 数据生成、蒸馏与知识过滤管线。"""

from .config import Recipe, load_recipe
from .pipeline import Pipeline, PipelineResult, run_pipeline
from .registry import get_strategy, list_strategies, register

__version__ = "0.1.0"

__all__ = [
    "Recipe",
    "load_recipe",
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
    "register",
    "get_strategy",
    "list_strategies",
]
