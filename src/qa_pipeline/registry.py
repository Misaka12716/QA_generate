"""策略注册表：按阶段名 + 策略名查找可拔插实现。"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}
_LOADED = False


class Strategy(Protocol):
    name: str

    def __init__(self, **params: Any) -> None: ...

    def run(self, items: list[Any], ctx: Any) -> list[Any]: ...


def register(stage: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        _REGISTRY.setdefault(stage, {})[name] = cls
        cls.stage = stage  # type: ignore[attr-defined]
        cls.strategy_name = name  # type: ignore[attr-defined]
        if not getattr(cls, "name", None):
            cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_strategy(stage: str, name: str) -> type:
    ensure_plugins()
    try:
        return _REGISTRY[stage][name]
    except KeyError as exc:
        known = list_strategies(stage)
        raise KeyError(
            f"未知策略 {stage}/{name}。可用：{known or '(该阶段尚无策略)'}"
        ) from exc


def build_strategy(stage: str, spec: dict[str, Any] | str | None, **defaults: Any) -> Any:
    if spec is None:
        raise KeyError(f"{stage} 未配置策略")
    if isinstance(spec, str):
        name, params = spec, {}
    else:
        spec = dict(spec)
        name = spec.pop("name")
        params = spec
    cls = get_strategy(stage, name)
    merged = {**defaults, **params}
    return cls(**merged)


def list_strategies(stage: str | None = None) -> dict[str, list[str]] | list[str]:
    ensure_plugins()
    if stage:
        return sorted(_REGISTRY.get(stage, {}))
    return {s: sorted(names) for s, names in sorted(_REGISTRY.items())}


def ensure_plugins() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import plugins  # noqa: F401

    plugins.load_all()
    _LOADED = True
