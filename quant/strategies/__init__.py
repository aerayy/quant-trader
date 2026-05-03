from __future__ import annotations

from typing import Any

from .base import Strategy, StrategyConfig
from .momentum import MomentumStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
}


def get_strategy(name: str, params: dict[str, Any]) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](StrategyConfig(name=name, params=params))


__all__ = ["Strategy", "StrategyConfig", "MomentumStrategy", "get_strategy", "REGISTRY"]
