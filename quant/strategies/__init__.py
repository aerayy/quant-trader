from __future__ import annotations

from typing import Any

from .base import Strategy, StrategyConfig
from .funding_mr import FundingMeanReversionStrategy
from .momentum import MomentumStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "funding_mr": FundingMeanReversionStrategy,
}

# Strategies that need auxiliary data beyond price klines must declare
# their kwargs here. CLI uses this to know what data to load.
AUX_DATA_KEYS: dict[str, tuple[str, ...]] = {
    "funding_mr": ("funding_data",),
}


def get_strategy(name: str, params: dict[str, Any], **kwargs: Any) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](StrategyConfig(name=name, params=params), **kwargs)


__all__ = [
    "Strategy", "StrategyConfig",
    "MomentumStrategy", "FundingMeanReversionStrategy",
    "get_strategy", "REGISTRY", "AUX_DATA_KEYS",
]
