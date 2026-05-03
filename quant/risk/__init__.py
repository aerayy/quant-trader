from __future__ import annotations

from typing import Any

import pandas as pd

from .base import RiskOverlay
from .stop_loss import StopLossOverlay
from .trend_filter import TrendFilterOverlay
from .vol_target import VolTargetOverlay

REGISTRY: dict[str, type[RiskOverlay]] = {
    "vol_target": VolTargetOverlay,
    "trend_filter": TrendFilterOverlay,
    "stop_loss": StopLossOverlay,
}


def build_overlay(spec: dict[str, Any]) -> RiskOverlay:
    """Construct one overlay from a config dict.
    Spec format: {"type": "<name>", **params}
    """
    spec = dict(spec)
    name = spec.pop("type", None)
    if name is None:
        raise ValueError(f"Risk overlay spec missing 'type': {spec}")
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown risk overlay: {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](spec)


def build_overlays(specs: list[dict] | None) -> list[RiskOverlay]:
    if not specs:
        return []
    return [build_overlay(s) for s in specs]


def apply_overlays(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    overlays: list[RiskOverlay],
) -> pd.DataFrame:
    for ov in overlays:
        weights = ov.transform(weights, prices)
    return weights


__all__ = [
    "RiskOverlay",
    "VolTargetOverlay", "TrendFilterOverlay", "StopLossOverlay",
    "REGISTRY", "build_overlay", "build_overlays", "apply_overlays",
]
