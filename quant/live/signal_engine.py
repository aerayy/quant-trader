from __future__ import annotations

from typing import Any

import pandas as pd

from quant.risk import apply_overlays, build_overlays
from quant.strategies import get_strategy


class SignalEngine:
    """Wraps strategy + overlays for live use. Each call computes target
    portfolio weights based on the latest available price history.
    """

    def __init__(
        self,
        strategy_name: str,
        params: dict[str, Any],
        overlay_specs: list[dict] | None = None,
        aux_data: dict[str, Any] | None = None,
    ):
        self.strategy_name = strategy_name
        self.params = params
        self.overlay_specs = overlay_specs or []
        self.aux_data = aux_data or {}

    def compute_target_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        """Returns the latest row of weights as a dict (symbol → weight)."""
        strategy = get_strategy(self.strategy_name, self.params, **self.aux_data)
        weights = strategy.generate_weights(prices)
        if self.overlay_specs:
            overlays = build_overlays(self.overlay_specs)
            weights = apply_overlays(weights, prices, overlays)
        weights = weights.reindex(prices.index).fillna(0.0)
        last = weights.iloc[-1].to_dict()
        return {k: float(v) for k, v in last.items()}
