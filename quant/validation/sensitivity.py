from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd
from tqdm import tqdm

from quant.backtest import CostModel, run_backtest
from quant.risk import RiskOverlay
from quant.strategies import get_strategy


def run_sensitivity(
    prices: pd.DataFrame,
    strategy_name: str,
    base_params: dict[str, Any],
    param_ranges: dict[str, list],
    cost: CostModel,
    initial_capital: float = 10_000.0,
    aux: dict[str, Any] | None = None,
    overlays: list[RiskOverlay] | None = None,
) -> pd.DataFrame:
    """Grid search over parameter ranges, returning a DataFrame with one
    row per combination including all backtest metrics.

    param_ranges keys must be 1 or 2 (otherwise the result becomes hard to
    visualize as a heatmap). aux is forwarded to get_strategy() so strategies
    that need auxiliary data (e.g. funding rates) get them on every iteration.
    """
    keys = list(param_ranges.keys())
    if not keys:
        raise ValueError("param_ranges must contain at least one parameter")
    if len(keys) > 2:
        raise ValueError("Only 1-2 parameters supported for sensitivity analysis")

    aux = aux or {}
    combos = list(product(*[param_ranges[k] for k in keys]))
    rows = []

    for combo in tqdm(combos, desc=f"sensitivity({','.join(keys)})"):
        params = dict(base_params)
        for k, v in zip(keys, combo):
            params[k] = v
        strategy = get_strategy(strategy_name, params, **aux)
        result = run_backtest(prices, strategy, cost, initial_capital, overlays=overlays)
        row = {k: v for k, v in zip(keys, combo)}
        row.update(result.metrics)
        rows.append(row)

    return pd.DataFrame(rows)
