from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.backtest import CostModel, compute_metrics, run_backtest
from quant.risk import RiskOverlay
from quant.strategies import Strategy


@dataclass
class OOSResult:
    is_metrics: dict
    oos_metrics: dict
    is_equity: pd.Series
    oos_equity: pd.Series
    train_pct: float
    split_date: pd.Timestamp

    def degradation(self, key: str) -> float:
        """Return (oos - is) / |is| for a metric. Negative = OOS worse than IS."""
        is_v = float(self.is_metrics.get(key, 0))
        oos_v = float(self.oos_metrics.get(key, 0))
        if is_v == 0:
            return 0.0
        return (oos_v - is_v) / abs(is_v)


def run_oos_split(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost: CostModel,
    initial_capital: float = 10_000.0,
    train_pct: float = 0.7,
    overlays: list[RiskOverlay] | None = None,
) -> OOSResult:
    """Run backtest on full price series, then split returns into IS/OOS at train_pct.

    This avoids cold-start bias from running backtests separately on each slice
    (signal lookback continues across the boundary).
    """
    full = run_backtest(prices, strategy, cost, initial_capital, overlays=overlays)

    n = len(full.returns)
    split = max(1, int(n * train_pct))

    is_returns = full.returns.iloc[:split]
    oos_returns = full.returns.iloc[split:]

    is_equity = (1 + is_returns).cumprod() * initial_capital
    oos_equity = (1 + oos_returns).cumprod() * initial_capital

    is_weights = full.weights.iloc[:split]
    oos_weights = full.weights.iloc[split:]

    # Lightweight trades count from weight changes
    is_trades = _count_trades_df(is_weights)
    oos_trades = _count_trades_df(oos_weights)

    is_metrics = compute_metrics(is_returns, is_equity, is_weights, is_trades)
    oos_metrics = compute_metrics(oos_returns, oos_equity, oos_weights, oos_trades)

    split_date = full.returns.index[split] if split < n else full.returns.index[-1]

    return OOSResult(
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        is_equity=is_equity,
        oos_equity=oos_equity,
        train_pct=train_pct,
        split_date=split_date,
    )


def _count_trades_df(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["time", "symbol", "delta_weight", "price"])
    rows = []
    for sym in weights.columns:
        delta = weights[sym].diff().fillna(weights[sym].iloc[0])
        for ts, d in delta[delta != 0].items():
            rows.append({"time": ts, "symbol": sym, "delta_weight": float(d), "price": float("nan")})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["time", "symbol", "delta_weight", "price"])
