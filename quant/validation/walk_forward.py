from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest import CostModel, compute_metrics, run_backtest
from quant.strategies import Strategy

from .oos import _count_trades_df


@dataclass
class WalkForwardResult:
    fold_metrics: list[dict]
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]]
    summary: dict


def run_walk_forward(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost: CostModel,
    initial_capital: float = 10_000.0,
    n_folds: int = 5,
) -> WalkForwardResult:
    """Run a single full backtest, then slice into N contiguous folds and
    compute fold-level metrics. This is k-fold-style time series validation,
    not true walk-forward optimization (the strategy has no per-fold tuning).
    """
    full = run_backtest(prices, strategy, cost, initial_capital)
    n = len(full.returns)
    if n < n_folds * 30:  # need at least 30 obs per fold
        n_folds = max(2, n // 30)

    fold_size = n // n_folds
    fold_metrics: list[dict] = []
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        rets = full.returns.iloc[start:end]
        wts = full.weights.iloc[start:end]
        if len(rets) < 2:
            continue
        eq = (1 + rets).cumprod() * initial_capital
        m = compute_metrics(rets, eq, wts, _count_trades_df(wts))
        fold_metrics.append(m)
        fold_dates.append((rets.index[0], rets.index[-1]))

    sharpes = [m["sharpe"] for m in fold_metrics]
    cagrs = [m["cagr"] for m in fold_metrics]

    summary = {
        "n_folds": len(fold_metrics),
        "sharpe_mean": float(np.mean(sharpes)) if sharpes else 0.0,
        "sharpe_std": float(np.std(sharpes)) if sharpes else 0.0,
        "sharpe_p25": float(np.percentile(sharpes, 25)) if sharpes else 0.0,
        "sharpe_p75": float(np.percentile(sharpes, 75)) if sharpes else 0.0,
        "sharpe_min": float(min(sharpes)) if sharpes else 0.0,
        "sharpe_max": float(max(sharpes)) if sharpes else 0.0,
        "cagr_mean": float(np.mean(cagrs)) if cagrs else 0.0,
        "positive_folds": int(sum(1 for s in sharpes if s > 0)),
        "total_folds": len(fold_metrics),
    }

    return WalkForwardResult(
        fold_metrics=fold_metrics,
        fold_dates=fold_dates,
        summary=summary,
    )
