from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.strategies import Strategy


@dataclass
class CostModel:
    fee_pct: float = 0.001       # 0.1% Binance spot taker
    slippage_pct: float = 0.0005  # 5 bps

    @property
    def per_turnover(self) -> float:
        """Cost charged per unit of |Δweight| (notional turnover fraction)."""
        return self.fee_pct + self.slippage_pct


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

    def report_text(self) -> str:
        m = self.metrics
        line = "─" * 50
        return "\n".join([
            line,
            f"  Total return:    {m['total_return']:+.2%}",
            f"  CAGR:            {m['cagr']:+.2%}",
            f"  Sharpe (net):    {m['sharpe']:.2f}",
            f"  Max drawdown:    {m['max_drawdown']:.2%}",
            f"  Calmar:          {m['calmar']:.2f}",
            f"  Win rate:        {m['win_rate']:.1%}",
            f"  Profit factor:   {m['profit_factor']:.2f}",
            f"  Trade count:     {m['trade_count']}",
            f"  Time in market:  {m['time_in_market']:.1%}",
            line,
        ])


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost: CostModel,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    raw_weights = strategy.generate_weights(prices).reindex(prices.index).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)

    # 1-bar lag: yesterday's signal applied to today's return
    target = raw_weights.shift(1).fillna(0.0)

    # Normalize so total |weight| <= 1 (prevent over-leverage from multi-asset signal)
    abs_sum = target.abs().sum(axis=1)
    scale = abs_sum.where(abs_sum > 1.0, 1.0)
    portfolio_weights = target.div(scale, axis=0)

    gross = (portfolio_weights * asset_returns).sum(axis=1)

    turnover = portfolio_weights.diff().abs().sum(axis=1).fillna(portfolio_weights.iloc[0].abs().sum())
    costs = turnover * cost.per_turnover

    net = gross - costs
    equity = (1.0 + net).cumprod() * initial_capital

    trades = _extract_trades(portfolio_weights, prices)
    metrics = _compute_metrics(net, equity, portfolio_weights, trades)

    return BacktestResult(
        equity_curve=equity,
        returns=net,
        weights=portfolio_weights,
        trades=trades,
        metrics=metrics,
    )


def _extract_trades(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in weights.columns:
        w = weights[symbol]
        delta = w.diff().fillna(w.iloc[0])
        nonzero = delta[delta != 0]
        for ts, d in nonzero.items():
            price = prices.loc[ts, symbol] if ts in prices.index and symbol in prices.columns else float("nan")
            rows.append({"time": ts, "symbol": symbol, "delta_weight": float(d), "price": float(price)})
    if not rows:
        return pd.DataFrame(columns=["time", "symbol", "delta_weight", "price"])
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict:
    if len(returns) < 2:
        return _zero_metrics()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    days = max((returns.index[-1] - returns.index[0]).days, 1)
    years = days / 365.25
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    periods_per_year = _infer_annualization(returns.index)
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    pos = returns[returns > 0]
    neg = returns[returns < 0]
    win_rate = float(len(pos) / len(returns)) if len(returns) else 0.0
    profit_factor = float(pos.sum() / abs(neg.sum())) if neg.sum() != 0 else float("inf")

    abs_w_sum = weights.abs().sum(axis=1)
    time_in_market = float((abs_w_sum > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": int(len(trades)),
        "time_in_market": time_in_market,
    }


def _zero_metrics() -> dict:
    return {
        "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
        "calmar": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
        "trade_count": 0, "time_in_market": 0.0,
    }


def _infer_annualization(index: pd.DatetimeIndex) -> int:
    """Return periods-per-year for Sharpe annualization. Crypto trades 365 days/yr."""
    if len(index) < 2:
        return 365
    try:
        median_delta = pd.Series(index).diff().dropna().median()
        seconds = float(median_delta.total_seconds())
    except Exception:
        return 365
    if seconds <= 0:
        return 365
    if seconds >= 86400:
        return 365
    if seconds >= 3600:
        return int(round(365 * 24 * 3600 / seconds))
    return int(round(365 * 24 * 3600 / seconds))
