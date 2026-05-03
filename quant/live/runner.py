from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data import DataStore
from quant.strategies import AUX_DATA_KEYS

from .paper_executor import PaperExecutor
from .signal_engine import SignalEngine
from .state import PaperState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _paper_db_path(cfg: dict) -> Path:
    return Path(cfg["data"]["cache_dir"]).parent / "paper.db"


def _load_aux_data(strategy_name: str, symbols: list[str], store: DataStore,
                   start: str, end: str) -> dict:
    aux: dict[str, Any] = {}
    keys = AUX_DATA_KEYS.get(strategy_name, ())
    if "funding_data" in keys:
        funding: dict[str, pd.Series] = {}
        for sym in symbols:
            f = store.load_funding(sym, start, end)
            if not f.empty:
                funding[sym] = f
        aux["funding_data"] = funding
    return aux


def _build_prices(store: DataStore, symbols: list[str], interval: str,
                  start: str, end: str) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        df = store.load(sym, interval, start, end)
        if df.empty:
            continue
        closes[sym] = df["close"].astype(float)
    if not closes:
        raise RuntimeError(
            "No price data available. Run `python -m quant download` first."
        )
    return pd.DataFrame(closes).sort_index().ffill().dropna(how="all")


def _compute_equity(state: PaperState,
                    prices: dict[str, float]) -> tuple[float, float, float]:
    cash = state.get_cash()
    positions = state.get_positions()
    pos_value = sum(qty * prices.get(sym, 0.0) for sym, qty in positions.items())
    total = cash + pos_value
    return cash, pos_value, total


def run_once(cfg: dict, console=None) -> dict:
    """Single iteration: build prices → compute signal → reconcile → record.

    Returns a summary dict that the CLI prints.
    """
    syms = cfg["data"]["symbols"]
    iv = cfg["data"]["interval"]
    s = str(cfg["data"]["start_date"])
    e = str(cfg["data"]["end_date"])

    store = DataStore(cfg["data"]["cache_dir"])
    prices = _build_prices(store, syms, iv, s, e)

    name = cfg["strategy"]
    params = cfg["strategies"][name]
    overlay_specs = cfg.get("risk", {}).get(name, [])
    aux = _load_aux_data(name, list(prices.columns), store, s, e)

    db_path = _paper_db_path(cfg)
    state = PaperState(db_path)
    initial = float(cfg["backtest"]["initial_capital"])

    is_first_run = state.get_cash() == 0.0 and not state.get_positions()
    if is_first_run:
        state.set_cash(initial)
        if console:
            console.print(f"[dim]First run: initialized state with "
                          f"{initial:.2f} USDT cash[/dim]")

    engine = SignalEngine(name, params, overlay_specs=overlay_specs, aux_data=aux)
    target_weights = engine.compute_target_weights(prices)

    current_prices = {sym: float(prices[sym].iloc[-1]) for sym in prices.columns}
    cash_before, pos_val_before, equity_before = _compute_equity(state, current_prices)

    executor = PaperExecutor(
        state,
        fee_pct=float(cfg["backtest"]["fee_pct"]),
        slippage_pct=float(cfg["backtest"]["slippage_pct"]),
    )
    ts = _utc_now()
    executed = executor.reconcile(
        target_weights, current_prices, equity_before, ts,
        reason=f"signal:{name}",
    )

    for sym, w in target_weights.items():
        state.record_signal(ts, sym, raw_weight=w, target_weight=w)

    cash_after, pos_val_after, equity_after = _compute_equity(state, current_prices)
    state.record_equity(ts, cash_after, pos_val_after, equity_after)

    state.close()

    return {
        "timestamp": ts.isoformat(),
        "is_first_run": is_first_run,
        "data_last_bar": prices.index[-1].isoformat(),
        "target_weights": target_weights,
        "current_prices": current_prices,
        "executed": executed,
        "cash": cash_after,
        "positions_value": pos_val_after,
        "equity": equity_after,
    }


def run_daemon(cfg: dict, console=None,
               signal_interval_seconds: int = 86400) -> None:
    """Long-running loop. Computes signal once per interval (default 1d).
    Catches per-iteration exceptions and backs off; Ctrl+C exits cleanly.
    """
    while True:
        try:
            result = run_once(cfg, console=console)
            if console:
                msg = f"[green]✓[/green] iteration @ {result['timestamp']}"
                if result["executed"]:
                    msg += f" — {len(result['executed'])} trade(s)"
                else:
                    msg += " — no trades"
                console.print(msg)
            time.sleep(signal_interval_seconds)
        except KeyboardInterrupt:
            if console:
                console.print("\n[yellow]Stopped by user[/yellow]")
            break
        except Exception as exc:
            if console:
                console.print(f"[red]Iteration error:[/red] {exc!r}")
            time.sleep(60)
