from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

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

    Returns a summary dict that the CLI prints. For paper trading, we ignore
    cfg.data.end_date and use all-cached data so each run sees the latest bars
    that bootstrap brought in.
    """
    syms = cfg["data"]["symbols"]
    iv = cfg["data"]["interval"]
    s = str(cfg["data"]["start_date"])
    e = "2099-12-31"  # use all data through cache end; paper trader is not period-bound

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


def run_live(cfg: dict, console=None, refresh_per_second: float = 1.0,
             poll_interval_seconds: int = 15) -> None:
    """WebSocket-driven paper trader with terminal dashboard + Telegram alerts.

    Lifecycle:
      1. REST-bootstrap recent klines into local cache.
      2. Initialize PaperState (sets initial cash on first run).
      3. Run signal once on bootstrapped data; alert any trades fired.
      4. Start Binance WebSocket (bar close + miniTicker).
      5. Start background poll thread that watches for new bar closes;
         when all symbols have a fresh bar, runs signal again.
      6. Run rich.live dashboard in main thread until Ctrl+C.
      7. On shutdown: stop WS, send "stopped" notification.

    No Binance API auth needed (public market data only). Telegram is
    optional — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to enable.
    """
    from .bootstrap import bootstrap_klines
    from .dashboard import Dashboard
    from .notifier import Notifier
    from .ws_client import BinanceWSClient

    name = cfg["strategy"]
    notifier = Notifier(console=console)
    db_path = _paper_db_path(cfg)

    if console:
        console.print("[bold]Step 1/4:[/bold] REST-bootstrap recent klines")
    bootstrap_klines(cfg, console=console)

    if console:
        console.print("[bold]Step 2/4:[/bold] initialize state + first signal")

    state = PaperState(db_path)
    initial = float(cfg["backtest"]["initial_capital"])
    if state.get_cash() == 0.0 and not state.get_positions():
        state.set_cash(initial)
        if console:
            console.print(f"[dim]First run: initialized state with "
                          f"{initial:.2f} USDT cash[/dim]")
    state.close()

    initial_result = run_once(cfg, console=console)
    if initial_result["executed"]:
        for t in initial_result["executed"]:
            notifier.trade(t["side"], t["symbol"], t["qty"], t["price"], t["notional"])
    notifier.signal(initial_result["target_weights"], strategy=name)

    last_processed_date = pd.Timestamp(initial_result["data_last_bar"]).date()

    if console:
        console.print("[bold]Step 3/4:[/bold] start Binance WebSocket")
    ws_client = BinanceWSClient(cfg["data"]["symbols"], cfg["data"]["interval"])
    ws_client.start()

    notifier.send(
        f"✅ Paper trader live\n"
        f"Strategy: `{name}`\n"
        f"Symbols: `{', '.join(cfg['data']['symbols'])}`\n"
        f"Equity: `{initial_result['equity']:,.2f}` USDT"
    )

    stop_flag = threading.Event()

    def poll_loop() -> None:
        nonlocal last_processed_date
        while not stop_flag.is_set():
            try:
                bars = ws_client.get_closed_bars()
                expected = len(ws_client.symbols)
                if len(bars) >= expected:
                    dates = [
                        pd.Timestamp(b["close_time"], unit="ms").date()
                        for b in bars.values()
                    ]
                    common = dates[0]
                    if all(d == common for d in dates) and common > last_processed_date:
                        try:
                            result = run_once(cfg, console=console)
                            for t in result["executed"]:
                                notifier.trade(
                                    t["side"], t["symbol"], t["qty"],
                                    t["price"], t["notional"],
                                )
                            notifier.signal(result["target_weights"], strategy=name)
                            last_processed_date = common
                        except Exception as exc:
                            notifier.error(f"Signal computation failed: {exc!r}")
            except Exception as exc:
                if console:
                    console.print(f"[yellow]poll loop error:[/yellow] {exc!r}")
            stop_flag.wait(poll_interval_seconds)

    poll_thread = threading.Thread(target=poll_loop, name="poll-loop", daemon=True)
    poll_thread.start()

    if console:
        console.print("[bold]Step 4/4:[/bold] dashboard (Ctrl+C to stop)")

    dashboard_state = PaperState(db_path)
    dashboard = Dashboard(
        dashboard_state, ws_client, strategy_name=name,
        refresh_per_second=refresh_per_second,
    )
    try:
        dashboard.run(console=console)
    finally:
        stop_flag.set()
        ws_client.stop()
        dashboard_state.close()
        notifier.send("👋 Paper trader stopped")
        if console:
            console.print("[yellow]Stopped[/yellow]")


def run_live_real(
    cfg: dict,
    console=None,
    testnet: bool = False,
    refresh_per_second: float = 1.0,
    poll_interval_seconds: int = 15,
) -> None:
    """REAL-MONEY live trading. Places actual Binance orders.

    Lifecycle (each step blocks on success):
      1. Connect to Binance with API key (testnet or mainnet).
      2. Pre-flight: verify TRADE perm enabled, WITHDRAWAL DISABLED, symbols
         tradable. Hard refuse if any critical issue.
      3. Fetch exchangeInfo (lot size, min notional) for configured symbols.
      4. REST-bootstrap recent klines into local cache.
      5. Reconcile local state with exchange (exchange = source of truth).
      6. Compute initial signal + place any orders the strategy requires.
      7. Start WebSocket, Telegram bot, background poll thread; run dashboard
         in main thread until Ctrl+C.

    Hard safety:
      - Risk manager checks daily loss + max DD + kill switch on every tick.
      - Order placement uses idempotent newClientOrderId.
      - Cash always re-fetched from exchange after trades (truth-source).
    """
    from binance.client import Client  # imported here to keep module-load light

    from .bootstrap import bootstrap_klines
    from .dashboard import Dashboard
    from .live_executor import LiveExecutor
    from .notifier import Notifier
    from .reconciler import Reconciler
    from .risk_manager import RiskManager
    from .safety import fetch_symbol_filters, preflight
    from .telegram_bot import TelegramBot
    from .ws_client import BinanceWSClient

    load_dotenv(override=False)
    name = cfg["strategy"]
    symbols = cfg["data"]["symbols"]
    notifier = Notifier(console=console)

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_API_KEY + BINANCE_API_SECRET required in .env for live mode. "
            "See .env.example for setup steps."
        )

    venue = "TESTNET" if testnet else "MAINNET"

    if console:
        console.print(f"[bold]Step 1/7:[/bold] connecting to Binance "
                      f"[{('yellow' if testnet else 'red bold')}]"
                      f"{venue}[/{('yellow' if testnet else 'red bold')}]")
    client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)

    if console:
        console.print("[bold]Step 2/7:[/bold] pre-flight checks "
                      "(API perms, withdrawal disabled, symbols tradable)")
    pf = preflight(client, symbols)
    if console:
        for issue in pf.issues:
            tag = "red" if not pf.ok else "yellow"
            console.print(f"  [{tag}]•[/{tag}] {issue}")
    if not pf.ok:
        notifier.error(f"Pre-flight FAILED on {venue}; refusing to start. "
                       f"Issues: {len(pf.issues)}")
        raise RuntimeError("Pre-flight failed; see terminal for issues to fix.")

    if console:
        console.print("[bold]Step 3/7:[/bold] fetch symbol filters (stepSize, minNotional)")
    symbol_filters = fetch_symbol_filters(client, symbols)
    if console:
        for sym, f in symbol_filters.items():
            console.print(f"  [dim]{sym}[/dim] step={f['stepSize']} "
                          f"minQty={f['minQty']} minNot={f['minNotional']}")

    if console:
        console.print("[bold]Step 4/7:[/bold] bootstrap recent klines from REST")
    bootstrap_klines(cfg, console=console)

    db_path = _paper_db_path(cfg)
    state = PaperState(db_path)

    iv = cfg["data"]["interval"]
    s = str(cfg["data"]["start_date"])
    e = "2099-12-31"
    store = DataStore(cfg["data"]["cache_dir"])
    prices = _build_prices(store, symbols, iv, s, e)
    current_prices_initial = {
        sym: float(prices[sym].iloc[-1]) for sym in prices.columns
    }

    if console:
        console.print("[bold]Step 5/7:[/bold] reconcile local state with exchange")
    reconciler = Reconciler(state, client, symbols, notifier=notifier)
    reconciler.initial_sync(current_prices_initial)

    risk_cfg = (cfg.get("live") or {}).get("risk") or {}
    risk = RiskManager(
        state,
        daily_loss_pct=float(risk_cfg.get("daily_loss_pct", 0.05)),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.15)),
        max_position_pct=float(risk_cfg.get("max_position_pct", 1.0)),
    )

    cash, pos_val, equity = _compute_equity(state, current_prices_initial)
    risk.update_equity_markers(equity)
    state.record_equity(_utc_now(), cash, pos_val, equity)
    state.close()

    if console:
        console.print("[bold]Step 6/7:[/bold] compute initial signal + place orders")
    initial_result = _run_real_iteration(cfg, client, symbol_filters, notifier, risk)
    notifier.signal(initial_result["target_weights"], strategy=name)

    if console:
        console.print("[bold]Step 7/7:[/bold] start WebSocket / Telegram / dashboard")

    ws_client = BinanceWSClient(symbols, cfg["data"]["interval"])
    ws_client.start()

    state_for_bot = PaperState(db_path)
    tg_bot = TelegramBot(state_for_bot, risk, notifier, client=client, symbols=symbols)
    if tg_bot.enabled:
        tg_bot.start()
        if console:
            console.print("[dim]Telegram bot active — try /status from your phone[/dim]")

    notifier.send(
        f"💰 *Live trader started* ({venue})\n"
        f"Strategy: `{name}`\n"
        f"Symbols: `{', '.join(symbols)}`\n"
        f"Equity: `{initial_result['equity']:,.2f}` USDT"
    )

    last_processed_date = pd.Timestamp(initial_result["data_last_bar"]).date()
    stop_flag = threading.Event()

    def poll_loop() -> None:
        nonlocal last_processed_date
        while not stop_flag.is_set():
            try:
                bars = ws_client.get_closed_bars()
                expected = len(ws_client.symbols)
                if len(bars) >= expected:
                    dates = [
                        pd.Timestamp(b["close_time"], unit="ms").date()
                        for b in bars.values()
                    ]
                    common = dates[0]
                    if all(d == common for d in dates) and common > last_processed_date:
                        try:
                            r = _run_real_iteration(cfg, client, symbol_filters, notifier, risk)
                            notifier.signal(r["target_weights"], strategy=name)
                            last_processed_date = common
                        except Exception as exc:
                            notifier.error(f"Iteration failed: {exc!r}")
            except Exception as exc:
                if console:
                    console.print(f"[yellow]poll error: {exc!r}[/yellow]")
            stop_flag.wait(poll_interval_seconds)

    poll_thread = threading.Thread(target=poll_loop, name="poll-loop-real", daemon=True)
    poll_thread.start()

    dashboard_state = PaperState(db_path)
    dashboard = Dashboard(
        dashboard_state, ws_client,
        strategy_name=f"{name} ({venue})",
        refresh_per_second=refresh_per_second,
    )
    try:
        dashboard.run(console=console)
    finally:
        stop_flag.set()
        ws_client.stop()
        tg_bot.stop()
        dashboard_state.close()
        state_for_bot.close()
        notifier.send("👋 Live trader stopped")
        if console:
            console.print("[yellow]Stopped[/yellow]")


def _run_real_iteration(cfg, client, symbol_filters, notifier, risk) -> dict:
    """One real-money iteration: prices → signal → live execution → state."""
    from .live_executor import LiveExecutor

    syms = cfg["data"]["symbols"]
    iv = cfg["data"]["interval"]
    s = str(cfg["data"]["start_date"])
    e = "2099-12-31"
    store = DataStore(cfg["data"]["cache_dir"])
    prices = _build_prices(store, syms, iv, s, e)

    name = cfg["strategy"]
    params = cfg["strategies"][name]
    overlay_specs = cfg.get("risk", {}).get(name, [])
    aux = _load_aux_data(name, list(prices.columns), store, s, e)

    db_path = _paper_db_path(cfg)
    state = PaperState(db_path)

    engine = SignalEngine(name, params, overlay_specs=overlay_specs, aux_data=aux)
    target_weights = engine.compute_target_weights(prices)

    current_prices = {sym: float(prices[sym].iloc[-1]) for sym in prices.columns}
    cash, pos_val, equity = _compute_equity(state, current_prices)

    executor = LiveExecutor(
        state, risk, client, symbol_filters, notifier=notifier,
        fee_pct=float(cfg["backtest"]["fee_pct"]),
    )
    ts = _utc_now()
    executed = executor.reconcile(
        target_weights, current_prices, equity, ts,
        reason=f"signal:{name}",
    )

    for sym, w in target_weights.items():
        state.record_signal(ts, sym, raw_weight=w, target_weight=w)

    cash, pos_val, equity = _compute_equity(state, current_prices)
    risk.update_equity_markers(equity)
    state.record_equity(ts, cash, pos_val, equity)
    state.close()

    return {
        "timestamp": ts.isoformat(),
        "data_last_bar": prices.index[-1].isoformat(),
        "target_weights": target_weights,
        "executed": executed,
        "cash": cash,
        "positions_value": pos_val,
        "equity": equity,
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
