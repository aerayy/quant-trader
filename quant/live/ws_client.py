from __future__ import annotations

import threading
from typing import Any, Callable


class BinanceWSClient:
    """Thin wrapper around python-binance ThreadedWebsocketManager.

    Subscribes per symbol to:
    - kline_<interval> (bar close detection)
    - miniTicker (live mark price)

    State exposed via thread-safe getters. No auth needed for these public
    streams. Caller is expected to call start() and stop() symmetrically.
    """

    def __init__(self, symbols: list[str], interval: str = "1d"):
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self._twm = None  # constructed in start() to defer ThreadedWebsocketManager import
        self._lock = threading.Lock()

        self.latest_price: dict[str, float] = {}
        self.latest_closed_bar: dict[str, dict] = {}

        self.on_bar_close: Callable[[str, dict], None] | None = None
        self.on_price: Callable[[str, float], None] | None = None

        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from binance import ThreadedWebsocketManager
        self._twm = ThreadedWebsocketManager()
        self._twm.start()
        for sym in self.symbols:
            self._twm.start_kline_socket(
                symbol=sym, interval=self.interval, callback=self._handle_kline,
            )
            self._twm.start_symbol_miniticker_socket(
                symbol=sym, callback=self._handle_ticker,
            )
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._twm.stop()
        except Exception:
            pass
        self._started = False

    # --- Callbacks ---

    def _handle_kline(self, msg: dict[str, Any]) -> None:
        if msg.get("e") == "error":
            return
        if msg.get("e") != "kline":
            return
        k = msg.get("k", {})
        if not k.get("x"):  # bar still forming
            return
        sym = msg.get("s") or k.get("s")
        if not sym:
            return
        try:
            bar = {
                "open_time": int(k["t"]),
                "close_time": int(k["T"]),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
        except (KeyError, TypeError, ValueError):
            return
        with self._lock:
            self.latest_closed_bar[sym] = bar
            self.latest_price[sym] = bar["close"]
        cb = self.on_bar_close
        if cb is not None:
            try:
                cb(sym, bar)
            except Exception:
                pass

    def _handle_ticker(self, msg: dict[str, Any]) -> None:
        sym = msg.get("s")
        if not sym:
            return
        try:
            price = float(msg.get("c", 0))
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        with self._lock:
            self.latest_price[sym] = price
        cb = self.on_price
        if cb is not None:
            try:
                cb(sym, price)
            except Exception:
                pass

    # --- Public getters ---

    def get_prices(self) -> dict[str, float]:
        with self._lock:
            return dict(self.latest_price)

    def get_closed_bars(self) -> dict[str, dict]:
        with self._lock:
            return dict(self.latest_closed_bar)

    @property
    def is_running(self) -> bool:
        return self._started
