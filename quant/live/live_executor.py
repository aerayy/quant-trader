from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any

from .risk_manager import RiskManager
from .state import PaperState


class LiveExecutor:
    """Real Binance market-order executor. Same reconcile() interface as
    PaperExecutor so the runner doesn't care which one is in use.

    Order placement uses idempotent newClientOrderId built from timestamp
    so retried calls don't double-fire. Lot size and minNotional are
    enforced per-symbol from exchangeInfo (passed in via filters).
    """

    def __init__(
        self,
        state: PaperState,
        risk: RiskManager,
        client: Any,
        symbol_filters: dict[str, dict[str, float]],
        notifier: Any = None,
        fee_pct: float = 0.001,
    ):
        self.state = state
        self.risk = risk
        self.client = client
        self.symbol_filters = symbol_filters
        self.notifier = notifier
        self.fee_pct = fee_pct

    def reconcile(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        total_equity: float,
        ts: datetime,
        reason: str = "",
    ) -> list[dict]:
        """Same shape as PaperExecutor.reconcile but places real orders."""
        allowed, why = self.risk.allowed(total_equity)
        if not allowed:
            if self.notifier:
                self.notifier.error(f"Reconcile blocked: {why}")
            return []

        positions = self.state.get_positions()
        cash = self._cash_from_exchange()  # exchange is source of truth
        executed: list[dict] = []

        for sym, target_w in target_weights.items():
            price = current_prices.get(sym)
            if price is None or price <= 0:
                continue

            target_qty = target_w * total_equity / price
            current_qty = positions.get(sym, 0.0)
            delta_qty = target_qty - current_qty

            filters = self.symbol_filters.get(sym, {
                "stepSize": 0.000001, "minQty": 0.0,
                "tickSize": 0.01, "minNotional": 10.0,
            })
            min_notional = filters["minNotional"]

            est_notional = abs(delta_qty * price)
            if est_notional < min_notional:
                continue

            side = "BUY" if delta_qty > 0 else "SELL"
            qty_abs = self._round_qty(abs(delta_qty), filters["stepSize"])
            if qty_abs < filters["minQty"]:
                continue
            if qty_abs * price < min_notional:
                continue

            pre_ok, why = self.risk.check_trade(sym, side, delta_qty, est_notional, total_equity)
            if not pre_ok:
                if self.notifier:
                    self.notifier.error(f"Trade blocked: {why}")
                continue

            client_order_id = f"qt-{int(time.time())}-{sym[:6]}-{side[:1]}"
            try:
                order = self.client.create_order(
                    symbol=sym,
                    side=side,
                    type="MARKET",
                    quantity=self._format_qty(qty_abs, filters["stepSize"]),
                    newClientOrderId=client_order_id,
                )
            except Exception as e:
                if self.notifier:
                    self.notifier.error(f"Order failed for {sym} {side}: {e!r}")
                continue

            fills = order.get("fills") or []
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_quote = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                avg_price = (total_quote / total_qty) if total_qty > 0 else price
                # Commission may be in BNB or other; for simplicity use quote-equivalent
                fee_quote = sum(self._fee_quote(f) for f in fills)
            else:
                exec_qty = float(order.get("executedQty", 0.0))
                cum_quote = float(order.get("cummulativeQuoteQty", 0.0))
                total_qty = exec_qty
                avg_price = (cum_quote / exec_qty) if exec_qty > 0 else price
                fee_quote = total_qty * avg_price * self.fee_pct  # estimate

            fill_notional = total_qty * avg_price

            # Update local state — exchange is truth, so re-fetch cash too
            new_qty = current_qty + (total_qty if side == "BUY" else -total_qty)

            existing = self.state.get_position_detail(sym)
            if existing and current_qty != 0.0 and side == "BUY":
                old_qty, old_avg = existing
                if (current_qty > 0 and total_qty > 0) and new_qty > 0:
                    new_avg = (old_qty * old_avg + total_qty * avg_price) / new_qty
                else:
                    new_avg = avg_price
            elif abs(new_qty) < 1e-9:
                new_avg = 0.0
            else:
                new_avg = avg_price

            self.state.record_trade(
                ts, sym, side, total_qty, avg_price, fee_quote, fill_notional,
                f"{reason}|{client_order_id}",
            )
            self.state.update_position(sym, new_qty, new_avg, ts)

            executed.append({
                "symbol": sym, "side": side, "qty": float(total_qty),
                "price": float(avg_price), "fee": float(fee_quote),
                "notional": float(fill_notional),
                "client_order_id": client_order_id,
                "order_id": order.get("orderId"),
            })

            if self.notifier:
                self.notifier.trade(side, sym, total_qty, avg_price, fill_notional)

        # Refresh cash from exchange after trades settle
        try:
            cash_now = self._cash_from_exchange()
            self.state.set_cash(cash_now)
        except Exception:
            self.state.set_cash(cash)  # fallback

        return executed

    # --- helpers ---

    def _cash_from_exchange(self) -> float:
        try:
            balances = self.client.get_account()["balances"]
            usdt = next((b for b in balances if b["asset"] == "USDT"), None)
            if usdt is None:
                return 0.0
            return float(usdt["free"]) + float(usdt["locked"])
        except Exception:
            return self.state.get_cash()

    @staticmethod
    def _round_qty(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        return math.floor(qty / step) * step

    @staticmethod
    def _format_qty(qty: float, step: float) -> str:
        # Determine decimals from step size (e.g., 0.00001 → 5 decimals)
        if step >= 1:
            return f"{int(qty)}"
        decimals = max(0, int(round(-math.log10(step))))
        return f"{qty:.{decimals}f}"

    @staticmethod
    def _fee_quote(fill: dict) -> float:
        """Estimate fill fee in quote currency (USDT). If commissionAsset
        is the quote, use directly; otherwise approximate as qty*price*fee_pct.
        """
        try:
            commission = float(fill.get("commission", 0))
            asset = fill.get("commissionAsset", "")
            if asset == "USDT":
                return commission
            # Approximate using fill price (rough)
            qty = float(fill["qty"])
            price = float(fill["price"])
            return qty * price * 0.001  # 0.1% default fee equivalent
        except Exception:
            return 0.0
