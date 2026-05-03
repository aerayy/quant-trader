from __future__ import annotations

from datetime import datetime

from .state import PaperState


class PaperExecutor:
    """Simulates market-order fills against current_prices using slippage
    and fee assumptions. Updates state.positions/cash and records trades.
    """

    def __init__(
        self,
        state: PaperState,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        min_notional: float = 10.0,
    ):
        self.state = state
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.min_notional = min_notional

    def reconcile(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        total_equity: float,
        ts: datetime,
        reason: str = "",
    ) -> list[dict]:
        """Bring positions in line with target weights. Returns list of
        executed trade dicts (one per symbol that crossed min_notional).
        """
        positions = self.state.get_positions()
        cash = self.state.get_cash()
        executed: list[dict] = []

        for sym, target_w in target_weights.items():
            price = current_prices.get(sym)
            if price is None or price <= 0:
                continue

            target_qty = target_w * total_equity / price
            current_qty = positions.get(sym, 0.0)
            delta_qty = target_qty - current_qty
            est_notional = abs(delta_qty * price)

            if est_notional < self.min_notional:
                continue

            side = "BUY" if delta_qty > 0 else "SELL"
            slip = 1.0 + self.slippage_pct if side == "BUY" else 1.0 - self.slippage_pct
            fill_price = price * slip
            fill_notional = abs(delta_qty * fill_price)
            fee = fill_notional * self.fee_pct

            new_qty = current_qty + delta_qty

            # Avg entry price update logic — same-direction adds blend, side
            # flips reset to fill_price, full closes reset to 0.
            existing = self.state.get_position_detail(sym)
            if existing and current_qty != 0.0:
                old_qty, old_avg = existing
                same_side_add = (current_qty > 0 and delta_qty > 0) or \
                                (current_qty < 0 and delta_qty < 0)
                if same_side_add and new_qty != 0.0:
                    new_avg = (old_qty * old_avg + delta_qty * fill_price) / new_qty
                elif abs(new_qty) < 1e-9:
                    new_avg = 0.0
                else:
                    new_avg = fill_price
            else:
                new_avg = fill_price if abs(new_qty) > 1e-9 else 0.0

            if side == "BUY":
                cash -= fill_notional + fee
            else:
                cash += fill_notional - fee

            self.state.record_trade(ts, sym, side, abs(delta_qty),
                                    fill_price, fee, fill_notional, reason)
            self.state.update_position(sym, new_qty, new_avg, ts)

            executed.append({
                "symbol": sym,
                "side": side,
                "qty": float(abs(delta_qty)),
                "price": float(fill_price),
                "fee": float(fee),
                "notional": float(fill_notional),
            })

        self.state.set_cash(cash)
        return executed
