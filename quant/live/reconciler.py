from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .state import PaperState


class Reconciler:
    """Source-of-truth sync: exchange ↔ local DB.

    On startup (and optionally periodically), fetches USDT cash and configured
    crypto balances from the exchange and writes them into PaperState. If a
    discrepancy is detected mid-run, alerts the user.

    Caveat: avg_entry_price is unknown for pre-existing exchange positions.
    For a fresh start, avg_entry defaults to current mark; subsequent trades
    correctly blend.
    """

    def __init__(
        self,
        state: PaperState,
        client: Any,
        symbols: list[str],
        notifier: Any = None,
    ):
        self.state = state
        self.client = client
        self.symbols = symbols
        self.notifier = notifier

    def initial_sync(self, current_prices: dict[str, float]) -> dict:
        """Pull exchange balances → local. Returns summary."""
        balances = self._fetch_balances()
        # USDT cash
        usdt = balances.get("USDT", 0.0)
        self.state.set_cash(usdt)

        # Crypto positions
        ts = datetime.now(timezone.utc).replace(microsecond=0)
        synced: dict[str, float] = {}
        for sym in self.symbols:
            base = sym.replace("USDT", "")  # naive but fine for *USDT pairs
            qty = balances.get(base, 0.0)
            if qty > 0:
                price = current_prices.get(sym, 0.0)
                # Preserve existing avg_entry if local already has same qty
                local = self.state.get_position_detail(sym)
                if local and abs(local[0] - qty) < 1e-9 and local[1] > 0:
                    avg = local[1]
                else:
                    avg = price if price > 0 else 0.0
                self.state.update_position(sym, qty, avg, ts)
                synced[sym] = qty
            else:
                # Make sure local doesn't have phantom
                local = self.state.get_position_detail(sym)
                if local and abs(local[0]) > 1e-9:
                    self.state.update_position(sym, 0.0, 0.0, ts)

        if self.notifier:
            lines = [f"🔄 *Initial sync*\nCash: `{usdt:,.2f}` USDT"]
            if synced:
                lines.append("Positions:")
                for sym, q in synced.items():
                    lines.append(f"  `{sym}`: `{q:.6f}`")
            else:
                lines.append("Positions: (flat)")
            self.notifier.send("\n".join(lines))

        return {"cash": usdt, "positions": synced}

    def detect_drift(
        self,
        current_prices: dict[str, float],
        threshold_pct: float = 0.01,
    ) -> list[str]:
        """Compare exchange balances vs local; return list of drift descriptions
        (asset name + delta) where delta exceeds threshold_pct of equity.
        """
        balances = self._fetch_balances()
        drifts: list[str] = []

        local_positions = self.state.get_positions()
        local_cash = self.state.get_cash()
        equity_estimate = local_cash + sum(
            q * current_prices.get(s, 0.0) for s, q in local_positions.items()
        )
        if equity_estimate <= 0:
            return drifts

        # Cash drift
        ex_cash = balances.get("USDT", 0.0)
        cash_diff = abs(ex_cash - local_cash)
        if cash_diff / equity_estimate > threshold_pct:
            drifts.append(f"USDT cash: local {local_cash:,.2f} vs exchange {ex_cash:,.2f}")

        # Position drift
        for sym in self.symbols:
            base = sym.replace("USDT", "")
            ex_qty = balances.get(base, 0.0)
            local_qty = local_positions.get(sym, 0.0)
            price = current_prices.get(sym, 0.0)
            if abs(ex_qty - local_qty) * price / equity_estimate > threshold_pct:
                drifts.append(f"{sym}: local {local_qty:.6f} vs exchange {ex_qty:.6f}")

        return drifts

    def _fetch_balances(self) -> dict[str, float]:
        try:
            data = self.client.get_account()
        except Exception as e:
            if self.notifier:
                self.notifier.error(f"Reconciler fetch failed: {e!r}")
            return {}
        out: dict[str, float] = {}
        for b in data.get("balances", []):
            free = float(b.get("free", 0))
            locked = float(b.get("locked", 0))
            qty = free + locked
            if qty > 0:
                out[b["asset"]] = qty
        return out
