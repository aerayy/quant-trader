from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .state import PaperState


# Kill switch lives in $HOME so user can `touch` it from any shell to stop.
DEFAULT_KILL_FILE = Path.home() / ".quant_kill"


class RiskManager:
    """Hard risk gates. Returns allowed=False with reason when any limit
    is breached, so callers can refuse to place trades or halt the system.

    Three gates:
      1. Kill file: if ~/.quant_kill exists, halt. User can create it via
         `touch ~/.quant_kill` from any shell, or via Telegram /kill.
      2. Daily loss: if today's equity drop ≥ daily_loss_pct, halt.
      3. Max drawdown: if equity drop from all-time-high ≥ max_drawdown_pct,
         halt; requires manual restart (more conservative than daily reset).

    State (start_of_day_equity, all_time_high) is persisted in PaperState's
    state_kv table so it survives restarts.
    """

    def __init__(
        self,
        state: PaperState,
        daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_position_pct: float = 1.0,
        kill_file: Path | str = DEFAULT_KILL_FILE,
    ):
        self.state = state
        self.daily_loss_pct = float(daily_loss_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.max_position_pct = float(max_position_pct)
        self.kill_file = Path(kill_file)
        self.last_block_reason: str = ""

    # --- Persisted markers ---

    def _get_kv(self, key: str) -> str | None:
        row = self.state.conn.execute(
            "SELECT value FROM state_kv WHERE key = ?", (key,),
        ).fetchone()
        return row[0] if row else None

    def _set_kv(self, key: str, value: str) -> None:
        with self.state.conn:
            self.state.conn.execute(
                "INSERT INTO state_kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # --- Daily / lifetime markers ---

    def update_equity_markers(self, total_equity: float) -> None:
        """Call once per signal tick. Updates all-time-high; rolls
        start-of-day on UTC date change.
        """
        # Lifetime high
        ath = self._get_kv("equity_ath")
        if ath is None or float(ath) < total_equity:
            self._set_kv("equity_ath", str(total_equity))
        # Daily start
        today = datetime.now(timezone.utc).date().isoformat()
        cur_day = self._get_kv("equity_day_date")
        if cur_day != today:
            self._set_kv("equity_day_date", today)
            self._set_kv("equity_day_start", str(total_equity))

    def get_day_start_equity(self) -> float | None:
        v = self._get_kv("equity_day_start")
        return float(v) if v is not None else None

    def get_ath(self) -> float | None:
        v = self._get_kv("equity_ath")
        return float(v) if v is not None else None

    # --- Gates ---

    def kill_active(self) -> tuple[bool, str]:
        if self.kill_file.exists():
            return True, f"Kill file present: {self.kill_file}"
        return False, ""

    def request_kill(self, reason: str = "user_request") -> None:
        self.kill_file.touch(exist_ok=True)
        self.kill_file.write_text(reason)

    def clear_kill(self) -> None:
        if self.kill_file.exists():
            self.kill_file.unlink()

    def allowed(self, total_equity: float) -> tuple[bool, str]:
        """Return (allowed, reason). Caller should NOT trade when allowed=False."""
        # 1. Kill file
        active, reason = self.kill_active()
        if active:
            self.last_block_reason = reason
            return False, reason

        # 2. Daily loss
        day_start = self.get_day_start_equity()
        if day_start is not None and day_start > 0:
            daily_change = (total_equity - day_start) / day_start
            if daily_change <= -self.daily_loss_pct:
                msg = (f"Daily loss limit breached: {daily_change:+.2%} "
                       f"(threshold -{self.daily_loss_pct:.0%})")
                self.last_block_reason = msg
                return False, msg

        # 3. Max drawdown from ATH
        ath = self.get_ath()
        if ath is not None and ath > 0:
            dd = (total_equity - ath) / ath
            if dd <= -self.max_drawdown_pct:
                msg = (f"Max drawdown breached: {dd:+.2%} from ATH "
                       f"{ath:,.2f} (threshold -{self.max_drawdown_pct:.0%})")
                self.last_block_reason = msg
                # Auto-arm kill so manual review is required to resume
                self.request_kill(reason=msg)
                return False, msg

        self.last_block_reason = ""
        return True, ""

    def check_trade(
        self,
        symbol: str,
        side: str,
        delta_qty: float,
        notional: float,
        total_equity: float,
    ) -> tuple[bool, str]:
        """Per-trade pre-check beyond the global allowed() gate."""
        if total_equity > 0 and self.max_position_pct < 1.0:
            if abs(notional) > self.max_position_pct * total_equity:
                msg = (f"Trade size {notional:,.2f} exceeds "
                       f"max_position_pct ({self.max_position_pct:.0%} of equity).")
                self.last_block_reason = msg
                return False, msg
        return True, ""
