from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv


class TelegramBot:
    """Minimal long-poll inbound Telegram bot, sync (no asyncio).

    Runs in a background thread. Authorizes only messages from TELEGRAM_CHAT_ID.
    Commands available:
        /status       Cash + positions + daily P&L
        /pnl          Last 7-day equity history
        /balance      Exchange balance vs local
        /pause        Stops trading via kill file (positions kept)
        /resume       Clears kill file
        /kill         Two-step: /kill then /confirm to flatten everything
        /help         List commands

    To enable: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env.
    """

    def __init__(
        self,
        state: Any,
        risk: Any,
        notifier: Any,
        client: Any | None = None,
        symbols: list[str] | None = None,
    ):
        load_dotenv(override=False)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
        self.state = state
        self.risk = risk
        self.notifier = notifier
        self.client = client
        self.symbols = symbols or []

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int | None = None
        self._kill_pending: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tg-bot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # --- Long poll loop ---

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                # Avoid busy looping on errors
                if self.notifier and self.notifier.console:
                    self.notifier.console.print(f"[yellow]tg bot error: {e!r}[/yellow]")
                self._stop.wait(10)

    def _tick(self) -> None:
        params = {"timeout": 25}
        if self._offset is not None:
            params["offset"] = self._offset
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{self.token}/getUpdates",
                params=params,
                timeout=30,
            )
        except Exception as e:
            self._stop.wait(5)
            return
        if r.status_code != 200:
            self._stop.wait(5)
            return
        data = r.json()
        if not data.get("ok"):
            self._stop.wait(5)
            return
        for upd in data.get("result", []):
            self._offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat = msg.get("chat", {})
            if str(chat.get("id")) != str(self.chat_id):
                continue  # unauthorized chat — silently ignore
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            self._handle_command(text)

    # --- Commands ---

    def _handle_command(self, text: str) -> None:
        cmd, *args = text.split()
        cmd = cmd.lower().split("@")[0]  # strip @BotName suffix if present

        if cmd == "/start" or cmd == "/help":
            self._reply_help()
        elif cmd == "/status":
            self._reply_status()
        elif cmd == "/pnl":
            self._reply_pnl()
        elif cmd == "/balance":
            self._reply_balance()
        elif cmd == "/pause":
            self.risk.request_kill(reason="paused via telegram")
            self._send("⏸️ Paused. Trading halted; positions kept. /resume to re-enable.")
        elif cmd == "/resume":
            self.risk.clear_kill()
            self._kill_pending = False
            self._send("▶️ Resumed.")
        elif cmd == "/kill":
            self._kill_pending = True
            self._send(
                "⚠️ *Confirm kill switch*\n"
                "This will halt the trader (positions stay open). "
                "Reply `/confirm` within 60s to proceed, or anything else to cancel."
            )
            # Auto-clear pending after 60s
            threading.Timer(60.0, self._clear_kill_pending).start()
        elif cmd == "/confirm":
            if self._kill_pending:
                self.risk.request_kill(reason="killed via telegram")
                self._kill_pending = False
                self._send(
                    "🔴 Kill confirmed. System will stop at next signal tick. "
                    "Positions remain open until you manually flatten."
                )
            else:
                self._send("Nothing pending. /kill first if you want to halt.")
        else:
            self._send(f"Unknown command: `{cmd}`. /help for options.")

    def _clear_kill_pending(self) -> None:
        if self._kill_pending:
            self._kill_pending = False

    # --- Reply builders ---

    def _reply_help(self) -> None:
        self._send(
            "*quant-trader bot*\n"
            "/status — positions + daily P&L\n"
            "/pnl — last 7d equity\n"
            "/balance — exchange vs local\n"
            "/pause — halt trading (positions stay)\n"
            "/resume — re-enable trading\n"
            "/kill — halt (two-step with /confirm)"
        )

    def _reply_status(self) -> None:
        cash = self.state.get_cash()
        positions = self.state.get_positions()
        day_start = self.risk.get_day_start_equity() if self.risk else None
        ath = self.risk.get_ath() if self.risk else None
        equity_now = cash + sum(
            qty * 0  # we don't have live prices here without WS state
            for sym, qty in positions.items()
        )
        # Just report cash + position notional from latest equity entry
        eq_hist = self.state.get_equity_history(limit=1)
        latest_eq = float(eq_hist["total_equity"].iloc[0]) if not eq_hist.empty else cash

        lines = [
            "*Status*",
            f"Cash: `{cash:,.2f}` USDT",
        ]
        if positions:
            lines.append("Positions:")
            for sym, qty in positions.items():
                detail = self.state.get_position_detail(sym)
                avg = detail[1] if detail else 0
                lines.append(f"  `{sym}`: qty=`{qty:.6f}` avg=`{avg:,.2f}`")
        else:
            lines.append("Positions: (flat)")
        lines.append(f"Equity (last tick): `{latest_eq:,.2f}` USDT")
        if day_start:
            delta = latest_eq - day_start
            pct = (delta / day_start * 100) if day_start else 0
            lines.append(f"Today: `{delta:+,.2f}` (`{pct:+.2f}%`)")
        if ath:
            dd = (latest_eq - ath) / ath * 100
            lines.append(f"From ATH: `{dd:+.2f}%`")
        kill, why = self.risk.kill_active() if self.risk else (False, "")
        if kill:
            lines.append(f"🔴 *KILL ACTIVE* — {why}")
        self._send("\n".join(lines))

    def _reply_pnl(self) -> None:
        df = self.state.get_equity_history(limit=10)
        if df.empty:
            self._send("No equity history yet.")
            return
        lines = ["*Equity history (latest first)*"]
        for _, r in df.iterrows():
            lines.append(f"`{r['timestamp'][:19]}`  `{r['total_equity']:>10,.2f}`")
        self._send("\n".join(lines))

    def _reply_balance(self) -> None:
        if self.client is None:
            self._send("Live client not attached (paper mode).")
            return
        try:
            data = self.client.get_account()
        except Exception as e:
            self._send(f"Balance fetch failed: `{e!r}`")
            return
        lines = ["*Exchange balances (>0)*"]
        for b in data.get("balances", []):
            free = float(b.get("free", 0))
            locked = float(b.get("locked", 0))
            qty = free + locked
            if qty > 0:
                lines.append(f"  `{b['asset']}`: free=`{free:.6f}` locked=`{locked:.6f}`")
        if len(lines) == 1:
            lines.append("  (none)")
        self._send("\n".join(lines))

    # --- Send wrapper ---

    def _send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception:
            pass
