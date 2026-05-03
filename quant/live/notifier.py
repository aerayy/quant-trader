from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


class Notifier:
    """Outbound-only notifications. If TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    are set in env (or .env), messages are sent to Telegram in addition to
    console. Otherwise console-only.

    No inbound bot polling here — that requires async/long-poll and is
    deferred to a later session. Setup steps for Telegram (one-time):
        1. Open Telegram, message @BotFather, /newbot, get token.
        2. Send your bot a /start message.
        3. curl https://api.telegram.org/bot<TOKEN>/getUpdates → find chat.id.
        4. Add to .env:
             TELEGRAM_BOT_TOKEN=<token>
             TELEGRAM_CHAT_ID=<chat_id>
    """

    def __init__(self, console: Any = None):
        load_dotenv(override=False)  # picks up .env if present
        self.console = console
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
        self.has_telegram = bool(self.token and self.chat_id)

    def _print(self, message: str, important: bool) -> None:
        if not self.console:
            print(message)
            return
        prefix = "[red bold]🔴[/red bold]" if important else "[cyan]ℹ[/cyan]"
        self.console.print(f"{prefix} {message}")

    def send(self, message: str, *, important: bool = False) -> None:
        self._print(message, important)
        if not self.has_telegram:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]Telegram send failed: {e!r}[/yellow]")

    def trade(
        self, side: str, symbol: str, qty: float, price: float, notional: float
    ) -> None:
        emoji = "🟢" if side == "BUY" else "🔴"
        self.send(
            f"{emoji} *{side}* `{symbol}`  qty=`{qty:.6f}`  "
            f"price=`{price:,.2f}`  notional=`{notional:,.2f}`"
        )

    def signal(self, weights: dict[str, float], strategy: str = "") -> None:
        suffix = f" ({strategy})" if strategy else ""
        lines = [f"📊 *Signal update*{suffix}"]
        for sym, w in weights.items():
            lines.append(f"  `{sym}`: {w:+.4f}")
        self.send("\n".join(lines))

    def daily_summary(
        self,
        equity: float,
        prev_equity: float,
        positions: dict[str, float],
        trade_count: int,
    ) -> None:
        delta = equity - prev_equity
        pct = (delta / prev_equity * 100.0) if prev_equity else 0.0
        emoji = "📈" if delta >= 0 else "📉"
        lines = [
            f"{emoji} *Daily summary*",
            f"Equity: `{equity:,.2f}` USDT",
            f"Δ today: `{delta:+,.2f}` (`{pct:+.2f}%`)",
            f"Trades: {trade_count}",
            "Positions:",
        ]
        if positions:
            for sym, qty in positions.items():
                lines.append(f"  `{sym}`: `{qty:.6f}`")
        else:
            lines.append("  (flat)")
        self.send("\n".join(lines))

    def error(self, message: str) -> None:
        self.send(f"⚠️ *Error*: {message}", important=True)

    def info(self, message: str) -> None:
        self.send(f"ℹ️ {message}")
