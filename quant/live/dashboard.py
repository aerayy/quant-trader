from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Dashboard:
    """rich.live terminal dashboard for the paper trader.

    Stacked panels: header (uptime/strategy), positions + P&L, recent trades,
    WS health + live prices.
    """

    def __init__(
        self,
        state: Any,
        ws_client: Any = None,
        strategy_name: str = "",
        refresh_per_second: float = 1.0,
    ):
        self.state = state
        self.ws_client = ws_client
        self.strategy_name = strategy_name
        self.refresh_per_second = refresh_per_second
        self._start = datetime.now(timezone.utc)
        self._last_signal_ts: str | None = None

    # --- Helpers ---

    def _uptime_str(self) -> str:
        elapsed = (datetime.now(timezone.utc) - self._start).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        if h:
            return f"{h}h{m:02d}m"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    # --- Panel builders ---

    def _header(self) -> Panel:
        ws_state = "[green]●[/green] connected" if (
            self.ws_client and self.ws_client.is_running
        ) else "[yellow]●[/yellow] not running"
        text = Text.from_markup(
            f"[bold cyan]quant-trader[/bold cyan] paper  "
            f"[dim]strategy=[/dim][magenta]{self.strategy_name}[/magenta]  "
            f"[dim]uptime=[/dim]{self._uptime_str()}  "
            f"[dim]ws=[/dim]{ws_state}"
        )
        return Panel(text, border_style="cyan", padding=(0, 1))

    def _positions(self) -> Panel:
        positions = self.state.get_positions()
        cash = self.state.get_cash()
        live_prices = self.ws_client.get_prices() if self.ws_client else {}

        table = Table(box=None, padding=(0, 1), show_header=True, header_style="dim")
        table.add_column("Symbol", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Entry", justify="right")
        table.add_column("Mark", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("P&L", justify="right")

        total_value = 0.0
        if positions:
            for sym, qty in positions.items():
                detail = self.state.get_position_detail(sym)
                avg = detail[1] if detail else 0.0
                mark = live_prices.get(sym, avg)
                value = qty * mark
                pnl = (mark - avg) * qty if avg else 0.0
                pnl_pct = (mark / avg - 1) * 100 if avg else 0.0
                pnl_text = (
                    f"[green]+{pnl:,.2f}[/green] [dim]({pnl_pct:+.2f}%)[/dim]"
                    if pnl >= 0
                    else f"[red]{pnl:,.2f}[/red] [dim]({pnl_pct:+.2f}%)[/dim]"
                )
                table.add_row(
                    sym,
                    f"{qty:.6f}",
                    f"{avg:,.2f}",
                    f"{mark:,.2f}",
                    f"{value:,.2f}",
                    pnl_text,
                )
                total_value += value
        else:
            table.add_row("[dim](flat)[/dim]", "", "", "", "", "")

        equity = cash + total_value
        footer = Text.from_markup(
            f"[bold]Cash:[/bold] {cash:,.2f}   "
            f"[bold]Positions:[/bold] {total_value:,.2f}   "
            f"[bold yellow]Equity:[/bold yellow] {equity:,.2f} USDT"
        )

        return Panel(Group(table, "", footer), title="Positions & P&L",
                     border_style="green", padding=(0, 1))

    def _trades(self) -> Panel:
        try:
            df = self.state.get_recent_trades(limit=8)
        except Exception:
            df = None
        if df is None or df.empty:
            return Panel(Text("(no trades yet)", style="dim"),
                         title="Recent Trades", border_style="blue", padding=(0, 1))

        table = Table(box=None, padding=(0, 1), show_header=True, header_style="dim")
        table.add_column("Time", style="dim")
        table.add_column("Side")
        table.add_column("Symbol", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Notional", justify="right")

        for _, r in df.iterrows():
            side = r["side"]
            color = "green" if side == "BUY" else "red"
            table.add_row(
                r["timestamp"][:19],
                f"[{color}]{side}[/{color}]",
                r["symbol"],
                f"{r['qty']:.6f}",
                f"{r['price']:,.2f}",
                f"{r['notional']:,.2f}",
            )
        return Panel(table, title="Recent Trades", border_style="blue", padding=(0, 1))

    def _live_prices(self) -> Panel:
        prices = self.ws_client.get_prices() if self.ws_client else {}
        symbols = self.ws_client.symbols if self.ws_client else []
        if not symbols:
            content = Text("WebSocket not running", style="yellow")
        else:
            lines = []
            for sym in symbols:
                p = prices.get(sym)
                lines.append(f"[bold]{sym}[/bold]  " + (
                    f"[white]{p:,.2f}[/white] USDT" if p else "[dim]--[/dim]"
                ))
            content = Text.from_markup("    ".join(lines))
        return Panel(content, title="Live Prices (WebSocket)",
                     border_style="magenta", padding=(0, 1))

    def render(self) -> Group:
        return Group(
            self._header(),
            self._positions(),
            self._trades(),
            self._live_prices(),
        )

    def run(self, console: Console | None = None) -> None:
        try:
            with Live(
                self.render(),
                refresh_per_second=self.refresh_per_second,
                screen=False,
                console=console,
            ) as live:
                while True:
                    time.sleep(1.0 / max(self.refresh_per_second, 0.1))
                    live.update(self.render())
        except KeyboardInterrupt:
            pass
