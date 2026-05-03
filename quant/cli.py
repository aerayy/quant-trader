from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from quant.backtest import CostModel, run_backtest
from quant.config import load_config
from quant.data import DataStore, download as data_download
from quant.strategies import get_strategy

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="quant-trader — backtest + (later) paper/live trading platform.",
)
console = Console()


@app.command()
def download(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    symbols: str = typer.Option(None, "--symbols", "-s",
                                help="Comma-separated, overrides config"),
    interval: str = typer.Option(None, "--interval", "-i",
                                 help="e.g., 1d, 4h, 1h, 1m. Overrides config."),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD; overrides config"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD; overrides config"),
):
    """Download Binance historical klines into local Parquet cache."""
    cfg = load_config(config)
    syms = symbols.split(",") if symbols else cfg["data"]["symbols"]
    iv = interval or cfg["data"]["interval"]
    s = start or str(cfg["data"]["start_date"])
    e = end or str(cfg["data"]["end_date"])

    store = DataStore(cfg["data"]["cache_dir"])
    console.print(f"[bold]Downloading[/bold] {syms} interval={iv} {s} → {e}")
    summary = data_download(syms, iv, s, e, store)

    total = sum(summary.values())
    console.print(f"[green]✓[/green] Downloaded {total} new month-files. "
                  f"Cache: [cyan]{store.cache_dir}[/cyan]")


@app.command()
def backtest(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    strategy_name: str = typer.Option(None, "--strategy", help="Override config strategy"),
):
    """Run backtest with the configured strategy/symbols/period."""
    cfg = load_config(config)

    syms = cfg["data"]["symbols"]
    iv = cfg["data"]["interval"]
    s = str(cfg["data"]["start_date"])
    e = str(cfg["data"]["end_date"])

    store = DataStore(cfg["data"]["cache_dir"])

    closes: dict[str, pd.Series] = {}
    for sym in syms:
        df = store.load(sym, iv, s, e)
        if df.empty:
            console.print(f"[yellow]⚠[/yellow] {sym}: no data in range, skipping")
            continue
        closes[sym] = df["close"].astype(float)

    if not closes:
        console.print("[red]✗[/red] No usable data. Run `python -m quant download` first.")
        raise typer.Exit(code=1)

    prices = pd.DataFrame(closes).sort_index()
    prices = prices.ffill().dropna(how="all")

    name = strategy_name or cfg["strategy"]["name"]
    params = cfg["strategy"]["params"]
    strategy = get_strategy(name, params)

    cm = CostModel(
        fee_pct=float(cfg["backtest"]["fee_pct"]),
        slippage_pct=float(cfg["backtest"]["slippage_pct"]),
    )

    console.print(
        f"[bold]Backtest[/bold] strategy=[cyan]{name}[/cyan] "
        f"symbols={list(prices.columns)} period={s}→{e}"
    )

    result = run_backtest(
        prices, strategy, cm,
        initial_capital=float(cfg["backtest"]["initial_capital"]),
    )
    console.print(result.report_text())

    out_dir = Path(cfg["reports"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"{name}_{ts}"

    result.equity_curve.to_csv(f"{base}.equity.csv", header=["equity"])
    result.trades.to_csv(f"{base}.trades.csv", index=False)

    plot_saved = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        result.equity_curve.plot(ax=axes[0], color="#2563eb", lw=1.5)
        axes[0].set_title(f"{name} — equity curve  (initial {cfg['backtest']['initial_capital']:,.0f} USDT)")
        axes[0].set_ylabel("Equity (USDT)")
        axes[0].grid(True, alpha=0.3)

        cum_max = result.equity_curve.cummax()
        dd = (result.equity_curve - cum_max) / cum_max
        axes[1].fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.5)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel("Date")

        fig.tight_layout()
        fig.savefig(f"{base}.equity.png", dpi=120)
        plt.close(fig)
        plot_saved = True
    except ImportError:
        pass

    artifacts = [f"{base}.equity.csv", f"{base}.trades.csv"]
    if plot_saved:
        artifacts.append(f"{base}.equity.png")
    console.print(f"[green]✓[/green] Saved: {', '.join(Path(a).name for a in artifacts)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
