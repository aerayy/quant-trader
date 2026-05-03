from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from quant.backtest import CostModel, run_backtest
from quant.config import load_config
from quant.data import DataStore, download as data_download, download_funding
from quant.risk import build_overlays
from quant.strategies import AUX_DATA_KEYS, get_strategy
from quant.validation import (
    assess,
    run_oos_split,
    run_sensitivity,
    run_walk_forward,
)

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
    data_type: str = typer.Option("all", "--type", "-t",
                                  help="klines | funding | all"),
):
    """Download Binance historical data into local Parquet cache.

    Spot klines from /spot/monthly/klines, funding rates from
    /futures/um/monthly/fundingRate.
    """
    cfg = load_config(config)
    syms = symbols.split(",") if symbols else cfg["data"]["symbols"]
    iv = interval or cfg["data"]["interval"]
    s = start or str(cfg["data"]["start_date"])
    e = end or str(cfg["data"]["end_date"])

    if data_type not in ("all", "klines", "funding"):
        console.print(f"[red]✗[/red] Invalid --type {data_type!r}; use klines|funding|all")
        raise typer.Exit(code=1)

    store = DataStore(cfg["data"]["cache_dir"])
    total = 0

    if data_type in ("all", "klines"):
        console.print(f"[bold]Klines[/bold] {syms} interval={iv} {s} → {e}")
        kline_summary = data_download(syms, iv, s, e, store)
        total += sum(kline_summary.values())

    if data_type in ("all", "funding"):
        console.print(f"[bold]Funding[/bold] {syms} {s} → {e}")
        funding_summary = download_funding(syms, s, e, store)
        total += sum(funding_summary.values())

    console.print(f"[green]✓[/green] Downloaded {total} new month-files. "
                  f"Cache: [cyan]{store.cache_dir}[/cyan]")


def _load_aux_data(strategy_name: str, symbols: list[str],
                   store: DataStore, start: str, end: str) -> dict:
    """Load auxiliary data (e.g., funding rates) for strategies that need it.
    Strategies declare their needed kwargs in strategies.AUX_DATA_KEYS.
    """
    aux: dict = {}
    keys = AUX_DATA_KEYS.get(strategy_name, ())

    if "funding_data" in keys:
        funding = {}
        missing = []
        for sym in symbols:
            f = store.load_funding(sym, start, end)
            if f.empty:
                missing.append(sym)
            else:
                funding[sym] = f
        if missing:
            console.print(f"[yellow]⚠[/yellow] Funding missing for {missing}; "
                          f"run `python -m quant download --type funding`")
        aux["funding_data"] = funding

    return aux


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

    name = strategy_name or cfg["strategy"]
    params = cfg["strategies"][name]
    aux = _load_aux_data(name, list(prices.columns), store, s, e)
    strategy = get_strategy(name, params, **aux)

    risk_specs = cfg.get("risk", {}).get(name, [])
    overlays = build_overlays(risk_specs)

    cm = CostModel(
        fee_pct=float(cfg["backtest"]["fee_pct"]),
        slippage_pct=float(cfg["backtest"]["slippage_pct"]),
    )

    risk_label = f" + risk[{', '.join(o['type'] for o in risk_specs)}]" if risk_specs else ""
    console.print(
        f"[bold]Backtest[/bold] strategy=[cyan]{name}[/cyan]{risk_label} "
        f"symbols={list(prices.columns)} period={s}→{e}"
    )

    result = run_backtest(
        prices, strategy, cm,
        initial_capital=float(cfg["backtest"]["initial_capital"]),
        overlays=overlays,
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


DEFAULT_SENSITIVITY_RANGES = {
    "momentum": {
        "lookback_days": [30, 45, 60, 75, 90, 120],
        "rebalance_days": [1, 3, 5, 10, 20],
    },
    "funding_mr": {
        "z_long_threshold": [-2.5, -2.0, -1.5, -1.0, -0.5],
        "z_window": [14, 21, 30, 60],
    },
}


@app.command()
def validate(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    strategy_name: str = typer.Option(None, "--strategy"),
    train_pct: float = typer.Option(0.7, "--train-pct",
                                    help="In-sample fraction for OOS split"),
    n_folds: int = typer.Option(5, "--folds",
                                help="Number of walk-forward folds"),
):
    """Run full validation: OOS split + walk-forward + sensitivity → verdict."""
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
            continue
        closes[sym] = df["close"].astype(float)
    if not closes:
        console.print("[red]✗[/red] No data; run `python -m quant download` first")
        raise typer.Exit(code=1)

    prices = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")

    name = strategy_name or cfg["strategy"]
    base_params = cfg["strategies"][name]
    aux = _load_aux_data(name, list(prices.columns), store, s, e)
    strategy = get_strategy(name, base_params, **aux)

    risk_specs = cfg.get("risk", {}).get(name, [])
    overlays = build_overlays(risk_specs)

    cm = CostModel(
        fee_pct=float(cfg["backtest"]["fee_pct"]),
        slippage_pct=float(cfg["backtest"]["slippage_pct"]),
    )
    capital = float(cfg["backtest"]["initial_capital"])

    if name not in DEFAULT_SENSITIVITY_RANGES:
        console.print(f"[red]✗[/red] No sensitivity grid for strategy {name!r}; "
                      f"available: {list(DEFAULT_SENSITIVITY_RANGES)}")
        raise typer.Exit(code=1)
    sens_ranges = DEFAULT_SENSITIVITY_RANGES[name]
    grid_size = 1
    for v in sens_ranges.values():
        grid_size *= len(v)

    risk_label = f" + risk[{', '.join(o['type'] for o in risk_specs)}]" if risk_specs else ""
    console.print(f"[bold]Validating[/bold] strategy=[cyan]{name}[/cyan]{risk_label} "
                  f"symbols={list(prices.columns)} period={s}→{e}")

    console.print(f"[dim]Step 1/3:[/dim] OOS split (train_pct={train_pct})")
    oos = run_oos_split(prices, strategy, cm, capital, train_pct=train_pct, overlays=overlays)

    console.print(f"[dim]Step 2/3:[/dim] Walk-forward ({n_folds} folds)")
    wf = run_walk_forward(prices, strategy, cm, capital, n_folds=n_folds, overlays=overlays)

    console.print(f"[dim]Step 3/3:[/dim] Sensitivity grid ({grid_size} combos)")
    sens = run_sensitivity(prices, name, base_params, sens_ranges, cm, capital,
                           aux=aux, overlays=overlays)

    verdict = assess(oos, wf, sens)

    console.print()
    console.print("[bold cyan]OOS Split[/bold cyan]")
    console.print(f"  IS  ({oos.train_pct:.0%}, → {oos.split_date.date()}): "
                  f"Sharpe {oos.is_metrics['sharpe']:+.2f}  "
                  f"CAGR {oos.is_metrics['cagr']:+.2%}  "
                  f"MaxDD {oos.is_metrics['max_drawdown']:.2%}")
    console.print(f"  OOS ({1 - oos.train_pct:.0%}, {oos.split_date.date()} →): "
                  f"Sharpe {oos.oos_metrics['sharpe']:+.2f}  "
                  f"CAGR {oos.oos_metrics['cagr']:+.2%}  "
                  f"MaxDD {oos.oos_metrics['max_drawdown']:.2%}")

    console.print()
    console.print("[bold cyan]Walk-Forward[/bold cyan]")
    for i, (m, (st, en)) in enumerate(zip(wf.fold_metrics, wf.fold_dates), 1):
        console.print(f"  Fold {i} [{st.date()} → {en.date()}]: "
                      f"Sharpe {m['sharpe']:+.2f}  CAGR {m['cagr']:+.2%}  "
                      f"MaxDD {m['max_drawdown']:.2%}")

    console.print()
    console.print("[bold cyan]Sensitivity — top 5 by Sharpe[/bold cyan]")
    top = sens.nlargest(5, "sharpe")
    for _, r in top.iterrows():
        params_str = "  ".join(f"{k}={int(r[k])}" for k in sens_ranges)
        console.print(f"  {params_str}  →  Sharpe {r['sharpe']:+.2f}  "
                      f"MaxDD {r['max_drawdown']:.2%}")

    console.print()
    console.print(verdict.report_text())

    out_dir = Path(cfg["reports"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"validate_{name}_{ts}"

    sens.to_csv(f"{base}.sensitivity.csv", index=False)
    _write_validation_report(
        f"{base}.md", name, oos, wf, sens, verdict,
        list(prices.columns), prices.index[0], prices.index[-1],
        base_params, train_pct,
    )

    saved = [f"{base.name}.md", f"{base.name}.sensitivity.csv"]
    if _save_sensitivity_heatmap(sens, sens_ranges, f"{base}.heatmap.png", name):
        saved.append(f"{base.name}.heatmap.png")
    if _save_oos_plot(oos, f"{base}.oos.png", name):
        saved.append(f"{base.name}.oos.png")
    if _save_walk_forward_plot(wf, f"{base}.walkforward.png", name):
        saved.append(f"{base.name}.walkforward.png")

    console.print(f"[green]✓[/green] Saved: {', '.join(saved)}")


def _write_validation_report(path, name, oos, wf, sens, verdict,
                             symbols, start, end, base_params, train_pct):
    lines = [
        f"# Validation Report — `{name}`",
        "",
        f"- Symbols: {', '.join(symbols)}",
        f"- Period: {start.date()} → {end.date()}",
        f"- Base params: `{base_params}`",
        "",
        f"## Verdict: **{verdict.decision}** (score {verdict.score}/3)",
        "",
        *verdict.notes,
        "",
        "## OOS Split",
        "",
        f"Train pct: {train_pct:.0%}; split date: {oos.split_date.date()}",
        "",
        "| Metric | In-sample | Out-of-sample |",
        "|---|---|---|",
        f"| Sharpe | {oos.is_metrics['sharpe']:+.2f} | {oos.oos_metrics['sharpe']:+.2f} |",
        f"| CAGR | {oos.is_metrics['cagr']:+.2%} | {oos.oos_metrics['cagr']:+.2%} |",
        f"| Max DD | {oos.is_metrics['max_drawdown']:.2%} | {oos.oos_metrics['max_drawdown']:.2%} |",
        f"| Win rate | {oos.is_metrics['win_rate']:.1%} | {oos.oos_metrics['win_rate']:.1%} |",
        f"| Profit factor | {oos.is_metrics['profit_factor']:.2f} | {oos.oos_metrics['profit_factor']:.2f} |",
        "",
        "## Walk-Forward",
        "",
        f"Folds: {wf.summary['n_folds']} · "
        f"Positive: {wf.summary['positive_folds']}/{wf.summary['total_folds']} · "
        f"Sharpe μ={wf.summary['sharpe_mean']:.2f} σ={wf.summary['sharpe_std']:.2f}",
        "",
        "| Fold | Period | Sharpe | CAGR | Max DD |",
        "|---|---|---|---|---|",
    ]
    for i, (m, (st, en)) in enumerate(zip(wf.fold_metrics, wf.fold_dates), 1):
        lines.append(f"| {i} | {st.date()} → {en.date()} | "
                     f"{m['sharpe']:+.2f} | {m['cagr']:+.2%} | {m['max_drawdown']:.2%} |")
    lines += [
        "",
        "## Sensitivity",
        "",
        f"Grid: {len(sens)} combos · "
        f"Sharpe mean={sens['sharpe'].mean():.2f} std={sens['sharpe'].std():.2f} "
        f"range=[{sens['sharpe'].min():.2f}, {sens['sharpe'].max():.2f}]",
        "",
        "Top 5 by Sharpe:",
        "",
    ]
    sens_keys = [c for c in sens.columns if c not in
                 ("total_return", "cagr", "sharpe", "max_drawdown", "calmar",
                  "win_rate", "profit_factor", "trade_count", "time_in_market")]
    header = "| " + " | ".join(sens_keys + ["Sharpe", "CAGR", "Max DD"]) + " |"
    sep = "|" + "|".join(["---"] * (len(sens_keys) + 3)) + "|"
    lines += [header, sep]
    for _, r in sens.nlargest(5, "sharpe").iterrows():
        row = ["{}".format(int(r[k]) if isinstance(r[k], (int, float)) and float(r[k]).is_integer() else r[k]) for k in sens_keys]
        row += [f"{r['sharpe']:+.2f}", f"{r['cagr']:+.2%}", f"{r['max_drawdown']:.2%}"]
        lines.append("| " + " | ".join(row) + " |")

    Path(path).write_text("\n".join(lines))


def _save_sensitivity_heatmap(sens, ranges, path, name) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    keys = list(ranges.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(keys) == 2:
        pivot = sens.pivot(index=keys[1], columns=keys[0], values="sharpe")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=8)
        plt.colorbar(im, ax=ax, label="Sharpe")
    else:
        ax.plot(sens[keys[0]], sens["sharpe"], marker="o", color="#2563eb")
        ax.set_xlabel(keys[0])
        ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.3)
    ax.set_title(f"{name} — Sharpe sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _save_oos_plot(oos, path, name) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(oos.is_equity.index, oos.is_equity.values, color="#2563eb",
            lw=1.5, label=f"In-sample ({oos.train_pct:.0%})")
    ax.plot(oos.oos_equity.index, oos.oos_equity.values, color="#dc2626",
            lw=1.5, label=f"Out-of-sample ({1 - oos.train_pct:.0%})")
    ax.axvline(oos.split_date, color="gray", linestyle="--", alpha=0.5, label="Split")
    ax.set_title(f"{name} — IS vs OOS equity (each rebased to initial capital)")
    ax.set_ylabel("Equity (USDT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _save_walk_forward_plot(wf, path, name) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    sharpes = [m["sharpe"] for m in wf.fold_metrics]
    labels = [f"F{i+1}\n{st.date().year}" for i, (st, _) in enumerate(wf.fold_dates)]
    colors = ["#16a34a" if s > 0 else "#dc2626" for s in sharpes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, sharpes, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(wf.summary["sharpe_mean"], color="blue", linestyle="--",
               alpha=0.6, label=f"mean={wf.summary['sharpe_mean']:.2f}")
    ax.set_title(f"{name} — walk-forward Sharpe by fold")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    app()


if __name__ == "__main__":
    main()
