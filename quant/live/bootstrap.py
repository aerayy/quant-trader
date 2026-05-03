from __future__ import annotations

from typing import Any

import pandas as pd

from quant.data import (
    BINANCE_VISION_URL,  # noqa: F401  (re-export for clarity)
    KLINE_COLUMNS,
    NUMERIC_COLS,
    DataStore,
)


def fetch_recent_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch the most recent N klines via Binance public REST API.

    No auth needed. Returns a DataFrame with the standard 12 kline columns,
    numeric columns coerced.
    """
    from binance.client import Client

    client = Client(requests_params={"timeout": 30})
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS[: len(raw[0]) if raw else 12])
    if list(df.columns)[: len(KLINE_COLUMNS)] != KLINE_COLUMNS[: len(df.columns)]:
        df.columns = KLINE_COLUMNS[: len(df.columns)]
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
    return df


def _filter_closed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the still-forming current bar (close_time in the future)."""
    if df.empty:
        return df
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    return df[df["close_time"] < now_ms].copy()


def bootstrap_klines(
    cfg: dict,
    limit: int = 200,
    console: Any = None,
) -> dict[str, int]:
    """Refresh local kline cache for each configured symbol with the most
    recent N closed bars from REST. Idempotent — merges with existing
    monthly Parquet files (REST data wins on conflicts).
    """
    syms: list[str] = cfg["data"]["symbols"]
    iv: str = cfg["data"]["interval"]
    store = DataStore(cfg["data"]["cache_dir"])
    summary: dict[str, int] = {}

    for sym in syms:
        try:
            fresh = fetch_recent_klines(sym, iv, limit=limit)
        except Exception as e:
            if console:
                console.print(f"[yellow]⚠[/yellow] {sym} REST fetch failed: {e!r}")
            summary[sym] = 0
            continue

        fresh = _filter_closed(fresh)
        if fresh.empty:
            summary[sym] = 0
            continue

        # The `ignore` column varies in dtype across archive eras (int vs
        # str vs object); we never use it, so drop to avoid concat dtype
        # collisions with existing Parquet.
        if "ignore" in fresh.columns:
            fresh = fresh.drop(columns=["ignore"])

        # Group by month, merge with existing Parquet, save back
        fresh["_month"] = pd.to_datetime(
            fresh["open_time"].astype("int64"), unit="ms"
        ).dt.to_period("M")

        n_bars = 0
        for month, group in fresh.groupby("_month"):
            year = int(month.year)
            mon = int(month.month)
            new_df = group.drop(columns=["_month"])

            if store.has(sym, iv, year, mon):
                existing = pd.read_parquet(store._path(sym, iv, year, mon))
                if "ignore" in existing.columns:
                    existing = existing.drop(columns=["ignore"])
                # Old archives may have µs timestamps; REST returns ms.
                # Normalize to ms BEFORE merging so duplicates collapse.
                for col in ("open_time", "close_time"):
                    if col in existing.columns:
                        v = existing[col].astype("int64")
                        mask = v > 1_000_000_000_000_000
                        if mask.any():
                            v[mask] = v[mask] // 1000
                            existing[col] = v
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates("open_time", keep="last")
                combined = combined.sort_values("open_time").reset_index(drop=True)
            else:
                combined = new_df.sort_values("open_time").reset_index(drop=True)

            store.save(combined, sym, iv, year, mon)
            n_bars += len(new_df)

        summary[sym] = n_bars
        if console:
            last_ts = pd.to_datetime(int(fresh["open_time"].iloc[-1]), unit="ms")
            console.print(
                f"[green]✓[/green] {sym}: refreshed {len(fresh)} bars "
                f"(latest {last_ts.date()})"
            )

    return summary
