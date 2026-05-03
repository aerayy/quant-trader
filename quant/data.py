from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

NUMERIC_COLS = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]


class DataStore:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str, year: int, month: int) -> Path:
        return self.cache_dir / symbol / interval / f"{year}-{month:02d}.parquet"

    def has(self, symbol: str, interval: str, year: int, month: int) -> bool:
        return self._path(symbol, interval, year, month).exists()

    def save(self, df: pd.DataFrame, symbol: str, interval: str, year: int, month: int) -> None:
        path = self._path(symbol, interval, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy")

    def load(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        sym_dir = self.cache_dir / symbol / interval
        if not sym_dir.exists():
            raise FileNotFoundError(
                f"No cached data for {symbol} {interval}. Run `python -m quant download` first."
            )
        parts = sorted(sym_dir.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files in {sym_dir}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        # Binance vision uses ms for ≤2024 archives but switched to µs for 2025+.
        # Normalize so values > 1e15 (16+ digits) are downcast µs → ms.
        ot = df["open_time"]
        mask_us = ot > 1_000_000_000_000_000
        if mask_us.any():
            df.loc[mask_us, "open_time"] = ot[mask_us] // 1000
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        return df.loc[start:end]


def _download_zip(url: str, timeout: int = 30) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # Some monthly files include a header row in 2025+; sniff it.
            first = f.readline()
            f.seek(0)
            has_header = b"open_time" in first.lower()
            df = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=None if has_header else KLINE_COLUMNS,
            )
    if list(df.columns)[: len(KLINE_COLUMNS)] != KLINE_COLUMNS[: len(df.columns)]:
        df.columns = KLINE_COLUMNS[: len(df.columns)]
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
    url = BINANCE_VISION_URL.format(symbol=symbol, interval=interval, year=year, month=month)
    return _download_zip(url)


def months_between(start: str, end: str) -> Iterable[tuple[int, int]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = pd.Timestamp(s.year, s.month, 1)
    while cur <= e:
        yield cur.year, cur.month
        cur = cur + pd.offsets.MonthBegin(1)


def download(
    symbols: list[str],
    interval: str,
    start: str,
    end: str,
    store: DataStore,
) -> dict[str, int]:
    months = list(months_between(start, end))
    summary: dict[str, int] = {}

    for symbol in symbols:
        pending = [(y, m) for y, m in months if not store.has(symbol, interval, y, m)]
        if not pending:
            print(f"[{symbol}] cached, skipping {len(months)} months")
            summary[symbol] = 0
            continue

        ok = 0
        for year, month in tqdm(pending, desc=f"{symbol} {interval}"):
            try:
                df = download_month(symbol, interval, year, month)
                store.save(df, symbol, interval, year, month)
                ok += 1
            except requests.HTTPError as e:
                # Current month may not be archived yet; skip with note.
                print(f"  [{symbol} {year}-{month:02d}] not available ({e.response.status_code})")
        summary[symbol] = ok

    return summary
