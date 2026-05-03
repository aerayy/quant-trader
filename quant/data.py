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

FUNDING_VISION_URL = (
    "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
    "{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

NUMERIC_COLS = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]

# Funding column names vary across Binance vision archive eras.
FUNDING_TIME_COLS = ("calc_time", "fundingTime")
FUNDING_RATE_COLS = ("last_funding_rate", "fundingRate")


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

    # --- Funding rate (perpetual futures) ---

    def _funding_path(self, symbol: str, year: int, month: int) -> Path:
        return self.cache_dir / "_funding" / symbol / f"{year}-{month:02d}.parquet"

    def has_funding(self, symbol: str, year: int, month: int) -> bool:
        return self._funding_path(symbol, year, month).exists()

    def save_funding(self, df: pd.DataFrame, symbol: str, year: int, month: int) -> None:
        path = self._funding_path(symbol, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy")

    def load_funding(self, symbol: str, start: str, end: str) -> pd.Series:
        sym_dir = self.cache_dir / "_funding" / symbol
        if not sym_dir.exists():
            return pd.Series(dtype=float, name="funding_rate")
        parts = sorted(sym_dir.glob("*.parquet"))
        if not parts:
            return pd.Series(dtype=float, name="funding_rate")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

        time_col = next((c for c in FUNDING_TIME_COLS if c in df.columns), None)
        rate_col = next((c for c in FUNDING_RATE_COLS if c in df.columns), None)
        if time_col is None or rate_col is None:
            raise ValueError(
                f"Funding parquet for {symbol} missing expected columns; got {list(df.columns)}"
            )

        # calc_time may be either ms-int64 (most archives) or ISO string;
        # fundingTime is always numeric. Handle both.
        col = df[time_col]
        if pd.api.types.is_numeric_dtype(col):
            raw = col.astype("int64")
            mask_us = raw > 1_000_000_000_000_000
            if mask_us.any():
                raw[mask_us] = raw[mask_us] // 1000
            t = pd.to_datetime(raw, unit="ms")
        else:
            t = pd.to_datetime(col)

        rates = pd.to_numeric(df[rate_col], errors="coerce")
        s = pd.Series(rates.values, index=t, name="funding_rate")
        s = s[~s.index.duplicated(keep="first")].sort_index()
        return s.loc[start:end]


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
            print(f"[{symbol}] kline cached, skipping {len(months)} months")
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


# --- Funding rate download ---

def _download_funding_zip(url: str, timeout: int = 30) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            sample = f.read(200).decode("utf-8", errors="ignore")
            f.seek(0)
            first = sample.split("\n")[0].strip()
            first_token = first.split(",")[0] if first else ""
            has_header = bool(first_token) and any(ch.isalpha() for ch in first_token)
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None,
                                 names=["fundingTime", "symbol", "fundingRate"])
    return df


def download_funding_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    url = FUNDING_VISION_URL.format(symbol=symbol, year=year, month=month)
    return _download_funding_zip(url)


def download_funding(
    symbols: list[str],
    start: str,
    end: str,
    store: DataStore,
) -> dict[str, int]:
    months = list(months_between(start, end))
    summary: dict[str, int] = {}

    for symbol in symbols:
        pending = [(y, m) for y, m in months if not store.has_funding(symbol, y, m)]
        if not pending:
            print(f"[{symbol}] funding cached, skipping {len(months)} months")
            summary[symbol] = 0
            continue

        ok = 0
        for year, month in tqdm(pending, desc=f"{symbol} funding"):
            try:
                df = download_funding_month(symbol, year, month)
                store.save_funding(df, symbol, year, month)
                ok += 1
            except requests.HTTPError as e:
                print(f"  [{symbol} {year}-{month:02d}] funding not available ({e.response.status_code})")
        summary[symbol] = ok

    return summary
