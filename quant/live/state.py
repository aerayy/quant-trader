from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS state_kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol           TEXT PRIMARY KEY,
    qty              REAL NOT NULL,
    avg_entry_price  REAL NOT NULL,
    last_update      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    fee         REAL NOT NULL,
    notional    REAL NOT NULL,
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);

CREATE TABLE IF NOT EXISTS equity (
    timestamp        TEXT PRIMARY KEY,
    cash             REAL NOT NULL,
    positions_value  REAL NOT NULL,
    total_equity     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    timestamp     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    raw_weight    REAL,
    target_weight REAL,
    PRIMARY KEY (timestamp, symbol)
);
"""


class PaperState:
    """SQLite-backed persistent state for the paper trader.

    Tables:
        state_kv   — small key/value store (e.g., 'cash')
        positions  — current positions per symbol
        trades     — append-only trade log
        equity     — equity history (per signal tick)
        signals    — target weight per symbol per signal tick
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- Cash ---

    def get_cash(self) -> float:
        row = self.conn.execute(
            "SELECT value FROM state_kv WHERE key = 'cash'"
        ).fetchone()
        return float(row[0]) if row else 0.0

    def set_cash(self, cash: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO state_kv (key, value) VALUES ('cash', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(cash),),
            )

    # --- Positions ---

    def get_positions(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT symbol, qty FROM positions WHERE qty != 0"
        ).fetchall()
        return {sym: float(qty) for sym, qty in rows}

    def get_position_detail(self, symbol: str) -> tuple[float, float] | None:
        row = self.conn.execute(
            "SELECT qty, avg_entry_price FROM positions WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else None

    def update_position(self, symbol: str, qty: float, avg_entry_price: float,
                        ts: datetime) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO positions (symbol, qty, avg_entry_price, last_update) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "qty = excluded.qty, avg_entry_price = excluded.avg_entry_price, "
                "last_update = excluded.last_update",
                (symbol, qty, avg_entry_price, ts.isoformat()),
            )

    # --- Trades ---

    def record_trade(self, ts: datetime, symbol: str, side: str, qty: float,
                     price: float, fee: float, notional: float,
                     reason: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO trades (timestamp, symbol, side, qty, price, fee, "
                "notional, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts.isoformat(), symbol, side, qty, price, fee, notional, reason),
            )

    def get_recent_trades(self, limit: int = 20) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT timestamp, symbol, side, qty, price, fee, notional, reason "
            "FROM trades ORDER BY id DESC LIMIT ?",
            self.conn,
            params=(int(limit),),
        )

    # --- Equity ---

    def record_equity(self, ts: datetime, cash: float, positions_value: float,
                      total_equity: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO equity (timestamp, cash, positions_value, total_equity) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(timestamp) DO UPDATE SET "
                "cash = excluded.cash, "
                "positions_value = excluded.positions_value, "
                "total_equity = excluded.total_equity",
                (ts.isoformat(), cash, positions_value, total_equity),
            )

    def get_equity_history(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT timestamp, cash, positions_value, total_equity FROM equity "
            "ORDER BY timestamp DESC LIMIT ?",
            self.conn,
            params=(int(limit),),
        )

    # --- Signals ---

    def record_signal(self, ts: datetime, symbol: str,
                      raw_weight: float, target_weight: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO signals (timestamp, symbol, raw_weight, target_weight) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(timestamp, symbol) DO UPDATE SET "
                "raw_weight = excluded.raw_weight, "
                "target_weight = excluded.target_weight",
                (ts.isoformat(), symbol, raw_weight, target_weight),
            )

    # --- Maintenance ---

    def reset(self) -> None:
        """Wipe all state. Schema preserved."""
        with self.conn:
            self.conn.executescript(
                "DELETE FROM positions; "
                "DELETE FROM trades; "
                "DELETE FROM equity; "
                "DELETE FROM signals; "
                "DELETE FROM state_kv; "
            )

    def close(self) -> None:
        self.conn.close()
