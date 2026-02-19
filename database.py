"""
Module de base de données SQLite pour les données de marché.

Tables :
  - instruments : métadonnées de chaque instrument
  - daily_prices : données OHLCV journalières
  - quality_log : journal des anomalies détectées
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from config import DB_PATH


def init_db() -> None:
    """Crée la base et les tables si elles n'existent pas."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT UNIQUE NOT NULL,
                ticker          TEXT NOT NULL,
                sector          TEXT NOT NULL,
                point_value     REAL NOT NULL,
                currency        TEXT NOT NULL,
                instrument_type TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now')),
                last_updated    TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_prices (
                instrument_id   INTEGER NOT NULL,
                date            TEXT NOT NULL,
                open            REAL,
                high            REAL,
                low             REAL,
                close           REAL NOT NULL,
                adj_close       REAL,
                volume          INTEGER,
                PRIMARY KEY (instrument_id, date),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            );

            CREATE TABLE IF NOT EXISTS quality_log (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_id   INTEGER NOT NULL,
                date            TEXT,
                check_type      TEXT NOT NULL,
                severity        TEXT NOT NULL,  -- 'WARNING' ou 'ERROR'
                message         TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            );

            CREATE INDEX IF NOT EXISTS idx_prices_date
                ON daily_prices(date);
            CREATE INDEX IF NOT EXISTS idx_prices_instrument
                ON daily_prices(instrument_id);
            CREATE INDEX IF NOT EXISTS idx_quality_instrument
                ON quality_log(instrument_id);
        """)
    print(f"✅ Base de données initialisée : {DB_PATH}")


@contextmanager
def get_connection():
    """Context manager pour les connexions SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_instrument(name: str, ticker: str, sector: str,
                       point_value: float, currency: str,
                       instrument_type: str) -> int:
    """Insère ou met à jour un instrument. Retourne l'instrument_id."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO instruments (name, ticker, sector, point_value,
                                     currency, instrument_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                ticker = excluded.ticker,
                sector = excluded.sector,
                point_value = excluded.point_value,
                currency = excluded.currency,
                instrument_type = excluded.instrument_type,
                last_updated = datetime('now')
        """, (name, ticker, sector, point_value, currency, instrument_type))

        cursor = conn.execute(
            "SELECT instrument_id FROM instruments WHERE name = ?", (name,)
        )
        return cursor.fetchone()[0]


def store_prices(instrument_id: int, df: pd.DataFrame) -> int:
    """
    Stocke les prix OHLCV dans la base.
    Le DataFrame doit avoir un DatetimeIndex et les colonnes
    Open, High, Low, Close, Volume (et optionnellement Adj Close).

    Retourne le nombre de lignes insérées/mises à jour.
    """
    if df.empty:
        return 0

    records = []
    for date, row in df.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        records.append((
            instrument_id,
            date_str,
            _safe_float(row.get("Open")),
            _safe_float(row.get("High")),
            _safe_float(row.get("Low")),
            _safe_float(row.get("Close")),
            _safe_float(row.get("Adj Close")),
            _safe_int(row.get("Volume")),
        ))

    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO daily_prices
                (instrument_id, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                adj_close = excluded.adj_close,
                volume = excluded.volume
        """, records)

        # Mettre à jour last_updated
        conn.execute("""
            UPDATE instruments SET last_updated = datetime('now')
            WHERE instrument_id = ?
        """, (instrument_id,))

    return len(records)


def log_quality_issue(instrument_id: int, date: str,
                       check_type: str, severity: str,
                       message: str) -> None:
    """Enregistre un problème de qualité dans le journal."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO quality_log
                (instrument_id, date, check_type, severity, message)
            VALUES (?, ?, ?, ?, ?)
        """, (instrument_id, date, check_type, severity, message))


def load_prices(instrument_name: str,
                start_date: str = None,
                end_date: str = None) -> pd.DataFrame:
    """
    Charge les prix depuis la base pour un instrument donné.
    Retourne un DataFrame avec DatetimeIndex.
    """
    query = """
        SELECT dp.date, dp.open, dp.high, dp.low, dp.close,
               dp.adj_close, dp.volume
        FROM daily_prices dp
        JOIN instruments i ON dp.instrument_id = i.instrument_id
        WHERE i.name = ?
    """
    params = [instrument_name]

    if start_date:
        query += " AND dp.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND dp.date <= ?"
        params.append(end_date)

    query += " ORDER BY dp.date"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])

    if not df.empty:
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    return df


def get_quality_report(instrument_name: str = None) -> pd.DataFrame:
    """Récupère le rapport de qualité, filtré optionnellement par instrument."""
    query = """
        SELECT i.name, ql.date, ql.check_type, ql.severity, ql.message
        FROM quality_log ql
        JOIN instruments i ON ql.instrument_id = i.instrument_id
    """
    params = []
    if instrument_name:
        query += " WHERE i.name = ?"
        params.append(instrument_name)
    query += " ORDER BY ql.created_at DESC"

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_data_summary() -> pd.DataFrame:
    """Résumé : nb de lignes, dates min/max par instrument."""
    query = """
        SELECT i.name, i.ticker, i.sector,
               COUNT(dp.date) as nb_rows,
               MIN(dp.date) as first_date,
               MAX(dp.date) as last_date,
               i.last_updated
        FROM instruments i
        LEFT JOIN daily_prices dp ON i.instrument_id = dp.instrument_id
        GROUP BY i.instrument_id
        ORDER BY i.sector, i.name
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


# ============================================================
# Helpers
# ============================================================
def _safe_float(val) -> float | None:
    """Convertit en float, retourne None si NaN ou None."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convertit en int, retourne None si NaN ou None."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else int(f)
    except (ValueError, TypeError):
        return None
