"""
Calcul des indicateurs pour tous les instruments en base.
Stocke les résultats dans une nouvelle table 'daily_indicators'.

Usage :
    python compute_indicators.py              # Tous les instruments
    python compute_indicators.py "S&P 500"    # Un seul instrument
    python compute_indicators.py --summary    # Résumé des dernières valeurs
"""

import sys
import sqlite3

import pandas as pd

from config import UNIVERSE, ATR_PERIOD, EMA_FAST, EMA_SLOW
from database import load_prices, get_connection, DB_PATH
from indicators import compute_all_indicators, indicators_summary


def init_indicators_table() -> None:
    """Crée la table daily_indicators si elle n'existe pas."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_indicators (
                instrument_id   INTEGER NOT NULL,
                date            TEXT NOT NULL,
                returns         REAL,
                log_returns     REAL,
                ema_fast        REAL,
                ema_slow        REAL,
                atr             REAL,
                atr_pct         REAL,
                ann_vol         REAL,
                exp_vol         REAL,
                entry_high      REAL,
                entry_low       REAL,
                exit_high       REAL,
                exit_low        REAL,
                ma_position     INTEGER,
                ewmac_16_64     REAL,
                PRIMARY KEY (instrument_id, date),
                FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
            );

            CREATE INDEX IF NOT EXISTS idx_indicators_date
                ON daily_indicators(date);
            CREATE INDEX IF NOT EXISTS idx_indicators_instrument
                ON daily_indicators(instrument_id);
        """)


def store_indicators(instrument_name: str, df: pd.DataFrame) -> int:
    """
    Stocke les indicateurs calculés dans daily_indicators.

    Args:
        instrument_name: nom de l'instrument
        df: DataFrame avec indicateurs (sortie de compute_all_indicators)

    Returns:
        Nombre de lignes insérées
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT instrument_id FROM instruments WHERE name = ?",
            (instrument_name,)
        )
        row = cursor.fetchone()
        if not row:
            print(f"  ❌ Instrument '{instrument_name}' non trouvé en base")
            return 0
        inst_id = row[0]

        records = []
        for date, r in df.iterrows():
            records.append((
                inst_id,
                date.strftime("%Y-%m-%d"),
                _safe(r.get("returns")),
                _safe(r.get("log_returns")),
                _safe(r.get(f"ema_{EMA_FAST}")),
                _safe(r.get(f"ema_{EMA_SLOW}")),
                _safe(r.get("atr")),
                _safe(r.get("atr_pct")),
                _safe(r.get("ann_vol")),
                _safe(r.get("exp_vol")),
                _safe(r.get("entry_high")),
                _safe(r.get("entry_low")),
                _safe(r.get("exit_high")),
                _safe(r.get("exit_low")),
                _safe_int(r.get("ma_position")),
                _safe(r.get("ewmac_16_64")),
            ))

        conn.executemany("""
            INSERT INTO daily_indicators
                (instrument_id, date, returns, log_returns,
                 ema_fast, ema_slow, atr, atr_pct,
                 ann_vol, exp_vol,
                 entry_high, entry_low, exit_high, exit_low,
                 ma_position, ewmac_16_64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, date) DO UPDATE SET
                returns = excluded.returns,
                log_returns = excluded.log_returns,
                ema_fast = excluded.ema_fast,
                ema_slow = excluded.ema_slow,
                atr = excluded.atr,
                atr_pct = excluded.atr_pct,
                ann_vol = excluded.ann_vol,
                exp_vol = excluded.exp_vol,
                entry_high = excluded.entry_high,
                entry_low = excluded.entry_low,
                exit_high = excluded.exit_high,
                exit_low = excluded.exit_low,
                ma_position = excluded.ma_position,
                ewmac_16_64 = excluded.ewmac_16_64
        """, records)

    return len(records)


def compute_for_instrument(instrument_name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Charge les prix, calcule les indicateurs, stocke en base.

    Returns:
        DataFrame enrichi
    """
    if verbose:
        print(f"\n  📐 Calcul des indicateurs pour {instrument_name}...", end=" ")

    df = load_prices(instrument_name)
    if df.empty:
        if verbose:
            print("⚠️  Aucune donnée")
        return pd.DataFrame()

    enriched = compute_all_indicators(df)
    nb = store_indicators(instrument_name, enriched)

    if verbose:
        print(f"✅ {nb} lignes")

    return enriched


def compute_all(verbose: bool = True) -> dict:
    """
    Calcule et stocke les indicateurs pour tous les instruments.

    Returns:
        dict : {instrument_name: DataFrame enrichi}
    """
    init_indicators_table()

    print("=" * 60)
    print("📐 CALCUL DES INDICATEURS — PHASE 1b")
    print(f"   EMA : {EMA_FAST} / {EMA_SLOW}")
    print(f"   ATR : {ATR_PERIOD} jours (EMA)")
    print(f"   Donchian : 100 / 50 jours")
    print(f"   EWMAC : 16 / 64 (Carver Starter)")
    print("=" * 60)

    results = {}
    for instrument in UNIVERSE:
        enriched = compute_for_instrument(instrument.name, verbose)
        if not enriched.empty:
            results[instrument.name] = enriched

    if verbose:
        print("\n" + "=" * 60)
        print("📊 DERNIÈRES VALEURS")
        print("=" * 60)
        for name, df in results.items():
            print(indicators_summary(df, name))

    print("\n✅ Phase 1b terminée — indicateurs stockés dans daily_indicators")
    return results


def print_last_values():
    """Affiche les dernières valeurs d'indicateurs depuis la base."""
    print("\n" + "=" * 60)
    print("📊 DERNIÈRES VALEURS D'INDICATEURS")
    print("=" * 60)

    for instrument in UNIVERSE:
        df = load_prices(instrument.name)
        if df.empty:
            continue
        enriched = compute_all_indicators(df)
        print(indicators_summary(enriched, instrument.name))


# ============================================================
# Helpers
# ============================================================

def _safe(val):
    """Convertit en float safe pour SQLite."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else round(f, 8)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Convertit en int safe pour SQLite."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else int(f)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        compute_all()
    elif args[0] == "--summary":
        print_last_values()
    else:
        # Instrument spécifique
        init_indicators_table()
        name = args[0]
        enriched = compute_for_instrument(name)
        if not enriched.empty:
            print(indicators_summary(enriched, name))
