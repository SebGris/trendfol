"""
Module de téléchargement des données via yfinance.

Télécharge l'historique OHLCV pour chaque instrument de l'univers
et le stocke en base SQLite.
"""

import time
from datetime import datetime

import yfinance as yf
import pandas as pd

from config import UNIVERSE, DOWNLOAD_START, DOWNLOAD_INTERVAL
from database import upsert_instrument, store_prices


def download_instrument(instrument, start: str = DOWNLOAD_START,
                         end: str = None) -> pd.DataFrame:
    """
    Télécharge les données d'un instrument via yfinance.

    Args:
        instrument: objet Instrument (config.py)
        start: date de début (YYYY-MM-DD)
        end: date de fin (défaut: aujourd'hui)

    Returns:
        DataFrame avec les colonnes OHLCV standard
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"  📥 Téléchargement {instrument.name} ({instrument.ticker})...", end=" ")

    try:
        ticker = yf.Ticker(instrument.ticker)
        df = ticker.history(start=start, end=end, interval=DOWNLOAD_INTERVAL)

        if df.empty:
            print("⚠️  Aucune donnée reçue")
            return pd.DataFrame()

        # Normaliser les colonnes (yfinance peut varier)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().replace(" ", "_")
            if "open" in col_lower:
                col_map[col] = "Open"
            elif "high" in col_lower:
                col_map[col] = "High"
            elif "low" in col_lower:
                col_map[col] = "Low"
            elif "close" in col_lower and "adj" not in col_lower:
                col_map[col] = "Close"
            elif "adj" in col_lower:
                col_map[col] = "Adj Close"
            elif "volume" in col_lower:
                col_map[col] = "Volume"

        df = df.rename(columns=col_map)

        # Garder uniquement les colonnes attendues
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        available = [c for c in expected_cols if c in df.columns]
        df = df[available]

        # S'assurer que l'index est bien un DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Retirer le timezone si présent (SQLite ne gère pas les tz)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        print(f"✅ {len(df)} lignes ({df.index.min().date()} → {df.index.max().date()})")
        return df

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return pd.DataFrame()


def download_all(start: str = DOWNLOAD_START) -> dict:
    """
    Télécharge et stocke en base tous les instruments de l'univers.

    Returns:
        dict : {nom_instrument: nb_lignes_stockées}
    """
    results = {}

    print("=" * 60)
    print("🚀 TÉLÉCHARGEMENT DE L'UNIVERS D'INSTRUMENTS")
    print(f"   Début : {start}")
    print(f"   Instruments : {len(UNIVERSE)}")
    print("=" * 60)

    for instrument in UNIVERSE:
        # Enregistrer l'instrument en base
        inst_id = upsert_instrument(
            name=instrument.name,
            ticker=instrument.ticker,
            sector=instrument.sector,
            point_value=instrument.point_value,
            currency=instrument.currency,
            instrument_type=instrument.instrument_type,
        )

        # Télécharger les données
        df = download_instrument(instrument, start=start)

        if df.empty:
            results[instrument.name] = 0
            continue

        # Stocker en base
        nb_stored = store_prices(inst_id, df)
        results[instrument.name] = nb_stored

        # Petite pause pour ne pas surcharger l'API Yahoo
        time.sleep(1)

    print("=" * 60)
    print("📊 RÉSUMÉ DU TÉLÉCHARGEMENT")
    print("-" * 40)
    total = 0
    for name, count in results.items():
        status = "✅" if count > 0 else "⚠️"
        print(f"  {status} {name:20s} : {count:>6} lignes")
        total += count
    print("-" * 40)
    print(f"  Total : {total} lignes")
    print("=" * 60)

    return results


if __name__ == "__main__":
    from database import init_db
    init_db()
    download_all()
