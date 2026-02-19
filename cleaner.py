"""
Module de contrôle qualité des données de marché.

Vérifie la cohérence OHLC, détecte les outliers, les gaps,
et enregistre les anomalies dans quality_log.

Référence : Principes de backtesting (Clenow) — Piège #5 "Mauvaise qualité des données"
"""

import numpy as np
import pandas as pd

from config import (
    UNIVERSE,
    OUTLIER_DAILY_RETURN_THRESHOLD,
    MAX_CONSECUTIVE_MISSING_DAYS,
    VALIDATE_OHLC_CONSISTENCY,
)
from database import (
    load_prices,
    log_quality_issue,
    get_connection,
)


def run_quality_checks(instrument_name: str) -> dict:
    """
    Exécute tous les contrôles qualité sur un instrument.

    Returns:
        dict avec le résumé des vérifications
    """
    df = load_prices(instrument_name)

    if df.empty:
        print(f"  ⚠️  {instrument_name} : aucune donnée en base")
        return {"status": "NO_DATA"}

    # Récupérer l'instrument_id
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT instrument_id FROM instruments WHERE name = ?",
            (instrument_name,)
        )
        inst_id = cursor.fetchone()[0]

    results = {
        "instrument": instrument_name,
        "nb_rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "checks": {},
    }

    # ---- Check 1 : Valeurs manquantes (NaN) ----
    nan_counts = df.isnull().sum()
    total_nans = nan_counts.sum()
    results["checks"]["missing_values"] = {
        "total_nans": int(total_nans),
        "by_column": nan_counts.to_dict(),
    }
    if total_nans > 0:
        for col, count in nan_counts.items():
            if count > 0:
                log_quality_issue(
                    inst_id, None, "MISSING_VALUES", "WARNING",
                    f"Colonne '{col}' : {count} valeurs manquantes "
                    f"({count/len(df)*100:.1f}%)"
                )

    # ---- Check 2 : Cohérence OHLC ----
    ohlc_issues = 0
    if VALIDATE_OHLC_CONSISTENCY and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        # High doit être >= Open, Close, Low
        bad_high = df[(df["High"] < df["Open"]) | (df["High"] < df["Close"])]
        # Low doit être <= Open, Close, High
        bad_low = df[(df["Low"] > df["Open"]) | (df["Low"] > df["Close"])]

        ohlc_issues = len(bad_high) + len(bad_low)

        for date in bad_high.index:
            log_quality_issue(
                inst_id, str(date.date()), "OHLC_CONSISTENCY", "ERROR",
                f"High ({bad_high.loc[date, 'High']:.4f}) < "
                f"Open ({bad_high.loc[date, 'Open']:.4f}) ou "
                f"Close ({bad_high.loc[date, 'Close']:.4f})"
            )
        for date in bad_low.index:
            log_quality_issue(
                inst_id, str(date.date()), "OHLC_CONSISTENCY", "ERROR",
                f"Low ({bad_low.loc[date, 'Low']:.4f}) > "
                f"Open ({bad_low.loc[date, 'Open']:.4f}) ou "
                f"Close ({bad_low.loc[date, 'Close']:.4f})"
            )

    results["checks"]["ohlc_consistency"] = {"issues": ohlc_issues}

    # ---- Check 3 : Outliers (variations journalières extrêmes) ----
    if "Close" in df.columns:
        returns = df["Close"].pct_change().dropna()
        outliers = returns[returns.abs() > OUTLIER_DAILY_RETURN_THRESHOLD]

        for date, ret in outliers.items():
            log_quality_issue(
                inst_id, str(date.date()), "OUTLIER_RETURN", "WARNING",
                f"Rendement journalier extrême : {ret:+.2%} "
                f"(seuil : ±{OUTLIER_DAILY_RETURN_THRESHOLD:.0%})"
            )

        results["checks"]["outliers"] = {
            "count": len(outliers),
            "max_positive": f"{returns.max():+.2%}" if len(returns) > 0 else "N/A",
            "max_negative": f"{returns.min():+.2%}" if len(returns) > 0 else "N/A",
            "threshold": f"±{OUTLIER_DAILY_RETURN_THRESHOLD:.0%}",
        }

    # ---- Check 4 : Gaps dans les dates ----
    if len(df) > 1:
        date_diffs = pd.Series(df.index).diff().dropna()
        # En jours calendaires — les weekends font normalement 3 jours
        long_gaps = date_diffs[date_diffs > pd.Timedelta(days=MAX_CONSECUTIVE_MISSING_DAYS)]

        for idx in long_gaps.index:
            gap_start = df.index[idx - 1]
            gap_end = df.index[idx]
            gap_days = (gap_end - gap_start).days
            log_quality_issue(
                inst_id, str(gap_start.date()), "DATE_GAP", "WARNING",
                f"Gap de {gap_days} jours calendaires "
                f"({gap_start.date()} → {gap_end.date()})"
            )

        results["checks"]["date_gaps"] = {
            "long_gaps_count": len(long_gaps),
            "max_gap_days": int(date_diffs.max().days) if len(date_diffs) > 0 else 0,
        }

    # ---- Check 5 : Prix nuls ou négatifs ----
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    zero_or_neg = 0
    for col in price_cols:
        bad = df[df[col] <= 0]
        zero_or_neg += len(bad)
        for date in bad.index:
            log_quality_issue(
                inst_id, str(date.date()), "ZERO_NEGATIVE_PRICE", "ERROR",
                f"Prix {col} = {bad.loc[date, col]:.4f} (≤ 0)"
            )

    results["checks"]["zero_negative_prices"] = {"count": zero_or_neg}

    # ---- Check 6 : Données suffisantes pour le backtest ----
    trading_days = len(df)
    years = trading_days / 252
    if years < 10:
        log_quality_issue(
            inst_id, None, "INSUFFICIENT_HISTORY", "WARNING",
            f"Seulement {years:.1f} années de données "
            f"({trading_days} jours). Minimum recommandé : 10 ans."
        )
    results["checks"]["history_years"] = round(years, 1)

    return results


def run_all_quality_checks() -> list:
    """Exécute les contrôles qualité sur tous les instruments."""
    print("=" * 60)
    print("🔍 CONTRÔLE QUALITÉ DES DONNÉES")
    print("=" * 60)

    all_results = []

    for instrument in UNIVERSE:
        print(f"\n{'─' * 40}")
        print(f"📋 {instrument.name}")
        print(f"{'─' * 40}")

        result = run_quality_checks(instrument.name)
        all_results.append(result)

        if result.get("status") == "NO_DATA":
            continue

        checks = result["checks"]

        # Affichage résumé
        print(f"  📅 Période : {result['first_date']} → {result['last_date']}"
              f" ({result['nb_rows']} lignes, {checks.get('history_years', '?')} ans)")

        nans = checks.get("missing_values", {}).get("total_nans", 0)
        print(f"  {'✅' if nans == 0 else '⚠️'} Valeurs manquantes : {nans}")

        ohlc = checks.get("ohlc_consistency", {}).get("issues", 0)
        print(f"  {'✅' if ohlc == 0 else '❌'} Cohérence OHLC : {ohlc} problèmes")

        outliers = checks.get("outliers", {}).get("count", 0)
        print(f"  {'✅' if outliers == 0 else '⚠️'} Outliers : {outliers} "
              f"(seuil {checks.get('outliers', {}).get('threshold', 'N/A')})")

        gaps = checks.get("date_gaps", {}).get("long_gaps_count", 0)
        print(f"  {'✅' if gaps == 0 else '⚠️'} Gaps > {MAX_CONSECUTIVE_MISSING_DAYS}j : {gaps}")

        zeros = checks.get("zero_negative_prices", {}).get("count", 0)
        print(f"  {'✅' if zeros == 0 else '❌'} Prix nuls/négatifs : {zeros}")

    print(f"\n{'=' * 60}")
    print("✅ Contrôle qualité terminé. Voir quality_log en base pour le détail.")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_all_quality_checks()
