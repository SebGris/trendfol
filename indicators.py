"""
Module d'indicateurs techniques — Phase 1b
=============================================

Briques réutilisables pour toutes les stratégies :
  - Rendements journaliers
  - EMA (configurable)
  - ATR (Average True Range)
  - Volatilité annualisée
  - Canaux de Donchian
  - Signaux de croisement (crossover)
  - EWMAC / Forecast (Carver)

Convention : toutes les fonctions prennent un DataFrame OHLCV
avec DatetimeIndex et retournent une Series ou un DataFrame.

Références :
  - Clenow: EMA 50/100, ATR 20j, volatilité pour sizing
  - Carver: EWMAC multi-vitesse, forecast scalars
  - Kaufman: True Range, Donchian channels
"""

import numpy as np
import pandas as pd

from config import ATR_PERIOD, EMA_FAST, EMA_SLOW, VOLATILITY_WINDOW


# ============================================================
# RENDEMENTS
# ============================================================

def daily_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Rendements journaliers simples (pct_change)."""
    return df[column].pct_change()


def log_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Rendements logarithmiques journaliers."""
    return np.log(df[column] / df[column].shift(1))


# ============================================================
# MOYENNES MOBILES
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Moyenne mobile exponentielle (EMA).

    Args:
        series: série de prix
        span: période (ex: 50, 100 jours)

    Returns:
        EMA de la série
    """
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple (SMA)."""
    return series.rolling(window=window).mean()


def ema_pair(df: pd.DataFrame, fast: int = EMA_FAST, slow: int = EMA_SLOW,
             column: str = "Close") -> pd.DataFrame:
    """
    Calcule une paire EMA rapide/lente (Clenow).

    Returns:
        DataFrame avec colonnes 'ema_fast', 'ema_slow'
    """
    return pd.DataFrame({
        f"ema_{fast}": ema(df[column], fast),
        f"ema_{slow}": ema(df[column], slow),
    }, index=df.index)


# ============================================================
# ATR — AVERAGE TRUE RANGE
# ============================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (Kaufman) :
      TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    """
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD,
        method: str = "ema") -> pd.Series:
    """
    Average True Range.

    Args:
        df: DataFrame OHLCV
        period: période de lissage (défaut: 20 — Clenow)
        method: 'ema' (Clenow, défaut) ou 'sma' (classique)

    Returns:
        ATR en valeur absolue (même unité que le prix)
    """
    tr = true_range(df)
    if method == "ema":
        return tr.ewm(span=period, adjust=False).mean()
    else:
        return tr.rolling(window=period).mean()


def atr_pct(df: pd.DataFrame, period: int = ATR_PERIOD,
            method: str = "ema") -> pd.Series:
    """ATR en pourcentage du prix Close (utile pour comparer entre instruments)."""
    return atr(df, period, method) / df["Close"] * 100


# ============================================================
# VOLATILITÉ ANNUALISÉE
# ============================================================

def annualized_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW,
                           column: str = "Close",
                           trading_days: int = 256) -> pd.Series:
    """
    Volatilité annualisée glissante (Carver : écart-type des rendements × √252).

    Args:
        df: DataFrame OHLCV
        window: fenêtre glissante en jours (défaut: 256 ≈ 1 an)
        column: colonne de prix
        trading_days: jours de trading par an (Carver utilise 256)

    Returns:
        Volatilité annualisée en décimal (ex: 0.20 = 20%)
    """
    ret = log_returns(df, column)
    return ret.rolling(window=window).std() * np.sqrt(trading_days)


def exponential_volatility(df: pd.DataFrame, span: int = 36,
                            column: str = "Close",
                            trading_days: int = 256) -> pd.Series:
    """
    Volatilité annualisée exponentielle (Carver: span=36 jours).
    Plus réactive que la volatilité glissante.

    Args:
        span: demi-vie exponentielle (Carver recommande 36j)
    """
    ret = log_returns(df, column)
    return ret.ewm(span=span).std() * np.sqrt(trading_days)


# ============================================================
# CANAUX DE DONCHIAN
# ============================================================

def donchian(df: pd.DataFrame, entry_period: int = 100,
             exit_period: int = 50) -> pd.DataFrame:
    """
    Canaux de Donchian (Clenow Breakout / Turtles).

    Args:
        entry_period: fenêtre pour les bandes d'entrée (défaut: 100j Clenow)
        exit_period: fenêtre pour les bandes de sortie (défaut: 50j Clenow)

    Returns:
        DataFrame avec colonnes :
          - entry_high  : plus haut sur entry_period (signal long)
          - entry_low   : plus bas sur entry_period (signal short)
          - exit_high   : plus haut sur exit_period (sortie short)
          - exit_low    : plus bas sur exit_period (sortie long)
    """
    return pd.DataFrame({
        "entry_high": df["High"].rolling(window=entry_period).max(),
        "entry_low": df["Low"].rolling(window=entry_period).min(),
        "exit_high": df["High"].rolling(window=exit_period).max(),
        "exit_low": df["Low"].rolling(window=exit_period).min(),
    }, index=df.index)


# ============================================================
# SIGNAUX DE CROISEMENT
# ============================================================

def crossover_signal(fast_series: pd.Series,
                     slow_series: pd.Series) -> pd.Series:
    """
    Détecte les croisements entre deux séries.

    Returns:
        Series avec :
          +1 = croisement haussier (fast passe au-dessus de slow)
          -1 = croisement baissier (fast passe en-dessous de slow)
           0 = pas de croisement
    """
    position = (fast_series > slow_series).astype(int)
    signal = position.diff()
    # diff() donne 1 (passage de 0→1) ou -1 (passage de 1→0)
    return signal.fillna(0).astype(int)


def ma_position(df: pd.DataFrame, fast: int = EMA_FAST, slow: int = EMA_SLOW,
                column: str = "Close") -> pd.Series:
    """
    Position continue basée sur les EMA (Clenow — always-in-market).

    Returns:
        +1 si EMA_fast > EMA_slow (haussier)
        -1 si EMA_fast < EMA_slow (baissier)
    """
    emas = ema_pair(df, fast, slow, column)
    fast_col = f"ema_{fast}"
    slow_col = f"ema_{slow}"
    return pd.Series(
        np.where(emas[fast_col] > emas[slow_col], 1, -1),
        index=df.index,
        name="ma_position",
    )


# ============================================================
# EWMAC — EXPONENTIALLY WEIGHTED MOVING AVERAGE CROSSOVER (Carver)
# ============================================================

# Forecast scalars par paire de vitesses (Carver, Systematic Trading)
FORECAST_SCALARS = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}

FORECAST_CAP = 20.0  # Carver: cap absolu à ±20


def ewmac_forecast(df: pd.DataFrame, fast: int = 16, slow: int = 64,
                    column: str = "Close",
                    vol_span: int = 36) -> pd.Series:
    """
    Forecast EWMAC (Carver, Systematic Trading ch.7).

    Calcul :
      1. raw = EMA(fast) - EMA(slow)
      2. volatility = ecart-type exponentiel des prix (span=36)
      3. scaled = (raw / volatility) × forecast_scalar
      4. capped à [-20, +20]

    Args:
        fast: période EMA rapide (ex: 16)
        slow: période EMA lente (ex: 64)
        vol_span: span pour la volatilité (Carver: 36)

    Returns:
        Forecast : valeur continue dans [-20, +20], moyenne abs ≈ 10
    """
    price = df[column]

    # 1. Raw crossover
    ema_fast = ema(price, fast)
    ema_slow = ema(price, slow)
    raw = ema_fast - ema_slow

    # 2. Normaliser par la volatilité du prix
    #    Carver utilise l'écart-type des prix (pas des rendements)
    #    pour que le raw crossover soit comparable entre instruments
    price_vol = price.ewm(span=vol_span).std()
    normalized = raw / price_vol

    # 3. Appliquer le forecast scalar
    speed_pair = (fast, slow)
    scalar = FORECAST_SCALARS.get(speed_pair, _estimate_scalar(fast, slow))
    scaled = normalized * scalar

    # 4. Capper à [-20, +20]
    capped = scaled.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)

    return capped.rename(f"ewmac_{fast}_{slow}")


def combined_forecast(df: pd.DataFrame,
                       speed_pairs: list = None,
                       weights: dict = None,
                       column: str = "Close") -> pd.Series:
    """
    Forecast combiné multi-vitesses (Carver, diversification).

    Args:
        speed_pairs: liste de tuples (fast, slow).
                     Défaut: [(8,32), (16,64), (32,128)]
        weights: dict {(fast,slow): weight}. Défaut: poids égaux.

    Returns:
        Forecast combiné, cappé à [-20, +20]
    """
    if speed_pairs is None:
        speed_pairs = [(8, 32), (16, 64), (32, 128)]

    if weights is None:
        w = 1.0 / len(speed_pairs)
        weights = {pair: w for pair in speed_pairs}

    forecasts = pd.DataFrame()
    for fast, slow in speed_pairs:
        fc = ewmac_forecast(df, fast, slow, column)
        forecasts[f"ewmac_{fast}_{slow}"] = fc

    # Pondération
    combined = sum(
        forecasts[f"ewmac_{fast}_{slow}"] * weights[(fast, slow)]
        for fast, slow in speed_pairs
    )

    # Recapper le résultat combiné
    return combined.clip(-FORECAST_CAP, FORECAST_CAP).rename("combined_forecast")


def _estimate_scalar(fast: int, slow: int) -> float:
    """
    Estime le forecast scalar pour des paires non standard.
    Interpolation log-linéaire sur les scalars connus.
    """
    known = sorted(FORECAST_SCALARS.items())
    fasts = [k[0][0] for k in known]
    scalars = [k[1] for k in known]

    # Interpolation log-linéaire
    log_fasts = np.log(fasts)
    log_scalars = np.log(scalars)
    log_fast = np.log(fast)

    estimated = np.interp(log_fast, log_fasts, log_scalars)
    return float(np.exp(estimated))


# ============================================================
# FONCTION UTILITAIRE : CALCUL COMPLET
# ============================================================

def compute_all_indicators(df: pd.DataFrame,
                            fast: int = EMA_FAST,
                            slow: int = EMA_SLOW,
                            atr_period: int = ATR_PERIOD) -> pd.DataFrame:
    """
    Calcule tous les indicateurs de base et les ajoute au DataFrame.

    Colonnes ajoutées :
      - returns       : rendements journaliers
      - log_returns   : rendements log
      - ema_{fast}    : EMA rapide
      - ema_{slow}    : EMA lente
      - atr           : Average True Range
      - atr_pct       : ATR en % du prix
      - ann_vol       : volatilité annualisée (rolling)
      - exp_vol       : volatilité annualisée (exponentielle)
      - entry_high    : Donchian haut (entrée)
      - entry_low     : Donchian bas (entrée)
      - exit_high     : Donchian haut (sortie)
      - exit_low      : Donchian bas (sortie)
      - ma_position   : signal EMA (+1/-1)
      - ewmac_16_64   : forecast EWMAC Carver (starter)

    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    result = df.copy()

    # Rendements
    result["returns"] = daily_returns(df)
    result["log_returns"] = log_returns(df)

    # Moyennes mobiles
    emas = ema_pair(df, fast, slow)
    result[f"ema_{fast}"] = emas[f"ema_{fast}"]
    result[f"ema_{slow}"] = emas[f"ema_{slow}"]

    # ATR
    result["atr"] = atr(df, atr_period)
    result["atr_pct"] = atr_pct(df, atr_period)

    # Volatilité
    result["ann_vol"] = annualized_volatility(df)
    result["exp_vol"] = exponential_volatility(df)

    # Donchian
    don = donchian(df)
    for col in don.columns:
        result[col] = don[col]

    # Signaux
    result["ma_position"] = ma_position(df, fast, slow)
    result["ewmac_16_64"] = ewmac_forecast(df, 16, 64)

    return result


# ============================================================
# AFFICHAGE RÉSUMÉ
# ============================================================

def indicators_summary(df: pd.DataFrame, instrument_name: str = "") -> str:
    """
    Génère un résumé texte des indicateurs pour un instrument.
    Suppose que compute_all_indicators() a été appelé.
    """
    last = df.iloc[-1]
    lines = [
        f"{'═' * 50}",
        f"📊 {instrument_name or 'Indicateurs'} — {df.index[-1].date()}",
        f"{'─' * 50}",
        f"  Close        : {last['Close']:>12.4f}",
    ]

    if "ema_50" in df.columns:
        lines.append(f"  EMA 50       : {last['ema_50']:>12.4f}")
    if "ema_100" in df.columns:
        lines.append(f"  EMA 100      : {last['ema_100']:>12.4f}")
    if "atr" in df.columns:
        lines.append(f"  ATR (20j)    : {last['atr']:>12.4f}")
    if "atr_pct" in df.columns:
        lines.append(f"  ATR %        : {last['atr_pct']:>11.2f}%")
    if "ann_vol" in df.columns and not np.isnan(last["ann_vol"]):
        lines.append(f"  Vol. annuelle: {last['ann_vol']:>11.1%}")
    if "exp_vol" in df.columns and not np.isnan(last["exp_vol"]):
        lines.append(f"  Vol. exp.    : {last['exp_vol']:>11.1%}")
    if "ma_position" in df.columns:
        pos = "🟢 LONG" if last["ma_position"] > 0 else "🔴 SHORT"
        lines.append(f"  Signal MA    :  {pos}")
    if "ewmac_16_64" in df.columns and not np.isnan(last["ewmac_16_64"]):
        fc = last["ewmac_16_64"]
        bar = "█" * int(abs(fc))
        direction = "↑" if fc > 0 else "↓"
        lines.append(f"  EWMAC 16/64  : {fc:>+11.2f} {direction} {bar}")

    lines.append(f"{'═' * 50}")
    return "\n".join(lines)
