"""
Stratégies de trading — Implémentations.
==========================================

Chaque stratégie est une fonction avec la signature :
    signal = strategy(date, row, instrument, positions) -> int
        +1 = long
        -1 = short
         0 = flat

Stratégie A : MA Crossover (Clenow)
    - Always-in-market
    - Long si EMA50 > EMA100, Short sinon
    - Simple, base de référence

Stratégie B : Breakout / Donchian (Clenow)
    - Entrée long sur nouveau plus haut 100j (si pas déjà long)
    - Entrée short sur nouveau plus bas 100j (si pas déjà short)
    - Sortie long si prix touche plus bas 50j
    - Sortie short si prix touche plus haut 50j

Stratégie C : Core Trend-Following (Clenow)
    - Combine A et B
    - Entrée long : nouveau plus haut 100j ET EMA50 > EMA100
    - Entrée short : nouveau plus bas 100j ET EMA50 < EMA100
    - Sortie long : plus bas 50j OU EMA50 < EMA100
    - Sortie short : plus haut 50j OU EMA50 > EMA100
    - CAGR historique : 15.8%, Sharpe 0.70

Stratégie D : EWMAC (Carver)
    - Forecast continu EWMAC 16/64 dans [-20, +20]
    - Zone morte ±2 pour éviter le whipsaw
    - Hystérésis : on ne sort que si le forecast change de signe

Stratégie E : Turtle System 1 (Kaufman / Faith)
    - Entrée long/short sur breakout 20j
    - Sortie sur Donchian 10j opposé
    - Stop-loss à 2 × ATR depuis l'entrée

Stratégie F : Turtle System 2 (Kaufman / Faith)
    - Entrée long/short sur breakout 55j
    - Sortie sur Donchian 20j opposé
    - Stop-loss à 2 × ATR depuis l'entrée

Références :
    - Clenow: Following the Trend, ch. 4
    - Carver: Systematic Trading, ch. 7 & Appendix B
    - Kaufman: Trading Systems and Methods, ch. 5
"""

from config import EMA_FAST, EMA_SLOW


# ============================================================
# STRATÉGIE A : MA CROSSOVER (Clenow)
# ============================================================

def strategy_ma_crossover(date, row, instrument: str,
                           positions: dict, fast: int = EMA_FAST,
                           slow: int = EMA_SLOW) -> int:
    """
    Stratégie A — MA Crossover (Always-in-market).

    Règles :
      - Long si EMA(fast) > EMA(slow)
      - Short si EMA(fast) < EMA(slow)

    C'est la stratégie la plus simple et la base de comparaison
    pour toutes les autres.
    """
    ema_fast = row.get(f"ema_{fast}")
    ema_slow = row.get(f"ema_{slow}")

    if ema_fast is None or ema_slow is None:
        return 0  # Pas assez de données (warmup)

    # Vérifier que les valeurs ne sont pas NaN
    import math
    if math.isnan(ema_fast) or math.isnan(ema_slow):
        return 0

    if ema_fast > ema_slow:
        return 1   # Bullish
    else:
        return -1  # Bearish


# ============================================================
# STRATÉGIE B : BREAKOUT / DONCHIAN (Clenow)
# ============================================================

def strategy_breakout(date, row, instrument: str,
                       positions: dict) -> int:
    """
    Stratégie B — Breakout sur canaux de Donchian.

    Règles :
      - Entrée long  : Close ≥ entry_high (plus haut 100j)
      - Entrée short : Close ≤ entry_low (plus bas 100j)
      - Sortie long  : Close ≤ exit_low (plus bas 50j)
      - Sortie short : Close ≥ exit_high (plus haut 50j)
    """
    import math

    close = row.get("Close")
    entry_high = row.get("entry_high")
    entry_low = row.get("entry_low")
    exit_high = row.get("exit_high")
    exit_low = row.get("exit_low")

    # Warmup check
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [close, entry_high, entry_low, exit_high, exit_low]):
        return 0

    current_pos = positions.get(instrument)
    current_dir = current_pos.direction if current_pos else 0

    # Sorties d'abord
    if current_dir == 1 and close <= exit_low:
        return 0  # Sortie long
    if current_dir == -1 and close >= exit_high:
        return 0  # Sortie short

    # Entrées
    if close >= entry_high:
        return 1   # Nouveau plus haut → long
    if close <= entry_low:
        return -1  # Nouveau plus bas → short

    # Maintenir la position existante
    return current_dir


# ============================================================
# STRATÉGIE C : CORE TREND-FOLLOWING (Clenow)
# ============================================================

def strategy_core(date, row, instrument: str,
                   positions: dict,
                   fast: int = EMA_FAST,
                   slow: int = EMA_SLOW) -> int:
    """
    Stratégie C — Core Trend-Following (combinaison MA + Breakout).

    Règles (Clenow, Following the Trend ch. 4) :
      - Trend haussier si EMA50 > EMA100
      - Trend baissier si EMA50 < EMA100
      - Entrée long  : nouveau plus haut 100j ET trend haussier
      - Entrée short : nouveau plus bas 100j ET trend baissier
      - Sortie long  : prix ≤ plus bas 50j OU trend baissier
      - Sortie short : prix ≥ plus haut 50j OU trend haussier

    C'est la stratégie de référence Clenow : CAGR 15.8%, Sharpe 0.70
    """
    import math

    close = row.get("Close")
    ema_fast_val = row.get(f"ema_{fast}")
    ema_slow_val = row.get(f"ema_{slow}")
    entry_high = row.get("entry_high")
    entry_low = row.get("entry_low")
    exit_high = row.get("exit_high")
    exit_low = row.get("exit_low")

    # Warmup check
    vals = [close, ema_fast_val, ema_slow_val, entry_high, entry_low, exit_high, exit_low]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return 0

    # Trend direction
    trend_bullish = ema_fast_val > ema_slow_val

    current_pos = positions.get(instrument)
    current_dir = current_pos.direction if current_pos else 0

    # Sorties d'abord
    if current_dir == 1:
        if close <= exit_low or not trend_bullish:
            return 0  # Sortie long
    if current_dir == -1:
        if close >= exit_high or trend_bullish:
            return 0  # Sortie short

    # Entrées (seulement si pas déjà en position)
    if current_dir == 0:
        if close >= entry_high and trend_bullish:
            return 1   # Entrée long
        if close <= entry_low and not trend_bullish:
            return -1  # Entrée short

    # Maintenir la position existante
    return current_dir


# ============================================================
# STRATÉGIE D : EWMAC (Carver, Systematic Trading)
# ============================================================

def strategy_ewmac(date, row, instrument: str,
                    positions: dict) -> int:
    """
    Stratégie D — EWMAC Forecast (Carver, Systematic Trading ch.7, App. B).

    Le forecast EWMAC 16/64 est continu dans [-20, +20] avec une
    valeur absolue moyenne de ~10. On convertit en signal :
      - forecast > +2  → long
      - forecast < -2  → short
      - |forecast| ≤ 2 → flat (zone morte pour éviter le whipsaw)

    La zone morte de ±2 évite les retournements fréquents quand
    le forecast oscille autour de zéro. Carver recommande un seuil
    bas pour ne pas manquer les tendances naissantes.

    NOTE : Dans une version avancée (Phase position sizing),
    la taille de position sera proportionnelle au forecast
    (forecast/10 × taille standard). Pour l'instant on utilise
    la taille standard du backtester.

    Réf : Carver, Systematic Trading p.117-123, 282-285
          Paires recommandées : 2:8, 4:16, 8:32, 16:64, 32:128, 64:256
          Forecast scalars : Table 49, App. B
    """
    import math

    ewmac = row.get("ewmac_16_64")

    if ewmac is None or (isinstance(ewmac, float) and math.isnan(ewmac)):
        return 0

    # Zone morte : éviter le whipsaw autour de zéro
    DEAD_ZONE = 2.0

    current_pos = positions.get(instrument)
    current_dir = current_pos.direction if current_pos else 0

    # Si déjà en position, on ne sort que si le forecast passe
    # de l'autre côté de la zone morte (hystérésis)
    if current_dir == 1:
        return 0 if ewmac < -DEAD_ZONE else 1
    if current_dir == -1:
        return 0 if ewmac > DEAD_ZONE else -1

    # Pas en position : entrer si forecast suffisamment fort
    if ewmac > DEAD_ZONE:
        return 1
    if ewmac < -DEAD_ZONE:
        return -1
    return 0


# ============================================================
# STRATÉGIE E : TURTLE SYSTEM 1 (Kaufman / Faith)
# ============================================================

def strategy_turtle_s1(date, row, instrument: str,
                        positions: dict) -> int:
    """
    Stratégie E — Turtle Trading System 1 (Kaufman, TSM ch.5).

    Règles :
      - Entrée long  : Close ≥ plus haut 20j
      - Entrée short : Close ≤ plus bas 20j
      - Sortie long  : Close ≤ plus bas 10j
      - Sortie short : Close ≥ plus haut 10j
      - Stop-loss    : 2 × ATR depuis le prix d'entrée

    Kaufman note que le S1 original avait un filtre (ignorer
    le signal si le trade S1 précédent était profitable). On
    n'implémente pas ce filtre ici pour garder la stratégie
    comparable aux autres.

    Le stop ATR utilise l'ATR à l'entrée (stocké dans Position.entry_atr)
    pour être cohérent avec la philosophie Turtle : "never risk more than
    2% of equity on any trade" (2N stop).

    Réf : Kaufman, Trading Systems and Methods, ch.5 "Turtle Rules"
          Curtis Faith, Way of the Turtle
    """
    import math

    close = row.get("Close")
    entry_high = row.get("turtle_s1_entry_high")  # 20-day high
    entry_low = row.get("turtle_s1_entry_low")    # 20-day low
    exit_high = row.get("turtle_s1_exit_high")    # 10-day high
    exit_low = row.get("turtle_s1_exit_low")      # 10-day low
    current_atr = row.get("atr")

    # Warmup check
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [close, entry_high, entry_low, exit_high, exit_low]):
        return 0

    current_pos = positions.get(instrument)
    current_dir = current_pos.direction if current_pos else 0

    # Stop-loss ATR (2N rule)
    if current_pos and current_atr is not None:
        atr_at_entry = current_pos.entry_atr
        if atr_at_entry > 0:
            if current_dir == 1:
                stop_price = current_pos.entry_price - 2.0 * atr_at_entry
                if close <= stop_price:
                    return 0  # Stop-loss long
            elif current_dir == -1:
                stop_price = current_pos.entry_price + 2.0 * atr_at_entry
                if close >= stop_price:
                    return 0  # Stop-loss short

    # Sorties Donchian
    if current_dir == 1 and close <= exit_low:
        return 0  # Sortie long : plus bas 10j
    if current_dir == -1 and close >= exit_high:
        return 0  # Sortie short : plus haut 10j

    # Entrées Donchian (seulement si flat)
    if current_dir == 0:
        if close >= entry_high:
            return 1   # Breakout haussier 20j
        if close <= entry_low:
            return -1  # Breakout baissier 20j

    # Maintenir la position
    return current_dir


# ============================================================
# STRATÉGIE F : TURTLE SYSTEM 2 (Kaufman / Faith)
# ============================================================

def strategy_turtle_s2(date, row, instrument: str,
                        positions: dict) -> int:
    """
    Stratégie F — Turtle Trading System 2 (Kaufman, TSM ch.5).

    Règles :
      - Entrée long  : Close ≥ plus haut 55j
      - Entrée short : Close ≤ plus bas 55j
      - Sortie long  : Close ≤ plus bas 20j
      - Sortie short : Close ≥ plus haut 20j
      - Stop-loss    : 2 × ATR depuis le prix d'entrée

    Pas de filtre (contrairement au S1 original).
    Système plus lent, conçu pour capturer les grandes tendances.

    Réf : Kaufman, Trading Systems and Methods, ch.5 "Turtle Rules"
    """
    import math

    close = row.get("Close")
    entry_high = row.get("turtle_s2_entry_high")  # 55-day high
    entry_low = row.get("turtle_s2_entry_low")    # 55-day low
    exit_high = row.get("turtle_s2_exit_high")    # 20-day high
    exit_low = row.get("turtle_s2_exit_low")      # 20-day low
    current_atr = row.get("atr")

    # Warmup check
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [close, entry_high, entry_low, exit_high, exit_low]):
        return 0

    current_pos = positions.get(instrument)
    current_dir = current_pos.direction if current_pos else 0

    # Stop-loss ATR (2N rule)
    if current_pos and current_atr is not None:
        atr_at_entry = current_pos.entry_atr
        if atr_at_entry > 0:
            if current_dir == 1:
                stop_price = current_pos.entry_price - 2.0 * atr_at_entry
                if close <= stop_price:
                    return 0
            elif current_dir == -1:
                stop_price = current_pos.entry_price + 2.0 * atr_at_entry
                if close >= stop_price:
                    return 0

    # Sorties Donchian
    if current_dir == 1 and close <= exit_low:
        return 0
    if current_dir == -1 and close >= exit_high:
        return 0

    # Entrées (seulement si flat)
    if current_dir == 0:
        if close >= entry_high:
            return 1
        if close <= entry_low:
            return -1

    return current_dir


# ============================================================
# REGISTRE DES STRATÉGIES
# ============================================================

STRATEGIES = {
    "ma_crossover": {
        "func": strategy_ma_crossover,
        "name": "Stratégie A — MA Crossover (Clenow)",
        "description": "EMA 50/100, always-in-market",
        "reference": "Following the Trend, ch. 4",
    },
    "breakout": {
        "func": strategy_breakout,
        "name": "Stratégie B — Breakout Donchian (Clenow)",
        "description": "Entrée 100j, sortie 50j",
        "reference": "Following the Trend, ch. 4",
    },
    "core": {
        "func": strategy_core,
        "name": "Stratégie C — Core Trend-Following (Clenow)",
        "description": "Breakout + MA filter",
        "reference": "Following the Trend, ch. 4 — CAGR 15.8%, Sharpe 0.70",
    },
    "ewmac": {
        "func": strategy_ewmac,
        "name": "Stratégie D — EWMAC (Carver)",
        "description": "Forecast EWMAC 16/64 avec zone morte ±2",
        "reference": "Systematic Trading, ch. 7 & App. B — paires 2:8 à 64:256",
    },
    "turtle_s1": {
        "func": strategy_turtle_s1,
        "name": "Stratégie E — Turtle System 1 (Kaufman)",
        "description": "Breakout 20j, sortie 10j, stop 2×ATR",
        "reference": "Trading Systems and Methods, ch. 5 — Turtle Rules",
    },
    "turtle_s2": {
        "func": strategy_turtle_s2,
        "name": "Stratégie F — Turtle System 2 (Kaufman)",
        "description": "Breakout 55j, sortie 20j, stop 2×ATR",
        "reference": "Trading Systems and Methods, ch. 5 — Turtle Rules",
    },
}
