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

Référence : Following the Trend (Clenow, ch. 4)
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
}
