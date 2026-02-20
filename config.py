"""
Configuration du projet Trend Following
Univers réduit Carver (Starter System) — 5 instruments
"""

from dataclasses import dataclass
from pathlib import Path

# ============================================================
# CHEMINS
# ============================================================
PROJECT_DIR = Path(__file__).parent
DB_PATH = PROJECT_DIR / "data" / "market_data.db"

# ============================================================
# UNIVERS D'INSTRUMENTS
# ============================================================
# Mapping : nom lisible → ticker yfinance
# On utilise des proxies accessibles via yfinance :
#   - Futures continus (*=F) quand disponibles
#   - FX via Yahoo (*=X)
#   - Index quand pas de futures dispo

@dataclass
class Instrument:
    name: str           # Nom lisible
    ticker: str         # Ticker yfinance
    sector: str         # Secteur (pour diversification)
    point_value: float  # Valeur d'un point (pour sizing futures)
    currency: str       # Devise du contrat
    instrument_type: str  # 'futures', 'fx', 'etf', 'index'

# -----------------------------------------------------------
# UNIVERS ÉLARGI (~25 instruments, 5 secteurs Clenow)
# -----------------------------------------------------------
# Sélection : les plus liquides par secteur, disponibles
# sur yfinance avec 15+ ans d'historique.
#
# Point values = valeurs réelles des contrats CME/CBOT/Eurex
# (utilisées pour le sizing même en mode fractionnaire)
# -----------------------------------------------------------

UNIVERSE = [
    # ── AGRICULTURAL (6) ──────────────────────────────────
    Instrument("Corn", "ZC=F", "agricultural", 50.0, "USD", "futures"),
    Instrument("Soybeans", "ZS=F", "agricultural", 50.0, "USD", "futures"),
    Instrument("Wheat", "ZW=F", "agricultural", 50.0, "USD", "futures"),
    Instrument("Sugar", "SB=F", "agricultural", 1120.0, "USD", "futures"),
    Instrument("Cotton", "CT=F", "agricultural", 500.0, "USD", "futures"),
    Instrument("Coffee", "KC=F", "agricultural", 375.0, "USD", "futures"),

    # ── NON-AGRICULTURAL (5) ──────────────────────────────
    Instrument("Gold", "GC=F", "non_agricultural", 100.0, "USD", "futures"),
    Instrument("Silver", "SI=F", "non_agricultural", 5000.0, "USD", "futures"),
    Instrument("Crude Oil", "CL=F", "non_agricultural", 1000.0, "USD", "futures"),
    Instrument("Natural Gas", "NG=F", "non_agricultural", 10000.0, "USD", "futures"),
    Instrument("Copper", "HG=F", "non_agricultural", 25000.0, "USD", "futures"),

    # ── CURRENCIES (5) ────────────────────────────────────
    Instrument("EURUSD", "EURUSD=X", "currencies", 125000.0, "USD", "fx"),
    Instrument("GBPUSD", "GBPUSD=X", "currencies", 62500.0, "USD", "fx"),
    Instrument("AUDUSD", "AUDUSD=X", "currencies", 100000.0, "USD", "fx"),
    Instrument("JPYUSD", "JPY=X", "currencies", 12500000.0, "USD", "fx"),
    Instrument("CADUSD", "CADUSD=X", "currencies", 100000.0, "USD", "fx"),

    # ── EQUITIES (5) ──────────────────────────────────────
    Instrument("S&P 500", "ES=F", "equities", 50.0, "USD", "futures"),
    Instrument("Nasdaq 100", "NQ=F", "equities", 20.0, "USD", "futures"),
    Instrument("Euro Stoxx 50", "^STOXX50E", "equities", 10.0, "EUR", "index"),
    Instrument("Nikkei 225", "^N225", "equities", 1000.0, "JPY", "index"),
    Instrument("FTSE 100", "^FTSE", "equities", 10.0, "GBP", "index"),

    # ── RATES (4) ─────────────────────────────────────────
    Instrument("US 10Y Note", "ZN=F", "rates", 1000.0, "USD", "futures"),
    Instrument("US 30Y Bond", "ZB=F", "rates", 1000.0, "USD", "futures"),
    Instrument("US 5Y Note", "ZF=F", "rates", 1000.0, "USD", "futures"),
    Instrument("Eurodollar", "GE=F", "rates", 2500.0, "USD", "futures"),
]

# Dictionnaire rapide par nom
UNIVERSE_MAP = {inst.name: inst for inst in UNIVERSE}

# Sous-univers prédéfinis
UNIVERSE_STARTER = [i for i in UNIVERSE if i.name in {
    "S&P 500", "Gold", "Corn", "Euro Stoxx 50", "AUDUSD",
}]

# -----------------------------------------------------------
# MICRO UNIVERS — Petit capital (€8 000+)
# -----------------------------------------------------------
# Basé sur Carver, Leveraged Trading, ch. 4-7 :
#   - AUDUSD : capital minimum ~$1 500 (spot FX/CFD)
#   - Gold   : capital minimum ~$2 400 (CFD/spread bet)
#   - 2 instruments de classes d'actifs différentes
#     → IDM = 1.2, risk target compte = 13% (Table 44)
#     → risk target instrument = 1.2 × 13% = 15.6%
# -----------------------------------------------------------
UNIVERSE_MICRO = [i for i in UNIVERSE if i.name in {"AUDUSD", "Gold"}]

# Paramètres Carver par taille d'univers (Leveraged Trading, Tables 43-44)
# clé = nombre d'instruments (classes d'actifs différentes)
CARVER_PARAMS = {
    1: {"idm": 1.00, "account_target": 0.12, "instrument_target": 0.120},
    2: {"idm": 1.20, "account_target": 0.13, "instrument_target": 0.156},
    3: {"idm": 1.48, "account_target": 0.14, "instrument_target": 0.207},
    4: {"idm": 1.56, "account_target": 0.17, "instrument_target": 0.265},
    5: {"idm": 1.70, "account_target": 0.19, "instrument_target": 0.323},
}

def carver_risk_factor(n_instruments: int) -> float:
    """
    Calcule le risk_factor équivalent Clenow à partir du
    framework Carver (Leveraged Trading).

    La formule Clenow : contracts = (equity × rf) / (ATR × pv)
    donne un impact journalier par position = equity × rf.

    Pour n positions :
        vol_annuelle ≈ rf × n × √256  (en simplifiant)

    On veut : vol_annuelle = instrument_target (15.6% pour 2 inst.)
    Mais chaque instrument ne reçoit que equity/n du capital.
    Comme le backtester utilise le capital total :
        rf = instrument_target / (n × √256)

    Réf : Carver, Leveraged Trading, Tables 43-44, ch. 7
    """
    import math
    params = CARVER_PARAMS.get(n_instruments, CARVER_PARAMS[5])
    return params["instrument_target"] / (n_instruments * math.sqrt(256))

UNIVERSE_BY_SECTOR = {}
for inst in UNIVERSE:
    UNIVERSE_BY_SECTOR.setdefault(inst.sector, []).append(inst)

# ============================================================
# PARAMÈTRES DE TÉLÉCHARGEMENT
# ============================================================
DOWNLOAD_START = "2005-01-01"   # ~20 ans d'historique
DOWNLOAD_INTERVAL = "1d"        # Daily data (seul timeframe utile)

# ============================================================
# PARAMÈTRES DE NETTOYAGE (Quality checks)
# ============================================================
# Seuil de variation journalière suspecte (Clenow: > 4% = suspect)
OUTLIER_DAILY_RETURN_THRESHOLD = 0.15   # 15% — on flag, on ne supprime pas
# Nombre max de jours consécutifs sans données
MAX_CONSECUTIVE_MISSING_DAYS = 5
# Vérification High >= Low, High >= Open/Close, Low <= Open/Close
VALIDATE_OHLC_CONSISTENCY = True

# ============================================================
# PARAMÈTRES INDICATEURS (Phase 1b — pas encore implémentés)
# ============================================================
ATR_PERIOD = 20             # Clenow: 20 jours EMA
EMA_FAST = 50               # Clenow: EMA rapide
EMA_SLOW = 100              # Clenow: EMA lente
VOLATILITY_WINDOW = 256     # Annualisation sur ~1 an de trading days

# Risk factor : impact journalier cible PAR POSITION
# ─────────────────────────────────────────────────────────
# Formule vol portfolio ≈ risk_factor × √positions × √256
# Avec 24 positions :
#   0.002  → vol ~80% (agressif — défaut Clenow pour ~50+ instruments)
#   0.001  → vol ~40% (modéré)
#   0.0006 → vol ~25% (cible Clenow avec notre univers de 24)
# ─────────────────────────────────────────────────────────
RISK_FACTOR = 0.002         # Clenow: 20 basis points pour position sizing
