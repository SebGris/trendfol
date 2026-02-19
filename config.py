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

UNIVERSE = [
    Instrument(
        name="S&P 500",
        ticker="ES=F",         # E-mini S&P 500 futures
        sector="equities",
        point_value=50.0,
        currency="USD",
        instrument_type="futures",
    ),
    Instrument(
        name="Gold",
        ticker="GC=F",         # Comex Gold futures
        sector="non_agricultural",
        point_value=100.0,
        currency="USD",
        instrument_type="futures",
    ),
    Instrument(
        name="Corn",
        ticker="ZC=F",         # CBOT Corn futures
        sector="agricultural",
        point_value=50.0,
        currency="USD",
        instrument_type="futures",
    ),
    Instrument(
        name="Euro Stoxx 50",
        ticker="^STOXX50E",    # Euro Stoxx 50 index
        sector="equities",
        point_value=10.0,
        currency="EUR",
        instrument_type="index",
    ),
    Instrument(
        name="AUDUSD",
        ticker="AUDUSD=X",    # AUD/USD spot FX
        sector="currencies",
        point_value=100_000.0,  # Standard FX lot
        currency="USD",
        instrument_type="fx",
    ),
]

# Dictionnaire rapide par nom
UNIVERSE_MAP = {inst.name: inst for inst in UNIVERSE}

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
RISK_FACTOR = 0.002         # Clenow: 20 basis points pour position sizing
