"""
Moteur de backtest — simulation jour par jour.
=================================================

Architecture :
  - Le moteur itère sur chaque jour
  - La stratégie produit un signal (jour J)
  - L'exécution se fait au prix d'ouverture du jour J+1 (anti look-ahead)
  - Le position sizing est calculé à l'entrée (Clenow: taille constante)
  - Les commissions sont déduites à chaque trade

Conventions :
  - Signal +1 = long, -1 = short, 0 = flat
  - Position size en nombre de contrats (arrondi au plancher)
  - PnL calculé en $ (prix × point_value × nb_contrats)

Référence : Clenow (Following the Trend), principes_backtesting_clenow.md
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from config import RISK_FACTOR


# ============================================================
# CONFIGURATION DES COÛTS
# ============================================================

@dataclass
class CostConfig:
    """Coûts de transaction."""
    commission_per_contract: float = 0.85      # Commission broker
    exchange_fee_per_contract: float = 1.50     # Frais de bourse
    slippage_pct: float = 0.0005               # 5 bps de slippage estimé

    @property
    def total_per_contract(self) -> float:
        return self.commission_per_contract + self.exchange_fee_per_contract


# ============================================================
# ÉTAT DU BACKTEST
# ============================================================

@dataclass
class Position:
    """Position ouverte."""
    instrument: str
    direction: int          # +1 long, -1 short
    contracts: float        # int pour futures, float pour CFD/fractionnaire
    entry_price: float
    entry_date: str
    point_value: float
    entry_atr: float        # ATR à l'entrée (pour trailing stop)
    peak_price: float = 0   # Pour trailing stop (long: plus haut)
    trough_price: float = float("inf")  # Pour trailing stop (short: plus bas)


@dataclass
class Trade:
    """Trade fermé."""
    instrument: str
    direction: int
    contracts: float        # int pour futures, float pour CFD/fractionnaire
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    pnl: float              # P&L en $ après coûts
    pnl_pct: float           # P&L en % du capital à l'entrée
    gross_pnl: float         # P&L brut (avant coûts)
    costs: float             # Coûts totaux (commissions + slippage)
    holding_days: int


# ============================================================
# MOTEUR DE BACKTEST
# ============================================================

class BacktestEngine:
    """
    Moteur de backtest événementiel jour par jour.

    Usage :
        engine = BacktestEngine(initial_capital=100_000)
        engine.run(data, strategy_func, instrument_config)

    Modes de sizing :
        - fractional=False (défaut) : nb entier de contrats (arrondi plancher)
          → réaliste mais nécessite un capital élevé (~$500k+ pour futures)
        - fractional=True : contrats fractionnaires (comme CFDs)
          → permet de tester la logique de la stratégie avec n'importe quel capital
    """

    def __init__(self, initial_capital: float = 100_000,
                 cost_config: CostConfig = None,
                 risk_factor: float = RISK_FACTOR,
                 fractional: bool = False):
        self.initial_capital = initial_capital
        self.costs = cost_config or CostConfig()
        self.risk_factor = risk_factor
        self.fractional = fractional

        # État
        self.capital = initial_capital
        self.positions: dict[str, Position] = {}  # instrument -> Position
        self.trades: list[Trade] = []
        self.equity_history: list[tuple] = []     # [(date, equity)]

    def run(self, data: dict[str, pd.DataFrame],
            strategy_func,
            instruments: dict[str, dict],
            progress: bool = True) -> tuple[pd.Series, list[dict]]:
        """
        Exécute le backtest.

        Args:
            data: {instrument_name: DataFrame OHLCV avec indicateurs}
            strategy_func: function(date, row, instrument, positions) -> signal
            instruments: {instrument_name: {point_value, currency, ...}}
            progress: afficher la progression

        Returns:
            (equity_curve, trades_list)
        """
        # Aligner toutes les dates
        all_dates = sorted(set().union(*(df.index for df in data.values())))

        # Pré-calculer les index de dates par instrument (évite O(n²))
        date_indices = {}
        for inst_name, df in data.items():
            idx_list = list(df.index)
            date_indices[inst_name] = {d: i for i, d in enumerate(idx_list)}

        if progress:
            n_total = len(all_dates)
            print(f"  📈 Backtest : {n_total} jours, "
                  f"{len(data)} instruments, capital initial ${self.initial_capital:,.0f}")

        for i, date in enumerate(all_dates):
            # 1. Mettre à jour le mark-to-market des positions ouvertes
            unrealized_pnl = self._mark_to_market(date, data)

            # 2. Enregistrer l'equity
            equity = self.capital + unrealized_pnl
            self.equity_history.append((date, equity))

            # 3. Pour chaque instrument, calculer le signal
            for inst_name, df in data.items():
                if date not in df.index:
                    continue

                row = df.loc[date]
                inst_config = instruments[inst_name]

                # Signal de la stratégie (jour J)
                signal = strategy_func(date, row, inst_name, self.positions)

                # Vérifier s'il y a un changement de position
                current_pos = self.positions.get(inst_name)
                current_dir = current_pos.direction if current_pos else 0

                if signal != current_dir:
                    # Exécution au prix d'ouverture J+1
                    idx_map = date_indices.get(inst_name, {})
                    next_idx = idx_map.get(date, -1)
                    if next_idx >= 0 and next_idx + 1 < len(df.index):
                        next_date = df.index[next_idx + 1]
                        exec_price = df.loc[next_date, "Open"]
                        exec_date = next_date

                        # Fermer la position existante
                        if current_pos:
                            self._close_position(inst_name, exec_price,
                                                 str(exec_date.date()),
                                                 inst_config)

                        # Ouvrir la nouvelle position (si signal != 0)
                        if signal != 0:
                            self._open_position(
                                inst_name, signal, exec_price,
                                str(exec_date.date()),
                                row.get("atr", 0),
                                inst_config,
                            )

            # Mettre à jour les trailing stops
            self._update_trailing(date, data)

        # Dernière equity
        if self.equity_history:
            final_unrealized = self._mark_to_market(all_dates[-1], data)
            # Remplacer le dernier point
            last_date = self.equity_history[-1][0]
            self.equity_history[-1] = (last_date, self.capital + final_unrealized)

        # Construire l'equity curve
        dates, equities = zip(*self.equity_history) if self.equity_history else ([], [])
        equity_curve = pd.Series(equities, index=pd.DatetimeIndex(dates), name="equity")

        # Convertir les trades en dicts
        trades_list = [
            {
                "instrument": t.instrument,
                "direction": t.direction,
                "contracts": t.contracts,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "gross_pnl": t.gross_pnl,
                "costs": t.costs,
                "holding_days": t.holding_days,
            }
            for t in self.trades
        ]

        if progress:
            print(f"  ✅ Terminé : {len(self.trades)} trades, "
                  f"equity finale ${equity_curve.iloc[-1]:,.0f}")

        return equity_curve, trades_list

    # ============================================================
    # GESTION DES POSITIONS
    # ============================================================

    def _open_position(self, instrument: str, direction: int,
                       price: float, date: str, current_atr: float,
                       inst_config: dict) -> None:
        """Ouvre une nouvelle position avec sizing ATR (Clenow)."""
        point_value = inst_config["point_value"]

        # Position sizing : Contracts = (Equity × risk_factor) / (ATR × PointValue)
        if current_atr > 0 and point_value > 0:
            raw_contracts = (self.capital * self.risk_factor) / (current_atr * point_value)

            if self.fractional:
                contracts = raw_contracts  # CFD mode : fractionnaire
            else:
                contracts = int(raw_contracts)  # Futures : entier arrondi plancher
        else:
            contracts = 0

        if contracts <= 0:
            return

        # Coûts d'entrée
        entry_cost = self._calc_costs(contracts, price)
        self.capital -= entry_cost

        self.positions[instrument] = Position(
            instrument=instrument,
            direction=direction,
            contracts=contracts,
            entry_price=price,
            entry_date=date,
            point_value=point_value,
            entry_atr=current_atr,
            peak_price=price if direction == 1 else 0,
            trough_price=price if direction == -1 else float("inf"),
        )

    def _close_position(self, instrument: str, price: float,
                        date: str, inst_config: dict) -> None:
        """Ferme une position et enregistre le trade."""
        pos = self.positions.pop(instrument, None)
        if pos is None:
            return

        # P&L brut
        price_diff = (price - pos.entry_price) * pos.direction
        gross_pnl = price_diff * pos.contracts * pos.point_value

        # Coûts de sortie
        exit_cost = self._calc_costs(pos.contracts, price)

        # P&L net
        entry_cost = self._calc_costs(pos.contracts, pos.entry_price)
        total_costs = entry_cost + exit_cost
        net_pnl = gross_pnl - exit_cost  # entry_cost déjà déduit à l'ouverture

        # Mettre à jour le capital
        self.capital += gross_pnl - exit_cost

        # P&L en % de l'equity au moment de l'entrée
        # (approximation : on utilise le capital courant comme proxy)
        entry_equity = self.capital - net_pnl + total_costs
        pnl_pct = net_pnl / entry_equity if entry_equity > 0 else 0

        # Durée
        try:
            holding_days = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days
        except Exception:
            holding_days = 0

        self.trades.append(Trade(
            instrument=pos.instrument,
            direction=pos.direction,
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_date=pos.entry_date,
            exit_date=date,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            gross_pnl=gross_pnl,
            costs=total_costs,
            holding_days=holding_days,
        ))

    def _calc_costs(self, contracts: int, price: float) -> float:
        """Calcule les coûts pour un nombre de contrats."""
        commission = contracts * self.costs.total_per_contract
        slippage = contracts * price * self.costs.slippage_pct
        return commission + slippage

    # ============================================================
    # MARK-TO-MARKET
    # ============================================================

    def _mark_to_market(self, date, data: dict) -> float:
        """Calcule le P&L non réalisé de toutes les positions ouvertes."""
        unrealized = 0
        for inst_name, pos in self.positions.items():
            df = data.get(inst_name)
            if df is None or date not in df.index:
                continue
            current_price = df.loc[date, "Close"]
            price_diff = (current_price - pos.entry_price) * pos.direction
            unrealized += price_diff * pos.contracts * pos.point_value
        return unrealized

    def _update_trailing(self, date, data: dict) -> None:
        """Met à jour les peak/trough pour les trailing stops."""
        for inst_name, pos in self.positions.items():
            df = data.get(inst_name)
            if df is None or date not in df.index:
                continue
            high = df.loc[date, "High"]
            low = df.loc[date, "Low"]
            if pos.direction == 1:
                pos.peak_price = max(pos.peak_price, high)
            else:
                pos.trough_price = min(pos.trough_price, low)
