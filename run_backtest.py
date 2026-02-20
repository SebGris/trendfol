"""
Runner de backtest — Phase 2
==============================

Usage :
    python run_backtest.py                      # Stratégie A (MA Crossover) par défaut
    python run_backtest.py --strategy breakout   # Stratégie B
    python run_backtest.py --strategy core        # Stratégie C
    python run_backtest.py --capital 200000       # Capital initial personnalisé
    python run_backtest.py --risk-factor 0.001    # Risk factor différent
    python run_backtest.py --all                  # Toutes les stratégies (comparaison)

Exécution : signal jour J → trade à l'ouverture jour J+1
"""

import sys
import argparse

import pandas as pd
import numpy as np

from config import UNIVERSE, UNIVERSE_STARTER, UNIVERSE_MAP, UNIVERSE_BY_SECTOR, RISK_FACTOR
from database import load_prices
from indicators import compute_all_indicators
from backtester import BacktestEngine, CostConfig
from strategies import STRATEGIES
from metrics import compute_metrics, format_metrics


def load_and_prepare_data(universe=None,
                          verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Charge les données depuis SQLite et calcule les indicateurs.

    Args:
        universe: liste d'Instrument (défaut: UNIVERSE complet)

    Returns:
        {instrument_name: DataFrame avec OHLCV + indicateurs}
    """
    if universe is None:
        universe = UNIVERSE

    data = {}
    skipped = []
    for inst in universe:
        if verbose:
            print(f"  📂 Chargement {inst.name}...", end=" ")

        df = load_prices(inst.name)
        if df.empty:
            if verbose:
                print("⚠️  Vide (pas encore téléchargé ?)")
            skipped.append(inst.name)
            continue

        # Calculer les indicateurs
        enriched = compute_all_indicators(df)
        data[inst.name] = enriched

        if verbose:
            print(f"✅ {len(enriched)} lignes")

    if skipped and verbose:
        print(f"\n  ⚠️  {len(skipped)} instruments sans données : {', '.join(skipped)}")
        print(f"  → Exécuter : python main.py --download")

    return data


def get_instrument_configs() -> dict[str, dict]:
    """Construit le dictionnaire de configuration des instruments."""
    return {
        inst.name: {
            "point_value": inst.point_value,
            "currency": inst.currency,
            "instrument_type": inst.instrument_type,
            "sector": inst.sector,
        }
        for inst in UNIVERSE
    }


def run_single_backtest(strategy_key: str,
                         data: dict,
                         instruments: dict,
                         initial_capital: float = 100_000,
                         risk_factor: float = RISK_FACTOR,
                         fractional: bool = False,
                         verbose: bool = True) -> dict:
    """
    Exécute un backtest pour une stratégie donnée.

    Returns:
        dict avec 'metrics', 'equity_curve', 'trades', 'strategy_info'
    """
    strategy_info = STRATEGIES[strategy_key]
    strategy_func = strategy_info["func"]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🎯 {strategy_info['name']}")
        print(f"   {strategy_info['description']}")
        print(f"   Réf: {strategy_info['reference']}")
        mode = "fractionnaire (CFD)" if fractional else "contrats entiers (futures)"
        print(f"   Sizing: {mode}, risk factor {risk_factor}")
        print(f"{'=' * 60}")

    # Créer le moteur
    engine = BacktestEngine(
        initial_capital=initial_capital,
        cost_config=CostConfig(),
        risk_factor=risk_factor,
        fractional=fractional,
    )

    # Exécuter
    equity_curve, trades = engine.run(data, strategy_func, instruments, progress=verbose)

    # Calculer les métriques
    metrics = compute_metrics(equity_curve, trades)

    if verbose:
        print()
        print(format_metrics(metrics, strategy_info["name"]))

    # Détail des trades par instrument
    if verbose and trades:
        print(f"\n  📋 RÉPARTITION PAR INSTRUMENT")
        print(f"  {'─' * 50}")
        trade_df = pd.DataFrame(trades)
        for inst_name in sorted(trade_df["instrument"].unique()):
            inst_trades = trade_df[trade_df["instrument"] == inst_name]
            n = len(inst_trades)
            pnl = inst_trades["pnl"].sum()
            wins = (inst_trades["pnl"] > 0).sum()
            print(f"  {inst_name:20s} : {n:3} trades, "
                  f"PnL ${pnl:>+12,.0f}, "
                  f"Win {wins}/{n} ({wins/n*100:.0f}%)")

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "strategy_info": strategy_info,
    }


def run_comparison(data: dict, instruments: dict,
                    initial_capital: float = 100_000,
                    risk_factor: float = RISK_FACTOR,
                    fractional: bool = False) -> None:
    """Compare toutes les stratégies côte à côte."""
    print("\n" + "🏆" * 20)
    print("  COMPARAISON DE TOUTES LES STRATÉGIES")
    mode = "fractionnaire (CFD)" if fractional else "contrats entiers (futures)"
    print(f"  Capital: ${initial_capital:,.0f} | Risk: {risk_factor} | Sizing: {mode}")
    print("🏆" * 20)

    results = {}
    for key in STRATEGIES:
        result = run_single_backtest(
            key, data, instruments,
            initial_capital, risk_factor, fractional, verbose=False
        )
        results[key] = result
        m = result["metrics"]
        print(f"\n  ✅ {STRATEGIES[key]['name']}")
        print(f"     CAGR: {m.cagr_pct:.2f}% | Sharpe: {m.sharpe_ratio:.3f} | "
              f"MaxDD: {m.max_drawdown_pct:.1f}% | Trades: {m.total_trades}")

    # Tableau comparatif
    print(f"\n{'═' * 70}")
    print(f"{'MÉTRIQUE':25s} | ", end="")
    for key in STRATEGIES:
        label = key[:12].center(14)
        print(f"{label} | ", end="")
    print()
    print(f"{'─' * 70}")

    rows = [
        ("CAGR (%)", lambda m: f"{m.cagr_pct:>+.2f}%"),
        ("Sharpe Ratio", lambda m: f"{m.sharpe_ratio:>.3f}"),
        ("Sortino Ratio", lambda m: f"{m.sortino_ratio:>.3f}"),
        ("Max Drawdown (%)", lambda m: f"{m.max_drawdown_pct:>.1f}%"),
        ("Volatilité ann. (%)", lambda m: f"{m.annualized_vol_pct:>.1f}%"),
        ("Calmar Ratio", lambda m: f"{m.calmar_ratio:>.3f}"),
        ("Trades", lambda m: f"{m.total_trades:>d}"),
        ("Win Rate (%)", lambda m: f"{m.win_rate_pct:>.1f}%"),
        ("Profit Factor", lambda m: f"{m.profit_factor:>.2f}"),
        ("Meilleur mois (%)", lambda m: f"{m.best_month_pct:>+.1f}%"),
        ("Pire mois (%)", lambda m: f"{m.worst_month_pct:>+.1f}%"),
        ("Mois profitables (%)", lambda m: f"{m.pct_profitable_months:>.1f}%"),
    ]

    for label, fmt_func in rows:
        print(f"{label:25s} | ", end="")
        for key in STRATEGIES:
            m = results[key]["metrics"]
            val = fmt_func(m).center(14)
            print(f"{val} | ", end="")
        print()

    print(f"{'═' * 70}")

    # Référence Clenow
    print(f"\n  📚 Référence Clenow (Following the Trend, Table 4.4) :")
    print(f"     Core model : CAGR 15.8%, Sharpe 0.70, MaxDD -39.4%, Vol 25.9%")
    print(f"     Breakout   : CAGR 14.3%, Sharpe 0.62, MaxDD -47.2%, Vol 27.5%")
    print(f"     MA Cross   : CAGR 12.7%, Sharpe 0.54, MaxDD -64.7%, Vol 30.9%")
    print(f"\n  ⚠️  Nos résultats diffèrent car :")
    print(f"     - Univers réduit ({len(data)} instruments vs ~70 chez Clenow)")
    print(f"     - Données yfinance (vs données institutionnelles)")
    print(f"     - Période différente")

    # Résumé par secteur
    print(f"\n  📊 RÉPARTITION PAR SECTEUR :")
    sectors = {}
    for inst_name in data:
        inst = UNIVERSE_MAP.get(inst_name)
        if inst:
            sectors.setdefault(inst.sector, []).append(inst_name)
    for sector, names in sorted(sectors.items()):
        print(f"     {sector:20s} : {len(names)} — {', '.join(names)}")

    if not fractional:
        print(f"\n  💡 CONSEIL : Avec ${initial_capital:,.0f}, la plupart des instruments")
        print(f"     donnent 0 contrats (ATR × PointValue > capital × risk_factor).")
        print(f"     → Relancer avec --fractional pour un backtest significatif :")
        print(f"     python run_backtest.py --all --fractional")


def main():
    parser = argparse.ArgumentParser(description="Backtest Trend Following")
    parser.add_argument("--strategy", type=str, default="ma_crossover",
                        choices=list(STRATEGIES.keys()),
                        help="Stratégie à tester")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Capital initial en USD")
    parser.add_argument("--risk-factor", type=float, default=RISK_FACTOR,
                        help="Risk factor (Clenow: 0.002 = 20 bps)")
    parser.add_argument("--all", action="store_true",
                        help="Comparer toutes les stratégies")
    parser.add_argument("--fractional", action="store_true",
                        help="Contrats fractionnaires (CFD mode). "
                             "Indispensable si capital < $500k")
    parser.add_argument("--universe", type=str, default="full",
                        choices=["starter", "full"],
                        help="Univers d'instruments : "
                             "'starter' (5 Carver) ou 'full' (25 multi-secteurs)")
    args = parser.parse_args()

    # Sélectionner l'univers
    if args.universe == "starter":
        universe = UNIVERSE_STARTER
    else:
        universe = UNIVERSE

    # Charger les données
    print(f"\n{'=' * 60}")
    print(f"📂 CHARGEMENT DES DONNÉES — univers '{args.universe}' ({len(universe)} instruments)")
    print(f"{'=' * 60}")
    data = load_and_prepare_data(universe)

    if not data:
        print("❌ Aucune donnée disponible. Exécuter d'abord python main.py")
        sys.exit(1)

    # Avertir si on a moins d'instruments que prévu
    if len(data) < len(universe):
        print(f"\n  ⚠️  {len(data)}/{len(universe)} instruments chargés.")
        print(f"  → Télécharger les manquants : python main.py --download")

    instruments = get_instrument_configs()

    if args.all:
        run_comparison(data, instruments, args.capital, args.risk_factor,
                       args.fractional)
    else:
        run_single_backtest(
            args.strategy, data, instruments,
            args.capital, args.risk_factor, args.fractional,
        )


if __name__ == "__main__":
    main()
