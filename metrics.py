"""
Métriques de performance pour le backtesting.

Calcule toutes les métriques standard :
  - CAGR, rendement total
  - Sharpe, Sortino, Calmar
  - Max Drawdown, durée du drawdown
  - Win rate, profit factor, payoff ratio
  - Nombre de trades, sample error

Référence : principes_backtesting_clenow.md
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    """Résultats complets d'un backtest."""
    # Rendement
    total_return_pct: float
    cagr_pct: float
    # Risque
    annualized_vol_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    # Ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    payoff_ratio: float
    sample_error_pct: float
    # Divers
    best_month_pct: float
    worst_month_pct: float
    pct_profitable_months: float
    avg_trade_duration_days: float
    # Séries
    equity_curve: pd.Series = None
    drawdown_series: pd.Series = None
    monthly_returns: pd.Series = None


def compute_metrics(equity_curve: pd.Series,
                    trades: list[dict],
                    trading_days_per_year: int = 256,
                    risk_free_rate: float = 0.0) -> BacktestMetrics:
    """
    Calcule toutes les métriques à partir de l'equity curve et de la liste des trades.

    Args:
        equity_curve: Series indexée par date, valeur du portefeuille
        trades: liste de dicts avec au minimum 'entry_date', 'exit_date',
                'pnl', 'pnl_pct', 'direction'
        trading_days_per_year: 256 (Carver) ou 252
        risk_free_rate: taux sans risque annuel (0 pour Sharpe simplifié)

    Returns:
        BacktestMetrics complet
    """
    # ---- Rendements journaliers ----
    daily_returns = equity_curve.pct_change().dropna()
    n_days = len(daily_returns)
    n_years = n_days / trading_days_per_year

    # ---- Rendement total et CAGR ----
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    if n_years > 0 and equity_curve.iloc[-1] > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1
    else:
        cagr = -1.0  # Perte totale

    # ---- Volatilité annualisée ----
    ann_vol = daily_returns.std() * np.sqrt(trading_days_per_year)

    # ---- Sharpe Ratio (simplifié, risk-free = 0) ----
    excess_returns = daily_returns - risk_free_rate / trading_days_per_year
    sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(trading_days_per_year)
              if daily_returns.std() > 0 else 0)

    # ---- Sortino Ratio ----
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(trading_days_per_year) if len(downside) > 0 else 0
    sortino = cagr / downside_std if downside_std > 0 else 0

    # ---- Drawdown ----
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # Durée du max drawdown
    dd_duration = _max_drawdown_duration(drawdown)

    # ---- Calmar Ratio ----
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # ---- Rendements mensuels ----
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    best_month = monthly.max() if len(monthly) > 0 else 0
    worst_month = monthly.min() if len(monthly) > 0 else 0
    pct_profitable_months = (monthly > 0).mean() * 100 if len(monthly) > 0 else 0

    # ---- Statistiques des trades ----
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]
        durations = [
            (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
            for t in trades if t.get("exit_date")
        ]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_pcts = [p for p in pnl_pcts if p > 0]
        loss_pcts = [p for p in pnl_pcts if p <= 0]

        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_trades * 100

        avg_win = np.mean(win_pcts) * 100 if win_pcts else 0
        avg_loss = np.mean(loss_pcts) * 100 if loss_pcts else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Sample error = 100% / √n (Clenow)
        sample_error = 100 / np.sqrt(n_trades) if n_trades > 0 else 100

        avg_duration = np.mean(durations) if durations else 0
    else:
        n_wins = n_losses = 0
        win_rate = avg_win = avg_loss = 0
        profit_factor = payoff_ratio = 0
        sample_error = 100
        avg_duration = 0

    return BacktestMetrics(
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        annualized_vol_pct=ann_vol * 100,
        max_drawdown_pct=max_dd * 100,
        max_drawdown_duration_days=dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=n_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        sample_error_pct=sample_error,
        best_month_pct=best_month * 100,
        worst_month_pct=worst_month * 100,
        pct_profitable_months=pct_profitable_months,
        avg_trade_duration_days=avg_duration,
        equity_curve=equity_curve,
        drawdown_series=drawdown,
        monthly_returns=monthly,
    )


def format_metrics(m: BacktestMetrics, name: str = "Backtest") -> str:
    """Formatage texte complet des métriques."""
    lines = [
        f"{'═' * 60}",
        f"📊 RÉSULTATS : {name}",
        f"{'═' * 60}",
        f"",
        f"  RENDEMENT",
        f"  {'─' * 40}",
        f"  Rendement total     : {m.total_return_pct:>10.1f}%",
        f"  CAGR                : {m.cagr_pct:>10.2f}%",
        f"  Meilleur mois       : {m.best_month_pct:>+10.2f}%",
        f"  Pire mois           : {m.worst_month_pct:>+10.2f}%",
        f"  Mois profitables    : {m.pct_profitable_months:>10.1f}%",
        f"",
        f"  RISQUE",
        f"  {'─' * 40}",
        f"  Volatilité annuelle : {m.annualized_vol_pct:>10.2f}%",
        f"  Max Drawdown        : {m.max_drawdown_pct:>10.2f}%",
        f"  Durée max DD        : {m.max_drawdown_duration_days:>10} jours",
        f"",
        f"  RATIOS",
        f"  {'─' * 40}",
        f"  Sharpe (0%)         : {m.sharpe_ratio:>10.3f}",
        f"  Sortino             : {m.sortino_ratio:>10.3f}",
        f"  Calmar              : {m.calmar_ratio:>10.3f}",
        f"",
        f"  TRADES",
        f"  {'─' * 40}",
        f"  Nombre total        : {m.total_trades:>10}",
        f"  Gagnants            : {m.winning_trades:>10} ({m.win_rate_pct:.1f}%)",
        f"  Perdants            : {m.losing_trades:>10} ({100 - m.win_rate_pct:.1f}%)",
        f"  Gain moyen          : {m.avg_win_pct:>+10.2f}%",
        f"  Perte moyenne       : {m.avg_loss_pct:>+10.2f}%",
        f"  Profit Factor       : {m.profit_factor:>10.2f}",
        f"  Payoff Ratio        : {m.payoff_ratio:>10.2f}",
        f"  Durée moy. trade    : {m.avg_trade_duration_days:>10.0f} jours",
        f"  Sample Error        : {m.sample_error_pct:>10.1f}%",
        f"",
    ]

    # Validation Clenow
    lines.append(f"  VALIDATION (Clenow)")
    lines.append(f"  {'─' * 40}")

    # Red flags
    flags = []
    if m.cagr_pct > 30:
        flags.append(f"  🚩 CAGR {m.cagr_pct:.1f}% > 30% — suspect d'overfitting")
    if m.sharpe_ratio > 2.0:
        flags.append(f"  🚩 Sharpe {m.sharpe_ratio:.2f} > 2.0 — suspect")
    if m.max_drawdown_pct > -10:
        flags.append(f"  🚩 MaxDD {m.max_drawdown_pct:.1f}% > -10% — irréaliste")
    if m.total_trades < 30:
        flags.append(f"  🚩 {m.total_trades} trades < 30 — statistiquement insuffisant")

    if flags:
        lines.extend(flags)
    else:
        lines.append(f"  ✅ Pas de red flags détectés")

    # Règle empirique DD ≈ 3× CAGR
    expected_dd = -3 * m.cagr_pct
    lines.append(f"  DD attendu (~3×CAGR) : {expected_dd:>+.1f}% (réel : {m.max_drawdown_pct:>+.1f}%)")

    lines.append(f"{'═' * 60}")
    return "\n".join(lines)


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calcule la durée maximale du drawdown en jours calendaires."""
    in_dd = drawdown < 0
    if not in_dd.any():
        return 0

    groups = (~in_dd).cumsum()
    dd_groups = drawdown[in_dd].groupby(groups[in_dd])

    max_duration = 0
    for _, group in dd_groups:
        if len(group) > 0:
            duration = (group.index[-1] - group.index[0]).days
            max_duration = max(max_duration, duration)

    return max_duration
