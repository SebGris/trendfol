"""
Visualisation des résultats de backtest.
==========================================

Génère un rapport HTML interactif avec :
  1. Equity curve (échelle log)
  2. Drawdown chart
  3. Heatmap des rendements mensuels
  4. Répartition par secteur
  5. Distribution des trades (durée, PnL)

Usage :
    from visualize import generate_report
    generate_report(equity_curve, trades, metrics, name="Core Strategy")

Ou via CLI :
    python run_backtest.py --strategy core --fractional --plot
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from config import UNIVERSE_MAP


def _js(obj) -> str:
    """Sérialise en JSON valide pour injection dans JavaScript.
    
    Convertit les types numpy (float64, int64, etc.) en types Python natifs
    pour éviter des sorties comme 'np.float64(-20699.0)' dans le HTML.
    """
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    return json.dumps(obj, cls=_NumpyEncoder)


def generate_report(equity_curve: pd.Series,
                    trades: list[dict],
                    metrics,
                    name: str = "Backtest",
                    output_dir: str = "reports") -> str:
    """
    Génère un rapport HTML complet.

    Returns:
        Chemin du fichier HTML généré
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    safe_name = name.replace(" ", "_").replace("/", "_")[:40]
    filepath = out / f"report_{safe_name}.html"

    # Préparer les données
    eq_data = _equity_data(equity_curve)
    dd_data = _drawdown_data(equity_curve)
    heatmap_data = _monthly_heatmap_data(equity_curve)
    sector_data = _sector_breakdown(trades)
    trade_dist = _trade_distribution(trades)
    yearly_data = _yearly_returns(equity_curve)

    # Choisir l'échelle : log si toutes les valeurs > 0, sinon linéaire
    eq_min = equity_curve.min() if len(equity_curve) > 0 else 0
    eq_scale = "logarithmic" if eq_min > 0 else "linear"

    html = _build_html(
        name=name,
        metrics=metrics,
        eq_data=eq_data,
        dd_data=dd_data,
        heatmap_data=heatmap_data,
        sector_data=sector_data,
        trade_dist=trade_dist,
        yearly_data=yearly_data,
        eq_scale=eq_scale,
    )

    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


def generate_comparison_report(results: dict,
                                output_dir: str = "reports") -> str:
    """
    Génère un rapport de comparaison multi-stratégies.

    Args:
        results: {strategy_key: {metrics, equity_curve, trades, strategy_info}}
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    filepath = out / "comparison_report.html"

    html = _build_comparison_html(results)
    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


# ============================================================
# PRÉPARATION DES DONNÉES
# ============================================================

def _equity_data(equity: pd.Series) -> list[dict]:
    # Sous-échantillonner si trop de points (>2000)
    if len(equity) > 2000:
        step = len(equity) // 2000
        equity = equity.iloc[::step]
    return [
        {"x": d.strftime("%Y-%m-%d"), "y": round(v, 2)}
        for d, v in equity.items()
    ]


def _drawdown_data(equity: pd.Series) -> list[dict]:
    cummax = equity.cummax()
    dd = ((equity - cummax) / cummax) * 100
    if len(dd) > 2000:
        step = len(dd) // 2000
        dd = dd.iloc[::step]
    return [
        {"x": d.strftime("%Y-%m-%d"), "y": round(v, 2)}
        for d, v in dd.items()
    ]


def _monthly_heatmap_data(equity: pd.Series) -> dict:
    """Rendements mensuels en % par année/mois."""
    daily_ret = equity.pct_change().dropna()
    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    years = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    grid = []
    for y in years:
        row = {"year": int(y)}
        year_data = monthly[monthly.index.year == y]
        year_total = 0
        for m in months:
            val = year_data[year_data.index.month == m]
            if len(val) > 0:
                v = round(float(val.iloc[0]), 1)
                row[f"m{m}"] = v
                year_total += v
            else:
                row[f"m{m}"] = None
        row["total"] = round(year_total, 1)
        grid.append(row)

    return {"month_names": month_names, "grid": grid}


def _sector_breakdown(trades: list[dict]) -> list[dict]:
    """PnL par secteur."""
    if not trades:
        return []

    sector_pnl = {}
    sector_trades = {}
    sector_wins = {}

    for t in trades:
        inst = UNIVERSE_MAP.get(t["instrument"])
        sector = inst.sector if inst else "unknown"
        sector_pnl[sector] = sector_pnl.get(sector, 0) + t["pnl"]
        sector_trades[sector] = sector_trades.get(sector, 0) + 1
        if t["pnl"] > 0:
            sector_wins[sector] = sector_wins.get(sector, 0) + 1

    result = []
    for sector in sorted(sector_pnl.keys()):
        n = sector_trades[sector]
        wins = sector_wins.get(sector, 0)
        result.append({
            "sector": sector,
            "pnl": round(sector_pnl[sector], 0),
            "trades": n,
            "win_rate": round(wins / n * 100, 1) if n > 0 else 0,
        })
    return result


def _trade_distribution(trades: list[dict]) -> dict:
    """Distribution des PnL et durées."""
    if not trades:
        return {"pnl_bins": [], "duration_bins": []}

    pnls = [t["pnl"] for t in trades]
    durations = [t["holding_days"] for t in trades]

    # PnL histogram (bins de $500)
    pnl_min = min(pnls)
    pnl_max = max(pnls)
    bin_size = max(500, (pnl_max - pnl_min) / 30)
    pnl_bins = []
    current = pnl_min
    while current < pnl_max:
        count = sum(1 for p in pnls if current <= p < current + bin_size)
        if count > 0:
            pnl_bins.append({
                "label": f"${current:,.0f}",
                "count": count,
                "color": "#22c55e" if current >= 0 else "#ef4444"
            })
        current += bin_size

    # Duration histogram
    dur_bins = []
    edges = [0, 7, 30, 90, 180, 365, 730, 9999]
    labels = ["<1w", "1w-1m", "1-3m", "3-6m", "6m-1y", "1-2y", ">2y"]
    for i in range(len(edges) - 1):
        count = sum(1 for d in durations if edges[i] <= d < edges[i + 1])
        dur_bins.append({"label": labels[i], "count": count})

    return {"pnl_bins": pnl_bins, "duration_bins": dur_bins}


def _yearly_returns(equity: pd.Series) -> list[dict]:
    """Rendement annuel."""
    daily_ret = equity.pct_change().dropna()
    yearly = daily_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    return [
        {"year": int(d.year), "return_pct": round(float(v), 2)}
        for d, v in yearly.items()
    ]


# ============================================================
# CONSTRUCTION HTML
# ============================================================

def _build_html(name, metrics, eq_data, dd_data, heatmap_data,
                sector_data, trade_dist, yearly_data, eq_scale="logarithmic") -> str:
    m = metrics
    eq_title = "Equity Curve (log)" if eq_scale == "logarithmic" else "Equity Curve"
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport — {name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 8px; color: #f8fafc; }}
  h2 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #94a3b8; font-weight: 500; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; }}

  /* Metrics cards */
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .card {{
    background: #1e293b; border-radius: 8px; padding: 16px;
    border: 1px solid #334155;
  }}
  .card-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
  .positive {{ color: #22c55e; }}
  .negative {{ color: #ef4444; }}
  .neutral {{ color: #e2e8f0; }}

  /* Charts */
  .chart-container {{
    background: #1e293b; border-radius: 8px; padding: 20px;
    border: 1px solid #334155; margin-bottom: 16px;
  }}
  .chart-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  @media (max-width: 900px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
  }}

  /* Heatmap */
  .heatmap {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .heatmap th {{ padding: 6px 8px; text-align: center; color: #94a3b8;
                 font-weight: 500; }}
  .heatmap td {{ padding: 6px 8px; text-align: center; border-radius: 4px; }}
  .heatmap tr:hover {{ background: #1e293b; }}

  /* Sector table */
  .sector-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .sector-table th {{ text-align: left; padding: 8px; color: #94a3b8;
                      border-bottom: 1px solid #334155; }}
  .sector-table td {{ padding: 8px; border-bottom: 1px solid #1e293b; }}
</style>
</head>
<body>
<div class="container">

<h1>📊 {name}</h1>
<p class="subtitle">
  {m.equity_curve.index[0].strftime('%Y-%m-%d') if m.equity_curve is not None else ''} →
  {m.equity_curve.index[-1].strftime('%Y-%m-%d') if m.equity_curve is not None else ''}
  &nbsp;|&nbsp; {m.total_trades} trades
</p>

<!-- METRICS CARDS -->
<div class="metrics-grid">
  <div class="card">
    <div class="card-label">CAGR</div>
    <div class="card-value {'positive' if m.cagr_pct > 0 else 'negative'}">{m.cagr_pct:+.2f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Sharpe Ratio</div>
    <div class="card-value {'positive' if m.sharpe_ratio > 0.5 else 'neutral'}">{m.sharpe_ratio:.3f}</div>
  </div>
  <div class="card">
    <div class="card-label">Sortino</div>
    <div class="card-value neutral">{m.sortino_ratio:.3f}</div>
  </div>
  <div class="card">
    <div class="card-label">Calmar</div>
    <div class="card-value neutral">{m.calmar_ratio:.3f}</div>
  </div>
  <div class="card">
    <div class="card-label">Max Drawdown</div>
    <div class="card-value negative">{m.max_drawdown_pct:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Volatilité ann.</div>
    <div class="card-value neutral">{m.annualized_vol_pct:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate</div>
    <div class="card-value neutral">{m.win_rate_pct:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Profit Factor</div>
    <div class="card-value {'positive' if m.profit_factor > 1 else 'negative'}">{m.profit_factor:.2f}</div>
  </div>
</div>

<!-- EQUITY CURVE -->
<div class="chart-container">
  <canvas id="equityChart" height="100"></canvas>
</div>

<!-- DRAWDOWN -->
<div class="chart-container">
  <canvas id="drawdownChart" height="60"></canvas>
</div>

<!-- YEARLY + SECTOR ROW -->
<div class="chart-row">
  <div class="chart-container">
    <canvas id="yearlyChart" height="140"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="sectorChart" height="140"></canvas>
  </div>
</div>

<!-- TRADE DISTRIBUTION -->
<div class="chart-row">
  <div class="chart-container">
    <canvas id="durationChart" height="140"></canvas>
  </div>
  <div class="chart-container">
    <h2 style="margin-top:0">Répartition par secteur</h2>
    <table class="sector-table">
      <tr><th>Secteur</th><th>PnL</th><th>Trades</th><th>Win Rate</th></tr>
      {''.join(f'''<tr>
        <td>{s['sector']}</td>
        <td class="{'positive' if s['pnl'] > 0 else 'negative'}">${s['pnl']:+,.0f}</td>
        <td>{s['trades']}</td>
        <td>{s['win_rate']:.1f}%</td>
      </tr>''' for s in sector_data)}
    </table>
  </div>
</div>

<!-- MONTHLY HEATMAP -->
<h2>Rendements mensuels (%)</h2>
<div class="chart-container" style="overflow-x: auto;">
  <table class="heatmap">
    <tr>
      <th>Année</th>
      {''.join(f'<th>{mn}</th>' for mn in heatmap_data['month_names'])}
      <th><b>Total</b></th>
    </tr>
    {''.join(_heatmap_row(row) for row in heatmap_data['grid'])}
  </table>
</div>

</div><!-- /container -->

<script>
const eqData = {_js(eq_data)};
const ddData = {_js(dd_data)};
const yearlyData = {_js(yearly_data)};
const sectorData = {_js(sector_data)};
const durationBins = {_js(trade_dist['duration_bins'])};

// Equity curve (log scale)
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    datasets: [{{
      label: 'Equity ($)',
      data: eqData,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      fill: true,
      pointRadius: 0,
      borderWidth: 1.5,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: '{eq_title}', color: '#94a3b8' }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'year' }},
            ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ type: '{eq_scale}',
            ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
            grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Drawdown
new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{
    datasets: [{{
      label: 'Drawdown (%)',
      data: ddData,
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239,68,68,0.15)',
      fill: true,
      pointRadius: 0,
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Drawdown', color: '#94a3b8' }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'year' }},
            ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#94a3b8', callback: v => v.toFixed(0) + '%' }},
            grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Yearly returns bar chart
new Chart(document.getElementById('yearlyChart'), {{
  type: 'bar',
  data: {{
    labels: yearlyData.map(d => d.year),
    datasets: [{{
      label: 'Rendement annuel (%)',
      data: yearlyData.map(d => d.return_pct),
      backgroundColor: yearlyData.map(d => d.return_pct >= 0 ? '#22c55e' : '#ef4444'),
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Rendements annuels', color: '#94a3b8' }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }},
            grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Sector PnL
new Chart(document.getElementById('sectorChart'), {{
  type: 'bar',
  data: {{
    labels: sectorData.map(d => d.sector),
    datasets: [{{
      label: 'PnL ($)',
      data: sectorData.map(d => d.pnl),
      backgroundColor: sectorData.map(d => d.pnl >= 0 ? '#22c55e' : '#ef4444'),
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'PnL par secteur', color: '#94a3b8' }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', callback: v => '$' + (v/1000).toFixed(0) + 'k' }},
            grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }}
    }}
  }}
}});

// Trade duration
new Chart(document.getElementById('durationChart'), {{
  type: 'bar',
  data: {{
    labels: durationBins.map(d => d.label),
    datasets: [{{
      label: 'Nombre de trades',
      data: durationBins.map(d => d.count),
      backgroundColor: '#6366f1',
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Durée des trades', color: '#94a3b8' }},
      legend: {{ display: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def _heatmap_row(row: dict) -> str:
    """Génère une ligne du tableau heatmap avec couleurs."""
    cells = f"<td><b>{row['year']}</b></td>"
    for m in range(1, 13):
        val = row.get(f"m{m}")
        if val is None:
            cells += "<td>—</td>"
        else:
            color = _heatmap_color(val)
            cells += f'<td style="background:{color};color:#fff;font-weight:500">{val:+.1f}</td>'
    # Total
    total = row.get("total", 0)
    tcolor = _heatmap_color(total)
    cells += f'<td style="background:{tcolor};color:#fff;font-weight:700">{total:+.1f}</td>'
    return f"<tr>{cells}</tr>"


def _heatmap_color(val: float) -> str:
    """Couleur pour la heatmap : rouge négatif, vert positif."""
    if val is None:
        return "transparent"
    # Clamp between -20 and +20 for color mapping
    clamped = max(-20, min(20, val))
    if clamped >= 0:
        intensity = min(clamped / 20, 1)
        r = int(30 * (1 - intensity) + 34 * intensity)
        g = int(41 * (1 - intensity) + 197 * intensity)
        b = int(59 * (1 - intensity) + 94 * intensity)
    else:
        intensity = min(abs(clamped) / 20, 1)
        r = int(30 * (1 - intensity) + 239 * intensity)
        g = int(41 * (1 - intensity) + 68 * intensity)
        b = int(59 * (1 - intensity) + 68 * intensity)
    return f"rgb({r},{g},{b})"


def _build_comparison_html(results: dict) -> str:
    """Rapport de comparaison multi-stratégies."""
    # Construire les datasets pour l'equity overlay
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']
    datasets_js = []
    all_min = float("inf")
    for i, (key, res) in enumerate(results.items()):
        if key == "ma_crossover":
            continue  # Skip MA crossover (cassé)
        eq = res["equity_curve"]
        all_min = min(all_min, eq.min())
        data = _equity_data(eq)
        color = colors[i % len(colors)]
        name = res["strategy_info"]["name"][:30]
        datasets_js.append(f"""{{
            label: '{name}',
            data: {_js(data)},
            borderColor: '{color}',
            pointRadius: 0,
            borderWidth: 2,
            fill: false,
        }}""")

    # Metrics comparison table
    rows_html = ""
    metric_labels = [
        ("CAGR", lambda m: f"{m.cagr_pct:+.2f}%"),
        ("Sharpe", lambda m: f"{m.sharpe_ratio:.3f}"),
        ("Sortino", lambda m: f"{m.sortino_ratio:.3f}"),
        ("Max DD", lambda m: f"{m.max_drawdown_pct:.1f}%"),
        ("Vol ann.", lambda m: f"{m.annualized_vol_pct:.1f}%"),
        ("Calmar", lambda m: f"{m.calmar_ratio:.3f}"),
        ("Trades", lambda m: f"{m.total_trades}"),
        ("Win Rate", lambda m: f"{m.win_rate_pct:.1f}%"),
        ("Profit Factor", lambda m: f"{m.profit_factor:.2f}"),
    ]

    filtered = {k: v for k, v in results.items() if k != "ma_crossover"}
    headers = "".join(f"<th>{v['strategy_info']['name'][:20]}</th>" for v in filtered.values())

    for label, fmt in metric_labels:
        cells = ""
        vals = [fmt(v["metrics"]) for v in filtered.values()]
        for val in vals:
            cells += f"<td>{val}</td>"
        rows_html += f"<tr><td><b>{label}</b></td>{cells}</tr>"

    comp_scale = "logarithmic" if all_min > 0 else "linear"
    scale_label = "échelle log" if comp_scale == "logarithmic" else "échelle linéaire"

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Comparaison des stratégies</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 16px; color: #f8fafc; }}
  h2 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #94a3b8; }}
  .chart-container {{
    background: #1e293b; border-radius: 8px; padding: 20px;
    border: 1px solid #334155; margin-bottom: 16px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ text-align: left; padding: 10px; color: #94a3b8;
       border-bottom: 2px solid #334155; }}
  td {{ padding: 10px; border-bottom: 1px solid #1e293b; }}
  tr:hover {{ background: #1e293b; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 Comparaison des stratégies</h1>

<div class="chart-container">
  <canvas id="compChart" height="120"></canvas>
</div>

<h2>Métriques</h2>
<div class="chart-container">
  <table>
    <tr><th>Métrique</th>{headers}</tr>
    {rows_html}
  </table>
</div>

</div>
<script>
new Chart(document.getElementById('compChart'), {{
  type: 'line',
  data: {{ datasets: [{','.join(datasets_js)}] }},
  options: {{
    responsive: true,
    interaction: {{ intersect: false, mode: 'index' }},
    plugins: {{
      title: {{ display: true, text: 'Equity Curves ({scale_label})', color: '#94a3b8' }},
      legend: {{ labels: {{ color: '#94a3b8' }} }}
    }},
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'year' }},
            ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ type: '{comp_scale}',
            ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
            grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
