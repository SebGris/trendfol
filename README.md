# Trend Following — Phase 1 : Infrastructure de données

## 🏗️ Structure du projet

```
trend_following/
├── config.py        # Univers (5 instruments Carver), paramètres globaux
├── database.py      # SQLite : schéma, CRUD, rapports
├── downloader.py    # Téléchargement yfinance → SQLite
├── cleaner.py       # 6 contrôles qualité (Clenow)
├── main.py          # Pipeline orchestrateur
└── data/
    └── market_data.db   # Base SQLite (créée automatiquement)
```

## ⚡ Démarrage rapide

### 1. Installer les dépendances

```bash
pip install yfinance pandas
```

### 2. Lancer le pipeline complet

```bash
cd trend_following
python main.py
```

Cela exécute dans l'ordre :
1. **Initialisation** de la base SQLite
2. **Téléchargement** des 5 instruments depuis 2005 (yfinance)
3. **Contrôle qualité** : 6 vérifications automatiques
4. **Rapport** résumé dans le terminal

### 3. Commandes individuelles

```bash
python main.py --download   # Téléchargement seul
python main.py --check      # Contrôle qualité seul
python main.py --summary    # Résumé des données en base
python main.py --quality    # Rapport des anomalies
```

## 📋 Univers d'instruments (Carver Starter System)

| Instrument   | Ticker yfinance | Secteur         | Point Value |
|-------------|-----------------|-----------------|-------------|
| S&P 500     | ES=F            | Equities        | 50          |
| Gold        | GC=F            | Non-agricultural| 100         |
| Corn        | ZC=F            | Agricultural    | 50          |
| Euro Stoxx  | ^STOXX50E       | Equities        | 10          |
| AUDUSD      | AUDUSD=X        | Currencies      | 100,000     |

## 🔍 Contrôles qualité implémentés

| # | Check                    | Sévérité | Référence          |
|---|--------------------------|----------|--------------------|
| 1 | Valeurs manquantes (NaN) | WARNING  | Clenow, piège #5   |
| 2 | Cohérence OHLC           | ERROR    | High≥Low, etc.     |
| 3 | Outliers (>15% daily)    | WARNING  | Clenow, piège #5   |
| 4 | Gaps de dates (>5j cal.) | WARNING  | Continuité données  |
| 5 | Prix nuls ou négatifs    | ERROR    | Validité basique    |
| 6 | Historique suffisant     | WARNING  | Clenow: min 10 ans |

## 🗄️ Schéma SQLite

**instruments** : métadonnées (nom, ticker, secteur, point_value...)
**daily_prices** : OHLCV journalier (clé: instrument_id + date)
**quality_log** : journal d'anomalies horodaté

## 🔜 Prochaines étapes

- **Phase 1b** : Indicateurs (ATR, EMA 50/100, volatilité annualisée)
- **Phase 2** : Backtest MA Crossover (stratégie A — Clenow)
