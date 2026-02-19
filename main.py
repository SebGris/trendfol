"""
Pipeline principal — Phase 1 : Infrastructure de données
=========================================================

Usage :
    python main.py              # Pipeline complet (download + quality check)
    python main.py --download   # Téléchargement seul
    python main.py --check      # Contrôle qualité seul
    python main.py --summary    # Résumé des données en base
    python main.py --quality    # Rapport des anomalies détectées
"""

import sys
from database import init_db, get_data_summary, get_quality_report
from downloader import download_all
from cleaner import run_all_quality_checks


def print_summary():
    """Affiche un résumé des données stockées en base."""
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES DONNÉES EN BASE")
    print("=" * 60)

    summary = get_data_summary()
    if summary.empty:
        print("  (aucune donnée)")
        return

    for _, row in summary.iterrows():
        print(f"\n  {row['name']:20s} ({row['ticker']})")
        print(f"    Secteur    : {row['sector']}")
        print(f"    Lignes     : {row['nb_rows']}")
        print(f"    Période    : {row['first_date']} → {row['last_date']}")
        print(f"    Mis à jour : {row['last_updated'] or 'jamais'}")


def print_quality_report():
    """Affiche les anomalies détectées."""
    print("\n" + "=" * 60)
    print("🔍 RAPPORT DE QUALITÉ")
    print("=" * 60)

    report = get_quality_report()
    if report.empty:
        print("  ✅ Aucune anomalie enregistrée")
        return

    # Résumé par type et sévérité
    summary = report.groupby(["name", "severity", "check_type"]).size()
    print(f"\n  {len(report)} anomalies au total :\n")

    for (name, severity, check_type), count in summary.items():
        icon = "❌" if severity == "ERROR" else "⚠️"
        print(f"  {icon} {name:20s} | {check_type:25s} | {count} occurrences")


def run_pipeline():
    """Exécute le pipeline complet."""
    print("\n" + "🚀" * 20)
    print("  PIPELINE PHASE 1 — INFRASTRUCTURE DE DONNÉES")
    print("🚀" * 20 + "\n")

    # Étape 1 : Initialiser la base
    init_db()

    # Étape 2 : Télécharger les données
    results = download_all()

    # Vérifier qu'on a bien des données
    if sum(results.values()) == 0:
        print("\n❌ Aucune donnée téléchargée. Vérifier la connexion réseau.")
        return

    # Étape 3 : Contrôle qualité
    run_all_quality_checks()

    # Étape 4 : Résumé final
    print_summary()
    print_quality_report()

    print("\n" + "=" * 60)
    print("✅ PHASE 1 TERMINÉE")
    print("   Prochaine étape : Phase 1b — Calcul des indicateurs (ATR, EMA)")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        run_pipeline()
    elif "--download" in args:
        init_db()
        download_all()
    elif "--check" in args:
        run_all_quality_checks()
    elif "--summary" in args:
        print_summary()
    elif "--quality" in args:
        print_quality_report()
    else:
        print(__doc__)
