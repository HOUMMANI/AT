#!/usr/bin/env python3
"""
CLI d'analyse technique pour les actions de la Bourse de Casablanca (BVC).

Utilisation:
    python scripts/analyze.py ATW
    python scripts/analyze.py IAM --period 6mo --chart
    python scripts/analyze.py BCP --period 1y --chart --save
    python scripts/analyze.py --liste-secteurs
    python scripts/analyze.py --secteur Banques
    python scripts/analyze.py --marche
"""

import sys
import os
import argparse
import logging

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import BVCDataFetcher
from src.data.tickers import BVC_TICKERS, list_sectors, get_ticker_info
from src.analysis import TechnicalAnalyzer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def cmd_analyze(args):
    """Analyse un symbole et affiche le rapport."""
    fetcher = BVCDataFetcher()

    print(f"\nRécupération des données pour {args.symbole} ({args.period})...")
    try:
        df = fetcher.get_ohlcv(args.symbole, period=args.period)
    except Exception as e:
        print(f"Erreur: impossible de récupérer les données pour '{args.symbole}': {e}")
        print("Vérifiez que le symbole est correct. Utilisez --liste pour voir tous les symboles.")
        sys.exit(1)

    if df.empty:
        print(f"Aucune donnée disponible pour {args.symbole}.")
        sys.exit(1)

    analyzer = TechnicalAnalyzer(df)
    report = analyzer.full_report()
    print(report)

    if args.chart or args.save:
        try:
            from src.visualization.charts import plot_chart
            import matplotlib.pyplot as plt

            save_path = None
            if args.save:
                save_path = f"{args.symbole}_{args.period}.png"

            fig = plot_chart(
                analyzer.compute_all(),
                show_volume=True,
                show_bollinger=True,
                show_macd=True,
                show_rsi=True,
                save_path=save_path,
            )

            if args.chart and not args.save:
                plt.show()
            elif args.save:
                print(f"Graphique sauvegardé: {save_path}")
            plt.close(fig)

        except ImportError as e:
            print(f"Impossible d'afficher le graphique: {e}")
            print("Installez matplotlib: pip install matplotlib")


def cmd_marche(args):
    """Affiche un aperçu du marché."""
    fetcher = BVCDataFetcher()

    print(f"\nAperçu du marché BVC (dernière période: {args.period})...")
    overview = fetcher.get_market_overview(period=args.period)

    if overview.empty:
        print("Impossible de récupérer les données du marché.")
        return

    print("\n" + "=" * 70)
    print("  APERCU DU MARCHE - BOURSE DE CASABLANCA (BVC)")
    print("=" * 70)
    print(f"{'Symbole':<10} {'Nom':<35} {'Cours':>10} {'Variation':>10}")
    print("-" * 70)

    for _, row in overview.iterrows():
        var = row["Variation (%)"]
        arrow = "▲" if var >= 0 else "▼"
        print(f"{row['Symbole']:<10} {row['Nom'][:34]:<35} {row['Dernier cours']:>8.2f}   {arrow}{abs(var):>6.2f}%")

    print("=" * 70)


def cmd_secteur(args):
    """Analyse toutes les actions d'un secteur."""
    fetcher = BVCDataFetcher()
    from src.data.tickers import get_tickers_by_sector

    secteur = args.secteur
    tickers = get_tickers_by_sector(secteur)

    if not tickers:
        print(f"Secteur '{secteur}' introuvable.")
        print(f"Secteurs disponibles: {', '.join(list_sectors())}")
        return

    print(f"\nAnalyse du secteur: {secteur} ({len(tickers)} actions)")
    print("=" * 60)

    for symbol, info in tickers.items():
        try:
            df = fetcher.get_ohlcv(symbol, period=args.period)
            if df.empty:
                continue
            analyzer = TechnicalAnalyzer(df)
            score_data = analyzer.score()
            summ = analyzer.summary()

            score = score_data["score"]
            reco = score_data["recommandation"]
            var = summ["variation_1j"]
            print(f"  {symbol:<8} {info['name'][:30]:<30} "
                  f"Score: {score:+5.1f}  {reco:<20}  "
                  f"1j: {var:+.2f}%")
        except Exception as e:
            print(f"  {symbol:<8} Erreur: {e}")

    print("=" * 60)


def cmd_liste(args):
    """Affiche la liste des actions disponibles."""
    secteur_filter = args.secteur if hasattr(args, "secteur") and args.secteur else None

    print("\nActions disponibles sur la Bourse de Casablanca (BVC)")
    print("=" * 65)
    print(f"{'Symbole':<10} {'Yahoo':<12} {'Secteur':<25} {'Nom'}")
    print("-" * 65)

    for symbol, info in sorted(BVC_TICKERS.items()):
        if secteur_filter and info["secteur"].lower() != secteur_filter.lower():
            continue
        print(f"{symbol:<10} {info['yahoo']:<12} {info['secteur']:<25} {info['name']}")

    print(f"\nTotal: {len(BVC_TICKERS)} instruments")
    print(f"Secteurs: {', '.join(list_sectors())}")


def cmd_compare(args):
    """Compare plusieurs actions."""
    fetcher = BVCDataFetcher()
    symbols = [s.strip().upper() for s in args.symboles.split(",")]

    print(f"\nComparaison: {', '.join(symbols)} ({args.period})")
    print("=" * 80)
    print(f"{'Symbole':<10} {'Nom':<30} {'Cours':>8} {'1j':>8} {'1m':>8} {'Score':>8} {'Signal'}")
    print("-" * 80)

    for symbol in symbols:
        try:
            df = fetcher.get_ohlcv(symbol, period=args.period)
            if df.empty:
                print(f"{symbol:<10} Pas de données")
                continue

            analyzer = TechnicalAnalyzer(df)
            summ = analyzer.summary()
            score_data = analyzer.score()

            var_1j = f"{summ['variation_1j']:+.2f}%" if summ["variation_1j"] else "N/A"
            var_1m = f"{summ['variation_1m']:+.2f}%" if summ["variation_1m"] else "N/A"

            print(f"{symbol:<10} {summ['nom'][:29]:<30} "
                  f"{summ['dernier_cours']:>8.2f} "
                  f"{var_1j:>8} {var_1m:>8} "
                  f"{score_data['score']:>+7.1f}  "
                  f"{score_data['recommandation']}")
        except Exception as e:
            print(f"{symbol:<10} Erreur: {e}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse technique des actions de la Bourse de Casablanca (BVC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/analyze.py ATW
  python scripts/analyze.py IAM --period 6mo --chart
  python scripts/analyze.py BCP --period 1y --chart --save
  python scripts/analyze.py --marche
  python scripts/analyze.py --secteur Banques
  python scripts/analyze.py --comparer ATW,BCP,BOA
  python scripts/analyze.py --liste
        """
    )

    parser.add_argument("symbole", nargs="?", help="Symbole BVC (ex: ATW, IAM, BCP)")
    parser.add_argument("--period", "-p", default="1y",
                        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                        help="Période d'analyse (défaut: 1y)")
    parser.add_argument("--chart", "-c", action="store_true",
                        help="Afficher le graphique interactif")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Sauvegarder le graphique en PNG")
    parser.add_argument("--marche", "-m", action="store_true",
                        help="Aperçu général du marché BVC")
    parser.add_argument("--secteur", help="Analyser toutes les actions d'un secteur")
    parser.add_argument("--liste", "-l", action="store_true",
                        help="Lister toutes les actions disponibles")
    parser.add_argument("--comparer", metavar="S1,S2,...",
                        help="Comparer plusieurs symboles (séparés par virgules)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mode verbeux")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.liste:
        cmd_liste(args)
    elif args.marche:
        cmd_marche(args)
    elif args.secteur and not args.symbole:
        cmd_secteur(args)
    elif args.comparer:
        args.symboles = args.comparer
        cmd_compare(args)
    elif args.symbole:
        cmd_analyze(args)
    else:
        parser.print_help()
        print("\nExemple rapide: python scripts/analyze.py ATW --chart")


if __name__ == "__main__":
    main()
