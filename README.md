# AT - Analyse Technique BVC

Outil d'analyse technique pour les actions cotées à la **Bourse des Valeurs de Casablanca (BVC)**.

## Fonctionnalités

- **Données en temps réel** via Yahoo Finance (actions marocaines avec suffixe `.CS`)
- **40+ actions** de la BVC répertoriées (banques, telecom, immobilier, mines, etc.)
- **Indicateurs techniques** complets :
  - Tendance : SMA, EMA, WMA, MACD, SuperTrend, Ichimoku, Pivots
  - Momentum : RSI, Stochastique, CCI, Williams %R, ROC, TSI
  - Volatilité : Bandes de Bollinger, ATR, Keltner, Donchian, Squeeze
  - Volume : OBV, VWAP, MFI, CMF, A/D, Volume Relatif
- **Signaux automatiques** d'achat/vente avec score global (-100 à +100)
- **Supports & Résistances** détectés automatiquement
- **Graphiques** avec chandeliers japonais et thème sombre
- **CLI** complète pour analyses rapides en ligne de commande

## Installation

```bash
git clone https://github.com/HOUMMANI/AT.git
cd AT
pip install -r requirements.txt
```

## Utilisation rapide

### CLI

```bash
# Analyser une action (rapport complet)
python scripts/analyze.py ATW

# Avec graphique interactif
python scripts/analyze.py IAM --period 6mo --chart

# Sauvegarder le graphique en PNG
python scripts/analyze.py BCP --period 1y --chart --save

# Aperçu du marché
python scripts/analyze.py --marche

# Analyser un secteur entier
python scripts/analyze.py --secteur Banques

# Comparer plusieurs actions
python scripts/analyze.py --comparer ATW,BCP,BOA,CIH

# Lister toutes les actions disponibles
python scripts/analyze.py --liste
```

### Python

```python
from src.data import BVCDataFetcher
from src.analysis import TechnicalAnalyzer
from src.visualization.charts import plot_chart

fetcher = BVCDataFetcher()
df = fetcher.get_ohlcv("ATW", period="1y")

analyzer = TechnicalAnalyzer(df)
print(analyzer.full_report())

# Score et signaux
score = analyzer.score()
print(f"Score: {score['score']} - {score['recommandation']}")

# Supports/Résistances
sr = analyzer.support_resistance()

# Graphique complet avec chandeliers
fig = plot_chart(analyzer.compute_all())
fig.savefig("ATW_analyse.png")

# Aperçu du marché
overview = fetcher.get_market_overview()
print(overview)
```

## Structure du projet

```
AT/
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Récupération données Yahoo Finance
│   │   └── tickers.py       # 40+ actions BVC répertoriées
│   ├── indicators/
│   │   ├── trend.py         # SMA, EMA, MACD, SuperTrend, Ichimoku, Pivots
│   │   ├── momentum.py      # RSI, Stochastique, CCI, Williams %R, ROC, TSI
│   │   ├── volatility.py    # Bollinger, ATR, Keltner, Donchian, Squeeze
│   │   └── volume.py        # OBV, VWAP, MFI, CMF, A/D, RVOL
│   ├── analysis/
│   │   └── analyzer.py      # Moteur d'analyse, signaux, score, S/R
│   └── visualization/
│       └── charts.py        # Chandeliers japonais + indicateurs
├── scripts/
│   └── analyze.py           # CLI principale
├── notebooks/               # Exemples Jupyter
└── requirements.txt
```

## Indicateurs implémentés

| Catégorie | Indicateurs |
|-----------|------------|
| **Tendance** | SMA (20/50/100/200), EMA (9/20/50), MACD, SuperTrend, Ichimoku, Pivots |
| **Momentum** | RSI, Stochastique, CCI, Williams %R, ROC, TSI |
| **Volatilité** | Bollinger Bands, ATR, Keltner, Donchian, HV, Squeeze |
| **Volume** | OBV, VWAP, MFI, CMF, A/D, RVOL |

## Symboles disponibles (exemples)

| Symbole | Nom | Secteur |
|---------|-----|---------|
| ATW | Attijariwafa Bank | Banques |
| BCP | Banque Centrale Populaire | Banques |
| IAM | Maroc Telecom | Télécommunications |
| MNG | Managem | Mines |
| LHM | LafargeHolcim Maroc | Matériaux |
| COSU | Cosumar | Agroalimentaire |
| TQM | Taqa Morocco | Énergie |
| MASI | MASI Index | Indice |

> `python scripts/analyze.py --liste` pour la liste complète.

## Licence

MIT
