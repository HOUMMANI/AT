"""
Analyse multi-timeframes (MTF) pour les actions BVC.

Principe : analyser le même actif sur plusieurs unités de temps
simultanément (journalier, hebdomadaire, mensuel) et détecter la
confluence — quand plusieurs timeframes confirment le même signal,
la fiabilité est nettement supérieure.

Règle de priorité des timeframes :
  Mensuel > Hebdomadaire > Journalier > H4 > H1

Exemple:
    mtf = MultiTimeframeAnalyzer("ATW")
    report = mtf.full_report()
    confluence = mtf.get_confluence()
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from ..data.fetcher import BVCDataFetcher
from ..analysis.analyzer import TechnicalAnalyzer
from ..patterns.candlesticks import CandlestickPatterns
from ..patterns.chart_patterns import ChartPatternDetector
from ..patterns.fibonacci import FibonacciAnalyzer
from ..patterns.trendlines import TrendlineDetector
from ..indicators.trend import sma, ema, macd
from ..indicators.momentum import rsi
from ..indicators.volatility import bollinger_bands, atr

logger = logging.getLogger(__name__)


# Libellés lisibles pour chaque timeframe
TIMEFRAME_LABELS = {
    "1d":  "Journalier (1J)",
    "1wk": "Hebdomadaire (1S)",
    "1mo": "Mensuel (1M)",
}

# Priorité des timeframes (plus grand = plus fort)
TIMEFRAME_WEIGHT = {
    "1mo": 3,
    "1wk": 2,
    "1d":  1,
}

# Périodes à récupérer pour chaque timeframe
TIMEFRAME_PERIOD = {
    "1d":  "2y",    # 2 ans de données journalières
    "1wk": "5y",    # 5 ans de données hebdomadaires
    "1mo": "10y",   # 10 ans de données mensuelles
}

# Paramètres des indicateurs adaptés au timeframe
TIMEFRAME_PARAMS = {
    "1d": {
        "rsi_period": 14,
        "sma_fast": 20,
        "sma_slow": 50,
        "sma_long": 200,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "atr_period": 14,
    },
    "1wk": {
        "rsi_period": 14,
        "sma_fast": 10,
        "sma_slow": 20,
        "sma_long": 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "atr_period": 14,
    },
    "1mo": {
        "rsi_period": 14,
        "sma_fast": 6,
        "sma_slow": 12,
        "sma_long": 24,
        "macd_fast": 6,
        "macd_slow": 12,
        "macd_signal": 4,
        "bb_period": 12,
        "atr_period": 10,
    },
}


@dataclass
class TimeframeAnalysis:
    """Résultat de l'analyse pour un timeframe donné."""
    timeframe: str
    label: str
    df: pd.DataFrame
    score: float
    recommandation: str
    tendance: str            # "HAUSSIERE" | "BAISSIERE" | "LATERALE"
    rsi_value: float
    rsi_signal: str
    macd_signal: str
    bb_signal: str
    ma_signal: str
    prix_actuel: float
    sma_fast: float
    sma_slow: float
    sma_long: float
    atr: float
    volatilite: str          # "FAIBLE" | "NORMALE" | "ELEVEE" | "TRES ELEVEE"
    candlestick_patterns: list = field(default_factory=list)
    chart_patterns: list = field(default_factory=list)
    supports: List[float] = field(default_factory=list)
    resistances: List[float] = field(default_factory=list)
    fib_support: Optional[float] = None
    fib_resistance: Optional[float] = None


@dataclass
class ConfluenceResult:
    """Résultat de la confluence entre timeframes."""
    direction: str           # "HAUSSIER" | "BAISSIER" | "MIXTE" | "NEUTRE"
    score_global: float      # -100 à +100
    score_pondere: float     # Score pondéré par poids des timeframes
    recommandation: str
    nb_timeframes_alignes: int
    nb_timeframes_total: int
    timeframes_haussiers: List[str]
    timeframes_baissiers: List[str]
    timeframes_neutres: List[str]
    zones_confluence: List[dict]    # Zones de prix où S/R coïncident
    description: str


class MultiTimeframeAnalyzer:
    """
    Analyse un actif BVC sur plusieurs timeframes et détecte la confluence.

    Exemple:
        mtf = MultiTimeframeAnalyzer("ATW")
        mtf.run()
        print(mtf.full_report())

        # Graphique comparatif des 3 timeframes
        fig = mtf.plot()
    """

    DEFAULT_TIMEFRAMES = ["1d", "1wk", "1mo"]

    def __init__(
        self,
        symbol: str,
        timeframes: List[str] = None,
        fetcher: BVCDataFetcher = None,
    ):
        """
        Args:
            symbol: Symbole BVC (ex: "ATW", "IAM")
            timeframes: Liste des timeframes (défaut: ["1d","1wk","1mo"])
            fetcher: Instance BVCDataFetcher (optionnel, créé automatiquement)
        """
        self.symbol = symbol.upper()
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.fetcher = fetcher or BVCDataFetcher()
        self._results: Dict[str, TimeframeAnalysis] = {}
        self._confluence: Optional[ConfluenceResult] = None
        self._ran = False

    def _fetch(self, timeframe: str) -> pd.DataFrame:
        """Récupère les données pour un timeframe donné."""
        period = TIMEFRAME_PERIOD.get(timeframe, "2y")
        return self.fetcher.get_ohlcv(self.symbol, period=period, interval=timeframe)

    def _compute_indicators(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Calcule les indicateurs avec les paramètres adaptés au timeframe."""
        p = TIMEFRAME_PARAMS.get(tf, TIMEFRAME_PARAMS["1d"])
        close = df["Close"]
        result = df.copy()

        result[f"SMA_fast"]   = sma(close, p["sma_fast"])
        result[f"SMA_slow"]   = sma(close, p["sma_slow"])
        result[f"SMA_long"]   = sma(close, p["sma_long"])
        result[f"EMA_fast"]   = ema(close, p["sma_fast"])
        result[f"RSI"]        = rsi(close, p["rsi_period"])

        macd_df               = macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        result["MACD"]        = macd_df["MACD"]
        result["MACD_Signal"] = macd_df["Signal"]
        result["MACD_Hist"]   = macd_df["Histogramme"]

        bb_df                 = bollinger_bands(close, p["bb_period"])
        result["BB_Haute"]    = bb_df["BB_Haute"]
        result["BB_Basse"]    = bb_df["BB_Basse"]
        result["BB_Milieu"]   = bb_df["BB_Milieu"]

        result["ATR"]         = atr(df, p["atr_period"])

        return result

    def _analyze_single(self, df: pd.DataFrame, tf: str) -> TimeframeAnalysis:
        """Analyse complète d'un DataFrame pour un timeframe."""
        df = self._compute_indicators(df, tf)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        prix = last["Close"]
        sma_fast_val = last.get("SMA_fast", np.nan)
        sma_slow_val = last.get("SMA_slow", np.nan)
        sma_long_val = last.get("SMA_long", np.nan)
        rsi_val = last.get("RSI", 50.0)
        macd_hist = last.get("MACD_Hist", 0.0)
        prev_hist = prev.get("MACD_Hist", 0.0)
        atr_val = last.get("ATR", 0.0)
        bb_haute = last.get("BB_Haute", np.nan)
        bb_basse = last.get("BB_Basse", np.nan)

        # --- Tendance directionnelle ---
        if not np.isnan(sma_fast_val) and not np.isnan(sma_slow_val):
            if prix > sma_fast_val > sma_slow_val:
                tendance = "HAUSSIERE"
            elif prix < sma_fast_val < sma_slow_val:
                tendance = "BAISSIERE"
            else:
                tendance = "LATERALE"
        else:
            tendance = "LATERALE"

        # --- Signal RSI ---
        if rsi_val < 30:
            rsi_signal = "SURVENTE"
        elif rsi_val > 70:
            rsi_signal = "SURACHAT"
        elif rsi_val < 45:
            rsi_signal = "FAIBLE"
        elif rsi_val > 55:
            rsi_signal = "FORT"
        else:
            rsi_signal = "NEUTRE"

        # --- Signal MACD ---
        if macd_hist > 0 and prev_hist <= 0:
            macd_signal = "CROISEMENT HAUSSIER"
        elif macd_hist < 0 and prev_hist >= 0:
            macd_signal = "CROISEMENT BAISSIER"
        elif macd_hist > 0:
            macd_signal = "HAUSSIER"
        else:
            macd_signal = "BAISSIER"

        # --- Signal Bollinger ---
        if not np.isnan(bb_haute) and not np.isnan(bb_basse):
            if prix > bb_haute:
                bb_signal = "SURACHAT"
            elif prix < bb_basse:
                bb_signal = "SURVENTE"
            elif prix > last.get("BB_Milieu", prix):
                bb_signal = "HAUT"
            else:
                bb_signal = "BAS"
        else:
            bb_signal = "NEUTRE"

        # --- Signal MA ---
        if not np.isnan(sma_fast_val) and not np.isnan(sma_slow_val):
            prev_fast = prev.get("SMA_fast", sma_fast_val)
            prev_slow = prev.get("SMA_slow", sma_slow_val)
            if sma_fast_val > sma_slow_val and prev_fast <= prev_slow:
                ma_signal = "GOLDEN CROSS"
            elif sma_fast_val < sma_slow_val and prev_fast >= prev_slow:
                ma_signal = "DEATH CROSS"
            elif sma_fast_val > sma_slow_val:
                ma_signal = "HAUSSIER"
            else:
                ma_signal = "BAISSIER"
        else:
            ma_signal = "NEUTRE"

        # --- Score ---
        score = self._compute_score(tendance, rsi_val, macd_hist, prev_hist, prix,
                                    sma_fast_val, sma_slow_val, sma_long_val,
                                    bb_haute, bb_basse)

        # --- Volatilité ---
        avg_atr_pct = (atr_val / prix * 100) if prix > 0 else 0
        if avg_atr_pct < 1:
            volatilite = "FAIBLE"
        elif avg_atr_pct < 2.5:
            volatilite = "NORMALE"
        elif avg_atr_pct < 5:
            volatilite = "ELEVEE"
        else:
            volatilite = "TRES ELEVEE"

        # --- Recommandation ---
        if score >= 60:
            recommandation = "FORT ACHAT"
        elif score >= 25:
            recommandation = "ACHAT"
        elif score >= -25:
            recommandation = "NEUTRE"
        elif score >= -60:
            recommandation = "VENTE"
        else:
            recommandation = "FORT VENTE"

        # --- Supports / Résistances ---
        supports = []
        resistances = []
        try:
            td = TrendlineDetector(df.rename(columns={
                "SMA_fast": "_", "SMA_slow": "__", "SMA_long": "___",
            }), min_touches=2)
            s = td.get_nearest_support()
            r = td.get_nearest_resistance()
            if s:
                supports.append(s.prix_actuel)
            if r:
                resistances.append(r.prix_actuel)
        except Exception:
            pass

        # --- Fibonacci ---
        fib_support = None
        fib_resistance = None
        try:
            fib = FibonacciAnalyzer(df, lookback=min(60, len(df)))
            analysis = fib.analyze()
            if analysis.prochain_support:
                fib_support = analysis.prochain_support.prix
            if analysis.prochaine_resistance:
                fib_resistance = analysis.prochaine_resistance.prix
        except Exception:
            pass

        # --- Patterns bougies ---
        candlestick_p = []
        try:
            cp = CandlestickPatterns(df)
            candlestick_p = cp.get_recent(lookback=5)
        except Exception:
            pass

        # --- Patterns graphiques ---
        chart_p = []
        try:
            if len(df) >= 30:
                detector = ChartPatternDetector(df)
                chart_p = detector.detect_all()
        except Exception:
            pass

        return TimeframeAnalysis(
            timeframe=tf,
            label=TIMEFRAME_LABELS.get(tf, tf),
            df=df,
            score=round(score, 1),
            recommandation=recommandation,
            tendance=tendance,
            rsi_value=round(rsi_val, 1),
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            bb_signal=bb_signal,
            ma_signal=ma_signal,
            prix_actuel=round(prix, 2),
            sma_fast=round(sma_fast_val, 2) if not np.isnan(sma_fast_val) else np.nan,
            sma_slow=round(sma_slow_val, 2) if not np.isnan(sma_slow_val) else np.nan,
            sma_long=round(sma_long_val, 2) if not np.isnan(sma_long_val) else np.nan,
            atr=round(atr_val, 2),
            volatilite=volatilite,
            candlestick_patterns=candlestick_p,
            chart_patterns=chart_p,
            supports=supports,
            resistances=resistances,
            fib_support=fib_support,
            fib_resistance=fib_resistance,
        )

    def _compute_score(
        self, tendance, rsi_val, macd_hist, prev_hist,
        prix, sma_fast, sma_slow, sma_long,
        bb_haute, bb_basse,
    ) -> float:
        """Score entre -100 et +100 pour un timeframe."""
        score = 0.0

        # Tendance (30 pts)
        if tendance == "HAUSSIERE":
            score += 30
        elif tendance == "BAISSIERE":
            score -= 30

        # RSI (20 pts)
        if rsi_val < 30:
            score += 20
        elif rsi_val > 70:
            score -= 20
        elif rsi_val < 50:
            score -= 5
        else:
            score += 5

        # MACD (25 pts)
        if macd_hist > 0 and prev_hist <= 0:
            score += 25
        elif macd_hist < 0 and prev_hist >= 0:
            score -= 25
        elif macd_hist > 0:
            score += 12
        else:
            score -= 12

        # Prix vs MA long (25 pts)
        if not np.isnan(sma_long) and sma_long > 0:
            if prix > sma_long * 1.02:
                score += 25
            elif prix < sma_long * 0.98:
                score -= 25
            elif prix > sma_long:
                score += 10
            else:
                score -= 10

        return max(-100, min(100, score))

    def run(self) -> Dict[str, TimeframeAnalysis]:
        """
        Exécute l'analyse sur tous les timeframes.

        Returns:
            Dict {timeframe: TimeframeAnalysis}
        """
        self._results = {}
        for tf in self.timeframes:
            print(f"  Analyse {TIMEFRAME_LABELS.get(tf, tf)}...")
            try:
                df = self._fetch(tf)
                if df.empty or len(df) < 10:
                    logger.warning(f"Données insuffisantes pour {tf}")
                    continue
                self._results[tf] = self._analyze_single(df, tf)
            except Exception as e:
                logger.warning(f"Erreur analyse {tf}: {e}")

        self._confluence = self._compute_confluence()
        self._ran = True
        return self._results

    def _compute_confluence(self) -> ConfluenceResult:
        """Calcule la confluence entre les différents timeframes."""
        if not self._results:
            return ConfluenceResult(
                direction="NEUTRE", score_global=0, score_pondere=0,
                recommandation="Données insuffisantes",
                nb_timeframes_alignes=0, nb_timeframes_total=0,
                timeframes_haussiers=[], timeframes_baissiers=[],
                timeframes_neutres=[], zones_confluence=[], description="",
            )

        haussiers = []
        baissiers = []
        neutres = []
        weighted_score = 0
        total_weight = 0

        for tf, analysis in self._results.items():
            w = TIMEFRAME_WEIGHT.get(tf, 1)
            total_weight += w
            weighted_score += analysis.score * w

            if analysis.score >= 25:
                haussiers.append(tf)
            elif analysis.score <= -25:
                baissiers.append(tf)
            else:
                neutres.append(tf)

        score_pondere = weighted_score / total_weight if total_weight > 0 else 0
        score_global = score_pondere

        # Direction
        n_h = len(haussiers)
        n_b = len(baissiers)
        n_total = len(self._results)

        if n_h == n_total:
            direction = "HAUSSIER"
            nb_alignes = n_h
        elif n_b == n_total:
            direction = "BAISSIER"
            nb_alignes = n_b
        elif n_h > n_b:
            direction = "PLUTOT HAUSSIER"
            nb_alignes = n_h
        elif n_b > n_h:
            direction = "PLUTOT BAISSIER"
            nb_alignes = n_b
        else:
            direction = "MIXTE"
            nb_alignes = 0

        # Recommandation pondérée
        if score_pondere >= 60:
            recommandation = "FORT SIGNAL ACHAT (tous TF alignés)"
        elif score_pondere >= 35:
            recommandation = "SIGNAL ACHAT"
        elif score_pondere >= 15:
            recommandation = "LÉGÈREMENT HAUSSIER"
        elif score_pondere >= -15:
            recommandation = "NEUTRE / ATTENTE"
        elif score_pondere >= -35:
            recommandation = "LÉGÈREMENT BAISSIER"
        elif score_pondere >= -60:
            recommandation = "SIGNAL VENTE"
        else:
            recommandation = "FORT SIGNAL VENTE (tous TF alignés)"

        # Zones de confluence S/R (niveaux présents sur ≥2 timeframes)
        zones = self._find_confluence_zones()

        # Description
        parts = []
        if haussiers:
            parts.append(f"Haussier sur {', '.join(TIMEFRAME_LABELS.get(t, t) for t in haussiers)}")
        if baissiers:
            parts.append(f"Baissier sur {', '.join(TIMEFRAME_LABELS.get(t, t) for t in baissiers)}")
        if neutres:
            parts.append(f"Neutre sur {', '.join(TIMEFRAME_LABELS.get(t, t) for t in neutres)}")
        description = ". ".join(parts) + "."

        return ConfluenceResult(
            direction=direction,
            score_global=round(score_global, 1),
            score_pondere=round(score_pondere, 1),
            recommandation=recommandation,
            nb_timeframes_alignes=nb_alignes,
            nb_timeframes_total=n_total,
            timeframes_haussiers=haussiers,
            timeframes_baissiers=baissiers,
            timeframes_neutres=neutres,
            zones_confluence=zones,
            description=description,
        )

    def _find_confluence_zones(self, tolerance_pct: float = 0.025) -> List[dict]:
        """
        Identifie les zones de prix où supports/résistances coïncident
        sur plusieurs timeframes.
        """
        all_levels = []
        for tf, analysis in self._results.items():
            weight = TIMEFRAME_WEIGHT.get(tf, 1)
            for s in analysis.supports:
                all_levels.append({"prix": s, "type": "support", "tf": tf, "weight": weight})
            for r in analysis.resistances:
                all_levels.append({"prix": r, "type": "resistance", "tf": tf, "weight": weight})
            if analysis.fib_support:
                all_levels.append({"prix": analysis.fib_support, "type": "fib_support", "tf": tf, "weight": weight})
            if analysis.fib_resistance:
                all_levels.append({"prix": analysis.fib_resistance, "type": "fib_resistance", "tf": tf, "weight": weight})

        if not all_levels:
            return []

        prices = [l["prix"] for l in all_levels]
        avg_price = np.mean(prices) if prices else 1
        tol = avg_price * tolerance_pct

        # Cluster les niveaux proches
        zones = []
        used = set()

        for i, level in enumerate(all_levels):
            if i in used:
                continue
            cluster = [level]
            cluster_idx = {i}
            for j, other in enumerate(all_levels):
                if j != i and j not in used and abs(level["prix"] - other["prix"]) <= tol:
                    cluster.append(other)
                    cluster_idx.add(j)

            if len(cluster) >= 2:
                avg_px = np.mean([l["prix"] for l in cluster])
                tfs = list(set(l["tf"] for l in cluster))
                types = list(set(l["type"] for l in cluster))
                total_weight = sum(l["weight"] for l in cluster)
                zones.append({
                    "prix": round(avg_px, 2),
                    "timeframes": tfs,
                    "types": types,
                    "nb_confluences": len(cluster),
                    "force": "FORTE" if len(tfs) >= 2 else "MOYENNE",
                    "poids_total": total_weight,
                })
                used.update(cluster_idx)

        return sorted(zones, key=lambda z: -z["poids_total"])

    def get_confluence(self) -> ConfluenceResult:
        """Retourne le résultat de confluence (lance run() si nécessaire)."""
        if not self._ran:
            self.run()
        return self._confluence

    def get_result(self, timeframe: str) -> Optional[TimeframeAnalysis]:
        """Retourne l'analyse d'un timeframe spécifique."""
        if not self._ran:
            self.run()
        return self._results.get(timeframe)

    def _trend_bar(self, score: float, width: int = 20) -> str:
        """Affiche une barre visuelle de score (-100 à +100)."""
        mid = width // 2
        filled = int(abs(score) / 100 * mid)
        if score >= 0:
            bar = " " * mid + "█" * filled + " " * (mid - filled)
        else:
            bar = " " * (mid - filled) + "█" * filled + " " * mid
        center = mid
        bar_list = list(bar)
        bar_list[center] = "|"
        return "[" + "".join(bar_list) + "]"

    def full_report(self) -> str:
        """Génère un rapport multi-timeframes complet."""
        if not self._ran:
            self.run()

        confluence = self._confluence
        symbol = self.symbol

        lines = [
            "=" * 70,
            f"  ANALYSE MULTI-TIMEFRAMES — {symbol}",
            "=" * 70,
        ]

        # ---- Résumé de confluence ----
        lines += [
            "",
            "┌─ CONFLUENCE DES TIMEFRAMES " + "─" * 41,
            f"│  Direction      : {confluence.direction}",
            f"│  Score pondéré  : {confluence.score_pondere:+.1f} / 100",
            f"│  Recommandation : {confluence.recommandation}",
            f"│  Alignement     : {confluence.nb_timeframes_alignes}/{confluence.nb_timeframes_total} timeframes",
            f"│  Analyse        : {confluence.description}",
            "└" + "─" * 68,
        ]

        # ---- Tableau comparatif ----
        lines += [
            "",
            f"  {'Timeframe':<22} {'Score':>7}  {'Signal':<10}  {'Tendance':<12}  {'RSI':>5}  {'MACD':<22}  {'Vol'}",
            "  " + "─" * 92,
        ]

        for tf in self.timeframes:
            if tf not in self._results:
                continue
            a = self._results[tf]
            label = TIMEFRAME_LABELS.get(tf, tf)
            bar = self._trend_bar(a.score, 12)
            lines.append(
                f"  {label:<22} {a.score:>+6.1f}  {a.recommandation:<10}  "
                f"{a.tendance:<12}  {a.rsi_value:>5.1f}  {a.macd_signal:<22}  {a.volatilite}"
            )

        # ---- Détail par timeframe ----
        for tf in self.timeframes:
            if tf not in self._results:
                continue
            a = self._results[tf]
            p = TIMEFRAME_PARAMS.get(tf, TIMEFRAME_PARAMS["1d"])
            label = TIMEFRAME_LABELS.get(tf, tf)
            bar = self._trend_bar(a.score)

            lines += [
                "",
                f"╔══ {label.upper()} {'═'*(65-len(label))}",
                f"║  Score          : {a.score:+.1f}/100  {bar}",
                f"║  Recommandation : {a.recommandation}",
                f"║  Tendance       : {a.tendance}",
                f"║  Prix actuel    : {a.prix_actuel:.2f} MAD",
                f"║",
                f"║  Moyennes mobiles (adaptées {label}) :",
                f"║    SMA {p['sma_fast']:>3} (rapide)  : {a.sma_fast} MAD",
                f"║    SMA {p['sma_slow']:>3} (lente)   : {a.sma_slow} MAD",
                f"║    SMA {p['sma_long']:>3} (long)    : {a.sma_long} MAD",
                f"║",
                f"║  Indicateurs :",
                f"║    RSI ({p['rsi_period']})          : {a.rsi_value:.1f}  → {a.rsi_signal}",
                f"║    MACD                : {a.macd_signal}",
                f"║    Bollinger           : {a.bb_signal}",
                f"║    MA Cross            : {a.ma_signal}",
                f"║    Volatilité (ATR)    : {a.atr:.2f} MAD  → {a.volatilite}",
            ]

            if a.supports:
                lines.append(f"║    Support dyn.        : {', '.join(str(s) for s in a.supports[:2])} MAD")
            if a.resistances:
                lines.append(f"║    Résistance dyn.     : {', '.join(str(r) for r in a.resistances[:2])} MAD")
            if a.fib_support:
                lines.append(f"║    Support Fibo        : {a.fib_support} MAD")
            if a.fib_resistance:
                lines.append(f"║    Résistance Fibo     : {a.fib_resistance} MAD")

            if a.candlestick_patterns:
                lines.append(f"║")
                lines.append(f"║  Patterns récents :")
                for cp in a.candlestick_patterns[:3]:
                    date_str = str(cp["date"])[:10]
                    lines.append(f"║    {date_str}  {cp['pattern']:<26}  {cp['direction']}")

            if a.chart_patterns:
                lines.append(f"║")
                lines.append(f"║  Configurations graphiques :")
                for gp in a.chart_patterns[:2]:
                    lines.append(f"║    {gp.nom} → {gp.direction}  "
                                 f"Breakout: {gp.niveau_breakout} MAD  "
                                 f"({gp.fiabilite})")

            lines.append(f"╚{'═'*68}")

        # ---- Zones de confluence ----
        if confluence.zones_confluence:
            lines += ["", "[ ZONES DE CONFLUENCE S/R (multi-timeframes) ]"]
            current_price = list(self._results.values())[0].prix_actuel
            for zone in confluence.zones_confluence[:6]:
                tfs_str = " + ".join(TIMEFRAME_LABELS.get(t, t) for t in zone["timeframes"])
                zone_type = "SUPPORT" if zone["prix"] < current_price else "RÉSISTANCE"
                dist = ((current_price - zone["prix"]) / current_price) * 100
                lines.append(
                    f"  {zone_type:<12} {zone['prix']:>8.2f} MAD  "
                    f"({dist:+.2f}%)  |  {tfs_str}  |  Force: {zone['force']}"
                )

        lines += ["", "=" * 70]
        return "\n".join(lines)
