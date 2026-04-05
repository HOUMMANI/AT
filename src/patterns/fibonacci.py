"""
Analyse de Fibonacci : retracements, extensions, fan, time zones.

Les niveaux de Fibonacci sont des zones de support/résistance clés
basées sur les ratios de la suite de Fibonacci (23.6%, 38.2%, 50%,
61.8%, 78.6%, 100%, 127.2%, 161.8%, 261.8%).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Ratios Fibonacci standards
FIBO_RETRACEMENTS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIBO_EXTENSIONS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
FIBO_ALL = sorted(set(FIBO_RETRACEMENTS + FIBO_EXTENSIONS))


@dataclass
class FibLevel:
    ratio: float
    prix: float
    type: str       # "retracement" | "extension"
    zone: str       # "support" | "resistance" | "neutre"
    distance_pct: float  # distance au prix actuel en %


@dataclass
class FibAnalysis:
    direction: str          # "HAUSSIER" | "BAISSIER"
    swing_high: float
    swing_low: float
    swing_high_date: object
    swing_low_date: object
    prix_actuel: float
    niveaux: List[FibLevel]
    niveau_actuel: Optional[FibLevel]   # niveau Fibo le plus proche
    prochain_support: Optional[FibLevel]
    prochaine_resistance: Optional[FibLevel]


class FibonacciAnalyzer:
    """
    Calcule et analyse les niveaux de Fibonacci sur un DataFrame OHLCV.

    Exemple:
        fib = FibonacciAnalyzer(df)
        analysis = fib.analyze()
        print(fib.report())

        # Niveaux personnalisés
        levels = fib.retracements(high=520, low=380)
    """

    def __init__(self, df: pd.DataFrame, lookback: int = 60):
        """
        Args:
            df: DataFrame OHLCV
            lookback: Période pour détecter le swing automatiquement (défaut: 60)
        """
        self.df = df.copy()
        self.lookback = min(lookback, len(df))
        self._swing_high = None
        self._swing_low = None
        self._swing_high_date = None
        self._swing_low_date = None
        self._detect_swing()

    def _detect_swing(self):
        """Détecte automatiquement le swing haut et bas récent."""
        df = self.df.tail(self.lookback)
        self._swing_high = df["High"].max()
        self._swing_high_date = df["High"].idxmax()
        self._swing_low = df["Low"].min()
        self._swing_low_date = df["Low"].idxmin()

    def set_swing(self, high: float, low: float,
                  high_date=None, low_date=None):
        """
        Définit manuellement le swing haut et bas.

        Args:
            high: Prix du swing haut
            low: Prix du swing bas
            high_date: Date du swing haut (optionnel)
            low_date: Date du swing bas (optionnel)
        """
        self._swing_high = high
        self._swing_low = low
        self._swing_high_date = high_date
        self._swing_low_date = low_date

    def retracements(
        self,
        high: Optional[float] = None,
        low: Optional[float] = None,
        ratios: List[float] = None,
    ) -> dict:
        """
        Calcule les niveaux de retracement de Fibonacci.

        Args:
            high: Prix haut (défaut: swing détecté)
            low: Prix bas (défaut: swing détecté)
            ratios: Ratios à calculer (défaut: FIBO_RETRACEMENTS)

        Returns:
            Dict {ratio: prix}
        """
        high = high or self._swing_high
        low = low or self._swing_low
        ratios = ratios or FIBO_RETRACEMENTS
        rang = high - low

        levels = {}
        for r in ratios:
            levels[r] = round(high - rang * r, 4)
        return levels

    def extensions(
        self,
        high: Optional[float] = None,
        low: Optional[float] = None,
        ratios: List[float] = None,
    ) -> dict:
        """
        Calcule les niveaux d'extension de Fibonacci.

        Args:
            high: Prix haut
            low: Prix bas
            ratios: Ratios (défaut: FIBO_EXTENSIONS)

        Returns:
            Dict {ratio: prix}
        """
        high = high or self._swing_high
        low = low or self._swing_low
        ratios = ratios or FIBO_EXTENSIONS
        rang = high - low

        levels = {}
        for r in ratios:
            # Extension depuis le bas (projection haussière)
            levels[r] = round(low + rang * r, 4)
        return levels

    def fan_lines(self) -> dict:
        """
        Calcule les lignes de Fan de Fibonacci.

        Les lignes fan partent du swing bas vers le swing haut en utilisant
        les ratios 38.2%, 50% et 61.8% du temps écoulé.

        Returns:
            Dict avec les pentes et points de départ des lignes
        """
        high = self._swing_high
        low = self._swing_low
        rang = high - low

        # Projections: fan depuis le creux
        fan_levels = {
            0.382: round(high - rang * 0.382, 2),
            0.500: round(high - rang * 0.500, 2),
            0.618: round(high - rang * 0.618, 2),
        }
        return fan_levels

    def time_zones(self, lookback: int = None) -> list:
        """
        Calcule les Fibonacci Time Zones à partir du swing bas.

        Les zones temporelles de Fibonacci projettent des moments clés
        potentiels dans le futur.

        Returns:
            Liste des indices de temps projetés
        """
        df = self.df
        lookback = lookback or self.lookback
        swing_idx = df.index.get_loc(self._swing_low_date) if self._swing_low_date in df.index else 0

        fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        zones = []
        for fib in fib_sequence:
            target_idx = swing_idx + fib
            if target_idx < len(df):
                zones.append({
                    "fib": fib,
                    "date": df.index[target_idx],
                    "prix_close": round(df["Close"].iloc[target_idx], 2),
                })
        return zones

    def find_nearest_level(
        self,
        price: float,
        levels: dict,
        tolerance_pct: float = 0.02,
    ) -> Optional[Tuple[float, float]]:
        """
        Trouve le niveau Fibonacci le plus proche du prix donné.

        Args:
            price: Prix de référence
            levels: Dict {ratio: prix}
            tolerance_pct: Tolérance en % (défaut: 2%)

        Returns:
            (ratio, prix_niveau) ou None
        """
        nearest = None
        min_dist = float("inf")

        for ratio, level_price in levels.items():
            dist = abs(price - level_price) / price
            if dist < min_dist and dist <= tolerance_pct:
                min_dist = dist
                nearest = (ratio, level_price)

        return nearest

    def analyze(self) -> FibAnalysis:
        """
        Analyse complète de Fibonacci.

        Détermine la direction du mouvement, calcule tous les niveaux,
        identifie le niveau actuel et les prochains supports/résistances.

        Returns:
            FibAnalysis avec tous les niveaux et le contexte
        """
        high = self._swing_high
        low = self._swing_low
        current = self.df["Close"].iloc[-1]

        # Déterminer la direction
        high_date = self._swing_high_date
        low_date = self._swing_low_date
        direction = "HAUSSIER" if low_date < high_date else "BAISSIER"

        # Calculer les retracements
        retrace_levels = self.retracements()

        # Construire la liste des niveaux
        niveaux = []
        for ratio, prix in sorted(retrace_levels.items(), key=lambda x: x[1], reverse=True):
            dist_pct = ((current - prix) / current) * 100
            zone = "resistance" if prix > current else "support"
            niveaux.append(FibLevel(
                ratio=ratio,
                prix=round(prix, 2),
                type="retracement",
                zone=zone,
                distance_pct=round(dist_pct, 2),
            ))

        # Ajouter les extensions
        ext_levels = self.extensions()
        for ratio, prix in sorted(ext_levels.items(), key=lambda x: x[1], reverse=True):
            if prix != high and prix != low:  # éviter les doublons
                dist_pct = ((current - prix) / current) * 100
                zone = "resistance" if prix > current else "support"
                niveaux.append(FibLevel(
                    ratio=ratio,
                    prix=round(prix, 2),
                    type="extension",
                    zone=zone,
                    distance_pct=round(dist_pct, 2),
                ))

        # Niveau actuel (le plus proche)
        nearest = min(niveaux, key=lambda l: abs(l.distance_pct)) if niveaux else None

        # Prochain support (prix < current, le plus proche par le bas)
        supports = [l for l in niveaux if l.prix < current]
        prochaine_support = min(supports, key=lambda l: abs(l.distance_pct)) if supports else None

        # Prochaine résistance
        resistances = [l for l in niveaux if l.prix > current]
        prochaine_resistance = min(resistances, key=lambda l: abs(l.distance_pct)) if resistances else None

        return FibAnalysis(
            direction=direction,
            swing_high=round(high, 2),
            swing_low=round(low, 2),
            swing_high_date=high_date,
            swing_low_date=low_date,
            prix_actuel=round(current, 2),
            niveaux=sorted(niveaux, key=lambda l: l.prix, reverse=True),
            niveau_actuel=nearest,
            prochain_support=prochaine_support,
            prochaine_resistance=prochaine_resistance,
        )

    def report(self) -> str:
        """Génère un rapport texte complet de l'analyse Fibonacci."""
        analysis = self.analyze()
        lines = [
            "\n[ ANALYSE FIBONACCI ]",
            f"  Swing Haut  : {analysis.swing_high} MAD  ({str(analysis.swing_high_date)[:10]})",
            f"  Swing Bas   : {analysis.swing_low} MAD  ({str(analysis.swing_low_date)[:10]})",
            f"  Mouvement   : {analysis.direction}",
            f"  Prix actuel : {analysis.prix_actuel} MAD",
            "",
            "  Niveaux de Retracement/Extension :",
            f"  {'Ratio':<10} {'Prix (MAD)':<14} {'Type':<14} {'Zone':<12} {'Distance'}",
            "  " + "-" * 62,
        ]

        for lvl in analysis.niveaux:
            marker = " ← ACTUEL" if analysis.niveau_actuel and lvl.prix == analysis.niveau_actuel.prix else ""
            lines.append(
                f"  {lvl.ratio*100:>5.1f}%   {lvl.prix:<14.2f} {lvl.type:<14} "
                f"{lvl.zone:<12} {lvl.distance_pct:+.2f}%{marker}"
            )

        lines.append("")
        if analysis.prochaine_resistance:
            r = analysis.prochaine_resistance
            lines.append(f"  Prochaine résistance Fibo : {r.prix} MAD ({r.ratio*100:.1f}%)")
        if analysis.prochain_support:
            s = analysis.prochain_support
            lines.append(f"  Prochain support Fibo     : {s.prix} MAD ({s.ratio*100:.1f}%)")

        return "\n".join(lines)
