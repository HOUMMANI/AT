"""
Détection automatique de lignes de tendance (trendlines).

Détecte :
- Lignes de tendance haussière (reliant les creux successifs croissants)
- Lignes de tendance baissière (reliant les sommets successifs décroissants)
- Force de la ligne (nombre de touches, R²)
- Cassures de tendance (breakouts)
- Niveaux actuels de support/résistance dynamique
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import argrelextrema


@dataclass
class Trendline:
    type: str               # "HAUSSIERE" | "BAISSIERE" | "HORIZONTALE"
    direction: str          # "SUPPORT" | "RESISTANCE"
    date_debut: object
    date_fin: object
    prix_debut: float
    prix_fin: float
    pente: float            # pente en MAD/jour
    r_squared: float        # qualité du fit (0-1)
    nb_touches: int         # nombre de points sur la ligne
    prix_actuel: float      # prix de la ligne aujourd'hui
    distance_pct: float     # distance au prix actuel (%)
    est_cassee: bool        # True si la ligne a été cassée récemment
    force: str              # "Faible" | "Moyenne" | "Forte" | "Très forte"


class TrendlineDetector:
    """
    Détecte et analyse les lignes de tendance automatiquement.

    Exemple:
        td = TrendlineDetector(df)
        lines = td.detect_all()
        print(td.report())

        # Ligne de tendance la plus forte
        best = td.get_strongest()
    """

    def __init__(self, df: pd.DataFrame, order: int = 5, min_touches: int = 2):
        """
        Args:
            df: DataFrame OHLCV
            order: Rayon de recherche des pivots (défaut: 5)
            min_touches: Nombre minimum de touches pour valider une ligne (défaut: 2)
        """
        if len(df) < 20:
            raise ValueError("Données insuffisantes (minimum 20 bougies)")
        self.df = df.copy()
        self.order = order
        self.min_touches = min_touches
        self._find_pivots()

    def _find_pivots(self):
        """Identifie les pivots hauts et bas locaux."""
        high = self.df["High"].values
        low = self.df["Low"].values

        peak_idx = argrelextrema(high, np.greater_equal, order=self.order)[0]
        trough_idx = argrelextrema(low, np.less_equal, order=self.order)[0]

        # Dédupliquer les pivots adjacents
        peak_idx = self._deduplicate(peak_idx)
        trough_idx = self._deduplicate(trough_idx)

        self.peaks = [(i, self.df.index[i], high[i]) for i in peak_idx]
        self.troughs = [(i, self.df.index[i], low[i]) for i in trough_idx]

    def _deduplicate(self, indices: np.ndarray, gap: int = 2) -> np.ndarray:
        """Supprime les pivots trop proches."""
        if len(indices) == 0:
            return indices
        result = [indices[0]]
        for idx in indices[1:]:
            if idx - result[-1] >= gap:
                result.append(idx)
        return np.array(result)

    def _fit_trendline(
        self,
        points: List[Tuple[int, object, float]],
        price_series: pd.Series,
        tolerance_pct: float = 0.02,
    ) -> Optional[Trendline]:
        """
        Ajuste une ligne de tendance sur un ensemble de points.

        Args:
            points: Liste de (index, date, prix)
            price_series: Série de prix (High pour résistance, Low pour support)
            tolerance_pct: Tolérance pour compter les touches

        Returns:
            Trendline ou None si insuffisant
        """
        if len(points) < 2:
            return None

        indices = np.array([p[0] for p in points])
        prices = np.array([p[2] for p in points])

        if len(indices) < 2:
            return None

        # Régression linéaire
        slope, intercept = np.polyfit(indices, prices, 1)

        # R² (qualité du fit)
        y_pred = slope * indices + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Compter les touches (tous les points proches de la ligne)
        all_indices = np.arange(len(price_series))
        line_values = slope * all_indices + intercept
        actual_values = price_series.values
        touches = 0
        for i in range(len(actual_values)):
            if abs(actual_values[i] - line_values[i]) / line_values[i] <= tolerance_pct:
                touches += 1

        if touches < self.min_touches:
            return None

        # Prix actuel de la ligne
        last_idx = len(self.df) - 1
        line_price_now = slope * last_idx + intercept
        current_price = self.df["Close"].iloc[-1]
        distance = ((current_price - line_price_now) / current_price) * 100

        # Force
        if touches >= 5 and r2 > 0.8:
            force = "Très forte"
        elif touches >= 4 and r2 > 0.6:
            force = "Forte"
        elif touches >= 3 and r2 > 0.4:
            force = "Moyenne"
        else:
            force = "Faible"

        # Cassure récente (prix a franchi la ligne dans les 5 dernières bougies)
        recent = self.df.tail(5)
        recent_indices = np.arange(last_idx - 4, last_idx + 1)
        recent_line = slope * recent_indices + intercept
        cassure = False

        if slope >= 0:  # ligne haussière → support
            cassure = any(recent["Close"].values[i] < recent_line[i] * 0.99
                          for i in range(len(recent)))
        else:           # ligne baissière → résistance
            cassure = any(recent["Close"].values[i] > recent_line[i] * 1.01
                          for i in range(len(recent)))

        # Type de ligne
        if abs(slope) < current_price * 0.0001:
            tl_type = "HORIZONTALE"
        elif slope > 0:
            tl_type = "HAUSSIERE"
        else:
            tl_type = "BAISSIERE"

        return Trendline(
            type=tl_type,
            direction="SUPPORT" if slope >= 0 else "RESISTANCE",
            date_debut=points[0][1],
            date_fin=points[-1][1],
            prix_debut=round(prices[0], 2),
            prix_fin=round(prices[-1], 2),
            pente=round(slope, 4),
            r_squared=round(r2, 3),
            nb_touches=touches,
            prix_actuel=round(line_price_now, 2),
            distance_pct=round(distance, 2),
            est_cassee=cassure,
            force=force,
        )

    def detect_support_lines(self) -> List[Trendline]:
        """
        Détecte les lignes de tendance haussières (support dynamique).
        Relie les creux successifs croissants.
        """
        lines = []
        troughs = self.troughs

        if len(troughs) < 2:
            return lines

        # Essayer toutes les paires de creux
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                t1_idx, t1_date, t1_price = troughs[i]
                t2_idx, t2_date, t2_price = troughs[j]

                # Ligne haussière : second creux plus haut
                if t2_price > t1_price:
                    line = self._fit_trendline(
                        [troughs[i], troughs[j]],
                        self.df["Low"],
                    )
                    if line and not line.est_cassee:
                        lines.append(line)

        # Dédupliquer les lignes similaires
        return self._deduplicate_lines(lines)

    def detect_resistance_lines(self) -> List[Trendline]:
        """
        Détecte les lignes de tendance baissières (résistance dynamique).
        Relie les sommets successifs décroissants.
        """
        lines = []
        peaks = self.peaks

        if len(peaks) < 2:
            return lines

        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1_idx, p1_date, p1_price = peaks[i]
                p2_idx, p2_date, p2_price = peaks[j]

                # Ligne baissière : second sommet plus bas
                if p2_price < p1_price:
                    line = self._fit_trendline(
                        [peaks[i], peaks[j]],
                        self.df["High"],
                    )
                    if line and not line.est_cassee:
                        lines.append(line)

        return self._deduplicate_lines(lines)

    def detect_horizontal_levels(self, tolerance_pct: float = 0.02) -> List[Trendline]:
        """
        Détecte les niveaux horizontaux de support/résistance clés.
        Un niveau est validé quand le prix y revient plusieurs fois.
        """
        levels = []
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]
        avg_price = close.mean()
        tol = avg_price * tolerance_pct

        # Regrouper les pivots par cluster de prix
        all_pivots = (
            [(h, "resistance") for _, _, h in self.peaks] +
            [(l, "support") for _, _, l in self.troughs]
        )

        used = set()
        for i, (price, ptype) in enumerate(all_pivots):
            if i in used:
                continue
            cluster = [price]
            cluster_types = [ptype]
            for j, (price2, ptype2) in enumerate(all_pivots):
                if j != i and j not in used and abs(price - price2) <= tol:
                    cluster.append(price2)
                    cluster_types.append(ptype2)
                    used.add(j)
            used.add(i)

            if len(cluster) >= 2:
                avg_level = np.mean(cluster)
                current = close.iloc[-1]
                distance = ((current - avg_level) / current) * 100

                touches = sum(
                    1 for idx in range(len(close))
                    if abs(close.iloc[idx] - avg_level) / avg_level <= tolerance_pct
                )

                zone = "RESISTANCE" if avg_level > current else "SUPPORT"
                force = "Très forte" if touches >= 5 else "Forte" if touches >= 4 else "Moyenne" if touches >= 3 else "Faible"

                levels.append(Trendline(
                    type="HORIZONTALE",
                    direction=zone,
                    date_debut=self.df.index[0],
                    date_fin=self.df.index[-1],
                    prix_debut=round(avg_level, 2),
                    prix_fin=round(avg_level, 2),
                    pente=0.0,
                    r_squared=1.0,
                    nb_touches=touches,
                    prix_actuel=round(avg_level, 2),
                    distance_pct=round(distance, 2),
                    est_cassee=False,
                    force=force,
                ))

        return sorted(levels, key=lambda l: abs(l.distance_pct))[:8]

    def _deduplicate_lines(self, lines: List[Trendline], pct_tol: float = 0.02) -> List[Trendline]:
        """Supprime les lignes trop similaires."""
        if not lines:
            return lines
        kept = []
        for line in sorted(lines, key=lambda l: (-l.nb_touches, -l.r_squared)):
            duplicate = False
            for k in kept:
                if (abs(line.prix_actuel - k.prix_actuel) / k.prix_actuel < pct_tol and
                        line.type == k.type):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(line)
        return kept[:6]  # garder les 6 meilleures

    def detect_all(self) -> List[Trendline]:
        """
        Détecte toutes les lignes de tendance actives.

        Returns:
            Liste de Trendline triée par force
        """
        all_lines = []
        all_lines.extend(self.detect_support_lines())
        all_lines.extend(self.detect_resistance_lines())
        all_lines.extend(self.detect_horizontal_levels())

        # Trier par force et nb touches
        order = {"Très forte": 0, "Forte": 1, "Moyenne": 2, "Faible": 3}
        return sorted(all_lines, key=lambda l: (order.get(l.force, 4), abs(l.distance_pct)))

    def get_strongest(self) -> Optional[Trendline]:
        """Retourne la ligne de tendance la plus forte."""
        lines = self.detect_all()
        return lines[0] if lines else None

    def get_nearest_support(self) -> Optional[Trendline]:
        """Retourne le support dynamique le plus proche en-dessous du prix."""
        current = self.df["Close"].iloc[-1]
        lines = [l for l in self.detect_all() if l.prix_actuel < current and not l.est_cassee]
        return min(lines, key=lambda l: abs(l.distance_pct)) if lines else None

    def get_nearest_resistance(self) -> Optional[Trendline]:
        """Retourne la résistance dynamique la plus proche au-dessus du prix."""
        current = self.df["Close"].iloc[-1]
        lines = [l for l in self.detect_all() if l.prix_actuel > current and not l.est_cassee]
        return min(lines, key=lambda l: abs(l.distance_pct)) if lines else None

    def report(self) -> str:
        """Génère un rapport texte des lignes de tendance."""
        lines = self.detect_all()
        current = self.df["Close"].iloc[-1]

        text = ["\n[ LIGNES DE TENDANCE ]"]

        if not lines:
            text.append("  Aucune ligne de tendance significative détectée.")
            return "\n".join(text)

        text.append(f"  {'Type':<14} {'Direction':<12} {'Prix ligne':<12} {'Distance':<12} {'Touches':<10} {'Force'}")
        text.append("  " + "-" * 72)

        for tl in lines:
            cassee = " [CASSÉE]" if tl.est_cassee else ""
            text.append(
                f"  {tl.type:<14} {tl.direction:<12} {tl.prix_actuel:<12.2f} "
                f"{tl.distance_pct:>+6.2f}%     {tl.nb_touches:<10} {tl.force}{cassee}"
            )

        # Résumé
        support = self.get_nearest_support()
        resistance = self.get_nearest_resistance()

        text.append("")
        if support:
            text.append(f"  Support dynamique   : {support.prix_actuel} MAD "
                        f"({support.distance_pct:+.2f}%, {support.force})")
        if resistance:
            text.append(f"  Résistance dynamique: {resistance.prix_actuel} MAD "
                        f"({resistance.distance_pct:+.2f}%, {resistance.force})")

        return "\n".join(text)
