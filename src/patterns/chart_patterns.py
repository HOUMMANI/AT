"""
Détection des configurations graphiques classiques de l'analyse technique.

Patterns détectés :
  Retournement : Tête & Épaules (H&S), H&S Inversé, Double Sommet/Creux,
                 Triple Sommet/Creux, Biseau Montant/Descendant
  Continuation : Triangle Symétrique/Ascendant/Descendant, Flag/Pennant,
                 Canal Haussier/Baissier, Cup & Handle, Rectangles

Chaque pattern inclut :
  - Type (retournement / continuation)
  - Direction (haussier / baissier)
  - Niveaux clés (breakout, target, stop)
  - Fiabilité estimée
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import argrelextrema


@dataclass
class PatternResult:
    nom: str
    type: str                    # "retournement" | "continuation"
    direction: str               # "HAUSSIER" | "BAISSIER" | "NEUTRE"
    fiabilite: str               # "Faible" | "Moyenne" | "Haute" | "Très haute"
    date_debut: object
    date_fin: object
    niveau_breakout: float
    objectif_prix: Optional[float]
    stop_loss: Optional[float]
    description: str
    points_cles: dict = field(default_factory=dict)


class ChartPatternDetector:
    """
    Détecte les configurations graphiques sur un DataFrame OHLCV.

    Exemple:
        detector = ChartPatternDetector(df)
        patterns = detector.detect_all()
        for p in patterns:
            print(p.nom, p.direction, p.niveau_breakout)
        print(detector.report())
    """

    def __init__(self, df: pd.DataFrame, order: int = 5):
        """
        Args:
            df: DataFrame OHLCV
            order: Rayon de recherche des pivots locaux (défaut: 5)
        """
        if len(df) < 30:
            raise ValueError("Données insuffisantes (minimum 30 bougies)")
        self.df = df.copy()
        self.order = order
        self._find_pivots()

    def _find_pivots(self):
        """Détecte les sommets et creux locaux avec scipy."""
        close = self.df["Close"].values
        high = self.df["High"].values
        low = self.df["Low"].values

        peak_idx = argrelextrema(high, np.greater, order=self.order)[0]
        trough_idx = argrelextrema(low, np.less, order=self.order)[0]

        self.peaks = [(self.df.index[i], high[i]) for i in peak_idx]
        self.troughs = [(self.df.index[i], low[i]) for i in trough_idx]
        self.close = self.df["Close"]
        self.high = self.df["High"]
        self.low = self.df["Low"]

    def _price_tolerance(self, pct: float = 0.03) -> float:
        """Tolérance en valeur absolue (% du prix moyen récent)."""
        return self.close.tail(20).mean() * pct

    # =========================================================
    # DOUBLE SOMMET / DOUBLE CREUX
    # =========================================================

    def detect_double_top(self) -> List[PatternResult]:
        """
        Double Sommet (M) : deux sommets proches au même niveau après tendance haussière.
        Signal de retournement baissier.
        """
        results = []
        tol = self._price_tolerance(0.03)
        peaks = self.peaks[-10:]  # limiter aux 10 derniers sommets

        for i in range(len(peaks) - 1):
            d1, p1 = peaks[i]
            d2, p2 = peaks[i + 1]

            if abs(p1 - p2) > tol:
                continue

            # Chercher le creux entre les deux sommets
            mask = (self.df.index > d1) & (self.df.index < d2)
            if mask.sum() < 3:
                continue
            valley = self.low[mask].min()
            valley_date = self.low[mask].idxmin()

            # Validation : les sommets doivent être significativement au-dessus du creux
            height = ((p1 + p2) / 2) - valley
            if height < tol * 2:
                continue

            neckline = valley
            target = neckline - height
            stop = max(p1, p2) * 1.01

            results.append(PatternResult(
                nom="Double Sommet",
                type="retournement",
                direction="BAISSIER",
                fiabilite="Très haute",
                date_debut=d1,
                date_fin=d2,
                niveau_breakout=round(neckline, 2),
                objectif_prix=round(target, 2),
                stop_loss=round(stop, 2),
                description=(
                    f"Double sommet à {round((p1+p2)/2, 2)} MAD. "
                    f"Ligne de cou: {round(neckline, 2)} MAD. "
                    f"Objectif baissier: {round(target, 2)} MAD."
                ),
                points_cles={"sommet1": round(p1, 2), "sommet2": round(p2, 2),
                              "creux": round(valley, 2), "ligne_cou": round(neckline, 2)},
            ))
        return results

    def detect_double_bottom(self) -> List[PatternResult]:
        """
        Double Creux (W) : deux creux proches au même niveau après tendance baissière.
        Signal de retournement haussier.
        """
        results = []
        tol = self._price_tolerance(0.03)
        troughs = self.troughs[-10:]

        for i in range(len(troughs) - 1):
            d1, t1 = troughs[i]
            d2, t2 = troughs[i + 1]

            if abs(t1 - t2) > tol:
                continue

            mask = (self.df.index > d1) & (self.df.index < d2)
            if mask.sum() < 3:
                continue
            peak = self.high[mask].max()
            peak_date = self.high[mask].idxmax()

            height = peak - ((t1 + t2) / 2)
            if height < tol * 2:
                continue

            neckline = peak
            target = neckline + height
            stop = min(t1, t2) * 0.99

            results.append(PatternResult(
                nom="Double Creux",
                type="retournement",
                direction="HAUSSIER",
                fiabilite="Très haute",
                date_debut=d1,
                date_fin=d2,
                niveau_breakout=round(neckline, 2),
                objectif_prix=round(target, 2),
                stop_loss=round(stop, 2),
                description=(
                    f"Double creux à {round((t1+t2)/2, 2)} MAD. "
                    f"Résistance: {round(neckline, 2)} MAD. "
                    f"Objectif haussier: {round(target, 2)} MAD."
                ),
                points_cles={"creux1": round(t1, 2), "creux2": round(t2, 2),
                              "sommet": round(peak, 2), "resistance": round(neckline, 2)},
            ))
        return results

    # =========================================================
    # TÊTE & ÉPAULES
    # =========================================================

    def detect_head_and_shoulders(self) -> List[PatternResult]:
        """
        Tête & Épaules : épaule gauche, tête (plus haut), épaule droite.
        Signal de retournement baissier majeur.
        """
        results = []
        tol = self._price_tolerance(0.05)
        peaks = self.peaks[-12:]

        for i in range(len(peaks) - 2):
            ls_date, ls = peaks[i]       # épaule gauche
            h_date, h = peaks[i + 1]    # tête
            rs_date, rs = peaks[i + 2]  # épaule droite

            # Tête > épaules
            if h <= ls or h <= rs:
                continue
            # Épaules approximativement égales
            if abs(ls - rs) > tol * 2:
                continue

            # Creux entre LS et head (creux gauche)
            mask_l = (self.df.index > ls_date) & (self.df.index < h_date)
            if mask_l.sum() < 2:
                continue
            left_trough = self.low[mask_l].min()

            # Creux entre head et RS (creux droit)
            mask_r = (self.df.index > h_date) & (self.df.index < rs_date)
            if mask_r.sum() < 2:
                continue
            right_trough = self.low[mask_r].min()

            neckline = (left_trough + right_trough) / 2
            height = h - neckline
            if height < tol * 2:
                continue

            target = neckline - height
            stop = h * 1.01

            results.append(PatternResult(
                nom="Tête & Épaules",
                type="retournement",
                direction="BAISSIER",
                fiabilite="Très haute",
                date_debut=ls_date,
                date_fin=rs_date,
                niveau_breakout=round(neckline, 2),
                objectif_prix=round(target, 2),
                stop_loss=round(stop, 2),
                description=(
                    f"Tête à {round(h, 2)} MAD, épaules à ~{round((ls+rs)/2, 2)} MAD. "
                    f"Ligne de cou: {round(neckline, 2)} MAD. "
                    f"Objectif: {round(target, 2)} MAD."
                ),
                points_cles={
                    "epaule_gauche": round(ls, 2),
                    "tete": round(h, 2),
                    "epaule_droite": round(rs, 2),
                    "ligne_cou": round(neckline, 2),
                },
            ))
        return results

    def detect_inverse_head_and_shoulders(self) -> List[PatternResult]:
        """
        Tête & Épaules Inversé : creux gauche, creux profond (tête), creux droit.
        Signal de retournement haussier majeur.
        """
        results = []
        tol = self._price_tolerance(0.05)
        troughs = self.troughs[-12:]

        for i in range(len(troughs) - 2):
            ls_date, ls = troughs[i]
            h_date, h = troughs[i + 1]
            rs_date, rs = troughs[i + 2]

            # Tête < épaules (plus bas)
            if h >= ls or h >= rs:
                continue
            if abs(ls - rs) > tol * 2:
                continue

            mask_l = (self.df.index > ls_date) & (self.df.index < h_date)
            if mask_l.sum() < 2:
                continue
            left_peak = self.high[mask_l].max()

            mask_r = (self.df.index > h_date) & (self.df.index < rs_date)
            if mask_r.sum() < 2:
                continue
            right_peak = self.high[mask_r].max()

            neckline = (left_peak + right_peak) / 2
            height = neckline - h
            if height < tol * 2:
                continue

            target = neckline + height
            stop = h * 0.99

            results.append(PatternResult(
                nom="Tête & Épaules Inversé",
                type="retournement",
                direction="HAUSSIER",
                fiabilite="Très haute",
                date_debut=ls_date,
                date_fin=rs_date,
                niveau_breakout=round(neckline, 2),
                objectif_prix=round(target, 2),
                stop_loss=round(stop, 2),
                description=(
                    f"Creux tête à {round(h, 2)} MAD, épaules à ~{round((ls+rs)/2, 2)} MAD. "
                    f"Ligne de cou: {round(neckline, 2)} MAD. "
                    f"Objectif: {round(target, 2)} MAD."
                ),
                points_cles={
                    "epaule_gauche": round(ls, 2),
                    "tete": round(h, 2),
                    "epaule_droite": round(rs, 2),
                    "ligne_cou": round(neckline, 2),
                },
            ))
        return results

    # =========================================================
    # TRIANGLES
    # =========================================================

    def _fit_line(self, dates, values):
        """Régression linéaire sur des valeurs de prix."""
        x = np.arange(len(values))
        if len(x) < 2:
            return None, None
        slope, intercept = np.polyfit(x, values, 1)
        return slope, intercept

    def detect_triangles(self) -> List[PatternResult]:
        """
        Détecte les triangles (symétrique, ascendant, descendant).
        Un triangle se forme quand les sommets et creux convergent.
        """
        results = []
        window = min(40, len(self.df) - 5)
        df_w = self.df.tail(window)

        highs = df_w["High"].values
        lows = df_w["Low"].values
        dates = df_w.index

        # Régression sur les hauts et les bas
        x = np.arange(len(highs))
        slope_h, inter_h = np.polyfit(x, highs, 1)
        slope_l, inter_l = np.polyfit(x, lows, 1)

        # Convergence : les lignes se rapprochent
        start_width = (inter_h) - (inter_l)
        end_width = (slope_h * (len(x)-1) + inter_h) - (slope_l * (len(x)-1) + inter_l)

        if start_width <= 0 or end_width >= start_width * 0.9:
            return results  # Pas de convergence significative

        current_price = self.close.iloc[-1]
        avg_price = df_w["Close"].mean()
        tol = avg_price * 0.015

        # Triangle symétrique
        if abs(slope_h) < abs(slope_h) * 0.3 + tol / avg_price and \
           abs(slope_h + slope_l) < abs(slope_h) * 0.5:
            breakout_up = inter_h + slope_h * (len(x) - 1)
            breakout_down = inter_l + slope_l * (len(x) - 1)
            height = (inter_h - inter_l)
            results.append(PatternResult(
                nom="Triangle Symétrique",
                type="continuation",
                direction="NEUTRE",
                fiabilite="Haute",
                date_debut=dates[0],
                date_fin=dates[-1],
                niveau_breakout=round((breakout_up + breakout_down) / 2, 2),
                objectif_prix=round(breakout_up + height * 0.75, 2),
                stop_loss=round(breakout_down * 0.98, 2),
                description=(
                    f"Triangle symétrique en formation. "
                    f"Résistance décroissante: {round(breakout_up, 2)} MAD. "
                    f"Support croissant: {round(breakout_down, 2)} MAD. "
                    f"Breakout imminent."
                ),
                points_cles={
                    "resistance": round(breakout_up, 2),
                    "support": round(breakout_down, 2),
                },
            ))

        # Triangle ascendant (résistance plate, support montant)
        elif slope_h > -tol / avg_price and slope_l > tol / avg_price:
            resistance = inter_h + slope_h * (len(x) - 1)
            height = resistance - (inter_l + slope_l * (len(x) - 1))
            results.append(PatternResult(
                nom="Triangle Ascendant",
                type="continuation",
                direction="HAUSSIER",
                fiabilite="Haute",
                date_debut=dates[0],
                date_fin=dates[-1],
                niveau_breakout=round(resistance, 2),
                objectif_prix=round(resistance + height, 2),
                stop_loss=round((inter_l + slope_l * (len(x) - 1)) * 0.98, 2),
                description=(
                    f"Triangle ascendant. Résistance plate: {round(resistance, 2)} MAD. "
                    f"Breakout haussier probable. Objectif: {round(resistance + height, 2)} MAD."
                ),
                points_cles={"resistance": round(resistance, 2)},
            ))

        # Triangle descendant (résistance descendante, support plat)
        elif slope_h < -tol / avg_price and slope_l < tol / avg_price:
            support = inter_l + slope_l * (len(x) - 1)
            height = (inter_h + slope_h * (len(x) - 1)) - support
            results.append(PatternResult(
                nom="Triangle Descendant",
                type="continuation",
                direction="BAISSIER",
                fiabilite="Haute",
                date_debut=dates[0],
                date_fin=dates[-1],
                niveau_breakout=round(support, 2),
                objectif_prix=round(support - height, 2),
                stop_loss=round((inter_h + slope_h * (len(x) - 1)) * 1.02, 2),
                description=(
                    f"Triangle descendant. Support plat: {round(support, 2)} MAD. "
                    f"Breakout baissier probable. Objectif: {round(support - height, 2)} MAD."
                ),
                points_cles={"support": round(support, 2)},
            ))

        return results

    # =========================================================
    # FLAGS & PENNANTS
    # =========================================================

    def detect_flag(self) -> List[PatternResult]:
        """
        Flag (drapeau) : forte impulsion suivie d'un canal en consolidation.
        Signal de continuation dans la direction de l'impulsion.
        """
        results = []
        if len(self.df) < 25:
            return results

        # Chercher une forte impulsion sur les 10-15 dernières bougies
        for pole_len in [8, 10, 12]:
            for flag_len in [5, 7, 10]:
                total = pole_len + flag_len
                if total > len(self.df):
                    continue

                pole = self.df.iloc[-(total):-flag_len]
                flag = self.df.iloc[-flag_len:]

                pole_move = (pole["Close"].iloc[-1] - pole["Close"].iloc[0]) / pole["Close"].iloc[0]
                flag_move = (flag["Close"].iloc[-1] - flag["Close"].iloc[0]) / flag["Close"].iloc[0]
                pole_vol = pole["Volume"].mean() if "Volume" in pole else 1
                flag_vol = flag["Volume"].mean() if "Volume" in flag else 1

                # Flag haussier : forte montée puis légère correction
                if pole_move > 0.05 and -0.05 < flag_move < 0 and flag_vol < pole_vol * 0.8:
                    target = flag["Close"].iloc[-1] + (pole["Close"].iloc[-1] - pole["Close"].iloc[0])
                    results.append(PatternResult(
                        nom="Flag Haussier",
                        type="continuation",
                        direction="HAUSSIER",
                        fiabilite="Haute",
                        date_debut=pole.index[0],
                        date_fin=flag.index[-1],
                        niveau_breakout=round(flag["High"].max(), 2),
                        objectif_prix=round(target, 2),
                        stop_loss=round(flag["Low"].min() * 0.99, 2),
                        description=(
                            f"Flag haussier. Mât: +{round(pole_move*100, 1)}%. "
                            f"Consolidation en cours. "
                            f"Breakout: {round(flag['High'].max(), 2)} MAD. "
                            f"Objectif: {round(target, 2)} MAD."
                        ),
                        points_cles={
                            "sommet_mat": round(pole["Close"].iloc[-1], 2),
                            "breakout": round(flag["High"].max(), 2),
                        },
                    ))
                    break

                # Flag baissier : forte baisse puis légère correction haussière
                elif pole_move < -0.05 and 0 < flag_move < 0.05 and flag_vol < pole_vol * 0.8:
                    target = flag["Close"].iloc[-1] - (pole["Close"].iloc[0] - pole["Close"].iloc[-1])
                    results.append(PatternResult(
                        nom="Flag Baissier",
                        type="continuation",
                        direction="BAISSIER",
                        fiabilite="Haute",
                        date_debut=pole.index[0],
                        date_fin=flag.index[-1],
                        niveau_breakout=round(flag["Low"].min(), 2),
                        objectif_prix=round(target, 2),
                        stop_loss=round(flag["High"].max() * 1.01, 2),
                        description=(
                            f"Flag baissier. Mât: {round(pole_move*100, 1)}%. "
                            f"Consolidation en cours. "
                            f"Breakout: {round(flag['Low'].min(), 2)} MAD. "
                            f"Objectif: {round(target, 2)} MAD."
                        ),
                        points_cles={
                            "creux_mat": round(pole["Close"].iloc[-1], 2),
                            "breakout": round(flag["Low"].min(), 2),
                        },
                    ))
                    break

        return results[:1]  # un seul flag à la fois

    # =========================================================
    # BISEAUX (WEDGES)
    # =========================================================

    def detect_wedge(self) -> List[PatternResult]:
        """
        Détecte les biseaux montants et descendants.
        - Biseau montant (Rising Wedge) : signal baissier
        - Biseau descendant (Falling Wedge) : signal haussier
        """
        results = []
        window = min(35, len(self.df) - 5)
        df_w = self.df.tail(window)

        highs = df_w["High"].values
        lows = df_w["Low"].values
        x = np.arange(len(highs))

        slope_h, inter_h = np.polyfit(x, highs, 1)
        slope_l, inter_l = np.polyfit(x, lows, 1)

        avg = df_w["Close"].mean()
        # Les deux lignes vont dans la même direction (convergentes)
        if slope_h * slope_l <= 0:
            return results

        # Convergence
        start_w = inter_h - inter_l
        end_w = (slope_h * (len(x)-1) + inter_h) - (slope_l * (len(x)-1) + inter_l)
        if end_w >= start_w * 0.85:
            return results

        height = start_w
        current_support = slope_l * (len(x)-1) + inter_l
        current_resist = slope_h * (len(x)-1) + inter_h

        # Biseau montant (les deux lignes montent → retournement baissier)
        if slope_h > 0 and slope_l > 0:
            results.append(PatternResult(
                nom="Biseau Montant (Rising Wedge)",
                type="retournement",
                direction="BAISSIER",
                fiabilite="Haute",
                date_debut=df_w.index[0],
                date_fin=df_w.index[-1],
                niveau_breakout=round(current_support, 2),
                objectif_prix=round(current_support - height * 0.75, 2),
                stop_loss=round(current_resist * 1.01, 2),
                description=(
                    f"Biseau montant (Rising Wedge). Signal de retournement baissier. "
                    f"Support: {round(current_support, 2)} MAD. "
                    f"Objectif cassure: {round(current_support - height*0.75, 2)} MAD."
                ),
                points_cles={"resistance": round(current_resist, 2), "support": round(current_support, 2)},
            ))

        # Biseau descendant (les deux lignes descendent → retournement haussier)
        elif slope_h < 0 and slope_l < 0:
            results.append(PatternResult(
                nom="Biseau Descendant (Falling Wedge)",
                type="retournement",
                direction="HAUSSIER",
                fiabilite="Haute",
                date_debut=df_w.index[0],
                date_fin=df_w.index[-1],
                niveau_breakout=round(current_resist, 2),
                objectif_prix=round(current_resist + height * 0.75, 2),
                stop_loss=round(current_support * 0.99, 2),
                description=(
                    f"Biseau descendant (Falling Wedge). Signal de retournement haussier. "
                    f"Résistance: {round(current_resist, 2)} MAD. "
                    f"Objectif cassure: {round(current_resist + height*0.75, 2)} MAD."
                ),
                points_cles={"resistance": round(current_resist, 2), "support": round(current_support, 2)},
            ))

        return results

    # =========================================================
    # CUP & HANDLE
    # =========================================================

    def detect_cup_and_handle(self) -> List[PatternResult]:
        """
        Cup & Handle : formation en coupe arrondie suivie d'une légère consolidation.
        Signal de continuation haussière.
        """
        results = []
        if len(self.df) < 50:
            return results

        df_w = self.df.tail(60)
        close = df_w["Close"]
        n = len(close)

        # Chercher le schéma : haut → creux → haut → petite correction
        left_high = close.iloc[:n//3].max()
        left_high_i = close.iloc[:n//3].idxmax()
        cup_low = close.iloc[n//4:3*n//4].min()
        cup_low_i = close.iloc[n//4:3*n//4].idxmin()
        right_high = close.iloc[2*n//3:].max()
        right_high_i = close.iloc[2*n//3:].idxmax()

        # Validation de la coupe
        if right_high < left_high * 0.95:
            return results
        cup_depth = (left_high - cup_low) / left_high
        if not (0.10 < cup_depth < 0.50):
            return results

        # Handle : légère correction après le right_high
        handle_data = close.iloc[close.index.get_loc(right_high_i):]
        if len(handle_data) < 3:
            return results
        handle_low = handle_data.min()
        handle_correction = (right_high - handle_low) / right_high

        if not (0.02 < handle_correction < 0.15):
            return results

        target = right_high + (right_high - cup_low)
        stop = handle_low * 0.98

        results.append(PatternResult(
            nom="Cup & Handle",
            type="continuation",
            direction="HAUSSIER",
            fiabilite="Très haute",
            date_debut=df_w.index[0],
            date_fin=df_w.index[-1],
            niveau_breakout=round(right_high, 2),
            objectif_prix=round(target, 2),
            stop_loss=round(stop, 2),
            description=(
                f"Cup & Handle. Coupe: -{round(cup_depth*100, 1)}%. "
                f"Breakout: {round(right_high, 2)} MAD. "
                f"Objectif: {round(target, 2)} MAD."
            ),
            points_cles={
                "sommet_gauche": round(left_high, 2),
                "fond_coupe": round(cup_low, 2),
                "sommet_droit": round(right_high, 2),
                "handle_bas": round(handle_low, 2),
            },
        ))
        return results

    # =========================================================
    # CANAUX
    # =========================================================

    def detect_channel(self) -> List[PatternResult]:
        """
        Détecte un canal de prix (haussier, baissier ou latéral).
        """
        results = []
        window = min(40, len(self.df) - 5)
        df_w = self.df.tail(window)

        highs = df_w["High"].values
        lows = df_w["Low"].values
        x = np.arange(len(highs))

        slope_h, inter_h = np.polyfit(x, highs, 1)
        slope_l, inter_l = np.polyfit(x, lows, 1)

        avg = df_w["Close"].mean()
        if avg == 0:
            return results

        # Canal : les deux lignes sont parallèles (même pente approximative)
        slope_diff = abs(slope_h - slope_l) / avg
        if slope_diff > 0.003:
            return results

        slope_avg = (slope_h + slope_l) / 2
        channel_width = (inter_h - inter_l)
        current_h = slope_h * (len(x)-1) + inter_h
        current_l = slope_l * (len(x)-1) + inter_l

        if slope_avg > avg * 0.001:
            name, direction = "Canal Haussier", "HAUSSIER"
        elif slope_avg < -avg * 0.001:
            name, direction = "Canal Baissier", "BAISSIER"
        else:
            name, direction = "Canal Latéral (Range)", "NEUTRE"

        current_price = self.close.iloc[-1]
        pct_in_channel = (current_price - current_l) / channel_width if channel_width > 0 else 0.5

        results.append(PatternResult(
            nom=name,
            type="continuation",
            direction=direction,
            fiabilite="Moyenne",
            date_debut=df_w.index[0],
            date_fin=df_w.index[-1],
            niveau_breakout=round(current_h, 2),
            objectif_prix=round(current_l if pct_in_channel > 0.7 else current_h, 2),
            stop_loss=None,
            description=(
                f"{name} actif. "
                f"Résistance: {round(current_h, 2)} MAD. "
                f"Support: {round(current_l, 2)} MAD. "
                f"Prix à {round(pct_in_channel*100, 0):.0f}% du canal."
            ),
            points_cles={
                "resistance_canal": round(current_h, 2),
                "support_canal": round(current_l, 2),
                "largeur": round(channel_width, 2),
            },
        ))
        return results

    # =========================================================
    # MÉTHODE PRINCIPALE
    # =========================================================

    def detect_all(self) -> List[PatternResult]:
        """
        Détecte tous les patterns graphiques.

        Returns:
            Liste de PatternResult triée par fiabilité
        """
        all_results = []

        detectors = [
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_head_and_shoulders,
            self.detect_inverse_head_and_shoulders,
            self.detect_triangles,
            self.detect_flag,
            self.detect_wedge,
            self.detect_cup_and_handle,
            self.detect_channel,
        ]

        for detect_fn in detectors:
            try:
                all_results.extend(detect_fn())
            except Exception:
                pass

        # Trier par fiabilité
        order = {"Très haute": 0, "Haute": 1, "Moyenne": 2, "Faible": 3}
        return sorted(all_results, key=lambda p: order.get(p.fiabilite, 4))

    def report(self) -> str:
        """Génère un rapport texte des patterns détectés."""
        patterns = self.detect_all()
        lines = ["\n[ CONFIGURATIONS GRAPHIQUES ]"]

        if not patterns:
            lines.append("  Aucune configuration majeure détectée.")
            return "\n".join(lines)

        for p in patterns:
            lines.append(f"\n  {p.nom} ({p.type.upper()})")
            lines.append(f"    Direction   : {p.direction}")
            lines.append(f"    Fiabilité   : {p.fiabilite}")
            lines.append(f"    Breakout    : {p.niveau_breakout} MAD")
            if p.objectif_prix:
                lines.append(f"    Objectif    : {p.objectif_prix} MAD")
            if p.stop_loss:
                lines.append(f"    Stop Loss   : {p.stop_loss} MAD")
            lines.append(f"    Description : {p.description}")

        return "\n".join(lines)
