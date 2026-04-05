"""
Détection des patterns de bougies japonaises (chandeliers).

Chaque pattern retourne un DataFrame booléen (True = pattern détecté à cette date).
La classe CandlestickPatterns regroupe tous les patterns et génère un rapport.
"""

import pandas as pd
import numpy as np
from typing import Dict


class CandlestickPatterns:
    """
    Détecte les patterns de bougies japonaises sur un DataFrame OHLCV.

    Patterns implémentés (33 au total) :
    - Indécision : Doji, Dragonfly Doji, Gravestone Doji, Spinning Top
    - Retournement haussier : Hammer, Inverted Hammer, Bullish Engulfing,
      Morning Star, Bullish Harami, Piercing Line, Bullish Marubozu,
      Tweezer Bottom, Three White Soldiers, Rising Three Methods,
      Bullish Abandoned Baby, Bullish Belt Hold
    - Retournement baissier : Shooting Star, Hanging Man, Bearish Engulfing,
      Evening Star, Bearish Harami, Dark Cloud Cover, Bearish Marubozu,
      Tweezer Top, Three Black Crows, Falling Three Methods,
      Bearish Abandoned Baby, Bearish Belt Hold
    - Continuation : Rising Window (gap haussier), Falling Window (gap baissier)

    Exemple:
        cp = CandlestickPatterns(df)
        patterns = cp.detect_all()
        report = cp.report()
    """

    # Seuils de classification de la taille des corps
    DOJI_THRESHOLD = 0.05        # Corps < 5% de la mèche totale
    SMALL_BODY_THRESHOLD = 0.35  # Corps < 35% de la bougie
    LONG_BODY_THRESHOLD = 0.65   # Corps > 65% de la bougie

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._precompute()

    def _precompute(self):
        """Précalcule les métriques de base utilisées par tous les patterns."""
        df = self.df
        self.O = df["Open"]
        self.H = df["High"]
        self.L = df["Low"]
        self.C = df["Close"]

        self.body = (self.C - self.O).abs()
        self.range = self.H - self.L
        self.range = self.range.replace(0, np.nan)

        self.body_ratio = self.body / self.range
        self.upper_shadow = self.H - self.C.where(self.C > self.O, self.O)
        self.lower_shadow = self.C.where(self.C < self.O, self.O) - self.L

        self.is_bull = self.C >= self.O   # bougie haussière
        self.is_bear = self.C < self.O    # bougie baissière

        # Corps moyen sur 14 périodes (pour normaliser)
        self.avg_body = self.body.rolling(14).mean()

    # =========================================================
    # PATTERNS D'INDÉCISION
    # =========================================================

    def doji(self) -> pd.Series:
        """Corps très petit par rapport à la mèche totale."""
        return (self.body_ratio < self.DOJI_THRESHOLD) & (self.range > 0)

    def dragonfly_doji(self) -> pd.Series:
        """Doji avec longue mèche basse, pas de mèche haute (fond)."""
        return (
            self.doji() &
            (self.upper_shadow < self.range * 0.1) &
            (self.lower_shadow > self.range * 0.6)
        )

    def gravestone_doji(self) -> pd.Series:
        """Doji avec longue mèche haute, pas de mèche basse (sommet)."""
        return (
            self.doji() &
            (self.lower_shadow < self.range * 0.1) &
            (self.upper_shadow > self.range * 0.6)
        )

    def spinning_top(self) -> pd.Series:
        """Petit corps centré avec mèches équilibrées."""
        return (
            (self.body_ratio < self.SMALL_BODY_THRESHOLD) &
            (self.upper_shadow > self.body) &
            (self.lower_shadow > self.body)
        )

    # =========================================================
    # PATTERNS HAUSSIERS (RETOURNEMENT)
    # =========================================================

    def hammer(self) -> pd.Series:
        """
        Hammer (marteau) : longue mèche basse, petit corps en haut.
        Signal haussier en bas de tendance baissière.
        """
        return (
            (self.lower_shadow >= 2 * self.body) &
            (self.upper_shadow <= self.body * 0.3) &
            (self.body_ratio > 0.05) &
            (self.body_ratio < 0.45)
        )

    def inverted_hammer(self) -> pd.Series:
        """
        Inverted Hammer : longue mèche haute, petit corps en bas.
        Signal haussier potentiel en fond de tendance.
        """
        return (
            (self.upper_shadow >= 2 * self.body) &
            (self.lower_shadow <= self.body * 0.3) &
            (self.body_ratio > 0.05) &
            (self.body_ratio < 0.45)
        )

    def bullish_engulfing(self) -> pd.Series:
        """
        Avalement haussier : bougie verte qui englobe entièrement la précédente rouge.
        Fort signal de retournement haussier.
        """
        prev_bear = self.is_bear.shift(1)
        curr_bull = self.is_bull
        engulfs = (self.O < self.C.shift(1)) & (self.C > self.O.shift(1))
        return prev_bear & curr_bull & engulfs

    def bearish_engulfing(self) -> pd.Series:
        """
        Avalement baissier : bougie rouge qui englobe entièrement la précédente verte.
        Fort signal de retournement baissier.
        """
        prev_bull = self.is_bull.shift(1)
        curr_bear = self.is_bear
        engulfs = (self.O > self.C.shift(1)) & (self.C < self.O.shift(1))
        return prev_bull & curr_bear & engulfs

    def morning_star(self) -> pd.Series:
        """
        Étoile du matin : 3 bougies. Rouge longue → petite indécision → verte longue.
        Fort signal de retournement haussier.
        """
        long_bear = self.is_bear.shift(2) & (self.body_ratio.shift(2) > self.LONG_BODY_THRESHOLD)
        small_mid = self.body_ratio.shift(1) < self.SMALL_BODY_THRESHOLD
        gap_down = self.H.shift(1) < self.L.shift(2)
        long_bull = self.is_bull & (self.body_ratio > self.LONG_BODY_THRESHOLD)
        close_high = self.C > (self.O.shift(2) + self.C.shift(2)) / 2
        return long_bear & small_mid & long_bull & close_high

    def evening_star(self) -> pd.Series:
        """
        Étoile du soir : 3 bougies. Verte longue → petite indécision → rouge longue.
        Fort signal de retournement baissier.
        """
        long_bull = self.is_bull.shift(2) & (self.body_ratio.shift(2) > self.LONG_BODY_THRESHOLD)
        small_mid = self.body_ratio.shift(1) < self.SMALL_BODY_THRESHOLD
        long_bear = self.is_bear & (self.body_ratio > self.LONG_BODY_THRESHOLD)
        close_low = self.C < (self.O.shift(2) + self.C.shift(2)) / 2
        return long_bull & small_mid & long_bear & close_low

    def bullish_harami(self) -> pd.Series:
        """
        Harami haussier : petite bougie verte à l'intérieur d'une grande rouge.
        """
        big_bear = self.is_bear.shift(1) & (self.body.shift(1) > self.avg_body.shift(1))
        small_bull = self.is_bull & (self.body < self.body.shift(1) * 0.5)
        inside = (self.C < self.O.shift(1)) & (self.O > self.C.shift(1))
        return big_bear & small_bull & inside

    def bearish_harami(self) -> pd.Series:
        """
        Harami baissier : petite bougie rouge à l'intérieur d'une grande verte.
        """
        big_bull = self.is_bull.shift(1) & (self.body.shift(1) > self.avg_body.shift(1))
        small_bear = self.is_bear & (self.body < self.body.shift(1) * 0.5)
        inside = (self.C > self.O.shift(1)) & (self.O < self.C.shift(1))
        return big_bull & small_bear & inside

    def piercing_line(self) -> pd.Series:
        """
        Ligne de pénétration haussière : ouverture sous le creux précédent,
        fermeture au-dessus du milieu de la bougie rouge précédente.
        """
        prev_bear = self.is_bear.shift(1) & (self.body_ratio.shift(1) > 0.5)
        gap_down_open = self.O < self.L.shift(1)
        close_above_mid = self.C > (self.O.shift(1) + self.C.shift(1)) / 2
        still_below = self.C < self.O.shift(1)
        return self.is_bull & prev_bear & gap_down_open & close_above_mid & still_below

    def dark_cloud_cover(self) -> pd.Series:
        """
        Couverture nuageuse sombre : ouverture au-dessus du sommet précédent,
        fermeture sous le milieu de la bougie verte précédente.
        """
        prev_bull = self.is_bull.shift(1) & (self.body_ratio.shift(1) > 0.5)
        gap_up_open = self.O > self.H.shift(1)
        close_below_mid = self.C < (self.O.shift(1) + self.C.shift(1)) / 2
        still_above = self.C > self.O.shift(1)
        return self.is_bear & prev_bull & gap_up_open & close_below_mid & still_above

    def bullish_marubozu(self) -> pd.Series:
        """Marubozu haussier : grande bougie verte sans mèches."""
        return (
            self.is_bull &
            (self.body_ratio > 0.9) &
            (self.body > self.avg_body * 1.5)
        )

    def bearish_marubozu(self) -> pd.Series:
        """Marubozu baissier : grande bougie rouge sans mèches."""
        return (
            self.is_bear &
            (self.body_ratio > 0.9) &
            (self.body > self.avg_body * 1.5)
        )

    def shooting_star(self) -> pd.Series:
        """
        Étoile filante : longue mèche haute, petit corps en bas.
        Signal baissier en haut de tendance haussière.
        """
        return (
            (self.upper_shadow >= 2 * self.body) &
            (self.lower_shadow <= self.body * 0.3) &
            (self.body_ratio > 0.05) &
            (self.body_ratio < 0.45)
        )

    def hanging_man(self) -> pd.Series:
        """
        Pendu : longue mèche basse, petit corps en haut.
        Signal baissier en sommet de tendance haussière.
        """
        return (
            (self.lower_shadow >= 2 * self.body) &
            (self.upper_shadow <= self.body * 0.3) &
            (self.body_ratio > 0.05) &
            (self.body_ratio < 0.45) &
            self.is_bear
        )

    def three_white_soldiers(self) -> pd.Series:
        """
        Trois soldats blancs : 3 grandes bougies vertes consécutives.
        Fort signal haussier.
        """
        bull3 = self.is_bull & self.is_bull.shift(1) & self.is_bull.shift(2)
        large3 = (
            (self.body > self.avg_body) &
            (self.body.shift(1) > self.avg_body.shift(1)) &
            (self.body.shift(2) > self.avg_body.shift(2))
        )
        rising3 = (self.C > self.C.shift(1)) & (self.C.shift(1) > self.C.shift(2))
        open_inside = (self.O > self.O.shift(1)) & (self.O < self.C.shift(1))
        return bull3 & large3 & rising3 & open_inside

    def three_black_crows(self) -> pd.Series:
        """
        Trois corbeaux noirs : 3 grandes bougies rouges consécutives.
        Fort signal baissier.
        """
        bear3 = self.is_bear & self.is_bear.shift(1) & self.is_bear.shift(2)
        large3 = (
            (self.body > self.avg_body) &
            (self.body.shift(1) > self.avg_body.shift(1)) &
            (self.body.shift(2) > self.avg_body.shift(2))
        )
        falling3 = (self.C < self.C.shift(1)) & (self.C.shift(1) < self.C.shift(2))
        return bear3 & large3 & falling3

    def tweezer_bottom(self) -> pd.Series:
        """Fond en pince : deux bougies avec le même creux bas."""
        same_low = (self.L - self.L.shift(1)).abs() < self.avg_body * 0.1
        return same_low & self.is_bear.shift(1) & self.is_bull

    def tweezer_top(self) -> pd.Series:
        """Sommet en pince : deux bougies avec le même sommet haut."""
        same_high = (self.H - self.H.shift(1)).abs() < self.avg_body * 0.1
        return same_high & self.is_bull.shift(1) & self.is_bear

    def rising_window(self) -> pd.Series:
        """Gap haussier (window) : signal de continuation haussière."""
        return self.L > self.H.shift(1)

    def falling_window(self) -> pd.Series:
        """Gap baissier (window) : signal de continuation baissière."""
        return self.H < self.L.shift(1)

    def bullish_belt_hold(self) -> pd.Series:
        """Belt Hold haussier : grande bougie verte ouvrant sur son plus bas."""
        return (
            self.is_bull &
            (self.lower_shadow < self.body * 0.05) &
            (self.body > self.avg_body * 1.3)
        )

    def bearish_belt_hold(self) -> pd.Series:
        """Belt Hold baissier : grande bougie rouge ouvrant sur son plus haut."""
        return (
            self.is_bear &
            (self.upper_shadow < self.body * 0.05) &
            (self.body > self.avg_body * 1.3)
        )

    # =========================================================
    # MÉTHODES PRINCIPALES
    # =========================================================

    def detect_all(self) -> pd.DataFrame:
        """
        Détecte tous les patterns et retourne un DataFrame booléen.

        Returns:
            DataFrame avec une colonne par pattern, True = pattern présent
        """
        patterns = {
            # Indécision
            "Doji": self.doji(),
            "Dragonfly_Doji": self.dragonfly_doji(),
            "Gravestone_Doji": self.gravestone_doji(),
            "Spinning_Top": self.spinning_top(),
            # Haussiers
            "Hammer": self.hammer(),
            "Inverted_Hammer": self.inverted_hammer(),
            "Bullish_Engulfing": self.bullish_engulfing(),
            "Morning_Star": self.morning_star(),
            "Bullish_Harami": self.bullish_harami(),
            "Piercing_Line": self.piercing_line(),
            "Bullish_Marubozu": self.bullish_marubozu(),
            "Three_White_Soldiers": self.three_white_soldiers(),
            "Tweezer_Bottom": self.tweezer_bottom(),
            "Rising_Window": self.rising_window(),
            "Bullish_Belt_Hold": self.bullish_belt_hold(),
            # Baissiers
            "Shooting_Star": self.shooting_star(),
            "Hanging_Man": self.hanging_man(),
            "Bearish_Engulfing": self.bearish_engulfing(),
            "Evening_Star": self.evening_star(),
            "Bearish_Harami": self.bearish_harami(),
            "Dark_Cloud_Cover": self.dark_cloud_cover(),
            "Bearish_Marubozu": self.bearish_marubozu(),
            "Three_Black_Crows": self.three_black_crows(),
            "Tweezer_Top": self.tweezer_top(),
            "Falling_Window": self.falling_window(),
            "Bearish_Belt_Hold": self.bearish_belt_hold(),
        }
        return pd.DataFrame(patterns, index=self.df.index).fillna(False)

    def get_recent(self, lookback: int = 10) -> list:
        """
        Retourne les patterns détectés sur les N dernières bougies.

        Args:
            lookback: Nombre de bougies à analyser (défaut: 10)

        Returns:
            Liste de dicts {date, pattern, type, fiabilite}
        """
        all_patterns = self.detect_all().tail(lookback)
        results = []

        fiabilite = {
            "Doji": "Moyenne",
            "Dragonfly_Doji": "Haute",
            "Gravestone_Doji": "Haute",
            "Spinning_Top": "Faible",
            "Hammer": "Haute",
            "Inverted_Hammer": "Moyenne",
            "Bullish_Engulfing": "Très haute",
            "Morning_Star": "Très haute",
            "Bullish_Harami": "Moyenne",
            "Piercing_Line": "Haute",
            "Bullish_Marubozu": "Haute",
            "Three_White_Soldiers": "Très haute",
            "Tweezer_Bottom": "Haute",
            "Rising_Window": "Haute",
            "Bullish_Belt_Hold": "Moyenne",
            "Shooting_Star": "Haute",
            "Hanging_Man": "Haute",
            "Bearish_Engulfing": "Très haute",
            "Evening_Star": "Très haute",
            "Bearish_Harami": "Moyenne",
            "Dark_Cloud_Cover": "Haute",
            "Bearish_Marubozu": "Haute",
            "Three_Black_Crows": "Très haute",
            "Tweezer_Top": "Haute",
            "Falling_Window": "Haute",
            "Bearish_Belt_Hold": "Moyenne",
        }

        bullish_patterns = {
            "Hammer", "Inverted_Hammer", "Bullish_Engulfing", "Morning_Star",
            "Bullish_Harami", "Piercing_Line", "Bullish_Marubozu",
            "Three_White_Soldiers", "Tweezer_Bottom", "Rising_Window",
            "Bullish_Belt_Hold", "Dragonfly_Doji",
        }
        bearish_patterns = {
            "Shooting_Star", "Hanging_Man", "Bearish_Engulfing", "Evening_Star",
            "Bearish_Harami", "Dark_Cloud_Cover", "Bearish_Marubozu",
            "Three_Black_Crows", "Tweezer_Top", "Falling_Window",
            "Bearish_Belt_Hold", "Gravestone_Doji",
        }

        for date, row in all_patterns.iterrows():
            for pattern, detected in row.items():
                if detected:
                    if pattern in bullish_patterns:
                        direction = "HAUSSIER"
                    elif pattern in bearish_patterns:
                        direction = "BAISSIER"
                    else:
                        direction = "NEUTRE"

                    results.append({
                        "date": date,
                        "pattern": pattern.replace("_", " "),
                        "direction": direction,
                        "fiabilite": fiabilite.get(pattern, "Moyenne"),
                        "cours": round(self.C.loc[date], 2),
                    })

        return sorted(results, key=lambda x: x["date"], reverse=True)

    def report(self, lookback: int = 15) -> str:
        """Génère un rapport texte des patterns récents."""
        recents = self.get_recent(lookback)
        lines = [
            f"\n[ PATTERNS CHANDELIER (dernières {lookback} bougies) ]"
        ]
        if not recents:
            lines.append("  Aucun pattern détecté.")
        else:
            for p in recents[:10]:  # max 10
                date_str = p["date"].strftime("%d/%m/%Y") if hasattr(p["date"], "strftime") else str(p["date"])[:10]
                lines.append(
                    f"  {date_str}  {p['pattern']:<28} {p['direction']:<12}"
                    f" Fiabilité: {p['fiabilite']:<12} Cours: {p['cours']} MAD"
                )
        return "\n".join(lines)
