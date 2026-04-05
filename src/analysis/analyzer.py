"""
Moteur d'analyse technique pour les actions BVC.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from ..indicators.trend import sma, ema, macd
from ..indicators.momentum import rsi, stochastic, cci, williams_r
from ..indicators.volatility import bollinger_bands, atr, historical_volatility
from ..indicators.volume import obv, volume_sma, mfi, cmf, relative_volume
from ..patterns.candlesticks import CandlestickPatterns
from ..patterns.chart_patterns import ChartPatternDetector
from ..patterns.fibonacci import FibonacciAnalyzer
from ..patterns.trendlines import TrendlineDetector

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Analyse technique complète d'une action BVC.

    Exemple d'utilisation:
        from src.data import BVCDataFetcher
        from src.analysis import TechnicalAnalyzer

        fetcher = BVCDataFetcher()
        df = fetcher.get_ohlcv("ATW", period="1y")

        analyzer = TechnicalAnalyzer(df)
        report = analyzer.full_report()
        print(report)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame OHLCV (Open, High, Low, Close, Volume)
        """
        self._validate_df(df)
        self.df = df.copy()
        self.symbol = df.attrs.get("symbol", "N/A")
        self.name = df.attrs.get("name", "N/A")
        self._indicators = None

    def _validate_df(self, df: pd.DataFrame):
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing}")
        if len(df) < 30:
            raise ValueError("Données insuffisantes (minimum 30 bougies requis)")

    def compute_all(self) -> pd.DataFrame:
        """
        Calcule tous les indicateurs techniques et retourne un DataFrame enrichi.
        """
        df = self.df.copy()
        close = df["Close"]

        # --- Moyennes mobiles ---
        df["SMA_20"] = sma(close, 20)
        df["SMA_50"] = sma(close, 50)
        df["SMA_100"] = sma(close, 100)
        df["SMA_200"] = sma(close, 200)
        df["EMA_9"] = ema(close, 9)
        df["EMA_20"] = ema(close, 20)
        df["EMA_50"] = ema(close, 50)

        # --- MACD ---
        macd_df = macd(close)
        df["MACD"] = macd_df["MACD"]
        df["MACD_Signal"] = macd_df["Signal"]
        df["MACD_Hist"] = macd_df["Histogramme"]

        # --- RSI ---
        df["RSI_14"] = rsi(close, 14)
        df["RSI_7"] = rsi(close, 7)

        # --- Stochastique ---
        stoch_df = stochastic(df)
        df["Stoch_%K"] = stoch_df["%K"]
        df["Stoch_%D"] = stoch_df["%D"]

        # --- CCI ---
        df["CCI_20"] = cci(df, 20)

        # --- Williams %R ---
        df["Williams_%R"] = williams_r(df, 14)

        # --- Bollinger Bands ---
        bb_df = bollinger_bands(close, 20)
        df["BB_Milieu"] = bb_df["BB_Milieu"]
        df["BB_Haute"] = bb_df["BB_Haute"]
        df["BB_Basse"] = bb_df["BB_Basse"]
        df["BB_Largeur"] = bb_df["BB_Largeur"]
        df["BB_%B"] = bb_df["BB_%B"]

        # --- ATR ---
        df["ATR_14"] = atr(df, 14)

        # --- Volatilité historique ---
        df["HV_20"] = historical_volatility(close, 20)

        # --- Volume ---
        df["Volume_SMA_20"] = volume_sma(df, 20)
        df["OBV"] = obv(df)
        df["MFI_14"] = mfi(df, 14)
        df["CMF_20"] = cmf(df, 20)
        df["RVOL"] = relative_volume(df, 20)

        self._indicators = df
        return df

    def get_signals(self) -> dict:
        """
        Génère des signaux d'achat/vente basés sur les indicateurs.

        Returns:
            Dictionnaire avec les signaux pour chaque indicateur
        """
        if self._indicators is None:
            self.compute_all()

        df = self._indicators
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        signals = {}

        # --- RSI ---
        rsi_val = last.get("RSI_14", np.nan)
        if not np.isnan(rsi_val):
            if rsi_val < 30:
                signals["RSI"] = {"valeur": round(rsi_val, 2), "signal": "ACHAT", "note": "Zone de survente (<30)"}
            elif rsi_val > 70:
                signals["RSI"] = {"valeur": round(rsi_val, 2), "signal": "VENTE", "note": "Zone de surachat (>70)"}
            else:
                signals["RSI"] = {"valeur": round(rsi_val, 2), "signal": "NEUTRE", "note": "Zone neutre (30-70)"}

        # --- MACD ---
        macd_val = last.get("MACD", np.nan)
        macd_signal = last.get("MACD_Signal", np.nan)
        macd_hist = last.get("MACD_Hist", np.nan)
        prev_hist = prev.get("MACD_Hist", np.nan)
        if not any(np.isnan(v) for v in [macd_val, macd_signal, macd_hist]):
            if macd_hist > 0 and prev_hist <= 0:
                signals["MACD"] = {"valeur": round(macd_val, 4), "signal": "ACHAT", "note": "Croisement haussier"}
            elif macd_hist < 0 and prev_hist >= 0:
                signals["MACD"] = {"valeur": round(macd_val, 4), "signal": "VENTE", "note": "Croisement baissier"}
            elif macd_hist > 0:
                signals["MACD"] = {"valeur": round(macd_val, 4), "signal": "HAUSSIER", "note": "Histogramme positif"}
            else:
                signals["MACD"] = {"valeur": round(macd_val, 4), "signal": "BAISSIER", "note": "Histogramme négatif"}

        # --- Bollinger Bands ---
        close_val = last["Close"]
        bb_haute = last.get("BB_Haute", np.nan)
        bb_basse = last.get("BB_Basse", np.nan)
        bb_milieu = last.get("BB_Milieu", np.nan)
        if not any(np.isnan(v) for v in [bb_haute, bb_basse, bb_milieu]):
            pct_b = (close_val - bb_basse) / (bb_haute - bb_basse)
            if pct_b > 1:
                signals["Bollinger"] = {"valeur": round(pct_b, 2), "signal": "VENTE", "note": "Prix au-dessus de la bande haute"}
            elif pct_b < 0:
                signals["Bollinger"] = {"valeur": round(pct_b, 2), "signal": "ACHAT", "note": "Prix en-dessous de la bande basse"}
            else:
                signals["Bollinger"] = {"valeur": round(pct_b, 2), "signal": "NEUTRE", "note": f"%B={round(pct_b*100, 1)}%"}

        # --- Moyennes mobiles (Golden/Death Cross) ---
        sma50 = last.get("SMA_50", np.nan)
        sma200 = last.get("SMA_200", np.nan)
        prev_sma50 = prev.get("SMA_50", np.nan)
        prev_sma200 = prev.get("SMA_200", np.nan)
        if not any(np.isnan(v) for v in [sma50, sma200, prev_sma50, prev_sma200]):
            if sma50 > sma200 and prev_sma50 <= prev_sma200:
                signals["MA_Cross"] = {"signal": "ACHAT", "note": "Golden Cross: SMA50 croise SMA200 à la hausse"}
            elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                signals["MA_Cross"] = {"signal": "VENTE", "note": "Death Cross: SMA50 croise SMA200 à la baisse"}
            elif sma50 > sma200:
                signals["MA_Cross"] = {"signal": "HAUSSIER", "note": f"SMA50 ({round(sma50, 2)}) > SMA200 ({round(sma200, 2)})"}
            else:
                signals["MA_Cross"] = {"signal": "BAISSIER", "note": f"SMA50 ({round(sma50, 2)}) < SMA200 ({round(sma200, 2)})"}

        # --- Stochastique ---
        stoch_k = last.get("Stoch_%K", np.nan)
        stoch_d = last.get("Stoch_%D", np.nan)
        if not any(np.isnan(v) for v in [stoch_k, stoch_d]):
            if stoch_k < 20:
                signals["Stochastique"] = {"valeur": round(stoch_k, 2), "signal": "ACHAT", "note": "Zone de survente (<20)"}
            elif stoch_k > 80:
                signals["Stochastique"] = {"valeur": round(stoch_k, 2), "signal": "VENTE", "note": "Zone de surachat (>80)"}
            else:
                signals["Stochastique"] = {"valeur": round(stoch_k, 2), "signal": "NEUTRE", "note": f"%K={round(stoch_k,1)}, %D={round(stoch_d,1)}"}

        # --- Volume ---
        rvol = last.get("RVOL", np.nan)
        if not np.isnan(rvol):
            if rvol > 2:
                signals["Volume"] = {"valeur": round(rvol, 2), "signal": "ALERTE", "note": f"Volume anormalement élevé (RVOL={round(rvol,1)}x)"}
            else:
                signals["Volume"] = {"valeur": round(rvol, 2), "signal": "NORMAL", "note": f"RVOL={round(rvol,2)}x la moyenne"}

        return signals

    def score(self) -> dict:
        """
        Calcule un score d'analyse technique global (-100 à +100).

        Returns:
            Dictionnaire avec score total et détail par catégorie
        """
        signals = self.get_signals()
        scores = {}
        weights = {
            "RSI": 20,
            "MACD": 25,
            "Bollinger": 15,
            "MA_Cross": 25,
            "Stochastique": 15,
        }

        signal_map = {
            "ACHAT": 1.0,
            "HAUSSIER": 0.5,
            "NEUTRE": 0.0,
            "BAISSIER": -0.5,
            "VENTE": -1.0,
            "ALERTE": 0.0,
            "NORMAL": 0.0,
        }

        total_weight = 0
        weighted_score = 0

        for indicator, weight in weights.items():
            if indicator in signals:
                sig = signals[indicator].get("signal", "NEUTRE")
                val = signal_map.get(sig, 0.0)
                scores[indicator] = {
                    "signal": sig,
                    "poids": weight,
                    "contribution": round(val * weight, 1),
                }
                weighted_score += val * weight
                total_weight += weight

        if total_weight > 0:
            final_score = (weighted_score / total_weight) * 100
        else:
            final_score = 0

        if final_score >= 50:
            recommendation = "FORT SIGNAL ACHAT"
        elif final_score >= 20:
            recommendation = "SIGNAL ACHAT"
        elif final_score >= -20:
            recommendation = "NEUTRE"
        elif final_score >= -50:
            recommendation = "SIGNAL VENTE"
        else:
            recommendation = "FORT SIGNAL VENTE"

        return {
            "score": round(final_score, 1),
            "recommandation": recommendation,
            "detail": scores,
        }

    def support_resistance(self, lookback: int = 50) -> dict:
        """
        Identifie les niveaux de support et résistance majeurs.

        Args:
            lookback: Nombre de périodes à analyser (défaut: 50)

        Returns:
            Dictionnaire avec supports et résistances
        """
        df = self.df.tail(lookback)
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Pivots locaux
        resistance_levels = []
        support_levels = []

        for i in range(2, len(df) - 2):
            # Résistance (high local)
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
               high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                resistance_levels.append(high.iloc[i])

            # Support (low local)
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
               low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                support_levels.append(low.iloc[i])

        current_price = close.iloc[-1]

        # Filtrer : supports < prix actuel, résistances > prix actuel
        supports = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
        resistances = sorted([r for r in resistance_levels if r > current_price])[:3]

        return {
            "prix_actuel": round(current_price, 2),
            "supports": [round(s, 2) for s in supports],
            "resistances": [round(r, 2) for r in resistances],
        }

    def summary(self) -> dict:
        """Retourne un résumé complet des statistiques de prix."""
        df = self.df
        close = df["Close"]

        return {
            "symbole": self.symbol,
            "nom": self.name,
            "dernier_cours": round(close.iloc[-1], 2),
            "ouverture": round(df["Open"].iloc[-1], 2),
            "plus_haut_jour": round(df["High"].iloc[-1], 2),
            "plus_bas_jour": round(df["Low"].iloc[-1], 2),
            "variation_1j": round(((close.iloc[-1] / close.iloc[-2]) - 1) * 100, 2) if len(close) > 1 else 0,
            "variation_1m": round(((close.iloc[-1] / close.iloc[-22]) - 1) * 100, 2) if len(close) > 22 else None,
            "variation_3m": round(((close.iloc[-1] / close.iloc[-63]) - 1) * 100, 2) if len(close) > 63 else None,
            "variation_1y": round(((close.iloc[-1] / close.iloc[-252]) - 1) * 100, 2) if len(close) > 252 else None,
            "plus_haut_52s": round(close.tail(252).max(), 2),
            "plus_bas_52s": round(close.tail(252).min(), 2),
            "volume_moyen_20j": int(df["Volume"].tail(20).mean()),
            "nb_periodes": len(df),
        }

    # =========================================================
    # PATTERNS
    # =========================================================

    def detect_candlestick_patterns(self, lookback: int = 15) -> list:
        """Détecte les patterns de bougies japonaises récents."""
        cp = CandlestickPatterns(self.df)
        return cp.get_recent(lookback)

    def detect_chart_patterns(self) -> list:
        """Détecte les configurations graphiques (H&S, triangles, etc.)."""
        try:
            detector = ChartPatternDetector(self.df)
            return detector.detect_all()
        except Exception as e:
            logger.warning(f"Détection chart patterns impossible: {e}")
            return []

    def fibonacci_analysis(self) -> object:
        """Retourne l'analyse Fibonacci complète."""
        fib = FibonacciAnalyzer(self.df)
        return fib.analyze()

    def trendline_analysis(self) -> list:
        """Détecte les lignes de tendance actives."""
        try:
            td = TrendlineDetector(self.df)
            return td.detect_all()
        except Exception as e:
            logger.warning(f"Détection trendlines impossible: {e}")
            return []

    def full_report(self, include_patterns: bool = True) -> str:
        """
        Génère un rapport d'analyse technique complet en texte.

        Args:
            include_patterns: Inclure la détection de patterns (défaut: True)
        """
        self.compute_all()
        summ = self.summary()
        signals = self.get_signals()
        score_data = self.score()
        sr = self.support_resistance()

        lines = [
            "=" * 65,
            f"  ANALYSE TECHNIQUE - {summ['symbole']} | {summ['nom']}",
            "=" * 65,
            "",
            "[ COURS ]",
            f"  Dernier cours    : {summ['dernier_cours']} MAD",
            f"  Ouverture        : {summ['ouverture']} MAD",
            f"  Plus haut / bas  : {summ['plus_haut_jour']} / {summ['plus_bas_jour']} MAD",
            f"  Variation 1j     : {summ['variation_1j']:+.2f}%",
        ]

        if summ["variation_1m"]:
            lines.append(f"  Variation 1 mois : {summ['variation_1m']:+.2f}%")
        if summ["variation_3m"]:
            lines.append(f"  Variation 3 mois : {summ['variation_3m']:+.2f}%")
        if summ["variation_1y"]:
            lines.append(f"  Variation 1 an   : {summ['variation_1y']:+.2f}%")

        lines += [
            f"  52 semaines H/B  : {summ['plus_haut_52s']} / {summ['plus_bas_52s']} MAD",
            f"  Volume moyen 20j : {summ['volume_moyen_20j']:,}",
            "",
            "[ SUPPORTS & RESISTANCES ]",
            f"  Prix actuel      : {sr['prix_actuel']} MAD",
        ]

        for i, r in enumerate(sr["resistances"], 1):
            lines.append(f"  Résistance R{i}    : {r} MAD")
        for i, s in enumerate(sr["supports"], 1):
            lines.append(f"  Support S{i}       : {s} MAD")

        lines += ["", "[ SIGNAUX TECHNIQUES ]"]
        for name, sig in signals.items():
            signal_str = sig.get("signal", "?")
            note = sig.get("note", "")
            lines.append(f"  {name:<18}: {signal_str:<12} | {note}")

        if include_patterns:
            # --- Patterns bougies ---
            try:
                cp = CandlestickPatterns(self.df)
                lines.append(cp.report(lookback=15))
            except Exception:
                pass

            # --- Lignes de tendance ---
            try:
                td = TrendlineDetector(self.df)
                lines.append(td.report())
            except Exception:
                pass

            # --- Fibonacci ---
            try:
                fib = FibonacciAnalyzer(self.df)
                lines.append(fib.report())
            except Exception:
                pass

            # --- Configurations graphiques ---
            try:
                detector = ChartPatternDetector(self.df)
                lines.append(detector.report())
            except Exception:
                pass

        lines += [
            "",
            "[ SCORE GLOBAL ]",
            f"  Score            : {score_data['score']:+.1f} / 100",
            f"  Recommandation   : {score_data['recommandation']}",
            "",
            "=" * 65,
        ]

        return "\n".join(lines)
