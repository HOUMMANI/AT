"""
Indicateurs de momentum : RSI, Stochastique, CCI, Williams %R, etc.
"""

import pandas as pd
import numpy as np


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcule le RSI (Relative Strength Index).

    Args:
        series: Série de prix de clôture
        window: Période de calcul (défaut: 14)

    Returns:
        Série RSI entre 0 et 100
        - RSI > 70 : Surachat (signal de vente potentiel)
        - RSI < 30 : Survente (signal d'achat potentiel)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values.name = f"RSI_{window}"
    return rsi_values


def stochastic(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Calcule l'oscillateur Stochastique (%K et %D).

    Args:
        df: DataFrame avec colonnes High, Low, Close
        k_window: Période %K (défaut: 14)
        d_window: Période %D (défaut: 3)
        smooth_k: Lissage de %K (défaut: 3)

    Returns:
        DataFrame avec colonnes: %K, %D
        - > 80 : Surachat
        - < 20 : Survente
    """
    lowest_low = df["Low"].rolling(k_window).min()
    highest_high = df["High"].rolling(k_window).max()

    k_raw = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(smooth_k).mean()
    d = k.rolling(d_window).mean()

    return pd.DataFrame({"%K": k, "%D": d}, index=df.index)


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calcule le CCI (Commodity Channel Index).

    Args:
        df: DataFrame avec colonnes High, Low, Close
        window: Période de calcul (défaut: 20)

    Returns:
        Série CCI
        - > +100 : Surachat
        - < -100 : Survente
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_tp = typical_price.rolling(window).mean()
    mean_deviation = typical_price.rolling(window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci_values = (typical_price - mean_tp) / (0.015 * mean_deviation)
    cci_values.name = f"CCI_{window}"
    return cci_values


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcule le Williams %R.

    Args:
        df: DataFrame avec colonnes High, Low, Close
        window: Période de calcul (défaut: 14)

    Returns:
        Série Williams %R entre -100 et 0
        - > -20 : Surachat
        - < -80 : Survente
    """
    highest_high = df["High"].rolling(window).max()
    lowest_low = df["Low"].rolling(window).min()
    wr = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)
    wr.name = f"Williams_%R_{window}"
    return wr


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Rate of Change (ROC) - Taux de variation.

    Args:
        series: Série de prix
        window: Période (défaut: 12)

    Returns:
        Série ROC en pourcentage
    """
    roc_values = ((series - series.shift(window)) / series.shift(window)) * 100
    roc_values.name = f"ROC_{window}"
    return roc_values


def momentum_indicator(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Indicateur de Momentum simple.

    Args:
        series: Série de prix
        window: Période (défaut: 10)

    Returns:
        Série momentum = Close(t) - Close(t-n)
    """
    mom = series - series.shift(window)
    mom.name = f"Momentum_{window}"
    return mom


def tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """
    True Strength Index (TSI).

    Args:
        series: Série de prix
        r: Période EMA rapide (défaut: 25)
        s: Période EMA lente (défaut: 13)

    Returns:
        Série TSI entre -100 et +100
    """
    delta = series.diff()
    abs_delta = delta.abs()

    ema1 = delta.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()

    abs_ema1 = abs_delta.ewm(span=r, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=s, adjust=False).mean()

    tsi_values = 100 * (ema2 / abs_ema2)
    tsi_values.name = "TSI"
    return tsi_values
