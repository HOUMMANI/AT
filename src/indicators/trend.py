"""
Indicateurs de tendance : Moyennes Mobiles, MACD, etc.
"""

import pandas as pd
import numpy as np


def moving_average(series: pd.Series, window: int, ma_type: str = "SMA") -> pd.Series:
    """
    Calcule une moyenne mobile.

    Args:
        series: Série de prix (généralement Close)
        window: Nombre de périodes
        ma_type: Type de moyenne ("SMA", "EMA", "WMA")

    Returns:
        Série avec les valeurs de la moyenne mobile
    """
    ma_type = ma_type.upper()
    if ma_type == "SMA":
        return series.rolling(window=window, min_periods=window).mean()
    elif ma_type == "EMA":
        return series.ewm(span=window, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    else:
        raise ValueError(f"Type de MA inconnu: {ma_type}. Utiliser SMA, EMA ou WMA.")


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return moving_average(series, window, "SMA")


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return moving_average(series, window, "EMA")


def wma(series: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average."""
    return moving_average(series, window, "WMA")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Calcule le MACD (Moving Average Convergence Divergence).

    Args:
        series: Série de prix de clôture
        fast: Période EMA rapide (défaut: 12)
        slow: Période EMA lente (défaut: 26)
        signal: Période de la ligne signal (défaut: 9)

    Returns:
        DataFrame avec colonnes: MACD, Signal, Histogramme
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogramme": histogram,
    }, index=series.index)


def supertrend(
    df: pd.DataFrame,
    period: int = 7,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Calcule le SuperTrend.

    Args:
        df: DataFrame avec colonnes High, Low, Close
        period: Période ATR (défaut: 7)
        multiplier: Multiplicateur ATR (défaut: 3.0)

    Returns:
        DataFrame avec colonnes: SuperTrend, Direction (1=haussier, -1=baissier)
    """
    from .volatility import atr as calc_atr

    atr_values = calc_atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2

    upper_band = hl2 + (multiplier * atr_values)
    lower_band = hl2 - (multiplier * atr_values)

    supertrend_val = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Upper band
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Lower band
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Direction
        if i == 1:
            direction.iloc[i] = 1
        elif supertrend_val.iloc[i - 1] == upper_band.iloc[i - 1]:
            direction.iloc[i] = -1 if df["Close"].iloc[i] > upper_band.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["Close"].iloc[i] < lower_band.iloc[i] else -1

        supertrend_val.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == -1 else upper_band.iloc[i]

    return pd.DataFrame({
        "SuperTrend": supertrend_val,
        "Direction": direction,
    }, index=df.index)


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les composantes de l'Ichimoku Cloud.

    Returns:
        DataFrame avec: Tenkan (9), Kijun (26), SenkouA, SenkouB, Chikou
    """
    # Tenkan-sen (Conversion Line) - 9 périodes
    high_9 = df["High"].rolling(9).max()
    low_9 = df["Low"].rolling(9).min()
    tenkan = (high_9 + low_9) / 2

    # Kijun-sen (Base Line) - 26 périodes
    high_26 = df["High"].rolling(26).max()
    low_26 = df["Low"].rolling(26).min()
    kijun = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A) - décalé de 26 périodes
    senkou_a = ((tenkan + kijun) / 2).shift(26)

    # Senkou Span B (Leading Span B) - 52 périodes, décalé de 26
    high_52 = df["High"].rolling(52).max()
    low_52 = df["Low"].rolling(52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)

    # Chikou Span (Lagging Span) - décalé de -26 périodes
    chikou = df["Close"].shift(-26)

    return pd.DataFrame({
        "Tenkan": tenkan,
        "Kijun": kijun,
        "SenkouA": senkou_a,
        "SenkouB": senkou_b,
        "Chikou": chikou,
    }, index=df.index)


def pivot_points(df: pd.DataFrame, method: str = "classic") -> pd.DataFrame:
    """
    Calcule les niveaux de pivot (supports et résistances).

    Args:
        df: DataFrame avec colonnes High, Low, Close
        method: Méthode de calcul ("classic", "fibonacci", "camarilla")

    Returns:
        DataFrame avec PP, R1, R2, R3, S1, S2, S3
    """
    H = df["High"]
    L = df["Low"]
    C = df["Close"]

    if method == "classic":
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        S1 = 2 * PP - H
        R2 = PP + (H - L)
        S2 = PP - (H - L)
        R3 = H + 2 * (PP - L)
        S3 = L - 2 * (H - PP)

    elif method == "fibonacci":
        PP = (H + L + C) / 3
        range_hl = H - L
        R1 = PP + 0.382 * range_hl
        S1 = PP - 0.382 * range_hl
        R2 = PP + 0.618 * range_hl
        S2 = PP - 0.618 * range_hl
        R3 = PP + 1.000 * range_hl
        S3 = PP - 1.000 * range_hl

    elif method == "camarilla":
        PP = (H + L + C) / 3
        range_hl = H - L
        R1 = C + range_hl * 1.1 / 12
        S1 = C - range_hl * 1.1 / 12
        R2 = C + range_hl * 1.1 / 6
        S2 = C - range_hl * 1.1 / 6
        R3 = C + range_hl * 1.1 / 4
        S3 = C - range_hl * 1.1 / 4

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    return pd.DataFrame({
        "PP": PP, "R1": R1, "R2": R2, "R3": R3,
        "S1": S1, "S2": S2, "S3": S3,
    }, index=df.index)
