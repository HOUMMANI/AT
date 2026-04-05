"""
Indicateurs de volatilité : Bandes de Bollinger, ATR, Keltner, etc.
"""

import pandas as pd
import numpy as np


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Calcule les Bandes de Bollinger.

    Args:
        series: Série de prix de clôture
        window: Période de la moyenne mobile (défaut: 20)
        num_std: Nombre d'écarts-types (défaut: 2)

    Returns:
        DataFrame avec colonnes: Milieu (SMA), Haute, Basse, Largeur, %B
    """
    sma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    width = (upper - lower) / sma * 100  # Largeur en %
    pct_b = (series - lower) / (upper - lower)  # %B

    return pd.DataFrame({
        "BB_Milieu": sma,
        "BB_Haute": upper,
        "BB_Basse": lower,
        "BB_Largeur": width,
        "BB_%B": pct_b,
    }, index=series.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcule l'Average True Range (ATR).

    Args:
        df: DataFrame avec colonnes High, Low, Close
        window: Période de calcul (défaut: 14)

    Returns:
        Série ATR
    """
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close_prev).abs(),
        "lc": (low - close_prev).abs(),
    }).max(axis=1)

    atr_values = tr.ewm(com=window - 1, adjust=False).mean()
    atr_values.name = f"ATR_{window}"
    return atr_values


def keltner_channels(
    df: pd.DataFrame,
    ema_window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calcule les Canaux de Keltner.

    Args:
        df: DataFrame avec colonnes High, Low, Close
        ema_window: Période EMA centrale (défaut: 20)
        atr_window: Période ATR (défaut: 10)
        multiplier: Multiplicateur ATR (défaut: 2)

    Returns:
        DataFrame avec colonnes: KC_Milieu, KC_Haute, KC_Basse
    """
    from .trend import ema as calc_ema
    middle = calc_ema(df["Close"], ema_window)
    atr_val = atr(df, atr_window)

    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return pd.DataFrame({
        "KC_Milieu": middle,
        "KC_Haute": upper,
        "KC_Basse": lower,
    }, index=df.index)


def donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calcule les Canaux de Donchian.

    Args:
        df: DataFrame avec colonnes High, Low
        window: Période (défaut: 20)

    Returns:
        DataFrame avec colonnes: DC_Haute, DC_Basse, DC_Milieu
    """
    upper = df["High"].rolling(window).max()
    lower = df["Low"].rolling(window).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({
        "DC_Haute": upper,
        "DC_Basse": lower,
        "DC_Milieu": middle,
    }, index=df.index)


def historical_volatility(series: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """
    Calcule la volatilité historique annualisée.

    Args:
        series: Série de prix de clôture
        window: Fenêtre de calcul (défaut: 20)
        trading_days: Jours de trading par an (BVC: ~252)

    Returns:
        Série de volatilité en pourcentage annualisé
    """
    log_returns = np.log(series / series.shift(1))
    vol = log_returns.rolling(window).std() * np.sqrt(trading_days) * 100
    vol.name = f"HV_{window}"
    return vol


def squeeze_momentum(df: pd.DataFrame, bb_window: int = 20, kc_window: int = 20) -> pd.DataFrame:
    """
    Détecte le Squeeze Momentum (compression entre BB et KC).

    Returns:
        DataFrame avec: Squeeze (bool), Momentum
    """
    bb = bollinger_bands(df["Close"], bb_window)
    kc = keltner_channels(df, kc_window)

    squeeze = (bb["BB_Haute"] < kc["KC_Haute"]) & (bb["BB_Basse"] > kc["KC_Basse"])

    # Momentum (delta)
    highest_high = df["High"].rolling(kc_window).max()
    lowest_low = df["Low"].rolling(kc_window).min()
    mid = (highest_high + lowest_low) / 2 + kc["KC_Milieu"] / 2
    momentum = df["Close"] - mid

    # Lissage
    from .trend import ema as calc_ema
    momentum_smooth = calc_ema(momentum, 5)

    return pd.DataFrame({
        "Squeeze": squeeze,
        "Momentum": momentum_smooth,
    }, index=df.index)
