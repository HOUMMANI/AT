"""
Indicateurs de volume : OBV, VWAP, MFI, CMF, etc.
"""

import pandas as pd
import numpy as np


def obv(df: pd.DataFrame) -> pd.Series:
    """
    Calcule l'On-Balance Volume (OBV).

    Args:
        df: DataFrame avec colonnes Close, Volume

    Returns:
        Série OBV
    """
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0
    obv_values = (direction * df["Volume"]).cumsum()
    obv_values.name = "OBV"
    return obv_values


def volume_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Moyenne mobile du volume.

    Args:
        df: DataFrame avec colonne Volume
        window: Période (défaut: 20)

    Returns:
        Série SMA du volume
    """
    vol_sma = df["Volume"].rolling(window).mean()
    vol_sma.name = f"Volume_SMA_{window}"
    return vol_sma


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calcule le VWAP (Volume Weighted Average Price).

    Note: Le VWAP se réinitialise chaque journée. Pour des données journalières,
    il est calculé sur toute la période (version approximative).

    Args:
        df: DataFrame avec colonnes High, Low, Close, Volume

    Returns:
        Série VWAP
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tp_vol = (typical_price * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].cumsum()
    vwap_values = cumulative_tp_vol / cumulative_vol
    vwap_values.name = "VWAP"
    return vwap_values


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcule le Money Flow Index (MFI).

    Args:
        df: DataFrame avec colonnes High, Low, Close, Volume
        window: Période (défaut: 14)

    Returns:
        Série MFI entre 0 et 100
        - > 80 : Surachat
        - < 20 : Survente
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]

    tp_diff = typical_price.diff()
    positive_flow = money_flow.where(tp_diff > 0, 0)
    negative_flow = money_flow.where(tp_diff < 0, 0)

    pos_sum = positive_flow.rolling(window).sum()
    neg_sum = negative_flow.rolling(window).sum()

    money_ratio = pos_sum / neg_sum.abs()
    mfi_values = 100 - (100 / (1 + money_ratio))
    mfi_values.name = f"MFI_{window}"
    return mfi_values


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calcule le Chaikin Money Flow (CMF).

    Args:
        df: DataFrame avec colonnes High, Low, Close, Volume
        window: Période (défaut: 20)

    Returns:
        Série CMF entre -1 et +1
        - > 0 : Pression acheteuse
        - < 0 : Pression vendeuse
    """
    high_low_diff = df["High"] - df["Low"]
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_diff.replace(0, np.nan)
    money_flow_volume = clv * df["Volume"]

    cmf_values = money_flow_volume.rolling(window).sum() / df["Volume"].rolling(window).sum()
    cmf_values.name = f"CMF_{window}"
    return cmf_values


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calcule la ligne Accumulation/Distribution.

    Args:
        df: DataFrame avec colonnes High, Low, Close, Volume

    Returns:
        Série A/D
    """
    high_low_diff = df["High"] - df["Low"]
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_diff.replace(0, np.nan)
    ad_values = (clv * df["Volume"]).cumsum()
    ad_values.name = "A/D"
    return ad_values


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calcule le Volume Relatif (RVOL).

    Args:
        df: DataFrame avec colonne Volume
        window: Période de comparaison (défaut: 20)

    Returns:
        Série RVOL (ratio volume actuel / volume moyen)
        - > 2 : Volume anormalement élevé (signal fort)
    """
    avg_vol = df["Volume"].rolling(window).mean()
    rvol = df["Volume"] / avg_vol
    rvol.name = f"RVOL_{window}"
    return rvol
