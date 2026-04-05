"""
Module de récupération des données de marché pour la BVC.
Sources (par ordre de priorité) :
  1. Stooq      — via pandas_datareader, gratuit, pas de rate-limit
  2. Yahoo Finance — fallback via yfinance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Union
import logging
import time

from .tickers import BVC_TICKERS, get_ticker_info

logger = logging.getLogger(__name__)


def _period_to_start(period: str) -> datetime:
    """Convertit une période (ex: '1y') en date de début."""
    now = datetime.now()
    mapping = {
        "1d": timedelta(days=1), "5d": timedelta(days=5),
        "1mo": timedelta(days=31), "3mo": timedelta(days=92),
        "6mo": timedelta(days=183), "1y": timedelta(days=365),
        "2y": timedelta(days=730), "5y": timedelta(days=1825),
        "10y": timedelta(days=3650), "ytd": timedelta(days=(now - datetime(now.year, 1, 1)).days),
        "max": timedelta(days=7300),
    }
    return now - mapping.get(period, timedelta(days=365))


def _fetch_stooq(yahoo_ticker: str, period: str = "1y",
                 start: Optional[str] = None, end: Optional[str] = None,
                 interval: str = "1d") -> pd.DataFrame:
    """
    Récupère les données depuis Stooq via pandas_datareader.
    Stooq utilise le même suffixe .CS que Yahoo pour les actions marocaines.
    Supporte uniquement interval=1d/1wk/1mo (pas d'intraday).
    """
    if interval not in ("1d", "1wk", "1mo"):
        return pd.DataFrame()

    try:
        from pandas_datareader import data as pdr

        # Stooq: le symbole doit être en minuscules
        stooq_sym = yahoo_ticker.lower()

        dt_start = datetime.strptime(start, "%Y-%m-%d") if start else _period_to_start(period)
        dt_end = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()

        df = pdr.DataReader(stooq_sym, "stooq", start=dt_start, end=dt_end)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Stooq renvoie Open/High/Low/Close/Volume
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Rééchantillonner si hebdo ou mensuel
        if interval == "1wk":
            df = df.resample("W").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
        elif interval == "1mo":
            df = df.resample("ME").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()

        return df

    except Exception as e:
        logger.debug(f"Stooq échec pour {yahoo_ticker}: {e}")
        return pd.DataFrame()


class BVCDataFetcher:
    """
    Récupère les données historiques des actions de la Bourse de Casablanca.

    Exemple d'utilisation:
        fetcher = BVCDataFetcher()
        df = fetcher.get_ohlcv("ATW")  # Attijariwafa Bank
        df = fetcher.get_ohlcv("IAM", period="1y")
    """

    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self):
        self._cache = {}

    def _resolve_yahoo_ticker(self, symbol: str) -> str:
        """Convertit un symbole BVC en ticker Yahoo Finance."""
        symbol = symbol.upper()
        info = get_ticker_info(symbol)
        if info:
            return info.get("yahoo", symbol)
        # Si déjà au format Yahoo (.CS) ou indice (^), retourner tel quel
        if symbol.endswith(".CS") or symbol.startswith("^"):
            return symbol
        # Ajouter suffixe .CS par défaut
        return f"{symbol}.CS"

    def get_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Récupère les données OHLCV (Open, High, Low, Close, Volume) d'une action.

        Args:
            symbol: Symbole BVC (ex: "ATW", "IAM") ou ticker Yahoo (ex: "ATW.CS")
            period: Période ("1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max")
            interval: Intervalle des données ("1d","1wk","1mo")
            start: Date de début au format "YYYY-MM-DD" (prioritaire sur period)
            end: Date de fin au format "YYYY-MM-DD"

        Returns:
            DataFrame avec colonnes: Open, High, Low, Close, Volume
        """
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Période invalide: {period}. Valeurs possibles: {self.VALID_PERIODS}")
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Intervalle invalide: {interval}. Valeurs possibles: {self.VALID_INTERVALS}")

        yahoo_ticker = self._resolve_yahoo_ticker(symbol)
        cache_key = f"{yahoo_ticker}_{period}_{interval}_{start}_{end}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # ── 1. Essayer Stooq (pas de rate-limit, fiable sur serveurs cloud) ──────
        df = _fetch_stooq(yahoo_ticker, period=period, start=start, end=end, interval=interval)

        # ── 2. Fallback Yahoo Finance avec retry ─────────────────────────────
        if df.empty:
            delays = [2, 6, 15]
            for delay in delays + [None]:
                try:
                    ticker = yf.Ticker(yahoo_ticker)
                    if start:
                        df = ticker.history(start=start, end=end, interval=interval)
                    else:
                        df = ticker.history(period=period, interval=interval)
                    if not df.empty:
                        break
                except Exception as e:
                    err = str(e).lower()
                    if ("ratelimit" in err or "429" in err or "too many" in err) and delay:
                        logger.warning(f"Yahoo rate-limit pour {symbol}, retry dans {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Yahoo erreur pour {symbol}: {e}")
                        break

        if df is None or df.empty:
            logger.warning(f"Aucune donnée disponible pour {symbol} ({yahoo_ticker})")
            return pd.DataFrame()

        # Nettoyer et standardiser
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()
        df.attrs["symbol"] = symbol
        df.attrs["yahoo_ticker"] = yahoo_ticker

        info = get_ticker_info(symbol)
        df.attrs["name"] = info.get("name", symbol) if info else symbol
        df.attrs["secteur"] = info.get("secteur", "N/A") if info else "N/A"

        self._cache[cache_key] = df
        logger.info(f"Données récupérées: {symbol} ({len(df)} lignes)")
        return df.copy()

    def get_multiple(
        self,
        symbols: list,
        period: str = "1y",
        interval: str = "1d",
    ) -> dict:
        """
        Récupère les données OHLCV de plusieurs actions.

        Returns:
            Dictionnaire {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, period=period, interval=interval)
            except Exception as e:
                logger.warning(f"Impossible de récupérer {symbol}: {e}")
        return results

    def get_info(self, symbol: str) -> dict:
        """Retourne les informations fondamentales d'une action."""
        yahoo_ticker = self._resolve_yahoo_ticker(symbol)
        try:
            ticker = yf.Ticker(yahoo_ticker)
            return ticker.info
        except Exception as e:
            logger.error(f"Erreur info {symbol}: {e}")
            return {}

    def get_sector_data(self, secteur: str, period: str = "1y") -> dict:
        """Récupère les données de toutes les actions d'un secteur."""
        from .tickers import get_tickers_by_sector
        tickers = get_tickers_by_sector(secteur)
        return self.get_multiple(list(tickers.keys()), period=period)

    def get_market_overview(self, period: str = "1mo") -> pd.DataFrame:
        """
        Retourne un aperçu du marché avec les variations des principales actions.
        """
        major = ["MASI", "ATW", "IAM", "BCP", "BOA", "CIH", "LHM", "MNG", "COSU", "LES"]
        data = self.get_multiple(major, period=period)

        rows = []
        for symbol, df in data.items():
            if df.empty:
                continue
            first_close = df["Close"].iloc[0]
            last_close = df["Close"].iloc[-1]
            variation = ((last_close - first_close) / first_close) * 100
            info = get_ticker_info(symbol)
            rows.append({
                "Symbole": symbol,
                "Nom": info.get("name", symbol) if info else symbol,
                "Secteur": info.get("secteur", "N/A") if info else "N/A",
                "Dernier cours": round(last_close, 2),
                "Variation (%)": round(variation, 2),
                "Volume moyen": int(df["Volume"].mean()),
            })

        overview = pd.DataFrame(rows)
        if not overview.empty:
            overview = overview.sort_values("Variation (%)", ascending=False)
        return overview

    def clear_cache(self):
        """Vide le cache des données."""
        self._cache.clear()
