"""
Module de récupération des données de marché pour la BVC.
Sources (par ordre de priorité) :
  1. Cache CSV local (data/cache/) — mis à jour par GitHub Actions quotidiennement
  2. Yahoo Finance API directe   — appel HTTP navigateur
  3. yfinance library             — fallback avec retry
"""

import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from typing import Optional
import logging
import time

from .tickers import BVC_TICKERS, get_ticker_info

logger = logging.getLogger(__name__)

# Répertoire du cache CSV (peuplé par GitHub Actions)
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "cache"
)

_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Referer": "https://finance.yahoo.com/",
}

_PERIOD_SECONDS = {
    "1d": 86400, "5d": 432000,
    "1mo": 2678400, "3mo": 8035200, "6mo": 16070400,
    "1y": 31536000, "2y": 63072000, "5y": 157680000,
    "10y": 315360000, "ytd": None, "max": 788400000,
}


# URL raw GitHub pour le cache (branche data-cache, pas de rate-limit)
_CACHE_RAW_URL = "https://raw.githubusercontent.com/HOUMMANI/AT/data-cache/data/cache/{symbol}.csv"


def _read_csv_cache(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Lit les données depuis le cache CSV (branche data-cache sur GitHub).
    Essaie d'abord le fichier local, puis l'URL raw GitHub.
    """
    df = pd.DataFrame()

    # 1. Fichier local (si présent — ex: développement local)
    path = os.path.join(_CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
        except Exception:
            pass

    # 2. URL raw GitHub (Streamlit Cloud lit directement depuis la branche data-cache)
    if df.empty:
        try:
            url = _CACHE_RAW_URL.format(symbol=symbol)
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), index_col="Date", parse_dates=True)
        except Exception as e:
            logger.debug(f"Cache raw URL échec pour {symbol}: {e}")

    if df.empty:
        return pd.DataFrame()

    # Filtrer par période
    secs = _PERIOD_SECONDS.get(period)
    if secs:
        cutoff = datetime.now() - timedelta(seconds=secs)
        df = df[df.index >= cutoff]

    return df if not df.empty else pd.DataFrame()


def _fetch_yahoo_direct(
    yahoo_ticker: str,
    period: str = "1y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Appel direct à l'API Yahoo Finance v8 avec headers navigateur."""
    now_ts = int(time.time())

    if start:
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    else:
        secs = _PERIOD_SECONDS.get(period)
        if secs is None:
            secs = int((datetime.now() - datetime(datetime.now().year, 1, 1)).total_seconds())
        start_ts = now_ts - secs

    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp()) if end else now_ts

    for base in ["query2", "query1"]:
        url = f"https://{base}.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
        params = {
            "period1": start_ts, "period2": end_ts,
            "interval": interval, "events": "history",
            "includeAdjustedClose": "true",
        }
        try:
            resp = requests.get(url, headers=_YAHOO_HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                continue

            data = resp.json()
            result = (data.get("chart", {}).get("result") or [])
            if not result:
                continue

            r = result[0]
            timestamps = r.get("timestamp", [])
            if not timestamps:
                continue

            quote = r["indicators"]["quote"][0]
            adj = r["indicators"].get("adjclose", [{}])
            closes = (adj[0].get("adjclose") if adj else None) or quote["close"]

            df = pd.DataFrame({
                "Open": quote["open"], "High": quote["high"],
                "Low": quote["low"], "Close": closes,
                "Volume": quote["volume"],
            }, index=pd.to_datetime(timestamps, unit="s", utc=True))

            df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            df = df.sort_index().dropna(subset=["Close"])
            if not df.empty:
                return df

        except Exception as e:
            logger.debug(f"Yahoo direct {base} échec pour {yahoo_ticker}: {e}")

    return pd.DataFrame()


class BVCDataFetcher:
    """
    Récupère les données historiques des actions de la Bourse de Casablanca.

    Stratégie :
      1. Appel HTTP direct à l'API Yahoo (headers navigateur) — pas de rate-limit
      2. Fallback yfinance library avec retry exponentiel

    Exemple :
        fetcher = BVCDataFetcher()
        df = fetcher.get_ohlcv("ATW")
        df = fetcher.get_ohlcv("IAM", period="1y")
    """

    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self):
        self._cache = {}

    def _resolve_yahoo_ticker(self, symbol: str) -> str:
        symbol = symbol.upper()
        info = get_ticker_info(symbol)
        if info:
            return info.get("yahoo", symbol)
        if symbol.endswith(".CS") or symbol.startswith("^"):
            return symbol
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
        Récupère les données OHLCV.
        Essaie l'API directe Yahoo en premier, yfinance en fallback.
        """
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Période invalide: {period}. Valeurs possibles: {self.VALID_PERIODS}")
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Intervalle invalide: {interval}")

        yahoo_ticker = self._resolve_yahoo_ticker(symbol)
        cache_key = f"{yahoo_ticker}_{period}_{interval}_{start}_{end}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # ── 1. Cache CSV (GitHub Actions, aucun appel API) ────────────────────
        df = _read_csv_cache(symbol, period=period)

        # ── 2. API Yahoo directe (headers navigateur) ─────────────────────────
        if df.empty:
            df = _fetch_yahoo_direct(yahoo_ticker, period=period, start=start,
                                      end=end, interval=interval)

        # ── 2. Fallback yfinance avec retry ────────────────────────────────────
        if df.empty:
            for delay in [3, 8, 20, None]:
                try:
                    ticker = yf.Ticker(yahoo_ticker)
                    df = ticker.history(
                        period=period if not start else "max",
                        start=start, end=end, interval=interval
                    )
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    err = str(e).lower()
                    if ("ratelimit" in err or "429" in err or "too many" in err) and delay:
                        logger.warning(f"yfinance rate-limit {symbol}, retry {delay}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"yfinance erreur {symbol}: {e}")
                        break

        if df is None or df.empty:
            logger.warning(f"Aucune donnée pour {symbol} ({yahoo_ticker})")
            return pd.DataFrame()

        # Standardiser les colonnes
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()

        info = get_ticker_info(symbol)
        df.attrs["symbol"] = symbol
        df.attrs["yahoo_ticker"] = yahoo_ticker
        df.attrs["name"] = info.get("name", symbol) if info else symbol
        df.attrs["secteur"] = info.get("secteur", "N/A") if info else "N/A"

        self._cache[cache_key] = df
        logger.info(f"Données récupérées: {symbol} ({len(df)} lignes)")
        return df.copy()

    def get_multiple(self, symbols: list, period: str = "1y", interval: str = "1d") -> dict:
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, period=period, interval=interval)
            except Exception as e:
                logger.warning(f"Impossible de récupérer {symbol}: {e}")
        return results

    def get_info(self, symbol: str) -> dict:
        yahoo_ticker = self._resolve_yahoo_ticker(symbol)
        try:
            return yf.Ticker(yahoo_ticker).info
        except Exception:
            return {}

    def get_sector_data(self, secteur: str, period: str = "1y") -> dict:
        from .tickers import get_tickers_by_sector
        return self.get_multiple(list(get_tickers_by_sector(secteur).keys()), period=period)

    def get_market_overview(self, period: str = "1mo") -> pd.DataFrame:
        major = ["ATW", "IAM", "BCP", "BOA", "CIH", "LHM", "MNG", "COSU", "LES", "WAA"]
        data = self.get_multiple(major, period=period)
        rows = []
        for symbol, df in data.items():
            if df.empty:
                continue
            first_close = df["Close"].iloc[0]
            last_close = df["Close"].iloc[-1]
            info = get_ticker_info(symbol)
            rows.append({
                "Symbole": symbol,
                "Nom": info.get("name", symbol) if info else symbol,
                "Secteur": info.get("secteur", "N/A") if info else "N/A",
                "Dernier cours": round(last_close, 2),
                "Variation (%)": round((last_close - first_close) / first_close * 100, 2),
                "Volume moyen": int(df["Volume"].mean()),
            })
        overview = pd.DataFrame(rows)
        if not overview.empty:
            overview = overview.sort_values("Variation (%)", ascending=False)
        return overview

    def clear_cache(self):
        self._cache.clear()
