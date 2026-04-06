"""
Module de récupération des données de marché pour la BVC.
Sources (par ordre de priorité) :
  1. tvdatafeed   — TradingView unofficial API, exchange CSE, pas de rate-limit
  2. leboursier.ma — Scraping du site marocain, données BVC directes
  3. Yahoo Finance  — Fallback API directe + yfinance
"""

import pandas as pd
import requests
import os
import io
from datetime import datetime, timedelta
from typing import Optional
import logging
import time

from .tickers import BVC_TICKERS, get_ticker_info

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "cache"
)

_CACHE_RAW_URL = "https://raw.githubusercontent.com/HOUMMANI/AT/data-cache/data/cache/{symbol}.csv"

_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 31, "3mo": 92,
    "6mo": 183, "1y": 365, "2y": 730, "5y": 1825,
    "10y": 3650, "ytd": None, "max": 7300,
}

_YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://finance.yahoo.com/",
}


# ─── Source 1 : TradingView via tvdatafeed ────────────────────────────────────

def _fetch_tvdatafeed(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Récupère les données depuis TradingView (exchange CSE pour la BVC).
    Utilise la librairie tvdatafeed (accès non-officiel, pas de clé API).
    """
    try:
        from tvdatafeed import TvDatafeed, Interval

        interval_map = {
            "1d": Interval.in_daily,
            "1wk": Interval.in_weekly,
            "1mo": Interval.in_monthly,
        }
        tv_interval = interval_map.get(interval, Interval.in_daily)

        days = _PERIOD_DAYS.get(period, 365)
        n_bars = min(days + 50, 5000)  # marge de sécurité

        tv = TvDatafeed()
        df = tv.get_hist(
            symbol=symbol,
            exchange="CSE",
            interval=tv_interval,
            n_bars=n_bars,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Renommer les colonnes TradingView → standard
        rename = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume"}
        df = df.rename(columns=rename)
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()

        # Filtrer par période
        if days and period != "max":
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff]

        return df.dropna(subset=["Close"]) if not df.empty else pd.DataFrame()

    except Exception as e:
        logger.debug(f"tvdatafeed échec pour {symbol}: {e}")
        return pd.DataFrame()


# ─── Source 2 : leboursier.ma ─────────────────────────────────────────────────

def _fetch_leboursier(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Scrape les données historiques depuis leboursier.ma.
    Source fiable pour toutes les actions BVC.
    """
    try:
        # Récupérer la liste des stocks pour trouver l'ID interne
        search_url = f"https://www.leboursier.ma/api?method=searchStock&name={symbol}&format=json"
        headers = {"User-Agent": "Mozilla/5.0 Chrome/122.0.0.0"}

        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()

        results = resp.json()
        if not results:
            return pd.DataFrame()

        # Trouver le bon ticker
        stock_id = None
        for r in results if isinstance(results, list) else [results]:
            ticker = r.get("ticker", "").upper()
            if ticker == symbol.upper():
                stock_id = r.get("id") or r.get("stockId")
                break

        if not stock_id:
            # Prendre le premier résultat
            r = results[0] if isinstance(results, list) else results
            stock_id = r.get("id") or r.get("stockId")

        if not stock_id:
            return pd.DataFrame()

        # Récupérer les données historiques
        hist_url = f"https://www.leboursier.ma/api?method=getHistoricalPrice&stock={stock_id}&format=json"
        resp2 = requests.get(hist_url, headers=headers, timeout=15)
        if resp2.status_code != 200:
            return pd.DataFrame()

        data = resp2.json()
        if not data:
            return pd.DataFrame()

        # Parser la réponse (format variable selon la version de l'API)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Normaliser les colonnes
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "date" in cl:
                col_map[c] = "Date"
            elif cl in ("open", "ouverture"):
                col_map[c] = "Open"
            elif cl in ("high", "haut", "max"):
                col_map[c] = "High"
            elif cl in ("low", "bas", "min"):
                col_map[c] = "Low"
            elif cl in ("close", "cloture", "dernier", "last"):
                col_map[c] = "Close"
            elif "vol" in cl:
                col_map[c] = "Volume"

        df = df.rename(columns=col_map)

        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        for c in ["Open", "High", "Low", "Volume"]:
            if c not in df.columns:
                df[c] = df["Close"]

        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")

        # Filtrer par période
        days = _PERIOD_DAYS.get(period, 365)
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff]

        return df.dropna(subset=["Close"])

    except Exception as e:
        logger.debug(f"leboursier.ma échec pour {symbol}: {e}")
        return pd.DataFrame()


# ─── Source 3 : Yahoo Finance API directe ────────────────────────────────────

def _fetch_yahoo_direct(yahoo_ticker: str, period: str = "1y",
                        start: Optional[str] = None, end: Optional[str] = None,
                        interval: str = "1d") -> pd.DataFrame:
    """Appel direct à l'API Yahoo Finance v8 avec headers navigateur."""
    import time as _time
    now_ts = int(_time.time())
    days = _PERIOD_DAYS.get(period, 365)
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp()) if start \
               else now_ts - (days or 365) * 86400
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp()) if end else now_ts

    for base in ["query2", "query1"]:
        try:
            url = f"https://{base}.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
            resp = requests.get(url, headers=_YAHOO_HEADERS, timeout=15,
                                params={"period1": start_ts, "period2": end_ts,
                                        "interval": interval, "events": "history"})
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
            df = pd.DataFrame({"Open": quote["open"], "High": quote["high"],
                                "Low": quote["low"], "Close": closes,
                                "Volume": quote["volume"]},
                               index=pd.to_datetime(timestamps, unit="s", utc=True))
            df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            df = df.sort_index().dropna(subset=["Close"])
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"Yahoo {base} échec: {e}")
    return pd.DataFrame()


# ─── Cache CSV (GitHub Actions) ───────────────────────────────────────────────

def _read_csv_cache(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Lit les données depuis le cache CSV (branche data-cache sur GitHub)."""
    df = pd.DataFrame()

    # Fichier local (développement)
    path = os.path.join(_CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
        except Exception:
            pass

    # URL raw GitHub (Streamlit Cloud)
    if df.empty:
        try:
            url = _CACHE_RAW_URL.format(symbol=symbol)
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), index_col="Date", parse_dates=True)
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame()

    days = _PERIOD_DAYS.get(period, 365)
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

    return df if not df.empty else pd.DataFrame()


# ─── BVCDataFetcher ───────────────────────────────────────────────────────────

class BVCDataFetcher:
    """
    Récupère les données OHLCV BVC depuis plusieurs sources :
    1. Cache CSV GitHub Actions
    2. TradingView (tvdatafeed, exchange CSE)
    3. leboursier.ma (scraping)
    4. Yahoo Finance API directe
    5. yfinance (fallback)
    """

    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                       "1d", "5d", "1wk", "1mo", "3mo"]

    def __init__(self):
        self._cache = {}

    def _resolve_yahoo_ticker(self, symbol: str) -> str:
        symbol = symbol.upper()
        info = get_ticker_info(symbol)
        if info:
            return info.get("yahoo", symbol)
        return f"{symbol}.CS" if not symbol.endswith(".CS") else symbol

    def get_ohlcv(self, symbol: str, period: str = "1y", interval: str = "1d",
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:

        if period not in self.VALID_PERIODS:
            raise ValueError(f"Période invalide: {period}")

        yahoo_ticker = self._resolve_yahoo_ticker(symbol)
        cache_key = f"{symbol}_{period}_{interval}_{start}_{end}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        df = pd.DataFrame()

        # 1. Cache CSV (GitHub Actions — le plus rapide)
        if not df.empty or (df := _read_csv_cache(symbol, period)):
            logger.info(f"{symbol}: données depuis cache CSV")

        # 2. TradingView (CSE exchange)
        if df.empty:
            df = _fetch_tvdatafeed(symbol, period=period, interval=interval)
            if not df.empty:
                logger.info(f"{symbol}: données depuis TradingView/CSE")

        # 3. leboursier.ma
        if df.empty:
            df = _fetch_leboursier(symbol, period=period)
            if not df.empty:
                logger.info(f"{symbol}: données depuis leboursier.ma")

        # 4. Yahoo Finance API directe
        if df.empty:
            df = _fetch_yahoo_direct(yahoo_ticker, period=period,
                                      start=start, end=end, interval=interval)
            if not df.empty:
                logger.info(f"{symbol}: données depuis Yahoo direct")

        # 5. yfinance fallback
        if df.empty:
            try:
                import yfinance as yf
                ticker = yf.Ticker(yahoo_ticker)
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    logger.info(f"{symbol}: données depuis yfinance")
            except Exception as e:
                logger.warning(f"{symbol}: yfinance échec: {e}")

        if df is None or df.empty:
            logger.warning(f"Aucune donnée pour {symbol}")
            return pd.DataFrame()

        # Standardiser
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
        return df.copy()

    def get_multiple(self, symbols: list, period: str = "1y", interval: str = "1d") -> dict:
        return {s: self.get_ohlcv(s, period=period, interval=interval)
                for s in symbols
                if not (df := self.get_ohlcv(s, period=period, interval=interval)).empty
                or True}

    def get_multiple(self, symbols: list, period: str = "1y", interval: str = "1d") -> dict:
        results = {}
        for s in symbols:
            try:
                results[s] = self.get_ohlcv(s, period=period, interval=interval)
            except Exception:
                pass
        return results

    def get_market_overview(self, period: str = "1mo") -> pd.DataFrame:
        major = ["ATW", "IAM", "BCP", "BOA", "CIH", "LHM", "MNG", "COSU", "LES", "WAA"]
        data = self.get_multiple(major, period=period)
        rows = []
        for symbol, df in data.items():
            if df.empty:
                continue
            info = get_ticker_info(symbol)
            rows.append({
                "Symbole": symbol,
                "Nom": info.get("name", symbol) if info else symbol,
                "Secteur": info.get("secteur", "N/A") if info else "N/A",
                "Dernier cours": round(df["Close"].iloc[-1], 2),
                "Variation (%)": round(
                    (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100, 2),
                "Volume moyen": int(df["Volume"].mean()),
            })
        return pd.DataFrame(rows).sort_values("Variation (%)", ascending=False) \
               if rows else pd.DataFrame()

    def get_info(self, symbol: str) -> dict:
        try:
            import yfinance as yf
            return yf.Ticker(self._resolve_yahoo_ticker(symbol)).info
        except Exception:
            return {}

    def get_sector_data(self, secteur: str, period: str = "1y") -> dict:
        from .tickers import get_tickers_by_sector
        return self.get_multiple(list(get_tickers_by_sector(secteur).keys()), period=period)

    def clear_cache(self):
        self._cache.clear()
