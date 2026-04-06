"""
Données en temps réel / quasi-temps réel pour la BVC.

Sources :
  1. Yahoo Finance (fast_info) — données avec ~15-20 min de délai, gratuit
  2. Yahoo Finance intraday (interval=1m/2m/5m) — séance en cours
  3. Scraper BVC / leboursier.ma — données en direct (scraping HTML)

Classes :
  RealTimeQuote   — snapshot d'un cours instantané
  RealTimeFetcher — récupère les données via Yahoo Finance
  BVCScraper      — scrape les sites publics de la BVC pour les cours live

Usage rapide :
    from src.data.realtime import RealTimeFetcher
    rt = RealTimeFetcher()

    # Cotation instantanée
    q = rt.get_quote("ATW")
    print(q)

    # Données intraday (dernière séance, 1 minute)
    df = rt.get_intraday("IAM", interval="1m")

    # Surveillance en boucle (toutes les 30s)
    rt.stream(["ATW","IAM","BCP"], callback=print, interval=30)

    # Tableau de bord de marché temps réel
    overview = rt.get_market_snapshot()
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Callable, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# DATACLASS : snapshot d'un cours
# ─────────────────────────────────────────────────────────────────

@dataclass
class RealTimeQuote:
    symbol:       str
    name:         str
    price:        float
    open_:        float
    high:         float
    low:          float
    prev_close:   float
    change:       float           # variation absolue
    change_pct:   float           # variation en %
    volume:       int
    avg_volume:   int
    high_52w:     float
    low_52w:      float
    market_cap:   Optional[float]
    timestamp:    datetime
    source:       str = "yahoo"
    is_live:      bool = False    # True si la séance est ouverte

    def __str__(self):
        arrow = "▲" if self.change >= 0 else "▼"
        live  = " [LIVE]" if self.is_live else " [CLÔTURE]"
        return (
            f"{self.symbol:<8} {self.name[:28]:<28} "
            f"{self.price:>9.2f} MAD  "
            f"{arrow}{abs(self.change_pct):>5.2f}%  "
            f"Vol: {self.volume:>10,}{live}"
        )


# ─────────────────────────────────────────────────────────────────
# REALTIME FETCHER — Yahoo Finance
# ─────────────────────────────────────────────────────────────────

class RealTimeFetcher:
    """
    Récupère les cotations en temps réel (ou quasi-réel) via Yahoo Finance.

    Yahoo Finance fournit des données avec un délai d'environ 15-20 minutes
    pour la BVC. En dehors des heures de cotation, renvoie la dernière clôture.

    Heures de cotation BVC : 09h30 – 15h30 (heure de Casablanca, UTC+1)
    """

    BVC_OPEN_HOUR  = 9
    BVC_OPEN_MIN   = 30
    BVC_CLOSE_HOUR = 15
    BVC_CLOSE_MIN  = 30

    def __init__(self):
        from ..data.fetcher import BVCDataFetcher
        self._fetcher = BVCDataFetcher()
        self._cache: Dict[str, RealTimeQuote] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._cache_seconds = 30   # TTL du cache en secondes

    def _resolve_yahoo(self, symbol: str) -> str:
        return self._fetcher._resolve_yahoo_ticker(symbol)

    def is_market_open(self) -> bool:
        """Vérifie si la BVC est actuellement ouverte."""
        import pytz
        try:
            tz = pytz.timezone("Africa/Casablanca")
            now = datetime.now(tz)
        except Exception:
            now = datetime.utcnow()
        if now.weekday() >= 5:   # samedi=5, dimanche=6
            return False
        h, m = now.hour, now.minute
        open_minutes  = self.BVC_OPEN_HOUR  * 60 + self.BVC_OPEN_MIN
        close_minutes = self.BVC_CLOSE_HOUR * 60 + self.BVC_CLOSE_MIN
        current_minutes = h * 60 + m
        return open_minutes <= current_minutes <= close_minutes

    def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[RealTimeQuote]:
        """
        Retourne la cotation instantanée d'un symbole.

        Args:
            symbol   : Symbole BVC (ex: "ATW") ou Yahoo (ex: "ATW.CS")
            use_cache: Utiliser le cache (défaut: True)

        Returns:
            RealTimeQuote ou None en cas d'erreur
        """
        import yfinance as yf

        symbol_up = symbol.upper()
        now = time.time()

        # Cache
        if use_cache and symbol_up in self._cache:
            if now - self._cache_ttl.get(symbol_up, 0) < self._cache_seconds:
                return self._cache[symbol_up]

        yahoo_sym = self._resolve_yahoo(symbol)
        from ..data.tickers import get_ticker_info
        info_bvc = get_ticker_info(symbol) or {}

        try:
            ticker = yf.Ticker(yahoo_sym)
            fi     = ticker.fast_info

            price     = float(fi.get("last_price", 0) or fi.get("regular_market_price", 0) or 0)
            prev_close= float(fi.get("previous_close", 0) or 0)
            open_     = float(fi.get("open", 0) or 0)
            high      = float(fi.get("day_high", 0) or 0)
            low       = float(fi.get("day_low", 0) or 0)
            volume    = int(fi.get("last_volume", 0) or fi.get("regular_market_volume", 0) or 0)
            avg_vol   = int(fi.get("three_month_average_volume", 0) or 0)
            high52    = float(fi.get("year_high", 0) or 0)
            low52     = float(fi.get("year_low", 0) or 0)
            mkt_cap   = fi.get("market_cap", None)

            if price == 0 and prev_close == 0:
                # Fallback : dernière clôture via history
                hist = ticker.history(period="2d")
                if not hist.empty:
                    price     = float(hist["Close"].iloc[-1])
                    prev_close= float(hist["Close"].iloc[-2]) if len(hist) > 1 else price

            change     = price - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0

            name = info_bvc.get("name", symbol_up)

            q = RealTimeQuote(
                symbol     = symbol_up,
                name       = name,
                price      = round(price, 2),
                open_      = round(open_, 2),
                high       = round(high, 2),
                low        = round(low, 2),
                prev_close = round(prev_close, 2),
                change     = round(change, 2),
                change_pct = round(change_pct, 2),
                volume     = volume,
                avg_volume = avg_vol,
                high_52w   = round(high52, 2),
                low_52w    = round(low52, 2),
                market_cap = mkt_cap,
                timestamp  = datetime.now(),
                source     = "yahoo",
                is_live    = self.is_market_open(),
            )

            self._cache[symbol_up] = q
            self._cache_ttl[symbol_up] = now
            return q

        except Exception as e:
            logger.error(f"Erreur get_quote({symbol}): {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, RealTimeQuote]:
        """Retourne les cotations de plusieurs symboles en parallèle."""
        import concurrent.futures
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(self.get_quote, s): s for s in symbols}
            for fut in concurrent.futures.as_completed(futures):
                sym = futures[fut]
                try:
                    q = fut.result()
                    if q: results[sym.upper()] = q
                except Exception as e:
                    logger.warning(f"Erreur {sym}: {e}")
        return results

    def get_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Données intraday pour la séance en cours (ou les N derniers jours).

        Args:
            symbol  : Symbole BVC
            interval: "1m","2m","5m","15m","30m","60m","90m"
            period  : "1d","5d","1mo"  (max 60j pour les minutes)

        Returns:
            DataFrame OHLCV avec index DatetimeIndex
        """
        import yfinance as yf

        valid_intervals = ["1m","2m","5m","15m","30m","60m","90m","1h"]
        if interval not in valid_intervals:
            raise ValueError(f"Intervalle invalide: {interval}. Valeurs: {valid_intervals}")

        yahoo_sym = self._resolve_yahoo(symbol)
        ticker    = yf.Ticker(yahoo_sym)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"Pas de données intraday pour {symbol}")
            return pd.DataFrame()

        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.attrs["symbol"] = symbol.upper()
        from ..data.tickers import get_ticker_info
        info = get_ticker_info(symbol)
        df.attrs["name"]   = info.get("name", symbol) if info else symbol
        df.attrs["interval"] = interval
        return df

    def get_market_snapshot(self, top_n: int = 20) -> pd.DataFrame:
        """
        Tableau de bord de marché instantané (les N principales valeurs BVC).

        Returns:
            DataFrame trié par variation décroissante
        """
        from ..data.tickers import BVC_TICKERS
        majors = [k for k, v in BVC_TICKERS.items() if v.get("secteur") != "Indice"][:top_n]
        # Ajouter le MASI
        majors.insert(0, "MASI")

        quotes = self.get_quotes(majors)
        rows = []
        for sym, q in quotes.items():
            rows.append({
                "Symbole":      q.symbol,
                "Nom":          q.name[:30],
                "Cours (MAD)":  q.price,
                "Variation %":  q.change_pct,
                "Variation":    q.change,
                "Volume":       q.volume,
                "52s Haut":     q.high_52w,
                "52s Bas":      q.low_52w,
                "Heure":        q.timestamp.strftime("%H:%M:%S"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Variation %", ascending=False)
        return df

    def stream(
        self,
        symbols: List[str],
        callback: Callable[[Dict[str, RealTimeQuote]], None],
        interval: int = 30,
        max_iterations: int = None,
    ):
        """
        Surveille les symboles en boucle et appelle callback à chaque refresh.

        Args:
            symbols       : Liste de symboles à surveiller
            callback      : Fonction appelée avec dict {symbol: RealTimeQuote}
            interval      : Intervalle de rafraîchissement en secondes (défaut: 30)
            max_iterations: Nombre max d'itérations (None = infini)

        Exemple:
            def on_update(quotes):
                for sym, q in quotes.items():
                    print(q)
            rt.stream(["ATW","IAM"], on_update, interval=30)
        """
        iteration = 0
        print(f"Surveillance de {len(symbols)} symbole(s) toutes les {interval}s. Ctrl+C pour arrêter.")
        try:
            while True:
                quotes = self.get_quotes(symbols)
                callback(quotes)
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nSurveillance arrêtée.")

    def stream_async(
        self,
        symbols: List[str],
        callback: Callable,
        interval: int = 30,
    ) -> threading.Thread:
        """
        Lance la surveillance dans un thread séparé (non-bloquant).

        Returns:
            Thread actif (appeler .stop_event.set() pour arrêter)
        """
        stop_event = threading.Event()

        def _run():
            while not stop_event.is_set():
                try:
                    quotes = self.get_quotes(symbols)
                    callback(quotes)
                except Exception as e:
                    logger.warning(f"Erreur stream: {e}")
                stop_event.wait(timeout=interval)

        t = threading.Thread(target=_run, daemon=True)
        t.stop_event = stop_event
        t.start()
        return t


# ─────────────────────────────────────────────────────────────────
# BVC SCRAPER — scraping des sites publics BVC
# ─────────────────────────────────────────────────────────────────

class BVCScraper:
    """
    Scrape les cours en direct depuis les sites publics de la BVC.

    Sources tentées (par ordre de priorité) :
      1. leboursier.ma — cours en direct
      2. casablanca-bourse.com — site officiel
      3. Fallback sur Yahoo Finance si tout échoue

    Note: Le scraping peut échouer si les sites changent leur structure HTML.
    Utiliser RealTimeFetcher comme alternative fiable.
    """

    SOURCES = [
        "https://www.leboursier.ma/api?method=getStockQuote&NumberOfResult=100&search=",
        "https://www.casablanca-bourse.com/bourseweb/Negociation.aspx",
    ]

    def __init__(self, fallback_fetcher: RealTimeFetcher = None):
        self._fallback = fallback_fetcher or RealTimeFetcher()
        self._session  = None

    def _get_session(self):
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/html, */*",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                "Referer": "https://www.leboursier.ma/",
            })
        return self._session

    def scrape_leboursier(self, symbol: str) -> Optional[RealTimeQuote]:
        """
        Tente de récupérer un cours depuis leboursier.ma.
        """
        try:
            session = self._get_session()
            url = f"https://www.leboursier.ma/api?method=getStockQuote&NumberOfResult=1&search={symbol}"
            resp = session.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            if not data or not isinstance(data, list):
                return None

            row = data[0]
            price     = float(row.get("currentValue", 0) or 0)
            prev      = float(row.get("previousClose", 0) or 0)
            change    = price - prev
            change_pct = (change / prev * 100) if prev else 0

            from ..data.tickers import get_ticker_info
            info = get_ticker_info(symbol) or {}

            return RealTimeQuote(
                symbol     = symbol.upper(),
                name       = row.get("name", info.get("name", symbol)),
                price      = round(price, 2),
                open_      = float(row.get("openValue", price) or price),
                high       = float(row.get("highValue", price) or price),
                low        = float(row.get("lowValue", price) or price),
                prev_close = round(prev, 2),
                change     = round(change, 2),
                change_pct = round(change_pct, 2),
                volume     = int(row.get("volume", 0) or 0),
                avg_volume = 0,
                high_52w   = float(row.get("high52w", 0) or 0),
                low_52w    = float(row.get("low52w", 0) or 0),
                market_cap = None,
                timestamp  = datetime.now(),
                source     = "leboursier.ma",
                is_live    = True,
            )
        except Exception as e:
            logger.debug(f"Scraping leboursier.ma échoué pour {symbol}: {e}")
            return None

    def scrape_all_leboursier(self) -> Dict[str, RealTimeQuote]:
        """Récupère tous les cours disponibles sur leboursier.ma."""
        try:
            session = self._get_session()
            url = "https://www.leboursier.ma/api?method=getStockQuote&NumberOfResult=200&search="
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = {}
            for row in data:
                try:
                    sym = row.get("ticker", "").upper()
                    if not sym: continue
                    price     = float(row.get("currentValue", 0) or 0)
                    prev      = float(row.get("previousClose", 0) or 0)
                    change    = price - prev
                    chg_pct   = (change / prev * 100) if prev else 0
                    results[sym] = RealTimeQuote(
                        symbol     = sym,
                        name       = row.get("name", sym),
                        price      = round(price, 2),
                        open_      = float(row.get("openValue", price) or price),
                        high       = float(row.get("highValue", price) or price),
                        low        = float(row.get("lowValue", price) or price),
                        prev_close = round(prev, 2),
                        change     = round(change, 2),
                        change_pct = round(chg_pct, 2),
                        volume     = int(row.get("volume", 0) or 0),
                        avg_volume = 0,
                        high_52w   = float(row.get("high52w", 0) or 0),
                        low_52w    = float(row.get("low52w", 0) or 0),
                        market_cap = None,
                        timestamp  = datetime.now(),
                        source     = "leboursier.ma",
                        is_live    = True,
                    )
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.warning(f"Scraping global leboursier.ma échoué: {e}")
            return {}

    def get_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """
        Récupère un cours en essayant toutes les sources par ordre.
        Fallback sur Yahoo Finance si le scraping échoue.
        """
        q = self.scrape_leboursier(symbol)
        if q and q.price > 0:
            return q
        logger.debug(f"Scraping échoué pour {symbol}, fallback Yahoo")
        return self._fallback.get_quote(symbol)

    def get_market_live(self) -> Dict[str, RealTimeQuote]:
        """
        Récupère l'ensemble des cours de la BVC depuis leboursier.ma.
        Fallback sur Yahoo pour les symboles manquants.
        """
        all_quotes = self.scrape_all_leboursier()
        if all_quotes:
            return all_quotes
        # Fallback complet
        logger.warning("Scraping global échoué, fallback Yahoo Finance")
        return self._fallback.get_quotes(list(
            __import__("src.data.tickers", fromlist=["BVC_TICKERS"]).BVC_TICKERS.keys()
        ))
