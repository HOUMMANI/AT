"""
Dashboard en temps réel avec auto-rafraîchissement.

Deux modes :
  1. LiveDashboard    — graphique matplotlib qui se met à jour automatiquement
                         (données intraday 5min + cotation instantanée)
  2. LiveTicker       — affichage terminal en mode texte (style Bloomberg)

Usage :
    from src.visualization.live_dashboard import LiveDashboard, LiveTicker

    # Graphique live (bloquant, Ctrl+C pour arrêter)
    dash = LiveDashboard("ATW", refresh=60)
    dash.run()

    # Ticker terminal multi-valeurs
    ticker = LiveTicker(["ATW","IAM","BCP","MNG"], refresh=30)
    ticker.run()
"""

import warnings
warnings.filterwarnings("ignore")

import time
import threading
import os
import sys
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.animation as animation
import logging

from .charts import DARK_THEME, _apply_dark_theme, _plot_candlesticks

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LIVE DASHBOARD — graphique auto-rafraîchi
# ─────────────────────────────────────────────

class LiveDashboard:
    """
    Dashboard graphique en temps réel pour une action BVC.

    Affiche (mis à jour toutes les N secondes) :
    ┌──────────────────────┬────────────────────┐
    │  Bougies intraday    │  Cotation live      │
    │  (5min, séance)      │  + variation        │
    │                      │  + volume           │
    ├──────────────────────┼────────────────────┤
    │  RSI + MACD intraday │  Carnet des ordres  │
    │                      │  (simulé)           │
    └──────────────────────┴────────────────────┘

    Args:
        symbol : Symbole BVC (ex: "ATW")
        refresh: Intervalle de rafraîchissement en secondes (défaut: 60)
        interval: Intervalle des bougies intraday (défaut: "5m")
        figsize: Taille de la figure
    """

    def __init__(
        self,
        symbol: str,
        refresh: int = 60,
        interval: str = "5m",
        figsize: tuple = (18, 10),
        dark_theme: bool = True,
    ):
        self.symbol   = symbol.upper()
        self.refresh  = refresh
        self.interval = interval
        self.figsize  = figsize
        self.dark     = dark_theme
        self.c        = DARK_THEME if dark_theme else {}
        self._running = False
        self._df_intraday: Optional[pd.DataFrame] = None
        self._quote = None

        from ..data.realtime import RealTimeFetcher
        self._rt = RealTimeFetcher()

    def _fetch_all(self):
        """Récupère les données (thread séparé)."""
        try:
            self._df_intraday = self._rt.get_intraday(
                self.symbol, interval=self.interval, period="1d"
            )
            self._quote = self._rt.get_quote(self.symbol, use_cache=False)
        except Exception as e:
            logger.warning(f"Erreur fetch live {self.symbol}: {e}")

    def _build_figure(self):
        """Crée la figure et les axes."""
        c = self.c
        fig = plt.figure(figsize=self.figsize, facecolor=c.get("bg","#0d1117"))
        gs  = gridspec.GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[3, 1.5],
            width_ratios=[3, 1],
            hspace=0.10, wspace=0.12,
            top=0.92, bottom=0.06, left=0.05, right=0.97,
        )
        ax_price  = fig.add_subplot(gs[0, 0])   # Bougies
        ax_info   = fig.add_subplot(gs[0, 1])   # Cotation live
        ax_indic  = fig.add_subplot(gs[1, 0])   # RSI/MACD
        ax_stats  = fig.add_subplot(gs[1, 1])   # Stats
        _apply_dark_theme(fig, [ax_price, ax_info, ax_indic, ax_stats])
        return fig, ax_price, ax_info, ax_indic, ax_stats

    def _draw_frame(self, fig, ax_price, ax_info, ax_indic, ax_stats):
        """Dessine un frame complet du dashboard."""
        c  = self.c
        df = self._df_intraday
        q  = self._quote

        for ax in [ax_price, ax_info, ax_indic, ax_stats]:
            ax.cla()
            ax.set_facecolor(c.get("ax_bg","#161b22"))

        now_str = datetime.now().strftime("%H:%M:%S")
        sym = self.symbol
        from ..data.tickers import get_ticker_info
        name = (get_ticker_info(sym) or {}).get("name", sym)

        # ── Panel 1 : Bougies intraday ──
        if df is not None and not df.empty:
            _plot_candlesticks(ax_price, df, c)

            # EMA 9 et 20
            from ..indicators.trend import ema as calc_ema
            close = df["Close"]
            e9  = calc_ema(close, 9)
            e20 = calc_ema(close, 20)
            ax_price.plot(df.index, e9,  color="#f0b429", lw=1.0, label="EMA9",  alpha=0.85)
            ax_price.plot(df.index, e20, color="#58a6ff", lw=1.0, label="EMA20", alpha=0.85)

            # Titre avec le cours live
            if q and q.price > 0:
                chg_clr = c.get("up","#26a641") if q.change_pct >= 0 else c.get("down","#da3633")
                arrow   = "▲" if q.change_pct >= 0 else "▼"
                ax_price.set_title(
                    f"{sym} — {name}   |   {q.price:.2f} MAD  "
                    f"{arrow}{abs(q.change_pct):.2f}%   [{self.interval}]  ·  {now_str}",
                    fontsize=11, color=c.get("text","#c9d1d9"), pad=5,
                )
            else:
                ax_price.set_title(f"{sym} — {name}  [{self.interval}]  ·  {now_str}",
                                   fontsize=11, color=c.get("text","#c9d1d9"))

            ax_price.legend(loc="upper left", fontsize=7.5,
                            facecolor=c.get("ax_bg","#161b22"),
                            edgecolor=c.get("grid","#21262d"),
                            labelcolor=c.get("text","#c9d1d9"))
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_price.set_ylabel("Prix (MAD)", color=c.get("text","#c9d1d9"), fontsize=8)
            plt.setp(ax_price.get_xticklabels(), rotation=25, ha="right", fontsize=7)
            ax_price.grid(True, color=c.get("grid","#21262d"), lw=0.4, alpha=0.6)

        else:
            ax_price.text(0.5, 0.5,
                          f"{sym}\nEn attente de données intraday...\n"
                          f"(BVC ouverte 09h30–15h30)",
                          ha="center", va="center",
                          fontsize=10, color=c.get("text","#c9d1d9"),
                          transform=ax_price.transAxes)

        # ── Panel 2 : Cotation live ──
        ax_info.axis("off")
        if q and q.price > 0:
            chg_clr = c.get("up","#26a641") if q.change >= 0 else c.get("down","#da3633")
            arrow   = "▲" if q.change >= 0 else "▼"
            live_lbl = "  LIVE" if q.is_live else "  CLÔTURE"
            live_clr = c.get("up","#26a641") if q.is_live else "#888"

            # Cours principal
            ax_info.text(0.5, 0.92, f"{q.price:.2f}", ha="center", va="center",
                         fontsize=24, color=chg_clr, fontweight="bold",
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.80, f"MAD", ha="center", va="center",
                         fontsize=10, color=c.get("text","#c9d1d9"),
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.70, f"{arrow} {q.change:+.2f}  ({q.change_pct:+.2f}%)",
                         ha="center", va="center", fontsize=12,
                         color=chg_clr, fontweight="bold",
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.61, live_lbl, ha="center", va="center",
                         fontsize=8, color=live_clr,
                         transform=ax_info.transAxes)

            # Stats
            stats = [
                ("Ouverture", f"{q.open_:.2f}"),
                ("Haut",      f"{q.high:.2f}"),
                ("Bas",       f"{q.low:.2f}"),
                ("Clôture préc.", f"{q.prev_close:.2f}"),
                ("Volume",    f"{q.volume:,}"),
                ("52s Haut",  f"{q.high_52w:.2f}"),
                ("52s Bas",   f"{q.low_52w:.2f}"),
                ("Heure",     q.timestamp.strftime("%H:%M:%S")),
            ]
            y_s = 0.50
            for label, value in stats:
                ax_info.text(0.10, y_s, label, ha="left", va="center",
                             fontsize=7.5, color=c.get("text","#c9d1d9"),
                             alpha=0.7, transform=ax_info.transAxes)
                ax_info.text(0.90, y_s, value, ha="right", va="center",
                             fontsize=7.5, color=c.get("text","#c9d1d9"),
                             fontweight="bold", transform=ax_info.transAxes)
                ax_info.axhline(y_s - 0.025, xmin=0.08, xmax=0.92,
                                color=c.get("grid","#21262d"), lw=0.4,
                                transform=ax_info.transAxes)
                y_s -= 0.052
        else:
            ax_info.text(0.5, 0.5, "Cotation\nindisponible",
                         ha="center", va="center",
                         fontsize=10, color=c.get("text","#c9d1d9"),
                         transform=ax_info.transAxes)

        # ── Panel 3 : RSI + MACD intraday ──
        if df is not None and not df.empty and len(df) >= 15:
            try:
                from ..indicators.momentum import rsi as calc_rsi
                from ..indicators.trend import macd as calc_macd
                close = df["Close"]
                rsi_v = calc_rsi(close, min(14, len(close)//2))
                macd_df = calc_macd(close)

                # RSI
                color_rsi = c.get("rsi","#58a6ff")
                color_up  = c.get("up","#26a641")
                color_dn  = c.get("down","#da3633")
                ax_indic.plot(df.index, rsi_v, color=color_rsi, lw=1.2, label="RSI(14)")
                ax_indic.axhline(70, color=color_dn, lw=0.7, linestyle="--", alpha=0.7)
                ax_indic.axhline(30, color=color_up, lw=0.7, linestyle="--", alpha=0.7)
                ax_indic.fill_between(df.index, rsi_v, 70, where=rsi_v >= 70,
                                      alpha=0.2, color=color_dn)
                ax_indic.fill_between(df.index, rsi_v, 30, where=rsi_v <= 30,
                                      alpha=0.2, color=color_up)
                ax_indic.set_ylim(0, 100)
                ax_indic.set_ylabel("RSI", color=c.get("text","#c9d1d9"), fontsize=7)
                ax_indic.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                plt.setp(ax_indic.get_xticklabels(), rotation=25, ha="right", fontsize=6.5)
                ax_indic.grid(True, color=c.get("grid","#21262d"), lw=0.4, alpha=0.5)
                rsi_now = rsi_v.iloc[-1]
                rsi_sig = "SURACHAT" if rsi_now > 70 else "SURVENTE" if rsi_now < 30 else "NEUTRE"
                rsi_clr = color_dn if rsi_now > 70 else color_up if rsi_now < 30 else c.get("text","#c9d1d9")
                ax_indic.text(0.99, 0.95, f"RSI {rsi_now:.1f} — {rsi_sig}",
                              ha="right", va="top", fontsize=7.5, color=rsi_clr,
                              fontweight="bold", transform=ax_indic.transAxes)
            except Exception as e:
                ax_indic.text(0.5, 0.5, f"RSI/MACD\n{e}", ha="center", va="center",
                              color=c.get("text","#c9d1d9"), transform=ax_indic.transAxes, fontsize=8)
        else:
            ax_indic.text(0.5, 0.5, "Données intraday\ninsuffisantes",
                          ha="center", va="center",
                          fontsize=9, color=c.get("text","#c9d1d9"),
                          transform=ax_indic.transAxes)

        # ── Panel 4 : Stats + badge marché ──
        ax_stats.axis("off")
        mkt_open = self._rt.is_market_open()
        badge_clr = c.get("up","#26a641") if mkt_open else "#555"
        badge_txt = "MARCHÉ OUVERT" if mkt_open else "MARCHÉ FERMÉ"
        ax_stats.text(0.5, 0.94, badge_txt, ha="center", va="center",
                      fontsize=9, color="white", fontweight="bold",
                      transform=ax_stats.transAxes,
                      bbox=dict(boxstyle="round,pad=0.4", fc=badge_clr, ec="none"))
        ax_stats.text(0.5, 0.84, "BVC · 09h30–15h30",
                      ha="center", va="center", fontsize=7.5,
                      color=c.get("text","#c9d1d9"), alpha=0.7,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.5, 0.75, f"Prochain refresh: {self.refresh}s",
                      ha="center", va="center", fontsize=7,
                      color=c.get("text","#c9d1d9"), alpha=0.6,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.5, 0.65, f"Source: Yahoo Finance",
                      ha="center", va="center", fontsize=7,
                      color=c.get("text","#c9d1d9"), alpha=0.5,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.5, 0.55, "Délai: ~15-20 min",
                      ha="center", va="center", fontsize=7,
                      color="#f0b429", alpha=0.7,
                      transform=ax_stats.transAxes)

        if df is not None and not df.empty:
            vol_ratio = ""
            try:
                avg_vol = df["Volume"].mean()
                last_vol = df["Volume"].iloc[-1]
                ratio = last_vol / avg_vol if avg_vol > 0 else 1
                vol_ratio = f"RVOL: {ratio:.1f}x"
            except Exception:
                pass

            ax_stats.text(0.5, 0.42, vol_ratio, ha="center", va="center",
                          fontsize=9, color=c.get("up","#26a641") if "2." in vol_ratio else c.get("text","#c9d1d9"),
                          fontweight="bold", transform=ax_stats.transAxes)

        fig.patch.set_facecolor(c.get("bg","#0d1117"))

    def run(self):
        """
        Lance le dashboard live (bloquant).
        Utilise matplotlib.animation pour le rafraîchissement.
        Appuyer sur 'q' ou fermer la fenêtre pour arrêter.
        """
        # Fetch initial
        print(f"Chargement des données pour {self.symbol}...")
        self._fetch_all()

        fig, ax_price, ax_info, ax_indic, ax_stats = self._build_figure()

        def _update(frame):
            # Fetch en arrière-plan
            t = threading.Thread(target=self._fetch_all, daemon=True)
            t.start()
            self._draw_frame(fig, ax_price, ax_info, ax_indic, ax_stats)
            return []

        # Premier dessin immédiat
        self._draw_frame(fig, ax_price, ax_info, ax_indic, ax_stats)

        ani = animation.FuncAnimation(
            fig, _update,
            interval=self.refresh * 1000,   # en millisecondes
            cache_frame_data=False,
            blit=False,
        )

        plt.tight_layout()
        print(f"Dashboard {self.symbol} lancé. Rafraîchissement toutes les {self.refresh}s.")
        print("Fermer la fenêtre ou Ctrl+C pour arrêter.")
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            plt.close(fig)

    def save_snapshot(self, path: str = None):
        """Sauvegarde un snapshot instantané du dashboard."""
        self._fetch_all()
        fig, ax_price, ax_info, ax_indic, ax_stats = self._build_figure()
        self._draw_frame(fig, ax_price, ax_info, ax_indic, ax_stats)
        path = path or f"{self.symbol}_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Snapshot sauvegardé: {path}")
        plt.close(fig)
        return path


# ─────────────────────────────────────────────
# LIVE TICKER — terminal texte style Bloomberg
# ─────────────────────────────────────────────

class LiveTicker:
    """
    Tableau de bord texte en temps réel dans le terminal.
    Affiche les cours de plusieurs actions BVC avec couleurs ANSI.

    Args:
        symbols : Liste de symboles à surveiller
        refresh : Intervalle de rafraîchissement en secondes
        source  : "yahoo" (défaut) ou "scraper" (leboursier.ma)
    """

    # Codes ANSI
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    CLEAR  = "\033[2J\033[H"

    def __init__(
        self,
        symbols: List[str],
        refresh: int = 30,
        source: str = "yahoo",
    ):
        self.symbols = [s.upper() for s in symbols]
        self.refresh = refresh
        self.source  = source

        if source == "scraper":
            from ..data.realtime import BVCScraper
            self._rt = BVCScraper()
        else:
            from ..data.realtime import RealTimeFetcher
            self._rt = RealTimeFetcher()

        self._prev_quotes = {}

    def _color_val(self, val: float, neutral: bool = False) -> str:
        if neutral: return self.WHITE
        if val > 0: return self.GREEN
        if val < 0: return self.RED
        return self.GRAY

    def _format_row(self, q) -> str:
        arrow  = "▲" if q.change_pct >= 0 else "▼"
        clr    = self.GREEN if q.change_pct >= 0 else self.RED
        live   = f"{self.CYAN}[LIVE]{self.RESET}" if q.is_live else f"{self.GRAY}[CLÔ]{self.RESET}"
        vol_str = f"{q.volume:>10,}" if q.volume else "          —"
        return (
            f"{self.BOLD}{q.symbol:<8}{self.RESET}"
            f"{q.name[:24]:<24} "
            f"{self.WHITE}{q.price:>9.2f} MAD{self.RESET}  "
            f"{clr}{arrow}{abs(q.change_pct):>5.2f}%  {q.change:>+7.2f}{self.RESET}  "
            f"Vol: {vol_str}  {live}"
        )

    def _print_table(self, quotes: dict):
        """Affiche le tableau mis en forme."""
        from ..data.realtime import RealTimeFetcher
        is_open = RealTimeFetcher().is_market_open()
        market_status = (f"{self.GREEN}● MARCHÉ OUVERT{self.RESET}"
                         if is_open else f"{self.GRAY}● MARCHÉ FERMÉ (09h30–15h30){self.RESET}")

        print(self.CLEAR, end="")
        print(f"{self.BOLD}{self.CYAN}══ BVC — COTATIONS EN TEMPS RÉEL ══  {market_status}{self.RESET}")
        print(f"{self.GRAY}Mis à jour: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  "
              f"│  Source: {'leboursier.ma' if self.source=='scraper' else 'Yahoo Finance (~15min de délai)'}  "
              f"│  Refresh: {self.refresh}s{self.RESET}")
        print(f"{self.GRAY}{'─'*100}{self.RESET}")
        print(f"  {self.BOLD}{'Symb':<8}{'Nom':<24}{'Cours':>12}  {'Var%':>7}  {'Var MAD':>8}  {'Volume':>14}  Statut{self.RESET}")
        print(f"{self.GRAY}{'─'*100}{self.RESET}")

        if not quotes:
            print(f"  {self.YELLOW}Aucune donnée disponible...{self.RESET}")
        else:
            for sym in self.symbols:
                q = quotes.get(sym)
                if q:
                    print("  " + self._format_row(q))
                else:
                    print(f"  {self.GRAY}{sym:<8}{'—':>24}  données indisponibles{self.RESET}")

        print(f"{self.GRAY}{'─'*100}{self.RESET}")
        print(f"  {self.GRAY}Ctrl+C pour arrêter{self.RESET}")

    def run(self):
        """Lance le ticker (bloquant). Ctrl+C pour arrêter."""
        print(f"Démarrage du ticker pour {len(self.symbols)} symbole(s)...")
        try:
            while True:
                quotes = self._rt.get_quotes(self.symbols) \
                    if hasattr(self._rt, "get_quotes") \
                    else {s: self._rt.get_quote(s) for s in self.symbols}

                self._print_table(quotes)
                time.sleep(self.refresh)
        except KeyboardInterrupt:
            print(f"\n{self.RESET}Ticker arrêté.")

    def run_once(self) -> dict:
        """Effectue un seul refresh et retourne les cotations."""
        quotes = self._rt.get_quotes(self.symbols) \
            if hasattr(self._rt, "get_quotes") \
            else {s: self._rt.get_quote(s) for s in self.symbols}
        self._print_table(quotes)
        return quotes
