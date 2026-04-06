"""
Visualisation des données de marché et indicateurs techniques.
Utilise matplotlib pour les graphiques statiques et plotly pour l'interactivité.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")


# Thème sombre par défaut
DARK_THEME = {
    "bg": "#0d1117",
    "ax_bg": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "up": "#26a641",       # vert haussier
    "down": "#da3633",     # rouge baissier
    "sma20": "#f0b429",    # jaune
    "sma50": "#58a6ff",    # bleu
    "sma200": "#bc8cff",   # violet
    "ema9": "#ff7b72",     # orange-rouge
    "macd": "#58a6ff",
    "signal": "#ff7b72",
    "hist_up": "#26a641",
    "hist_down": "#da3633",
    "rsi": "#58a6ff",
    "bb_upper": "#f0b429",
    "bb_lower": "#f0b429",
    "bb_fill": "#f0b42915",
    "volume": "#30363d",
    "volume_up": "#26a64180",
    "volume_down": "#da363380",
}


def _apply_dark_theme(fig, axes):
    """Applique le thème sombre à tous les axes."""
    c = DARK_THEME
    fig.patch.set_facecolor(c["bg"])
    for ax in axes:
        ax.set_facecolor(c["ax_bg"])
        ax.tick_params(colors=c["text"])
        ax.xaxis.label.set_color(c["text"])
        ax.yaxis.label.set_color(c["text"])
        ax.title.set_color(c["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(c["grid"])
        ax.grid(True, color=c["grid"], linewidth=0.5, alpha=0.7)


def plot_patterns(
    ax,
    df: pd.DataFrame,
    show_candlestick_patterns: bool = True,
    show_chart_patterns: bool = True,
    show_trendlines: bool = True,
    show_fibonacci: bool = True,
    dark_theme: bool = True,
):
    """
    Superpose les patterns et configurations graphiques sur un axe matplotlib.

    Args:
        ax: Axe matplotlib (graphique principal)
        df: DataFrame OHLCV
        show_candlestick_patterns: Afficher les marqueurs de patterns bougies
        show_chart_patterns: Afficher les zones de configurations graphiques
        show_trendlines: Afficher les lignes de tendance
        show_fibonacci: Afficher les niveaux Fibonacci
        dark_theme: Thème sombre
    """
    c = DARK_THEME if dark_theme else {}
    up_color = c.get("up", "#26a641")
    down_color = c.get("down", "#da3633")
    text_color = c.get("text", "#c9d1d9")
    grid_color = c.get("grid", "#21262d")

    # --- Patterns de bougies japonaises ---
    if show_candlestick_patterns:
        try:
            from ..patterns.candlesticks import CandlestickPatterns
            cp = CandlestickPatterns(df)
            recent = cp.get_recent(lookback=20)

            bullish_markers = []
            bearish_markers = []

            for p in recent:
                date = p["date"]
                if date not in df.index:
                    continue
                price = df.loc[date, "Low"] if p["direction"] == "HAUSSIER" else df.loc[date, "High"]
                if p["direction"] == "HAUSSIER":
                    bullish_markers.append((date, price))
                elif p["direction"] == "BAISSIER":
                    bearish_markers.append((date, price))

            # Triangles haussiers (pointe vers le haut) sous les bougies
            if bullish_markers:
                dates, prices = zip(*bullish_markers)
                offset = df["High"].mean() * 0.015
                ax.scatter(dates, [p - offset for p in prices],
                           marker="^", color=up_color, s=80, zorder=5, alpha=0.9,
                           label="Pattern haussier")

            # Triangles baissiers (pointe vers le bas) au-dessus des bougies
            if bearish_markers:
                dates, prices = zip(*bearish_markers)
                offset = df["High"].mean() * 0.015
                ax.scatter(dates, [p + offset for p in prices],
                           marker="v", color=down_color, s=80, zorder=5, alpha=0.9,
                           label="Pattern baissier")

            # Annotations pour les patterns les plus récents (max 3)
            shown = 0
            for p in recent[:5]:
                if shown >= 3:
                    break
                date = p["date"]
                if date not in df.index:
                    continue
                name = p["pattern"]
                if len(name) > 16:
                    name = name[:14] + ".."
                price = df.loc[date, "Close"]
                offset = df["High"].mean() * 0.03
                y = price - offset if p["direction"] == "HAUSSIER" else price + offset
                va = "top" if p["direction"] == "HAUSSIER" else "bottom"
                ax.annotate(
                    name,
                    xy=(date, price),
                    xytext=(date, y),
                    fontsize=7,
                    color=up_color if p["direction"] == "HAUSSIER" else down_color,
                    ha="center", va=va,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=c.get("ax_bg", "#161b22"), alpha=0.7, ec="none"),
                )
                shown += 1
        except Exception:
            pass

    # --- Lignes de tendance ---
    if show_trendlines:
        try:
            from ..patterns.trendlines import TrendlineDetector
            td = TrendlineDetector(df)
            lines = td.detect_all()

            tl_colors = {
                ("HAUSSIERE", "SUPPORT"): up_color,
                ("BAISSIERE", "RESISTANCE"): down_color,
                ("HORIZONTALE", "SUPPORT"): "#58a6ff",
                ("HORIZONTALE", "RESISTANCE"): "#f0b429",
            }

            x_all = np.arange(len(df))
            plotted = 0
            for tl in lines[:6]:
                if tl.force in ("Faible",) and plotted >= 3:
                    continue
                color = tl_colors.get((tl.type, tl.direction), "#888888")
                linestyle = "--" if tl.type == "HORIZONTALE" else "-"
                alpha = 0.8 if tl.force in ("Forte", "Très forte") else 0.5

                if tl.type == "HORIZONTALE":
                    ax.axhline(
                        y=tl.prix_actuel,
                        color=color, linewidth=1.0,
                        linestyle=linestyle, alpha=alpha,
                        label=f"S/R {tl.prix_actuel:.2f}"
                    )
                    ax.annotate(
                        f"{tl.direction[0]}{tl.prix_actuel:.1f}",
                        xy=(df.index[-1], tl.prix_actuel),
                        fontsize=7, color=color, alpha=0.9,
                        ha="right", va="center",
                    )
                else:
                    # Calculer les points de la ligne sur toute la période
                    if tl.date_debut in df.index and tl.date_fin in df.index:
                        i_start = df.index.get_loc(tl.date_debut)
                        i_end = len(df) - 1
                        slope = tl.pente
                        intercept = tl.prix_debut - slope * i_start
                        y_vals = slope * x_all[i_start:] + intercept
                        ax.plot(
                            df.index[i_start:], y_vals,
                            color=color, linewidth=1.2,
                            linestyle=linestyle, alpha=alpha,
                        )
                plotted += 1
        except Exception:
            pass

    # --- Niveaux Fibonacci ---
    if show_fibonacci:
        try:
            from ..patterns.fibonacci import FibonacciAnalyzer
            fib = FibonacciAnalyzer(df)
            analysis = fib.analyze()

            fib_colors = {
                0.0: "#888888",
                0.236: "#58a6ff",
                0.382: "#f0b429",
                0.500: "#ffffff",
                0.618: "#f0b429",
                0.786: "#58a6ff",
                1.0: "#888888",
            }

            current = df["Close"].iloc[-1]
            # Afficher seulement les niveaux proches du prix (±20%)
            for lvl in analysis.niveaux:
                if abs(lvl.distance_pct) > 20:
                    continue
                if lvl.type != "retracement":
                    continue
                color = fib_colors.get(lvl.ratio, "#aaaaaa")
                linestyle = ":" if lvl.ratio not in (0.382, 0.500, 0.618) else "-."
                ax.axhline(
                    y=lvl.prix,
                    color=color, linewidth=0.8,
                    linestyle=linestyle, alpha=0.6,
                )
                ax.annotate(
                    f"Fib {lvl.ratio*100:.1f}%  {lvl.prix:.1f}",
                    xy=(df.index[len(df)//4], lvl.prix),
                    fontsize=6.5, color=color, alpha=0.8,
                    ha="left", va="bottom",
                )
        except Exception:
            pass

    # --- Configurations graphiques ---
    if show_chart_patterns:
        try:
            from ..patterns.chart_patterns import ChartPatternDetector
            detector = ChartPatternDetector(df)
            patterns = detector.detect_all()

            for p in patterns[:3]:
                color = up_color if p.direction == "HAUSSIER" else down_color if p.direction == "BAISSIER" else "#aaaaaa"

                # Ligne de breakout
                ax.axhline(
                    y=p.niveau_breakout,
                    color=color, linewidth=1.5,
                    linestyle="-.", alpha=0.85,
                )
                ax.annotate(
                    f"▶ {p.nom}  Breakout: {p.niveau_breakout}",
                    xy=(df.index[-len(df)//5], p.niveau_breakout),
                    fontsize=7.5, color=color, fontweight="bold",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=c.get("ax_bg", "#161b22"), alpha=0.8, ec=color, lw=0.8),
                )

                # Objectif de prix
                if p.objectif_prix:
                    ax.axhline(
                        y=p.objectif_prix,
                        color=color, linewidth=0.8,
                        linestyle=":", alpha=0.5,
                    )
                    ax.annotate(
                        f"  Obj: {p.objectif_prix}",
                        xy=(df.index[-len(df)//5], p.objectif_prix),
                        fontsize=7, color=color, alpha=0.7,
                        ha="left", va="bottom",
                    )
        except Exception:
            pass


def plot_chart(
    df: pd.DataFrame,
    title: Optional[str] = None,
    show_volume: bool = True,
    show_sma: bool = True,
    show_bollinger: bool = True,
    show_macd: bool = True,
    show_rsi: bool = True,
    show_patterns: bool = True,
    show_trendlines: bool = True,
    show_fibonacci: bool = True,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
) -> plt.Figure:
    """
    Crée un graphique complet avec chandeliers japonais, indicateurs et patterns.

    Args:
        df: DataFrame OHLCV (avec indicateurs calculés si disponibles)
        title: Titre du graphique
        show_volume: Afficher le panneau volume
        show_bollinger: Afficher les bandes de Bollinger
        show_macd: Afficher le panneau MACD
        show_rsi: Afficher le panneau RSI
        show_patterns: Afficher les patterns de bougies et configurations graphiques
        show_trendlines: Afficher les lignes de tendance
        show_fibonacci: Afficher les niveaux Fibonacci
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder (optionnel)
        dark_theme: Utiliser le thème sombre

    Returns:
        Figure matplotlib
    """
    from ..analysis.analyzer import TechnicalAnalyzer
    from ..indicators.trend import sma as calc_sma, ema as calc_ema, macd as calc_macd
    from ..indicators.momentum import rsi as calc_rsi
    from ..indicators.volatility import bollinger_bands as calc_bb

    # Calculer les indicateurs si pas déjà présents
    if "SMA_20" not in df.columns:
        analyzer = TechnicalAnalyzer(df)
        df = analyzer.compute_all()

    c = DARK_THEME if dark_theme else {
        "bg": "white", "ax_bg": "white", "grid": "#e0e0e0",
        "text": "black", "up": "#26a641", "down": "#da3633",
        "sma20": "#f0b429", "sma50": "#2196F3", "sma200": "#9C27B0",
        "ema9": "#FF5722", "macd": "#2196F3", "signal": "#FF5722",
        "hist_up": "#26a641", "hist_down": "#da3633",
        "rsi": "#2196F3", "bb_upper": "#f0b429", "bb_lower": "#f0b429",
        "bb_fill": "#f0b42915", "volume": "#90A4AE",
        "volume_up": "#26a64180", "volume_down": "#da363380",
    }

    # Définir les panneaux
    n_panels = 1 + int(show_volume) + int(show_macd) + int(show_rsi)
    heights = [4]
    if show_volume:
        heights.append(1)
    if show_macd:
        heights.append(1.5)
    if show_rsi:
        heights.append(1.5)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=heights, hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    if dark_theme:
        _apply_dark_theme(fig, axes)

    ax_main = axes[0]
    panel_idx = 1

    # --- Chandeliers japonais ---
    _plot_candlesticks(ax_main, df, c)

    # --- Moyennes mobiles ---
    if show_sma and "SMA_20" in df.columns:
        ax_main.plot(df.index, df["SMA_20"], color=c["sma20"], linewidth=1, label="SMA 20", alpha=0.9)
        ax_main.plot(df.index, df["SMA_50"], color=c["sma50"], linewidth=1, label="SMA 50", alpha=0.9)
        if "SMA_200" in df.columns and df["SMA_200"].notna().sum() > 10:
            ax_main.plot(df.index, df["SMA_200"], color=c["sma200"], linewidth=1, label="SMA 200", alpha=0.9)

    # --- Bollinger Bands ---
    if show_bollinger and "BB_Haute" in df.columns:
        ax_main.plot(df.index, df["BB_Haute"], color=c["bb_upper"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_main.plot(df.index, df["BB_Basse"], color=c["bb_lower"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_main.fill_between(df.index, df["BB_Haute"], df["BB_Basse"], alpha=0.05, color=c["bb_upper"])

    # Titre et légende
    symbol = df.attrs.get("symbol", "")
    name = df.attrs.get("name", "")
    last_close = df["Close"].iloc[-1]
    chg = ((df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1) * 100
    chg_str = f"{chg:+.2f}%"
    chg_color = c["up"] if chg >= 0 else c["down"]

    chart_title = title or f"{symbol} - {name}"
    ax_main.set_title(
        f"{chart_title}   |   {last_close:.2f} MAD   {chg_str}",
        fontsize=13, color=c["text"], pad=10
    )
    ax_main.legend(loc="upper left", facecolor=c["ax_bg"], edgecolor=c["grid"],
                   labelcolor=c["text"], fontsize=8)
    ax_main.set_ylabel("Prix (MAD)", color=c["text"])
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- Volume ---
    if show_volume:
        ax_vol = axes[panel_idx]
        panel_idx += 1
        colors = [c["volume_up"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else c["volume_down"] for i in range(len(df))]
        ax_vol.bar(df.index, df["Volume"], color=colors, width=0.8)
        if "Volume_SMA_20" in df.columns:
            ax_vol.plot(df.index, df["Volume_SMA_20"], color=c["sma20"], linewidth=1, alpha=0.8)
        ax_vol.set_ylabel("Volume", color=c["text"], fontsize=8)
        plt.setp(ax_vol.get_xticklabels(), visible=False)

    # --- MACD ---
    if show_macd and "MACD" in df.columns:
        ax_macd = axes[panel_idx]
        panel_idx += 1
        ax_macd.plot(df.index, df["MACD"], color=c["macd"], linewidth=1.2, label="MACD")
        ax_macd.plot(df.index, df["MACD_Signal"], color=c["signal"], linewidth=1.2, label="Signal")
        hist = df["MACD_Hist"]
        hist_colors = [c["hist_up"] if v >= 0 else c["hist_down"] for v in hist]
        ax_macd.bar(df.index, hist, color=hist_colors, alpha=0.6, width=0.8)
        ax_macd.axhline(y=0, color=c["grid"], linewidth=0.8)
        ax_macd.set_ylabel("MACD", color=c["text"], fontsize=8)
        ax_macd.legend(loc="upper left", facecolor=c["ax_bg"], edgecolor=c["grid"],
                       labelcolor=c["text"], fontsize=7)
        plt.setp(ax_macd.get_xticklabels(), visible=False)

    # --- RSI ---
    if show_rsi and "RSI_14" in df.columns:
        ax_rsi = axes[panel_idx]
        panel_idx += 1
        ax_rsi.plot(df.index, df["RSI_14"], color=c["rsi"], linewidth=1.2, label="RSI 14")
        ax_rsi.axhline(y=70, color=c["down"], linewidth=0.8, linestyle="--", alpha=0.7, label="70")
        ax_rsi.axhline(y=30, color=c["up"], linewidth=0.8, linestyle="--", alpha=0.7, label="30")
        ax_rsi.axhline(y=50, color=c["grid"], linewidth=0.5, alpha=0.5)
        ax_rsi.fill_between(df.index, df["RSI_14"], 70,
                             where=df["RSI_14"] >= 70, alpha=0.2, color=c["down"])
        ax_rsi.fill_between(df.index, df["RSI_14"], 30,
                             where=df["RSI_14"] <= 30, alpha=0.2, color=c["up"])
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color=c["text"], fontsize=8)
        ax_rsi.legend(loc="upper left", facecolor=c["ax_bg"], edgecolor=c["grid"],
                      labelcolor=c["text"], fontsize=7)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax_rsi.get_xticklabels(), rotation=30, ha="right")

    # --- Patterns, lignes de tendance et Fibonacci sur le graphique principal ---
    if show_patterns or show_trendlines or show_fibonacci:
        plot_patterns(
            ax_main, df,
            show_candlestick_patterns=show_patterns,
            show_chart_patterns=show_patterns,
            show_trendlines=show_trendlines,
            show_fibonacci=show_fibonacci,
            dark_theme=dark_theme,
        )

    # Mise à jour de la légende après ajout des patterns
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_main.legend(
            handles[:8], labels[:8],  # limiter à 8 éléments
            loc="upper left",
            facecolor=c["ax_bg"], edgecolor=c["grid"],
            labelcolor=c["text"], fontsize=7,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Graphique sauvegardé: {save_path}")

    return fig


def _plot_candlesticks(ax, df: pd.DataFrame, c: dict):
    """Trace les chandeliers japonais."""
    width = 0.6
    width2 = 0.1

    up = df[df["Close"] >= df["Open"]]
    down = df[df["Close"] < df["Open"]]

    # Corps des bougies haussières
    ax.bar(up.index, up["Close"] - up["Open"], width, bottom=up["Open"],
           color=c["up"], alpha=0.9, zorder=2)
    ax.bar(up.index, up["High"] - up["Close"], width2, bottom=up["Close"],
           color=c["up"], alpha=0.9, zorder=2)
    ax.bar(up.index, up["Low"] - up["Open"], width2, bottom=up["Open"],
           color=c["up"], alpha=0.9, zorder=2)

    # Corps des bougies baissières
    ax.bar(down.index, down["Close"] - down["Open"], width, bottom=down["Open"],
           color=c["down"], alpha=0.9, zorder=2)
    ax.bar(down.index, down["High"] - down["Open"], width2, bottom=down["Open"],
           color=c["down"], alpha=0.9, zorder=2)
    ax.bar(down.index, down["Low"] - down["Close"], width2, bottom=down["Close"],
           color=c["down"], alpha=0.9, zorder=2)


def plot_indicators(
    df: pd.DataFrame,
    indicators: List[str] = None,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
) -> plt.Figure:
    """
    Affiche des indicateurs spécifiques dans des panneaux séparés.

    Args:
        df: DataFrame avec indicateurs calculés
        indicators: Liste d'indicateurs à afficher
                   Options: "RSI", "MACD", "Stochastique", "CCI", "MFI", "Williams"
        figsize: Taille de la figure
        save_path: Chemin de sauvegarde
        dark_theme: Thème sombre

    Returns:
        Figure matplotlib
    """
    if indicators is None:
        indicators = ["RSI", "MACD", "Stochastique"]

    c = DARK_THEME if dark_theme else {"bg": "white", "ax_bg": "white",
                                        "grid": "#e0e0e0", "text": "black",
                                        "up": "#26a641", "down": "#da3633",
                                        "rsi": "#2196F3", "macd": "#2196F3",
                                        "signal": "#FF5722", "hist_up": "#26a641",
                                        "hist_down": "#da3633"}

    n = len(indicators)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    if dark_theme:
        _apply_dark_theme(fig, axes)

    for i, indicator in enumerate(indicators):
        ax = axes[i]
        ax.set_title(indicator, color=c["text"], fontsize=10)

        if indicator == "RSI" and "RSI_14" in df.columns:
            ax.plot(df.index, df["RSI_14"], color=c.get("rsi", "#58a6ff"), linewidth=1.2)
            ax.axhline(70, color=c["down"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(30, color=c["up"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(50, color=c["grid"], linewidth=0.5)
            ax.fill_between(df.index, df["RSI_14"], 70, where=df["RSI_14"] >= 70,
                             alpha=0.2, color=c["down"])
            ax.fill_between(df.index, df["RSI_14"], 30, where=df["RSI_14"] <= 30,
                             alpha=0.2, color=c["up"])
            ax.set_ylim(0, 100)
            ax.set_ylabel("RSI", color=c["text"], fontsize=8)

        elif indicator == "MACD" and "MACD" in df.columns:
            ax.plot(df.index, df["MACD"], color=c.get("macd", "#58a6ff"),
                    linewidth=1.2, label="MACD")
            ax.plot(df.index, df["MACD_Signal"], color=c.get("signal", "#ff7b72"),
                    linewidth=1.2, label="Signal")
            hist = df["MACD_Hist"]
            hist_colors = [c["hist_up"] if v >= 0 else c["hist_down"] for v in hist]
            ax.bar(df.index, hist, color=hist_colors, alpha=0.6, width=0.8)
            ax.axhline(0, color=c["grid"], linewidth=0.8)
            ax.legend(loc="upper left", facecolor=c.get("ax_bg", "white"),
                      edgecolor=c["grid"], labelcolor=c["text"], fontsize=7)
            ax.set_ylabel("MACD", color=c["text"], fontsize=8)

        elif indicator == "Stochastique" and "Stoch_%K" in df.columns:
            ax.plot(df.index, df["Stoch_%K"], color=c.get("macd", "#58a6ff"),
                    linewidth=1.2, label="%K")
            ax.plot(df.index, df["Stoch_%D"], color=c.get("signal", "#ff7b72"),
                    linewidth=1.2, label="%D")
            ax.axhline(80, color=c["down"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(20, color=c["up"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_ylim(0, 100)
            ax.legend(loc="upper left", facecolor=c.get("ax_bg", "white"),
                      edgecolor=c["grid"], labelcolor=c["text"], fontsize=7)
            ax.set_ylabel("Stoch", color=c["text"], fontsize=8)

        elif indicator == "CCI" and "CCI_20" in df.columns:
            ax.plot(df.index, df["CCI_20"], color=c.get("macd", "#58a6ff"), linewidth=1.2)
            ax.axhline(100, color=c["down"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(-100, color=c["up"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(0, color=c["grid"], linewidth=0.5)
            ax.set_ylabel("CCI", color=c["text"], fontsize=8)

        elif indicator == "MFI" and "MFI_14" in df.columns:
            ax.plot(df.index, df["MFI_14"], color=c.get("macd", "#58a6ff"), linewidth=1.2)
            ax.axhline(80, color=c["down"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(20, color=c["up"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_ylabel("MFI", color=c["text"], fontsize=8)

        elif indicator == "Williams" and "Williams_%R" in df.columns:
            ax.plot(df.index, df["Williams_%R"], color=c.get("macd", "#58a6ff"), linewidth=1.2)
            ax.axhline(-20, color=c["down"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(-80, color=c["up"], linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_ylim(-100, 0)
            ax.set_ylabel("W%R", color=c["text"], fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")

    symbol = df.attrs.get("symbol", "")
    fig.suptitle(f"Indicateurs - {symbol}", color=c["text"], fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


def plot_heatmap(
    returns_df: pd.DataFrame,
    title: str = "Performance des actions BVC",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Affiche une heatmap des performances des actions.

    Args:
        returns_df: DataFrame des rendements (lignes=dates, colonnes=symboles)
        title: Titre
        figsize: Taille
        save_path: Chemin de sauvegarde

    Returns:
        Figure matplotlib
    """
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["ax_bg"])

        sns.heatmap(
            returns_df.T,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Rendement (%)"},
        )
        ax.set_title(title, color=DARK_THEME["text"], fontsize=13)
        plt.tight_layout()

    except ImportError:
        print("Note: installez 'seaborn' pour la heatmap (pip install seaborn)")
        fig = plt.figure(figsize=figsize)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig
