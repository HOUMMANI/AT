"""
Visualisation multi-timeframes pour les actions BVC.

Fonctions :
  plot_mtf_overview    : Panneau comparatif des 3 timeframes côte à côte
  plot_mtf_confluence  : Graphique journalier annoté avec niveaux MTF
  plot_timeframe       : Graphique complet pour un seul timeframe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")

from .charts import DARK_THEME, _apply_dark_theme, _plot_candlesticks, plot_patterns


# Étiquettes courtes pour les timeframes
TF_SHORT = {"1d": "J", "1wk": "H", "1mo": "M"}
TF_FULL = {"1d": "Journalier", "1wk": "Hebdo", "1mo": "Mensuel"}


def plot_mtf_overview(
    mtf_analyzer,
    figsize: tuple = (20, 14),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
) -> plt.Figure:
    """
    Crée un panneau 3-en-1 : Journalier | Hebdomadaire | Mensuel.

    Chaque panneau affiche :
    - Chandeliers japonais
    - SMA rapide, lente et longue (adaptées au timeframe)
    - Bandes de Bollinger
    - RSI sous chaque graphique
    - Niveau de score et recommandation

    Args:
        mtf_analyzer: Instance de MultiTimeframeAnalyzer (déjà run())
        figsize: Taille de la figure
        save_path: Chemin de sauvegarde
        dark_theme: Thème sombre

    Returns:
        Figure matplotlib
    """
    if not mtf_analyzer._ran:
        mtf_analyzer.run()

    results = mtf_analyzer._results
    confluence = mtf_analyzer._confluence
    symbol = mtf_analyzer.symbol
    tfs = [tf for tf in mtf_analyzer.timeframes if tf in results]
    n = len(tfs)

    if n == 0:
        raise ValueError("Aucune donnée disponible")

    c = DARK_THEME if dark_theme else {}
    up = c.get("up", "#26a641")
    down = c.get("down", "#da3633")
    text_color = c.get("text", "#c9d1d9")
    grid_color = c.get("grid", "#21262d")
    bg = c.get("bg", "#0d1117")
    ax_bg = c.get("ax_bg", "#161b22")

    # Layout : n colonnes × 2 lignes (prix + RSI), + ligne titre en haut
    fig = plt.figure(figsize=figsize, facecolor=bg)
    outer = gridspec.GridSpec(
        2, n,
        height_ratios=[3.5, 1.2],
        hspace=0.08,
        wspace=0.06,
        top=0.91, bottom=0.06,
    )

    # Titre principal
    score_str = f"{confluence.score_pondere:+.1f}"
    score_color = up if confluence.score_pondere >= 0 else down
    fig.suptitle(
        f"{symbol}  —  Analyse Multi-Timeframes  |  "
        f"Score: {score_str}  |  {confluence.recommandation}",
        fontsize=14, color=text_color, fontweight="bold", y=0.97,
    )

    axes_price = []
    axes_rsi = []

    for col, tf in enumerate(tfs):
        a = results[tf]
        df = a.df
        label = TF_FULL.get(tf, tf)

        ax_p = fig.add_subplot(outer[0, col])
        ax_r = fig.add_subplot(outer[1, col], sharex=ax_p)
        axes_price.append(ax_p)
        axes_rsi.append(ax_r)

        _apply_dark_theme(fig, [ax_p, ax_r])

        # Limiter à 100 dernières périodes pour la lisibilité
        display_df = df.tail(100)

        # --- Chandeliers ---
        _plot_candlesticks(ax_p, display_df, c)

        # --- SMA adaptées ---
        sma_colors = [c.get("sma20", "#f0b429"), c.get("sma50", "#58a6ff"), c.get("sma200", "#bc8cff")]
        sma_labels = []
        from ..analysis.multi_timeframe import TIMEFRAME_PARAMS
        p = TIMEFRAME_PARAMS.get(tf, TIMEFRAME_PARAMS["1d"])

        for sma_col, sma_c, sma_label in [
            ("SMA_fast", sma_colors[0], f"SMA{p['sma_fast']}"),
            ("SMA_slow", sma_colors[1], f"SMA{p['sma_slow']}"),
            ("SMA_long", sma_colors[2], f"SMA{p['sma_long']}"),
        ]:
            if sma_col in display_df.columns and display_df[sma_col].notna().sum() > 3:
                ax_p.plot(display_df.index, display_df[sma_col],
                          color=sma_c, linewidth=1.0, label=sma_label, alpha=0.85)

        # --- Bollinger Bands ---
        if "BB_Haute" in display_df.columns:
            ax_p.plot(display_df.index, display_df["BB_Haute"],
                      color=c.get("bb_upper", "#f0b429"), linewidth=0.7,
                      linestyle="--", alpha=0.6)
            ax_p.plot(display_df.index, display_df["BB_Basse"],
                      color=c.get("bb_lower", "#f0b429"), linewidth=0.7,
                      linestyle="--", alpha=0.6)
            ax_p.fill_between(display_df.index,
                              display_df["BB_Haute"], display_df["BB_Basse"],
                              alpha=0.04, color=c.get("bb_upper", "#f0b429"))

        # Score badge
        score_bg = up if a.score >= 25 else down if a.score <= -25 else "#888888"
        ax_p.text(
            0.98, 0.98,
            f"{a.score:+.0f}",
            transform=ax_p.transAxes,
            fontsize=11, fontweight="bold",
            color="white", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc=score_bg, alpha=0.85, ec="none"),
        )

        # Titre du panneau
        ax_p.set_title(
            f"{label}   |   {a.recommandation}   |   Tendance: {a.tendance}",
            fontsize=9, color=text_color, pad=4,
        )
        ax_p.legend(loc="upper left", fontsize=7,
                    facecolor=ax_bg, edgecolor=grid_color, labelcolor=text_color)
        ax_p.set_ylabel("Prix (MAD)" if col == 0 else "", color=text_color, fontsize=8)
        plt.setp(ax_p.get_xticklabels(), visible=False)

        # --- RSI ---
        if "RSI" in display_df.columns:
            rsi_vals = display_df["RSI"]
            ax_r.plot(display_df.index, rsi_vals,
                      color=c.get("rsi", "#58a6ff"), linewidth=1.2)
            ax_r.axhline(70, color=down, linewidth=0.7, linestyle="--", alpha=0.7)
            ax_r.axhline(30, color=up, linewidth=0.7, linestyle="--", alpha=0.7)
            ax_r.axhline(50, color=grid_color, linewidth=0.5, alpha=0.5)
            ax_r.fill_between(display_df.index, rsi_vals, 70,
                              where=rsi_vals >= 70, alpha=0.2, color=down)
            ax_r.fill_between(display_df.index, rsi_vals, 30,
                              where=rsi_vals <= 30, alpha=0.2, color=up)
            ax_r.set_ylim(0, 100)
            ax_r.set_ylabel(f"RSI({p['rsi_period']})" if col == 0 else "", color=text_color, fontsize=7)

            # Valeur RSI actuelle
            rsi_now = rsi_vals.iloc[-1]
            rsi_clr = down if rsi_now > 70 else up if rsi_now < 30 else text_color
            ax_r.text(0.02, 0.85, f"RSI {rsi_now:.1f}",
                      transform=ax_r.transAxes, fontsize=7.5,
                      color=rsi_clr, fontweight="bold")

        ax_r.xaxis.set_major_formatter(mdates.DateFormatter(
            "%b %Y" if tf == "1mo" else "%m/%Y" if tf == "1wk" else "%d/%m"
        ))
        plt.setp(ax_r.get_xticklabels(), rotation=25, ha="right", fontsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
        print(f"Graphique MTF sauvegardé: {save_path}")

    return fig


def plot_mtf_confluence(
    mtf_analyzer,
    base_tf: str = "1d",
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
) -> plt.Figure:
    """
    Graphique du timeframe de base (journalier par défaut) enrichi
    des niveaux des autres timeframes en overlay.

    Ajoute sur le graphique journalier :
    - Niveaux de S/R hebdomadaires (lignes épaisses bleues)
    - Niveaux de S/R mensuels (lignes épaisses violettes)
    - Zones de confluence (fond coloré)
    - Niveaux Fibonacci (tous timeframes)

    Args:
        mtf_analyzer: Instance de MultiTimeframeAnalyzer (déjà run())
        base_tf: Timeframe de base du graphique (défaut: "1d")
        figsize: Taille de la figure
        save_path: Chemin de sauvegarde
        dark_theme: Thème sombre
    """
    if not mtf_analyzer._ran:
        mtf_analyzer.run()

    results = mtf_analyzer._results
    confluence = mtf_analyzer._confluence
    symbol = mtf_analyzer.symbol

    if base_tf not in results:
        available = list(results.keys())
        base_tf = available[0] if available else None
    if not base_tf:
        raise ValueError("Aucune donnée disponible")

    a_base = results[base_tf]
    df = a_base.df
    c = DARK_THEME if dark_theme else {}
    up = c.get("up", "#26a641")
    down = c.get("down", "#da3633")
    text_color = c.get("text", "#c9d1d9")
    grid_color = c.get("grid", "#21262d")
    bg = c.get("bg", "#0d1117")
    ax_bg = c.get("ax_bg", "#161b22")

    # Layout : prix + MACD + RSI
    fig = plt.figure(figsize=figsize, facecolor=bg)
    gs = gridspec.GridSpec(3, 1, height_ratios=[4.5, 1.5, 1.5], hspace=0.06)
    ax_main = fig.add_subplot(gs[0])
    ax_macd = fig.add_subplot(gs[1], sharex=ax_main)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_main)
    _apply_dark_theme(fig, [ax_main, ax_macd, ax_rsi])

    display_df = df.tail(200)

    # --- Chandeliers ---
    _plot_candlesticks(ax_main, display_df, c)

    # --- SMA base ---
    from ..analysis.multi_timeframe import TIMEFRAME_PARAMS, TIMEFRAME_LABELS
    p_base = TIMEFRAME_PARAMS.get(base_tf, TIMEFRAME_PARAMS["1d"])
    sma_colors = [c.get("sma20", "#f0b429"), c.get("sma50", "#58a6ff"), c.get("sma200", "#bc8cff")]
    for sma_col, sma_c, sname in [
        ("SMA_fast", sma_colors[0], f"SMA{p_base['sma_fast']}"),
        ("SMA_slow", sma_colors[1], f"SMA{p_base['sma_slow']}"),
        ("SMA_long", sma_colors[2], f"SMA{p_base['sma_long']}"),
    ]:
        if sma_col in display_df.columns and display_df[sma_col].notna().sum() > 3:
            ax_main.plot(display_df.index, display_df[sma_col],
                         color=sma_c, linewidth=1.0, label=sname, alpha=0.8)

    # --- Niveaux des autres timeframes ---
    tf_level_colors = {"1wk": "#58a6ff", "1mo": "#bc8cff"}
    tf_level_styles = {"1wk": "-", "1mo": "-"}
    tf_lw = {"1wk": 1.5, "1mo": 2.0}

    for tf, a in results.items():
        if tf == base_tf:
            continue
        color = tf_level_colors.get(tf, "#888888")
        lw = tf_lw.get(tf, 1.5)
        ls = tf_level_styles.get(tf, "--")
        short = TF_SHORT.get(tf, tf)

        for s in a.supports[:2]:
            ax_main.axhline(s, color=color, linewidth=lw, linestyle="--", alpha=0.7)
            ax_main.annotate(
                f"S({short}) {s:.1f}",
                xy=(display_df.index[-1], s),
                fontsize=7, color=color, ha="right", va="top", alpha=0.85,
            )

        for r in a.resistances[:2]:
            ax_main.axhline(r, color=color, linewidth=lw, linestyle="--", alpha=0.7)
            ax_main.annotate(
                f"R({short}) {r:.1f}",
                xy=(display_df.index[-1], r),
                fontsize=7, color=color, ha="right", va="bottom", alpha=0.85,
            )

        if a.fib_support:
            ax_main.axhline(a.fib_support, color=color, linewidth=0.9,
                            linestyle=":", alpha=0.5)
        if a.fib_resistance:
            ax_main.axhline(a.fib_resistance, color=color, linewidth=0.9,
                            linestyle=":", alpha=0.5)

    # --- Zones de confluence ---
    current_price = a_base.prix_actuel
    for zone in confluence.zones_confluence[:5]:
        is_support = zone["prix"] < current_price
        zone_color = up if is_support else down
        ax_main.axhspan(
            zone["prix"] * 0.998, zone["prix"] * 1.002,
            alpha=0.12, color=zone_color,
        )
        ax_main.annotate(
            f"⚡ Confluence {zone['prix']:.1f}",
            xy=(display_df.index[len(display_df) // 3], zone["prix"]),
            fontsize=7.5, color=zone_color, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=ax_bg, alpha=0.7, ec=zone_color, lw=0.6),
        )

    # Titre
    score_str = f"{confluence.score_pondere:+.1f}"
    ax_main.set_title(
        f"{symbol} — {TF_FULL.get(base_tf, base_tf)} avec niveaux MTF  |  "
        f"Score global: {score_str}  |  {confluence.recommandation}",
        fontsize=11, color=text_color, pad=8,
    )
    ax_main.legend(loc="upper left", fontsize=7.5,
                   facecolor=ax_bg, edgecolor=grid_color, labelcolor=text_color)
    ax_main.set_ylabel("Prix (MAD)", color=text_color)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- MACD ---
    if "MACD" in display_df.columns:
        ax_macd.plot(display_df.index, display_df["MACD"],
                     color=c.get("macd", "#58a6ff"), linewidth=1.2, label="MACD")
        ax_macd.plot(display_df.index, display_df["MACD_Signal"],
                     color=c.get("signal", "#ff7b72"), linewidth=1.2, label="Signal")
        hist = display_df["MACD_Hist"]
        hist_colors = [c.get("hist_up", up) if v >= 0 else c.get("hist_down", down) for v in hist]
        ax_macd.bar(display_df.index, hist, color=hist_colors, alpha=0.6, width=0.8)
        ax_macd.axhline(0, color=grid_color, linewidth=0.7)
        ax_macd.set_ylabel("MACD", color=text_color, fontsize=8)
        ax_macd.legend(loc="upper left", fontsize=7,
                       facecolor=ax_bg, edgecolor=grid_color, labelcolor=text_color)
        plt.setp(ax_macd.get_xticklabels(), visible=False)

    # --- RSI ---
    if "RSI" in display_df.columns:
        rsi_vals = display_df["RSI"]
        ax_rsi.plot(display_df.index, rsi_vals,
                    color=c.get("rsi", "#58a6ff"), linewidth=1.2, label=f"RSI({p_base['rsi_period']})")
        ax_rsi.axhline(70, color=down, linewidth=0.7, linestyle="--", alpha=0.7)
        ax_rsi.axhline(30, color=up, linewidth=0.7, linestyle="--", alpha=0.7)
        ax_rsi.axhline(50, color=grid_color, linewidth=0.4)
        ax_rsi.fill_between(display_df.index, rsi_vals, 70,
                            where=rsi_vals >= 70, alpha=0.2, color=down)
        ax_rsi.fill_between(display_df.index, rsi_vals, 30,
                            where=rsi_vals <= 30, alpha=0.2, color=up)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color=text_color, fontsize=8)
        plt.setp(ax_rsi.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Légende des niveaux MTF (hors du graphique)
    legend_elements = []
    import matplotlib.patches as mpatches
    for tf, clr in tf_level_colors.items():
        if tf in results:
            legend_elements.append(
                mpatches.Patch(color=clr, label=f"S/R {TF_FULL.get(tf, tf)}", alpha=0.7)
            )
    legend_elements.append(
        mpatches.Patch(color="#ffffff", alpha=0.3, label="Zone de confluence ⚡")
    )
    ax_main.legend(
        handles=ax_main.get_legend_handles_labels()[0][:6] + legend_elements,
        labels=ax_main.get_legend_handles_labels()[1][:6] + [e.get_label() for e in legend_elements],
        loc="upper left", fontsize=7,
        facecolor=ax_bg, edgecolor=grid_color, labelcolor=text_color,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
        print(f"Graphique confluence sauvegardé: {save_path}")

    return fig


def plot_timeframe(
    df: pd.DataFrame,
    timeframe: str = "1d",
    show_patterns: bool = True,
    show_trendlines: bool = True,
    show_fibonacci: bool = True,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
) -> plt.Figure:
    """
    Graphique complet pour un seul timeframe spécifique.
    Adapte automatiquement les paramètres des indicateurs au timeframe.

    Args:
        df: DataFrame OHLCV
        timeframe: "1d", "1wk" ou "1mo"
        show_patterns: Afficher les patterns
        show_trendlines: Afficher les lignes de tendance
        show_fibonacci: Afficher Fibonacci
        figsize: Taille
        save_path: Chemin de sauvegarde
        dark_theme: Thème sombre
    """
    from ..analysis.multi_timeframe import TIMEFRAME_PARAMS, TIMEFRAME_LABELS
    from .charts import plot_chart

    # Injecter les paramètres adaptés au timeframe dans le DataFrame
    p = TIMEFRAME_PARAMS.get(timeframe, TIMEFRAME_PARAMS["1d"])
    from ..indicators.trend import sma as calc_sma, macd as calc_macd
    from ..indicators.momentum import rsi as calc_rsi
    from ..indicators.volatility import bollinger_bands as calc_bb

    close = df["Close"]
    df = df.copy()
    df["SMA_20"] = calc_sma(close, p["sma_fast"])
    df["SMA_50"] = calc_sma(close, p["sma_slow"])
    df["SMA_200"] = calc_sma(close, p["sma_long"])

    macd_df = calc_macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    df["MACD"] = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["Signal"]
    df["MACD_Hist"] = macd_df["Histogramme"]

    df["RSI_14"] = calc_rsi(close, p["rsi_period"])

    bb = calc_bb(close, p["bb_period"])
    df["BB_Haute"] = bb["BB_Haute"]
    df["BB_Basse"] = bb["BB_Basse"]
    df["BB_Milieu"] = bb["BB_Milieu"]

    label = TIMEFRAME_LABELS.get(timeframe, timeframe)
    if "name" in df.attrs:
        df.attrs["name"] = f"{df.attrs['name']} ({label})"

    return plot_chart(
        df,
        title=f"{df.attrs.get('symbol', '')} — {label}",
        show_patterns=show_patterns,
        show_trendlines=show_trendlines,
        show_fibonacci=show_fibonacci,
        figsize=figsize,
        save_path=save_path,
        dark_theme=dark_theme,
    )
