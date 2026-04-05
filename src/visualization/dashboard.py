"""
Dashboard 8 panneaux pour l'analyse technique BVC.

Layout 2x4 :
  P1 Bougies OHLCV  | P2 Tracés avancés  | P3 Signaux MTF  | P4 Volume Profile
  P5 Oscillateurs   | P6 Saisonnalité     | P7 Volatilité   | P8 Score global

Usage:
    from src.visualization.dashboard import plot_dashboard, AVAILABLE_OVERLAYS
    fig = plot_dashboard(df, overlays=["fibonacci","ichimoku"], mtf_analyzer=mtf)
    fig.savefig("dashboard.png", dpi=150, bbox_inches="tight")
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow, Wedge, Arc
from typing import List, Optional
import logging

from .charts import DARK_THEME, _apply_dark_theme, _plot_candlesticks

logger = logging.getLogger(__name__)

AVAILABLE_OVERLAYS = {
    "fibonacci":            "Retracements Fibonacci",
    "trendlines":           "Lignes de tendance",
    "ichimoku":             "Ichimoku Cloud",
    "pivots":               "Points Pivots",
    "patterns":             "Configurations graphiques",
    "support_resistance":   "Zones S/R horizontales",
    "regression":           "Canal de régression",
    "candlestick_patterns": "Patterns bougies (▲▼)",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _c(dark):
    return DARK_THEME if dark else {
        "bg":"white","ax_bg":"white","grid":"#e0e0e0","text":"black",
        "up":"#26a641","down":"#da3633","sma20":"#f0b429","sma50":"#2196F3",
        "sma200":"#9C27B0","macd":"#2196F3","signal":"#FF5722",
        "hist_up":"#26a641","hist_down":"#da3633","rsi":"#2196F3",
        "bb_upper":"#f0b429","bb_lower":"#f0b429","volume_up":"#26a64180",
        "volume_down":"#da363380",
    }

def _sig_color(sig: str, c: dict) -> str:
    s = str(sig).upper()
    if any(x in s for x in ("ACHAT","HAUSSIER","GOLDEN","SURVENTE","FORT","SOLDAT","CREUX")):
        return c.get("up","#26a641")
    if any(x in s for x in ("VENTE","BAISSIER","DEATH","SURACHAT","CORB","SOMMET")):
        return c.get("down","#da3633")
    return c.get("grid","#444444")


# ─────────────────────────────────────────────
# PANEL 1 — OHLCV BOUGIES PURES
# ─────────────────────────────────────────────

def _draw_panel1(ax_price, ax_vol, df: pd.DataFrame, c: dict):
    """Chandeliers japonais + volume. Aucun indicateur."""
    _plot_candlesticks(ax_price, df, c)
    ax_price.set_ylabel("Prix (MAD)", color=c["text"], fontsize=8)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax_price.get_xticklabels(), visible=False)
    ax_price.set_title("OHLCV — Bougies Japonaises", color=c["text"], fontsize=9, pad=4)

    colors = [c["volume_up"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
              else c["volume_down"] for i in range(len(df))]
    ax_vol.bar(df.index, df["Volume"], color=colors, width=0.8)
    ax_vol.set_ylabel("Vol", color=c["text"], fontsize=7)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax_vol.get_xticklabels(), rotation=25, ha="right", fontsize=6)


# ─────────────────────────────────────────────
# PANEL 2 — TRACÉS AVANCÉS (overlays choisis)
# ─────────────────────────────────────────────

def _draw_panel2(ax, df: pd.DataFrame, overlays: List[str], c: dict):
    """Chandeliers + overlays sélectionnables."""
    _plot_candlesticks(ax, df, c)

    active_labels = []

    # ── Fibonacci ──
    if "fibonacci" in overlays:
        try:
            from ..patterns.fibonacci import FibonacciAnalyzer
            fib = FibonacciAnalyzer(df)
            analysis = fib.analyze()
            fib_colors = {0.0:"#888",0.236:"#58a6ff",0.382:"#f0b429",
                          0.500:"#fff",0.618:"#f0b429",0.786:"#58a6ff",1.0:"#888"}
            cur = df["Close"].iloc[-1]
            for lvl in analysis.niveaux:
                if lvl.type != "retracement" or abs(lvl.distance_pct) > 18:
                    continue
                clr = fib_colors.get(lvl.ratio, "#aaa")
                ls  = "-." if lvl.ratio in (0.382,0.5,0.618) else ":"
                ax.axhline(lvl.prix, color=clr, linewidth=0.8, linestyle=ls, alpha=0.7)
                ax.annotate(f"Fib {lvl.ratio*100:.1f}%  {lvl.prix:.1f}",
                            xy=(df.index[5], lvl.prix), fontsize=6,
                            color=clr, alpha=0.85, va="bottom")
            active_labels.append("Fibonacci")
        except Exception: pass

    # ── Régression linéaire ──
    if "regression" in overlays:
        try:
            n = min(60, len(df))
            x = np.arange(n)
            y = df["Close"].iloc[-n:].values
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            resid = y - y_fit
            std   = resid.std()
            dates_reg = df.index[-n:]
            ax.plot(dates_reg, y_fit,         color="#58a6ff", lw=1.2, linestyle="--", alpha=0.8, label="Régression")
            ax.plot(dates_reg, y_fit + 2*std, color="#58a6ff", lw=0.7, linestyle=":",  alpha=0.5)
            ax.plot(dates_reg, y_fit - 2*std, color="#58a6ff", lw=0.7, linestyle=":",  alpha=0.5)
            ax.fill_between(dates_reg, y_fit-2*std, y_fit+2*std, alpha=0.05, color="#58a6ff")
            active_labels.append("Régression")
        except Exception: pass

    # ── Ichimoku ──
    if "ichimoku" in overlays:
        try:
            from ..indicators.trend import ichimoku
            ich = ichimoku(df)
            ax.plot(df.index, ich["Tenkan"], color="#ff7b72", lw=0.9, alpha=0.8, label="Tenkan")
            ax.plot(df.index, ich["Kijun"],  color="#58a6ff", lw=0.9, alpha=0.8, label="Kijun")
            sa = ich["SenkouA"].dropna()
            sb = ich["SenkouB"].dropna()
            idx_common = sa.index.intersection(sb.index)
            if len(idx_common) > 0:
                ax.fill_between(idx_common,
                                sa.loc[idx_common], sb.loc[idx_common],
                                where=sa.loc[idx_common] >= sb.loc[idx_common],
                                alpha=0.12, color="#26a641")
                ax.fill_between(idx_common,
                                sa.loc[idx_common], sb.loc[idx_common],
                                where=sa.loc[idx_common] <  sb.loc[idx_common],
                                alpha=0.12, color="#da3633")
            active_labels.append("Ichimoku")
        except Exception: pass

    # ── Points Pivots ──
    if "pivots" in overlays:
        try:
            from ..indicators.trend import pivot_points
            pp = pivot_points(df.tail(1))
            last = pp.iloc[-1]
            pv_colors = {"PP":"#fff","R1":"#da3633","R2":"#da3633","R3":"#da3633",
                         "S1":"#26a641","S2":"#26a641","S3":"#26a641"}
            for level, clr in pv_colors.items():
                val = last.get(level, np.nan)
                if np.isnan(val): continue
                ax.axhline(val, color=clr, lw=0.9, linestyle="--", alpha=0.7)
                ax.annotate(f"{level} {val:.1f}", xy=(df.index[-1], val),
                            fontsize=6, color=clr, ha="right", va="bottom", alpha=0.85)
            active_labels.append("Pivots")
        except Exception: pass

    # ── Lignes de tendance ──
    if "trendlines" in overlays:
        try:
            from ..patterns.trendlines import TrendlineDetector
            td = TrendlineDetector(df)
            lines = td.detect_all()
            x_all = np.arange(len(df))
            for tl in lines[:4]:
                clr = c["up"] if tl.direction == "SUPPORT" else c["down"]
                if tl.type == "HORIZONTALE":
                    ax.axhline(tl.prix_actuel, color=clr, lw=0.9, linestyle="--", alpha=0.6)
                else:
                    if tl.date_debut in df.index:
                        i0 = df.index.get_loc(tl.date_debut)
                        slope = tl.pente
                        inter = tl.prix_debut - slope * i0
                        yv = slope * x_all[i0:] + inter
                        ax.plot(df.index[i0:], yv, color=clr, lw=1.0, alpha=0.65)
            active_labels.append("Tendances")
        except Exception: pass

    # ── S/R horizontaux ──
    if "support_resistance" in overlays:
        try:
            from ..patterns.trendlines import TrendlineDetector
            td  = TrendlineDetector(df)
            hor = td.detect_horizontal_levels()
            cur = df["Close"].iloc[-1]
            for h in hor[:6]:
                clr = c["up"] if h.prix_actuel < cur else c["down"]
                ax.axhline(h.prix_actuel, color=clr, lw=1.1, linestyle="-.", alpha=0.55)
            active_labels.append("S/R")
        except Exception: pass

    # ── Configurations graphiques ──
    if "patterns" in overlays:
        try:
            from ..patterns.chart_patterns import ChartPatternDetector
            det = ChartPatternDetector(df)
            pats = det.detect_all()
            for p in pats[:2]:
                clr = c["up"] if p.direction=="HAUSSIER" else c["down"] if p.direction=="BAISSIER" else "#aaa"
                ax.axhline(p.niveau_breakout, color=clr, lw=1.5, linestyle="-.", alpha=0.85)
                ax.annotate(f"▶ {p.nom}  {p.niveau_breakout}",
                            xy=(df.index[len(df)//5], p.niveau_breakout),
                            fontsize=7, color=clr, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.25", fc=c["ax_bg"], alpha=0.8, ec=clr, lw=0.7))
                if p.objectif_prix:
                    ax.axhline(p.objectif_prix, color=clr, lw=0.7, linestyle=":", alpha=0.5)
            active_labels.append("Patterns")
        except Exception: pass

    # ── Marqueurs bougies ──
    if "candlestick_patterns" in overlays:
        try:
            from ..patterns.candlesticks import CandlestickPatterns
            cp   = CandlestickPatterns(df)
            recs = cp.get_recent(lookback=20)
            off  = df["Close"].mean() * 0.015
            bull_d, bull_p, bear_d, bear_p = [], [], [], []
            for r in recs:
                d = r["date"]
                if d not in df.index: continue
                if r["direction"] == "HAUSSIER":
                    bull_d.append(d); bull_p.append(df.loc[d,"Low"] - off)
                elif r["direction"] == "BAISSIER":
                    bear_d.append(d); bear_p.append(df.loc[d,"High"] + off)
            if bull_d: ax.scatter(bull_d, bull_p, marker="^", color=c["up"],   s=70, zorder=5, alpha=0.9)
            if bear_d: ax.scatter(bear_d, bear_p, marker="v", color=c["down"], s=70, zorder=5, alpha=0.9)
            active_labels.append("Bougies▲▼")
        except Exception: pass

    # Légende overlays actifs
    ax.set_title("TRACÉS AVANCÉS", color=c["text"], fontsize=9, pad=4)
    if active_labels:
        leg_text = "Actifs: " + " · ".join(active_labels)
        ax.text(0.01, 0.01, leg_text, transform=ax.transAxes,
                fontsize=6.5, color=c["text"], alpha=0.8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc=c["ax_bg"], alpha=0.7, ec=c["grid"]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("Prix (MAD)", color=c["text"], fontsize=8)


# ─────────────────────────────────────────────
# PANEL 3 — SIGNAUX MTF (tableau coloré)
# ─────────────────────────────────────────────

def _draw_panel3(ax, mtf_analyzer, df: pd.DataFrame, c: dict):
    """Heatmap table des signaux pour chaque timeframe × indicateur."""
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("SIGNAUX MULTI-TIMEFRAMES", color=c["text"], fontsize=9, pad=4)

    INDICATORS = ["Tendance","RSI","MACD","Bollinger","Volume"]
    TF_LABELS  = {"1d":"Journalier","1wk":"Hebdo","1mo":"Mensuel"}

    def _score_to_signal(score):
        if score >= 50: return "FORT ACHAT"
        if score >= 20: return "ACHAT"
        if score >= -20: return "NEUTRE"
        if score >= -50: return "VENTE"
        return "FORT VENTE"

    # Construire la data depuis mtf_analyzer ou analyser le df seul
    columns, data = [], {ind: [] for ind in INDICATORS}
    if mtf_analyzer and mtf_analyzer._ran:
        for tf, res in mtf_analyzer._results.items():
            columns.append(TF_LABELS.get(tf, tf))
            data["Tendance"].append(res.tendance)
            data["RSI"].append(res.rsi_signal)
            data["MACD"].append(res.macd_signal)
            data["Bollinger"].append(res.bb_signal)
            data["Volume"].append("ELEVÉ" if res.volatilite in ("ELEVEE","TRES ELEVEE") else "NORMAL")
    else:
        # Fallback : signaux journaliers seulement
        try:
            from ..analysis.analyzer import TechnicalAnalyzer
            an = TechnicalAnalyzer(df)
            sigs = an.get_signals()
            columns = ["Journalier"]
            data["Tendance"].append(sigs.get("MA_Cross",{}).get("signal","NEUTRE"))
            data["RSI"].append(sigs.get("RSI",{}).get("signal","NEUTRE"))
            data["MACD"].append(sigs.get("MACD",{}).get("signal","NEUTRE"))
            data["Bollinger"].append(sigs.get("Bollinger",{}).get("signal","NEUTRE"))
            data["Volume"].append(sigs.get("Volume",{}).get("signal","NORMAL"))
        except Exception:
            columns = ["N/A"]
            for k in data: data[k] = ["–"]

    if not columns:
        ax.text(0.5,0.5,"Données MTF indisponibles",ha="center",va="center",
                color=c["text"],transform=ax.transAxes,fontsize=9)
        return

    n_rows = len(INDICATORS)
    n_cols = len(columns)
    cell_w = 0.85 / n_cols
    cell_h = 0.72 / n_rows
    x0, y0 = 0.07, 0.12

    # Headers colonnes
    for j, col in enumerate(columns):
        ax.text(x0 + (j+0.5)*cell_w, y0 + n_rows*cell_h + 0.04, col,
                ha="center", va="bottom", fontsize=8, color=c["text"],
                fontweight="bold", transform=ax.transAxes)

    # Headers lignes
    for i, ind in enumerate(INDICATORS):
        ax.text(x0 - 0.01, y0 + (n_rows-1-i+0.5)*cell_h, ind,
                ha="right", va="center", fontsize=7.5, color=c["text"],
                transform=ax.transAxes)

    # Cellules
    for i, ind in enumerate(INDICATORS):
        for j, col in enumerate(columns):
            sig = data[ind][j] if j < len(data[ind]) else "–"
            clr = _sig_color(sig, c)
            rect = mpatches.FancyBboxPatch(
                (x0 + j*cell_w + 0.005, y0 + (n_rows-1-i)*cell_h + 0.005),
                cell_w - 0.01, cell_h - 0.01,
                boxstyle="round,pad=0.01",
                linewidth=0, facecolor=clr + "55",
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            short = sig[:10] if len(sig) > 10 else sig
            ax.text(x0 + (j+0.5)*cell_w, y0 + (n_rows-1-i+0.5)*cell_h,
                    short, ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold", transform=ax.transAxes)

    # Score global si MTF
    if mtf_analyzer and mtf_analyzer._ran and mtf_analyzer._confluence:
        conf = mtf_analyzer._confluence
        sc   = conf.score_pondere
        clr  = c["up"] if sc > 20 else c["down"] if sc < -20 else "#888"
        ax.text(0.5, 0.04, f"Score global : {sc:+.1f}  —  {conf.recommandation}",
                ha="center", va="center", fontsize=8, color=clr,
                fontweight="bold", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc=c["ax_bg"], ec=clr, lw=0.8))


# ─────────────────────────────────────────────
# PANEL 4 — VOLUME PROFILE (VPVR)
# ─────────────────────────────────────────────

def _compute_vpvr(df: pd.DataFrame, n_bins: int = 40):
    lo, hi = df["Low"].min(), df["High"].max()
    if hi <= lo: return np.array([]), np.array([])
    bins   = np.linspace(lo, hi, n_bins + 1)
    vol_pr = np.zeros(n_bins)
    for _, row in df.iterrows():
        clo, chi = row["Low"], row["High"]
        vol  = row["Volume"]
        span = chi - clo
        if span < 1e-9: span = 1e-9
        for b in range(n_bins):
            b_lo, b_hi = bins[b], bins[b+1]
            overlap = min(chi, b_hi) - max(clo, b_lo)
            if overlap > 0:
                vol_pr[b] += vol * (overlap / span)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, vol_pr

def _draw_panel4(ax, df: pd.DataFrame, c: dict):
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("VOLUME PROFILE (VPVR)", color=c["text"], fontsize=9, pad=4)

    try:
        display = df.tail(80)
        centers, vol_pr = _compute_vpvr(display, n_bins=40)
        if len(centers) == 0: raise ValueError("empty")

        max_vol = vol_pr.max()
        if max_vol == 0: raise ValueError("zero volume")

        poc_idx = np.argmax(vol_pr)
        poc_price = centers[poc_idx]

        # Value Area (70% around POC)
        sorted_idx = np.argsort(vol_pr)[::-1]
        total_vol  = vol_pr.sum()
        va_vol, va_bins = 0, []
        for idx in sorted_idx:
            if va_vol >= 0.70 * total_vol: break
            va_bins.append(idx)
            va_vol += vol_pr[idx]
        va_bins = sorted(va_bins)
        vah = centers[max(va_bins)] if va_bins else centers[-1]
        val = centers[min(va_bins)] if va_bins else centers[0]

        # Axe "prix" → positions Y dans [0.08, 0.92] de l'axe normalisé
        p_min, p_max = centers[0], centers[-1]
        p_range = p_max - p_min if p_max > p_min else 1

        def py(price):
            return 0.08 + 0.84 * (price - p_min) / p_range

        # Zone Value Area
        y_val = py(val)
        y_vah = py(vah)
        rect_va = mpatches.FancyBboxPatch(
            (0.55, y_val), 0.42, y_vah - y_val,
            boxstyle="square,pad=0", linewidth=0,
            facecolor="#1e3a5f", alpha=0.45, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect_va)

        # Barres horizontales de volume
        for i, (price, vol) in enumerate(zip(centers, vol_pr)):
            bar_w = 0.40 * (vol / max_vol)
            clr   = "#f0b429" if i == poc_idx else ("#26a641" if price >= poc_price else "#da3633")
            alpha = 0.85 if i == poc_idx else 0.55
            bar   = mpatches.FancyBboxPatch(
                (0.55, py(price) - 0.005), bar_w, 0.012,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=clr, alpha=alpha,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(bar)

        # Prix axis (droite)
        for price in [p_min, val, poc_price, vah, p_max]:
            y = py(price)
            clr = "#f0b429" if price == poc_price else c["text"]
            fw  = "bold" if price == poc_price else "normal"
            ax.text(0.97, y, f"{price:.1f}", ha="right", va="center",
                    fontsize=6.5, color=clr, fontweight=fw,
                    transform=ax.transAxes)

        # Annotations
        ax.text(0.57, py(poc_price), "  POC", ha="left", va="center",
                fontsize=7, color="#f0b429", fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.57, py(vah)+0.01, "  VAH", ha="left", va="bottom",
                fontsize=6.5, color="#58a6ff", transform=ax.transAxes)
        ax.text(0.57, py(val)-0.01, "  VAL", ha="left", va="top",
                fontsize=6.5, color="#58a6ff", transform=ax.transAxes)

        # Mini-chandeliers à gauche (axe temporel)
        n = len(display)
        x_scale = 0.50 / max(n, 1)
        for k in range(n):
            row  = display.iloc[k]
            xc   = 0.01 + k * x_scale
            y_o  = py(row["Open"])
            y_cl = py(row["Close"])
            y_h  = py(row["High"])
            y_l  = py(row["Low"])
            clr  = c["up"] if row["Close"] >= row["Open"] else c["down"]
            body_h = abs(y_cl - y_o)
            body_y = min(y_o, y_cl)
            ax.add_patch(mpatches.Rectangle(
                (xc - x_scale*0.3, body_y), x_scale*0.6, max(body_h, 0.003),
                linewidth=0, facecolor=clr, alpha=0.85,
                transform=ax.transAxes, clip_on=False
            ))
            ax.plot([xc, xc], [y_l, y_h], color=clr, lw=0.5,
                    alpha=0.7, transform=ax.transAxes)

        ax.text(0.25, 0.01, f"Dernières {len(display)} séances",
                ha="center", va="bottom", fontsize=6.5, color=c["text"],
                alpha=0.7, transform=ax.transAxes)

    except Exception as e:
        ax.text(0.5, 0.5, f"Volume Profile\nindisponible\n({e})",
                ha="center", va="center", color=c["text"],
                fontsize=8, transform=ax.transAxes)


# ─────────────────────────────────────────────
# PANEL 5 — OSCILLATEURS (jauges horizontales)
# ─────────────────────────────────────────────

def _draw_panel5(ax, df: pd.DataFrame, c: dict):
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("OSCILLATEURS", color=c["text"], fontsize=9, pad=4)

    from ..indicators.momentum import rsi as calc_rsi, stochastic, cci, williams_r

    try:
        close = df["Close"]
        rsi_v  = calc_rsi(close, 14).iloc[-1]
        stoch  = stochastic(df)["%K"].iloc[-1]
        cci_v  = cci(df, 20).iloc[-1]
        wpr    = williams_r(df, 14).iloc[-1]
    except Exception:
        rsi_v = stoch = cci_v = wpr = np.nan

    oscillators = [
        {
            "name": "RSI (14)",
            "value": rsi_v, "min": 0, "max": 100,
            "zones": [(0,30,"#26a641"),(30,70,"#444"),(70,100,"#da3633")],
            "signal": ("SURVENTE" if rsi_v < 30 else "SURACHAT" if rsi_v > 70 else "NEUTRE") if not np.isnan(rsi_v) else "N/A",
        },
        {
            "name": "Stoch %K (14)",
            "value": stoch, "min": 0, "max": 100,
            "zones": [(0,20,"#26a641"),(20,80,"#444"),(80,100,"#da3633")],
            "signal": ("SURVENTE" if stoch < 20 else "SURACHAT" if stoch > 80 else "NEUTRE") if not np.isnan(stoch) else "N/A",
        },
        {
            "name": "CCI (20)",
            "value": cci_v, "min": -200, "max": 200,
            "zones": [(-200,-100,"#26a641"),(-100,100,"#444"),(100,200,"#da3633")],
            "signal": ("SURVENTE" if cci_v < -100 else "SURACHAT" if cci_v > 100 else "NEUTRE") if not np.isnan(cci_v) else "N/A",
        },
        {
            "name": "Williams %R (14)",
            "value": wpr, "min": -100, "max": 0,
            "zones": [(-100,-80,"#26a641"),(-80,-20,"#444"),(-20,0,"#da3633")],
            "signal": ("SURVENTE" if wpr < -80 else "SURACHAT" if wpr > -20 else "NEUTRE") if not np.isnan(wpr) else "N/A",
        },
    ]

    n = len(oscillators)
    bar_h  = 0.10
    y_start = 0.82

    for i, osc in enumerate(oscillators):
        y = y_start - i * 0.20
        v, mn, mx = osc["value"], osc["min"], osc["max"]

        # Fond de la jauge
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.20, y), 0.60, bar_h,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=c["grid"], alpha=0.7,
            transform=ax.transAxes, clip_on=False
        ))

        # Zones colorées
        for z_lo, z_hi, z_clr in osc["zones"]:
            x0 = 0.20 + 0.60 * (z_lo - mn) / (mx - mn)
            ww = 0.60 * (z_hi - z_lo) / (mx - mn)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y), ww, bar_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=z_clr, alpha=0.18,
                transform=ax.transAxes, clip_on=False
            ))

        # Marqueur valeur actuelle
        if not np.isnan(v):
            xv = 0.20 + 0.60 * np.clip((v - mn) / (mx - mn), 0, 1)
            ax.add_patch(mpatches.FancyBboxPatch(
                (xv - 0.005, y - 0.01), 0.010, bar_h + 0.02,
                boxstyle="round,pad=0.002", linewidth=0,
                facecolor="white", alpha=0.95,
                transform=ax.transAxes, clip_on=False
            ))

        # Textes
        ax.text(0.19, y + bar_h/2, osc["name"],
                ha="right", va="center", fontsize=7.5,
                color=c["text"], transform=ax.transAxes)
        sig_clr = c["up"] if "SURV" in osc["signal"] else c["down"] if "SURACH" in osc["signal"] else c["text"]
        val_str = f"{v:.1f}" if not np.isnan(v) else "N/A"
        ax.text(0.82, y + bar_h/2, f"{val_str}  {osc['signal']}",
                ha="left", va="center", fontsize=7.5,
                color=sig_clr, fontweight="bold", transform=ax.transAxes)


# ─────────────────────────────────────────────
# PANEL 6 — SAISONNALITÉ & PERFORMANCE
# ─────────────────────────────────────────────

def _draw_panel6(ax, df: pd.DataFrame, c: dict):
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("SAISONNALITÉ & PERFORMANCE", color=c["text"], fontsize=9, pad=4)

    try:
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)

        # Rendements mensuels
        monthly = close.resample("ME").last().pct_change() * 100
        monthly.index = monthly.index.to_period("M")
        years  = sorted(monthly.index.year.unique())
        months = list(range(1, 13))
        MONTH_NAMES = ["Jan","Fév","Mar","Avr","Mai","Jun",
                       "Jul","Aoû","Sep","Oct","Nov","Déc"]

        if len(years) >= 2:
            # Heatmap
            grid = np.full((len(years), 12), np.nan)
            for idx in monthly.index:
                yi = years.index(idx.year)
                mi = idx.month - 1
                grid[yi, mi] = monthly[idx]

            ax_inner = ax.inset_axes([0.02, 0.38, 0.96, 0.58])
            ax_inner.set_facecolor(c["ax_bg"])
            vmax = min(10, np.nanpercentile(np.abs(grid[~np.isnan(grid)]), 95)) if np.any(~np.isnan(grid)) else 5
            im = ax_inner.imshow(grid, aspect="auto", cmap="RdYlGn",
                                 vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax_inner.set_xticks(range(12))
            ax_inner.set_xticklabels(MONTH_NAMES, fontsize=6, color=c["text"])
            ax_inner.set_yticks(range(len(years)))
            ax_inner.set_yticklabels([str(y) for y in years], fontsize=6, color=c["text"])
            ax_inner.tick_params(colors=c["text"])
            for spine in ax_inner.spines.values():
                spine.set_edgecolor(c["grid"])
            # Annoter les cellules
            for yi in range(len(years)):
                for mi in range(12):
                    val = grid[yi, mi]
                    if not np.isnan(val):
                        txt_clr = "white" if abs(val) > vmax * 0.5 else c["text"]
                        ax_inner.text(mi, yi, f"{val:.1f}", ha="center", va="center",
                                      fontsize=5, color=txt_clr)
        else:
            ax.text(0.5, 0.7, "Données insuffisantes\npour la heatmap\n(< 2 ans)",
                    ha="center", va="center", fontsize=8, color=c["text"],
                    transform=ax.transAxes)

        # Performance normalisée (base 100)
        ax_perf = ax.inset_axes([0.02, 0.01, 0.96, 0.33])
        ax_perf.set_facecolor(c["ax_bg"])
        perf = (close / close.iloc[0]) * 100
        ax_perf.plot(perf.index, perf.values, color=c["up"], lw=1.2, label=df.attrs.get("symbol","Actif"))
        ax_perf.axhline(100, color=c["grid"], lw=0.6, linestyle="--", alpha=0.5)
        ax_perf.set_ylabel("Base 100", color=c["text"], fontsize=6)
        ax_perf.tick_params(colors=c["text"], labelsize=5.5)
        ax_perf.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for spine in ax_perf.spines.values():
            spine.set_edgecolor(c["grid"])
        ax_perf.grid(True, color=c["grid"], lw=0.4, alpha=0.5)
        perf_total = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
        ax_perf.text(0.99, 0.95, f"Total: {perf_total:+.1f}%",
                     ha="right", va="top", fontsize=7, color=c["up"] if perf_total >= 0 else c["down"],
                     fontweight="bold", transform=ax_perf.transAxes)

    except Exception as e:
        ax.text(0.5, 0.5, f"Saisonnalité\nindisponible\n({e})",
                ha="center", va="center", color=c["text"],
                fontsize=8, transform=ax.transAxes)


# ─────────────────────────────────────────────
# PANEL 7 — VOLATILITÉ & RÉGIMES DE MARCHÉ
# ─────────────────────────────────────────────

def _draw_panel7(ax, df: pd.DataFrame, c: dict):
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("VOLATILITÉ & RÉGIMES", color=c["text"], fontsize=9, pad=4)

    try:
        from ..indicators.volatility import bollinger_bands, atr as calc_atr

        close = df["Close"]
        bb    = bollinger_bands(close, 20)
        bb_w  = bb["BB_Largeur"].dropna()
        atr_v = (calc_atr(df, 14) / close * 100).dropna()

        # Régime par percentile de BB Width
        pct    = bb_w.rank(pct=True)
        regime = pd.Series("NORMAL", index=bb_w.index)
        regime[pct > 0.70] = "TRENDING"
        regime[pct < 0.30] = "COMPRESSION"

        # Panel BB Width (haut)
        ax_bb = ax.inset_axes([0.04, 0.50, 0.92, 0.42])
        ax_bb.set_facecolor(c["ax_bg"])
        for spine in ax_bb.spines.values():
            spine.set_edgecolor(c["grid"])
        ax_bb.tick_params(colors=c["text"], labelsize=5.5)
        ax_bb.grid(True, color=c["grid"], lw=0.4, alpha=0.5)

        # Colorier les régimes en background
        REGIME_COLORS = {"TRENDING":"#1a3a5c","COMPRESSION":"#3a2a00","NORMAL":c["ax_bg"]}
        prev_reg = None
        seg_start = None
        for date, reg in regime.items():
            if reg != prev_reg:
                if prev_reg is not None:
                    ax_bb.axvspan(seg_start, date, alpha=0.35,
                                  color=REGIME_COLORS.get(prev_reg, c["ax_bg"]))
                seg_start = date
                prev_reg = reg
        if prev_reg and seg_start:
            ax_bb.axvspan(seg_start, bb_w.index[-1], alpha=0.35,
                          color=REGIME_COLORS.get(prev_reg, c["ax_bg"]))

        ax_bb.plot(bb_w.index, bb_w.values, color="#f0b429", lw=1.1)
        ax_bb.set_ylabel("BB Width %", color=c["text"], fontsize=6)
        ax_bb.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.setp(ax_bb.get_xticklabels(), visible=False)

        # Régime actuel
        cur_reg = regime.iloc[-1]
        reg_clr = {"TRENDING":"#58a6ff","COMPRESSION":"#f0b429","NORMAL":"#888"}.get(cur_reg,"#888")
        ax_bb.text(0.99, 0.95, f"Régime: {cur_reg}", ha="right", va="top",
                   fontsize=7.5, color=reg_clr, fontweight="bold",
                   transform=ax_bb.transAxes,
                   bbox=dict(boxstyle="round,pad=0.25", fc=c["ax_bg"], ec=reg_clr, lw=0.7))

        # Panel ATR% (bas)
        ax_atr = ax.inset_axes([0.04, 0.04, 0.92, 0.40])
        ax_atr.set_facecolor(c["ax_bg"])
        for spine in ax_atr.spines.values():
            spine.set_edgecolor(c["grid"])
        ax_atr.tick_params(colors=c["text"], labelsize=5.5)
        ax_atr.grid(True, color=c["grid"], lw=0.4, alpha=0.5)
        ax_atr.fill_between(atr_v.index, 0, atr_v.values,
                            color="#bc8cff", alpha=0.45)
        ax_atr.plot(atr_v.index, atr_v.values, color="#bc8cff", lw=0.8)
        ax_atr.set_ylabel("ATR %", color=c["text"], fontsize=6)
        ax_atr.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        plt.setp(ax_atr.get_xticklabels(), rotation=25, ha="right", fontsize=5.5)

        # Légende régimes
        leg = [
            mpatches.Patch(color="#1a3a5c", alpha=0.7, label="TRENDING"),
            mpatches.Patch(color="#3a2a00", alpha=0.7, label="COMPRESSION"),
            mpatches.Patch(color="#444",    alpha=0.7, label="NORMAL"),
        ]
        ax_bb.legend(handles=leg, loc="upper left", fontsize=5.5,
                     facecolor=c["ax_bg"], edgecolor=c["grid"], labelcolor=c["text"])

    except Exception as e:
        ax.text(0.5, 0.5, f"Volatilité\nindisponible\n({e})",
                ha="center", va="center", color=c["text"],
                fontsize=8, transform=ax.transAxes)


# ─────────────────────────────────────────────
# PANEL 8 — SCORE GLOBAL (SPEEDOMETER)
# ─────────────────────────────────────────────

def _draw_panel8(ax, df: pd.DataFrame, analyzer, c: dict):
    ax.axis("off")
    ax.set_facecolor(c["ax_bg"])
    ax.set_title("SYNTHÈSE GLOBALE", color=c["text"], fontsize=9, pad=4)

    # Calculer le score
    score = 0.0
    recommandation = "N/A"
    try:
        if analyzer is None:
            from ..analysis.analyzer import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer(df)
        sc_data = analyzer.score()
        score   = sc_data["score"]
        recommandation = sc_data["recommandation"]
    except Exception:
        pass

    # ── Speedometer (demi-cercle) ──
    cx, cy = 0.50, 0.68
    r_out, r_in = 0.28, 0.16

    # Arcs colorés : -100 → +100 sur 180°
    # angle 180° (gauche) = -100 ; angle 0° (droite) = +100
    arc_zones = [
        (-100, -60, "#8b0000"),   # rouge foncé
        (-60,  -20, "#da3633"),   # rouge
        (-20,   20, "#555555"),   # gris
        ( 20,   60, "#26a641"),   # vert
        ( 60,  100, "#006400"),   # vert foncé
    ]

    for (s_lo, s_hi, clr) in arc_zones:
        theta1 = 180 - (s_hi + 100) * 0.90   # mapping -100..+100 → 180..0
        theta2 = 180 - (s_lo + 100) * 0.90
        wedge  = Wedge((cx, cy), r_out, theta1, theta2,
                       width=r_out - r_in,
                       facecolor=clr, alpha=0.85,
                       transform=ax.transAxes)
        ax.add_patch(wedge)

    # Fond intérieur (demi-cercle)
    wedge_bg = Wedge((cx, cy), r_in, 0, 180,
                     facecolor=c["ax_bg"], alpha=1.0,
                     transform=ax.transAxes)
    ax.add_patch(wedge_bg)

    # Aiguille
    angle_deg = 180 - (score + 100) * 0.90
    angle_rad = np.deg2rad(angle_deg)
    needle_len = r_out * 0.85
    nx = cx + needle_len * np.cos(angle_rad)
    ny = cy + needle_len * np.sin(angle_rad)
    needle_clr = c["up"] if score > 20 else c["down"] if score < -20 else "#aaa"
    ax.annotate("", xy=(nx, ny), xytext=(cx, cy),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->,head_width=0.015,head_length=0.015",
                                color=needle_clr, lw=2.0))
    # Centre de l'aiguille
    ax.add_patch(plt.Circle((cx, cy), 0.018, color=c["text"],
                             transform=ax.transAxes, zorder=5))

    # Repères -100, -50, 0, +50, +100
    for val, label in [(-100,"-100"),(-50,"-50"),(0,"0"),(50,"+50"),(100,"+100")]:
        a_deg = 180 - (val + 100) * 0.90
        a_rad = np.deg2rad(a_deg)
        rx = cx + (r_out + 0.035) * np.cos(a_rad)
        ry = cy + (r_out + 0.035) * np.sin(a_rad)
        ax.text(rx, ry, label, ha="center", va="center",
                fontsize=6, color=c["text"], transform=ax.transAxes)

    # Score au centre
    score_clr = c["up"] if score > 20 else c["down"] if score < -20 else "#aaa"
    ax.text(cx, cy - 0.06, f"{score:+.0f}", ha="center", va="center",
            fontsize=22, color=score_clr, fontweight="bold",
            transform=ax.transAxes)
    ax.text(cx, cy - 0.13, recommandation, ha="center", va="center",
            fontsize=8, color=score_clr, fontweight="bold",
            transform=ax.transAxes)

    # ── Statistiques clés (bas) ──
    close = df["Close"]
    stats = []
    try:
        last  = close.iloc[-1]
        prev  = close.iloc[-2] if len(close) > 1 else last
        var1j = (last / prev - 1) * 100
        hi52  = close.tail(252).max()
        lo52  = close.tail(252).min()
        vol20 = int(df["Volume"].tail(20).mean())
        stats = [
            ("Cours",      f"{last:.2f} MAD"),
            ("Variation",  f"{var1j:+.2f}%",  c["up"] if var1j >= 0 else c["down"]),
            ("52s Haut",   f"{hi52:.2f}"),
            ("52s Bas",    f"{lo52:.2f}"),
            ("Vol moy 20j",f"{vol20:,}"),
        ]
    except Exception:
        pass

    y_s = 0.30
    for item in stats:
        label_s, value_s = item[0], item[1]
        val_clr = item[2] if len(item) > 2 else c["text"]
        ax.text(0.18, y_s, label_s, ha="left",  va="center", fontsize=7.5,
                color=c["text"], alpha=0.75, transform=ax.transAxes)
        ax.text(0.82, y_s, value_s, ha="right", va="center", fontsize=7.5,
                color=val_clr, fontweight="bold", transform=ax.transAxes)
        ax.plot([0.16, 0.84], [y_s - 0.025, y_s - 0.025],
                color=c["grid"], lw=0.4, alpha=0.5, transform=ax.transAxes)
        y_s -= 0.055


# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def plot_dashboard(
    df: pd.DataFrame,
    overlays: Optional[List[str]] = None,
    mtf_analyzer=None,
    analyzer=None,
    figsize: tuple = (32, 18),
    save_path: Optional[str] = None,
    dark_theme: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Dashboard 8 panneaux pour l'analyse technique BVC.

    Args:
        df           : DataFrame OHLCV (données journalières recommandées)
        overlays     : Liste des tracés pour le Panel 2.
                       None → tous actifs.
                       Valeurs possibles : "fibonacci", "trendlines", "ichimoku",
                       "pivots", "patterns", "support_resistance",
                       "regression", "candlestick_patterns"
        mtf_analyzer : Instance MultiTimeframeAnalyzer déjà run() (optionnel)
        analyzer     : Instance TechnicalAnalyzer (optionnel, créé si absent)
        figsize      : Taille de la figure (défaut: 32×18)
        save_path    : Chemin PNG pour sauvegarde
        dark_theme   : Thème sombre (défaut: True)
        title        : Titre global (optionnel)

    Returns:
        Figure matplotlib
    """
    c = _c(dark_theme)
    if overlays is None:
        overlays = list(AVAILABLE_OVERLAYS.keys())

    # Limiter les données affichées pour la performance
    display_df = df.tail(200).copy()

    # ── Layout 2 lignes × 4 colonnes ──
    fig = plt.figure(figsize=figsize, facecolor=c["bg"])

    # Ligne titre
    fig.suptitle(
        title or f"{df.attrs.get('symbol','?')} — Dashboard Analyse Technique BVC",
        fontsize=14, color=c["text"], fontweight="bold", y=0.985,
    )

    outer = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.30, wspace=0.28,
        top=0.96, bottom=0.04,
        left=0.03, right=0.98,
    )

    # Panel 1 : OHLCV (avec sous-panneau volume)
    p1_inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0],
        height_ratios=[4, 1], hspace=0.04,
    )
    ax_p1_price = fig.add_subplot(p1_inner[0])
    ax_p1_vol   = fig.add_subplot(p1_inner[1], sharex=ax_p1_price)
    _apply_dark_theme(fig, [ax_p1_price, ax_p1_vol])

    # Panels 2–8 (simples)
    ax_p2 = fig.add_subplot(outer[0, 1])
    ax_p3 = fig.add_subplot(outer[0, 2])
    ax_p4 = fig.add_subplot(outer[0, 3])
    ax_p5 = fig.add_subplot(outer[1, 0])
    ax_p6 = fig.add_subplot(outer[1, 1])
    ax_p7 = fig.add_subplot(outer[1, 2])
    ax_p8 = fig.add_subplot(outer[1, 3])
    _apply_dark_theme(fig, [ax_p2, ax_p3, ax_p4, ax_p5, ax_p6, ax_p7, ax_p8])

    # ── Dessiner chaque panel ──
    try: _draw_panel1(ax_p1_price, ax_p1_vol, display_df, c)
    except Exception as e: ax_p1_price.text(0.5,0.5,f"P1 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p1_price.transAxes)

    try: _draw_panel2(ax_p2, display_df, overlays, c)
    except Exception as e: ax_p2.text(0.5,0.5,f"P2 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p2.transAxes)

    try: _draw_panel3(ax_p3, mtf_analyzer, display_df, c)
    except Exception as e: ax_p3.text(0.5,0.5,f"P3 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p3.transAxes)

    try: _draw_panel4(ax_p4, display_df, c)
    except Exception as e: ax_p4.text(0.5,0.5,f"P4 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p4.transAxes)

    try: _draw_panel5(ax_p5, display_df, c)
    except Exception as e: ax_p5.text(0.5,0.5,f"P5 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p5.transAxes)

    try: _draw_panel6(ax_p6, df, c)  # df complet pour la saisonnalité
    except Exception as e: ax_p6.text(0.5,0.5,f"P6 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p6.transAxes)

    try: _draw_panel7(ax_p7, display_df, c)
    except Exception as e: ax_p7.text(0.5,0.5,f"P7 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p7.transAxes)

    try: _draw_panel8(ax_p8, display_df, analyzer, c)
    except Exception as e: ax_p8.text(0.5,0.5,f"P8 erreur:\n{e}",ha="center",va="center",color=c["text"],transform=ax_p8.transAxes)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Dashboard sauvegardé : {save_path}")

    return fig
