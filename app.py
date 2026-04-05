"""
AT — Analyse Technique BVC
Application Streamlit pour l'analyse des actions de la Bourse de Casablanca.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use("Agg")

# ─── Config page ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AT — Analyse Technique BVC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS dark theme léger ────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0e1117; }
.metric-card {
    background: #1a1d27;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    border-left: 3px solid #00d4aa;
}
.up   { color: #00d4aa; font-weight: bold; }
.down { color: #ff4b4b; font-weight: bold; }
h1, h2, h3 { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ─── Imports projet ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_tickers():
    from src.data.tickers import BVC_TICKERS, list_sectors
    return BVC_TICKERS, list_sectors()

try:
    BVC_TICKERS, SECTORS = _load_tickers()
except Exception as e:
    st.error(f"Erreur de chargement des tickers: {e}")
    st.stop()

# ─── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 AT — BVC")
    st.caption("Analyse Technique · Bourse de Casablanca")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 Analyse", "📊 Dashboard", "🌐 Multi-Timeframe", "🏪 Marché", "⚖️ Comparaison"],
        label_visibility="collapsed",
    )

    st.divider()

    symbol_list = sorted(BVC_TICKERS.keys())
    symbol_names = {s: f"{s} — {BVC_TICKERS[s]['name']}" for s in symbol_list}

    default_idx = symbol_list.index("ATW") if "ATW" in symbol_list else 0
    selected_symbol = st.selectbox(
        "Symbole",
        symbol_list,
        index=default_idx,
        format_func=lambda s: symbol_names[s],
    )

    period = st.select_slider(
        "Période",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        value="1y",
    )

    st.divider()
    st.caption("Données: Yahoo Finance (BVC ~15 min delay)")

# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(symbol, per):
    """Fetch OHLCV with retry + exponential backoff for Yahoo rate limits."""
    import yfinance as yf
    yahoo_sym = BVC_TICKERS.get(symbol, {}).get("yahoo", symbol + ".CS")
    delays = [3, 8, 20]
    for attempt, delay in enumerate(delays + [None]):
        try:
            ticker = yf.Ticker(yahoo_sym)
            df = ticker.history(period=per, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception as e:
            err = str(e).lower()
            if "ratelimit" in err or "429" in err or "too many" in err:
                if delay is not None:
                    time.sleep(delay)
                    continue
            return None
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_realtime(symbol):
    try:
        import yfinance as yf
        yahoo_sym = BVC_TICKERS.get(symbol, {}).get("yahoo", symbol + ".CS")
        ticker = yf.Ticker(yahoo_sym)
        info = ticker.fast_info
        return {
            "price": getattr(info, "last_price", None),
            "open": getattr(info, "open", None),
            "high": getattr(info, "day_high", None),
            "low": getattr(info, "day_low", None),
            "volume": getattr(info, "last_volume", None),
            "high_52w": getattr(info, "year_high", None),
            "low_52w": getattr(info, "year_low", None),
        }
    except Exception:
        return None


def color(val, positive_good=True):
    if val is None:
        return "—"
    ok = val >= 0 if positive_good else val <= 0
    css = "up" if ok else "down"
    sign = "+" if val > 0 else ""
    return f'<span class="{css}">{sign}{val:.2f}%</span>'


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyse":
    info = BVC_TICKERS.get(selected_symbol, {})
    st.title(f"{selected_symbol} — {info.get('name', '')}")
    st.caption(f"Secteur: {info.get('secteur', '—')}  ·  Yahoo: {info.get('yahoo', '—')}")

    with st.spinner("Récupération des données..."):
        df = fetch_data(selected_symbol, period)
        qt = fetch_realtime(selected_symbol)

    if df is None or df.empty:
        st.warning("⏳ Yahoo Finance limite les requêtes depuis les serveurs cloud. Réessaie dans quelques secondes.")
        st.stop()

    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else last
    var_1j = (last - prev) / prev * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dernier cours", f"{last:.2f} MAD",
              delta=f"{var_1j:+.2f}%",
              delta_color="normal")

    if qt and qt.get("open"):
        c2.metric("Ouverture", f"{qt['open']:.2f}")
        c3.metric("Haut", f"{qt['high']:.2f}")
        c4.metric("Bas", f"{qt['low']:.2f}")
        c5.metric("Volume", f"{int(qt['volume']):,}" if qt.get("volume") else "—")
    else:
        c2.metric("Haut 52s", f"{df['High'].tail(252).max():.2f}")
        c3.metric("Bas 52s", f"{df['Low'].tail(252).min():.2f}")
        c4.metric("Vol. moyen", f"{int(df['Volume'].mean()):,}")
        c5.metric("Nb. séances", str(len(df)))

    st.divider()

    col_chart, col_report = st.columns([2, 1])

    with col_chart:
        chart_opts = st.multiselect(
            "Options graphique",
            ["Bollinger", "MACD", "RSI", "Patterns", "Fibonacci", "Trendlines"],
            default=["Bollinger", "RSI"],
        )

        with st.spinner("Calcul des indicateurs..."):
            try:
                from src.analysis import TechnicalAnalyzer
                from src.visualization.charts import plot_chart

                analyzer = TechnicalAnalyzer(df)
                df_ind = analyzer.compute_all()

                fig = plot_chart(
                    df_ind,
                    show_volume=True,
                    show_bollinger="Bollinger" in chart_opts,
                    show_macd="MACD" in chart_opts,
                    show_rsi="RSI" in chart_opts,
                    show_patterns="Patterns" in chart_opts,
                    show_fibonacci="Fibonacci" in chart_opts,
                    show_trendlines="Trendlines" in chart_opts,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Erreur graphique: {e}")

    with col_report:
        st.subheader("Rapport")
        with st.spinner("Analyse..."):
            try:
                from src.analysis import TechnicalAnalyzer
                analyzer = TechnicalAnalyzer(df)
                score_data = analyzer.score()

                score = score_data["score"]
                reco = score_data["recommandation"]

                score_color = "#00d4aa" if score > 0 else "#ff4b4b"
                st.markdown(f"""
                <div style="text-align:center; padding:20px; background:#1a1d27;
                            border-radius:12px; margin-bottom:16px;">
                    <div style="font-size:42px; font-weight:bold; color:{score_color};">
                        {score:+.1f}
                    </div>
                    <div style="font-size:18px; color:#aaa; margin-top:4px;">
                        {reco}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Rapport complet", expanded=False):
                    report_txt = analyzer.full_report(include_patterns=False)
                    st.text(report_txt)

                st.subheader("Signaux")
                signals = analyzer.get_signals()
                for sig in signals[:8]:
                    icon = "🟢" if "ACHAT" in sig.upper() or "HAUSSIER" in sig.upper() else \
                           "🔴" if "VENTE" in sig.upper() or "BAISSIER" in sig.upper() else "🟡"
                    st.markdown(f"{icon} {sig}")

            except Exception as e:
                st.error(f"Erreur analyse: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    info = BVC_TICKERS.get(selected_symbol, {})
    st.title(f"Dashboard — {selected_symbol}")

    with st.sidebar:
        st.subheader("Overlays panel 2")
        OVERLAYS_ALL = [
            "fibonacci", "trendlines", "ichimoku", "pivots",
            "patterns", "support_resistance", "regression", "candlestick_patterns"
        ]
        selected_overlays = st.multiselect(
            "Tracés avancés",
            OVERLAYS_ALL,
            default=["fibonacci", "trendlines", "support_resistance"],
        )
        no_mtf = st.checkbox("Sans multi-timeframe", value=False)

    with st.spinner("Chargement des données..."):
        df = fetch_data(selected_symbol, period)

    if df is None or df.empty:
        st.warning("⏳ Données indisponibles. Réessaie dans quelques secondes.")
        st.stop()

    with st.spinner("Génération du dashboard 8 panneaux..."):
        try:
            from src.analysis import TechnicalAnalyzer, MultiTimeframeAnalyzer
            from src.visualization.dashboard import plot_dashboard

            analyzer = TechnicalAnalyzer(df)

            mtf = None
            if not no_mtf:
                try:
                    from src.data import BVCDataFetcher
                    fetcher = BVCDataFetcher()
                    mtf = MultiTimeframeAnalyzer(selected_symbol, fetcher=fetcher)
                    mtf.run()
                except Exception:
                    mtf = None

            fig = plot_dashboard(
                df,
                overlays=selected_overlays if selected_overlays else None,
                mtf_analyzer=mtf,
                analyzer=analyzer,
                figsize=(28, 16),
                title=f"{selected_symbol} — {info.get('name', '')} · Dashboard BVC",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.error(f"Erreur dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MULTI-TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Multi-Timeframe":
    st.title(f"Multi-Timeframe — {selected_symbol}")

    with st.spinner("Chargement des 3 timeframes..."):
        try:
            from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
            from src.visualization.mtf_charts import plot_mtf_overview, plot_mtf_confluence
            from src.data import BVCDataFetcher

            fetcher = BVCDataFetcher()
            mtf = MultiTimeframeAnalyzer(selected_symbol, timeframes=["1d", "1wk", "1mo"], fetcher=fetcher)
            mtf.run()

            with st.expander("Rapport MTF", expanded=True):
                st.text(mtf.full_report())

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Vue d'ensemble")
                fig1 = plot_mtf_overview(mtf)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with col2:
                st.subheader("Confluence")
                fig2 = plot_mtf_confluence(mtf)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        except Exception as e:
            st.error(f"Erreur MTF: {e}")
            import traceback
            st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MARCHÉ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏪 Marché":
    st.title("Aperçu du Marché BVC")

    with st.sidebar:
        market_period = st.select_slider(
            "Période marché",
            options=["1mo", "3mo", "6mo", "1y"],
            value="1mo",
        )
        sector_filter = st.selectbox(
            "Secteur",
            ["Tous"] + sorted(SECTORS),
        )

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_market(per):
        rows = []
        for sym, sym_info in list(BVC_TICKERS.items())[:25]:
            df_s = fetch_data(sym, per)
            if df_s is not None and not df_s.empty:
                last = df_s["Close"].iloc[-1]
                prev = df_s["Close"].iloc[-2] if len(df_s) > 1 else last
                rows.append({
                    "Symbole": sym,
                    "Nom": sym_info.get("name", sym)[:28],
                    "Secteur": sym_info.get("secteur", "—"),
                    "Dernier cours": last,
                    "Variation (%)": (last - prev) / prev * 100,
                })
            time.sleep(0.3)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    with st.spinner("Récupération des données marché (peut prendre ~30s)..."):
        overview = fetch_market(market_period)

    if overview is None or overview.empty:
        st.warning("⏳ Données indisponibles — Yahoo Finance limite les requêtes. Réessaie dans 1 minute.")
        st.stop()
    else:
        if sector_filter != "Tous":
            filtered_syms = {s for s, i in BVC_TICKERS.items() if i.get("secteur") == sector_filter}
            overview = overview[overview["Symbole"].isin(filtered_syms)]

        var = overview["Variation (%)"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valeurs analysées", len(overview))
        c2.metric("En hausse", int((var > 0).sum()))
        c3.metric("En baisse", int((var < 0).sum()))
        c4.metric("Variation moyenne", f"{var.mean():+.2f}%",
                  delta=f"{var.mean():+.2f}%", delta_color="normal")

        st.divider()

        display = overview.sort_values("Variation (%)", ascending=False)
        st.dataframe(
            display.style
                .applymap(lambda v: "color: #00d4aa" if v >= 0 else "color: #ff4b4b",
                          subset=["Variation (%)"])
                .format({"Dernier cours": "{:.2f}", "Variation (%)": "{:+.2f}%"}),
            use_container_width=True,
            height=600,
        )

        st.subheader("Variation par action")
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in display["Variation (%)"]]
        ax.bar(display["Symbole"], display["Variation (%)"], color=colors, width=0.7)
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.tick_params(colors="#aaa", rotation=45)
        ax.set_ylabel("Variation (%)", color="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — COMPARAISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Comparaison":
    st.title("Comparaison de valeurs")

    with st.sidebar:
        compare_symbols = st.multiselect(
            "Valeurs à comparer",
            symbol_list,
            default=["ATW", "BCP", "IAM"] if all(s in symbol_list for s in ["ATW", "BCP", "IAM"])
                    else symbol_list[:3],
            format_func=lambda s: symbol_names[s],
        )
        compare_period = st.select_slider(
            "Période",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            value="6mo",
            key="cmp_period",
        )
        normalize = st.checkbox("Normaliser (base 100)", value=True)

    if len(compare_symbols) < 2:
        st.info("Sélectionnez au moins 2 valeurs dans la barre latérale.")
        st.stop()

    results = []
    close_data = {}

    progress = st.progress(0, text="Chargement...")
    for i, sym in enumerate(compare_symbols):
        progress.progress((i + 1) / len(compare_symbols), text=f"Chargement {sym}...")
        df_s = fetch_data(sym, compare_period)
        if df_s is not None and not df_s.empty:
            close_data[sym] = df_s["Close"]
            try:
                from src.analysis import TechnicalAnalyzer
                analyzer = TechnicalAnalyzer(df_s)
                score_data = analyzer.score()
                last_price = df_s["Close"].iloc[-1]
                prev_price = df_s["Close"].iloc[-2]
                results.append({
                    "Symbole": sym,
                    "Nom": BVC_TICKERS[sym]["name"][:30],
                    "Cours": last_price,
                    "1j (%)": (last_price - prev_price) / prev_price * 100,
                    f"Perf. {compare_period} (%)": (df_s["Close"].iloc[-1] / df_s["Close"].iloc[0] - 1) * 100,
                    "Score": score_data["score"],
                    "Signal": score_data["recommandation"],
                })
            except Exception:
                pass
    progress.empty()

    if results:
        df_res = pd.DataFrame(results).set_index("Symbole")
        st.subheader("Tableau comparatif")
        st.dataframe(
            df_res.style
                .applymap(lambda v: "color: #00d4aa" if v >= 0 else "color: #ff4b4b",
                          subset=["1j (%)", f"Perf. {compare_period} (%)", "Score"])
                .format({
                    "Cours": "{:.2f}",
                    "1j (%)": "{:+.2f}%",
                    f"Perf. {compare_period} (%)": "{:+.2f}%",
                    "Score": "{:+.1f}",
                }),
            use_container_width=True,
        )

    if close_data:
        st.subheader("Évolution des cours")
        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        palette = ["#00d4aa", "#4fc3f7", "#ffb74d", "#f48fb1", "#ce93d8",
                   "#80cbc4", "#a5d6a7", "#fff176", "#ff8a65", "#90a4ae"]
        for i, (sym, series) in enumerate(close_data.items()):
            if normalize:
                series = series / series.iloc[0] * 100
            ax.plot(series.index, series.values, label=sym,
                    color=palette[i % len(palette)], linewidth=1.8)
        ax.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="#ddd")
        ax.tick_params(colors="#aaa")
        ax.set_ylabel("Base 100" if normalize else "MAD", color="#aaa")
        ax.grid(color="#2a2a2a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if normalize:
            ax.axhline(100, color="#555", linewidth=0.8, linestyle="--")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
