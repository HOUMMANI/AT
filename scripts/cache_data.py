"""
Script de mise en cache des données BVC.
Lancé par GitHub Actions — IPs non bloquées par Yahoo Finance.
Sauvegarde un CSV par symbole dans data/cache/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Tous les tickers BVC
try:
    from src.data.tickers import BVC_TICKERS
    SYMBOLS = {sym: info.get("yahoo", sym + ".CS") for sym, info in BVC_TICKERS.items()}
except ImportError:
    SYMBOLS = {
        "ATW": "ATW.CS", "IAM": "IAM.CS", "BCP": "BCP.CS", "BOA": "BOA.CS",
        "CIH": "CIH.CS", "WAA": "WAA.CS", "LHM": "LHM.CS", "MNG": "MNG.CS",
        "COSU": "COSU.CS", "LES": "LES.CS", "HPS": "HPS.CS", "TMA": "TMA.CS",
        "MSA": "MSA.CS", "ADI": "ADI.CS", "AFM": "AFM.CS", "ALM": "ALM.CS",
        "CMT": "CMT.CS", "DHO": "DHO.CS", "SNA": "SNA.CS", "JET": "JET.CS",
    }

PERIODS = {
    "1y":  365,
    "5y":  1825,
}


def fetch_symbol(yahoo_sym: str, days: int = 365) -> pd.DataFrame:
    """Fetch avec retry depuis GitHub Actions (pas de rate-limit Yahoo)."""
    for attempt in range(3):
        try:
            ticker = yf.Ticker(yahoo_sym)
            df = ticker.history(period=f"{days}d", auto_adjust=True)
            if df is not None and not df.empty:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print(f"  Tentative {attempt+1}/3 échouée pour {yahoo_sym}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def save_cache(sym: str, df: pd.DataFrame):
    """Sauvegarde le DataFrame en CSV dans data/cache/."""
    if df.empty:
        return
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df.sort_index()
    path = os.path.join(CACHE_DIR, f"{sym}.csv")
    df.to_csv(path)
    print(f"  ✓ {sym}: {len(df)} lignes → {path}")


def main():
    print(f"\n=== Cache BVC Data — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    ok, fail = 0, 0
    for sym, yahoo_sym in SYMBOLS.items():
        print(f"Fetching {sym} ({yahoo_sym})...")
        df = fetch_symbol(yahoo_sym, days=365 * 5)  # 5 ans d'historique
        if not df.empty:
            save_cache(sym, df)
            ok += 1
        else:
            print(f"  ✗ {sym}: aucune donnée")
            fail += 1
        time.sleep(0.5)  # pause courtoise

    # Mettre à jour le timestamp de la dernière mise à jour
    meta_path = os.path.join(CACHE_DIR, "_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"last_update: {datetime.now().isoformat()}\n")
        f.write(f"symbols_ok: {ok}\n")
        f.write(f"symbols_fail: {fail}\n")

    print(f"\n=== Terminé: {ok} OK, {fail} échecs ===\n")


if __name__ == "__main__":
    main()
