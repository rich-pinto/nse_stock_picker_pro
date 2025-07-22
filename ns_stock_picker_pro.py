"""
nse_stock_picker_pro.py
───────────────────────
Short‑lists up to 5 Nifty‑100 stocks for intraday↔10‑day trades
using an advanced multi‑factor score and risk:reward gating.

DEPENDENCIES
------------
pip install yfinance pandas ta numpy
"""

import warnings, requests, pandas as pd, numpy as np, yfinance as yf
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime

warnings.filterwarnings("ignore")

# ────────── CONFIG ──────────
LOOKBACK_DAYS = "6mo"        # more history for ATR/ADX
EMA1, EMA2, EMA3 = 20, 50, 100
ADX_THRESH       = 20        # trend strength
VOL_MULT         = 1.5       # volume surge factor
BB_SQUEEZE_PCT   = 0.5       # BB width < 50 % of 120‑day avg
ATR_STOP_MULT    = 1.0       # Stop = Price − 1·ATR
ATR_TGT_MULT     = 2.0       # Target = Price + 2·ATR   →  R:R ≈ 0.5
MAX_RR           = 0.6       # keep trades with RR ≤ 0.6
MIN_SCORE        = 6         # out of 10
TOP_N            = 5         # how many picks to print
# ────────────────────────────

def fetch_nifty100_symbols() -> list[str]:
    url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    return [f"{s.strip()}.NS" for s in pd.read_csv(url)["Symbol"].dropna()]

def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Handle MultiIndex columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def analyze(ticker: str) -> dict | None:
    df = yf.download(ticker, period=LOOKBACK_DAYS, interval="1d",
                     progress=False, auto_adjust=False, threads=False)
    if df.empty:
        return None
    df = flatten_df(df).dropna()

    if len(df) < 120:
        return None  # not enough history

    close, high, low, volume = (df[c].astype(float) for c in ["Close", "High", "Low", "Volume"])

    # ── Indicators ─────────────────────────────────────────────
    ema20  = EMAIndicator(close, window=EMA1).ema_indicator()
    ema50  = EMAIndicator(close, window=EMA2).ema_indicator()
    ema100 = EMAIndicator(close, window=EMA3).ema_indicator()

    macd    = MACD(close)
    macd_line, macd_signal = macd.macd(), macd.macd_signal()

    rsi      = RSIIndicator(close).rsi()
    adx      = ADXIndicator(high, low, close).adx()
    atr      = AverageTrueRange(high, low, close).average_true_range()

    bb       = BollingerBands(close)
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # ── Feature Flags ──────────────────────────────────────────
    today = -1
    f_close_above_ema20     = close[today] > ema20[today]
    f_macd_line_cross_up    = macd_line[today] > macd_signal[today]
    f_rsi_mid_cross         = (rsi.iloc[-2] < 50) and (rsi[today] > 50)
    f_volume_surge          = volume[today] > VOL_MULT * volume.rolling(20).mean()[today]
    f_adx_trending          = adx[today] > ADX_THRESH
    f_bb_squeeze_break      = bb_width[today] > BB_SQUEEZE_PCT * bb_width.rolling(120).mean()[today]
    f_price_near_ema_cluster = (
        abs(close[today] - ema20[today]) / close[today] < 0.01 and
        abs(close[today] - ema50[today]) / close[today] < 0.01 and
        abs(close[today] - ema100[today]) / close[today] < 0.015
    )
    f_breakout_20d_high     = close[today] >= close.rolling(20).max()[today]
    f_5d_momentum           = close[today] > close.shift(5)[today]

    # ── Scoring (0‑10) ────────────────────────────────────────
    score = sum([
        f_close_above_ema20,
        f_macd_line_cross_up,
        45 < rsi[today] < 65,
        f_volume_surge,
        f_5d_momentum,
        f_adx_trending,
        f_bb_squeeze_break,
        f_rsi_mid_cross,
        f_price_near_ema_cluster,
        f_breakout_20d_high
    ])

    # ── Risk:Reward ───────────────────────────────────────────
    stop   = round(close[today] - ATR_STOP_MULT * atr[today], 2)
    target = round(close[today] + ATR_TGT_MULT * atr[today], 2)
    rr     = round((ATR_STOP_MULT * atr[today]) / (ATR_TGT_MULT * atr[today]), 2)  # ≈0.5

    return {
        "Ticker": ticker.replace(".NS", ""),
        "Price": round(close[today], 2),
        "Score": score,
        "RR": rr,
        "Target": target,
        "Stop": stop,
        "RSI": round(rsi[today], 1),
        "ADX": round(adx[today], 1),
        "VolSurge": f_volume_surge
    }

def main():
    print(f"\n📈 NSE Stock‑Picker Pro — {datetime.now():%d‑%b‑%Y %H:%M %Z}\n")
    picks = []
    for i, sym in enumerate(fetch_nifty100_symbols(), 1):
        res = analyze(sym)
        if res and res["Score"] >= MIN_SCORE and res["RR"] <= MAX_RR:
            picks.append(res)

    if not picks:
        print("❌ No stocks met all criteria today.")
        return

    df = (pd.DataFrame(picks)
            .sort_values(["Score", "VolSurge", "ADX"], ascending=[False]*3)
            .head(TOP_N)
            .reset_index(drop=True))

    print("🏆 Candidates (holding window ≈1‑10 days)\n")
    print(df[["Ticker","Price","Score","RR","Target","Stop","RSI","ADX"]]
            .to_string(index=False, justify="center"))

    print("\nℹ️  Filters used:")
    print("   • Score ≥", MIN_SCORE, "out of 10 (EMA trend, MACD cross, RSI, volume surge …)")
    print("   • Risk:Reward ≤", MAX_RR, "(ATR‑based 1:2 default)")
    print("   • ADX >", ADX_THRESH, "for trend strength")
    print("   • Bollinger squeeze + breakout confirmation")
    print("🔔 Always double‑check news, earnings dates, and liquidity before entering.")

if __name__ == "__main__":
    main()
