import streamlit as st
import pandas as pd
from nse_stock_picker_pro import fetch_nifty100_symbols, analyze

st.set_page_config(page_title="NSE Stock Picker Pro", layout="wide")

st.title("📈 NSE Stock Picker Pro")
st.markdown("""
A multi-factor stock screener for Nifty 100
Intraday to short-term (1–10 days)
Uses ATR, Risk:Reward, MACD, RSI, ADX, volume surge, and more.
""")

run_button = st.button("🔍 Run Stock Screener")

if run_button:
    st.info("Running analysis... please wait (30–60s).")
    picks = []

    symbols = fetch_nifty100_symbols()
    for i, symbol in enumerate(symbols, 1):
        res = analyze(symbol)
        if res and res["Score"] >= 6 and res["RR"] <= 0.6:
            picks.append(res)

    if not picks:
        st.error("❌ No qualifying stocks found.")
    else:
        df = pd.DataFrame(picks).sort_values(["Score", "ADX"], ascending=[False, False])
        st.success(f"✅ Found {len(df)} strong candidate(s).")
        st.dataframe(df[["Ticker", "Price", "Score", "RR", "Target", "Stop", "RSI", "ADX"]].reset_index(drop=True))
