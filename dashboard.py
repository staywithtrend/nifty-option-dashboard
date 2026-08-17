import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty Signal Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for clean UI metrics
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.markdown("## 🔑 FYERS API Login")

app_id = st.sidebar.text_input("FYERS App ID", value="IONVEW8SCZ-100", help="Must include -100 suffix")
access_token = st.sidebar.text_input("FYERS Access Token", type="password", help="Paste long JWT access token")

st.sidebar.markdown("---")
selected_index = st.sidebar.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
auto_refresh = st.sidebar.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=60)
oi_lookback = st.sidebar.slider("OI-change comparison window (minutes)", min_value=5, max_value=60, value=15)

# Symbol & Strike Step Mapping
SYMBOL_CONFIG = {
    "NIFTY": {"fyers_symbol": "NSE:NIFTY50-INDEX", "step": 50},
    "BANKNIFTY": {"fyers_symbol": "NSE:NIFTYBANK-INDEX", "step": 100},
    "FINNIFTY": {"fyers_symbol": "NSE:FINNIFTY-INDEX", "step": 50}
}

# -------------------------------------------------------------------
# API DATA FETCHERS
# -------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_fyers_data(app_id, token, symbol):
    """Fetch live option chain and spot price quotes directly from FYERS v3 REST API"""
    if not token or not app_id:
        return None, None, "App ID and Access Token are required."

    headers = {
        "Authorization": f"{app_id}:{token}",
        "Content-Type": "application/json"
    }
    
    # 1. Fetch Option Chain
    chain_url = f"https://api-t1.fyers.in/data/options-chain-v3?symbol={symbol}&strikecount=20"
    quotes_url = f"https://api-t1.fyers.in/data/quotes?symbols={symbol}"
    
    spot_price = 0.0
    
    try:
        # Fetch Spot Price via Quotes API
        q_res = requests.get(quotes_url, headers=headers, timeout=8)
        if q_res.status_code == 200:
            q_data = q_res.json()
            if q_data.get("s") == "ok" and "d" in q_data and len(q_data["d"]) > 0:
                spot_price = float(q_data["d"][0].get("v", {}).get("lp", 0.0))

        # Fetch Option Chain Data
        c_res = requests.get(chain_url, headers=headers, timeout=10)
        if c_res.status_code != 200:
            return None, spot_price, f"HTTP {c_res.status_code}: {c_res.text}"

        c_data = c_res.json()
        if c_data.get("s") != "ok":
            return None, spot_price, f"FYERS Error ({c_data.get('code')}): {c_data.get('message', 'Failed to fetch chain')}"

        payload = c_data.get("data", {})
        
        # Fallback for spot price if quote api wasn't populated
        if spot_price == 0.0:
            spot_price = float(payload.get("underVal", 0.0))

        return payload, spot_price, None

    except Exception as e:
        return None, 0.0, f"Connection Exception: {str(e)}"

# -------------------------------------------------------------------
# DASHBOARD LOGIC
# -------------------------------------------------------------------
if access_token:
    config = SYMBOL_CONFIG[selected_index]
    chain_payload, spot_price, error_msg = fetch_fyers_data(app_id, access_token, config["fyers_symbol"])

    if error_msg:
        st.error(f"❌ {error_msg}")
        st.info("💡 Ensure your Access Token is fresh and your App ID ends with `-100`.")
    elif chain_payload and "optionsChain" in chain_payload:
        raw_options = chain_payload["optionsChain"]
        df = pd.DataFrame(raw_options)

        if not df.empty:
            # Separate Call & Put Dataframes
            ce_df = df[df['option_type'] == 'CE'].copy()
            pe_df = df[df['option_type'] == 'PE'].copy()

            # Calculate ATM Strike
            strike_step = config["step"]
            atm_strike = round(spot_price / strike_step) * strike_step if spot_price > 0 else df['strike_price'].median()

            # Totals & PCR
            total_call_oi = int(ce_df['oi'].sum()) if 'oi' in ce_df.columns else 0
            total_put_oi = int(pe_df['oi'].sum()) if 'oi' in pe_df.columns else 0
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

            # Max Pain Calculation
            strikes = df['strike_price'].unique()
            pain_dict = {}
            for s in strikes:
                c_loss = ce_df.apply(lambda r: max(0, s - r['strike_price']) * r['oi'], axis=1).sum() if not ce_df.empty else 0
                p_loss = pe_df.apply(lambda r: max(0, r['strike_price'] - s) * r['oi'], axis=1).sum() if not pe_df.empty else 0
                pain_dict[s] = c_loss + p_loss
            
            max_pain = min(pain_dict, key=pain_dict.get) if pain_dict else atm_strike

            # ATM Pricing & Straddle Calculation
            atm_ce = ce_df[ce_df['strike_price'] == atm_strike]
            atm_pe = pe_df[pe_df['strike_price'] == atm_strike]
            
            ce_ltp = float(atm_ce['ltp'].values[0]) if not atm_ce.empty and 'ltp' in atm_ce.columns else 0.0
            pe_ltp = float(atm_pe['ltp'].values[0]) if not atm_pe.empty and 'ltp' in atm_pe.columns else 0.0
            
            atm_straddle = round(ce_ltp + pe_ltp, 2)
            expected_move = round(atm_straddle * 0.85, 1)

            # IV Skew Calculation
            ce_iv = float(atm_ce['iv'].values[0]) if not atm_ce.empty and 'iv' in atm_ce.columns and atm_ce['iv'].values[0] is not None else 0.0
            pe_iv = float(atm_pe['iv'].values[0]) if not atm_pe.empty and 'iv' in atm_pe.columns and atm_pe['iv'].values[0] is not None else 0.0
            iv_skew = round(pe_iv - ce_iv, 2)

            # Positioning Read
            if pcr > 1.25:
                pos_read = "Bullish bias — Strong Put writing support"
            elif pcr < 0.75:
                pos_read = "Bearish bias — Aggressive Call writing overhead"
            else:
                pos_read = "Neutral / mixed positioning"

            # ---------------------------------------------------------------
            # HEADER & TOP SUMMARY METRICS
            # ---------------------------------------------------------------
            curr_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"## ⚡ **{selected_index} — Spot {spot_price:.2f}** · Updated {curr_time}")
            st.caption(f"Positioning read: **{pos_read}** — pairing options positioning with technical trend.")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("PCR (OI)", f"{pcr}")
            col2.metric("Max Pain", f"{max_pain}")
            col3.metric("Expected Move (1σ)", f"±{expected_move}")
            col4.metric("ATM Straddle", f"{atm_straddle}")
            col5.metric("IV Skew (PE-CE)", f"{iv_skew:.2f}%" if iv_skew != 0 else "—")

            st.markdown("---")

            # ---------------------------------------------------------------
            # TOP OI WALLS & CHARTS
            # ---------------------------------------------------------------
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🛡️ Top Call & Put OI Walls")
                top_ce = ce_df.nlargest(3, 'oi')[['strike_price', 'oi', 'ltp']] if 'oi' in ce_df.columns else pd.DataFrame()
                top_pe = pe_df.nlargest(3, 'oi')[['strike_price', 'oi', 'ltp']] if 'oi' in pe_df.columns else pd.DataFrame()
                
                walls_df = pd.DataFrame({
                    "Resistance (Call Wall)": top_ce['strike_price'].values if not top_ce.empty else [],
                    "Call OI": top_ce['oi'].values if not top_ce.empty else [],
                    "Support (Put Wall)": top_pe['strike_price'].values if not top_pe.empty else [],
                    "Put OI": top_pe['oi'].values if not top_pe.empty else []
                })
                st.dataframe(walls_df, use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("📊 Intraday OI Distribution")
                pivot_oi = df.pivot(index='strike_price', columns='option_type', values='oi').fillna(0)
                # Display around ATM range (± 10 strikes)
                atm_range_pivot = pivot_oi.loc[(pivot_oi.index >= atm_strike - (10 * strike_step)) & (pivot_oi.index <= atm_strike + (10 * strike_step))]
                st.bar_chart(atm_range_pivot, height=260)

            st.markdown("---")

            # ---------------------------------------------------------------
            # OPTION CHAIN TABLE AROUND ATM
            # ---------------------------------------------------------------
            st.subheader("🎯 Option Chain around ATM")

            # Filter ±7 strikes around ATM
            filtered_strikes = [s for s in strikes if abs(s - atm_strike) <= (7 * strike_step)]
            filtered_strikes.sort()

            chain_rows = []
            for s in filtered_strikes:
                c_row = ce_df[ce_df['strike_price'] == s]
                p_row = pe_df[pe_df['strike_price'] == s]

                c_oi = int(c_row['oi'].values[0]) if not c_row.empty and 'oi' in c_row.columns else 0
                c_oichg = int(c_row['oich'].values[0]) if not c_row.empty and 'oich' in c_row.columns else 0
                c_iv = float(c_row['iv'].values[0]) if not c_row.empty and 'iv' in c_row.columns and c_row['iv'].values[0] is not None else 0.0
                c_ltp = float(c_row['ltp'].values[0]) if not c_row.empty and 'ltp' in c_row.columns else 0.0

                p_ltp = float(p_row['ltp'].values[0]) if not p_row.empty and 'ltp' in p_row.columns else 0.0
                p_iv = float(p_row['iv'].values[0]) if not p_row.empty and 'iv' in p_row.columns and p_row['iv'].values[0] is not None else 0.0
                p_oichg = int(p_row['oich'].values[0]) if not p_row.empty and 'oich' in p_row.columns else 0
                p_oi = int(p_row['oi'].values[0]) if not p_row.empty and 'oi' in p_row.columns else 0

                chain_rows.append({
                    "Call OI": f"{c_oi:,}",
                    "Call OI Chg": f"{c_oichg:+,}",
                    "Call IV": f"{c_iv:.2f}",
                    "Call Price": f"{c_ltp:.2f}",
                    "Strike Price": f"👉 {s}" if s == atm_strike else f"{s}",
                    "Put Price": f"{p_ltp:.2f}",
                    "Put IV": f"{p_iv:.2f}",
                    "Put OI Chg": f"{p_oichg:+,}",
                    "Put OI": f"{p_oi:,}"
                })

            chain_table_df = pd.DataFrame(chain_rows)
            st.dataframe(chain_table_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Please enter your **FYERS Access Token** in the sidebar to stream live market data.")
