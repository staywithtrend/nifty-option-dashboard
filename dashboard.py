import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime, date

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty Option Dashboard",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------------
# BLACK-SCHOLES IV SOLVER
# -------------------------------------------------------------------
def cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * cdf(-d2) - S * cdf(-d1)

def calculate_iv(price, S, K, T, option_type='CE', r=0.07):
    """Calculates Implied Volatility (%) using Newton-Raphson & Bisection fallback"""
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0

    intrinsic = max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
    if price <= intrinsic:
        return 0.0

    # Newton-Raphson Method
    sigma = 0.20
    for _ in range(30):
        p = bs_call_price(S, K, T, r, sigma) if option_type == 'CE' else bs_put_price(S, K, T, r, sigma)
        diff = p - price
        if abs(diff) < 1e-4:
            return sigma * 100.0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * pdf(d1)
        if vega < 1e-6:
            break
        sigma = sigma - diff / vega
        if sigma <= 0:
            break

    # Bisection Fallback
    low, high = 0.001, 5.0
    for _ in range(35):
        mid = (low + high) / 2.0
        p = bs_call_price(S, K, T, r, mid) if option_type == 'CE' else bs_put_price(S, K, T, r, mid)
        if abs(p - price) < 1e-4:
            return mid * 100.0
        if p < price:
            low = mid
        else:
            high = mid
            
    return ((low + high) / 2.0) * 100.0

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.markdown("## 🔑 FYERS API Login")

app_id = st.sidebar.text_input("FYERS App ID", value="IONVEW8SCZ-100")
access_token = st.sidebar.text_input("FYERS Access Token", type="password")

st.sidebar.markdown("---")
selected_index = st.sidebar.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
auto_refresh = st.sidebar.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=60)
oi_lookback = st.sidebar.slider("OI-change comparison window (minutes)", min_value=5, max_value=60, value=15)

# Index Symbol and Strike Step Config
SYMBOL_CONFIG = {
    "NIFTY": {"fyers_symbol": "NSE:NIFTY50-INDEX", "step": 50},
    "BANKNIFTY": {"fyers_symbol": "NSE:NIFTYBANK-INDEX", "step": 100},
    "FINNIFTY": {"fyers_symbol": "NSE:FINNIFTY-INDEX", "step": 50}
}

# -------------------------------------------------------------------
# DATA FETCHING
# -------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_fyers_data(app_id, token, symbol):
    if not token or not app_id:
        return None, 0.0, "App ID and Access Token are required."

    headers = {
        "Authorization": f"{app_id}:{token}",
        "Content-Type": "application/json"
    }
    
    chain_url = f"https://api-t1.fyers.in/data/options-chain-v3?symbol={symbol}&strikecount=20"
    quotes_url = f"https://api-t1.fyers.in/data/quotes?symbols={symbol}"
    
    spot_price = 0.0
    
    try:
        # Get Spot Price
        q_res = requests.get(quotes_url, headers=headers, timeout=8)
        if q_res.status_code == 200:
            q_data = q_res.json()
            if q_data.get("s") == "ok" and "d" in q_data and len(q_data["d"]) > 0:
                spot_price = float(q_data["d"][0].get("v", {}).get("lp", 0.0))

        # Get Option Chain
        c_res = requests.get(chain_url, headers=headers, timeout=10)
        if c_res.status_code != 200:
            return None, spot_price, f"HTTP {c_res.status_code}: {c_res.text}"

        c_data = c_res.json()
        if c_data.get("s") != "ok":
            return None, spot_price, f"FYERS Error: {c_data.get('message', 'Failed to fetch chain')}"

        payload = c_data.get("data", {})
        if spot_price == 0.0:
            spot_price = float(payload.get("underVal", 0.0))

        return payload, spot_price, None

    except Exception as e:
        return None, 0.0, f"Connection Error: {str(e)}"

# -------------------------------------------------------------------
# MAIN DASHBOARD UI
# -------------------------------------------------------------------
if access_token:
    config = SYMBOL_CONFIG[selected_index]
    chain_payload, spot_price, error_msg = fetch_fyers_data(app_id, access_token, config["fyers_symbol"])

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif chain_payload and "optionsChain" in chain_payload:
        raw_options = chain_payload["optionsChain"]
        df = pd.DataFrame(raw_options)

        if not df.empty:
            ce_df = df[df['option_type'] == 'CE'].copy()
            pe_df = df[df['option_type'] == 'PE'].copy()

            # Calculate ATM Strike
            strike_step = config["step"]
            atm_strike = round(spot_price / strike_step) * strike_step if spot_price > 0 else df['strike_price'].median()

            # Totals & PCR
            total_call_oi = int(ce_df['oi'].sum()) if 'oi' in ce_df.columns else 0
            total_put_oi = int(pe_df['oi'].sum()) if 'oi' in pe_df.columns else 0
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

            # Max Pain
            strikes = sorted(df['strike_price'].unique())
            pain_dict = {}
            for s in strikes:
                c_loss = ce_df.apply(lambda r: max(0, s - r['strike_price']) * r['oi'], axis=1).sum() if not ce_df.empty else 0
                p_loss = pe_df.apply(lambda r: max(0, r['strike_price'] - s) * r['oi'], axis=1).sum() if not pe_df.empty else 0
                pain_dict[s] = c_loss + p_loss
            
            max_pain = min(pain_dict, key=pain_dict.get) if pain_dict else atm_strike

            # Sentiment Readout
            if pcr > 1.25:
                sentiment = "Bullish"
            elif pcr < 0.75:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"

            # ---------------------------------------------------------------
            # TOP HEADER & METRICS SUMMARY
            # ---------------------------------------------------------------
            st.title(f"⚡ {selected_index} Options Signal Dashboard")
            st.markdown(f"### **{selected_index} — Spot {spot_price:.2f}**")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("PCR (OI)", f"{pcr}")
            m2.metric("Max Pain", f"{max_pain}")
            m3.metric("Total Call OI", f"{total_call_oi:,}")
            m4.metric("Total Put OI", f"{total_put_oi:,}")
            m5.metric("Sentiment", f"{sentiment}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # OPEN INTEREST DISTRIBUTION CHART
            # ---------------------------------------------------------------
            st.subheader("📊 Open Interest (OI) Distribution by Strike")
            pivot_oi = df.pivot(index='strike_price', columns='option_type', values='oi').fillna(0)
            
            # Focus on strikes around ATM (± 10 strikes)
            atm_range_pivot = pivot_oi.loc[
                (pivot_oi.index >= atm_strike - (10 * strike_step)) & 
                (pivot_oi.index <= atm_strike + (10 * strike_step))
            ]
            st.bar_chart(atm_range_pivot, height=320)

            st.markdown("---")

            # ---------------------------------------------------------------
            # EXACT ORIGINAL OPTION CHAIN TABLE
            # ---------------------------------------------------------------
            # Determine days to expiry for IV calculation
            days_to_expiry = 3.0
            if "expiryData" in chain_payload and len(chain_payload["expiryData"]) > 0:
                exp_date_str = chain_payload["expiryData"][0].get("date", "")
                try:
                    exp_dt = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                    days_to_expiry = max((exp_dt - date.today()).days, 0.5)
                except Exception:
                    pass
            
            T = days_to_expiry / 365.0
            vix_fallback = float(chain_payload.get("indiavixData", {}).get("ltp", 14.0)) if isinstance(chain_payload.get("indiavixData"), dict) else 14.0

            # Filter strikes around ATM (± 8 strikes)
            filtered_strikes = [s for s in strikes if abs(s - atm_strike) <= (8 * strike_step)]
            filtered_strikes.sort()

            table_rows = []
            for s in filtered_strikes:
                c_row = ce_df[ce_df['strike_price'] == s]
                p_row = pe_df[pe_df['strike_price'] == s]

                c_oi = int(c_row['oi'].values[0]) if not c_row.empty and 'oi' in c_row.columns else 0
                c_oichg = int(c_row['oich'].values[0]) if not c_row.empty and 'oich' in c_row.columns else 0
                c_ltp = float(c_row['ltp'].values[0]) if not c_row.empty and 'ltp' in c_row.columns else 0.0

                p_ltp = float(p_row['ltp'].values[0]) if not p_row.empty and 'ltp' in p_row.columns else 0.0
                p_oichg = int(p_row['oich'].values[0]) if not p_row.empty and 'oich' in p_row.columns else 0
                p_oi = int(p_row['oi'].values[0]) if not p_row.empty and 'oi' in p_row.columns else 0

                # Dynamic BS IV Calculation
                calc_c_iv = calculate_iv(c_ltp, spot_price, s, T, 'CE')
                calc_p_iv = calculate_iv(p_ltp, spot_price, s, T, 'PE')

                c_iv_final = calc_c_iv if calc_c_iv > 0 else vix_fallback
                p_iv_final = calc_p_iv if calc_p_iv > 0 else vix_fallback

                table_rows.append({
                    "Call OI": f"{c_oi:,}",
                    "Call OI Chg": f"{c_oichg:+,}",
                    "Call IV": f"{c_iv_final:.2f}",
                    "Call Price": f"{c_ltp:.2f}",
                    "Strike Price": f"👉 {s}" if s == atm_strike else f"{s}",
                    "Put Price": f"{p_ltp:.2f}",
                    "Put IV": f"{p_iv_final:.2f}",
                    "Put OI Chg": f"{p_oichg:+,}",
                    "Put OI": f"{p_oi:,}"
                })

            chain_table_df = pd.DataFrame(table_rows)
            st.dataframe(chain_table_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Enter your **FYERS Access Token** in the sidebar to load the dashboard.")
