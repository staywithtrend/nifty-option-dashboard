import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime, date

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty Options Analytics",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------------
# CUSTOM CSS FOR COMPACT FONTS & NO TRUNCATION
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Sans-Serif Clean Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Reduced Heading Sizes */
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important; margin-top: 1rem !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
    h4 { font-size: 0.95rem !important; font-weight: 600 !important; }

    /* Compact Metric Boxes */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 8px 12px !important;
        border-radius: 8px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #495057 !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }

    /* Small & Compact Dataframe Text */
    .stDataFrame {
        font-size: 0.82rem !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem !important;
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# FALLBACK BLACK-SCHOLES IV SOLVER
# -------------------------------------------------------------------
def cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        return S * cdf(d1) - K * math.exp(-r * T) * cdf(d2)
    else:
        return K * math.exp(-r * T) * cdf(-d2) - S * cdf(-d1)

def compute_fallback_iv(price, S, K, T, option_type, r=0.07):
    if price <= 0 or S <= 0 or K <= 0:
        return 0.0
    T = max(T, 1.0 / 365.0)
    intrinsic = max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
    if price <= intrinsic:
        return 0.0

    low, high = 0.01, 3.0
    for _ in range(25):
        mid = (low + high) / 2.0
        p = bs_price(S, K, T, r, mid, option_type)
        if p < price:
            low = mid
        else:
            high = mid
    return round(((low + high) / 2.0) * 100.0, 2)

def extract_iv_from_row(row_series, spot_price, strike_price, T, option_type):
    if row_series.empty:
        return 0.0
    row_dict = row_series.iloc[0].to_dict()

    greeks_data = row_dict.get('greeks')
    if isinstance(greeks_data, dict):
        iv_val = greeks_data.get('iv', 0.0)
        if iv_val and float(iv_val) > 0:
            return float(iv_val)

    if 'iv' in row_dict and pd.notnull(row_dict['iv']):
        iv_val = float(row_dict['iv'])
        if iv_val > 0:
            return iv_val

    ltp = float(row_dict.get('ltp', 0.0))
    return compute_fallback_iv(ltp, spot_price, strike_price, T, option_type)

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.markdown("### 🔑 FYERS Credentials")
app_id = st.sidebar.text_input("FYERS App ID", value="IONVEW8SCZ-100")
access_token = st.sidebar.text_input("FYERS Access Token", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Symbol Selection")
selected_index = st.sidebar.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)

SYMBOL_CONFIG = {
    "NIFTY": {"fyers_symbol": "NSE:NIFTY50-INDEX", "step": 50, "lot_size": 65},
    "BANKNIFTY": {"fyers_symbol": "NSE:NIFTYBANK-INDEX", "step": 100, "lot_size": 30},
    "FINNIFTY": {"fyers_symbol": "NSE:FINNIFTY-INDEX", "step": 50, "lot_size": 60}
}

# -------------------------------------------------------------------
# API FETCH FUNCTION
# -------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_fyers_data(app_id, token, symbol):
    if not token or not app_id:
        return None, 0.0, "App ID and Access Token are required."

    headers = {"Authorization": f"{app_id}:{token}", "Content-Type": "application/json"}
    chain_url = f"https://api-t1.fyers.in/data/options-chain-v3?symbol={symbol}&strikecount=30&greeks=1"
    quotes_url = f"https://api-t1.fyers.in/data/quotes?symbols={symbol}"
    
    spot_price = 0.0
    try:
        q_res = requests.get(quotes_url, headers=headers, timeout=8)
        if q_res.status_code == 200:
            q_data = q_res.json()
            if q_data.get("s") == "ok" and "d" in q_data and len(q_data["d"]) > 0:
                spot_price = float(q_data["d"][0].get("v", {}).get("lp", 0.0))

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
# MAIN APPLICATION LOGIC
# -------------------------------------------------------------------
if access_token:
    config = SYMBOL_CONFIG[selected_index]
    lot_size = config["lot_size"]
    chain_payload, spot_price, error_msg = fetch_fyers_data(app_id, access_token, config["fyers_symbol"])

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif chain_payload and "optionsChain" in chain_payload:
        
        expiry_list = []
        if "expiryData" in chain_payload:
            for exp in chain_payload["expiryData"]:
                expiry_list.append(exp.get("date", ""))
        
        selected_expiry = st.sidebar.selectbox("Select Expiry Date", expiry_list, index=0) if expiry_list else "Current Expiry"

        # Calculate Days to Expiry (DTE)
        days_to_exp = 3.0
        if selected_expiry and selected_expiry != "Current Expiry":
            try:
                exp_dt = datetime.strptime(selected_expiry, "%d-%m-%Y").date()
                days_to_exp = max((exp_dt - date.today()).days, 0.5)
            except Exception:
                pass
        T = days_to_exp / 365.0

        raw_options = chain_payload["optionsChain"]
        df = pd.DataFrame(raw_options)

        if not df.empty:
            ce_df = df[df['option_type'] == 'CE'].copy()
            pe_df = df[df['option_type'] == 'PE'].copy()

            ce_df['oi_contracts'] = (ce_df['oi'] / lot_size).round().astype(int)
            ce_df['oich_contracts'] = (ce_df['oich'] / lot_size).round().astype(int)
            pe_df['oi_contracts'] = (pe_df['oi'] / lot_size).round().astype(int)
            pe_df['oich_contracts'] = (pe_df['oich'] / lot_size).round().astype(int)

            strike_step = config["step"]
            atm_strike = round(spot_price / strike_step) * strike_step if spot_price > 0 else df['strike_price'].median()

            # Walls
            call_wall_row = ce_df.loc[ce_df['oi_contracts'].idxmax()] if not ce_df.empty else None
            put_wall_row = pe_df.loc[pe_df['oi_contracts'].idxmax()] if not pe_df.empty else None

            call_wall_strike = int(call_wall_row['strike_price']) if call_wall_row is not None else 0
            put_wall_strike = int(put_wall_row['strike_price']) if put_wall_row is not None else 0

            # Totals & Differences
            total_call_oi = int(ce_df['oi_contracts'].sum()) if not ce_df.empty else 0
            total_put_oi = int(pe_df['oi_contracts'].sum()) if not pe_df.empty else 0
            net_oi_diff = total_put_oi - total_call_oi

            total_call_oichg = int(ce_df['oich_contracts'].sum()) if not ce_df.empty else 0
            total_put_oichg = int(pe_df['oich_contracts'].sum()) if not pe_df.empty else 0
            net_oichg_diff = total_put_oichg - total_call_oichg

            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

            # --- 1 SD EXPECTED MOVE CALCULATION ---
            atm_ce_row = ce_df[ce_df['strike_price'] == atm_strike]
            atm_pe_row = pe_df[pe_df['strike_price'] == atm_strike]
            
            atm_ce_iv = extract_iv_from_row(atm_ce_row, spot_price, atm_strike, T, 'CE')
            atm_pe_iv = extract_iv_from_row(atm_pe_row, spot_price, atm_strike, T, 'PE')
            
            avg_atm_iv = (atm_ce_iv + atm_pe_iv) / 2.0 if (atm_ce_iv + atm_pe_iv) > 0 else 15.0
            expected_move_pts = spot_price * (avg_atm_iv / 100.0) * math.sqrt(T)
            
            sd_upper = spot_price + expected_move_pts
            sd_lower = spot_price - expected_move_pts

            # ---------------------------------------------------------------
            # HEADER & SUMMARY METRICS
            # ---------------------------------------------------------------
            st.markdown(f"# ⚡ {selected_index} Options Analytics")
            st.markdown(f"**Spot Price:** `{spot_price:.2f}` | **ATM Strike:** `{int(atm_strike)}` | **Expiry:** `{selected_expiry}` (`{days_to_exp}` Days) | **ATM IV:** `{avg_atm_iv:.2f}%`")

            st.markdown("### 📊 Market Summary & Differences")

            m1, m2, m3, m4, m5 = st.columns(5)
            
            with m1:
                st.metric(label="🧱 Call / Put Walls", value=f"{call_wall_strike} / {put_wall_strike}")
                st.caption(f"Call Resistance / Put Support")

            with m2:
                st.metric(label="📉 Net OI Diff (Put - Call)", value=f"{net_oi_diff:+,}")
                st.caption(f"Total Put OI - Total Call OI")

            with m3:
                st.metric(label="⚡ Net OI Chg Diff", value=f"{net_oichg_diff:+,}")
                st.caption(f"Put Chg - Call Chg")

            with m4:
                st.metric(label="📊 PCR Ratio", value=f"{pcr}")
                sentiment = "🐂 Bullish" if pcr > 1.25 else ("🐻 Bearish" if pcr < 0.75 else "⚖️ Neutral")
                st.caption(f"Sentiment: **{sentiment}**")

            with m5:
                st.metric(label="🎯 1 SD Expected Move", value=f"±{expected_move_pts:.1f} pts")
                st.caption(f"Range: **{sd_lower:.0f} - {sd_upper:.0f}**")

            st.markdown("---")

            # ---------------------------------------------------------------
            # TOP 3 HIGHEST OI & OI CHANGE TABLES (2-ROW SPACIOUS GRID)
            # ---------------------------------------------------------------
            st.markdown("### 🔥 Top 3 Highest OI & Highest Change in OI")

            top_call_oi = ce_df.nlargest(3, 'oi_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not ce_df.empty else pd.DataFrame()
            top_call_oichg = ce_df.nlargest(3, 'oich_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not ce_df.empty else pd.DataFrame()

            top_put_oi = pe_df.nlargest(3, 'oi_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not pe_df.empty else pd.DataFrame()
            top_put_oichg = pe_df.nlargest(3, 'oich_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not pe_df.empty else pd.DataFrame()

            def format_top_table(d):
                if d.empty:
                    return d
                res = d.copy()
                tot_oi = res['oi_contracts'].sum()
                tot_oichg = res['oich_contracts'].sum()

                res['strike_price'] = res['strike_price'].astype(int).astype(str)
                res['oi_contracts'] = res['oi_contracts'].apply(lambda x: f"{int(x):,}")
                res['oich_contracts'] = res['oich_contracts'].apply(lambda x: f"{int(x):+,}")

                tot_row = pd.DataFrame([{
                    'strike_price': 'Total',
                    'oi_contracts': f"{int(tot_oi):,}",
                    'oich_contracts': f"{int(tot_oichg):+,}"
                }])
                res = pd.concat([res, tot_row], ignore_index=True)
                res.columns = ['Strike', 'OI (Contracts)', 'OI Chg (Contracts)']
                return res

            c_top1, c_top2 = st.columns(2)
            with c_top1:
                st.markdown("#### 🔹 Top 3 Call Highest OI")
                st.dataframe(format_top_table(top_call_oi), hide_index=True, use_container_width=True)
            with c_top2:
                st.markdown("#### 🔹 Top 3 Call Highest OI Chg")
                st.dataframe(format_top_table(top_call_oichg), hide_index=True, use_container_width=True)

            p_top1, p_top2 = st.columns(2)
            with p_top1:
                st.markdown("#### 🔸 Top 3 Put Highest OI")
                st.dataframe(format_top_table(top_put_oi), hide_index=True, use_container_width=True)
            with p_top2:
                st.markdown("#### 🔸 Top 3 Put Highest OI Chg")
                st.dataframe(format_top_table(top_put_oichg), hide_index=True, use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------------
            # STRIKE-BY-STRIKE DIFFERENCES TABLE
            # ---------------------------------------------------------------
            st.markdown("### 📋 Option Differences Table (Put minus Call)")

            strikes = sorted(df['strike_price'].unique())
            filtered_strikes = [s for s in strikes if abs(s - atm_strike) <= (10 * strike_step)]

            diff_rows = []
            for s in filtered_strikes:
                c_row = ce_df[ce_df['strike_price'] == s]
                p_row = pe_df[pe_df['strike_price'] == s]

                c_oi = int(c_row['oi_contracts'].values[0]) if not c_row.empty else 0
                c_oichg = int(c_row['oich_contracts'].values[0]) if not c_row.empty else 0
                c_iv = extract_iv_from_row(c_row, spot_price, s, T, 'CE')

                p_oi = int(p_row['oi_contracts'].values[0]) if not p_row.empty else 0
                p_oichg = int(p_row['oich_contracts'].values[0]) if not p_row.empty else 0
                p_iv = extract_iv_from_row(p_row, spot_price, s, T, 'PE')

                # Requested Calculations
                oi_diff = p_oi - c_oi
                oichg_diff = p_oichg - c_oichg
                iv_skew = p_iv - c_iv

                diff_rows.append({
                    "Strike": int(s),
                    "Put OI - Call OI": oi_diff,
                    "Put OI Chg - Call OI Chg": oichg_diff,
                    "PE IV - CE IV": round(iv_skew, 2),
                    "Call OI": c_oi,
                    "Put OI": p_oi,
                    "Call IV": c_iv,
                    "PE IV": p_iv
                })

            diff_df = pd.DataFrame(diff_rows)

            # Styling Difference Columns
            def style_diffs(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #2e7d32; font-weight: bold;' # Green for positive put excess
                    elif val < 0:
                        return 'color: #c62828; font-weight: bold;' # Red for negative call excess
                return ''

            styled_diff_df = diff_df.style.applymap(
                style_diffs, subset=['Put OI - Call OI', 'Put OI Chg - Call OI Chg', 'PE IV - CE IV']
            ).format({
                "Strike": "{:d}",
                "Put OI - Call OI": "{:+,}",
                "Put OI Chg - Call OI Chg": "{:+,}",
                "PE IV - CE IV": "{:+.2f}",
                "Call OI": "{:,}",
                "Put OI": "{:,}",
                "Call IV": "{:.2f}",
                "PE IV": "{:.2f}"
            })

            st.dataframe(styled_diff_df, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------------
            # STACKED PLOTLY OI DIFFERENCE CHART
            # ---------------------------------------------------------------
            st.markdown("### 📊 OI Difference Chart (Put OI - Call OI)")
            
            fig_diff = go.Figure()
            colors = ['#2e7d32' if x >= 0 else '#c62828' for x in diff_df['Put OI - Call OI']]
            
            fig_diff.add_trace(go.Bar(
                x=diff_df['Strike'],
                y=diff_df['Put OI - Call OI'],
                marker_color=colors,
                name='Net OI Difference'
            ))
            fig_diff.update_layout(
                xaxis_title='Strike Price',
                yaxis_title='Put OI - Call OI (Contracts)',
                margin=dict(l=20, r=20, t=20, b=20),
                height=350
            )
            st.plotly_chart(fig_diff, use_container_width=True)

else:
    st.info("👈 Please enter your **FYERS Access Token** in the sidebar to view data.")
