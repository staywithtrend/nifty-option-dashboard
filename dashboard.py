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
    page_title="Nifty Options Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

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
st.sidebar.markdown("## 🔑 FYERS API Credentials")

app_id = st.sidebar.text_input("FYERS App ID", value="IONVEW8SCZ-100")
access_token = st.sidebar.text_input("FYERS Access Token", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Symbol & Expiry Selection")
selected_index = st.sidebar.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)

# Lot Size configurations
SYMBOL_CONFIG = {
    "NIFTY": {"fyers_symbol": "NSE:NIFTY50-INDEX", "step": 50, "lot_size": 65},
    "BANKNIFTY": {"fyers_symbol": "NSE:NIFTYBANK-INDEX", "step": 100, "lot_size": 30},
    "FINNIFTY": {"fyers_symbol": "NSE:FINNIFTY-INDEX", "step": 50, "lot_size": 60}
}

# -------------------------------------------------------------------
# API DATA FETCHERS
# -------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_fyers_data(app_id, token, symbol):
    if not token or not app_id:
        return None, 0.0, "App ID and Access Token are required."

    headers = {
        "Authorization": f"{app_id}:{token}",
        "Content-Type": "application/json"
    }
    
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
        
        # --- DYNAMIC EXPIRY SELECTION DROPDOWN ---
        expiry_list = []
        if "expiryData" in chain_payload and len(chain_payload["expiryData"]) > 0:
            for exp in chain_payload["expiryData"]:
                expiry_list.append(exp.get("date", ""))
        
        if expiry_list:
            selected_expiry = st.sidebar.selectbox("Select Expiry Date", expiry_list, index=0)
        else:
            selected_expiry = "Current Expiry"

        auto_refresh = st.sidebar.slider("Auto-refresh (seconds)", min_value=15, max_value=300, value=60)

        raw_options = chain_payload["optionsChain"]
        df = pd.DataFrame(raw_options)

        if not df.empty:
            ce_df = df[df['option_type'] == 'CE'].copy()
            pe_df = df[df['option_type'] == 'PE'].copy()

            # Convert raw quantities to Contracts (Lots)
            ce_df['oi_contracts'] = (ce_df['oi'] / lot_size).round().astype(int)
            ce_df['oich_contracts'] = (ce_df['oich'] / lot_size).round().astype(int)
            pe_df['oi_contracts'] = (pe_df['oi'] / lot_size).round().astype(int)
            pe_df['oich_contracts'] = (pe_df['oich'] / lot_size).round().astype(int)

            strike_step = config["step"]
            atm_strike = round(spot_price / strike_step) * strike_step if spot_price > 0 else df['strike_price'].median()

            # ---------------------------------------------------------------
            # CALL WALL & PUT WALL CALCULATIONS (IN CONTRACTS)
            # ---------------------------------------------------------------
            call_wall_row = ce_df.loc[ce_df['oi_contracts'].idxmax()] if not ce_df.empty and 'oi_contracts' in ce_df.columns else None
            put_wall_row = pe_df.loc[pe_df['oi_contracts'].idxmax()] if not pe_df.empty and 'oi_contracts' in pe_df.columns else None

            call_wall_strike = int(call_wall_row['strike_price']) if call_wall_row is not None else 0
            call_wall_oi = int(call_wall_row['oi_contracts']) if call_wall_row is not None else 0
            call_wall_oichg = int(call_wall_row['oich_contracts']) if call_wall_row is not None else 0

            put_wall_strike = int(put_wall_row['strike_price']) if put_wall_row is not None else 0
            put_wall_oi = int(put_wall_row['oi_contracts']) if put_wall_row is not None else 0
            put_wall_oichg = int(put_wall_row['oich_contracts']) if put_wall_row is not None else 0

            # ---------------------------------------------------------------
            # HEADER & WALL METRICS
            # ---------------------------------------------------------------
            st.title(f"⚡ {selected_index} Options Signal Dashboard")
            st.markdown(f"### **Spot Price: {spot_price:.2f}** | **ATM Strike: {int(atm_strike)}** | **Expiry: {selected_expiry}** | **Lot Size: {lot_size}**")

            st.markdown("---")
            st.subheader("🧱 Key Market Walls & Level Summary (in Contracts)")

            w1, w2, w3, w4 = st.columns(4)
            
            with w1:
                st.metric(
                    label="🟢 CALL WALL (Resistance)",
                    value=f"Strike {call_wall_strike:,}",
                    delta=f"OI: {call_wall_oi:,} Contracts"
                )
                st.caption(f"OI Chg: **{call_wall_oichg:+,} Contracts**")

            with w2:
                st.metric(
                    label="🔴 PUT WALL (Support)",
                    value=f"Strike {put_wall_strike:,}",
                    delta=f"OI: {put_wall_oi:,} Contracts"
                )
                st.caption(f"OI Chg: **{put_wall_oichg:+,} Contracts**")

            total_call_oi = int(ce_df['oi_contracts'].sum()) if 'oi_contracts' in ce_df.columns else 0
            total_put_oi = int(pe_df['oi_contracts'].sum()) if 'oi_contracts' in pe_df.columns else 0
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

            with w3:
                st.metric(label="📊 Put-Call Ratio (PCR)", value=f"{pcr}")
                sentiment = "🐂 Bullish" if pcr > 1.25 else ("🐻 Bearish" if pcr < 0.75 else "⚖️ Neutral")
                st.caption(f"Market Sentiment: **{sentiment}**")

            with w4:
                st.metric(label="📈 Total Call / Put Contracts", value=f"{total_put_oi:,} / {total_call_oi:,}")
                st.caption(f"Net Diff: **{(total_put_oi - total_call_oi):+,} Contracts**")

            st.markdown("---")

            # ---------------------------------------------------------------
            # TOP 3 HIGHEST OI & HIGHEST CHANGE IN OI (WITH TOTAL ROW)
            # ---------------------------------------------------------------
            st.subheader("🔥 Top 3 Highest OI & Highest Change in OI (Contracts)")

            t1, t2, t3, t4 = st.columns(4)

            top_call_oi = ce_df.nlargest(3, 'oi_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not ce_df.empty else pd.DataFrame()
            top_call_oichg = ce_df.nlargest(3, 'oich_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not ce_df.empty else pd.DataFrame()

            top_put_oi = pe_df.nlargest(3, 'oi_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not pe_df.empty else pd.DataFrame()
            top_put_oichg = pe_df.nlargest(3, 'oich_contracts')[['strike_price', 'oi_contracts', 'oich_contracts']] if not pe_df.empty else pd.DataFrame()

            def format_top_table(d):
                if d.empty:
                    return d
                res = d.copy()

                # Calculate totals for top 3
                tot_oi = res['oi_contracts'].sum()
                tot_oichg = res['oich_contracts'].sum()

                # Format existing rows
                res['strike_price'] = res['strike_price'].astype(int).astype(str)
                res['oi_contracts'] = res['oi_contracts'].apply(lambda x: f"{int(x):,}")
                res['oich_contracts'] = res['oich_contracts'].apply(lambda x: f"{int(x):+,}")

                # Append Total Row
                tot_row = pd.DataFrame([{
                    'strike_price': 'Total',
                    'oi_contracts': f"{int(tot_oi):,}",
                    'oich_contracts': f"{int(tot_oichg):+,}"
                }])

                res = pd.concat([res, tot_row], ignore_index=True)
                res.columns = ['Strike', 'OI (Contracts)', 'OI Chg (Contracts)']
                return res

            with t1:
                st.markdown("#### 🔹 Top 3 Call Highest OI")
                st.dataframe(format_top_table(top_call_oi), hide_index=True, use_container_width=True)

            with t2:
                st.markdown("#### 🔹 Top 3 Call Highest OI Chg")
                st.dataframe(format_top_table(top_call_oichg), hide_index=True, use_container_width=True)

            with t3:
                st.markdown("#### 🔸 Top 3 Put Highest OI")
                st.dataframe(format_top_table(top_put_oi), hide_index=True, use_container_width=True)

            with t4:
                st.markdown("#### 🔸 Top 3 Put Highest OI Chg")
                st.dataframe(format_top_table(top_put_oichg), hide_index=True, use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------------
            # STACKED PLOTLY CHARTS (IN CONTRACTS)
            # ---------------------------------------------------------------
            strikes = sorted(df['strike_price'].unique())
            filtered_strikes = [s for s in strikes if abs(s - atm_strike) <= (10 * strike_step)]
            filtered_strikes.sort()

            chart_ce = ce_df[ce_df['strike_price'].isin(filtered_strikes)].sort_values('strike_price')
            chart_pe = pe_df[pe_df['strike_price'].isin(filtered_strikes)].sort_values('strike_price')

            # 1. Total Open Interest Chart
            st.subheader("📊 Total Open Interest in Contracts (Call vs Put)")
            fig_total = go.Figure()
            fig_total.add_trace(go.Bar(
                x=chart_ce['strike_price'],
                y=chart_ce['oi_contracts'],
                name='Call OI (Resistance)',
                marker_color='#1E88E5' # Blue
            ))
            fig_total.add_trace(go.Bar(
                x=chart_pe['strike_price'],
                y=chart_pe['oi_contracts'],
                name='Put OI (Support)',
                marker_color='#E53935' # Red
            ))
            fig_total.update_layout(
                barmode='group',
                xaxis_title='Strike Price',
                yaxis_title='Total Contracts',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                height=380
            )
            st.plotly_chart(fig_total, use_container_width=True)

            # 2. Change in Open Interest Chart
            st.subheader("⚡ Change in Open Interest in Contracts (Call vs Put)")
            fig_chg = go.Figure()
            fig_chg.add_trace(go.Bar(
                x=chart_ce['strike_price'],
                y=chart_ce['oich_contracts'],
                name='Call OI Change',
                marker_color='#1565C0' # Darker Blue
            ))
            fig_chg.add_trace(go.Bar(
                x=chart_pe['strike_price'],
                y=chart_pe['oich_contracts'],
                name='Put OI Change',
                marker_color='#C62828' # Darker Red
            ))
            fig_chg.update_layout(
                barmode='group',
                xaxis_title='Strike Price',
                yaxis_title='Change in Contracts',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                height=380
            )
            st.plotly_chart(fig_chg, use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------------
            # OPTION CHAIN TABLE WITH TOP 3 HIGHLIGHTING ONLY (IN CONTRACTS)
            # ---------------------------------------------------------------
            st.subheader("📋 Option Chain (Values in Contracts | Top 3 Highlighted)")

            days_to_exp = 3.0
            if selected_expiry and selected_expiry != "Current Expiry":
                try:
                    exp_dt = datetime.strptime(selected_expiry, "%d-%m-%Y").date()
                    days_to_exp = max((exp_dt - date.today()).days, 0.5)
                except Exception:
                    pass
            T = days_to_exp / 365.0

            table_rows = []
            for s in filtered_strikes:
                c_row = ce_df[ce_df['strike_price'] == s]
                p_row = pe_df[pe_df['strike_price'] == s]

                c_oi = int(c_row['oi_contracts'].values[0]) if not c_row.empty and 'oi_contracts' in c_row.columns else 0
                c_oichg = int(c_row['oich_contracts'].values[0]) if not c_row.empty and 'oich_contracts' in c_row.columns else 0
                c_ltp = float(c_row['ltp'].values[0]) if not c_row.empty and 'ltp' in c_row.columns else 0.0
                c_iv = extract_iv_from_row(c_row, spot_price, s, T, 'CE')

                p_ltp = float(p_row['ltp'].values[0]) if not p_row.empty and 'ltp' in p_row.columns else 0.0
                p_iv = extract_iv_from_row(p_row, spot_price, s, T, 'PE')
                p_oichg = int(p_row['oich_contracts'].values[0]) if not p_row.empty and 'oich_contracts' in p_row.columns else 0
                p_oi = int(p_row['oi_contracts'].values[0]) if not p_row.empty and 'oi_contracts' in p_row.columns else 0

                table_rows.append({
                    "Call OI (Contracts)": c_oi,
                    "Call OI Chg (Contracts)": c_oichg,
                    "Call IV": c_iv,
                    "Call Price": c_ltp,
                    "Strike Price": s,
                    "Put Price": p_ltp,
                    "Put IV": p_iv,
                    "Put OI Chg (Contracts)": p_oichg,
                    "Put OI (Contracts)": p_oi
                })

            chain_df = pd.DataFrame(table_rows)

            # Function to style ONLY top 3 values per column
            def style_top3_only(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)

                def apply_colors(series, col_name, color_map):
                    unique_vals = sorted([v for v in series.unique() if v > 0], reverse=True)
                    top3_vals = unique_vals[:3]
                    
                    for idx, val in series.items():
                        if val in top3_vals:
                            rank = top3_vals.index(val) + 1
                            styles.loc[idx, col_name] = color_map[rank]

                # Call colors
                call_oi_colors = {
                    1: 'background-color: #0D47A1; color: white; font-weight: bold;',
                    2: 'background-color: #1976D2; color: white; font-weight: bold;',
                    3: 'background-color: #64B5F6; color: black; font-weight: bold;'
                }
                call_oichg_colors = {
                    1: 'background-color: #1565C0; color: white; font-weight: bold;',
                    2: 'background-color: #2196F3; color: white; font-weight: bold;',
                    3: 'background-color: #90CAF9; color: black; font-weight: bold;'
                }

                # Put colors
                put_oi_colors = {
                    1: 'background-color: #B71C1C; color: white; font-weight: bold;',
                    2: 'background-color: #D32F2F; color: white; font-weight: bold;',
                    3: 'background-color: #EF5350; color: white; font-weight: bold;'
                }
                put_oichg_colors = {
                    1: 'background-color: #C62828; color: white; font-weight: bold;',
                    2: 'background-color: #F44336; color: white; font-weight: bold;',
                    3: 'background-color: #E57373; color: black; font-weight: bold;'
                }

                apply_colors(df['Call OI (Contracts)'], 'Call OI (Contracts)', call_oi_colors)
                apply_colors(df['Call OI Chg (Contracts)'], 'Call OI Chg (Contracts)', call_oichg_colors)
                apply_colors(df['Put OI (Contracts)'], 'Put OI (Contracts)', put_oi_colors)
                apply_colors(df['Put OI Chg (Contracts)'], 'Put OI Chg (Contracts)', put_oichg_colors)

                return styles

            styled_chain = chain_df.style.apply(style_top3_only, axis=None).format({
                "Call OI (Contracts)": "{:,}",
                "Call OI Chg (Contracts)": "{:+,}",
                "Call IV": "{:.2f}",
                "Call Price": "{:.2f}",
                "Strike Price": "{:d}",
                "Put Price": "{:.2f}",
                "Put IV": "{:.2f}",
                "Put OI Chg (Contracts)": "{:+,}",
                "Put OI (Contracts)": "{:,}"
            })

            # Render Table
            st.dataframe(styled_chain, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------------
            # FIXED SUMMARY BAR BELOW TABLE (TOTAL CONTRACTS)
            # ---------------------------------------------------------------
            st.markdown("---")
            st.subheader("📌 Option Chain Totals (Fixed Below Table - In Contracts)")

            tot_c_oi = chain_df['Call OI (Contracts)'].sum()
            tot_c_oichg = chain_df['Call OI Chg (Contracts)'].sum()
            tot_p_oi = chain_df['Put OI (Contracts)'].sum()
            tot_p_oichg = chain_df['Put OI Chg (Contracts)'].sum()

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Call OI", f"{tot_c_oi:,} Contracts")
            b2.metric("Total Call OI Change", f"{tot_c_oichg:+,} Contracts")
            b3.metric("Total Put OI Change", f"{tot_p_oichg:+,} Contracts")
            b4.metric("Total Put OI", f"{tot_p_oi:,} Contracts")

else:
    st.info("👈 Enter your **FYERS Access Token** in the sidebar to load the dashboard.")
