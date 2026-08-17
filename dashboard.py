import streamlit as st
import requests
import pandas as pd

# Page Config
st.set_page_config(page_title="Nifty Signal Dashboard", layout="wide")

st.title("⚡ NIFTY Options Signal Dashboard")

# Sidebar Configuration
st.sidebar.header("🔑 FYERS API Login")

app_id = st.sidebar.text_input("FYERS App ID", value="IONVEW8SCZ-100", help="Must include -100 suffix")
access_token = st.sidebar.text_input("FYERS Access Token", type="password", help="Paste your long access token here")

selected_index = st.sidebar.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
st.sidebar.slider("Auto-refresh (seconds)", min_value=10, max_value=300, value=60)

# Symbol Mapping
SYMBOL_MAP = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX"
}

def fetch_option_chain(app_id, token, symbol):
    if not token or not app_id:
        return None, "Please provide both App ID and Access Token."

    headers = {
        "Authorization": f"{app_id}:{token}",
        "Content-Type": "application/json"
    }
    
    url = f"https://api-t1.fyers.in/data/options-chain-v3?symbol={symbol}&strikecount=15"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: {response.text}"
        
        data = response.json()
        if data.get("s") != "ok":
            return None, f"FYERS API Error ({data.get('code')}): {data.get('message', 'Unknown Error')}"
        
        return data.get("data", {}), None

    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# Dashboard Logic
if access_token:
    with st.spinner(f"Fetching live option chain for {selected_index}..."):
        fyers_symbol = SYMBOL_MAP[selected_index]
        chain_data, error_msg = fetch_option_chain(app_id, access_token, fyers_symbol)

    if error_msg:
        st.error(f"❌ {error_msg}")
    else:
        spot_price = chain_data.get("underVal", 0.0)
        options = chain_data.get("optionsChain", [])

        if not options:
            st.warning("⚠️ No option chain data returned.")
        else:
            df = pd.DataFrame(options)
            ce_df = df[df['option_type'] == 'CE']
            pe_df = df[df['option_type'] == 'PE']

            total_call_oi = ce_df['oi'].sum() if 'oi' in ce_df.columns else 0
            total_put_oi = pe_df['oi'].sum() if 'oi' in pe_df.columns else 0
            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0

            strikes = df['strike_price'].unique()
            pain_dict = {}
            for strike in strikes:
                call_loss = ce_df.apply(lambda row: max(0, strike - row['strike_price']) * row['oi'], axis=1).sum() if not ce_df.empty else 0
                put_loss = pe_df.apply(lambda row: max(0, row['strike_price'] - strike) * row['oi'], axis=1).sum() if not pe_df.empty else 0
                pain_dict[strike] = call_loss + put_loss
            
            max_pain_strike = min(pain_dict, key=pain_dict.get) if pain_dict else 0

            # Metrics Section
            st.markdown(f"### **{selected_index} — Spot {spot_price:.2f}**")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("PCR (OI)", f"{pcr}")
            col2.metric("Max Pain", f"{max_pain_strike}")
            col3.metric("Total Call OI", f"{total_call_oi:,}")
            col4.metric("Total Put OI", f"{total_put_oi:,}")
            col5.metric("Sentiment", "Bullish" if pcr > 1.0 else ("Bearish" if pcr < 0.8 else "Neutral"))

            st.markdown("---")
            st.subheader("📊 Open Interest (OI) Distribution by Strike")
            pivot_df = df.pivot(index='strike_price', columns='option_type', values='oi').fillna(0)
            st.bar_chart(pivot_df)
else:
    st.info("👈 Enter your **FYERS Access Token** in the sidebar to stream live data.")
