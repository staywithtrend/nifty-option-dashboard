"""
Nifty Signal Dashboard — live NSE option-chain dashboard.

Features:
- Index selector
- Expiry selector
- PCR / Max Pain / IV skew / expected move / ATM straddle
- Top CE/PE OI walls
- Option chain around ATM (includes Day OI Change, Window OI Delta, 2-decimal IVs)
- Intraday ATM CE OI vs PE OI graph
- Intraday NIFTY spot graph
- Intraday history collected from the first refresh of the day until close
"""

import time
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import altair as alt

from nse_fetcher import NseSession
from signals import (
    chain_to_dataframe,
    compute_pcr,
    compute_max_pain,
    compute_iv_skew,
    compute_expected_move,
    top_oi_strikes,
    oi_change_vs_previous,
    bias_verdict,
)

st.set_page_config(page_title="Nifty Signal Dashboard", layout="wide")

SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "nse" not in st.session_state:
    st.session_state.nse = NseSession()

if "history" not in st.session_state:
    st.session_state.history = {}  # (symbol, expiry) -> list of snapshots

if "last_day" not in st.session_state:
    st.session_state.last_day = date.today()


def reset_history_for_new_day():
    today = date.today()
    if st.session_state.last_day != today:
        st.session_state.history = {}
        st.session_state.last_day = today


reset_history_for_new_day()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Nifty Signal Dashboard")

symbol = st.sidebar.selectbox("Index", SYMBOLS, index=0)
refresh_secs = st.sidebar.slider(
    "Auto-refresh (seconds)", 30, 300, 60, step=15
)
oi_lookback_mins = st.sidebar.slider(
    "OI-change comparison window (minutes)", 5, 60, 15, step=5
)
strike_window = st.sidebar.slider(
    "Strikes shown around ATM", 5, 25, 10
)

st.sidebar.caption(
    "Data: NSE option chain. PCR/IV/max-pain are positioning reads, "
    "not trade calls."
)


# ---------------------------------------------------------------------
# Fetch raw chain for expiries discovery
# ---------------------------------------------------------------------
placeholder = st.empty()

try:
    raw = st.session_state.nse.get_option_chain(symbol)
except RuntimeError as e:
    st.error(
        f"Fetch failed: {e}. "
        "NSE may temporarily rate-limit automated requests; "
        "the next refresh will retry."
    )
    st.stop()

available_expiries = raw["records"].get("expiryDates", [])
if not available_expiries:
    st.error(f"No expiry dates returned by NSE for {symbol}.")
    st.stop()

default_expiry_index = 0

selected_expiry = st.sidebar.selectbox(
    "Expiry",
    available_expiries,
    index=default_expiry_index,
    key="selected_expiry",
)

history_key = (symbol, selected_expiry)

# ---------------------------------------------------------------------
# Build current snapshot for the SELECTED expiry.
# ---------------------------------------------------------------------
if selected_expiry == available_expiries[0]:
    selected_raw = raw
else:
    try:
        selected_raw = st.session_state.nse.get_option_chain(
            symbol,
            expiry=selected_expiry,
        )
    except RuntimeError as e:
        st.error(f"Could not load {selected_expiry}: {e}")
        st.stop()

df, spot, expiry = chain_to_dataframe(selected_raw, selected_expiry)

if df.empty or "strike" not in df.columns:
    st.error(
        f"No option-chain rows were returned for {symbol} "
        f"expiry {selected_expiry}."
    )
    st.stop()

now = datetime.now(ZoneInfo("Asia/Kolkata"))

# ---------------------------------------------------------------------
# Intraday history tracking
# ---------------------------------------------------------------------
hist = st.session_state.history.setdefault(history_key, [])

atm_idx = (df["strike"] - spot).abs().idxmin()
atm_row = df.loc[atm_idx]

snapshot = {
    "timestamp": now,
    "time": now.strftime("%H:%M"),
    "spot": float(spot),
    "atm_strike": float(atm_row["strike"]),
    "atm_ce_oi": float(atm_row["ce_oi"]),
    "atm_pe_oi": float(atm_row["pe_oi"]),
}

if not hist or hist[-1]["timestamp"].strftime("%H:%M:%S") != now.strftime("%H:%M:%S"):
    hist.append(snapshot)

st.session_state.history[history_key] = [
    x for x in hist
    if x["timestamp"].date() == now.date()
]

hist = st.session_state.history[history_key]

# ---------------------------------------------------------------------
# OI comparison snapshot history
# ---------------------------------------------------------------------
if "df_history" not in st.session_state:
    st.session_state.df_history = {}

df_hist_key = (symbol, selected_expiry)
df_snapshots = st.session_state.df_history.setdefault(df_hist_key, [])

if (
    not df_snapshots
    or df_snapshots[-1][0].strftime("%H:%M:%S") != now.strftime("%H:%M:%S")
):
    df_snapshots.append((now, df.copy()))

st.session_state.df_history[df_hist_key] = [
    (t, d) for t, d in df_snapshots
    if t.date() == now.date()
]

df_snapshots = st.session_state.df_history[df_hist_key]

df_prev = None
for t, d in df_snapshots:
    if (now - t).total_seconds() >= oi_lookback_mins * 60:
        df_prev = d

merged = oi_change_vs_previous(df, df_prev)

# ---------------------------------------------------------------------
# Current positioning calculations
# ---------------------------------------------------------------------
pcr = compute_pcr(df)
max_pain = compute_max_pain(df)
iv_skew = compute_iv_skew(df, spot)
exp_move = compute_expected_move(df, spot)
top_ce, top_pe = top_oi_strikes(df, n=3)
verdict, notes = bias_verdict(pcr, iv_skew, max_pain, spot)


# ---------------------------------------------------------------------
# Main Dashboard UI
# ---------------------------------------------------------------------
with placeholder.container():

    st.subheader(
        f"{symbol} — Spot {spot:,.2f}  ·  Expiry {expiry}  ·  "
        f"Updated {now.strftime('%H:%M:%S')}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PCR (OI)", pcr if pcr is not None else "—")
    c2.metric("Max Pain", f"{max_pain:,.0f}" if max_pain else "—")
    c3.metric(
        "Expected Move (1σ)",
        f"±{exp_move['expected_move_1sigma']}",
    )
    c4.metric("ATM Straddle", exp_move["straddle_price"])
    c5.metric(
        "IV Skew (PE-CE)",
        iv_skew["skew"] if iv_skew["skew"] is not None else "—",
    )

    st.markdown(f"**Positioning read: {verdict}**")
    for n in notes:
        st.caption(f"• {n}")

    st.caption(
        "This is a read of current options positioning, not a prediction "
        "or trade recommendation — pair it with your own technical view."
    )

    # -------------------------------------------------------------
    # Intraday charts
    # -------------------------------------------------------------
    st.markdown("## Intraday OI & Price")

    chart_data = pd.DataFrame(hist)

    if len(chart_data) >= 1:
        chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])
        chart_data = chart_data.set_index("timestamp")

        st.markdown(
            f"**ATM CE OI vs ATM PE OI — {selected_expiry}**"
        )
        st.caption(
            "ATM strike is recalculated at every snapshot. "
            "The table below shows when the ATM strike changes."
        )

        oi_chart = chart_data[
            ["atm_ce_oi", "atm_pe_oi"]
        ].rename(
            columns={
                "atm_ce_oi": "ATM Call OI",
                "atm_pe_oi": "ATM Put OI",
            }
        )

        st.line_chart(
            oi_chart,
            use_container_width=True,
            height=320,
        )

        latest = chart_data.iloc[-1]

        m1, m2, m3 = st.columns(3)
        m1.metric("Current ATM", f"{latest['atm_strike']:,.0f}")
        m2.metric("ATM Call OI", f"{latest['atm_ce_oi']:,.0f}")
        m3.metric("ATM Put OI", f"{latest['atm_pe_oi']:,.0f}")

    else:
        st.info(
            "Intraday history will build from the first snapshot. "
            "For a complete 9:15 AM–3:30 PM graph, keep the dashboard "
            "running throughout the session."
        )

    # -------------------------------------------------------------
    # OI walls
    # -------------------------------------------------------------
    st.markdown("## OI Walls (support/resistance by open interest)")

    call_walls = (
        df.nlargest(3, "ce_oi")
        [["strike", "ce_oi", "ce_oi_change"]]
        .copy()
    )
    call_walls.columns = ["Strike", "Call OI", "Call OI Change"]

    put_walls = (
        df.nlargest(3, "pe_oi")
        [["strike", "pe_oi", "pe_oi_change"]]
        .copy()
    )
    put_walls.columns = ["Strike", "Put OI", "Put OI Change"]

    call_total = pd.DataFrame([{
        "Strike": "TOTAL",
        "Call OI": call_walls["Call OI"].sum(),
        "Call OI Change": call_walls["Call OI Change"].sum(),
    }])

    put_total = pd.DataFrame([{
        "Strike": "TOTAL",
        "Put OI": put_walls["Put OI"].sum(),
        "Put OI Change": put_walls["Put OI Change"].sum(),
    }])

    call_display = pd.concat(
        [call_walls, call_total], ignore_index=True
    )
    put_display = pd.concat(
        [put_walls, put_total], ignore_index=True
    )

    full_call_oi = df["ce_oi"].sum()
    full_call_oi_change = df["ce_oi_change"].sum()
    full_put_oi = df["pe_oi"].sum()
    full_put_oi_change = df["pe_oi_change"].sum()

    wcol1, wcol2 = st.columns(2)

    with wcol1:
        st.write("Top Call OI (resistance-side)")
        st.dataframe(
            call_display.style.format({
                "Strike": lambda x: f"{float(x):,.0f}" if x != "TOTAL" else x,
                "Call OI": "{:,.0f}",
                "Call OI Change": "{:+,.0f}",
            }),
            hide_index=True,
            use_container_width=True,
            height=205,
        )
        st.caption(
            f"Full-chain Call OI: {full_call_oi:,.0f}  |  "
            f"Full-chain OI Change: {full_call_oi_change:+,.0f}"
        )

    with wcol2:
        st.write("Top Put OI (support-side)")
        st.dataframe(
            put_display.style.format({
                "Strike": lambda x: f"{float(x):,.0f}" if x != "TOTAL" else x,
                "Put OI": "{:,.0f}",
                "Put OI Change": "{:+,.0f}",
            }),
            hide_index=True,
            use_container_width=True,
            height=205,
        )
        st.caption(
            f"Full-chain Put OI: {full_put_oi:,.0f}  |  "
            f"Full-chain OI Change: {full_put_oi_change:+,.0f}"
        )

    # -------------------------------------------------------------
    # Chain around ATM
    # -------------------------------------------------------------
    st.markdown(
        f"## Chain around ATM (±{strike_window} strikes)"
    )

    atm_idx = (df["strike"] - spot).abs().idxmin()
    lo = max(0, atm_idx - strike_window)
    hi = min(len(df), atm_idx + strike_window + 1)

    display_df = (
        merged if merged is not None else df
    ).iloc[lo:hi]

    if merged is not None:
        cols_to_show = [
            "strike",
            "ce_oi",
            "ce_oi_change",
            "ce_oi_delta",
            "ce_iv",
            "ce_ltp",
            "pe_ltp",
            "pe_iv",
            "pe_oi_delta",
            "pe_oi_change",
            "pe_oi",
        ]
        st.caption(
            f"OI delta columns compare against snapshot from ~{oi_lookback_mins} min ago. "
            f"OI change columns show full-day open interest change."
        )
    else:
        cols_to_show = [
            "strike",
            "ce_oi",
            "ce_oi_change",
            "ce_iv",
            "ce_ltp",
            "pe_ltp",
            "pe_iv",
            "pe_oi_change",
            "pe_oi",
        ]
        st.caption(
            "OI delta columns will appear after the initial lookback window passes."
        )

    # Filter columns that exist in DataFrame
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    chain_display = display_df[cols_to_show].copy()

    # Precision and sign formatting
    chain_format = {}
    if "strike" in chain_display.columns:
        chain_format["strike"] = "{:,.0f}"
    if "ce_oi" in chain_display.columns:
        chain_format["ce_oi"] = "{:,.0f}"
    if "ce_oi_change" in chain_display.columns:
        chain_format["ce_oi_change"] = "{:+,.0f}"
    if "ce_oi_delta" in chain_display.columns:
        chain_format["ce_oi_delta"] = "{:+,.0f}"
    if "ce_iv" in chain_display.columns:
        chain_format["ce_iv"] = "{:.2f}"
    if "ce_ltp" in chain_display.columns:
        chain_format["ce_ltp"] = "{:.2f}"
    if "pe_ltp" in chain_display.columns:
        chain_format["pe_ltp"] = "{:.2f}"
    if "pe_iv" in chain_display.columns:
        chain_format["pe_iv"] = "{:.2f}"
    if "pe_oi_delta" in chain_display.columns:
        chain_format["pe_oi_delta"] = "{:+,.0f}"
    if "pe_oi_change" in chain_display.columns:
        chain_format["pe_oi_change"] = "{:+,.0f}"
    if "pe_oi" in chain_display.columns:
        chain_format["pe_oi"] = "{:,.0f}"

    st.dataframe(
        chain_display.style.format(
            chain_format,
            na_rep="—",
        ),
        hide_index=True,
        use_container_width=True,
        height=450,
    )

    # -------------------------------------------------------------
    # Altair OI Visualizations
    # -------------------------------------------------------------
    st.markdown("## OI by Strike")

    chart_slice = df.iloc[lo:hi].copy()
    chart_slice["Strike"] = chart_slice["strike"].astype(float)

    # Chart 1 — Total OI by Strike
    oi_long = chart_slice[
        ["Strike", "ce_oi", "pe_oi"]
    ].melt(
        id_vars="Strike",
        value_vars=["ce_oi", "pe_oi"],
        var_name="Side",
        value_name="OI",
    )

    oi_long["Side"] = oi_long["Side"].map({
        "ce_oi": "Call OI",
        "pe_oi": "Put OI",
    })

    oi_bars = (
        alt.Chart(oi_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "Strike:O",
                sort=sorted(chart_slice["Strike"].unique().tolist()),
                axis=alt.Axis(title="Strike", labelAngle=-45),
            ),
            xOffset=alt.XOffset("Side:N", title=None),
            y=alt.Y(
                "OI:Q",
                title="Open Interest",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Side:N",
                title=None,
                scale=alt.Scale(
                    domain=["Call OI", "Put OI"],
                    range=["#0066CC", "#E91E63"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Strike:Q", title="Strike", format=".0f"),
                alt.Tooltip("Side:N", title="Side"),
                alt.Tooltip("OI:Q", title="OI", format=","),
            ],
        )
    )

    oi_labels = (
        alt.Chart(oi_long)
        .mark_text(
            dy=-6,
            fontSize=10,
        )
        .encode(
            x=alt.X(
                "Strike:O",
                sort=sorted(chart_slice["Strike"].unique().tolist()),
            ),
            xOffset=alt.XOffset("Side:N"),
            y=alt.Y("OI:Q"),
            text=alt.Text("OI:Q", format=","),
            color=alt.Color("Side:N", legend=None),
        )
    )

    if float(chart_slice["Strike"].min()) <= float(spot) <= float(chart_slice["Strike"].max()):
        oi_chart = (
            (oi_bars + oi_labels)
            .properties(
                height=390,
                title=f"ATM CE OI vs PE OI  |  Spot {spot:,.0f}",
            )
        )
    else:
        oi_chart = (oi_bars + oi_labels).properties(
            height=390,
            title="ATM CE OI vs PE OI",
        )

    st.altair_chart(oi_chart, use_container_width=True)

    # Chart 2 — Day's Change in OI
    change_long = chart_slice[
        ["Strike", "ce_oi_change", "pe_oi_change"]
    ].melt(
        id_vars="Strike",
        value_vars=["ce_oi_change", "pe_oi_change"],
        var_name="Side",
        value_name="OI Change",
    )

    change_long["Side"] = change_long["Side"].map({
        "ce_oi_change": "Call OI Change",
        "pe_oi_change": "Put OI Change",
    })

    change_bars = (
        alt.Chart(change_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "Strike:O",
                sort=sorted(chart_slice["Strike"].unique().tolist()),
                axis=alt.Axis(title="Strike", labelAngle=-45),
            ),
            xOffset=alt.XOffset("Side:N", title=None),
            y=alt.Y(
                "OI Change:Q",
                title="Change in Open Interest",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Side:N",
                title=None,
                scale=alt.Scale(
                    domain=["Call OI Change", "Put OI Change"],
                    range=["#0066CC", "#E91E63"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Strike:Q", title="Strike", format=".0f"),
                alt.Tooltip("Side:N", title="Side"),
                alt.Tooltip(
                    "OI Change:Q",
                    title="OI Change",
                    format="+,",
                ),
            ],
        )
    )

    change_labels = (
        alt.Chart(change_long)
        .mark_text(
            dy=-7,
            fontSize=10,
        )
        .encode(
            x=alt.X(
                "Strike:O",
                sort=sorted(chart_slice["Strike"].unique().tolist()),
            ),
            xOffset=alt.XOffset("Side:N"),
            y=alt.Y("OI Change:Q"),
            text=alt.Text("OI Change:Q", format="+,"),
            color=alt.Color(
                "Side:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Call OI Change", "Put OI Change"],
                    range=["#0066CC", "#E91E63"],
                ),
            ),
        )
    )

    change_chart = (
        change_bars + change_labels
    ).properties(
        height=390,
        title="ATM CE OI Change vs PE OI Change",
    )

    st.altair_chart(change_chart, use_container_width=True)

    st.caption(
        "Positive OI Change = fresh OI addition. "
        "Negative OI Change = OI reduction. "
        "Hover over any bar for exact values."
    )


# ---------------------------------------------------------------------
# Auto refresh loop
# ---------------------------------------------------------------------
time.sleep(refresh_secs)
st.rerun()
