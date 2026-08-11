"""
signals.py
Turns raw NSE option-chain JSON into the signals that actually matter:
PCR, OI buildup/unwind per strike, max pain, IV skew, and expected move.

Every function here is a pure read of the current chain snapshot except
oi_change_vs_previous, which needs a prior snapshot to compare against —
the dashboard is responsible for keeping that history.
"""

import pandas as pd


def chain_to_dataframe(raw_json, expiry=None):
    """
    Flattens NSE's nested JSON into a clean per-strike DataFrame with
    separate CE/PE columns. If expiry is None, uses the nearest expiry.
    """
    records = raw_json["records"]["data"]
    if expiry is None:
        expiry = raw_json["records"]["expiryDates"][0]

    rows = []
    for r in records:
        if r.get("expiryDate") != expiry:
            continue
        strike = r["strikePrice"]
        ce = r.get("CE", {})
        pe = r.get("PE", {})
        rows.append({
            "strike": strike,
            "ce_oi": ce.get("openInterest", 0),
            "ce_oi_change": ce.get("changeinOpenInterest", 0),
            "ce_volume": ce.get("totalTradedVolume", 0),
            "ce_iv": ce.get("impliedVolatility", 0),
            "ce_ltp": ce.get("lastPrice", 0),
            "pe_oi": pe.get("openInterest", 0),
            "pe_oi_change": pe.get("changeinOpenInterest", 0),
            "pe_volume": pe.get("totalTradedVolume", 0),
            "pe_iv": pe.get("impliedVolatility", 0),
            "pe_ltp": pe.get("lastPrice", 0),
        })
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    spot = raw_json["records"]["underlyingValue"]
    return df, spot, expiry


def compute_pcr(df):
    """Put-Call Ratio by OI. >1 = more put OI (bullish lean by convention), <1 = call-heavy."""
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    if total_ce_oi == 0:
        return None
    return round(total_pe_oi / total_ce_oi, 3)


def compute_max_pain(df):
    """
    Strike where option WRITERS lose the least — i.e. where total payout
    to option buyers (CE+PE) is minimized at expiry.
    """
    strikes = df["strike"].values
    min_pain = None
    min_strike = None
    for s in strikes:
        ce_loss = ((s - df["strike"]).clip(lower=0) * df["ce_oi"]).sum()
        pe_loss = ((df["strike"] - s).clip(lower=0) * df["pe_oi"]).sum()
        total = ce_loss + pe_loss
        if min_pain is None or total < min_pain:
            min_pain = total
            min_strike = s
    return min_strike


def compute_iv_skew(df, spot, width=3):
    """
    Average CE IV vs PE IV for strikes within `width` steps of ATM.
    Positive skew (PE IV > CE IV) usually reflects downside hedging demand.
    """
    atm_idx = (df["strike"] - spot).abs().idxmin()
    lo = max(0, atm_idx - width)
    hi = min(len(df), atm_idx + width + 1)
    window = df.iloc[lo:hi]
    ce_iv = window.loc[window["ce_iv"] > 0, "ce_iv"].mean()
    pe_iv = window.loc[window["pe_iv"] > 0, "pe_iv"].mean()
    return {
        "ce_iv_avg": round(ce_iv, 2) if pd.notna(ce_iv) else None,
        "pe_iv_avg": round(pe_iv, 2) if pd.notna(pe_iv) else None,
        "skew": round(pe_iv - ce_iv, 2) if pd.notna(ce_iv) and pd.notna(pe_iv) else None,
    }


def compute_expected_move(df, spot):
    """ATM straddle price * 0.68 ≈ market-implied 1-sigma move to expiry."""
    atm_idx = (df["strike"] - spot).abs().idxmin()
    row = df.iloc[atm_idx]
    straddle = row["ce_ltp"] + row["pe_ltp"]
    return {
        "atm_strike": row["strike"],
        "straddle_price": round(straddle, 2),
        "expected_move_1sigma": round(straddle * 0.68, 2),
    }


def top_oi_strikes(df, n=3):
    """Top N strikes by CE OI and PE OI — these behave like support/resistance walls."""
    top_ce = df.nlargest(n, "ce_oi")[["strike", "ce_oi"]]
    top_pe = df.nlargest(n, "pe_oi")[["strike", "pe_oi"]]
    return top_ce, top_pe


def oi_change_vs_previous(df_now, df_prev):
    """
    Merges current vs previous snapshot to flag fresh buildup/unwind per strike.
    Requires the dashboard to have cached a previous snapshot at least a
    few minutes old — comparing to itself (0 seconds) is meaningless.
    """
    if df_prev is None:
        return None
    merged = df_now.merge(
        df_prev[["strike", "ce_oi", "pe_oi"]],
        on="strike", how="left", suffixes=("", "_prev")
    )
    merged["ce_oi_delta"] = merged["ce_oi"] - merged["ce_oi_prev"].fillna(merged["ce_oi"])
    merged["pe_oi_delta"] = merged["pe_oi"] - merged["pe_oi_prev"].fillna(merged["pe_oi"])
    return merged


def bias_verdict(pcr, iv_skew, max_pain, spot):
    """
    Combines PCR + IV skew + max-pain-vs-spot into a single plain-English read.
    This is a heuristic summary of positioning, NOT a trade signal or prediction —
    label it that way wherever it's shown.
    """
    score = 0
    notes = []

    if pcr is not None:
        if pcr > 1.15:
            score += 1
            notes.append(f"PCR {pcr} — put-heavy (bullish lean by convention)")
        elif pcr < 0.85:
            score -= 1
            notes.append(f"PCR {pcr} — call-heavy (bearish lean by convention)")
        else:
            notes.append(f"PCR {pcr} — balanced")

    skew = iv_skew.get("skew")
    if skew is not None:
        if skew > 1.5:
            score -= 1
            notes.append(f"IV skew +{skew} — puts pricier, downside hedging demand")
        elif skew < -1.5:
            score += 1
            notes.append(f"IV skew {skew} — calls pricier, upside chase")
        else:
            notes.append(f"IV skew {skew} — flat")

    if max_pain is not None:
        diff_pct = (spot - max_pain) / spot * 100
        notes.append(f"Spot is {diff_pct:+.2f}% vs max pain ({max_pain})")

    if score >= 1:
        verdict = "Mild bullish lean"
    elif score <= -1:
        verdict = "Mild bearish lean"
    else:
        verdict = "Neutral / mixed positioning"

    return verdict, notes
