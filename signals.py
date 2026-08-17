"""
signals.py
Turns raw option-chain data into key analytical signals:
PCR, OI buildup/unwind per strike, max pain, IV skew, and expected move.
"""

import pandas as pd


def chain_to_dataframe(raw_json, expiry=None):
    """
    Flattens raw JSON into a clean per-strike DataFrame with separate CE/PE columns.
    Handles FYERS & NSE dictionary structures flexibly.
    """
    records = raw_json.get("records", {})
    spot = float(records.get("underlyingValue", 0.0))
    raw_chain = records.get("data", [])

    if expiry is None and records.get("expiryDates"):
        expiry = records["expiryDates"][0]

    rows = []
    for r in raw_chain:
        # Expiry filtering if present in item level
        exp_val = r.get("expiryDate") or r.get("expiry") or r.get("expiry_date")
        if expiry and exp_val and str(exp_val) != str(expiry):
            continue

        strike = float(r.get("strikePrice", r.get("strike_price", r.get("strike", 0))))

        # Extract Call & Put details flexibly
        ce = r.get("CE", r.get("call", {}))
        pe = r.get("PE", r.get("put", {}))

        rows.append({
            "strike": strike,
            "ce_oi": float(ce.get("openInterest", ce.get("oi", ce.get("open_interest", 0)))),
            "ce_oi_change": float(ce.get("changeinOpenInterest", ce.get("oichange", ce.get("change_in_oi", 0)))),
            "ce_volume": float(ce.get("totalTradedVolume", ce.get("volume", 0))),
            "ce_iv": float(ce.get("impliedVolatility", ce.get("option_iv", ce.get("iv", 0)))),
            "ce_ltp": float(ce.get("lastPrice", ce.get("ltp", ce.get("last_price", 0)))),
            "pe_oi": float(pe.get("openInterest", pe.get("oi", pe.get("open_interest", 0)))),
            "pe_oi_change": float(pe.get("changeinOpenInterest", pe.get("oichange", pe.get("change_in_oi", 0)))),
            "pe_volume": float(pe.get("totalTradedVolume", pe.get("volume", 0))),
            "pe_iv": float(pe.get("impliedVolatility", pe.get("option_iv", pe.get("iv", 0)))),
            "pe_ltp": float(pe.get("lastPrice", pe.get("ltp", pe.get("last_price", 0)))),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("strike").reset_index(drop=True)

    return df, spot, expiry


def compute_pcr(df):
    """Put-Call Ratio by OI. >1 = more put OI (bullish lean), <1 = call-heavy."""
    if df.empty or "ce_oi" not in df.columns:
        return None
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
    if df.empty or "strike" not in df.columns:
        return None
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
    if df.empty or "strike" not in df.columns or spot == 0:
        return {"ce_iv_avg": None, "pe_iv_avg": None, "skew": None}

    atm_idx = (df["strike"] - spot).abs().idxmin()
    lo = max(0, atm_idx - width)
    hi = min(len(df), atm_idx + width + 1)
    window = df.iloc[lo:hi]
    
    ce_iv = window.loc[window["ce_iv"] > 0, "ce_iv"].mean()
    pe_iv = window.loc[window["pe_iv"] > 0, "pe_iv"].mean()
    
    ce_avg = round(ce_iv, 2) if pd.notna(ce_iv) else None
    pe_avg = round(pe_iv, 2) if pd.notna(pe_iv) else None
    skew = round(pe_iv - ce_iv, 2) if (ce_avg is not None and pe_avg is not None) else None

    return {
        "ce_iv_avg": ce_avg,
        "pe_iv_avg": pe_avg,
        "skew": skew,
    }


def compute_expected_move(df, spot):
    """ATM straddle price * 0.68 ≈ market-implied 1-sigma move to expiry."""
    if df.empty or "strike" not in df.columns or spot == 0:
        return {"atm_strike": 0, "straddle_price": 0.0, "expected_move_1sigma": 0.0}

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
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_ce = df.nlargest(n, "ce_oi")[["strike", "ce_oi"]]
    top_pe = df.nlargest(n, "pe_oi")[["strike", "pe_oi"]]
    return top_ce, top_pe


def oi_change_vs_previous(df_now, df_prev):
    """
    Merges current vs previous snapshot to flag fresh buildup/unwind per strike.
    """
    if df_prev is None or df_now.empty:
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
    This is a heuristic summary of positioning, NOT a trade signal or prediction.
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

    skew = iv_skew.get("skew") if isinstance(iv_skew, dict) else None
    if skew is not None:
        if skew > 1.5:
            score -= 1
            notes.append(f"IV skew +{skew} — puts pricier, downside hedging demand")
        elif skew < -1.5:
            score += 1
            notes.append(f"IV skew {skew} — calls pricier, upside chase")
        else:
            notes.append(f"IV skew {skew} — flat")

    if max_pain is not None and spot > 0:
        diff_pct = (spot - max_pain) / spot * 100
        notes.append(f"Spot is {diff_pct:+.2f}% vs max pain ({max_pain:,.0f})")

    if score >= 1:
        verdict = "Mild bullish lean"
    elif score <= -1:
        verdict = "Mild bearish lean"
    else:
        verdict = "Neutral / mixed positioning"

    return verdict, notes
