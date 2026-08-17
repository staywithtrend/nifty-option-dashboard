"""
nse_fetcher.py — robust NSE option-chain adapter.

Provides NseSession compatible with app.py and signals.py, with 
automatic fallback session handling for Akamai/NSE rate limiting.
"""

from __future__ import annotations

import time
from typing import Any
import pandas as pd

# Compatibility patch for curl_cffi / PNSEA
try:
    import curl_cffi.requests as requests
    from curl_cffi.requests import Session
except ImportError as exc:
    raise ImportError(
        "curl_cffi is required. Run: pip install curl-cffi pnsea"
    ) from exc

try:
    from pnsea import NSE
except ImportError:
    NSE = None

SUPPORTED_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


class NseSession:
    """Persistent NSE option chain session with auto-retry & cookie handling."""

    def __init__(self):
        self.pnsea = NSE() if NSE is not None else None
        self._last_fetch = 0.0
        self._min_interval = 3.0
        
        # Direct session fallback setup
        self.session = Session(impersonate="chrome120")
        self.has_cookies = False

    @staticmethod
    def _clean_number(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if value != value:  # NaN check
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _respect_rate_limit(self):
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _init_direct_cookies(self):
        """Fetch valid session cookies from NSE homepage if PNSEA fails."""
        try:
            url = "https://www.nseindia.com/option-chain"
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                self.has_cookies = True
        except Exception:
            self.has_cookies = False

    def _fetch_direct(self, symbol: str) -> dict:
        """Direct fallback API request matching native NSE JSON structure."""
        if not self.has_cookies:
            self._init_direct_cookies()
            time.sleep(0.5)

        api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        headers = {
            "Referer": f"https://www.nseindia.com/option-chain?symbol={symbol}",
            "Accept": "application/json, text/plain, */*",
        }
        
        res = self.session.get(api_url, headers=headers, timeout=10)
        if res.status_code in (401, 403):
            self.has_cookies = False
            raise RuntimeError("NSE Session expired / IP blocked")

        return res.json()

    def _fetch_once(self, symbol: str, expiry: str | None = None):
        self._respect_rate_limit()

        # Attempt 1: PNSEA Library
        if self.pnsea is not None:
            try:
                if expiry:
                    result = self.pnsea.options.option_chain(
                        symbol, expiry_date=expiry
                    )
                else:
                    result = self.pnsea.options.option_chain(symbol)

                self._last_fetch = time.monotonic()

                if isinstance(result, (tuple, list)) and len(result) >= 3:
                    df, expiries, underlying = result[0], result[1], result[2]
                    if df is not None and not getattr(df, "empty", True):
                        return df, list(expiries), self._clean_number(underlying)
            except Exception:
                # Fall through to direct cURL fetcher on PNSEA failure
                pass

        # Attempt 2: Direct curl_cffi fallback
        raw_json = self._fetch_direct(symbol)
        records = raw_json.get("records", {})
        expiries = records.get("expiryDates", [])
        underlying = self._clean_number(records.get("underlyingValue"))
        raw_data = records.get("data", [])

        if not raw_data:
            raise RuntimeError(f"NSE returned empty data for {symbol}")

        # Convert raw JSON records into expected DataFrame format
        rows = []
        target_expiry = expiry if expiry else (expiries[0] if expiries else None)

        for item in raw_data:
            if target_expiry and item.get("expiryDate") != target_expiry:
                continue

            strike = item.get("strikePrice")
            ce_data = item.get("CE", {})
            pe_data = item.get("PE", {})

            rows.append({
                "strikePrice": strike,
                "CE_openInterest": ce_data.get("openInterest"),
                "CE_changeinOpenInterest": ce_data.get("changeinOpenInterest"),
                "CE_totalTradedVolume": ce_data.get("totalTradedVolume"),
                "CE_impliedVolatility": ce_data.get("impliedVolatility"),
                "CE_lastPrice": ce_data.get("lastPrice"),
                "PE_openInterest": pe_data.get("openInterest"),
                "PE_changeinOpenInterest": pe_data.get("changeinOpenInterest"),
                "PE_totalTradedVolume": pe_data.get("totalTradedVolume"),
                "PE_impliedVolatility": pe_data.get("impliedVolatility"),
                "PE_lastPrice": pe_data.get("lastPrice"),
            })

        df = pd.DataFrame(rows)
        self._last_fetch = time.monotonic()
        return df, expiries, underlying

    def get_option_chain(
        self,
        symbol="NIFTY",
        retries=3,
        expiry: str | None = None,
    ):
        symbol = str(symbol).upper().strip()

        if symbol not in SUPPORTED_SYMBOLS:
            raise RuntimeError(
                f"Unsupported index '{symbol}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_SYMBOLS))}"
            )

        last_error = None

        for attempt in range(1, retries + 1):
            try:
                df, expiries, underlying = self._fetch_once(
                    symbol, expiry=expiry
                )

                selected_expiry = str(expiry) if expiry else str(expiries[0])
                rows = []

                for _, row in df.iterrows():
                    strike = self._clean_number(row.get("strikePrice"))
                    if strike <= 0:
                        continue

                    ce = {
                        "openInterest": self._clean_number(row.get("CE_openInterest")),
                        "changeinOpenInterest": self._clean_number(row.get("CE_changeinOpenInterest")),
                        "totalTradedVolume": self._clean_number(row.get("CE_totalTradedVolume")),
                        "impliedVolatility": self._clean_number(row.get("CE_impliedVolatility")),
                        "lastPrice": self._clean_number(row.get("CE_lastPrice")),
                    }

                    pe = {
                        "openInterest": self._clean_number(row.get("PE_openInterest")),
                        "changeinOpenInterest": self._clean_number(row.get("PE_changeinOpenInterest")),
                        "totalTradedVolume": self._clean_number(row.get("PE_totalTradedVolume")),
                        "impliedVolatility": self._clean_number(row.get("PE_impliedVolatility")),
                        "lastPrice": self._clean_number(row.get("PE_lastPrice")),
                    }

                    rows.append({
                        "expiryDate": selected_expiry,
                        "strikePrice": strike,
                        "CE": ce,
                        "PE": pe,
                    })

                if not rows:
                    raise RuntimeError(
                        f"NSE returned no usable strike rows for {symbol} / {selected_expiry}."
                    )

                return {
                    "records": {
                        "underlyingValue": underlying,
                        "expiryDates": expiries,
                        "data": rows,
                    }
                }

            except Exception as exc:
                last_error = str(exc)
                self.has_cookies = False  # Reset cookies on error
                if attempt < retries:
                    time.sleep(2.0 * attempt)

        suffix = f" / {expiry}" if expiry else ""
        raise RuntimeError(
            f"Failed to fetch NSE option chain for {symbol}{suffix}: {last_error}"
        )


if __name__ == "__main__":
    session = NseSession()
    data = session.get_option_chain("NIFTY")
    records = data["records"]
    print("Underlying spot:", records["underlyingValue"])
    print("Expiry dates:", records["expiryDates"][:3])
    print("Nearest expiry strikes:", len(records["data"]))
