"""
nse_fetcher.py — robust NSE option-chain adapter.

The dashboard can request a specific expiry:
    get_option_chain("NIFTY", expiry="18-Aug-2026")

PNSEA supports filtering the option chain by expiry_date, which is important
because the default option_chain() response can return the nearest-expiry
chain while still reporting the list of all available expiries.
"""

from __future__ import annotations

import time
from typing import Any

# --- FIX: Compatibility patch for PNSEA and curl_cffi ---
try:
    import curl_cffi.requests
    if not hasattr(curl_cffi.requests, "RequestException"):
        try:
            from curl_cffi.requests.errors import RequestsError
            curl_cffi.requests.RequestException = RequestsError
        except Exception:
            curl_cffi.requests.RequestException = Exception
except Exception:
    pass
# ---------------------------------------------------------

try:
    from pnsea import NSE
except ImportError as exc:
    raise ImportError(
        "PNSEA is not installed. Run: python -m pip install -U pnsea==1.1"
    ) from exc


SUPPORTED_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


class NseSession:
    """Persistent PNSEA-backed NSE session used by dashboard.py."""

    def __init__(self):
        self.nse = NSE()
        self._last_fetch = 0.0
        self._min_interval = 3.0

    @staticmethod
    def _clean_number(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if value != value:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _respect_rate_limit(self):
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _fetch_once(self, symbol: str, expiry: str | None = None):
        self._respect_rate_limit()

        if expiry:
            result = self.nse.options.option_chain(
                symbol,
                expiry_date=expiry,
            )
        else:
            result = self.nse.options.option_chain(symbol)

        self._last_fetch = time.monotonic()

        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise RuntimeError(
                f"Unexpected PNSEA option-chain response for {symbol}."
            )

        df, expiries, underlying = result[0], result[1], result[2]

        if df is None or getattr(df, "empty", True):
            suffix = f" for expiry {expiry}" if expiry else ""
            raise RuntimeError(
                f"NSE returned an empty option chain for {symbol}{suffix}."
            )

        if not expiries:
            raise RuntimeError(f"NSE returned no expiry dates for {symbol}.")

        return df, list(expiries), self._clean_number(underlying)

    def get_option_chain(
        self,
        symbol="NIFTY",
        retries=3,
        expiry: str | None = None,
    ):
        """
        Return raw JSON compatible with the existing signals.py.

        Supported symbols:
            NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY

        expiry:
            Optional NSE expiry string such as '18-Aug-2026'.
        """
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
                    symbol,
                    expiry=expiry,
                )

                # When a specific expiry was requested, that is the expiry
                # represented by the returned DataFrame.
                selected_expiry = str(expiry) if expiry else str(expiries[0])

                rows = []

                for _, row in df.iterrows():
                    strike = self._clean_number(row.get("strikePrice"))
                    if strike <= 0:
                        continue

                    ce = {
                        "openInterest": self._clean_number(
                            row.get("CE_openInterest")
                        ),
                        "changeinOpenInterest": self._clean_number(
                            row.get("CE_changeinOpenInterest")
                        ),
                        "totalTradedVolume": self._clean_number(
                            row.get("CE_totalTradedVolume")
                        ),
                        "impliedVolatility": self._clean_number(
                            row.get("CE_impliedVolatility")
                        ),
                        "lastPrice": self._clean_number(
                            row.get("CE_lastPrice")
                        ),
                    }

                    pe = {
                        "openInterest": self._clean_number(
                            row.get("PE_openInterest")
                        ),
                        "changeinOpenInterest": self._clean_number(
                            row.get("PE_changeinOpenInterest")
                        ),
                        "totalTradedVolume": self._clean_number(
                            row.get("PE_totalTradedVolume")
                        ),
                        "impliedVolatility": self._clean_number(
                            row.get("PE_impliedVolatility")
                        ),
                        "lastPrice": self._clean_number(
                            row.get("PE_lastPrice")
                        ),
                    }

                    rows.append(
                        {
                            "expiryDate": selected_expiry,
                            "strikePrice": strike,
                            "CE": ce,
                            "PE": pe,
                        }
                    )

                if not rows:
                    raise RuntimeError(
                        f"NSE returned no usable strike rows for "
                        f"{symbol} / {selected_expiry}."
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
                if attempt < retries:
                    time.sleep(2.0 * attempt)

        suffix = f" / {expiry}" if expiry else ""
        raise RuntimeError(
            f"Failed to fetch NSE option chain for {symbol}{suffix}: "
            f"{last_error}"
        )


if __name__ == "__main__":
    session = NseSession()

    # Test nearest expiry.
    data = session.get_option_chain("NIFTY")
    records = data["records"]
    print("Underlying spot:", records["underlyingValue"])
    print("Expiry dates:", records["expiryDates"][:3])
    print("Nearest expiry strikes:", len(records["data"]))

    # Test a selectable expiry if available.
    if len(records["expiryDates"]) > 1:
        target = records["expiryDates"][1]
        selected = session.get_option_chain("NIFTY", expiry=target)
        print(f"{target} strikes:", len(selected["records"]["data"]))
