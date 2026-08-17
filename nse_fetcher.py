"""
nse_fetcher.py — FYERS API & NSE Option Chain Adapter.
"""

from __future__ import annotations
import time
from typing import Any
import pandas as pd

try:
    from fyers_apiv3 import fyersModel
    HAS_FYERS = True
except ImportError:
    HAS_FYERS = False

SUPPORTED_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


class NseSession:
    """Option chain fetcher using FYERS API v3."""

    def __init__(self, client_id: str | None = None, access_token: str | None = None):
        self.client_id = client_id
        self.access_token = access_token
        self.fyers = None

        if HAS_FYERS and client_id and access_token:
            self.fyers = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                is_async=False,
                log_path=""
            )

    @staticmethod
    def _clean_number(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            val = float(value)
            return default if val != val else val
        except (TypeError, ValueError):
            return default

    def get_option_chain(
        self,
        symbol: str = "NIFTY",
        retries: int = 3,
        expiry: str | None = None,
        client_id: str | None = None,
        access_token: str | None = None,
    ) -> dict:
        symbol = str(symbol).upper().strip()

        if symbol not in SUPPORTED_SYMBOLS:
            raise RuntimeError(
                f"Unsupported index '{symbol}'. Use one of: {', '.join(sorted(SUPPORTED_SYMBOLS))}"
            )

        cid = client_id or self.client_id
        token = access_token or self.access_token

        if not cid or not token:
            raise RuntimeError(
                "FYERS credentials missing. Please enter your App ID and Access Token in the dashboard sidebar."
            )

        fyers_client = self.fyers
        if fyers_client is None or cid != self.client_id or token != self.access_token:
            if not HAS_FYERS:
                raise RuntimeError("fyers-apiv3 package is not installed.")
            fyers_client = fyersModel.FyersModel(
                client_id=cid,
                token=token,
                is_async=False,
                log_path=""
            )

        fyers_symbol = f"NSE:{symbol}-INDEX"
        data_req = {
            "symbol": fyers_symbol,
            "strikecount": 30,
            "timestamp": ""
        }

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = fyers_client.optionchain(data=data_req)
                if not isinstance(response, dict) or response.get("s") != "ok":
                    msg = response.get("message", "API response error") if isinstance(response, dict) else str(response)
                    raise RuntimeError(f"FYERS API error: {msg}")

                chain_data = response.get("data", {})
                options_chain = chain_data.get("optionsChain", [])
                
                expiry_data = chain_data.get("expiryData", [])
                underlying = 0.0
                if expiry_data:
                    underlying = self._clean_number(expiry_data[0].get("underlyingValue"))

                expiries = sorted(list({
                    item.get("expiry") for item in options_chain if item.get("expiry")
                }))

                selected_expiry = expiry if expiry else (expiries[0] if expiries else "")
                rows = []

                for strike_info in options_chain:
                    if selected_expiry and strike_info.get("expiry") != selected_expiry:
                        continue

                    strike = self._clean_number(strike_info.get("strike_price"))
                    if strike <= 0:
                        continue

                    ce_data = strike_info.get("ce", {}) or {}
                    pe_data = strike_info.get("pe", {}) or {}

                    ce = {
                        "openInterest": self._clean_number(ce_data.get("oi")),
                        "changeinOpenInterest": self._clean_number(ce_data.get("oichange")),
                        "totalTradedVolume": self._clean_number(ce_data.get("volume")),
                        "impliedVolatility": self._clean_number(ce_data.get("optionAskIv")),
                        "lastPrice": self._clean_number(ce_data.get("ltp")),
                    }

                    pe = {
                        "openInterest": self._clean_number(pe_data.get("oi")),
                        "changeinOpenInterest": self._clean_number(pe_data.get("oichange")),
                        "totalTradedVolume": self._clean_number(pe_data.get("volume")),
                        "impliedVolatility": self._clean_number(pe_data.get("optionAskIv")),
                        "lastPrice": self._clean_number(pe_data.get("ltp")),
                    }

                    rows.append({
                        "expiryDate": selected_expiry,
                        "strikePrice": strike,
                        "CE": ce,
                        "PE": pe,
                    })

                if not rows:
                    raise RuntimeError(
                        f"No option chain data returned for {symbol} / {selected_expiry}."
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
                    time.sleep(1.5)

        raise RuntimeError(f"Failed to fetch option chain: {last_error}")


if __name__ == "__main__":
    print("NseSession ready for FYERS API.")
