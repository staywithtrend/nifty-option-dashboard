"""
NSE & FYERS Data Fetcher for Option Chain data.
Handles API requests to FYERS and standardizes responses.
"""

from fyers_apiv3 import fyersModel


class NseSession:
    def __init__(self):
        pass

    def get_option_chain(self, symbol="NIFTY", expiry=None, client_id=None, access_token=None):
        """
        Fetches the live option chain from FYERS API and maps it into a format
        compatible with the signals processor.
        """
        if not client_id or not access_token:
            raise RuntimeError("FYERS Client ID and Access Token are required in the sidebar.")

        # Initialize FYERS API Model
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            is_async=False,
            token=access_token,
            log_path=""
        )

        # Map index symbols to FYERS Option Chain symbol format
        symbol_map = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX",
            "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX"
        }

        fyers_symbol = symbol_map.get(symbol, f"NSE:{symbol}-INDEX")

        data = {
            "symbol": fyers_symbol,
            "strikecount": 25,
            "timestamp": ""
        }

        # Query FYERS Option Chain endpoint
        response = fyers.optionchain(data=data)

        if not response or response.get("s") != "ok":
            error_msg = response.get("message", "Unknown error or invalid token from FYERS API") if isinstance(response, dict) else str(response)
            raise RuntimeError(f"FYERS API Error: {error_msg}")

        chain_data = response.get("data", {})
        
        # Extract Expiries list
        raw_expiries = chain_data.get("expiryData", [])
        expiry_dates = [exp.get("date") for exp in raw_expiries] if raw_expiries else []

        # Standardize payload output for signals.py
        return {
            "records": {
                "expiryDates": expiry_dates,
                "data": chain_data.get("optionsChain", []),
                "underlyingValue": float(chain_data.get("strikePrice", 0) or chain_data.get("lastPrice", 0))
            }
        }
