# Nifty Options Signal Dashboard (NSE public data)

Live PCR, OI buildup/unwind, max pain, IV skew, and expected move for
NIFTY / BANKNIFTY / FINNIFTY / MIDCPNIFTY — pulled straight from NSE's
own public option-chain feed. No broker login, no API key.

## Setup (one-time)

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run dashboard.py
```

It opens in your browser at `http://localhost:8501`. Leave the terminal
running — it auto-refreshes on the interval you set in the sidebar.

## What it shows

- **PCR (OI)** — put/call ratio by open interest across the chain
- **Max Pain** — the strike where option writers lose least at expiry
- **Expected Move** — ATM straddle price × 0.68 ≈ market-implied 1-sigma move to expiry
- **IV Skew** — average put IV minus call IV near ATM (hedging demand vs upside chase)
- **OI Walls** — top strikes by call OI and put OI (informal support/resistance)
- **OI change table** — per-strike CE/PE OI delta vs N minutes ago (set in sidebar)
- **Positioning read** — a plain-English one-line summary combining the above

**This is a read of current options positioning, not a trade signal or
prediction.** Pair it with your own technical/price view — same as
you'd treat PCR or max pain from any options screener.

## Notes on reliability

- NSE occasionally rate-limits or briefly blocks non-browser requests.
  The fetcher re-warms its session and retries automatically; if a
  refresh fails it'll usually recover on the next cycle.
- OI-change columns need a second snapshot to compare against, so
  they're blank for the first few minutes after you start the app.
- Works outside market hours too, but shows the last traded/closing
  chain rather than live-moving data.

## Natural next steps (not built yet)

- Persist snapshots to disk/SQLite so OI history survives a restart
- Add your RSVM technical signal as a second panel alongside the
  options-positioning read
- Swap in Fyers API once your API app is approved, for real-time tick
  data instead of NSE's ~3-second-delayed polling
