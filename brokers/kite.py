"""
brokers/kite.py
---------------
Zerodha Kite Connect adapter.

Requires: pip install kiteconnect

KEY GOTCHAS (these will bite you):
1. Access token expires DAILY at ~6 AM IST. Re-login every day.
2. Historical data has a CALL LIMIT: typically 60 days per minute-resolution
   request. We chunk requests automatically.
3. Rate limit: 3 requests/second, 200/minute. Honor them or get banned.
4. Instrument tokens change with each contract; you cannot hardcode them.
5. Options expiry data goes back roughly 365 days for minute bars; older
   data is daily-only (a hard limit set by Kite).
6. `from` and `to` are INCLUSIVE in Kite's API. A request for
   '2024-01-01' to '2024-01-01' returns one day. Don't over-shoot.

Reference: https://kite.trade/docs/connect/v3/historical/
"""
from __future__ import annotations
import time
import logging
from datetime import datetime, date, timedelta
from typing import List
import pandas as pd

from .base import BrokerAdapter, OHLCBar, OptionContract

log = logging.getLogger(__name__)

try:
    from kiteconnect import KiteConnect  # noqa: F401
    HAS_KITE = True
except ImportError:
    HAS_KITE = False


class KiteAdapter(BrokerAdapter):
    """
    Zerodha Kite Connect adapter.

    Login flow is multi-step and INTERACTIVE on first run:
      1. Go to login URL printed by login()
      2. Authorize, copy 'request_token' from redirect URL
      3. Pass to generate_session(request_token, api_secret)
      4. Save the access_token; it's valid until next 6 AM IST
    """

    RATE_LIMIT_DELAY = 0.34  # ~3 req/sec; safer than the documented limit
    HIST_MINUTE_CHUNK_DAYS = 60   # Kite's hard cap for minute resolution

    def __init__(self, api_key: str, api_secret: str,
                 access_token: str | None = None):
        if not HAS_KITE:
            raise ImportError("kiteconnect not installed. "
                              "pip install kiteconnect")
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        if access_token:
            self.kite.set_access_token(access_token)
        self._instrument_cache: pd.DataFrame | None = None
        self._last_call_ts = 0.0

    # ----- Auth -----
    def login(self) -> None:
        """
        For headless / programmatic use, the user must run the interactive
        login flow once per day and paste the request_token. This method
        verifies the existing access_token works.
        """
        if not self.access_token:
            url = self.kite.login_url()
            raise RuntimeError(
                f"\n[Kite Login Required]\n"
                f"1. Open this URL in a browser: {url}\n"
                f"2. After login, copy 'request_token' from the redirect URL\n"
                f"3. Run: KiteAdapter.generate_session(request_token, api_secret)\n"
                f"4. Save the returned access_token to config.yaml\n"
            )
        # Verify by making a cheap call
        try:
            self.kite.profile()
        except Exception as e:
            raise RuntimeError(f"Kite access token invalid or expired: {e}")

    def generate_session(self, request_token: str) -> str:
        """One-time per day: exchange request_token for access_token."""
        data = self.kite.generate_session(request_token,
                                          api_secret=self.api_secret)
        self.access_token = data["access_token"]
        self.kite.set_access_token(self.access_token)
        return self.access_token

    # ----- Rate-limited call wrapper -----
    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_call_ts = time.time()

    # ----- Instruments -----
    def _load_instruments(self) -> pd.DataFrame:
        """
        Kite's instrument master is a CSV with ALL instruments. Cache it for
        the session — it's ~5MB and we don't want to fetch repeatedly.
        Schema columns of interest:
          instrument_token, exchange_token, tradingsymbol, name, last_price,
          expiry, strike, tick_size, lot_size, instrument_type, segment, exchange
        """
        if self._instrument_cache is not None:
            return self._instrument_cache
        self._throttle()
        instruments = self.kite.instruments(exchange="NFO")  # F&O segment
        df = pd.DataFrame(instruments)
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
        self._instrument_cache = df
        log.info(f"Loaded {len(df)} NFO instruments")
        return df

    def list_option_contracts(self, underlying: str,
                              as_of: date,
                              max_dte_days: int = 60) -> List[OptionContract]:
        """
        Return option contracts plausibly tradeable on `as_of`.
        - expiry >= as_of (not yet expired)
        - expiry <= as_of + max_dte_days (heuristic listing-window filter)
        """
        from datetime import timedelta as _td
        df = self._load_instruments()
        max_exp = as_of + _td(days=max_dte_days)
        mask = (
            (df["name"] == underlying) &
            (df["instrument_type"].isin(["CE", "PE"])) &
            (df["expiry"].notna()) &
            (df["expiry"] >= as_of) &
            (df["expiry"] <= max_exp)
        )
        sub = df.loc[mask]
        return [
            OptionContract(
                underlying=row["name"],
                expiry=row["expiry"],
                strike=int(row["strike"]),
                is_call=(row["instrument_type"] == "CE"),
                instrument_token=int(row["instrument_token"]),
                tradingsymbol=row["tradingsymbol"],
            )
            for _, row in sub.iterrows()
        ]

    # ----- Historical bars -----
    def _resolve_kite_resolution(self, resolution: str) -> str:
        return {
            "1minute": "minute",
            "5minute": "5minute",
            "15minute": "15minute",
            "day": "day",
        }[resolution]

    def get_historical_bars(self, instrument_token: int,
                            from_dt: datetime, to_dt: datetime,
                            resolution: str) -> List[OHLCBar]:
        """
        Fetch with automatic chunking (Kite caps minute requests at ~60 days).
        We always pass continuous=False; corp actions on options are a
        non-issue (cash-settled), but for robustness add no adjustment.
        """
        kite_res = self._resolve_kite_resolution(resolution)
        chunk_days = (self.HIST_MINUTE_CHUNK_DAYS
                      if resolution in ("1minute", "5minute") else 365)
        all_bars: List[OHLCBar] = []
        cursor = from_dt
        while cursor <= to_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), to_dt)
            self._throttle()
            try:
                raw = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=cursor,
                    to_date=chunk_end,
                    interval=kite_res,
                    oi=True,  # we want OI on options
                )
            except Exception as e:
                log.warning(f"Kite hist fetch failed for token "
                            f"{instrument_token} {cursor}-{chunk_end}: {e}")
                cursor = chunk_end + timedelta(days=1)
                continue
            for r in raw:
                all_bars.append(OHLCBar(
                    timestamp=pd.Timestamp(r["date"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=int(r["volume"]),
                    oi=int(r.get("oi") or 0) or None,
                ))
            cursor = chunk_end + timedelta(days=1)
        return all_bars

    def get_index_bars(self, symbol: str, from_dt: datetime, to_dt: datetime,
                       resolution: str) -> List[OHLCBar]:
        """
        For NSE indices like 'NIFTY 50' or 'INDIA VIX', the instrument lives
        in NSE segment, not NFO. Token lookup needed.
        """
        self._throttle()
        nse_inst = pd.DataFrame(self.kite.instruments(exchange="NSE"))
        match = nse_inst[nse_inst["tradingsymbol"] == symbol.replace(" ", "")]
        if match.empty:
            # Some indices use 'name' not 'tradingsymbol'
            match = nse_inst[nse_inst["name"] == symbol]
        if match.empty:
            raise ValueError(f"Index {symbol} not found in NSE instruments")
        token = int(match.iloc[0]["instrument_token"])
        return self.get_historical_bars(token, from_dt, to_dt, resolution)

    # ----- Diagnostics -----
    def get_rate_limit_remaining(self) -> dict:
        """Kite doesn't expose this directly; return our own counter."""
        return {"min_delay_remaining_s":
                max(0, self.RATE_LIMIT_DELAY -
                    (time.time() - self._last_call_ts))}
