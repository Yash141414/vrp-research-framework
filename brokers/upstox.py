"""
brokers/upstox.py
-----------------
Upstox V2 API adapter. Free historical data is a notable advantage.

Requires: pip install upstox-python-sdk

GOTCHAS:
1. Access tokens last ~1 day. OAuth2 flow needed daily.
2. Historical 1-minute data for options goes back ~1 year.
3. Instrument keys use the format e.g. "NSE_FO|36702" — different from Kite.
4. Lot sizes and tick sizes are encoded in the instruments JSON; do not
   hardcode them.
"""
from __future__ import annotations
import time
import logging
import requests
from datetime import datetime, date, timedelta
from typing import List
import pandas as pd

from .base import BrokerAdapter, OHLCBar, OptionContract

log = logging.getLogger(__name__)


class UpstoxAdapter(BrokerAdapter):
    BASE = "https://api.upstox.com/v2"
    RATE_LIMIT_DELAY = 0.5  # conservative

    def __init__(self, api_key: str, api_secret: str,
                 redirect_uri: str, access_token: str | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self._last_call_ts = 0.0
        self._instruments: pd.DataFrame | None = None

    def _headers(self) -> dict:
        if not self.access_token:
            raise RuntimeError("No access token. Run login() flow first.")
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_call_ts = time.time()

    # ----- Auth -----
    def login(self) -> None:
        if not self.access_token:
            url = (f"https://api.upstox.com/v2/login/authorization/dialog?"
                   f"response_type=code&client_id={self.api_key}&"
                   f"redirect_uri={self.redirect_uri}")
            raise RuntimeError(
                f"\n[Upstox Login Required]\n"
                f"1. Open {url}\n"
                f"2. After login, copy 'code' from redirect\n"
                f"3. Run: UpstoxAdapter.exchange_code(code)\n"
            )
        # Verify
        r = requests.get(f"{self.BASE}/user/profile", headers=self._headers())
        if r.status_code != 200:
            raise RuntimeError(f"Upstox token invalid: {r.text}")

    def exchange_code(self, code: str) -> str:
        r = requests.post(f"{self.BASE}/login/authorization/token", data={
            "code": code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        })
        r.raise_for_status()
        self.access_token = r.json()["access_token"]
        return self.access_token

    # ----- Instruments -----
    def _load_instruments(self) -> pd.DataFrame:
        if self._instruments is not None:
            return self._instruments
        # Upstox publishes a daily JSON dump
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        df = pd.read_json(url, compression="gzip")
        df["expiry"] = pd.to_datetime(df.get("expiry"), errors="coerce").dt.date
        self._instruments = df
        return df

    def list_option_contracts(self, underlying: str,
                              as_of: date,
                              max_dte_days: int = 60) -> List[OptionContract]:
        from datetime import timedelta as _td
        df = self._load_instruments()
        max_exp = as_of + _td(days=max_dte_days)
        mask = (
            (df.get("asset_symbol") == underlying) &
            (df.get("instrument_type").isin(["CE", "PE"])) &
            (df["expiry"].notna()) &
            (df["expiry"] >= as_of) &
            (df["expiry"] <= max_exp)
        )
        sub = df.loc[mask]
        return [
            OptionContract(
                underlying=row["asset_symbol"],
                expiry=row["expiry"],
                strike=int(row["strike_price"]),
                is_call=(row["instrument_type"] == "CE"),
                instrument_token=row["instrument_key"],   # Upstox uses string keys
                tradingsymbol=row["trading_symbol"],
            )
            for _, row in sub.iterrows()
        ]

    # ----- Historical -----
    def get_historical_bars(self, instrument_token, from_dt: datetime,
                            to_dt: datetime, resolution: str) -> List[OHLCBar]:
        # Upstox interval format: '1minute', '30minute', 'day', 'week', 'month'
        interval = resolution if resolution != "1minute" else "1minute"
        all_bars = []
        cursor = from_dt
        while cursor <= to_dt:
            chunk_end = min(cursor + timedelta(days=30), to_dt)  # safe chunk
            self._throttle()
            url = (f"{self.BASE}/historical-candle/{instrument_token}/"
                   f"{interval}/{chunk_end.date()}/{cursor.date()}")
            r = requests.get(url, headers=self._headers())
            if r.status_code != 200:
                log.warning(f"Upstox hist fetch failed: {r.status_code} {r.text}")
                cursor = chunk_end + timedelta(days=1)
                continue
            data = r.json().get("data", {}).get("candles", [])
            for c in data:
                # candle = [timestamp, open, high, low, close, volume, oi]
                all_bars.append(OHLCBar(
                    timestamp=pd.Timestamp(c[0]),
                    open=float(c[1]), high=float(c[2]),
                    low=float(c[3]), close=float(c[4]),
                    volume=int(c[5]),
                    oi=int(c[6]) if len(c) > 6 else None,
                ))
            cursor = chunk_end + timedelta(days=1)
        return all_bars

    def get_index_bars(self, symbol: str, from_dt: datetime, to_dt: datetime,
                       resolution: str) -> List[OHLCBar]:
        # Upstox: NSE indices use NSE_INDEX| segment
        # "NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"
        token = f"NSE_INDEX|{symbol}"
        return self.get_historical_bars(token, from_dt, to_dt, resolution)

    def get_rate_limit_remaining(self) -> dict:
        return {"throttle_remaining_s":
                max(0, self.RATE_LIMIT_DELAY -
                    (time.time() - self._last_call_ts))}
