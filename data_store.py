"""
data_store.py
-------------
Local cache for fetched market data. Uses Parquet (faster, smaller,
type-safe) partitioned by symbol and month.

WHY THIS MATTERS: a single re-fetch of 3 years of 1-minute Nifty options
can take HOURS due to rate limits. Once fetched, never re-fetch.

Schema is validated on every write. Bad data = caught early, not at
backtest time.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date
import pandas as pd

from .brokers.base import OHLCBar, OptionContract

log = logging.getLogger(__name__)


# Required columns and dtypes for any cached bars table
BAR_SCHEMA = {
    "timestamp": "datetime64[ns]",  # IST naive; we standardize to IST
    "open":      "float64",
    "high":      "float64",
    "low":       "float64",
    "close":     "float64",
    "volume":    "int64",
    "oi":        "Int64",   # nullable int
}


def bars_to_df(bars: List[OHLCBar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=BAR_SCHEMA.keys()).astype(BAR_SCHEMA)
    df = pd.DataFrame([b.__dict__ for b in bars])
    # Sanity check
    assert df["timestamp"].is_monotonic_increasing or df["timestamp"].is_unique, \
        "Bars must be unique-timestamped or sorted ascending"
    assert (df["high"] >= df["low"]).all(), "high < low detected — bad data"
    assert (df["volume"] >= 0).all(), "negative volume detected"
    return df.astype(BAR_SCHEMA, errors="ignore")


class DataStore:
    """
    Layout under data_dir:
        data_dir/
          spot/{symbol}.parquet         e.g. NIFTY50.parquet
          vix/{symbol}.parquet
          options/{underlying}/{expiry_yyyy_mm_dd}/{strike}_{CE|PE}.parquet
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "spot").mkdir(parents=True, exist_ok=True)
        (self.root / "vix").mkdir(parents=True, exist_ok=True)
        (self.root / "options").mkdir(parents=True, exist_ok=True)

    # ----- Spot / index -----
    def _spot_path(self, symbol: str) -> Path:
        clean = symbol.replace(" ", "_").upper()
        return self.root / "spot" / f"{clean}.parquet"

    def save_spot(self, symbol: str, bars: List[OHLCBar]) -> None:
        df = bars_to_df(bars)
        path = self._spot_path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset="timestamp").sort_values("timestamp")
        df.to_parquet(path, index=False)
        log.info(f"Saved {len(df)} bars to {path}")

    def load_spot(self, symbol: str) -> pd.DataFrame:
        path = self._spot_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=BAR_SCHEMA.keys()).astype(BAR_SCHEMA)
        return pd.read_parquet(path)

    # ----- VIX -----
    def _vix_path(self, symbol: str) -> Path:
        clean = symbol.replace(" ", "_").upper()
        return self.root / "vix" / f"{clean}.parquet"

    def save_vix(self, symbol: str, bars: List[OHLCBar]) -> None:
        df = bars_to_df(bars)
        path = self._vix_path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset="timestamp").sort_values("timestamp")
        df.to_parquet(path, index=False)
        log.info(f"Saved {len(df)} VIX bars to {path}")

    def load_vix(self, symbol: str) -> pd.DataFrame:
        path = self._vix_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=BAR_SCHEMA.keys()).astype(BAR_SCHEMA)
        return pd.read_parquet(path)

    # ----- Options -----
    def _option_path(self, contract: OptionContract) -> Path:
        exp = contract.expiry.strftime("%Y_%m_%d")
        leg = "CE" if contract.is_call else "PE"
        return (self.root / "options" / contract.underlying / exp /
                f"{contract.strike}_{leg}.parquet")

    def save_option(self, contract: OptionContract,
                    bars: List[OHLCBar]) -> None:
        df = bars_to_df(bars)
        if df.empty:
            return
        path = self._option_path(contract)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset="timestamp").sort_values("timestamp")
        df.to_parquet(path, index=False)

    def load_option(self, underlying: str, expiry: date, strike: int,
                    is_call: bool) -> pd.DataFrame:
        leg = "CE" if is_call else "PE"
        path = (self.root / "options" / underlying /
                expiry.strftime("%Y_%m_%d") / f"{strike}_{leg}.parquet")
        if not path.exists():
            return pd.DataFrame(columns=BAR_SCHEMA.keys()).astype(BAR_SCHEMA)
        return pd.read_parquet(path)

    def list_cached_expiries(self, underlying: str) -> List[date]:
        d = self.root / "options" / underlying
        if not d.exists():
            return []
        out = []
        for child in d.iterdir():
            try:
                out.append(datetime.strptime(child.name, "%Y_%m_%d").date())
            except ValueError:
                continue
        return sorted(out)

    def list_cached_strikes(self, underlying: str,
                             expiry: date) -> List[tuple]:
        """Returns list of (strike, is_call) tuples cached for given expiry."""
        d = (self.root / "options" / underlying /
             expiry.strftime("%Y_%m_%d"))
        if not d.exists():
            return []
        out = []
        for f in d.glob("*.parquet"):
            stem = f.stem  # e.g. "18000_CE"
            try:
                strike, leg = stem.rsplit("_", 1)
                out.append((int(strike), leg == "CE"))
            except ValueError:
                continue
        return sorted(out)
