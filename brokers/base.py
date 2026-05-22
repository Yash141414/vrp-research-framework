"""
brokers/base.py
---------------
Abstract interface every broker adapter must implement.

All adapters return data in a STANDARDIZED schema so downstream code is
broker-agnostic. If the broker can't supply a field, it must return None
or NaN — never silently substitute.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional
import pandas as pd


@dataclass(frozen=True)
class OHLCBar:
    """Standard OHLC bar. timestamp is timezone-aware IST."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None  # open interest, options/futures only


@dataclass(frozen=True)
class OptionContract:
    """Identity of a single option contract."""
    underlying: str            # e.g. 'NIFTY'
    expiry: date
    strike: int
    is_call: bool              # True=CE, False=PE
    instrument_token: int      # broker-specific id
    tradingsymbol: str         # broker-specific symbol


class BrokerAdapter(ABC):
    """
    Each broker subclass implements these. NO method here may have
    look-ahead — historical fetches must respect (from_dt, to_dt) bounds
    exactly and never return data outside them.
    """

    # --- Auth ---
    @abstractmethod
    def login(self) -> None:
        """Establish session; raise on failure with broker-specific reason."""
        ...

    # --- Instruments ---
    @abstractmethod
    def list_option_contracts(self, underlying: str,
                              as_of: date,
                              max_dte_days: int = 60) -> List[OptionContract]:
        """
        Return option contracts for `underlying` that would have been
        tradeable on `as_of`.

        Filters applied:
          - expiry >= as_of           (contract not yet expired)
          - expiry <= as_of + max_dte_days
              (HEURISTIC: contract was already listed on as_of. NSE doesn't
              expose listed_date, but weeklies list ~2-3wks before expiry
              and monthlies ~3mo before. 60 days covers near-month + 2-3
              weeklies without admitting far-month look-ahead.)

        Subclasses must apply BOTH filters. Tightening max_dte_days makes
        the filter more conservative (fewer false-positive contracts).
        """
        ...

    # --- Historical bars ---
    @abstractmethod
    def get_historical_bars(self, instrument_token: int,
                            from_dt: datetime, to_dt: datetime,
                            resolution: str) -> List[OHLCBar]:
        """
        Fetch OHLC bars for one instrument in [from_dt, to_dt].
        resolution: '1minute', '5minute', '15minute', 'day'
        """
        ...

    @abstractmethod
    def get_index_bars(self, symbol: str, from_dt: datetime, to_dt: datetime,
                       resolution: str) -> List[OHLCBar]:
        """Fetch index (spot) bars: NIFTY 50, INDIA VIX, etc."""
        ...

    # --- Diagnostics ---
    @abstractmethod
    def get_rate_limit_remaining(self) -> dict:
        """Return current rate-limit status — important to avoid 429s."""
        ...
