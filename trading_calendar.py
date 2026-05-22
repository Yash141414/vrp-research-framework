"""
trading_calendar.py
-------------------
Trading-day calendar derived from cached spot data. We do NOT hardcode the
NSE holiday list — instead we infer "trading day" from observed intraday
bars. A day with >= `min_bars_per_day` 1-min bars is a full trading day;
fewer means a half-day (e.g. Muhurat Trading, partial sessions) or no
session at all.

Why infer rather than hardcode:
  - Removes a maintenance burden (NSE holiday list changes yearly).
  - Stays in sync with the data we actually fetched (if a trading day's
    data was missed for any reason, treating it as "not trading" prevents
    silently using stale or partial data downstream).
"""
from __future__ import annotations
from datetime import date
from typing import Optional
import pandas as pd


FULL_DAY_MIN_BARS = 300   # 09:15-15:30 1-min bars ~= 375; pad for missing bars


def trading_days_from_spot(spot_df: pd.DataFrame,
                           min_bars_per_day: int = FULL_DAY_MIN_BARS) -> list[date]:
    """Return sorted unique trading-day dates inferred from a spot dataframe."""
    if spot_df.empty:
        return []
    s = spot_df.copy()
    s["d"] = pd.to_datetime(s["timestamp"]).dt.date
    counts = s.groupby("d").size()
    return sorted(counts[counts >= min_bars_per_day].index.tolist())


def prev_trading_day(target: date, calendar: list[date]) -> Optional[date]:
    """Most recent trading day strictly before `target`. None if not found."""
    earlier = [d for d in calendar if d < target]
    return earlier[-1] if earlier else None


def next_trading_day(target: date, calendar: list[date]) -> Optional[date]:
    """Earliest trading day strictly after `target`. None if not found."""
    later = [d for d in calendar if d > target]
    return later[0] if later else None


def missing_trading_days(calendar: list[date],
                          start: date, end: date,
                          weekday_only: bool = True) -> list[date]:
    """
    Calendar weekdays in [start, end] NOT present in `calendar`. Useful for
    spotting fetch gaps. `weekday_only` skips Sat/Sun.
    """
    have = set(calendar)
    out = []
    cur = start
    while cur <= end:
        if weekday_only and cur.weekday() >= 5:
            cur = pd.Timestamp(cur).date() + pd.Timedelta(days=1).to_pytimedelta()
            continue
        if cur not in have:
            out.append(cur)
        cur = (pd.Timestamp(cur) + pd.Timedelta(days=1)).date()
    return out
