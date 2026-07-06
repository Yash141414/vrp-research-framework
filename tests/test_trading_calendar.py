"""Tests for the inferred trading-day calendar."""
from __future__ import annotations
from datetime import date
import pandas as pd
import pytest

from trading_calendar import (
    trading_days_from_spot,
    prev_trading_day,
    next_trading_day,
    missing_trading_days,
    FULL_DAY_MIN_BARS,
)


def _spot_with_days(days: list[date], bars_per_day: int) -> pd.DataFrame:
    rows = []
    for d in days:
        base = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
        for i in range(bars_per_day):
            rows.append({"timestamp": base + pd.Timedelta(minutes=i),
                         "close": 100.0 + i * 0.01})
    return pd.DataFrame(rows)


def test_trading_days_only_full_days_qualify():
    days = [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]
    df = _spot_with_days(days[:2], FULL_DAY_MIN_BARS + 10)
    df_partial = _spot_with_days([days[2]], 50)
    spot = pd.concat([df, df_partial], ignore_index=True)
    cal = trading_days_from_spot(spot)
    assert cal == days[:2]


def test_prev_and_next_trading_day():
    cal = [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 22)]
    # Wednesday between Tue and Fri (Wed/Thu missing)
    assert prev_trading_day(date(2026, 5, 21), cal) == date(2026, 5, 19)
    assert next_trading_day(date(2026, 5, 21), cal) == date(2026, 5, 22)
    # Boundary: returns None outside calendar
    assert prev_trading_day(date(2026, 5, 18), cal) is None
    assert next_trading_day(date(2026, 5, 22), cal) is None


def test_missing_trading_days_skips_weekends():
    # 2026-05-18 (Mon), 19 (Tue), 20 (Wed), 21 (Thu), 22 (Fri); 23/24 = weekend
    cal = [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 22)]
    missing = missing_trading_days(cal, date(2026, 5, 18), date(2026, 5, 24))
    # Weekdays Mon..Fri = 5 days; calendar has 3 of them; missing 2 (Wed, Thu)
    assert missing == [date(2026, 5, 20), date(2026, 5, 21)]


def test_missing_trading_days_empty_when_complete():
    cal = [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20),
           date(2026, 5, 21), date(2026, 5, 22)]
    missing = missing_trading_days(cal, date(2026, 5, 18), date(2026, 5, 22))
    assert missing == []


def test_empty_spot_returns_empty_calendar():
    assert trading_days_from_spot(pd.DataFrame()) == []
