"""Tests for the H4 signal evaluators and trade simulator."""
from __future__ import annotations
from datetime import date, time, timedelta
import numpy as np
import pandas as pd
import pytest

from hypothesis_h4_intraday_buy import (
    H4Params, evaluate_h4a, evaluate_h4b, simulate_trade,
    _nearest_weekly_expiry,
)


def _make_spot_minutes(base_date: date, opens: list[tuple[time, float]],
                       n_bars_per_open: int = 1) -> pd.DataFrame:
    """Build a spot 1-min dataframe. `opens` is a list of (start_time, value)
    pairs; from each, n_bars_per_open consecutive 1-min bars are emitted with
    flat OHLC = value."""
    rows = []
    for t, v in opens:
        ts0 = pd.Timestamp(base_date) + pd.Timedelta(hours=t.hour,
                                                       minutes=t.minute)
        for i in range(n_bars_per_open):
            rows.append({
                "timestamp": ts0 + pd.Timedelta(minutes=i),
                "open": v, "high": v, "low": v, "close": v,
                "volume": 1000,
            })
    return pd.DataFrame(rows)


def _spot_with_full_day(base_date: date, opens: list[tuple[time, float]],
                         pad_full_day: bool = True) -> pd.DataFrame:
    """Same as _make_spot_minutes but pads the day with ~300 bars of flat data
    so trading_days_from_spot would recognize it. Not used by H4 evaluators
    directly, but useful when also testing calendar interaction."""
    df = _make_spot_minutes(base_date, opens)
    return df


# ----- H4a: gap continuation -----

def test_h4a_gap_up_fires_long_ce():
    base = date(2026, 5, 18)  # Monday
    prev = date(2026, 5, 15)  # Friday
    spot = pd.concat([
        # Prev day close at 15:29
        _make_spot_minutes(prev, [(time(15, 29), 22000.0)]),
        # Today open at 9:15 with +0.5% gap
        _make_spot_minutes(base, [(time(9, 15), 22110.0)]),
        # Entry-time bar
        _make_spot_minutes(base, [(time(9, 20), 22120.0)]),
    ], ignore_index=True)
    spot_idx = spot.set_index("timestamp").sort_index()
    params = H4Params()
    sig = evaluate_h4a(spot_idx, pd.Timestamp(base), params)
    assert sig is not None
    direction, atm, entry_spot, entry_ts = sig
    assert direction == "long_ce"
    assert atm == 22100   # 22120 rounded to nearest 50
    assert entry_ts.time() == time(9, 20)


def test_h4a_gap_down_fires_long_pe():
    base = date(2026, 5, 18)
    prev = date(2026, 5, 15)
    spot = pd.concat([
        _make_spot_minutes(prev, [(time(15, 29), 22000.0)]),
        _make_spot_minutes(base, [(time(9, 15), 21890.0)]),    # -0.5% gap
        _make_spot_minutes(base, [(time(9, 20), 21880.0)]),
    ], ignore_index=True)
    spot_idx = spot.set_index("timestamp").sort_index()
    sig = evaluate_h4a(spot_idx, pd.Timestamp(base), H4Params())
    assert sig is not None
    assert sig[0] == "long_pe"


def test_h4a_no_gap_returns_none():
    base = date(2026, 5, 18)
    prev = date(2026, 5, 15)
    spot = pd.concat([
        _make_spot_minutes(prev, [(time(15, 29), 22000.0)]),
        # gap of +0.1% — under 0.3% threshold
        _make_spot_minutes(base, [(time(9, 15), 22022.0)]),
        _make_spot_minutes(base, [(time(9, 20), 22025.0)]),
    ], ignore_index=True)
    spot_idx = spot.set_index("timestamp").sort_index()
    assert evaluate_h4a(spot_idx, pd.Timestamp(base), H4Params()) is None


def test_h4a_no_prev_close_returns_none():
    base = date(2026, 5, 18)
    spot = _make_spot_minutes(base, [(time(9, 15), 22100.0),
                                       (time(9, 20), 22100.0)])
    spot_idx = spot.set_index("timestamp").sort_index()
    assert evaluate_h4a(spot_idx, pd.Timestamp(base), H4Params()) is None


# ----- H4b: opening-range breakout -----

def _orb_window_spot(base_date: date, range_high: float, range_low: float,
                      entry_price: float) -> pd.DataFrame:
    """Synthesize a 09:15-09:30 ORB window where high/low cover the band,
    plus an entry bar at 09:31 at entry_price."""
    rows = []
    # 15 minutes of bars in 09:15-09:29 inclusive
    for m in range(15):
        ts = pd.Timestamp(base_date) + pd.Timedelta(hours=9, minutes=15 + m)
        # Alternate so high/low hit the band
        if m % 2 == 0:
            o = h = (range_high + range_low) / 2
            l = c = range_low
            h = range_high
        else:
            o = c = (range_high + range_low) / 2
            l = range_low
            h = range_high
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l,
                     "close": c, "volume": 1000})
    # Entry bar at 09:31
    rows.append({
        "timestamp": pd.Timestamp(base_date) + pd.Timedelta(hours=9, minutes=31),
        "open": entry_price, "high": entry_price, "low": entry_price,
        "close": entry_price, "volume": 1000,
    })
    return pd.DataFrame(rows)


def test_h4b_break_above_fires_long_ce():
    base = date(2026, 5, 18)
    spot = _orb_window_spot(base, range_high=22050, range_low=21950,
                             entry_price=22075)
    spot_idx = spot.set_index("timestamp").sort_index()
    sig = evaluate_h4b(spot_idx, pd.Timestamp(base), H4Params())
    assert sig is not None
    assert sig[0] == "long_ce"
    # 22075 rounds to 22100
    assert sig[1] == 22100


def test_h4b_break_below_fires_long_pe():
    base = date(2026, 5, 18)
    spot = _orb_window_spot(base, range_high=22050, range_low=21950,
                             entry_price=21925)
    spot_idx = spot.set_index("timestamp").sort_index()
    sig = evaluate_h4b(spot_idx, pd.Timestamp(base), H4Params())
    assert sig is not None
    assert sig[0] == "long_pe"


def test_h4b_inside_range_returns_none():
    base = date(2026, 5, 18)
    spot = _orb_window_spot(base, range_high=22050, range_low=21950,
                             entry_price=22000)
    spot_idx = spot.set_index("timestamp").sort_index()
    assert evaluate_h4b(spot_idx, pd.Timestamp(base), H4Params()) is None


def test_h4b_insufficient_window_returns_none():
    base = date(2026, 5, 18)
    # Only 5 bars in the window — under 10-bar minimum
    rows = []
    for m in range(5):
        ts = pd.Timestamp(base) + pd.Timedelta(hours=9, minutes=15 + m)
        rows.append({"timestamp": ts, "open": 22000.0, "high": 22050.0,
                     "low": 21950.0, "close": 22000.0, "volume": 100})
    rows.append({"timestamp": pd.Timestamp(base) +
                 pd.Timedelta(hours=9, minutes=31),
                 "open": 22075.0, "high": 22075.0, "low": 22075.0,
                 "close": 22075.0, "volume": 100})
    spot = pd.DataFrame(rows)
    spot_idx = spot.set_index("timestamp").sort_index()
    assert evaluate_h4b(spot_idx, pd.Timestamp(base), H4Params()) is None


# ----- simulate_trade -----

def _option_bars(base_date: date, entry_time: time,
                 prices: list[float]) -> pd.DataFrame:
    """Build a sequence of 1-min option bars starting at entry_time.
    `prices` is the close price per minute; OHL = close for simplicity."""
    rows = []
    ts0 = pd.Timestamp(base_date) + pd.Timedelta(
        hours=entry_time.hour, minutes=entry_time.minute)
    for i, p in enumerate(prices):
        rows.append({
            "timestamp": ts0 + pd.Timedelta(minutes=i),
            "open": p, "high": p, "low": p, "close": p,
            "volume": 1000,
        })
    return pd.DataFrame(rows)


def test_simulate_tp_hit():
    base = date(2026, 5, 18)
    entry_t = time(9, 20)
    # Entry at 100, rises to 121 (TP at +20pts on Rs1500/75 = 20pts)
    prices = [100.0] + [105.0, 110.0, 121.0, 100.0]
    opt = _option_bars(base, entry_t, prices)
    params = H4Params()
    ts = pd.Timestamp(base) + pd.Timedelta(hours=9, minutes=20)
    outcome = simulate_trade(opt, ts, params,
                              signal_name="h4a", trade_date=base,
                              direction="long_ce", expiry=base + timedelta(days=3),
                              atm_strike=22000)
    assert outcome is not None
    assert outcome.exit_reason == "tp"
    # TP price = 100 + 1500/75 = 120
    assert outcome.exit_premium == pytest.approx(120.0)
    assert outcome.gross_pnl_inr == pytest.approx((120.0 - 100.0) * 75)


def test_simulate_sl_hit():
    base = date(2026, 5, 18)
    entry_t = time(9, 20)
    # Entry at 100, drops to 80 (SL at -1000/75 = -13.33pts -> 86.67)
    prices = [100.0] + [95.0, 90.0, 80.0]
    opt = _option_bars(base, entry_t, prices)
    params = H4Params()
    ts = pd.Timestamp(base) + pd.Timedelta(hours=9, minutes=20)
    outcome = simulate_trade(opt, ts, params,
                              signal_name="h4a", trade_date=base,
                              direction="long_ce", expiry=base + timedelta(days=3),
                              atm_strike=22000)
    assert outcome is not None
    assert outcome.exit_reason == "sl"
    # SL price = 100 - 1000/75 ~= 86.667
    assert outcome.exit_premium == pytest.approx(100.0 - 1000.0/75, abs=1e-9)


def test_simulate_time_stop():
    """Premium drifts sideways under TP, exit at 15:15."""
    base = date(2026, 5, 18)
    entry_t = time(9, 20)
    # 356 bars from 09:20 -> ~15:16. Premium stays at 100.
    # Add some early-morning bars > scratch threshold so scratch doesn't fire.
    n = 356
    # Mix prices so max_unrealized_by_scratch >= scratch_pts (500/75 ~6.67)
    # Insert bars with high enough early on:
    prices = [100.0]  # entry bar
    # Then morning bars hovering with brief +7 spike to defeat scratch
    for i in range(n):
        prices.append(107.5 if i < 5 else 100.0)
    opt = _option_bars(base, entry_t, prices)
    params = H4Params()
    ts = pd.Timestamp(base) + pd.Timedelta(hours=9, minutes=20)
    outcome = simulate_trade(opt, ts, params,
                              signal_name="h4a", trade_date=base,
                              direction="long_ce", expiry=base + timedelta(days=3),
                              atm_strike=22000)
    assert outcome is not None
    assert outcome.exit_reason == "time_stop"


def test_simulate_scratch_rule():
    """No meaningful move by 11:30 -> exit at 11:30 bar close."""
    base = date(2026, 5, 18)
    entry_t = time(9, 20)
    # ~130 bars from 09:20 -> 11:30. Premium hovers at 100, never reaches
    # entry + scratch_pts (~106.67).
    n = 130
    prices = [100.0] + [100.5] * n
    opt = _option_bars(base, entry_t, prices)
    params = H4Params()
    ts = pd.Timestamp(base) + pd.Timedelta(hours=9, minutes=20)
    outcome = simulate_trade(opt, ts, params,
                              signal_name="h4a", trade_date=base,
                              direction="long_ce", expiry=base + timedelta(days=3),
                              atm_strike=22000)
    assert outcome is not None
    assert outcome.exit_reason == "scratch"


# ----- _nearest_weekly_expiry -----

def test_nearest_expiry_includes_today_by_default():
    expiries = [date(2026, 5, 14), date(2026, 5, 21), date(2026, 5, 28)]
    # On the 21st (expiry), default include=True picks 21st
    got = _nearest_weekly_expiry(expiries, date(2026, 5, 21),
                                  include_expiry_day=True)
    assert got == date(2026, 5, 21)


def test_nearest_expiry_excludes_today_when_flag_off():
    expiries = [date(2026, 5, 14), date(2026, 5, 21), date(2026, 5, 28)]
    got = _nearest_weekly_expiry(expiries, date(2026, 5, 21),
                                  include_expiry_day=False)
    assert got == date(2026, 5, 28)


def test_nearest_expiry_none_when_no_future():
    expiries = [date(2026, 5, 7), date(2026, 5, 14)]
    assert _nearest_weekly_expiry(expiries, date(2026, 5, 21),
                                   include_expiry_day=True) is None
