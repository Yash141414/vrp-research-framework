"""Tests for the realized-variance math in hypothesis_h1_vrp."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

from Apology.Proj.Nifty_momentum_system.nifty_data_layer.hypothesis_h1_vrp import realized_variance_window


def _make_constant_log_return_spot(start: str,
                                    n_bars: int,
                                    bar_freq: str = "5min",
                                    log_ret_per_bar: float = 0.001,
                                    s0: float = 100.0) -> pd.DataFrame:
    """Build a 5-min spot series where every bar has the same log-return."""
    idx = pd.date_range(start=start, periods=n_bars, freq=bar_freq)
    closes = s0 * np.exp(np.arange(n_bars) * log_ret_per_bar)
    return pd.DataFrame({"close": closes}, index=idx)


def test_rv_constant_log_returns_matches_closed_form():
    """For constant per-bar log return r over N bars in a `days`-day window,
    sum of squared returns = (N-1) * r^2; annualized factor = 365/days."""
    days = 30
    bar_freq = "5min"
    # 5-min bars in 30 calendar days = 30 * 24 * 12 = 8640
    n_bars = days * 24 * 12
    r = 0.001
    spot_5min = _make_constant_log_return_spot(
        "2026-01-01", n_bars=n_bars, bar_freq=bar_freq,
        log_ret_per_bar=r, s0=100.0,
    )

    start = pd.Timestamp("2026-01-01")
    rv = realized_variance_window(spot_5min, start, days=days)

    expected_rv_period = (n_bars - 1) * (r ** 2)
    expected_rv_annualized = expected_rv_period * (365.0 / days)
    assert rv == pytest.approx(expected_rv_annualized, rel=1e-9)


def test_rv_zero_returns_is_zero():
    days = 30
    n_bars = days * 24 * 12
    spot_5min = _make_constant_log_return_spot(
        "2026-01-01", n_bars=n_bars, log_ret_per_bar=0.0,
    )
    rv = realized_variance_window(spot_5min, pd.Timestamp("2026-01-01"),
                                   days=days)
    assert rv == pytest.approx(0.0)


def test_rv_insufficient_bars_returns_nan():
    # Only 50 bars in window — function requires >= 100
    spot_5min = _make_constant_log_return_spot(
        "2026-01-01", n_bars=50, log_ret_per_bar=0.001,
    )
    rv = realized_variance_window(spot_5min, pd.Timestamp("2026-01-01"),
                                   days=30)
    assert math.isnan(rv)
