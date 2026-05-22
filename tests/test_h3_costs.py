"""Tests for the cost model in hypothesis_h3_costs.

Run with:  python -m pytest nifty_data_layer/tests
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from Apology.Proj.Nifty_momentum_system.nifty_data_layer.hypothesis_h3_costs import (
    leg_cost,
    round_trip_short_straddle_cost,
    compute_net_pnl,
    _resolve_lot_size,
    BROKERAGE_PER_ORDER,
    STT_SELL_PREM_PCT,
    EXCH_TXN_PCT,
    SEBI_PCT,
    GST_PCT,
    STAMP_BUY_PCT,
    SLIPPAGE_PCT,
)


def _expected_leg_cost(premium: float, qty: int, is_buy: bool) -> float:
    notional = premium * qty
    brok = BROKERAGE_PER_ORDER
    exch = notional * EXCH_TXN_PCT
    sebi = notional * SEBI_PCT
    stt = 0.0 if is_buy else notional * STT_SELL_PREM_PCT
    stamp = notional * STAMP_BUY_PCT if is_buy else 0.0
    gst = (brok + exch + sebi) * GST_PCT
    slip = notional * SLIPPAGE_PCT
    return brok + exch + sebi + stt + stamp + gst + slip


def test_leg_cost_sell_breakdown():
    # Sell side incurs STT, no stamp duty.
    got = leg_cost(premium=100.0, qty=75, is_buy=False)
    assert got == pytest.approx(_expected_leg_cost(100.0, 75, False))


def test_leg_cost_buy_breakdown():
    # Buy side incurs stamp duty, no STT.
    got = leg_cost(premium=100.0, qty=75, is_buy=True)
    assert got == pytest.approx(_expected_leg_cost(100.0, 75, True))


def test_leg_cost_sell_vs_buy_has_stt_swap():
    # The difference between sell and buy (same premium) should equal
    # STT (on sell) minus stamp duty (on buy).
    sell = leg_cost(100.0, 75, is_buy=False)
    buy = leg_cost(100.0, 75, is_buy=True)
    notional = 100.0 * 75
    delta = sell - buy
    expected = notional * (STT_SELL_PREM_PCT - STAMP_BUY_PCT)
    assert delta == pytest.approx(expected)


def test_round_trip_short_straddle_cost_sums_four_legs():
    ce_in, pe_in = 120.0, 100.0
    ce_out, pe_out = 60.0, 50.0
    qty = 75
    got = round_trip_short_straddle_cost(ce_in, pe_in, ce_out, pe_out, qty)
    expected = (
        _expected_leg_cost(ce_in, qty, False) +
        _expected_leg_cost(pe_in, qty, False) +
        _expected_leg_cost(ce_out, qty, True) +
        _expected_leg_cost(pe_out, qty, True)
    )
    assert got == pytest.approx(expected)


def test_resolve_lot_size_missing_raises():
    with pytest.raises(ValueError, match="lot_sizes"):
        _resolve_lot_size({}, "NIFTY")
    with pytest.raises(ValueError, match="lot_sizes"):
        _resolve_lot_size({"lot_sizes": {"BANKNIFTY": 30}}, "NIFTY")


def test_resolve_lot_size_present_returns_int():
    assert _resolve_lot_size({"lot_sizes": {"NIFTY": 75}}, "NIFTY") == 75


def _minimal_h2_row(overnight_pnl=10.0, intraday_pnl=-5.0,
                    ce_in=120.0, pe_in=100.0,
                    ce_out=60.0, pe_out=50.0) -> dict:
    return {
        "expiry": pd.Timestamp("2026-05-22").date(),
        "trade_date": pd.Timestamp("2026-05-19").date(),
        "dte": 3,
        "vix": 14.5,
        "overnight_atm": 22000,
        "intraday_atm": 22000,
        "ce_overnight_entry": ce_in,
        "pe_overnight_entry": pe_in,
        "ce_overnight_exit": ce_out,
        "pe_overnight_exit": pe_out,
        "ce_intraday_entry": ce_out,
        "pe_intraday_entry": pe_out,
        "ce_intraday_exit": ce_in,
        "pe_intraday_exit": pe_in,
        "overnight_pnl": overnight_pnl,
        "intraday_pnl": intraday_pnl,
    }


def test_compute_net_pnl_subtracts_real_cost():
    lot = 75
    overnight_pnl = 10.0  # premium points
    df = pd.DataFrame([_minimal_h2_row(overnight_pnl=overnight_pnl)])
    out, mean_cost = compute_net_pnl(df, lot_size=lot, qty_lots=1)

    qty = lot * 1
    gross = overnight_pnl * qty
    expected_cost = round_trip_short_straddle_cost(120.0, 100.0, 60.0, 50.0, qty)
    assert out["gross_overnight_pnl_rupees"].iloc[0] == pytest.approx(gross)
    assert out["overnight_cost_inr"].iloc[0] == pytest.approx(expected_cost)
    assert out["net_overnight_pnl"].iloc[0] == pytest.approx(gross - expected_cost)
    assert mean_cost > 0


def test_compute_net_pnl_missing_columns_raises():
    df = pd.DataFrame([{"overnight_pnl": 1.0, "intraday_pnl": 1.0}])
    with pytest.raises(ValueError, match="missing columns"):
        compute_net_pnl(df, lot_size=75, qty_lots=1)


def test_compute_net_pnl_nan_premium_row_gets_nan_cost():
    row = _minimal_h2_row()
    row["ce_overnight_entry"] = np.nan
    df = pd.DataFrame([row])
    out, _ = compute_net_pnl(df, lot_size=75, qty_lots=1)
    assert np.isnan(out["overnight_cost_inr"].iloc[0])
    assert np.isnan(out["net_overnight_pnl"].iloc[0])
