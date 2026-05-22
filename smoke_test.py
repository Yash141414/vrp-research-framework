"""
smoke_test.py
-------------
End-to-end pipeline smoke test on SYNTHETIC data. Verifies the full chain
(spot/VIX/option ingestion -> health_check -> H4 backtest -> paper-trade
replay) runs without exception. Does NOT validate statistical correctness
- synthetic data has no real edge.

Use this BEFORE committing to a multi-hour real-data fetch, to make sure
the toolchain works end-to-end. If smoke passes, the only remaining
unknowns when you run on real data are signal/regime quality, not wiring.

Run:  python -m nifty_data_layer.smoke_test
"""
from __future__ import annotations
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from .brokers.base import OHLCBar, OptionContract
from .data_store import DataStore
from .trading_calendar import trading_days_from_spot
from .hypothesis_h4_intraday_buy import (
    H4Params, backtest as h4_backtest,
    report as h4_report, interpret as h4_interpret,
)
from .risk_gate import RiskGate, RiskConfig
from .paper_trader import PaperTrader, SimulatedFillEngine
from .health_check import report_spot, report_vix, report_options

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("smoke_test")

UNDERLYING = "NIFTY"
SPOT_SYMBOL = "NIFTY 50"
VIX_SYMBOL = "INDIA VIX"


# ----- Synthetic data generators -----

def _weekdays(start: date, n: int) -> list[date]:
    out, cur = [], start
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _gen_spot_bars(trading_days: list[date], seed: int = 42) -> list[OHLCBar]:
    """1-min spot bars with mean-reverting walk + occasional >0.3% gaps."""
    rng = np.random.default_rng(seed)
    bars: list[OHLCBar] = []
    spot = 22000.0
    anchor = 22000.0
    for d in trading_days:
        # ~40% of days have a >0.3% gap
        if rng.random() < 0.4:
            sign = 1 if rng.random() < 0.5 else -1
            gap = sign * rng.uniform(0.003, 0.007)
        else:
            gap = rng.uniform(-0.001, 0.001)
        cur = spot * (1 + gap)
        for m in range(375):  # 09:15..15:29
            ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15 + m)
            mr = (anchor - cur) * 0.0005   # mean-revert
            step = rng.normal(0, 5) + mr
            new_close = cur + step
            spread = abs(rng.normal(0, 2))
            high = max(cur, new_close) + spread
            low = min(cur, new_close) - spread
            bars.append(OHLCBar(
                timestamp=ts, open=cur, high=high, low=low, close=new_close,
                volume=1000 + int(abs(rng.normal(0, 100))), oi=None,
            ))
            cur = new_close
        spot = cur
    return bars


def _gen_vix_bars(trading_days: list[date], seed: int = 7) -> list[OHLCBar]:
    rng = np.random.default_rng(seed)
    return [
        OHLCBar(
            timestamp=pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=29),
            open=(v := float(rng.uniform(10.0, 22.0))),
            high=v, low=v, close=v, volume=0, oi=None,
        )
        for d in trading_days
    ]


def _nearest_thursday(d: date) -> date:
    offset = (3 - d.weekday()) % 7
    if offset == 0:
        offset = 7
    return d + timedelta(days=offset)


def _gen_option_bars(strike: int, is_call: bool, expiry: date,
                     spot_idx: pd.DataFrame, active_days: list[date],
                     seed: int) -> list[OHLCBar]:
    """Intrinsic + simple time-decay value + noise. Shape-realistic, not
    Black-Scholes — fine for smoke testing the pipeline."""
    rng = np.random.default_rng(seed)
    bars: list[OHLCBar] = []
    for d in active_days:
        dte = (expiry - d).days
        if dte < 0:
            continue
        day_start = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
        day_end = pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=30)
        day_spot = spot_idx.loc[(spot_idx.index >= day_start) &
                                 (spot_idx.index < day_end)]
        # Linear time-decay: tv ~ 5 + 15 * dte (so dte=0 has ~5, dte=7 ~110)
        for ts, row in day_spot.iterrows():
            S = float(row["close"])
            intrinsic = max(0.0, (S - strike) if is_call else (strike - S))
            tv = max(5.0, 5.0 + 15.0 * dte)
            premium = intrinsic + tv + rng.normal(0, 3)
            premium = max(0.5, premium)
            spread = max(0.2, abs(rng.normal(0, 2)))
            high = premium + spread
            low = max(0.5, premium - spread)
            bars.append(OHLCBar(
                timestamp=ts, open=premium, high=high, low=low,
                close=premium, volume=500, oi=None,
            ))
    return bars


def populate_synthetic(data_dir: Path, n_trading_days: int = 35,
                        seed: int = 42) -> dict:
    """Write synthetic spot/VIX/options into `data_dir`. Returns metadata."""
    start = date(2025, 1, 6)  # Monday
    trading_days = _weekdays(start, n_trading_days)
    store = DataStore(data_dir)

    log.info(f"Generating spot bars for {len(trading_days)} trading days")
    spot_bars = _gen_spot_bars(trading_days, seed=seed)
    store.save_spot(SPOT_SYMBOL, spot_bars)

    log.info(f"Generating VIX bars")
    vix_bars = _gen_vix_bars(trading_days, seed=seed + 1)
    store.save_vix(VIX_SYMBOL, vix_bars)

    # Index spot by timestamp for option generation
    spot_df = pd.DataFrame([b.__dict__ for b in spot_bars])
    spot_idx = spot_df.set_index("timestamp").sort_index()

    # Weekly expiries that fall within the trading-day range
    expiries = sorted({_nearest_thursday(d) for d in trading_days
                       if _nearest_thursday(d) <= trading_days[-1]})

    # Wide strike range so ATM lookups don't miss when spot drifts
    strikes = list(range(21500, 22501, 50))

    log.info(f"Generating option bars for {len(expiries)} expiries x "
             f"{len(strikes)} strikes x 2 legs")
    for i, exp in enumerate(expiries):
        active = [d for d in trading_days
                  if exp - timedelta(days=7) <= d <= exp]
        for strike in strikes:
            for is_call in (True, False):
                bars = _gen_option_bars(strike, is_call, exp, spot_idx, active,
                                         seed=seed + i * 1000 + strike +
                                         (1 if is_call else 0))
                if not bars:
                    continue
                contract = OptionContract(
                    underlying=UNDERLYING, expiry=exp, strike=strike,
                    is_call=is_call, instrument_token=0,
                    tradingsymbol=(f"{UNDERLYING}{exp:%y%b%d}{strike}"
                                   f"{'CE' if is_call else 'PE'}"),
                )
                store.save_option(contract, bars)

    return {
        "trading_days": trading_days,
        "expiries": expiries,
        "strikes": strikes,
    }


# ----- Smoke runner -----

def run_smoke(temp_data_dir: Path) -> dict:
    """Execute the full pipeline against synthetic data. Returns a dict of
    summary metrics; raises if any stage fails."""
    log.info(f"Synthetic data dir: {temp_data_dir}")
    meta = populate_synthetic(temp_data_dir)
    log.info(f"Populated {len(meta['trading_days'])} trading days, "
             f"{len(meta['expiries'])} expiries, "
             f"{len(meta['strikes'])} strikes")

    store = DataStore(temp_data_dir)
    spot_df = store.load_spot(SPOT_SYMBOL)

    # --- Stage A: health check ---
    log.info("Stage A: health_check report")
    cal = trading_days_from_spot(spot_df)
    spot_rep = report_spot(store, SPOT_SYMBOL)
    vix_rep = report_vix(store, VIX_SYMBOL, cal)
    opt_rep = report_options(store, UNDERLYING, strikes_around=5)
    log.info(f"  spot: {spot_rep['trading_days_observed']} trading days")
    log.info(f"  vix:  {vix_rep['trading_days_observed']} observations")
    log.info(f"  opts: {len(opt_rep)} expiries")
    assert spot_rep["trading_days_observed"] > 0
    assert vix_rep["trading_days_observed"] > 0
    assert len(opt_rep) > 0

    # --- Stage B: H4 backtest, both signals, default TP ---
    log.info("Stage B: H4 backtest (h4a + h4b, TP=1500)")
    h4_results = {}
    for signal in ("h4a", "h4b"):
        params = H4Params(tp_inr=1500.0)
        trades = h4_backtest(store, spot_df, UNDERLYING, signal, params)
        stats = h4_report(trades, params)
        print(h4_interpret(stats, signal, params.tp_inr))
        h4_results[signal] = {
            "n_trades": stats.get("n_trades", 0),
            "mean_net_pnl_inr": stats.get("mean_net_pnl_inr", 0.0),
        }

    # --- Stage C: paper-trade replay (h4a only, smoke purposes) ---
    log.info("Stage C: paper_trader.run_replay (h4a, TP=1500)")
    run_dir = temp_data_dir / "paper" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    gate = RiskGate(
        config=RiskConfig(),
        state_path=run_dir / "risk_state.json",
        audit_path=run_dir / "risk_audit.jsonl",
    )
    gate.load()
    trader = PaperTrader(
        store=store, spot=spot_df, underlying=UNDERLYING,
        params=H4Params(tp_inr=1500.0), signal="h4a",
        risk_gate=gate, fill_engine=SimulatedFillEngine(),
        trades_path=run_dir / "trades.parquet",
    )
    trader.run_replay()
    snap = gate.snapshot()
    log.info(f"  paper trades closed: {snap['closed_trades_total']}")
    log.info(f"  cumulative pnl:     Rs {snap['cumulative_pnl_inr']:.2f}")
    log.info(f"  halted:             {snap['halted']}"
             + (f" ({snap['halt_reason']})" if snap['halted'] else ""))

    print()
    print("=" * 60)
    print("SMOKE OK — full pipeline ran end-to-end on synthetic data.")
    print("=" * 60)

    return {
        "trading_days": len(meta["trading_days"]),
        "expiries": len(meta["expiries"]),
        "spot_obs": spot_rep["trading_days_observed"],
        "vix_obs": vix_rep["trading_days_observed"],
        "h4_results": h4_results,
        "paper_trades": snap["closed_trades_total"],
        "paper_pnl": snap["cumulative_pnl_inr"],
        "paper_halted": snap["halted"],
    }


def main():
    with tempfile.TemporaryDirectory(prefix="nifty_smoke_") as tmp:
        run_smoke(Path(tmp))


if __name__ == "__main__":
    main()
