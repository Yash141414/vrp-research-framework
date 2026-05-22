"""
health_check.py
---------------
Pre-flight data sanity report. Run this BEFORE H1/H2/H3 to catch:
  - Missing trading days in spot/VIX
  - Sparse option coverage by expiry (which strikes are populated?)
  - Day-level bar counts (full vs partial trading days)
  - Misalignment between spot trading days and VIX trading days

A failing hypothesis test with bad input data tells you nothing. This
script gives you an unambiguous picture of what's in the cache before
any statistical work runs.

USAGE:
    python -m nifty_data_layer.health_check --config config.yaml
"""
from __future__ import annotations
import argparse
from datetime import date
import yaml
import pandas as pd

from .config_loader import load_config
from .data_store import DataStore
from .trading_calendar import (trading_days_from_spot, missing_trading_days,
                               FULL_DAY_MIN_BARS)


def _bar_counts_per_day(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return (df.assign(d=pd.to_datetime(df["timestamp"]).dt.date)
              .groupby("d").size())


def report_spot(store: DataStore, symbol: str) -> dict:
    df = store.load_spot(symbol)
    counts = _bar_counts_per_day(df)
    full_days = counts[counts >= FULL_DAY_MIN_BARS]
    return {
        "symbol": symbol,
        "rows": len(df),
        "first_day": counts.index.min() if len(counts) else None,
        "last_day": counts.index.max() if len(counts) else None,
        "trading_days_observed": len(full_days),
        "partial_days_observed": int((counts < FULL_DAY_MIN_BARS).sum()),
        "median_bars_per_full_day": int(full_days.median()) if len(full_days) else 0,
    }


def report_vix(store: DataStore, symbol: str,
               spot_calendar: list[date]) -> dict:
    df = store.load_vix(symbol)
    if df.empty:
        return {"symbol": symbol, "rows": 0, "trading_days_observed": 0,
                "missing_vs_spot": len(spot_calendar)}
    days = sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())
    spot_set = set(spot_calendar)
    return {
        "symbol": symbol,
        "rows": len(df),
        "first_day": days[0],
        "last_day": days[-1],
        "trading_days_observed": len(days),
        "missing_vs_spot": len([d for d in spot_calendar if d not in set(days)]),
    }


def report_options(store: DataStore, underlying: str,
                    strikes_around: int) -> list[dict]:
    out = []
    expiries = store.list_cached_expiries(underlying)
    for exp in expiries:
        strikes = store.list_cached_strikes(underlying, exp)
        n_ce = sum(1 for _, is_call in strikes if is_call)
        n_pe = sum(1 for _, is_call in strikes if not is_call)
        expected_legs = (2 * strikes_around + 1) * 2  # CE+PE per strike
        out.append({
            "expiry": exp,
            "ce_strikes": n_ce,
            "pe_strikes": n_pe,
            "total_legs": n_ce + n_pe,
            "expected_legs": expected_legs,
            "coverage_pct": round(100 * (n_ce + n_pe) / expected_legs, 1)
                            if expected_legs else 0.0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--strikes-around", type=int, default=5,
                    help="Match --strikes-around used in fetch.py")
    ap.add_argument("--start", help="YYYY-MM-DD: window start for gap analysis")
    ap.add_argument("--end", help="YYYY-MM-DD: window end for gap analysis")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"])

    spot_sym = cfg["universe"]["spot_symbol"]
    vix_sym = cfg["universe"]["vix_symbol"]
    underlying = cfg["universe"]["options_underlying"]

    print("=" * 70)
    print("DATA HEALTH CHECK")
    print("=" * 70)

    # Spot
    spot_df = store.load_spot(spot_sym)
    spot_rep = report_spot(store, spot_sym)
    print("\n[Spot]")
    for k, v in spot_rep.items():
        print(f"  {k:30s}: {v}")

    spot_cal = trading_days_from_spot(spot_df)

    # VIX
    vix_rep = report_vix(store, vix_sym, spot_cal)
    print("\n[VIX]")
    for k, v in vix_rep.items():
        print(f"  {k:30s}: {v}")
    if vix_rep.get("missing_vs_spot", 0) > 0:
        print(f"  WARN: {vix_rep['missing_vs_spot']} spot trading days have "
              f"no VIX observation.")

    # Window gap analysis (optional)
    if args.start and args.end:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end)
        missing = missing_trading_days(spot_cal, s, e)
        print(f"\n[Gap analysis] {args.start}..{args.end}")
        print(f"  weekdays missing from spot cache: {len(missing)}")
        if missing and len(missing) <= 25:
            print(f"  -> {', '.join(d.isoformat() for d in missing)}")
        elif missing:
            print(f"  -> first 25: "
                  f"{', '.join(d.isoformat() for d in missing[:25])}, ...")

    # Options
    print(f"\n[Options: {underlying}]")
    opt_rows = report_options(store, underlying, args.strikes_around)
    if not opt_rows:
        print("  (no expiries cached)")
    else:
        full = sum(1 for r in opt_rows if r["coverage_pct"] >= 95)
        partial = sum(1 for r in opt_rows
                       if 0 < r["coverage_pct"] < 95)
        empty = sum(1 for r in opt_rows if r["coverage_pct"] == 0)
        print(f"  expiries cached       : {len(opt_rows)}")
        print(f"  fully covered (>=95%) : {full}")
        print(f"  partially covered     : {partial}")
        print(f"  empty                 : {empty}")
        # Show worst 10 by coverage
        opt_rows.sort(key=lambda r: r["coverage_pct"])
        if partial or empty:
            print("  worst-covered expiries:")
            for r in opt_rows[:10]:
                if r["coverage_pct"] < 95:
                    print(f"    {r['expiry']}  "
                          f"{r['total_legs']:>2}/{r['expected_legs']} legs "
                          f"({r['coverage_pct']}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
