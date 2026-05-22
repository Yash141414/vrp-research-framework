"""
validation_report.py
--------------------
Aggregates the three hypothesis tests into a single go/no-go report
and decides whether strategy work on this data is justified.

DECISION TREE:
   H1 fails -> STOP. No VRP, no edge.
   H1 passes, H2 fails -> Maybe try other VRP harvest structures (not day-night).
   H1 + H2 pass, H3 fails -> Reduce costs first, then retest.
   All three pass -> PROCEED to paper trading (Step 8).

This script prints a structured summary you should save with each
re-run for audit trail. Future regime changes will require re-running.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import yaml

from .hypothesis_h1_vrp import compute_vrp_series, report as h1_report
from .hypothesis_h2_daynight import compute_daynight_pnls, report as h2_report
from .hypothesis_h3_costs import (compute_net_pnl, report as h3_report,
                                  _resolve_lot_size)
from .data_store import DataStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    store = DataStore(cfg["data_dir"])
    underlying = cfg["universe"]["options_underlying"]

    print("=" * 70)
    print(f"VALIDATION REPORT  generated {datetime.now().isoformat()}")
    print("=" * 70)

    # H1
    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    vix = store.load_vix(cfg["universe"]["vix_symbol"])
    if spot.empty or vix.empty:
        print("Cannot run: spot or VIX data missing.")
        return
    vrp_df = compute_vrp_series(spot, vix)
    h1 = h1_report(vrp_df)
    print("\nH1 (VRP):")
    h1_pass = (h1.get("p_value_one_sided", 1) < 0.05 and
               h1.get("fraction_positive", 0) > 0.5)
    print(f"  result = {'PASS' if h1_pass else 'FAIL'}")
    print(f"  mean VRP = {h1.get('mean_vrp', 0):.6f}")
    print(f"  p-value  = {h1.get('p_value_one_sided', 1):.4f}")

    if not h1_pass:
        print("\n=> H1 failed. STOP. No deployable strategy.")
        return

    # H2
    df2 = compute_daynight_pnls(
        store, spot, vix, underlying=underlying,
        min_volume_per_leg=int(cfg.get("min_volume_per_leg", 0)),
    )
    h2 = h2_report(df2)
    print("\nH2 (Day-Night Asymmetry):")
    h2_pass = (h2.get("paired_p_one_sided", 1) < 0.05 and
               h2.get("mean_overnight_pnl", 0) >
               h2.get("mean_intraday_pnl", 0))
    print(f"  result = {'PASS' if h2_pass else 'FAIL'}")
    print(f"  overnight mean = {h2.get('mean_overnight_pnl', 0):.4f}")
    print(f"  intraday mean  = {h2.get('mean_intraday_pnl', 0):.4f}")

    if not h2_pass:
        print("\n=> H2 failed. VRP exists but day-night structure doesn't.")
        print("   Consider testing other VRP-harvest structures.")
        return

    # H3 — real costs from per-trade premiums; lot_size from config.
    lot_size = _resolve_lot_size(cfg, underlying)
    df3, mean_cost = compute_net_pnl(df2, lot_size=lot_size)
    h3 = h3_report(df3, cost_per_trade=mean_cost)
    print("\nH3 (Cost Survival):")
    sharpe = h3.get("overnight_net_sharpe_annualized", -99)
    h3_pass = sharpe > 0.5
    print(f"  result = {'PASS' if h3_pass else 'FAIL'}")
    print(f"  mean cost / trade (INR) = {mean_cost:.2f}")
    print(f"  net annualized Sharpe   = {sharpe:.3f}")

    print("\n" + "=" * 70)
    if h1_pass and h2_pass and h3_pass:
        print("OVERALL: ALL THREE PASS. Proceed to Step 8 (paper trading).")
        print("Do NOT skip paper trading. Do NOT increase capital before 3 months.")
    else:
        print("OVERALL: NOT READY for capital deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
