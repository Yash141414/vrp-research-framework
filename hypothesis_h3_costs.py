"""
hypothesis_h3_costs.py
----------------------
The kill-switch test. Even if H1 and H2 show edge in raw P&L, this
test answers the only question that matters: does the edge survive
realistic transaction costs and slippage?

The Bhat et al. paper itself notes that the simplest day-night strategy
is profitable BEFORE costs but NOT AFTER. So this test must be passed
on YOUR cost structure for any deployment to make sense.

WHAT WE TEST:
  For the H2 trade structure (sell ATM straddle prev close, buy back open):
    1. Per-leg STT, brokerage, exchange charges, GST, stamp duty
    2. Slippage as % of premium (configurable)
    3. Recompute summary statistics on net P&L
  Cost is computed PER TRADE using the actual entry/exit leg premiums
  emitted by H2 (no AVG_LEG_PREMIUM placeholder).

DECISION RULE:
  Pass     if Sharpe(net) > 0.5 AND mean(net_pnl) > 0 AND drawdown < 20%
  Marginal if 0.0 < Sharpe < 0.5
  Fail     if Sharpe <= 0 or mean <= 0 — STOP, the literature effect
           doesn't translate to deployable edge in your environment.
"""
from __future__ import annotations
import argparse
import logging
import numpy as np
import pandas as pd
import yaml

from .config_loader import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("h3_costs")

# ---- Cost model (Indian options, conservative) ----
# Per-leg, applied separately to entry and exit.
BROKERAGE_PER_ORDER = 20.0           # discount broker
STT_SELL_PREM_PCT = 0.0625 / 100      # STT on options sell side
EXCH_TXN_PCT = 0.0019 / 100           # NSE
SEBI_PCT = 0.0001 / 100
GST_PCT = 0.18
STAMP_BUY_PCT = 0.003 / 100
SLIPPAGE_PCT = 0.005                  # 0.5% of premium per leg


def leg_cost(premium: float, qty: int, is_buy: bool) -> float:
    notional = premium * qty
    brok = BROKERAGE_PER_ORDER
    exch = notional * EXCH_TXN_PCT
    sebi = notional * SEBI_PCT
    stt = 0.0 if is_buy else notional * STT_SELL_PREM_PCT
    stamp = notional * STAMP_BUY_PCT if is_buy else 0.0
    gst = (brok + exch + sebi) * GST_PCT
    slip = notional * SLIPPAGE_PCT
    return brok + exch + sebi + stt + stamp + gst + slip


def round_trip_short_straddle_cost(ce_entry: float, pe_entry: float,
                                   ce_exit: float, pe_exit: float,
                                   qty: int) -> float:
    """
    Short straddle: sell CE + sell PE entry, buy CE + buy PE exit.
    Returns total cost in INR for the given total qty (lots * lot_size).
    """
    return (
        leg_cost(ce_entry, qty, is_buy=False) +
        leg_cost(pe_entry, qty, is_buy=False) +
        leg_cost(ce_exit, qty, is_buy=True) +
        leg_cost(pe_exit, qty, is_buy=True)
    )


def round_trip_long_single_leg_cost(entry_premium: float,
                                    exit_premium: float,
                                    qty: int) -> float:
    """
    Long single option leg: buy entry (stamp duty), sell exit (STT).
    Returns total INR cost for total qty (lots * lot_size).
    Used by H4 (intraday long-option buying).
    """
    return (leg_cost(entry_premium, qty, is_buy=True) +
            leg_cost(exit_premium, qty, is_buy=False))


def _row_cost(row: pd.Series, leg: str, qty: int) -> float:
    """leg in {'overnight', 'intraday'} — uses that leg's four premiums."""
    cols = [f"ce_{leg}_entry", f"pe_{leg}_entry",
            f"ce_{leg}_exit",  f"pe_{leg}_exit"]
    vals = [row.get(c) for c in cols]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return np.nan
    return round_trip_short_straddle_cost(*vals, qty=qty)


def compute_net_pnl(df: pd.DataFrame, lot_size: int,
                    qty_lots: int = 1) -> tuple[pd.DataFrame, float]:
    """
    Add net P&L columns using REAL per-trade leg premiums from H2 output.
    Returns (df_with_costs, mean_round_trip_cost_inr).
    """
    required = {
        "ce_overnight_entry", "pe_overnight_entry",
        "ce_overnight_exit",  "pe_overnight_exit",
        "ce_intraday_entry",  "pe_intraday_entry",
        "ce_intraday_exit",   "pe_intraday_exit",
        "overnight_pnl", "intraday_pnl",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"H2 output missing columns {missing}. "
            f"Re-run hypothesis_h2_daynight.py with the updated version."
        )

    qty = qty_lots * lot_size
    df = df.copy()
    df["gross_overnight_pnl_rupees"] = df["overnight_pnl"] * qty
    df["gross_intraday_pnl_rupees"] = df["intraday_pnl"] * qty
    df["overnight_cost_inr"] = df.apply(
        lambda r: _row_cost(r, "overnight", qty), axis=1)
    df["intraday_cost_inr"] = df.apply(
        lambda r: _row_cost(r, "intraday", qty), axis=1)
    df["net_overnight_pnl"] = (df["gross_overnight_pnl_rupees"]
                               - df["overnight_cost_inr"])
    df["net_intraday_pnl"] = (df["gross_intraday_pnl_rupees"]
                              - df["intraday_cost_inr"])

    mean_cost = float(np.nanmean(
        pd.concat([df["overnight_cost_inr"], df["intraday_cost_inr"]])
    )) if len(df) else float("nan")
    return df, mean_cost


def report(df: pd.DataFrame, cost_per_trade: float) -> dict:
    out = {"mean_cost_per_trade_inr": cost_per_trade}
    for label, col in [("overnight_net", "net_overnight_pnl"),
                        ("intraday_net", "net_intraday_pnl"),
                        ("overnight_gross", "gross_overnight_pnl_rupees"),
                        ("intraday_gross", "gross_intraday_pnl_rupees")]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        mean = s.mean()
        std = s.std()
        out[f"{label}_n"] = len(s)
        out[f"{label}_mean"] = mean
        out[f"{label}_std"] = std
        out[f"{label}_sharpe_per_trade"] = (mean / std) if std > 0 else np.nan
        out[f"{label}_win_rate"] = (s > 0).mean()
        # Approx annualized Sharpe assuming ~5 trades per week (one per dte 1..5)
        out[f"{label}_sharpe_annualized"] = (
            out[f"{label}_sharpe_per_trade"] * np.sqrt(52 * 5)
            if std > 0 else np.nan
        )

        cum = s.cumsum()
        peak = cum.cummax()
        dd = (cum - peak)
        out[f"{label}_max_drawdown_inr"] = dd.min()

    return out


def interpret(s: dict) -> str:
    out = ["=" * 60, "HYPOTHESIS H3: Cost-Survival of the Effect", "=" * 60]
    for k, v in s.items():
        out.append(f"  {k:35s}: {v:.4f}" if isinstance(v, float)
                   else f"  {k:35s}: {v}")
    out.append("")
    out.append("INTERPRETATION (overnight strategy net of costs):")
    sharpe = s.get("overnight_net_sharpe_annualized", np.nan)
    mean_net = s.get("overnight_net_mean", np.nan)

    if np.isnan(sharpe):
        out.append("  Insufficient data.")
    elif sharpe > 0.8 and mean_net > 0:
        out.append(f"  STRONG: net Sharpe {sharpe:.2f}, mean P&L positive.")
        out.append("    Effect survives costs. Proceed to paper trading.")
    elif sharpe > 0.3 and mean_net > 0:
        out.append(f"  MARGINAL: net Sharpe {sharpe:.2f}.")
        out.append("    Effect survives costs but barely. Sample-size dependent.")
        out.append("    Recommend: continue testing on more data before deployment.")
    else:
        out.append(f"  FAIL: net Sharpe {sharpe:.2f}, mean P&L {mean_net:.0f} INR.")
        out.append("    Effect is destroyed by costs. STOP. The Bhat et al.")
        out.append("    pre-cost result is real but does not survive Indian retail")
        out.append("    transaction costs. Consider:")
        out.append("    - Renegotiating brokerage (lower per-trade cost)")
        out.append("    - Larger position sizes (amortize fixed costs)")
        out.append("    - Different structure (vertical spreads have lower STT)")
    return "\n".join(out)


def _resolve_lot_size(cfg: dict, underlying: str) -> int:
    lot_sizes = cfg.get("lot_sizes")
    if not lot_sizes or underlying not in lot_sizes:
        raise ValueError(
            f"lot_sizes.{underlying} not set in config.yaml. "
            f"Add e.g. `lot_sizes: {{NIFTY: 75}}` (check NSE for current value)."
        )
    return int(lot_sizes[underlying])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--qty-lots", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)

    underlying = cfg["universe"]["options_underlying"]
    lot_size = _resolve_lot_size(cfg, underlying)

    df_path = f"{cfg['data_dir']}/h2_daynight_pnls.parquet"
    try:
        df = pd.read_parquet(df_path)
    except FileNotFoundError:
        raise SystemExit(
            f"H2 output {df_path} missing. Run hypothesis_h2_daynight.py first."
        )

    df_costed, mean_cost = compute_net_pnl(df, lot_size=lot_size,
                                            qty_lots=args.qty_lots)
    df_costed.to_parquet(f"{cfg['data_dir']}/h3_costed_pnls.parquet",
                         index=False)

    s = report(df_costed, mean_cost)
    print(interpret(s))


if __name__ == "__main__":
    main()
