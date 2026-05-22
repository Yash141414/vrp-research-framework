"""
hypothesis_h2_daynight.py
-------------------------
Tests the day-night asymmetry in Nifty option returns.

CLAIM (Bhat, Pandey, Rao 2024):
    Short-option overnight returns: positive and significant
    Short-option intraday returns:  negative or weakly positive
    Net effect: VRP earned mostly overnight

METHODOLOGY (replication design):
For each weekly expiry, on each day from 7 days-to-expiry to 1 day-to-expiry:
  1. Overnight leg:
       - Pick ATM strike using LAST trading day's 15:29 spot (no look-ahead).
       - Sell straddle at prev close (15:29), buy back at today's open (09:15).
  2. Intraday leg:
       - Pick ATM strike using today's 09:15 spot.
       - Sell straddle at today's open, buy back at today's close (15:29).
  3. Emit BOTH legs' per-contract entry/exit premiums so H3 can compute
     real round-trip costs (no AVG_LEG_PREMIUM placeholder).

Note we use NON-DELTA-HEDGED short straddle as a proxy. True replication
of Bhat needs delta hedges using futures, which we omit for simplicity in
this validation step. The straddle structure is approximately delta-neutral
at inception so this is a defensible proxy at a single point in time.

Output:
  - Mean overnight straddle P&L (should be positive if hypothesis holds)
  - Mean intraday straddle P&L (should be < overnight)
  - Paired t-test: overnight vs intraday
  - Distribution by VIX regime (high VIX -> stronger effect expected per literature)

INTERPRETATION RULE:
  Pass if E[overnight_pnl] > E[intraday_pnl] with p < 0.05
"""
from __future__ import annotations
import argparse
import logging
from datetime import timedelta, date
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .config_loader import load_config
from .data_store import DataStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("h2_daynight")


def _atm_from_spot(spot_close: float, step: int = 50) -> int:
    return int(round(spot_close / step) * step)


def _spot_close_at_or_before(spot_indexed: pd.DataFrame,
                             target: pd.Timestamp,
                             lookback_min: int = 5) -> float | None:
    """Return spot close of the last bar in [target-lookback_min, target]."""
    window = spot_indexed.loc[
        (spot_indexed.index >= target - pd.Timedelta(minutes=lookback_min)) &
        (spot_indexed.index <= target)
    ]
    if window.empty:
        return None
    return float(window.iloc[-1]["close"])


def _prev_trading_close_ts(spot_indexed: pd.DataFrame,
                           day_ts: pd.Timestamp,
                           max_lookback_days: int = 5) -> pd.Timestamp | None:
    """Find the 15:29 (or last available) bar on the most recent trading day
    strictly before day_ts. Walks back day-by-day to skip weekends/holidays."""
    for back in range(1, max_lookback_days + 1):
        target = ((day_ts - pd.Timedelta(days=back)).normalize()
                  + pd.Timedelta(hours=15, minutes=29))
        window = spot_indexed.loc[
            (spot_indexed.index >= target - pd.Timedelta(minutes=15)) &
            (spot_indexed.index <= target + pd.Timedelta(minutes=1))
        ]
        if not window.empty:
            return window.index[-1]
    return None


def _option_close_at(opt_df: pd.DataFrame,
                     target: pd.Timestamp,
                     tolerance_min: int = 5,
                     min_volume: int = 0) -> float | None:
    """Find option close at the first bar in [target, target+tolerance_min]
    with volume >= min_volume. Returns None if no qualifying bar."""
    if opt_df.empty:
        return None
    window = opt_df[(opt_df["timestamp"] >= target) &
                    (opt_df["timestamp"] <= target + pd.Timedelta(minutes=tolerance_min))]
    if min_volume > 0:
        window = window[window["volume"] >= min_volume]
    if window.empty:
        return None
    return float(window.iloc[0]["close"])


def compute_daynight_pnls(store: DataStore, spot: pd.DataFrame,
                          vix: pd.DataFrame,
                          underlying: str = "NIFTY",
                          min_volume_per_leg: int = 0) -> pd.DataFrame:
    """
    For each expiry x each day-in-week, compute:
      - overnight straddle pnl (sell at prev trading close, buy back at open)
      - intraday straddle pnl  (sell at open,               buy back at close)

    Strike is chosen WITHOUT look-ahead:
      - overnight strike from prev trading close spot
      - intraday  strike from today's 09:15 spot
    """
    spot_idx = spot.set_index("timestamp").sort_index()
    expiries = store.list_cached_expiries(underlying)
    rows = []

    for exp in expiries:
        for dte_days in range(6, 0, -1):  # 6,5,4,3,2,1
            day_ts = pd.Timestamp(exp - timedelta(days=dte_days))
            open_t = day_ts.normalize() + pd.Timedelta(hours=9, minutes=15)
            close_t = day_ts.normalize() + pd.Timedelta(hours=15, minutes=29)

            # --- Overnight ATM: from PREV trading close spot ---
            prev_close_ts = _prev_trading_close_ts(spot_idx, day_ts)
            if prev_close_ts is None:
                continue
            prev_spot = _spot_close_at_or_before(spot_idx, prev_close_ts,
                                                 lookback_min=2)
            overnight_atm = (_atm_from_spot(prev_spot)
                             if prev_spot is not None else None)

            # --- Intraday ATM: from today's 09:15 spot ---
            today_open_spot = _spot_close_at_or_before(spot_idx, open_t,
                                                       lookback_min=3)
            if today_open_spot is None:
                # market not open today (holiday) — skip
                continue
            intraday_atm = _atm_from_spot(today_open_spot)

            # --- Overnight leg ---
            ce_on_entry = pe_on_entry = ce_on_exit = pe_on_exit = None
            overnight_pnl = np.nan
            if overnight_atm is not None:
                ce_on = store.load_option(underlying, exp, overnight_atm,
                                          is_call=True)
                pe_on = store.load_option(underlying, exp, overnight_atm,
                                          is_call=False)
                if not ce_on.empty and not pe_on.empty:
                    ce_on_entry = _option_close_at(ce_on, prev_close_ts,
                                                    min_volume=min_volume_per_leg)
                    pe_on_entry = _option_close_at(pe_on, prev_close_ts,
                                                    min_volume=min_volume_per_leg)
                    ce_on_exit = _option_close_at(ce_on, open_t,
                                                   min_volume=min_volume_per_leg)
                    pe_on_exit = _option_close_at(pe_on, open_t,
                                                   min_volume=min_volume_per_leg)
                    if all(v is not None for v in
                           (ce_on_entry, pe_on_entry, ce_on_exit, pe_on_exit)):
                        overnight_pnl = ((ce_on_entry + pe_on_entry)
                                         - (ce_on_exit + pe_on_exit))

            # --- Intraday leg ---
            ce_id = store.load_option(underlying, exp, intraday_atm,
                                      is_call=True)
            pe_id = store.load_option(underlying, exp, intraday_atm,
                                      is_call=False)
            ce_id_entry = pe_id_entry = ce_id_exit = pe_id_exit = None
            intraday_pnl = np.nan
            if not ce_id.empty and not pe_id.empty:
                ce_id_entry = _option_close_at(ce_id, open_t,
                                                min_volume=min_volume_per_leg)
                pe_id_entry = _option_close_at(pe_id, open_t,
                                                min_volume=min_volume_per_leg)
                ce_id_exit = _option_close_at(ce_id, close_t,
                                               min_volume=min_volume_per_leg)
                pe_id_exit = _option_close_at(pe_id, close_t,
                                               min_volume=min_volume_per_leg)
                if all(v is not None for v in
                       (ce_id_entry, pe_id_entry, ce_id_exit, pe_id_exit)):
                    intraday_pnl = ((ce_id_entry + pe_id_entry)
                                    - (ce_id_exit + pe_id_exit))

            # VIX at open
            vix_today = vix[(vix["timestamp"] >= day_ts) &
                            (vix["timestamp"] < day_ts + pd.Timedelta(days=1))]
            vix_val = (float(vix_today.iloc[0]["close"])
                       if not vix_today.empty else np.nan)

            rows.append({
                "expiry": exp,
                "trade_date": day_ts.date(),
                "dte": dte_days,
                "vix": vix_val,
                "overnight_atm": overnight_atm,
                "intraday_atm": intraday_atm,
                "ce_overnight_entry": ce_on_entry,
                "pe_overnight_entry": pe_on_entry,
                "ce_overnight_exit": ce_on_exit,
                "pe_overnight_exit": pe_on_exit,
                "ce_intraday_entry": ce_id_entry,
                "pe_intraday_entry": pe_id_entry,
                "ce_intraday_exit": ce_id_exit,
                "pe_intraday_exit": pe_id_exit,
                "overnight_pnl": overnight_pnl,
                "intraday_pnl": intraday_pnl,
            })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict:
    on = df["overnight_pnl"].dropna()
    intra = df["intraday_pnl"].dropna()
    paired = df[["overnight_pnl", "intraday_pnl"]].dropna()

    out = {
        "n_overnight_obs": len(on),
        "n_intraday_obs": len(intra),
        "n_paired_obs": len(paired),
        "mean_overnight_pnl": on.mean() if len(on) > 0 else np.nan,
        "mean_intraday_pnl": intra.mean() if len(intra) > 0 else np.nan,
        "median_overnight_pnl": on.median() if len(on) > 0 else np.nan,
        "median_intraday_pnl": intra.median() if len(intra) > 0 else np.nan,
    }

    # Paired t-test: H0: overnight = intraday, H1: overnight > intraday
    if len(paired) >= 30:
        diff = paired["overnight_pnl"] - paired["intraday_pnl"]
        t = diff.mean() / (diff.std() / np.sqrt(len(diff)))
        p_one = 1 - stats.t.cdf(t, df=len(diff) - 1)
        out["paired_t_stat"] = t
        out["paired_p_one_sided"] = p_one
        out["mean_difference"] = diff.mean()

    return out


def interpret(stats_dict: dict) -> str:
    out = ["=" * 60,
           "HYPOTHESIS H2: Day-Night Asymmetry in Option Returns",
           "=" * 60]
    for k, v in stats_dict.items():
        out.append(f"  {k:30s}: {v:.4f}" if isinstance(v, float)
                   else f"  {k:30s}: {v}")
    out.append("")
    out.append("INTERPRETATION:")
    if stats_dict.get("n_paired_obs", 0) < 30:
        out.append("  Too few paired observations to test (need >=30).")
        return "\n".join(out)

    on_mean = stats_dict["mean_overnight_pnl"]
    intra_mean = stats_dict["mean_intraday_pnl"]
    p = stats_dict.get("paired_p_one_sided", 1.0)

    if on_mean > 0 and on_mean > intra_mean and p < 0.05:
        out.append("  STRONG support for day-night asymmetry.")
        out.append("    Overnight straddle P&L is significantly higher.")
        out.append("    A short-overnight strategy is justified to test next.")
    elif on_mean > intra_mean:
        out.append("  Overnight > intraday, but not statistically significant.")
        out.append("  Larger sample needed; do not commit capital.")
    else:
        out.append("  NO support: overnight pnl is not greater than intraday.")
        out.append("  The Bhat et al. result does not replicate on this data.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"])

    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    vix = store.load_vix(cfg["universe"]["vix_symbol"])
    if spot.empty:
        raise SystemExit("No spot data — run fetch.py first.")

    df = compute_daynight_pnls(
        store, spot, vix,
        underlying=cfg["universe"]["options_underlying"],
        min_volume_per_leg=int(cfg.get("min_volume_per_leg", 0)),
    )
    out_path = f"{cfg['data_dir']}/h2_daynight_pnls.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} day-night observations to {out_path}")

    stats_dict = report(df)
    print(interpret(stats_dict))


if __name__ == "__main__":
    main()
