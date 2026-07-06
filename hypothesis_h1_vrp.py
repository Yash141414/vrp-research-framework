"""
hypothesis_h1_vrp.py
--------------------
Tests whether a Variance Risk Premium (VRP) exists in Nifty options.

DEFINITION:
    VRP_t = IV_t - RV_t
where:
    IV_t = (India VIX_t / 100)^2               [annualized implied variance]
    RV_t = sum of squared 5-min log returns over next 30 cal days, annualized
           by (365 / days) to match VIX's calendar-day convention

If VRP > 0 on average AND the difference is statistically significant,
the literature's claim is supported on YOUR data.

KEY DESIGN DECISIONS (and why):
- Use 5-min returns, not 1-min. 1-min has microstructure noise that
  inflates RV. Andersen-Bollerslev showed 5-min is the sweet spot.
- Annualize using calendar-day convention to match VIX. Some literature
  uses trading-day; the conversion factor is ~252/365. We do calendar
  to be apples-to-apples with India VIX which uses calendar days.
- Do NOT include the realization period in the VIX value's computation
  (avoid look-ahead): VIX at t is paired with realized vol over [t, t+30d].

OUTPUT:
- Mean VRP, t-statistic, sign-test, distribution by VIX regime.
- Plot: rolling 60-day mean VRP. Should be predominantly positive.
"""
from __future__ import annotations
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from scipy import stats

try:
    from .config_loader import load_config
    from .data_store import DataStore
except ImportError:
    from config_loader import load_config  # type: ignore[no-redef]
    from data_store import DataStore  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("h1_vrp")


def realized_variance_window(spot_5min: pd.DataFrame,
                              start: pd.Timestamp,
                              days: int = 30) -> float:
    """
    Annualized realized variance over [start, start+days calendar days].
    Returns variance in same units as IV^2 (decimal, not %).
    """
    end = start + pd.Timedelta(days=days)
    window = spot_5min[(spot_5min.index >= start) & (spot_5min.index < end)]
    if len(window) < 100:  # need enough bars
        return np.nan
    log_rets = np.log(window["close"] / window["close"].shift(1)).dropna()
    # Squared returns sum
    rv_period = (log_rets ** 2).sum()
    # Annualize: scale by (calendar days in year) / (calendar days in window)
    rv_annualized = rv_period * (365.0 / days)
    return rv_annualized


def compute_vrp_series(spot_1min: pd.DataFrame,
                       vix_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with daily VRP observations:
      [date, iv, rv, vrp]
    """
    # Resample spot to 5-min for RV
    spot_1min = spot_1min.set_index("timestamp").sort_index()
    spot_5min = spot_1min["close"].resample("5min").last().dropna().to_frame()

    vix_daily = vix_daily.set_index("timestamp").sort_index()
    # Use end-of-day VIX value
    vix_eod = vix_daily["close"].resample("1D").last().dropna()

    rows = []
    for d, vix_val in vix_eod.items():
        iv2 = (vix_val / 100.0) ** 2  # annualized variance
        # Pair with FORWARD 30-day RV (no look-ahead)
        rv2 = realized_variance_window(spot_5min, d, days=30)
        rows.append({"date": d, "iv": iv2, "rv": rv2, "vrp": iv2 - rv2})
    df = pd.DataFrame(rows).dropna(subset=["rv"])
    return df


def report(vrp_df: pd.DataFrame) -> dict:
    vrp = vrp_df["vrp"].dropna()
    n = len(vrp)
    mean = vrp.mean()
    std = vrp.std()
    se = std / np.sqrt(n)
    t = mean / se if se > 0 else np.nan
    # One-sided test: VRP > 0
    p_one_side = 1 - stats.t.cdf(t, df=n - 1) if not np.isnan(t) else np.nan
    # Sign test
    pos_frac = (vrp > 0).mean()
    sign_p = stats.binomtest(int((vrp > 0).sum()), n, p=0.5,
                              alternative="greater").pvalue
    return {
        "n_obs": n,
        "mean_vrp": mean,
        "std_vrp": std,
        "t_statistic": t,
        "p_value_one_sided": p_one_side,
        "fraction_positive": pos_frac,
        "sign_test_p_value": sign_p,
        "annualized_vol_iv_mean_pct": np.sqrt(vrp_df["iv"].mean()) * 100,
        "annualized_vol_rv_mean_pct": np.sqrt(vrp_df["rv"].mean()) * 100,
    }


def interpret(stats_dict: dict) -> str:
    out = ["=" * 60, "HYPOTHESIS H1: Variance Risk Premium (IV > RV)",
           "=" * 60]
    for k, v in stats_dict.items():
        out.append(f"  {k:35s}: {v:.6f}" if isinstance(v, float)
                   else f"  {k:35s}: {v}")

    out.append("")
    out.append("INTERPRETATION:")
    p = stats_dict["p_value_one_sided"]
    pos = stats_dict["fraction_positive"]

    if p < 0.01 and pos > 0.55:
        out.append("  ✓ STRONG support for VRP. Mean IV significantly above mean RV.")
        out.append("  ✓ Proceed to H2 (day-night asymmetry test).")
    elif p < 0.05 and pos > 0.50:
        out.append("  ~ MODERATE support for VRP. Effect present but smaller than literature.")
        out.append("  ~ Caution: strategy edge may be marginal after costs.")
    else:
        out.append("  ✗ NO support for VRP in this sample. Stop.")
        out.append("  ✗ The literature's effect is not present in your data window.")
        out.append("  ✗ Either the data has issues, the regime has changed,")
        out.append("    or option-selling strategies will not work here.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"])

    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    vix = store.load_vix(cfg["universe"]["vix_symbol"])
    if spot.empty or vix.empty:
        raise SystemExit("Run fetch.py first to populate the cache.")

    vrp_df = compute_vrp_series(spot, vix)
    out_path = Path(cfg["data_dir"]) / "h1_vrp_series.parquet"
    vrp_df.to_parquet(out_path, index=False)
    log.info(f"Saved VRP series to {out_path} ({len(vrp_df)} obs)")

    stats_dict = report(vrp_df)
    print(interpret(stats_dict))


if __name__ == "__main__":
    main()
