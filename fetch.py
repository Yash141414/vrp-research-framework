"""
fetch.py
--------
Driver script that fetches: spot, VIX, and the ATM±N strikes of each weekly
expiry of Nifty options across the configured date range.

USAGE:
    python fetch.py --start 2022-01-01 --end 2024-12-31 --strikes-around 5

What gets fetched per trading day:
  - Nifty 50 spot, 1-min bars
  - India VIX, 1-min bars (or daily if intraday VIX not available)
  - For the current and next weekly expiry: ATM ± N strikes,
    both CE and PE, 1-min bars

Why not all strikes: storage and rate-limit cost is enormous, and the
hypothesis tests only need ATM-vicinity options. ATM±5 strikes
(50pt steps = ATM±250pts) covers the high-liquidity region.
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import yaml
import pandas as pd

from .data_store import DataStore
from .brokers import BrokerAdapter
from .config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("fetch")


def make_adapter(cfg: dict) -> BrokerAdapter:
    primary = cfg["primary_broker"]
    if primary == "kite":
        from .brokers.kite import KiteAdapter
        return KiteAdapter(
            api_key=cfg["kite"]["api_key"],
            api_secret=cfg["kite"]["api_secret"],
            access_token=cfg["kite"].get("access_token") or None,
        )
    elif primary == "upstox":
        from .brokers.upstox import UpstoxAdapter
        return UpstoxAdapter(
            api_key=cfg["upstox"]["api_key"],
            api_secret=cfg["upstox"]["api_secret"],
            redirect_uri=cfg["upstox"]["redirect_uri"],
            access_token=cfg["upstox"].get("access_token") or None,
        )
    else:
        raise ValueError(f"Broker {primary} not implemented yet")


def round_to_strike(p: float, step: int = 50) -> int:
    return int(round(p / step) * step)


def find_weekly_expiries(adapter: BrokerAdapter, underlying: str,
                          start: date, end: date,
                          step_days: int = 30) -> list[date]:
    """
    Walk the [start, end] range in `step_days` chunks, asking the broker
    "what contracts are tradeable as of this date" at each point. Union
    the results to recover every expiry that was ever listed in the window.

    NOTE on historical discovery: live broker instrument masters reflect
    CURRENT state. Already-expired contracts are typically not returned by
    today's master, so backfilling historical windows beyond what is
    currently listed may yield an empty set. Run fetch.py periodically to
    accumulate the cache while contracts are still listed.
    """
    expiries: set[date] = set()
    cursor = start
    listing_window = 14   # match the listing-window heuristic in adapters
    while cursor <= end:
        contracts = adapter.list_option_contracts(
            underlying, as_of=cursor, max_dte_days=step_days + listing_window)
        for c in contracts:
            if start <= c.expiry <= end + timedelta(days=listing_window):
                expiries.add(c.expiry)
        cursor += timedelta(days=step_days)
    return sorted(expiries)


def _df_covers_window(df: pd.DataFrame, start: datetime, end: datetime,
                      min_bars: int = 1) -> bool:
    """True if `df` has >= min_bars in [start, end]."""
    if df.empty:
        return False
    ts = pd.to_datetime(df["timestamp"])
    in_window = ((ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))).sum()
    return in_window >= min_bars


def fetch_spot_and_vix(adapter: BrokerAdapter, store: DataStore,
                       start: datetime, end: datetime,
                       cfg: dict, force: bool = False) -> None:
    spot_sym = cfg["universe"]["spot_symbol"]
    vix_sym = cfg["universe"]["vix_symbol"]
    res = cfg["resolution"]

    # Spot
    cached_spot = store.load_spot(spot_sym)
    expected_min_bars = max(1, int((end - start).days * 0.5 * 300))  # rough
    if not force and _df_covers_window(cached_spot, start, end,
                                       min_bars=expected_min_bars):
        log.info(f"Spot {spot_sym} already covers {start.date()}..{end.date()}; "
                 f"skip (pass --force to refetch).")
    else:
        log.info(f"Fetching spot bars for {spot_sym} "
                 f"{start.date()}..{end.date()}")
        spot_bars = adapter.get_index_bars(spot_sym, start, end, res)
        store.save_spot(spot_sym, spot_bars)

    # VIX
    cached_vix = store.load_vix(vix_sym)
    if not force and _df_covers_window(cached_vix, start, end, min_bars=1):
        log.info(f"VIX {vix_sym} already cached; skip.")
        return
    log.info(f"Fetching VIX bars for {vix_sym} {start.date()}..{end.date()}")
    try:
        vix_bars = adapter.get_index_bars(vix_sym, start, end, res)
        store.save_vix(vix_sym, vix_bars)
    except Exception as e:
        log.warning(f"VIX intraday unavailable, falling back to daily: {e}")
        vix_bars = adapter.get_index_bars(vix_sym, start, end, "day")
        store.save_vix(vix_sym, vix_bars)


def fetch_options(adapter: BrokerAdapter, store: DataStore,
                  underlying: str, start: date, end: date,
                  strikes_around: int, resolution: str,
                  force: bool = False) -> None:
    """
    For every expiry in the range, determine the spot price near that
    expiry's listing/active period, then fetch ATM±N CE+PE 1-minute bars.

    We need spot already cached before this runs.
    """
    spot_df = store.load_spot("NIFTY 50")
    if spot_df.empty:
        raise RuntimeError("Spot data not cached. Run fetch_spot_and_vix first.")
    spot_df = spot_df.set_index("timestamp").sort_index()

    expiries = find_weekly_expiries(adapter, underlying, start, end)
    log.info(f"Found {len(expiries)} expiries in range")

    for exp in expiries:
        # Look up spot at start of expiry week (Monday closest to exp - 4 days)
        ref_dt = pd.Timestamp(exp - timedelta(days=4))
        nearest = spot_df.index[spot_df.index <= ref_dt]
        if len(nearest) == 0:
            log.warning(f"No spot data near {exp}, skipping")
            continue
        spot_at_listing = float(spot_df.loc[nearest[-1], "close"])
        atm = round_to_strike(spot_at_listing)
        target_strikes = [atm + i * 50 for i in
                          range(-strikes_around, strikes_around + 1)]

        # Get all contracts for this expiry
        contracts_today = adapter.list_option_contracts(
            underlying, as_of=exp - timedelta(days=7))
        relevant = [c for c in contracts_today
                    if c.expiry == exp and c.strike in target_strikes]
        log.info(f"Expiry {exp}: ATM={atm}, fetching {len(relevant)} contracts")

        # Active window: from 7 days before expiry to expiry day
        active_start = datetime.combine(
            max(date.fromisoformat(str(start)), exp - timedelta(days=7)),
            datetime.min.time())
        active_end = datetime.combine(exp, datetime.max.time())

        for c in relevant:
            if not force:
                cached = store.load_option(underlying, c.expiry, c.strike,
                                            c.is_call)
                # Active window spans ~7 calendar days; ~5 trading days * 375
                # bars/day = ~1875 expected. Skip if we have at least 60%.
                if _df_covers_window(cached, active_start, active_end,
                                     min_bars=1000):
                    continue
            try:
                bars = adapter.get_historical_bars(
                    c.instrument_token, active_start, active_end, resolution)
                store.save_option(c, bars)
            except Exception as e:
                log.warning(f"Failed to fetch {c.tradingsymbol}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--strikes-around", type=int, default=5,
                    help="ATM±N strikes to fetch per expiry")
    ap.add_argument("--skip-options", action="store_true",
                    help="Fetch only spot+VIX (faster sanity check)")
    ap.add_argument("--force", action="store_true",
                    help="Refetch even if cache already covers the window")
    args = ap.parse_args()

    cfg = load_config(args.config)

    adapter = make_adapter(cfg)
    adapter.login()  # may raise interactive instructions
    store = DataStore(cfg["data_dir"])

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    start_dt = datetime.combine(start_d, datetime.min.time())
    end_dt = datetime.combine(end_d, datetime.max.time())

    fetch_spot_and_vix(adapter, store, start_dt, end_dt, cfg,
                       force=args.force)
    if not args.skip_options:
        fetch_options(
            adapter, store,
            underlying=cfg["universe"]["options_underlying"],
            start=start_d, end=end_d,
            strikes_around=args.strikes_around,
            resolution=cfg["resolution"],
            force=args.force,
        )

    log.info("Fetch complete. Run hypothesis tests next.")


if __name__ == "__main__":
    main()
