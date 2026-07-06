"""
hypothesis_h4_intraday_buy.py
-----------------------------
Intraday long-option buying, 1 lot at a time, max 1 trade/day.

Tested signals (run in parallel, independent stats):
  H4a  Gap continuation:
         At 09:20 IST, if gap = (today_open - prev_close) / prev_close
         |gap| >= 0.3%: gap up -> long ATM CE, gap down -> long ATM PE.
  H4b  Opening-range breakout (ORB):
         Track 09:15-09:30 high/low. At 09:31 IST:
         spot > range_high -> long ATM CE; spot < range_low -> long ATM PE.

Trade mechanics (per trade):
  - 1 lot ATM CE or PE.
  - SL  : Rs 1,000 loss (5% of Rs 20,000 capital).
  - TP  : default Rs 1,500 (1.5x R); CLI sweeps {1000, 1500, 2000, 2500}.
  - Time stop: 15:15 IST hard exit.
  - Scratch  : if max-unrealized < +Rs 500 by 11:30, exit at that bar's close.
  - Entry cutoff: no entries after 14:30 (irrelevant for H4a/H4b which fire
                  in the morning, but enforced for the live framework).

Pass criteria for each (signal, TP) cell:
  - Net annualized Sharpe > 0.5
  - Mean net P&L per trade > 0 (after fees)
  - Max drawdown < Rs 4,000 (20% of capital)

This module is BACKTEST ONLY. Live paths live in paper_trader.py +
risk_gate.py once H4 passes here.
"""
from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal, Optional
import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from .config_loader import load_config
    from .data_store import DataStore
    from .trading_calendar import trading_days_from_spot
    from .hypothesis_h3_costs import round_trip_long_single_leg_cost
except ImportError:
    from config_loader import load_config  # type: ignore[no-redef]
    from data_store import DataStore  # type: ignore[no-redef]
    from trading_calendar import trading_days_from_spot  # type: ignore[no-redef]
    from hypothesis_h3_costs import round_trip_long_single_leg_cost  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("h4_intraday_buy")


Direction = Literal["long_ce", "long_pe"]


@dataclass
class H4Params:
    capital_inr: float = 20_000.0
    lot_size: int = 75
    qty_lots: int = 1
    sl_inr: float = 1_000.0
    tp_inr: float = 1_500.0
    scratch_time: time = time(11, 30)
    scratch_min_profit_inr: float = 500.0
    time_stop: time = time(15, 15)
    entry_cutoff: time = time(14, 30)
    strike_step: int = 50
    gap_threshold_pct: float = 0.3
    orb_window_start: time = time(9, 15)
    orb_window_end: time = time(9, 30)
    h4a_entry_time: time = time(9, 20)
    h4b_entry_time: time = time(9, 31)
    include_expiry_day: bool = True


@dataclass
class TradeOutcome:
    """One simulated trade."""
    signal: str
    trade_date: date
    direction: Direction
    expiry: date
    atm_strike: int
    entry_time: pd.Timestamp
    entry_premium: float
    exit_time: pd.Timestamp
    exit_premium: float
    exit_reason: str
    gross_pnl_pts: float
    gross_pnl_inr: float
    fees_inr: float
    net_pnl_inr: float
    max_unrealized_pts: float


# -------- Signal evaluators --------

def _spot_close_at(spot_idx: pd.DataFrame, ts: pd.Timestamp,
                   tol_min: int = 2) -> Optional[float]:
    window = spot_idx.loc[
        (spot_idx.index >= ts - pd.Timedelta(minutes=tol_min)) &
        (spot_idx.index <= ts + pd.Timedelta(minutes=tol_min))
    ]
    if window.empty:
        return None
    return float(window.iloc[-1]["close"])


def _prev_trading_close(spot_idx: pd.DataFrame, day: pd.Timestamp,
                        max_lookback_days: int = 5) -> Optional[float]:
    for back in range(1, max_lookback_days + 1):
        ts = ((day - pd.Timedelta(days=back)).normalize()
              + pd.Timedelta(hours=15, minutes=29))
        v = _spot_close_at(spot_idx, ts, tol_min=15)
        if v is not None:
            return v
    return None


def _round_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def evaluate_h4a(spot_idx: pd.DataFrame, day: pd.Timestamp,
                 params: H4Params) -> Optional[tuple[Direction, int, float, pd.Timestamp]]:
    """Gap continuation at 09:20. Returns (direction, atm, entry_premium_ref_spot, ts) or None.

    Note: entry_premium_ref_spot is the spot at entry; caller resolves option premium.
    """
    prev_close = _prev_trading_close(spot_idx, day)
    if prev_close is None:
        return None
    today_open_ts = day.normalize() + pd.Timedelta(hours=9, minutes=15)
    today_open = _spot_close_at(spot_idx, today_open_ts, tol_min=2)
    if today_open is None:
        return None
    gap_pct = (today_open - prev_close) / prev_close * 100.0
    if abs(gap_pct) < params.gap_threshold_pct:
        return None
    direction: Direction = "long_ce" if gap_pct > 0 else "long_pe"
    entry_ts = day.normalize() + pd.Timedelta(
        hours=params.h4a_entry_time.hour,
        minutes=params.h4a_entry_time.minute,
    )
    entry_spot = _spot_close_at(spot_idx, entry_ts, tol_min=2)
    if entry_spot is None:
        return None
    return direction, _round_strike(entry_spot, params.strike_step), entry_spot, entry_ts


def evaluate_h4b(spot_idx: pd.DataFrame, day: pd.Timestamp,
                 params: H4Params) -> Optional[tuple[Direction, int, float, pd.Timestamp]]:
    """ORB at 09:31."""
    window_start = day.normalize() + pd.Timedelta(
        hours=params.orb_window_start.hour,
        minutes=params.orb_window_start.minute,
    )
    window_end = day.normalize() + pd.Timedelta(
        hours=params.orb_window_end.hour,
        minutes=params.orb_window_end.minute,
    )
    bars = spot_idx.loc[(spot_idx.index >= window_start) &
                        (spot_idx.index < window_end)]
    if len(bars) < 10:   # need most of the 15-min window
        return None
    range_high = float(bars["high"].max())
    range_low = float(bars["low"].min())
    entry_ts = day.normalize() + pd.Timedelta(
        hours=params.h4b_entry_time.hour,
        minutes=params.h4b_entry_time.minute,
    )
    entry_spot = _spot_close_at(spot_idx, entry_ts, tol_min=2)
    if entry_spot is None:
        return None
    if entry_spot > range_high:
        direction: Direction = "long_ce"
    elif entry_spot < range_low:
        direction = "long_pe"
    else:
        return None
    return direction, _round_strike(entry_spot, params.strike_step), entry_spot, entry_ts


# -------- Option-data + trade simulator --------

def _nearest_weekly_expiry(cached_expiries: list[date], day: date,
                            include_expiry_day: bool) -> Optional[date]:
    """Nearest cached expiry on or after `day`."""
    threshold = day if include_expiry_day else day + timedelta(days=1)
    future = [e for e in cached_expiries if e >= threshold]
    return future[0] if future else None


def _load_atm_option(store: DataStore, underlying: str, expiry: date,
                     strike: int, direction: Direction) -> pd.DataFrame:
    is_call = (direction == "long_ce")
    df = store.load_option(underlying, expiry, strike, is_call=is_call)
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def simulate_trade(opt_df: pd.DataFrame, entry_ts: pd.Timestamp,
                   params: H4Params,
                   signal_name: str, trade_date: date,
                   direction: Direction, expiry: date,
                   atm_strike: int) -> Optional[TradeOutcome]:
    """
    Walk minute bars forward from entry_ts and exit on first triggered rule.
    Returns None if entry can't be filled (no bar at entry_ts).
    """
    qty = params.qty_lots * params.lot_size
    sl_pts = params.sl_inr / qty                 # premium drop that triggers SL
    tp_pts = params.tp_inr / qty                 # premium gain that triggers TP
    scratch_pts = params.scratch_min_profit_inr / qty
    time_stop_ts = (entry_ts.normalize()
                    + pd.Timedelta(hours=params.time_stop.hour,
                                    minutes=params.time_stop.minute))
    scratch_ts = (entry_ts.normalize()
                  + pd.Timedelta(hours=params.scratch_time.hour,
                                  minutes=params.scratch_time.minute))

    # Find entry bar
    entry_row = opt_df[(opt_df["timestamp"] >= entry_ts) &
                       (opt_df["timestamp"] <= entry_ts + pd.Timedelta(minutes=5))]
    if entry_row.empty:
        return None
    entry_bar = entry_row.iloc[0]
    entry_premium = float(entry_bar["close"])
    if entry_premium <= 0:
        return None

    sl_price = entry_premium - sl_pts
    tp_price = entry_premium + tp_pts

    forward = opt_df[opt_df["timestamp"] > entry_bar["timestamp"]]
    max_unrealized_pts = 0.0
    exit_premium: Optional[float] = None
    exit_ts: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    max_unrealized_by_scratch: float = 0.0

    for _, bar in forward.iterrows():
        bar_ts: pd.Timestamp = bar["timestamp"]
        b_high = float(bar["high"])
        b_low = float(bar["low"])
        b_close = float(bar["close"])

        # Track max unrealized in option premium points
        max_unrealized_pts = max(max_unrealized_pts, b_high - entry_premium)
        if bar_ts <= scratch_ts:
            max_unrealized_by_scratch = max(max_unrealized_by_scratch,
                                            b_high - entry_premium)

        # Time stop check (end of day)
        if bar_ts >= time_stop_ts:
            exit_premium = b_close
            exit_ts = bar_ts
            exit_reason = "time_stop"
            break

        # Scratch rule (only check at the bar AT scratch_ts)
        if (bar_ts >= scratch_ts and
                max_unrealized_by_scratch < scratch_pts and
                exit_reason is None):
            exit_premium = b_close
            exit_ts = bar_ts
            exit_reason = "scratch"
            break

        # SL/TP intraminute. Pessimistic order: SL first if both possible.
        sl_hit = b_low <= sl_price
        tp_hit = b_high >= tp_price
        if sl_hit and tp_hit:
            exit_premium = sl_price
            exit_ts = bar_ts
            exit_reason = "sl_pessimistic_both"
            break
        if sl_hit:
            exit_premium = sl_price
            exit_ts = bar_ts
            exit_reason = "sl"
            break
        if tp_hit:
            exit_premium = tp_price
            exit_ts = bar_ts
            exit_reason = "tp"
            break

    if exit_premium is None:
        # No more bars (end of session reached without trigger) — use last close
        last = forward.iloc[-1] if not forward.empty else entry_bar
        exit_premium = float(last["close"])
        exit_ts = last["timestamp"]
        exit_reason = "session_end"

    gross_pnl_pts = exit_premium - entry_premium
    gross_pnl_inr = gross_pnl_pts * qty
    fees_inr = round_trip_long_single_leg_cost(entry_premium, exit_premium, qty)
    net_pnl_inr = gross_pnl_inr - fees_inr

    return TradeOutcome(
        signal=signal_name,
        trade_date=trade_date,
        direction=direction,
        expiry=expiry,
        atm_strike=atm_strike,
        entry_time=entry_bar["timestamp"],
        entry_premium=entry_premium,
        exit_time=exit_ts,
        exit_premium=exit_premium,
        exit_reason=exit_reason,
        gross_pnl_pts=gross_pnl_pts,
        gross_pnl_inr=gross_pnl_inr,
        fees_inr=fees_inr,
        net_pnl_inr=net_pnl_inr,
        max_unrealized_pts=max_unrealized_pts,
    )


# -------- Backtest driver --------

SIGNAL_EVALUATORS = {
    "h4a": evaluate_h4a,
    "h4b": evaluate_h4b,
}


def backtest(store: DataStore, spot: pd.DataFrame, underlying: str,
             signal: str, params: H4Params) -> pd.DataFrame:
    """Run one (signal, TP) cell of the backtest. Returns trades DataFrame."""
    if signal not in SIGNAL_EVALUATORS:
        raise ValueError(f"unknown signal {signal}; choose from "
                         f"{list(SIGNAL_EVALUATORS)}")
    evaluator = SIGNAL_EVALUATORS[signal]

    spot_idx = spot.set_index("timestamp").sort_index()
    trading_days = trading_days_from_spot(spot)
    cached_expiries = store.list_cached_expiries(underlying)
    log.info(f"[{signal}] {len(trading_days)} trading days, "
             f"{len(cached_expiries)} cached expiries")

    trades: list[TradeOutcome] = []
    for d in trading_days:
        day_ts = pd.Timestamp(d)
        sig = evaluator(spot_idx, day_ts, params)
        if sig is None:
            continue
        direction, atm, _, entry_ts = sig
        expiry = _nearest_weekly_expiry(cached_expiries, d,
                                         params.include_expiry_day)
        if expiry is None:
            continue
        opt_df = _load_atm_option(store, underlying, expiry, atm, direction)
        if opt_df.empty:
            continue
        outcome = simulate_trade(opt_df, entry_ts, params,
                                  signal_name=signal, trade_date=d,
                                  direction=direction, expiry=expiry,
                                  atm_strike=atm)
        if outcome is not None:
            trades.append(outcome)

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


# -------- Statistics --------

def report(trades: pd.DataFrame, params: H4Params) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    net = trades["net_pnl_inr"]
    wins = net[net > 0]
    losses = net[net <= 0]

    mean = float(net.mean())
    std = float(net.std()) if len(net) > 1 else float("nan")
    sharpe_per_trade = (mean / std) if std and std > 0 else float("nan")
    sharpe_annualized = (sharpe_per_trade * np.sqrt(252)
                          if not np.isnan(sharpe_per_trade) else float("nan"))

    # Max drawdown on cumulative net P&L
    cum = net.cumsum()
    peak = cum.cummax()
    dd_series = cum - peak
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    # T-test: H0 mean = 0, H1 mean > 0
    if len(net) >= 10 and std and std > 0:
        t_stat = mean / (std / np.sqrt(len(net)))
        p_one = float(1.0 - stats.t.cdf(t_stat, df=len(net) - 1))
    else:
        t_stat = float("nan")
        p_one = float("nan")

    profit_factor = (float(wins.sum() / abs(losses.sum()))
                     if len(losses) and losses.sum() != 0 else float("inf"))

    # Pass-criteria evaluation
    passes = (
        not np.isnan(sharpe_annualized) and sharpe_annualized > 0.5 and
        mean > 0 and
        abs(max_dd) < 0.20 * params.capital_inr
    )

    return {
        "n_trades": int(len(net)),
        "n_wins": int(len(wins)),
        "win_rate": float(len(wins) / len(net)),
        "mean_net_pnl_inr": mean,
        "median_net_pnl_inr": float(net.median()),
        "mean_winner_inr": float(wins.mean()) if len(wins) else 0.0,
        "mean_loser_inr": float(losses.mean()) if len(losses) else 0.0,
        "std_net_pnl_inr": std,
        "sharpe_per_trade": sharpe_per_trade,
        "sharpe_annualized": sharpe_annualized,
        "max_drawdown_inr": max_dd,
        "profit_factor": profit_factor,
        "t_statistic": t_stat,
        "p_value_one_sided": p_one,
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
        "passes_h4": bool(passes),
    }


def interpret(s: dict, signal: str, tp: float) -> str:
    lines = ["=" * 60,
             f"H4 [{signal}] TP=Rs{int(tp)}",
             "=" * 60]
    for k, v in s.items():
        if isinstance(v, float):
            lines.append(f"  {k:30s}: {v:.4f}")
        else:
            lines.append(f"  {k:30s}: {v}")
    lines.append("")
    if s.get("n_trades", 0) < 30:
        lines.append("  WARN: < 30 trades. Statistics unreliable; "
                     "extend data window.")
    if s.get("passes_h4"):
        lines.append("  -> Signal/TP passes H4 backtest criteria. "
                     "Eligible for paper trading next.")
    else:
        lines.append("  -> Signal/TP fails one or more of: net Sharpe>0.5, "
                     "mean net P&L>0, |DD|<20% capital.")
    return "\n".join(lines)


# -------- CLI --------

DEFAULT_TP_SWEEP = (1_000.0, 1_500.0, 2_000.0, 2_500.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--signal", choices=["h4a", "h4b", "both"], default="both",
                    help="Which signal to test (default: both)")
    ap.add_argument("--tp-inr", type=float, default=None,
                    help="If set, test only this TP value; else sweep "
                         f"{DEFAULT_TP_SWEEP}")
    ap.add_argument("--exclude-expiry-day", action="store_true",
                    help="Skip trades that would use today's expiry "
                         "(gamma/theta cliff).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"])
    underlying = cfg["universe"]["options_underlying"]
    lot_size = int(cfg.get("lot_sizes", {}).get(underlying, 75))

    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    if spot.empty:
        raise SystemExit("No spot data. Run fetch.py first.")

    signals = ["h4a", "h4b"] if args.signal == "both" else [args.signal]
    tp_grid = (args.tp_inr,) if args.tp_inr is not None else DEFAULT_TP_SWEEP

    out_rows = []
    for sig in signals:
        for tp in tp_grid:
            params = H4Params(
                lot_size=lot_size,
                tp_inr=tp,
                include_expiry_day=not args.exclude_expiry_day,
            )
            trades = backtest(store, spot, underlying, sig, params)
            trades_path = (f"{cfg['data_dir']}/h4_{sig}_tp{int(tp)}"
                           f"_trades.parquet")
            if not trades.empty:
                trades.to_parquet(trades_path, index=False)
            s = report(trades, params)
            print(interpret(s, sig, tp))
            out_rows.append({"signal": sig, "tp_inr": tp, **s})

    # Save summary across the grid
    summary_path = f"{cfg['data_dir']}/h4_summary.parquet"
    pd.DataFrame(out_rows).to_parquet(summary_path, index=False)
    log.info(f"H4 grid summary saved to {summary_path}")


if __name__ == "__main__":
    main()
