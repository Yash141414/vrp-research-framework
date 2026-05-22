"""
paper_trader.py
---------------
Paper-trading harness for H4. Same code path that live will use, with
one swap: a SimulatedFillEngine instead of a broker API.

Two modes:
  --mode replay : walks cached historical data, calls the risk gate at
                  every tick, simulates fills. Use this to validate gate
                  logic + see what live would have done.
  --mode live   : (placeholder) subscribes to broker tick feed at wall-
                  clock time. NOT BUILT YET. Will be added once H4
                  backtest + replay are validated.

The paper trader writes:
  - trades parquet           : every closed trade
  - risk_state.json          : risk-gate state (persisted across runs)
  - risk_audit.jsonl         : append-only audit log

These are kept under {data_dir}/paper/{run_name}/ so multiple paper runs
(e.g. h4a vs h4b) don't stomp on each other.
"""
from __future__ import annotations
import argparse
import logging
import time as _pytime
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import yaml

from .config_loader import load_config
from .data_store import DataStore
from .trading_calendar import trading_days_from_spot
from .risk_gate import RiskGate, RiskConfig
from .hypothesis_h3_costs import round_trip_long_single_leg_cost
from .hypothesis_h4_intraday_buy import (
    H4Params, SIGNAL_EVALUATORS, _load_atm_option, _nearest_weekly_expiry,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("paper_trader")


# ----- Fill engine -----

@dataclass
class Fill:
    timestamp: pd.Timestamp
    price: float
    qty: int
    side: str   # "buy" | "sell"


class SimulatedFillEngine:
    """
    Bar-close fills with no extra slippage on top of the cost model's
    0.5% slippage component. SL/TP fills happen at the trigger price, not
    bar close (intraminute trigger).
    """

    def fill_at_close(self, bar: pd.Series, qty: int, side: str) -> Fill:
        return Fill(timestamp=bar["timestamp"], price=float(bar["close"]),
                    qty=qty, side=side)

    def fill_at_price(self, ts: pd.Timestamp, price: float, qty: int,
                      side: str) -> Fill:
        return Fill(timestamp=ts, price=float(price), qty=qty, side=side)


# ----- Paper trader -----

class PaperTrader:
    def __init__(self, store: DataStore, spot: pd.DataFrame,
                 underlying: str, params: H4Params, signal: str,
                 risk_gate: RiskGate, fill_engine: SimulatedFillEngine,
                 trades_path: Path):
        if signal not in SIGNAL_EVALUATORS:
            raise ValueError(f"unknown signal {signal}")
        self.store = store
        self.spot_idx = spot.set_index("timestamp").sort_index()
        self.underlying = underlying
        self.params = params
        self.signal = signal
        self.signal_fn: Callable = SIGNAL_EVALUATORS[signal]
        self.risk_gate = risk_gate
        self.fill_engine = fill_engine
        self.trades_path = Path(trades_path)
        self._trades: list[dict] = []

    def _simulate_position(self, opt_df: pd.DataFrame,
                            entry_fill: Fill,
                            params: H4Params) -> tuple[Fill, str, float]:
        """Walk forward bar-by-bar with risk gate active. Returns
        (exit_fill, exit_reason, max_unrealized_pts)."""
        qty = params.qty_lots * params.lot_size
        sl_pts = params.sl_inr / qty
        tp_pts = params.tp_inr / qty
        scratch_pts = params.scratch_min_profit_inr / qty
        time_stop_ts = (entry_fill.timestamp.normalize()
                        + pd.Timedelta(hours=params.time_stop.hour,
                                        minutes=params.time_stop.minute))
        scratch_ts = (entry_fill.timestamp.normalize()
                      + pd.Timedelta(hours=params.scratch_time.hour,
                                      minutes=params.scratch_time.minute))
        entry_premium = entry_fill.price
        sl_price = entry_premium - sl_pts
        tp_price = entry_premium + tp_pts

        forward = opt_df[opt_df["timestamp"] > entry_fill.timestamp]
        max_unrealized_pts = 0.0
        max_unrealized_by_scratch = 0.0

        for _, bar in forward.iterrows():
            bar_ts: pd.Timestamp = bar["timestamp"]
            b_high = float(bar["high"])
            b_low = float(bar["low"])
            b_close = float(bar["close"])

            max_unrealized_pts = max(max_unrealized_pts, b_high - entry_premium)
            if bar_ts <= scratch_ts:
                max_unrealized_by_scratch = max(max_unrealized_by_scratch,
                                                 b_high - entry_premium)

            # Risk-gate single-trade SL check (uses unrealized at bar close)
            unrealized_inr = (b_close - entry_premium) * qty
            tick_action = self.risk_gate.on_position_tick(unrealized_inr)
            if tick_action == "flatten_single_trade_sl":
                exit_fill = self.fill_engine.fill_at_close(bar, qty, "sell")
                return exit_fill, "risk_gate_single_trade_sl", max_unrealized_pts

            # Time stop
            if bar_ts >= time_stop_ts:
                exit_fill = self.fill_engine.fill_at_close(bar, qty, "sell")
                return exit_fill, "time_stop", max_unrealized_pts

            # Scratch rule
            if (bar_ts >= scratch_ts and
                    max_unrealized_by_scratch < scratch_pts):
                exit_fill = self.fill_engine.fill_at_close(bar, qty, "sell")
                return exit_fill, "scratch", max_unrealized_pts

            # Intraminute SL/TP (pessimistic: SL first if both)
            sl_hit = b_low <= sl_price
            tp_hit = b_high >= tp_price
            if sl_hit and tp_hit:
                exit_fill = self.fill_engine.fill_at_price(
                    bar_ts, sl_price, qty, "sell")
                return exit_fill, "sl_pessimistic_both", max_unrealized_pts
            if sl_hit:
                exit_fill = self.fill_engine.fill_at_price(
                    bar_ts, sl_price, qty, "sell")
                return exit_fill, "sl", max_unrealized_pts
            if tp_hit:
                exit_fill = self.fill_engine.fill_at_price(
                    bar_ts, tp_price, qty, "sell")
                return exit_fill, "tp", max_unrealized_pts

        # Session end: exit at last available bar
        last = forward.iloc[-1] if not forward.empty else None
        if last is None:
            return entry_fill, "session_end_no_bars", max_unrealized_pts
        exit_fill = self.fill_engine.fill_at_close(last, qty, "sell")
        return exit_fill, "session_end", max_unrealized_pts

    def run_replay(self, start: Optional[date] = None,
                    end: Optional[date] = None) -> pd.DataFrame:
        cached_expiries = self.store.list_cached_expiries(self.underlying)
        all_days = trading_days_from_spot(
            self.spot_idx.reset_index().rename(columns={"index": "timestamp"})
            if "timestamp" not in self.spot_idx.reset_index().columns
            else self.spot_idx.reset_index()
        )
        days = [d for d in all_days
                if (start is None or d >= start)
                and (end is None or d <= end)]
        log.info(f"[{self.signal}] replay over {len(days)} trading days, "
                 f"{len(cached_expiries)} cached expiries")

        for d in days:
            day_ts = pd.Timestamp(d)
            sig = self.signal_fn(self.spot_idx, day_ts, self.params)
            if sig is None:
                continue
            direction, atm, _, entry_ts = sig

            # Risk gate check BEFORE trying to enter
            ok, reason = self.risk_gate.can_open(entry_ts.to_pydatetime())
            if not ok:
                self.risk_gate.audit.write("can_open_denied", {
                    "date": d.isoformat(),
                    "signal": self.signal,
                    "reason": reason,
                })
                if reason.startswith("halted"):
                    log.warning(f"Halted on {d}: {reason}; stopping replay.")
                    break
                continue

            expiry = _nearest_weekly_expiry(cached_expiries, d,
                                             self.params.include_expiry_day)
            if expiry is None:
                continue
            opt_df = _load_atm_option(self.store, self.underlying, expiry,
                                       atm, direction)
            if opt_df.empty:
                continue
            # Find entry bar
            entry_window = opt_df[(opt_df["timestamp"] >= entry_ts) &
                                  (opt_df["timestamp"] <=
                                   entry_ts + pd.Timedelta(minutes=5))]
            if entry_window.empty:
                continue
            entry_bar = entry_window.iloc[0]
            qty = self.params.qty_lots * self.params.lot_size
            entry_fill = self.fill_engine.fill_at_close(entry_bar, qty, "buy")
            self.risk_gate.on_position_opened({
                "date": d.isoformat(),
                "signal": self.signal,
                "direction": direction,
                "expiry": expiry.isoformat(),
                "atm_strike": atm,
                "entry_premium": entry_fill.price,
            })

            exit_fill, exit_reason, max_unr = self._simulate_position(
                opt_df, entry_fill, self.params)

            gross_pnl_pts = exit_fill.price - entry_fill.price
            gross_pnl_inr = gross_pnl_pts * qty
            fees_inr = round_trip_long_single_leg_cost(
                entry_fill.price, exit_fill.price, qty)
            net_pnl_inr = gross_pnl_inr - fees_inr

            self.risk_gate.on_trade_closed(net_pnl_inr, {
                "date": d.isoformat(),
                "signal": self.signal,
                "direction": direction,
                "expiry": expiry.isoformat(),
                "atm_strike": atm,
                "entry_premium": entry_fill.price,
                "exit_premium": exit_fill.price,
                "exit_reason": exit_reason,
                "gross_pnl_inr": gross_pnl_inr,
                "fees_inr": fees_inr,
            })
            self._trades.append({
                "trade_date": d,
                "signal": self.signal,
                "direction": direction,
                "expiry": expiry,
                "atm_strike": atm,
                "entry_time": entry_fill.timestamp,
                "entry_premium": entry_fill.price,
                "exit_time": exit_fill.timestamp,
                "exit_premium": exit_fill.price,
                "exit_reason": exit_reason,
                "gross_pnl_pts": gross_pnl_pts,
                "gross_pnl_inr": gross_pnl_inr,
                "fees_inr": fees_inr,
                "net_pnl_inr": net_pnl_inr,
                "max_unrealized_pts": max_unr,
            })
            # End-of-day rollover for risk gate
            self.risk_gate.on_day_end(d)

        if self._trades:
            df = pd.DataFrame(self._trades)
            self.trades_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.trades_path, index=False)
            log.info(f"Saved {len(df)} trades to {self.trades_path}")
            return df
        return pd.DataFrame()


# ----- CLI -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--signal", choices=["h4a", "h4b"], required=True)
    ap.add_argument("--run-name", default=None,
                    help="Subdirectory under {data_dir}/paper/ for this run. "
                         "Defaults to {signal}_{date}.")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--start", help="YYYY-MM-DD: replay window start")
    ap.add_argument("--end", help="YYYY-MM-DD: replay window end")
    ap.add_argument("--tp-inr", type=float, default=1500.0)
    ap.add_argument("--exclude-expiry-day", action="store_true")
    args = ap.parse_args()

    if args.mode == "live":
        raise SystemExit("live mode not implemented yet; use --mode replay")

    cfg = load_config(args.config)
    store = DataStore(cfg["data_dir"])
    underlying = cfg["universe"]["options_underlying"]
    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    if spot.empty:
        raise SystemExit("No spot data. Run fetch.py first.")
    lot_size = int(cfg.get("lot_sizes", {}).get(underlying, 75))

    run_name = args.run_name or f"{args.signal}_{date.today().isoformat()}"
    run_dir = Path(cfg["data_dir"]) / "paper" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    risk_config = RiskConfig()  # defaults match H4 spec for Rs 20k
    risk_gate = RiskGate(
        config=risk_config,
        state_path=run_dir / "risk_state.json",
        audit_path=run_dir / "risk_audit.jsonl",
    )
    risk_gate.load()

    params = H4Params(
        lot_size=lot_size,
        tp_inr=args.tp_inr,
        include_expiry_day=not args.exclude_expiry_day,
    )

    trader = PaperTrader(
        store=store, spot=spot, underlying=underlying,
        params=params, signal=args.signal,
        risk_gate=risk_gate,
        fill_engine=SimulatedFillEngine(),
        trades_path=run_dir / "trades.parquet",
    )

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    df = trader.run_replay(start=start, end=end)

    snap = risk_gate.snapshot()
    print("=" * 60)
    print(f"PAPER REPLAY [{args.signal}]  TP=Rs{int(args.tp_inr)}")
    print("=" * 60)
    print(f"  trades_closed       : {snap['closed_trades_total']}")
    print(f"  cumulative_pnl_inr  : {snap['cumulative_pnl_inr']:.2f}")
    print(f"  drawdown_inr        : {snap['drawdown_inr']:.2f}")
    print(f"  consecutive_loss_days: {snap['consecutive_loss_days']}")
    print(f"  halted              : {snap['halted']}"
          + (f"  ({snap['halt_reason']})" if snap['halted'] else ""))
    print(f"  trades parquet      : {trader.trades_path}")
    print(f"  audit log           : {run_dir / 'risk_audit.jsonl'}")


if __name__ == "__main__":
    main()
