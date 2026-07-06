"""
pipeline.py
-----------
Single-command orchestrator for the four runtime steps:

  Step 1: fetch real data         -> brokers + DataStore
  Step 2: health_check report     -> coverage sanity
  Step 3: H4 backtest             -> H4a + H4b across TP sweep
  Step 4: paper-trade replay      -> winning (signal, TP) cell + risk gate

PREREQUISITES (you must provide these):
  - config.yaml with broker api_key + api_secret + access_token
    (run `python -m nifty_data_layer.login_kite` daily to refresh the token)
  - Broker account (Kite or Upstox) with historical-data API access

This orchestrator wraps the programmatic APIs of the underlying modules,
so the same CLIs (`python -m nifty_data_layer.fetch`, etc.) work
identically if you want to run any stage individually.

Run:
  # Full pipeline (steps 1-4)
  python -m nifty_data_layer.pipeline --start 2023-01-01 --end 2026-05-15

  # Skip the slow fetch if data is already cached:
  python -m nifty_data_layer.pipeline --start 2023-01-01 --end 2026-05-15 --skip-fetch

  # Just the analysis stages (no paper trade):
  python -m nifty_data_layer.pipeline --start ... --end ... --skip-fetch --skip-paper

Flags:
  --skip-fetch        : do not call broker; use cached data as-is
  --skip-health       : skip the health_check stage
  --skip-h4           : skip the H4 backtest
  --skip-paper        : skip the paper-trade replay
  --force             : refetch data even if cache covers the window
  --signal {h4a,h4b,both}
  --tp-inr <N>        : override the H4 TP value (default: sweep 1k..2.5k)
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd

from .config_loader import load_config
from .data_store import DataStore
from .trading_calendar import trading_days_from_spot
from .health_check import report_spot, report_vix, report_options
from .hypothesis_h4_intraday_buy import (
    H4Params, backtest as h4_backtest,
    report as h4_report, interpret as h4_interpret,
    DEFAULT_TP_SWEEP,
)
from .risk_gate import RiskGate, RiskConfig
from .paper_trader import PaperTrader, SimulatedFillEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ----- Stage 1: fetch -----

def step_fetch(cfg: dict, start: date, end: date, force: bool) -> None:
    _banner("STEP 1: FETCH (broker -> local cache)")
    from .fetch import make_adapter, fetch_spot_and_vix, fetch_options

    adapter = make_adapter(cfg)
    adapter.login()  # interactive instructions raised on missing token
    store = DataStore(cfg["data_dir"])

    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    fetch_spot_and_vix(adapter, store, start_dt, end_dt, cfg, force=force)
    fetch_options(
        adapter, store,
        underlying=cfg["universe"]["options_underlying"],
        start=start, end=end,
        strikes_around=int(cfg.get("strikes_around", 5)),
        resolution=cfg["resolution"],
        spot_symbol=cfg["universe"]["spot_symbol"],
        force=force,
    )
    log.info("Step 1 complete: data cached locally.")


# ----- Stage 2: health -----

def step_health(cfg: dict, start: date, end: date) -> bool:
    _banner("STEP 2: HEALTH CHECK")
    store = DataStore(cfg["data_dir"])
    spot_sym = cfg["universe"]["spot_symbol"]
    vix_sym = cfg["universe"]["vix_symbol"]
    underlying = cfg["universe"]["options_underlying"]

    spot_df = store.load_spot(spot_sym)
    if spot_df.empty:
        log.error("No spot data cached. Did Step 1 succeed?")
        return False

    spot_rep = report_spot(store, spot_sym)
    cal = trading_days_from_spot(spot_df)
    vix_rep = report_vix(store, vix_sym, cal)
    opt_rep = report_options(store, underlying,
                              strikes_around=int(cfg.get("strikes_around", 5)))

    print(f"[Spot] {spot_rep['trading_days_observed']} trading days, "
          f"{spot_rep['rows']} bars  ({spot_rep['first_day']}..{spot_rep['last_day']})")
    print(f"[VIX]  {vix_rep['trading_days_observed']} observations "
          f"(missing vs spot: {vix_rep.get('missing_vs_spot', 0)})")
    print(f"[Options] {len(opt_rep)} expiries cached")
    full = sum(1 for r in opt_rep if r["coverage_pct"] >= 95)
    partial = sum(1 for r in opt_rep if 0 < r["coverage_pct"] < 95)
    empty = sum(1 for r in opt_rep if r["coverage_pct"] == 0)
    print(f"  fully covered (>=95%) : {full}")
    print(f"  partially covered     : {partial}")
    print(f"  empty                 : {empty}")
    log.info("Step 2 complete.")
    return spot_rep["trading_days_observed"] > 0 and len(opt_rep) > 0


# ----- Stage 3: H4 backtest -----

def step_h4(cfg: dict, signals: list[str], tp_sweep: tuple[float, ...]
            ) -> list[dict]:
    _banner("STEP 3: H4 BACKTEST (intraday long-option buying)")
    store = DataStore(cfg["data_dir"])
    spot_df = store.load_spot(cfg["universe"]["spot_symbol"])
    underlying = cfg["universe"]["options_underlying"]
    lot_size = int(cfg.get("lot_sizes", {}).get(underlying, 75))

    results = []
    for sig in signals:
        for tp in tp_sweep:
            params = H4Params(lot_size=lot_size, tp_inr=tp)
            trades = h4_backtest(store, spot_df, underlying, sig, params)
            stats = h4_report(trades, params)
            print(h4_interpret(stats, sig, tp))
            results.append({"signal": sig, "tp_inr": tp, **stats})
            if not trades.empty:
                p = Path(cfg["data_dir"]) / (f"h4_{sig}_tp{int(tp)}"
                                              f"_trades.parquet")
                trades.to_parquet(p, index=False)

    df = pd.DataFrame(results)
    df.to_parquet(Path(cfg["data_dir"]) / "h4_summary.parquet", index=False)
    log.info("Step 3 complete.")
    return results


def _pick_winning_cell(h4_results: list[dict]) -> dict | None:
    """Return the (signal, tp) row with the highest annualized Sharpe among
    cells that pass H4 criteria. None if nothing passes."""
    passing = [r for r in h4_results if r.get("passes_h4")]
    if not passing:
        return None
    passing.sort(key=lambda r: r.get("sharpe_annualized", -999),
                 reverse=True)
    return passing[0]


# ----- Stage 4: paper -----

def step_paper(cfg: dict, signal: str, tp_inr: float,
                start: date | None, end: date | None,
                run_name: str | None = None) -> None:
    _banner(f"STEP 4: PAPER-TRADE REPLAY ({signal}, TP=Rs{int(tp_inr)})")
    store = DataStore(cfg["data_dir"])
    spot = store.load_spot(cfg["universe"]["spot_symbol"])
    underlying = cfg["universe"]["options_underlying"]
    lot_size = int(cfg.get("lot_sizes", {}).get(underlying, 75))

    run_name = run_name or f"{signal}_tp{int(tp_inr)}_{date.today().isoformat()}"
    run_dir = Path(cfg["data_dir"]) / "paper" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gate = RiskGate(
        config=RiskConfig(),
        state_path=run_dir / "risk_state.json",
        audit_path=run_dir / "risk_audit.jsonl",
    )
    gate.load()

    trader = PaperTrader(
        store=store, spot=spot, underlying=underlying,
        params=H4Params(lot_size=lot_size, tp_inr=tp_inr),
        signal=signal,
        risk_gate=gate, fill_engine=SimulatedFillEngine(),
        trades_path=run_dir / "trades.parquet",
    )
    trader.run_replay(start=start, end=end)

    snap = gate.snapshot()
    print(f"  paper trades closed     : {snap['closed_trades_total']}")
    print(f"  cumulative P&L (INR)    : {snap['cumulative_pnl_inr']:.2f}")
    print(f"  drawdown      (INR)     : {snap['drawdown_inr']:.2f}")
    print(f"  consecutive loss days   : {snap['consecutive_loss_days']}")
    print(f"  halted                  : {snap['halted']}"
          + (f"  ({snap['halt_reason']})" if snap['halted'] else ""))
    print(f"  trades parquet          : {trader.trades_path}")
    print(f"  risk audit log          : {run_dir / 'risk_audit.jsonl'}")
    log.info("Step 4 complete.")


# ----- CLI -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", help="YYYY-MM-DD (required unless --skip-fetch)")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-health", action="store_true")
    ap.add_argument("--skip-h4", action="store_true")
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Refetch even if cache covers the window")
    ap.add_argument("--signal", choices=["h4a", "h4b", "both"], default="both")
    ap.add_argument("--tp-inr", type=float, default=None,
                    help="Override TP; if unset, sweep "
                         f"{DEFAULT_TP_SWEEP}")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}. "
                         f"Copy config.example.yaml -> config.yaml first.")
    cfg = load_config(cfg_path)

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    # Step 1
    if not args.skip_fetch:
        if start is None or end is None:
            raise SystemExit("--start and --end required for fetch. "
                             "Use --skip-fetch to use existing cache.")
        step_fetch(cfg, start, end, force=args.force)

    # Step 2
    if not args.skip_health:
        ok = step_health(cfg, start or date(1970, 1, 1),
                          end or date.today())
        if not ok:
            log.error("Health check failed; halting pipeline.")
            sys.exit(2)

    # Step 3
    signals = (["h4a", "h4b"] if args.signal == "both" else [args.signal])
    tp_sweep = ((args.tp_inr,) if args.tp_inr is not None
                else DEFAULT_TP_SWEEP)
    h4_results = []
    if not args.skip_h4:
        h4_results = step_h4(cfg, signals, tp_sweep)

    # Step 4: paper-trade the winning cell if any
    if not args.skip_paper:
        if not h4_results:
            log.warning("No H4 results available — running paper with "
                        f"signal={signals[0]} TP={int(tp_sweep[0])} "
                        "as a fallback.")
            chosen_sig, chosen_tp = signals[0], tp_sweep[0]
        else:
            winner = _pick_winning_cell(h4_results)
            if winner is None:
                log.warning("No H4 cell passed criteria. Paper trading the "
                            "best Sharpe cell anyway for diagnostic value.")
                best = max(h4_results,
                           key=lambda r: r.get("sharpe_annualized", -999))
                chosen_sig, chosen_tp = best["signal"], best["tp_inr"]
            else:
                chosen_sig, chosen_tp = winner["signal"], winner["tp_inr"]
                log.info(f"Winning cell: {chosen_sig} TP=Rs{int(chosen_tp)} "
                         f"(Sharpe={winner.get('sharpe_annualized'):.2f})")
        step_paper(cfg, chosen_sig, chosen_tp, start, end)

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
