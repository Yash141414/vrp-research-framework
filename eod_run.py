"""
eod_run.py
----------
End-of-day runner. Run this once per trading day, after the market closes
(post 15:30 IST), to keep paper trading in sync with reality.

Steps performed:
  1. Refresh the broker access token (delegates to login_kite or
     login_upstox based on primary_broker in config). Interactive — paste
     the request_token / code when prompted.
  2. Fetch today's spot, VIX, and option bars (incremental — only fetches
     what's missing).
  3. For each configured signal (default h4a + h4b), replay today through
     the paper trader. Risk-gate state and audit log persist across runs.
  4. Print a summary table of today's P&L, cumulative P&L, drawdown,
     trade count, and halt status per signal.

IDEMPOTENT: re-running on a day already processed is a no-op (won't
double-count). Each signal's run directory keeps a `processed_dates.json`
marker file. Pass --force-replay to override (useful only if you know
why and have manually reset the risk state).

Usage:
  # Normal daily run (today)
  python -m nifty_data_layer.eod_run --config nifty_data_layer/config.yaml

  # Backfill a specific past trading day (assumes data is cached)
  python -m nifty_data_layer.eod_run --day 2026-05-15 --skip-login

  # Skip token refresh and fetch (everything already in place)
  python -m nifty_data_layer.eod_run --skip-login --skip-fetch

  # Schedule as cron at 19:00 IST on weekdays:
  #   0 19 * * 1-5  cd /path/to/Apology && python -m nifty_data_layer.eod_run >> eod.log 2>&1
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import yaml

from .config_loader import load_config
from .data_store import DataStore
from .risk_gate import RiskGate, RiskConfig
from .paper_trader import PaperTrader, SimulatedFillEngine
from .hypothesis_h4_intraday_buy import H4Params

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("eod_run")


# ----- Helpers -----

def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _today() -> date:
    return datetime.now().date()


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    try:
        return load_config(path)
    except ValueError as e:
        raise SystemExit(str(e))


# ----- Idempotency marker -----

def _load_processed(run_dir: Path) -> set[str]:
    p = run_dir / "processed_dates.json"
    if not p.exists():
        return set()
    try:
        with p.open() as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        log.warning(f"processed_dates.json at {p} is corrupt; treating as empty.")
        return set()


def _mark_processed(run_dir: Path, day: date) -> None:
    p = run_dir / "processed_dates.json"
    existing = _load_processed(run_dir)
    existing.add(day.isoformat())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(sorted(existing), f)
    tmp.replace(p)


# ----- Step 1: login -----

def _run_login(primary_broker: str, config_path: Path) -> bool:
    if primary_broker == "upstox":
        module = "nifty_data_layer.login_upstox"
    elif primary_broker == "kite":
        module = "nifty_data_layer.login_kite"
    else:
        log.warning(f"No login helper for primary_broker={primary_broker!r}. "
                    f"Skipping; assuming token is valid.")
        return True
    log.info(f"Refreshing access token via {module} (interactive)...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", module, "--config", str(config_path)]
        )
    except KeyboardInterrupt:
        log.info("Login cancelled by user.")
        return False
    if r.returncode != 0:
        log.error(f"{module} exited with code {r.returncode}.")
        return False
    return True


# ----- Step 2: fetch -----

def _run_fetch(config_path: Path, day: date) -> bool:
    log.info(f"Fetching market data for {day.isoformat()} (incremental)...")
    r = subprocess.run([
        sys.executable, "-m", "nifty_data_layer.fetch",
        "--config", str(config_path),
        "--start", day.isoformat(),
        "--end", day.isoformat(),
    ])
    if r.returncode != 0:
        log.error(f"fetch exited with code {r.returncode}.")
        return False
    return True


# ----- Step 3: paper replay per signal -----

def _replay_one_signal(cfg: dict, signal: str, tp_inr: float,
                       day: date, force: bool) -> dict:
    underlying = cfg["universe"]["options_underlying"]
    lot_size = int(cfg.get("lot_sizes", {}).get(underlying, 75))
    spot_sym = cfg["universe"]["spot_symbol"]

    run_name = f"{signal}_tp{int(tp_inr)}_eod"
    run_dir = Path(cfg["data_dir"]) / "paper" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    processed = _load_processed(run_dir)
    if not force and day.isoformat() in processed:
        log.info(f"[{signal}] {day} already processed; skipping. "
                 f"Use --force-replay to override.")
        # Return current cached state for the summary
        state_path = run_dir / "risk_state.json"
        if state_path.exists():
            with state_path.open() as f:
                return {"signal": signal, "skipped": True,
                         "state": json.load(f)}
        return {"signal": signal, "skipped": True, "state": {}}

    store = DataStore(cfg["data_dir"])
    spot = store.load_spot(spot_sym)
    if spot.empty:
        return {"signal": signal, "error": "no spot data cached"}

    gate = RiskGate(
        config=RiskConfig(),
        state_path=run_dir / "risk_state.json",
        audit_path=run_dir / "risk_audit.jsonl",
    )
    gate.load()
    if gate.state.halted:
        log.warning(f"[{signal}] risk gate is HALTED "
                    f"({gate.state.halt_reason}). Skipping replay. "
                    f"To resume, call risk_gate.reset_halt('reason') "
                    f"manually after reviewing the audit log.")
        return {"signal": signal, "halted": True,
                 "halt_reason": gate.state.halt_reason,
                 "state": gate.state.to_dict()}

    trader = PaperTrader(
        store=store, spot=spot, underlying=underlying,
        params=H4Params(lot_size=lot_size, tp_inr=tp_inr),
        signal=signal, risk_gate=gate,
        fill_engine=SimulatedFillEngine(),
        trades_path=run_dir / "trades.parquet",
    )
    trader.run_replay(start=day, end=day)
    _mark_processed(run_dir, day)
    return {"signal": signal, "skipped": False,
             "state": gate.snapshot()}


# ----- Step 4: summary -----

def _print_summary(day: date, summaries: list[dict]) -> None:
    _banner(f"DAILY SUMMARY  {day.isoformat()}")
    header = (f"  {'signal':10s} {'today P&L':>12s} {'cum P&L':>12s} "
              f"{'drawdown':>12s} {'trades':>8s} {'halted':>10s}")
    print(header)
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
    for s in summaries:
        sig = s["signal"]
        if s.get("error"):
            print(f"  {sig:10s}  ERROR: {s['error']}")
            continue
        st = s.get("state") or {}
        today_pnl = st.get("today_pnl_inr", 0.0)
        cum = st.get("cumulative_pnl_inr", 0.0)
        dd = st.get("drawdown_inr", st.get("current_drawdown_inr", 0.0))
        # current_drawdown is a property; if we got raw dict it may be absent
        if dd == 0.0 and "cum_peak_pnl_inr" in st:
            dd = st["cumulative_pnl_inr"] - st["cum_peak_pnl_inr"]
        trades = st.get("closed_trades_total",
                         len(st.get("closed_pnls", [])))
        halted = st.get("halted", False)
        halt_str = (f"YES({s.get('halt_reason') or st.get('halt_reason')})"
                    if halted else "no")
        skipped_tag = " (cached)" if s.get("skipped") else ""
        print(f"  {sig:10s} {today_pnl:>+12.2f} {cum:>+12.2f} "
              f"{dd:>+12.2f} {trades:>8d} {halt_str:>10s}{skipped_tag}")


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nifty_data_layer/config.yaml")
    ap.add_argument("--day", default=None,
                    help="YYYY-MM-DD to process (default: today).")
    ap.add_argument("--signals", default="h4a,h4b",
                    help="Comma-separated list of signals to replay.")
    ap.add_argument("--tp-inr", type=float, default=1500.0,
                    help="TP per trade in INR. Default 1500.")
    ap.add_argument("--skip-login", action="store_true",
                    help="Skip token refresh.")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Skip data fetch.")
    ap.add_argument("--skip-replay", action="store_true",
                    help="Skip paper replay (just login + fetch).")
    ap.add_argument("--force-replay", action="store_true",
                    help="Re-replay even if today is marked processed.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)

    day = (date.fromisoformat(args.day) if args.day else _today())
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    _banner(f"EOD RUN  {day.isoformat()}  signals={signals}  "
            f"TP=Rs{int(args.tp_inr)}")

    if not _is_weekday(day):
        log.info(f"{day.isoformat()} is a weekend. Nothing to do.")
        return

    # Step 1: login
    if not args.skip_login:
        primary = cfg.get("primary_broker", "upstox")
        if not _run_login(primary, cfg_path):
            log.error("Login failed; aborting EOD run.")
            sys.exit(1)
    else:
        log.info("Skipping login (--skip-login).")

    # Step 2: fetch
    if not args.skip_fetch:
        if not _run_fetch(cfg_path, day):
            log.error("Fetch failed; aborting EOD run.")
            sys.exit(1)
    else:
        log.info("Skipping fetch (--skip-fetch).")

    # Step 3: paper replay per signal
    summaries: list[dict] = []
    if not args.skip_replay:
        for sig in signals:
            _banner(f"PAPER REPLAY  {sig}  TP=Rs{int(args.tp_inr)}")
            summaries.append(_replay_one_signal(
                cfg, sig, args.tp_inr, day, args.force_replay,
            ))
    else:
        log.info("Skipping replay (--skip-replay).")

    # Step 4: summary
    if summaries:
        _print_summary(day, summaries)


if __name__ == "__main__":
    main()
