"""
verify_config.py
----------------
Pre-flight sanity check for config.yaml. Run this BEFORE the daily
login_kite / pipeline runs so typos and missing fields fail fast (1
second) instead of after a partial fetch (hours).

Checks (no broker calls by default):
  - YAML parses
  - primary_broker is one of {kite, upstox, angelone}
  - api_key/api_secret present and not placeholder strings
  - data_dir is a writable absolute or relative path
  - universe.spot_symbol, options_underlying, vix_symbol present
  - resolution is supported
  - date_range.start <= date_range.end (if present)
  - lot_sizes[underlying] is a positive int
  - broker SDK import works
  - (with --probe) tries a single broker auth call to verify access_token

Run:
  python -m nifty_data_layer.verify_config --config config.yaml
  python -m nifty_data_layer.verify_config --config config.yaml --probe
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date
from pathlib import Path
import yaml

from .config_loader import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verify_config")

SUPPORTED_BROKERS = {"kite", "upstox", "angelone"}
SUPPORTED_RESOLUTIONS = {"1minute", "5minute", "15minute", "day"}


class Check:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def report_and_exit(self) -> None:
        print()
        if self.warnings:
            print("WARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")
            print()
        if self.errors:
            print("ERRORS:")
            for e in self.errors:
                print(f"  - {e}")
            print()
            print(f"FAIL ({len(self.errors)} errors, "
                  f"{len(self.warnings)} warnings)")
            sys.exit(1)
        print(f"OK ({len(self.warnings)} warnings, 0 errors)")
        sys.exit(0)


def _is_placeholder(s) -> bool:
    if not isinstance(s, str):
        return True
    return (not s) or s.startswith("YOUR_") or s.strip() == ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--probe", action="store_true",
                    help="Actually call the broker to verify access_token "
                         "works (1 cheap API call).")
    args = ap.parse_args()

    chk = Check()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        chk.err(f"config not found: {cfg_path}. "
                f"Copy config.example.yaml -> {cfg_path.name} and fill in "
                f"your broker credentials.")
        chk.report_and_exit()

    try:
        cfg = load_config(cfg_path)
    except yaml.YAMLError as e:
        chk.err(f"YAML parse error: {e}")
        chk.report_and_exit()
    except (FileNotFoundError, ValueError) as e:
        chk.err(str(e))
        chk.report_and_exit()

    if not isinstance(cfg, dict):
        chk.err("Top-level config must be a mapping/dict.")
        chk.report_and_exit()

    # primary_broker
    primary = cfg.get("primary_broker")
    if primary not in SUPPORTED_BROKERS:
        chk.err(f"primary_broker must be one of {SUPPORTED_BROKERS}, "
                f"got {primary!r}")

    # Per-broker creds
    if primary == "kite":
        kc = cfg.get("kite") or {}
        if _is_placeholder(kc.get("api_key")):
            chk.err("kite.api_key is missing or still a placeholder.")
        if _is_placeholder(kc.get("api_secret")):
            chk.err("kite.api_secret is missing or still a placeholder.")
        if _is_placeholder(kc.get("access_token")):
            chk.warn("kite.access_token is empty. Run "
                     "`python -m nifty_data_layer.login_kite` to generate one.")
    elif primary == "upstox":
        uc = cfg.get("upstox") or {}
        if _is_placeholder(uc.get("api_key")):
            chk.err("upstox.api_key is missing or still a placeholder.")
        if _is_placeholder(uc.get("api_secret")):
            chk.err("upstox.api_secret is missing or still a placeholder.")
        if _is_placeholder(uc.get("redirect_uri")):
            chk.err("upstox.redirect_uri is missing.")
        if _is_placeholder(uc.get("access_token")):
            chk.warn("upstox.access_token is empty.")
    elif primary == "angelone":
        chk.err("angelone adapter is not implemented yet; pick kite or upstox.")

    # data_dir
    dd = cfg.get("data_dir")
    if not dd:
        chk.err("data_dir is missing.")
    else:
        p = Path(dd)
        try:
            p.mkdir(parents=True, exist_ok=True)
            # Smoke-test writability
            t = p / ".verify_write_test"
            t.write_text("ok"); t.unlink()
        except OSError as e:
            chk.err(f"data_dir {dd} is not writable: {e}")

    # universe
    uni = cfg.get("universe") or {}
    for key in ("spot_symbol", "options_underlying", "vix_symbol"):
        if not uni.get(key):
            chk.err(f"universe.{key} is missing.")

    # resolution
    res = cfg.get("resolution")
    if res not in SUPPORTED_RESOLUTIONS:
        chk.err(f"resolution must be one of {SUPPORTED_RESOLUTIONS}, "
                f"got {res!r}")

    # date_range
    dr = cfg.get("date_range") or {}
    s = dr.get("start"); e = dr.get("end")
    if s and e:
        try:
            ds = date.fromisoformat(str(s))
            de = date.fromisoformat(str(e))
            if ds > de:
                chk.err(f"date_range.start ({s}) is after end ({e}).")
        except ValueError as exc:
            chk.err(f"date_range parse error: {exc}")

    # lot_sizes
    underlying = uni.get("options_underlying")
    ls = cfg.get("lot_sizes") or {}
    if underlying and underlying not in ls:
        chk.err(f"lot_sizes.{underlying} is missing. NSE lot for NIFTY is "
                f"currently 75 (verify against your broker).")
    elif underlying and not isinstance(ls.get(underlying), int):
        chk.err(f"lot_sizes.{underlying} must be a positive int.")
    elif underlying and ls.get(underlying) <= 0:
        chk.err(f"lot_sizes.{underlying} must be positive.")

    # min_volume_per_leg
    mv = cfg.get("min_volume_per_leg", 0)
    if not isinstance(mv, int) or mv < 0:
        chk.err("min_volume_per_leg must be a non-negative int.")

    # Broker SDK import
    if primary == "kite":
        try:
            import kiteconnect  # noqa: F401
        except ImportError:
            chk.warn("kiteconnect not installed; "
                     "`pip install kiteconnect` before fetch.")
    elif primary == "upstox":
        try:
            import requests  # noqa: F401
        except ImportError:
            chk.warn("requests not installed (needed by upstox adapter).")

    # Live probe (optional)
    if args.probe and not chk.errors:
        log.info("Probing broker auth...")
        try:
            if primary == "kite":
                from .brokers.kite import KiteAdapter
                ad = KiteAdapter(
                    api_key=cfg["kite"]["api_key"],
                    api_secret=cfg["kite"]["api_secret"],
                    access_token=cfg["kite"].get("access_token") or None,
                )
                ad.login()
                log.info("Kite probe OK.")
            elif primary == "upstox":
                from .brokers.upstox import UpstoxAdapter
                ad = UpstoxAdapter(
                    api_key=cfg["upstox"]["api_key"],
                    api_secret=cfg["upstox"]["api_secret"],
                    redirect_uri=cfg["upstox"]["redirect_uri"],
                    access_token=cfg["upstox"].get("access_token") or None,
                )
                ad.login()
                log.info("Upstox probe OK.")
        except Exception as exc:
            chk.err(f"Broker auth probe failed: {exc}")

    chk.report_and_exit()


if __name__ == "__main__":
    main()
