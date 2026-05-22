"""
login_kite.py
-------------
Daily Kite access-token refresh. Kite tokens expire at ~06:00 IST each
morning; this script automates the request_token -> access_token exchange
and writes the new token back to config.yaml.

Flow (interactive, run once per day before fetch / paper / live):
  1. Reads api_key + api_secret from config.yaml.
  2. Prints the login URL and waits for you to paste the request_token
     from the redirect URL after you authorize.
  3. Calls Kite to exchange the request_token for an access_token.
  4. Writes the new access_token back to config.yaml (preserves all other
     fields and YAML comments are preserved with ruamel if installed,
     otherwise pyyaml round-trip).

Run:  python -m nifty_data_layer.login_kite --config config.yaml
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import yaml

from .config_loader import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("login_kite")


def _extract_request_token(input_str: str) -> str:
    """Accept either a raw request_token or a full redirect URL containing it."""
    s = input_str.strip()
    if "request_token=" in s:
        # e.g. https://localhost/?action=login&type=login&status=success&request_token=ABC123
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(s).query)
        if "request_token" in q and q["request_token"]:
            return q["request_token"][0]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--request-token", default=None,
                    help="If provided, skip the interactive prompt and use "
                         "this value (or a full redirect URL containing it)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found at {cfg_path}. "
                         f"Copy config.example.yaml -> config.yaml and "
                         f"fill in your Kite api_key/api_secret first.")

    cfg = load_config(cfg_path)

    kite_cfg = cfg.get("kite") or {}
    api_key = kite_cfg.get("api_key")
    api_secret = kite_cfg.get("api_secret")
    if not api_key or not api_secret or api_key.startswith("YOUR_"):
        raise SystemExit("config.yaml is missing valid kite.api_key / "
                         "kite.api_secret. Fill those before logging in.")

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise SystemExit("kiteconnect not installed. "
                         "pip install kiteconnect")

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    if args.request_token:
        rt = _extract_request_token(args.request_token)
    else:
        print("\n[Kite Login]")
        print("1. Open this URL in your browser and authorize:")
        print(f"   {login_url}")
        print("2. After login, Kite redirects to your registered redirect URL.")
        print("   Copy the FULL redirect URL or just the request_token value.")
        raw = input("Paste request_token (or full redirect URL): ").strip()
        if not raw:
            raise SystemExit("empty input; aborting.")
        rt = _extract_request_token(raw)

    log.info(f"Exchanging request_token (length {len(rt)}) for access_token...")
    try:
        data = kite.generate_session(rt, api_secret=api_secret)
    except Exception as e:
        raise SystemExit(f"Kite session generation failed: {e}")

    access_token = data.get("access_token")
    if not access_token:
        raise SystemExit(f"No access_token in Kite response: {data}")

    # Write back to config.yaml. pyyaml doesn't preserve comments, but the
    # user can always restore them from config.example.yaml. Acceptable
    # tradeoff vs adding a ruamel.yaml dependency.
    cfg["kite"]["access_token"] = access_token
    tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    tmp.replace(cfg_path)

    log.info(f"access_token updated in {cfg_path}")
    print("\nLogin successful. Token is valid until ~06:00 IST tomorrow.")
    print(f"User: {data.get('user_id')}  Name: {data.get('user_name')}")


if __name__ == "__main__":
    main()
