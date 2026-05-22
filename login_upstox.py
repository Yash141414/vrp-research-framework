"""
login_upstox.py
---------------
Daily Upstox access-token refresh via OAuth2. Equivalent to login_kite.py
but for Upstox. Tokens expire roughly daily; run this each morning before
fetch / paper / live.

Flow (interactive):
  1. Reads api_key + api_secret + redirect_uri from config.yaml.
  2. Prints the OAuth dialog URL and waits for you to paste the `code`
     from the redirect URL after you authorize.
  3. Exchanges the code for an access_token via Upstox V2 API.
  4. Writes the new access_token back to config.yaml.

Run:  python -m nifty_data_layer.login_upstox --config config.yaml
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import yaml
import requests

from .config_loader import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("login_upstox")


UPSTOX_BASE = "https://api.upstox.com/v2"


def _extract_code(input_str: str) -> str:
    """Accept either a raw `code` or a full redirect URL containing it."""
    s = input_str.strip()
    if "code=" in s:
        q = parse_qs(urlparse(s).query)
        if "code" in q and q["code"]:
            return q["code"][0]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--code", default=None,
                    help="If provided, skip the interactive prompt and use "
                         "this value (or a full redirect URL containing it).")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found at {cfg_path}. "
                         f"Copy config.example.yaml -> config.yaml and "
                         f"fill in your Upstox api_key/api_secret first.")

    cfg = load_config(cfg_path)

    uc = cfg.get("upstox") or {}
    api_key = uc.get("api_key")
    api_secret = uc.get("api_secret")
    redirect_uri = uc.get("redirect_uri")
    if not api_key or api_key.startswith("YOUR_") or not api_key.strip():
        raise SystemExit("config.yaml is missing upstox.api_key.")
    if not api_secret or api_secret.startswith("YOUR_") or not api_secret.strip():
        raise SystemExit("config.yaml is missing upstox.api_secret.")
    if not redirect_uri:
        raise SystemExit("config.yaml is missing upstox.redirect_uri.")

    oauth_url = (
        f"{UPSTOX_BASE}/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={api_key}"
        f"&redirect_uri={redirect_uri}"
    )

    if args.code:
        code = _extract_code(args.code)
    else:
        print("\n[Upstox Login]")
        print("1. Open this URL in your browser and authorize:")
        print(f"   {oauth_url}")
        print("2. After login, Upstox redirects to your registered "
              "redirect_uri.")
        print("   Copy the FULL redirect URL or just the `code=...` value.")
        raw = input("Paste code (or full redirect URL): ").strip()
        if not raw:
            raise SystemExit("empty input; aborting.")
        code = _extract_code(raw)

    log.info(f"Exchanging code (length {len(code)}) for access_token...")
    try:
        r = requests.post(
            f"{UPSTOX_BASE}/login/authorization/token",
            data={
                "code": code,
                "client_id": api_key,
                "client_secret": api_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
    except requests.RequestException as e:
        raise SystemExit(f"Network error contacting Upstox: {e}")

    if r.status_code != 200:
        raise SystemExit(f"Upstox token exchange failed (HTTP {r.status_code}): "
                         f"{r.text}")

    payload = r.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise SystemExit(f"No access_token in Upstox response: {payload}")

    # Write back. pyyaml strips comments — user keeps the commented version
    # in config.example.yaml for reference.
    cfg["upstox"]["access_token"] = access_token
    tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    tmp.replace(cfg_path)

    log.info(f"access_token updated in {cfg_path}")
    print(f"\nLogin successful. User: {payload.get('user_name', '(unknown)')}.")
    print("Token is valid until tomorrow morning IST.")


if __name__ == "__main__":
    main()
