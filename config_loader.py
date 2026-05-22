"""
config_loader.py
----------------
Single entrypoint that loads config.yaml with optional .env / process-env
overrides for sensitive fields (broker credentials).

Resolution order (later beats earlier):
  1. config.yaml
  2. .env file (parsed inline; no python-dotenv dependency)
  3. process environment variables

The .env file is searched in this order:
  - explicit `dotenv_path` argument
  - same directory as config.yaml
  - parent directory (repo root, when config lives in a package dir)
  - current working directory

Env vars that override config (all optional):
  KITE_API_KEY        -> kite.api_key
  KITE_API_SECRET     -> kite.api_secret
  KITE_ACCESS_TOKEN   -> kite.access_token
  UPSTOX_API_KEY      -> upstox.api_key
  UPSTOX_API_SECRET   -> upstox.api_secret
  UPSTOX_REDIRECT_URI -> upstox.redirect_uri
  UPSTOX_ACCESS_TOKEN -> upstox.access_token

Login helpers (login_kite/login_upstox) still write tokens back to
config.yaml because that's the simplest place to persist them. If you
prefer to keep tokens in .env only, move them manually after login.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import yaml


ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "KITE_API_KEY":         ("kite",   "api_key"),
    "KITE_API_SECRET":      ("kite",   "api_secret"),
    "KITE_ACCESS_TOKEN":    ("kite",   "access_token"),
    "UPSTOX_API_KEY":       ("upstox", "api_key"),
    "UPSTOX_API_SECRET":    ("upstox", "api_secret"),
    "UPSTOX_REDIRECT_URI":  ("upstox", "redirect_uri"),
    "UPSTOX_ACCESS_TOKEN":  ("upstox", "access_token"),
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Tiny .env parser: KEY=VALUE per line, # comments, blank lines ignored.
    Strips surrounding single/double quotes. Returns {} if file is missing."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes if symmetric
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def _find_dotenv(cfg_path: Path,
                 explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    candidates = [
        cfg_path.parent / ".env",
        cfg_path.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_config(path: str | Path,
                dotenv_path: Optional[str | Path] = None) -> dict:
    """Load YAML, layer .env on top, then os.environ on top. Returns dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} is not a valid YAML mapping")

    # Layer 2: .env file
    dotenv = _find_dotenv(cfg_path,
                          Path(dotenv_path) if dotenv_path else None)
    dotenv_dict = _parse_dotenv(dotenv) if dotenv else {}
    for env_key, (section, field) in ENV_OVERRIDES.items():
        v = dotenv_dict.get(env_key)
        if v:
            cfg.setdefault(section, {})[field] = v

    # Layer 3: process environment (highest precedence)
    for env_key, (section, field) in ENV_OVERRIDES.items():
        v = os.environ.get(env_key)
        if v:
            cfg.setdefault(section, {})[field] = v

    return cfg
