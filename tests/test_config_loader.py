"""Tests for config_loader: YAML -> .env -> os.environ precedence."""
from __future__ import annotations
import os
from pathlib import Path
import yaml
import pytest

from Apology.Proj.Nifty_momentum_system.nifty_data_layer.config_loader import (
    load_config, _parse_dotenv, ENV_OVERRIDES,
)


# ----- .env parsing -----

def test_parse_dotenv_basic(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nKITE_API_KEY=abc123\n\nUPSTOX_API_KEY=xyz\n")
    got = _parse_dotenv(p)
    assert got == {"KITE_API_KEY": "abc123", "UPSTOX_API_KEY": "xyz"}


def test_parse_dotenv_strips_quotes(tmp_path):
    p = tmp_path / ".env"
    p.write_text('KITE_API_KEY="quoted_val"\nUPSTOX_API_KEY=\'single_q\'\n')
    got = _parse_dotenv(p)
    assert got["KITE_API_KEY"] == "quoted_val"
    assert got["UPSTOX_API_KEY"] == "single_q"


def test_parse_dotenv_missing_file_is_empty(tmp_path):
    assert _parse_dotenv(tmp_path / "does_not_exist") == {}


def test_parse_dotenv_ignores_garbage_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text("not a kv pair\n=no_key\nKITE_API_KEY=ok\n#KITE_API_SECRET=skip\n")
    got = _parse_dotenv(p)
    assert got == {"KITE_API_KEY": "ok"}


# ----- load_config precedence -----

def _write_cfg(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        yaml.safe_dump(payload, f)


def test_load_config_yaml_only(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"kite": {"api_key": "from_yaml"}})
    cfg = load_config(cfg_path)
    assert cfg["kite"]["api_key"] == "from_yaml"


def test_load_config_dotenv_overrides_yaml(tmp_path, monkeypatch):
    # Wipe any process env vars that would beat .env
    for k in ENV_OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"kite": {"api_key": "from_yaml"}})
    (tmp_path / ".env").write_text("KITE_API_KEY=from_dotenv\n")
    cfg = load_config(cfg_path)
    assert cfg["kite"]["api_key"] == "from_dotenv"


def test_load_config_processenv_overrides_dotenv(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"kite": {"api_key": "from_yaml"}})
    (tmp_path / ".env").write_text("KITE_API_KEY=from_dotenv\n")
    monkeypatch.setenv("KITE_API_KEY", "from_processenv")
    cfg = load_config(cfg_path)
    assert cfg["kite"]["api_key"] == "from_processenv"


def test_load_config_creates_section_if_missing(tmp_path, monkeypatch):
    for k in ENV_OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"primary_broker": "upstox"})  # no upstox section
    (tmp_path / ".env").write_text("UPSTOX_API_KEY=k\nUPSTOX_API_SECRET=s\n")
    cfg = load_config(cfg_path)
    assert cfg["upstox"]["api_key"] == "k"
    assert cfg["upstox"]["api_secret"] == "s"


def test_load_config_blank_env_falls_back_to_yaml(tmp_path, monkeypatch):
    for k in ENV_OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"kite": {"api_key": "from_yaml"}})
    (tmp_path / ".env").write_text("KITE_API_KEY=\n")   # blank value
    cfg = load_config(cfg_path)
    assert cfg["kite"]["api_key"] == "from_yaml"


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_non_mapping_raises(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("- just a list\n- of things\n")
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_load_config_explicit_dotenv_path(tmp_path, monkeypatch):
    for k in ENV_OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    cfg_path = tmp_path / "config.yaml"
    _write_cfg(cfg_path, {"kite": {"api_key": "from_yaml"}})
    alt_env = tmp_path / "custom.env"
    alt_env.write_text("KITE_API_KEY=from_explicit\n")
    cfg = load_config(cfg_path, dotenv_path=alt_env)
    assert cfg["kite"]["api_key"] == "from_explicit"
