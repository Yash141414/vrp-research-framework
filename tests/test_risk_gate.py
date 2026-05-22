"""Tests for the risk gate: pre-trade checks + kill-switch state machine."""
from __future__ import annotations
from datetime import datetime, date, time
from pathlib import Path
import json
import pytest

from Apology.Proj.Nifty_momentum_system.nifty_data_layer.risk_gate import RiskGate, RiskConfig, RiskState


def _fresh_gate(tmp_path: Path, **config_overrides) -> RiskGate:
    cfg = RiskConfig(**config_overrides)
    gate = RiskGate(
        config=cfg,
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
    )
    gate.load()
    return gate


def test_can_open_allows_fresh_morning(tmp_path):
    gate = _fresh_gate(tmp_path)
    now = datetime(2026, 5, 18, 9, 20)
    ok, reason = gate.can_open(now)
    assert ok is True
    assert reason == "ok"


def test_can_open_denies_after_entry_cutoff(tmp_path):
    gate = _fresh_gate(tmp_path)
    now = datetime(2026, 5, 18, 14, 35)
    ok, reason = gate.can_open(now)
    assert ok is False
    assert "entry_cutoff" in reason


def test_can_open_denies_when_position_open(tmp_path):
    gate = _fresh_gate(tmp_path)
    now = datetime(2026, 5, 18, 9, 20)
    ok, reason = gate.can_open(now, position_open=True)
    assert ok is False
    assert reason == "position_already_open"


def test_can_open_denies_after_one_trade_today(tmp_path):
    gate = _fresh_gate(tmp_path)
    now = datetime(2026, 5, 18, 9, 20)
    gate.can_open(now)
    gate.on_position_opened({"signal": "h4a"})
    gate.on_trade_closed(realized_inr=200.0, meta={"signal": "h4a"})
    # Try second trade same day
    ok, reason = gate.can_open(datetime(2026, 5, 18, 12, 0))
    assert ok is False
    assert reason == "daily_trade_limit_reached"


def test_can_open_denies_when_daily_loss_limit_hit(tmp_path):
    gate = _fresh_gate(tmp_path)
    # Pretend we already lost the daily max in a prior trade today
    gate._roll_to_day(date(2026, 5, 18))
    gate.state.today_pnl_inr = -1500.0
    gate.state.today_trades_opened = 0   # didn't actually open trade
    gate._persist()
    ok, reason = gate.can_open(datetime(2026, 5, 18, 9, 20))
    assert ok is False
    assert reason == "daily_loss_limit_reached"


def test_on_position_tick_flatten_at_single_trade_sl(tmp_path):
    gate = _fresh_gate(tmp_path)
    # 5% of Rs20k = Rs1000
    action = gate.on_position_tick(unrealized_inr=-1001.0)
    assert action == "flatten_single_trade_sl"
    action = gate.on_position_tick(unrealized_inr=-999.0)
    assert action == "hold"


def test_cumulative_drawdown_halts_strategy(tmp_path):
    gate = _fresh_gate(tmp_path)
    gate.can_open(datetime(2026, 5, 18, 9, 20))
    gate.on_position_opened({"trade": 1})
    # Big single loss > cumulative DD limit
    gate.on_trade_closed(realized_inr=-4100.0, meta={"trade": 1})
    assert gate.state.halted is True
    assert "cum_drawdown" in (gate.state.halt_reason or "")
    # Future can_open is blocked
    ok, reason = gate.can_open(datetime(2026, 5, 19, 9, 20))
    assert ok is False
    assert reason.startswith("halted:")


def test_three_consecutive_loss_days_halts(tmp_path):
    gate = _fresh_gate(tmp_path)
    for i, d in enumerate([date(2026, 5, 18), date(2026, 5, 19),
                            date(2026, 5, 20)]):
        gate.can_open(datetime(d.year, d.month, d.day, 9, 20))
        gate.on_position_opened({"day": d.isoformat()})
        gate.on_trade_closed(realized_inr=-500.0, meta={"day": d.isoformat()})
        gate.on_day_end(d)
    # After three loss days, next day should be halted
    assert gate.state.halted is True
    assert "consecutive_loss_days" in (gate.state.halt_reason or "")


def test_consecutive_loss_resets_on_winning_day(tmp_path):
    gate = _fresh_gate(tmp_path)
    # Two loss days
    for d in [date(2026, 5, 18), date(2026, 5, 19)]:
        gate.can_open(datetime(d.year, d.month, d.day, 9, 20))
        gate.on_position_opened({"day": d.isoformat()})
        gate.on_trade_closed(realized_inr=-300.0, meta={"day": d.isoformat()})
        gate.on_day_end(d)
    # Then a winning day
    d = date(2026, 5, 20)
    gate.can_open(datetime(d.year, d.month, d.day, 9, 20))
    gate.on_position_opened({"day": d.isoformat()})
    gate.on_trade_closed(realized_inr=500.0, meta={"day": d.isoformat()})
    gate.on_day_end(d)
    assert gate.state.consecutive_loss_days == 0
    assert gate.state.halted is False


def test_ttest_halt_on_persistent_losses(tmp_path):
    # Disable other halt paths so we isolate the t-test trigger.
    gate = _fresh_gate(tmp_path, ttest_every_n_trades=20,
                       ttest_p_threshold=0.30,
                       cumulative_drawdown_inr=1e9,
                       max_consecutive_loss_days=999)
    # 20 trades whose distribution has mean<0 with std>0 (varied losses).
    losses = [-100, -150, -80, -120, -90, -110, -130, -70, -140, -60,
              -100, -120, -90, -110, -80, -150, -70, -130, -100, -90]
    for i, p in enumerate(losses):
        d = date(2026, 5, 1) + (date(2026, 5, 2) - date(2026, 5, 1)) * i
        gate.can_open(datetime(d.year, d.month, d.day, 9, 20))
        gate.on_position_opened({"i": i})
        gate.on_trade_closed(realized_inr=float(p), meta={"i": i})
        gate.on_day_end(d)
    assert gate.state.halted is True
    assert "ttest" in (gate.state.halt_reason or "")


def test_persistence_roundtrip(tmp_path):
    gate = _fresh_gate(tmp_path)
    gate.can_open(datetime(2026, 5, 18, 9, 20))
    gate.on_position_opened({"x": 1})
    gate.on_trade_closed(realized_inr=250.0, meta={"x": 1})
    gate.on_day_end(date(2026, 5, 18))

    # Recreate
    gate2 = RiskGate(
        config=RiskConfig(),
        state_path=tmp_path / "state.json",
        audit_path=tmp_path / "audit.jsonl",
    )
    gate2.load()
    assert gate2.state.cumulative_pnl_inr == pytest.approx(250.0)
    assert gate2.state.consecutive_loss_days == 0


def test_reset_halt_clears_state(tmp_path):
    gate = _fresh_gate(tmp_path)
    gate._halt("test_reason")
    assert gate.state.halted is True
    gate.reset_halt("manual review complete")
    assert gate.state.halted is False
    assert gate.state.halt_reason is None
    assert gate.state.consecutive_loss_days == 0


def test_audit_log_records_events(tmp_path):
    gate = _fresh_gate(tmp_path)
    gate.can_open(datetime(2026, 5, 18, 9, 20))
    gate.on_position_opened({"signal": "h4a"})
    gate.on_trade_closed(realized_inr=100.0, meta={"signal": "h4a"})
    gate.on_day_end(date(2026, 5, 18))

    audit_lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    events = [json.loads(line)["event"] for line in audit_lines]
    assert "position_opened" in events
    assert "trade_closed" in events
    assert "day_closed" in events
