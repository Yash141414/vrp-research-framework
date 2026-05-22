"""
risk_gate.py
------------
Pre-trade risk checks, kill-switch state machine, and audit log.
Used by both the paper trader and (eventually) the live trader. Same code
path, same gates — the only thing that changes is what fills your orders.

Capital and thresholds are tuned for Rs 20,000 starting capital, 1 lot at
a time, max 1 trade per day (H4 spec). All thresholds are CONFIGURABLE
via RiskConfig — change them via config.yaml, not by editing this file.

The gate is a STATE MACHINE persisted to a JSON file on every state
change. This survives process restarts: if the bot crashes at 14:45, the
next run sees the same halted state, same cumulative DD, same consecutive
loss days. No silent state loss.

Decisions:
  can_open()         -> (allowed: bool, reason: str)
  on_position_tick() -> action: str ("hold" | "flatten_single_trade_sl")
  on_trade_closed()  -> mutates state; may flip halted=True
  on_day_end()       -> rolls daily counters; may set halted=True for
                        consecutive-loss-days

A halt is STICKY: once halted, no new trades open until the operator
calls reset_halt(reason) which is itself audit-logged.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
from scipy import stats

log = logging.getLogger("risk_gate")


# ----- Config -----

@dataclass
class RiskConfig:
    """Capital-tuned thresholds. Defaults are the H4 spec for Rs 20k."""
    capital_inr: float = 20_000.0
    single_trade_max_loss_inr: float = 1_000.0   # flattens that position
    daily_max_loss_inr: float = 1_500.0          # halts for the day
    max_consecutive_loss_days: int = 3           # halts for review
    cumulative_drawdown_inr: float = 4_000.0     # halts strategy
    entry_cutoff_time: time = time(14, 30)       # no new entries after
    ttest_every_n_trades: int = 20               # sample-size health check
    ttest_p_threshold: float = 0.10              # halt if Sharpe<0 at p<this


# ----- State -----

@dataclass
class RiskState:
    """Persisted to disk on every mutation."""
    cumulative_pnl_inr: float = 0.0           # since first trade
    cum_peak_pnl_inr: float = 0.0             # for drawdown calc
    today_date: Optional[str] = None          # ISO date
    today_pnl_inr: float = 0.0
    today_trades_opened: int = 0
    today_trades_closed: int = 0
    consecutive_loss_days: int = 0
    last_closed_day_pnl: Optional[float] = None
    last_closed_day_date: Optional[str] = None
    halted: bool = False
    halt_reason: Optional[str] = None
    halt_ts: Optional[str] = None
    closed_pnls: list[float] = field(default_factory=list)   # for t-test

    @property
    def current_drawdown_inr(self) -> float:
        return self.cumulative_pnl_inr - self.cum_peak_pnl_inr  # <= 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(**d)


# ----- Audit log -----

class AuditLog:
    """Append-only JSONL log. Every decision and state change is recorded.
    Cheap, grep-able, and survives process death."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            **payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ----- Risk gate -----

class RiskGate:
    """
    Stateful pre-trade and intra-trade risk enforcement.

    Lifecycle:
        gate = RiskGate(config, state_path, audit_path)
        gate.load()
        ...
        ok, reason = gate.can_open(now)
        if ok: enter trade
        while position open:
            action = gate.on_position_tick(unrealized_inr)
            if action == "flatten_single_trade_sl": exit
        gate.on_trade_closed(realized_inr, trade_meta)
        ...
        gate.on_day_end(today)
    """

    def __init__(self, config: RiskConfig, state_path: Path,
                 audit_path: Path):
        self.config = config
        self.state_path = Path(state_path)
        self.audit = AuditLog(audit_path)
        self.state = RiskState()

    # ----- Persistence -----

    def load(self) -> None:
        if self.state_path.exists():
            with self.state_path.open() as f:
                self.state = RiskState.from_dict(json.load(f))
            log.info(f"Loaded risk state: halted={self.state.halted}, "
                     f"cum_pnl={self.state.cumulative_pnl_inr:.2f}, "
                     f"dd={self.state.current_drawdown_inr:.2f}")
        else:
            log.info("No prior risk state; starting fresh.")
            self.state = RiskState()
            self._persist()

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.state.to_dict(), f, default=str, indent=2)
        tmp.replace(self.state_path)

    # ----- Internal helpers -----

    def _roll_to_day(self, today: date) -> None:
        """Switch today's-state to a new calendar day. Idempotent."""
        today_iso = today.isoformat()
        if self.state.today_date == today_iso:
            return
        # If we're closing a day with trades, count it for consecutive-loss tracking
        if (self.state.today_date is not None and
                self.state.today_trades_closed > 0):
            self.state.last_closed_day_pnl = self.state.today_pnl_inr
            self.state.last_closed_day_date = self.state.today_date
            if self.state.today_pnl_inr < 0:
                self.state.consecutive_loss_days += 1
            else:
                self.state.consecutive_loss_days = 0
            self.audit.write("day_closed", {
                "date": self.state.today_date,
                "pnl_inr": self.state.today_pnl_inr,
                "trades": self.state.today_trades_closed,
                "consecutive_loss_days": self.state.consecutive_loss_days,
            })
            # Consecutive-loss-days halt
            if (self.state.consecutive_loss_days >=
                    self.config.max_consecutive_loss_days and
                    not self.state.halted):
                self._halt(f"consecutive_loss_days="
                           f"{self.state.consecutive_loss_days}")
        # Reset daily counters
        self.state.today_date = today_iso
        self.state.today_pnl_inr = 0.0
        self.state.today_trades_opened = 0
        self.state.today_trades_closed = 0
        self._persist()

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_ts = datetime.now().isoformat()
        self.audit.write("halted", {"reason": reason,
                                     "cum_pnl": self.state.cumulative_pnl_inr,
                                     "drawdown": self.state.current_drawdown_inr})
        log.warning(f"RISK HALT: {reason}")

    # ----- Public API -----

    def can_open(self, now: datetime,
                 position_open: bool = False) -> tuple[bool, str]:
        """Return (allowed, reason). Reason is informative either way."""
        self._roll_to_day(now.date())
        if self.state.halted:
            return False, f"halted:{self.state.halt_reason}"
        if position_open:
            return False, "position_already_open"
        if now.time() >= self.config.entry_cutoff_time:
            return False, f"entry_cutoff_{self.config.entry_cutoff_time}"
        if (self.state.today_pnl_inr <= -self.config.daily_max_loss_inr):
            return False, "daily_loss_limit_reached"
        if self.state.today_trades_opened >= 1:
            # H4 spec: max 1 trade per day.
            return False, "daily_trade_limit_reached"
        # Cumulative drawdown check
        if (abs(self.state.current_drawdown_inr) >=
                self.config.cumulative_drawdown_inr):
            self._halt(f"cum_drawdown="
                       f"{abs(self.state.current_drawdown_inr):.0f}")
            self._persist()
            return False, "cumulative_drawdown_limit"
        return True, "ok"

    def on_position_opened(self, meta: dict) -> None:
        self.state.today_trades_opened += 1
        self.audit.write("position_opened", meta)
        self._persist()

    def on_position_tick(self, unrealized_inr: float) -> str:
        """Returns 'hold' or 'flatten_single_trade_sl'."""
        if unrealized_inr <= -self.config.single_trade_max_loss_inr:
            return "flatten_single_trade_sl"
        return "hold"

    def on_trade_closed(self, realized_inr: float, meta: dict) -> None:
        """Update state after a position closes. Triggers daily-loss /
        cumulative-DD / t-test halts as appropriate."""
        self.state.today_trades_closed += 1
        self.state.today_pnl_inr += realized_inr
        self.state.cumulative_pnl_inr += realized_inr
        if self.state.cumulative_pnl_inr > self.state.cum_peak_pnl_inr:
            self.state.cum_peak_pnl_inr = self.state.cumulative_pnl_inr
        self.state.closed_pnls.append(float(realized_inr))

        self.audit.write("trade_closed", {
            "pnl_inr": realized_inr,
            "today_pnl_inr": self.state.today_pnl_inr,
            "cum_pnl_inr": self.state.cumulative_pnl_inr,
            "drawdown_inr": self.state.current_drawdown_inr,
            **meta,
        })

        # Daily loss halt (for the rest of today)
        if self.state.today_pnl_inr <= -self.config.daily_max_loss_inr:
            # We don't flip global `halted` for daily; can_open() checks the
            # daily counter directly. But we record it.
            self.audit.write("daily_loss_limit_hit", {
                "today_pnl_inr": self.state.today_pnl_inr,
            })

        # Cumulative drawdown halt (sticky)
        if (abs(self.state.current_drawdown_inr) >=
                self.config.cumulative_drawdown_inr and not self.state.halted):
            self._halt(f"cum_drawdown="
                       f"{abs(self.state.current_drawdown_inr):.0f}")

        # Sample-size t-test halt
        n = len(self.state.closed_pnls)
        if (n > 0 and n % self.config.ttest_every_n_trades == 0
                and not self.state.halted):
            arr = np.array(self.state.closed_pnls, dtype=float)
            mean = arr.mean()
            std = arr.std()
            if std > 0:
                t_stat = mean / (std / np.sqrt(len(arr)))
                p_one = float(1.0 - stats.t.cdf(t_stat, df=len(arr) - 1))
                # Halt if mean<0 AND we're confident it's negative (1-p_one<thresh)
                sharpe_neg = mean < 0
                p_neg = float(stats.t.cdf(t_stat, df=len(arr) - 1))
                self.audit.write("ttest_check", {
                    "n": n, "mean": mean, "p_value_pos": p_one,
                    "p_value_neg": p_neg,
                })
                if sharpe_neg and p_neg < self.config.ttest_p_threshold:
                    self._halt(f"ttest_sharpe_negative_n={n}_p={p_neg:.3f}")

        self._persist()

    def on_day_end(self, day: date) -> None:
        """Explicit end-of-day call (idempotent). _roll_to_day handles the
        accounting; this just lets the caller advance the cursor."""
        self._roll_to_day(day + timedelta(days=1))

    def reset_halt(self, operator_reason: str) -> None:
        """Manual restart of a halted strategy. AUDIT-LOGGED."""
        if not self.state.halted:
            return
        self.audit.write("halt_reset", {
            "prior_reason": self.state.halt_reason,
            "operator_reason": operator_reason,
        })
        self.state.halted = False
        self.state.halt_reason = None
        self.state.halt_ts = None
        self.state.consecutive_loss_days = 0
        self._persist()
        log.info(f"Halt reset by operator: {operator_reason}")

    def snapshot(self) -> dict:
        """Read-only view of current state, for logging/monitoring."""
        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "cumulative_pnl_inr": self.state.cumulative_pnl_inr,
            "drawdown_inr": self.state.current_drawdown_inr,
            "today_date": self.state.today_date,
            "today_pnl_inr": self.state.today_pnl_inr,
            "today_trades_opened": self.state.today_trades_opened,
            "consecutive_loss_days": self.state.consecutive_loss_days,
            "closed_trades_total": len(self.state.closed_pnls),
        }
