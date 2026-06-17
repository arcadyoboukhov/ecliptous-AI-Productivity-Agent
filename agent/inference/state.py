from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
import json

from agent.storage.db import get_connection


class ActivityState(Enum):
    UNKNOWN = "unknown"
    IDLE = "idle"
    # Keep ACTIVE_DRIFT as an alias for historical compatibility, but expose
    # ACTIVE_UNALIGNED as the canonical name requested in the UX spec.
    ACTIVE_UNALIGNED = "active_unaligned"
    ACTIVE_DRIFT = "active_unaligned"
    ACTIVE_ALIGNED = "active_aligned"
    CONTRADICTORY = "contradictory"
    PAUSED = "paused"


@dataclass
class StateSnapshot:
    state: ActivityState
    since: datetime


class StateManager:
    """State manager with simple hysteresis and pending/committed semantics.

    - Keeps only the current and previous committed snapshots in memory.
    - Tracks a pending candidate state and only commits after a minimum duration.
    - Emits enriched `STATE_CHANGE` events with reason and previous duration.
    """

    MIN_IDLE_SECONDS = 10  # Reduced from 300 to 10 seconds
    MIN_DRIFT_SECONDS = 5   # Reduced from 120 to 5 seconds
    MIN_CONTRADICTION_SECONDS = 5  # Reduced from 60 to 5 seconds

    THRESHOLDS: Dict[ActivityState, int] = {
        ActivityState.IDLE: MIN_IDLE_SECONDS,
        ActivityState.ACTIVE_DRIFT: MIN_DRIFT_SECONDS,
        ActivityState.CONTRADICTORY: MIN_CONTRADICTION_SECONDS,
        ActivityState.ACTIVE_ALIGNED: 0,
        ActivityState.PAUSED: 0,
        ActivityState.UNKNOWN: 0,
    }

    def __init__(self):
        self.current_state: Optional[StateSnapshot] = None
        self.previous_state: Optional[StateSnapshot] = None

        # pending candidate state (not yet committed)
        self.pending_state: Optional[ActivityState] = None
        self.pending_since: Optional[datetime] = None

    def _persist_event(self, ts: datetime, payload: dict) -> None:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (timestamp, event_type, payload) VALUES (?, ?, ?)",
                (ts.isoformat(), "STATE_CHANGE", json.dumps(payload)),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _derive_reason(self, candidate: ActivityState, context: Optional[dict]) -> str:
        if candidate == ActivityState.UNKNOWN:
            return "No committed state"
        if candidate == ActivityState.IDLE:
            return "Idle detector"
        if candidate == ActivityState.PAUSED:
            return "Session/task paused"
        if candidate == ActivityState.CONTRADICTORY:
            return "Conflicting signals: task present + strong input"
        if candidate == ActivityState.ACTIVE_ALIGNED:
            return "Active task + input"
        # ACTIVE_UNALIGNED
        if context:
            if not context.get("active_task"):
                return "No active task"
            score = context.get("input_activity_score")
            if score is not None and score < 1:
                return "Low input density"
        return "Unaligned activity"

    def update(self, candidate: ActivityState, context: Optional[dict] = None, ts: Optional[datetime] = None) -> bool:
        """Process a candidate state from the inference engine.

        Returns True if a committed transition occurred, False otherwise.
        """
        now = ts or datetime.now()

        # Initial commit if we have no state yet
        if self.current_state is None:
            self.current_state = StateSnapshot(candidate, now)
            payload = {
                "from": None,
                "to": candidate.name,
                "duration_prev": 0,
                "reason": "initial",
            }
            # Include standard context fields (always present for clarity)
            try:
                payload.update({
                    "active_task": context.get("active_task") if context else None,
                    "active_app": context.get("active_app") if context else None,
                    "is_idle": context.get("is_idle") if context else None,
                    "input_activity_score": context.get("input_activity_score") if context else 0.0,
                    "session_active": context.get("session_active") if context else False,
                    "intensity": context.get("intensity") if context else None,
                })
                # Keep small 'confidence' float for backward compatibility
                if context and context.get("intensity") is not None:
                    payload["confidence"] = (context.get("intensity") or 0.0) / 100.0
                else:
                    payload.setdefault("confidence", 0.0)
            except Exception:
                pass
            self._persist_event(now, payload)
            return True

        # If candidate matches committed state, update intensity but don't change state
        if candidate == self.current_state.state:
            self.pending_state = None
            self.pending_since = None
            
            # Update intensity even if state hasn't changed (for live UI updates)
            payload = {
                "from": self.current_state.state.name,
                "to": candidate.name,
                "duration_prev": 0,
                "reason": "intensity_update",
            }
            try:
                payload.update({
                    "active_task": context.get("active_task") if context else None,
                    "active_app": context.get("active_app") if context else None,
                    "is_idle": context.get("is_idle") if context else None,
                    "input_activity_score": context.get("input_activity_score") if context else 0.0,
                    "session_active": context.get("session_active") if context else False,
                    "intensity": context.get("intensity") if context else None,
                })
                if context and context.get("intensity") is not None:
                    payload["confidence"] = (context.get("intensity") or 0.0) / 100.0
                else:
                    payload.setdefault("confidence", 0.0)
            except Exception:
                pass
            self._persist_event(now, payload)
            return True  # Changed from False to True to indicate we persisted

        # New candidate: start or continue pending
        if self.pending_state is None or self.pending_state != candidate:
            self.pending_state = candidate
            self.pending_since = now
            return False

        # Candidate is same as pending; check if it persisted long enough
        elapsed = (now - self.pending_since).total_seconds()
        threshold = self.THRESHOLDS.get(candidate, 0)
        if elapsed < threshold:
            return False

        # Commit transition
        prev = self.current_state
        duration_prev = int((now - prev.since).total_seconds()) if prev else 0
        reason = self._derive_reason(candidate, context)

        payload = {
            "from": prev.state.name if prev else None,
            "to": candidate.name,
            "duration_prev": duration_prev,
            "reason": reason,
        }
        # Include standard context fields (always present for clarity)
        try:
            payload.update({
                "active_task": context.get("active_task") if context else None,
                "active_app": context.get("active_app") if context else None,
                "is_idle": context.get("is_idle") if context else None,
                "input_activity_score": context.get("input_activity_score") if context else 0.0,
                "session_active": context.get("session_active") if context else False,
                "intensity": context.get("intensity") if context else None,
            })
            if context and context.get("intensity") is not None:
                payload["confidence"] = (context.get("intensity") or 0.0) / 100.0
            else:
                payload.setdefault("confidence", 0.0)
        except Exception:
            pass

        # Persist event at commit time
        self._persist_event(now, payload)

        # Update snapshots: current becomes previous; new current uses pending_since
        self.previous_state = self.current_state
        commit_since = self.pending_since or now
        self.current_state = StateSnapshot(candidate, commit_since)

        # Clear pending
        self.pending_state = None
        self.pending_since = None

        return True
