
import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from agent.session.gate import SessionGate, get_session_gate, SessionAlreadyActive, NoActiveSession
from agent.session.signal_buffer import SignalBuffer


@dataclass
class SessionTask:
    """A task within a session."""
    task_id: str
    session_id: str  # Foreign key (required)
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    state: str = "ACTIVE"  # ACTIVE | PAUSED | COMPLETED
    accumulated_seconds: float = 0.0  # Tracks paused/resumed duration
    activity_type: str = "HYBRID"  # PASSIVE | ACTIVE | HYBRID
    
    def is_active(self) -> bool:
        return self.end_time is None and self.state == "ACTIVE"
    
    def duration_seconds(self) -> float:
        """Total duration: accumulated + current active period."""
        if not self.start_time:
            return self.accumulated_seconds
        
        # If completed, return accumulated (start_time/end_time may have been reset during pause/resume)
        if self.state == "COMPLETED":
            return self.accumulated_seconds
        
        # If paused, return accumulated (not counting current period)
        if self.state == "PAUSED":
            return self.accumulated_seconds
        
        # Task is ACTIVE: add current active period to accumulated
        current = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return self.accumulated_seconds + current
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "state": self.state,
            "accumulated_seconds": self.accumulated_seconds,
            "activity_type": self.activity_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionTask":
        return cls(
            task_id=data["task_id"],
            session_id=data["session_id"],
            name=data["name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            state=data.get("state", "ACTIVE"),
            accumulated_seconds=data.get("accumulated_seconds", 0.0),
            activity_type=data.get("activity_type", "HYBRID").upper()
        )


@dataclass
class Session:
    
    session_id: str
    name: str
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    tasks: Dict[str, SessionTask] = field(default_factory=dict)
    signals: Optional[SignalBuffer] = None
    accumulated_seconds: float = 0.0  # Tracks paused/resumed duration
    intensity_history: List[Tuple[datetime, float]] = field(default_factory=list)  # (timestamp, intensity_score)
    
    def is_active(self) -> bool:
        """Session is active if started but not ended."""
        return self.started_at is not None and self.ended_at is None
    
    def is_paused(self) -> bool:
        """Session is paused if it was started but signals are not being collected."""
        # Determined by SessionGate; this is a helper for reporting
        return self.started_at is not None and self.ended_at is None
    
    def duration_seconds(self) -> float:
        """Total duration: accumulated + current active period."""
        # If session hasn't been started, return accumulated
        if not self.started_at:
            return self.accumulated_seconds
        
        # If session is ended, return accumulated (started_at may have been reset during pause/resume)
        if self.ended_at is not None:
            return self.accumulated_seconds
        
        # Session is currently active; add current active period
        current = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return self.accumulated_seconds + current
    
    def add_task(self, task: SessionTask):
        """Add a task to this session."""
        if task.session_id != self.session_id:
            raise ValueError(f"Task {task.task_id} belongs to {task.session_id}, not {self.session_id}")
        self.tasks[task.task_id] = task
    
    def get_active_task(self) -> Optional[SessionTask]:
        """Return the first active task (should be at most one)."""
        for task in self.tasks.values():
            if task.is_active():
                return task
        return None
    
    def end_all_tasks(self, end_time: datetime):
        """Called when session is ended; ends all tasks."""
        for task in self.tasks.values():
            if task.state != "COMPLETED":
                if task.end_time is None:
                    task.end_time = end_time
                task.state = "COMPLETED"
    
    def record_intensity(self, intensity_score: float, timestamp: Optional[datetime] = None):
        """Record an intensity score with timestamp."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.intensity_history.append((timestamp, intensity_score))
    
    def get_average_intensity(self) -> Optional[float]:
        """Calculate average intensity from history."""
        if not self.intensity_history:
            return None
        return sum(score for _, score in self.intensity_history) / len(self.intensity_history)
    
    def get_intensity_stats(self) -> dict:
        """Get intensity statistics."""
        if not self.intensity_history:
            return {
                "count": 0,
                "average": None,
                "min": None,
                "max": None,
                "latest": None
            }
        scores = [score for _, score in self.intensity_history]
        return {
            "count": len(scores),
            "average": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "latest": scores[-1] if scores else None
        }
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "signals": self.signals.to_dict() if self.signals else None,
            "accumulated_seconds": self.accumulated_seconds,
            "intensity_history": [[ts.isoformat(), score] for ts, score in self.intensity_history]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        s = cls(
            session_id=data["session_id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            accumulated_seconds=data.get("accumulated_seconds", 0.0)
        )
        for task_data in data.get("tasks", {}).values():
            task = SessionTask.from_dict(task_data)
            s.tasks[task.task_id] = task
        
        if data.get("signals"):
            s.signals = SignalBuffer.from_dict(data["signals"])
        
        # Restore intensity history
        if data.get("intensity_history"):
            s.intensity_history = [
                (datetime.fromisoformat(ts), score)
                for ts, score in data["intensity_history"]
            ]
        
        return s


class SessionManager:
    """
    Manages session lifecycle and coordinates with SessionGate and SignalBuffer.
    
    Responsibilities:
    - CRUD for sessions
    - Coordinate with SessionGate (start/stop tracking)
    - Create/destroy SignalBuffers
    - Persist sessions
    - Maintain active session state
    """
    
    SESSIONS_FILE = os.path.join("sessions_v2.json")
    STATE_FILE = os.path.join(".session_state")  # Stores active session ID
    
    def __init__(self, gate: Optional[SessionGate] = None):
        self.gate = gate or get_session_gate()
        self.sessions: Dict[str, Session] = {}
        self.current_signal_buffer: Optional[SignalBuffer] = None
        self._load_sessions()
        self._restore_gate_state()
        # Clean up any ghost sessions that were started but not properly paused
        self.cleanup_ghost_sessions()
        # Start background intensity monitoring thread
        self._start_intensity_monitor()
    
    def _start_intensity_monitor(self):
        """Start a background thread that records session intensity every 10 seconds."""
        import threading
        import time
        from agent.storage.db import log_intensity_score
        
        def _monitor_loop():
            while True:
                time.sleep(10)
                try:
                    if self.gate.is_active():
                        session_id = self.gate.active_session_id
                        if session_id is None:
                            continue
                        session = self.get_session(session_id)
                        if session and session.signals:
                            metrics = session.signals.metrics_since(10)
                            intensity_score = metrics.get("intensity", 0)
                            timestamp = datetime.now(timezone.utc)
                            
                            # Record intensity in session history (for JSON persistence)
                            session.record_intensity(intensity_score, timestamp)
                            
                            # Save to SQLite database
                            try:
                                log_intensity_score(session_id, intensity_score, timestamp, window_seconds=10)
                            except Exception:
                                pass
                            
                            # Persist sessions to JSON
                            self._persist()
                except Exception as e:
                    pass
        
        threading.Thread(target=_monitor_loop, daemon=True, name="intensity-monitor").start()
    
    def _restore_gate_state(self):
        """Restore the active session ID from persistent storage.
        
        NOTE: We do NOT auto-restore active sessions. If a session was paused
        (active_session_id is None in state file), it stays paused even if
        started_at is set. This prevents ghost sessions from auto-resuming.
        """
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                    active_session_id = data.get("active_session_id")
                    # Only restore if explicitly saved as active
                    if active_session_id and active_session_id in self.sessions:
                        session = self.sessions[active_session_id]
                        # Only restore if session hasn't been ended
                        # NOTE: started_at may be None after session is paused/ended, so check ended_at instead
                        if not session.ended_at:
                            self.gate.active_session_id = active_session_id
                            # Recreate signal buffer if it should exist
                            if not self.current_signal_buffer and session.signals is None:
                                now = datetime.now(timezone.utc)
                                session.signals = SignalBuffer(session_id=active_session_id, start_time=now)
                                self.current_signal_buffer = session.signals
        except Exception as e:
            # Silently ignore state file errors; this is best-effort
            pass
    
    def cleanup_ghost_sessions(self):
        """Pause any sessions that are started but gate says are not active.
        
        This handles the case where UI crashed or was force-closed without
        properly pausing sessions. These 'ghost' sessions should be paused.
        """
        paused_count = 0
        for session_id, session in self.sessions.items():
            # Skip if already ended
            if session.ended_at is not None:
                continue
            # If session is started but gate says it's not active, it's a ghost
            if session.started_at is not None and self.gate.active_session_id != session_id:
                try:
                    # Ensure signal buffer exists before pausing
                    if not session.signals:
                        now = datetime.now(timezone.utc)
                        session.signals = SignalBuffer(session_id=session_id, start_time=now)
                    
                    # Accumulate duration before pausing
                    if session.started_at:
                        duration = (datetime.now(timezone.utc) - session.started_at).total_seconds()
                        session.accumulated_seconds += duration
                        session.started_at = None  # Clear start time
                    
                    self._persist()
                    self._save_gate_state()
                    paused_count += 1
                except Exception as e:
                    pass
        
        if paused_count > 0:
            pass
    
    def _save_gate_state(self):
        """Persist the active session ID."""
        try:
            data = {
                "active_session_id": self.gate.active_session_id
            }
            with open(self.STATE_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def create_session(self, name: str, session_id: Optional[str] = None) -> Session:
        """
        Create a new session (not started yet).
        
        Does NOT enable tracking.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=session_id,
            name=name,
            created_at=now
        )
        self.sessions[session_id] = session
        self._persist()
        return session
    
    def start_session(self, session_id: str) -> Session:
        """
        Start a session, enabling tracking via SessionGate.
        
        Preconditions:
        - Session exists
        - No other session is active (enforced by gate)
        
        Raises SessionAlreadyActive if another session is active.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Ask gate to activate this session. If another session is active,
        # raise SessionAlreadyActive (tests expect this behavior).
        self.gate.start(session_id)
        
        # Initialize tracking
        now = datetime.now(timezone.utc)
        session.started_at = now
        session.signals = SignalBuffer(session_id=session_id, start_time=now)
        self.current_signal_buffer = session.signals

        # Seed minimal recent activity so UI health checks do not raise before
        # real input events arrive. This is a short-lived bootstrap only.
        try:
            session.signals.ensure_recent_activity_baseline()
        except Exception:
            pass

        # Ensure input hooks are attached (best-effort). This will be a no-op
        # if `pynput` is unavailable or if hooks are already attached.
        try:
            from agent.signals.input import attach_input_hooks
            # Pass self so hooks can find sessions via SessionManager if needed
            attach_input_hooks(self, self.gate, debug=False)
        except Exception:
            # Do not fail session start if hooks cannot be attached
            pass

        # Record current active window immediately so the session has at least
        # one app/window record (useful for verification and UI feedback).
        try:
            from agent.signals.active_window import get_active_window
            aw = get_active_window()
            if aw and session.signals:
                proc = aw.get("process_name") or str(aw)
                title = aw.get("window_title") or ""
                session.signals.record_app_window(proc, title)
        except Exception:
            # Best-effort only; do not raise on failures to query active window
            pass
        
        # Log state change to database
        try:
            from agent.storage.db import log_state_change
            log_state_change("ACTIVE_UNALIGNED", session_id=session_id, timestamp=now)
        except Exception:
            pass
        
        self._persist()
        self._save_gate_state()
        return session
    
    def pause_session(self, session_id: str) -> Session:
        """
        Pause a session (stop collecting signals, but keep it open).
        
        Also pauses all tasks in the session.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Verify it's the active session
        if self.gate.active_session_id != session_id:
            raise ValueError(f"Session {session_id} is not active")
        
        # Accumulate duration before pausing
        if session.started_at:
            duration = (datetime.now(timezone.utc) - session.started_at).total_seconds()
            session.accumulated_seconds += duration
            session.started_at = None  # Clear start_at so duration stops counting
        
        # Pause all active tasks in the session
        for task in session.tasks.values():
            if task.state == "ACTIVE":
                task.state = "PAUSED"
        
        # Revert to DEFAULT preset when pausing session (pauses all tasks)
        if session.signals:
            session.signals.set_preset("DEFAULT")
        
        # Stop signal collection
        self.gate.stop()
        self.current_signal_buffer = None
        
        # Log state change to database
        try:
            from agent.storage.db import log_state_change
            log_state_change("PAUSED", session_id=session_id)
        except Exception:
            pass
        
        self._persist()
        self._save_gate_state()
        return session
    
    def resume_session(self, session_id: str) -> Session:
        """
        Resume a paused session (re-enable signal collection and duration counter).
        
        Tasks remain paused after resume (must be manually restarted).
        Failsafe: If the session was never actually started (created but never had
        start_session() called), this will initialize it as if start_session() was called.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Verify no other session is active
        if self.gate.active_session_id is not None:
            raise SessionAlreadyActive(
                f"Another session {self.gate.active_session_id} is active"
            )
        
        # Failsafe: If this is a fresh session that was never started, initialize it
        # like start_session() would do
        was_never_started = session.accumulated_seconds == 0.0 and not hasattr(session, '_ever_started')
        if was_never_started:
            # Mark that we've initialized this session (using object.__setattr__ for dynamic attr)
            object.__setattr__(session, '_ever_started', True)
            # Seed minimal baseline activity like start_session does
            try:
                if not session.signals:
                    now = datetime.now(timezone.utc)
                    session.signals = SignalBuffer(session_id=session_id, start_time=now)
                session.signals.ensure_recent_activity_baseline()
            except Exception:
                pass
        
        # Always re-attach input hooks when resuming (in case app was restarted)
        try:
            from agent.signals.input import attach_input_hooks
            attach_input_hooks(self, self.gate, debug=False)
        except Exception:
            pass
        
        # Re-enable signal collection
        self.gate.start(session_id)
        
        # Restart the duration counter from this point
        now = datetime.now(timezone.utc)
        session.started_at = now
        
        # Create a new signal buffer for this resumed period
        session.signals = SignalBuffer(session_id=session_id, start_time=now)
        self.current_signal_buffer = session.signals
        
        # Record current active window like start_session does
        try:
            from agent.signals.active_window import get_active_window
            aw = get_active_window()
            if aw and session.signals:
                proc = aw.get("process_name") or str(aw)
                title = aw.get("window_title") or ""
                session.signals.record_app_window(proc, title)
        except Exception:
            pass
        
        # Log state change to database
        try:
            from agent.storage.db import log_state_change
            log_state_change("ACTIVE_UNALIGNED", session_id=session_id, timestamp=now)
        except Exception:
            pass
        
        self._persist()
        self._save_gate_state()
        return session
    
    def end_session(self, session_id: str) -> Session:
        """
        End a session, stopping tracking and finalizing all tasks.
        
        Preconditions:
        - Session exists
        - Session has been started
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        now = datetime.now(timezone.utc)
        
        # Accumulate duration before ending
        if session.started_at:
            duration = (datetime.now(timezone.utc) - session.started_at).total_seconds()
            session.accumulated_seconds += duration
            session.started_at = None
        
        # End all tasks in the session
        session.end_all_tasks(now)
        
        # Revert to DEFAULT preset when ending session (ends all tasks)
        if session.signals:
            session.signals.set_preset("DEFAULT")
        
        # Stop tracking
        session.ended_at = now
        if self.gate.active_session_id == session_id:
            self.gate.stop()
            self.current_signal_buffer = None
        
        # Log state change to database
        try:
            from agent.storage.db import log_state_change
            log_state_change("COMPLETED", session_id=session_id, timestamp=now)
        except Exception:
            pass
        
        self._persist()
        self._save_gate_state()
        return session
    
    def uncomplete_session(self, session_id: str) -> Session:
        """
        Uncomplete an ended session (make it paused again, not active).
        
        Tasks are restored to PAUSED state and remain paused.
        Preconditions:
        - Session exists
        - Session has been ended
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        if session.ended_at is None:
            raise ValueError(f"Session is not ended (state: active/paused). Uncomplete only works on ended sessions.")
        
        # Verify no other session is active
        if self.gate.active_session_id is not None:
            raise SessionAlreadyActive(
                f"Another session {self.gate.active_session_id} is active"
            )
        
        # Uncomplete all tasks (restore to PAUSED, NOT ACTIVE)
        for task in session.tasks.values():
            if task.state == "COMPLETED":
                task.state = "PAUSED"  # Restore to paused state
                task.end_time = None
        
        # Revert to DEFAULT preset when uncompleting (tasks become paused)
        if session.signals:
            session.signals.set_preset("DEFAULT")
        
        # Mark session as paused (not active yet)
        session.started_at = None  # Not active, just paused
        session.ended_at = None
        
        # Do NOT re-enable tracking yet; user must explicitly resume()
        # (gate stays inactive)
        session.signals = None
        self.current_signal_buffer = None
        
        self._persist()
        self._save_gate_state()
        return session
    
    def create_task(self, session_id: str, name: str, task_id: Optional[str] = None, activity_type: str = "HYBRID") -> SessionTask:
        """
        Create a task in a session.
        
        Note: Can be created even if session is paused.
        Does NOT start the task or require the session to be active.
        """
        allowed_types = {"PASSIVE", "ACTIVE", "HYBRID"}
        activity_type = (activity_type or "HYBRID").upper()
        if activity_type not in allowed_types:
            raise ValueError(f"Invalid activity_type '{activity_type}'. Choose one of {sorted(allowed_types)}")
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        
        now = datetime.now(timezone.utc)
        task = SessionTask(
            task_id=task_id,
            session_id=session_id,
            name=name,
            start_time=now,
            activity_type=activity_type
        )
        session.add_task(task)
        
        self._persist()
        return task
    
    def start_task(self, task_id: str) -> SessionTask:
        """
        Start a task (mark it as active within its session).
        
        Preconditions:
        - Task exists
        - Task's session is active (must be checked against SessionGate)
        
        Raises ValueError if session is not active.
        """
        # Find the task
        task = None
        session = None
        for s in self.sessions.values():
            if task_id in s.tasks:
                task = s.tasks[task_id]
                session = s
                break
        
        if task is None or session is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        # Verify session is active
        if not self.gate.is_active() or self.gate.active_session_id != session.session_id:
            raise ValueError(
                f"Cannot start task: session {session.session_id} is not active. "
                f"Start or resume the session first."
            )
        
        # Pause any other active task in the session
        current = session.get_active_task()
        if current and current.task_id != task_id:
            # Accumulate duration when pausing
            pause_time = datetime.now(timezone.utc)
            if current.start_time:
                duration = (pause_time - current.start_time).total_seconds()
                current.accumulated_seconds += duration
            current.end_time = pause_time
            current.state = "PAUSED"
        
        # Start this task (fresh start_time, continue accumulating)
        task.start_time = datetime.now(timezone.utc)
        task.end_time = None
        task.state = "ACTIVE"
        
        # Switch to task's activity type preset
        if session.signals:
            session.signals.set_preset(task.activity_type)
        
        self._persist()
        return task
    
    def end_task(self, task_id: str) -> SessionTask:
        """
        End a task.
        
        Preconditions:
        - Task exists
        """
        # Find the task
        task = None
        for s in self.sessions.values():
            if task_id in s.tasks:
                task = s.tasks[task_id]
                break
        
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        # Accumulate duration before ending
        if task.start_time:
            duration = (datetime.now(timezone.utc) - task.start_time).total_seconds()
            task.accumulated_seconds += duration
        
        task.end_time = datetime.now(timezone.utc)
        task.state = "COMPLETED"
        
        # Revert to DEFAULT preset when task ends
        for s in self.sessions.values():
            if task_id in s.tasks and s.signals:
                s.signals.set_preset("DEFAULT")
                break
        
        self._persist()
        return task
    
    def pause_task(self, task_id: str) -> SessionTask:
        """
        Pause an active task (stops duration counter).
        
        Preconditions:
        - Task exists
        - Task is ACTIVE
        """
        # Find the task
        task = None
        for s in self.sessions.values():
            if task_id in s.tasks:
                task = s.tasks[task_id]
                break
        
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        if task.state != "ACTIVE":
            raise ValueError(f"Task is not active (state: {task.state})")
        
        # Accumulate duration before pausing
        if task.start_time:
            duration = (datetime.now(timezone.utc) - task.start_time).total_seconds()
            task.accumulated_seconds += duration
        
        task.end_time = datetime.now(timezone.utc)
        task.state = "PAUSED"
        
        # Revert to DEFAULT preset when task pauses
        for s in self.sessions.values():
            if task_id in s.tasks and s.signals:
                s.signals.set_preset("DEFAULT")
                break
        
        self._persist()
        return task
    
    def resume_task(self, task_id: str) -> SessionTask:
        """
        Resume a paused task (restart duration counter from current accumulated time).
        
        Preconditions:
        - Task exists
        - Task is PAUSED
        - Task's session is active
        """
        # Find the task
        task = None
        session = None
        for s in self.sessions.values():
            if task_id in s.tasks:
                task = s.tasks[task_id]
                session = s
                break
        
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        if session is None:
            raise ValueError(f"Task {task_id} has no associated session")
        
        if task.state != "PAUSED":
            raise ValueError(f"Task is not paused (state: {task.state})")
        
        # Verify session is active
        if not self.gate.is_active() or self.gate.active_session_id != session.session_id:
            raise ValueError(
                f"Cannot resume task: session {session.session_id} is not active. "
                f"Start or resume the session first."
            )
        
        # Pause any other active task in the session
        current = session.get_active_task()
        if current and current.task_id != task_id:
            if current.start_time:
                duration = (datetime.now(timezone.utc) - current.start_time).total_seconds()
                current.accumulated_seconds += duration
            current.end_time = datetime.now(timezone.utc)
            current.state = "PAUSED"
        
        # Resume this task
        task.start_time = datetime.now(timezone.utc)
        task.end_time = None
        task.state = "ACTIVE"
        
        # Switch to task's activity type preset
        if session.signals:
            session.signals.set_preset(task.activity_type)
        
        self._persist()
        return task
    
    def uncomplete_task(self, task_id: str) -> SessionTask:
        """
        Uncomplete a finished task (make it active again from end of duration).
        
        Preconditions:
        - Task exists
        - Task is COMPLETED
        - Task's session is active
        """
        # Find the task
        task = None
        session = None
        for s in self.sessions.values():
            if task_id in s.tasks:
                task = s.tasks[task_id]
                session = s
                break
        
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        if session is None:
            raise ValueError(f"Task {task_id} has no associated session")
        
        if task.state != "COMPLETED":
            raise ValueError(f"Task is not completed (state: {task.state}). Uncomplete only works on completed tasks.")
        
        # Verify session is active
        if not self.gate.is_active() or self.gate.active_session_id != session.session_id:
            raise ValueError(
                f"Cannot uncomplete task: session {session.session_id} is not active. "
                f"Start or resume the session first."
            )
        
        # Pause any other active task in the session
        current = session.get_active_task()
        if current and current.task_id != task_id:
            if current.start_time:
                duration = (datetime.now(timezone.utc) - current.start_time).total_seconds()
                current.accumulated_seconds += duration
            current.end_time = datetime.now(timezone.utc)
            current.state = "PAUSED"
        
        # Uncomplete: resume from accumulated duration
        task.start_time = datetime.now(timezone.utc)
        task.end_time = None
        task.state = "ACTIVE"
        
        self._persist()
        return task
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[Session]:
        """Return all sessions, sorted by creation time."""
        return sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session (only allowed if not active)."""
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Cannot delete an active session
        if self.gate.active_session_id == session_id:
            raise ValueError(f"Cannot delete active session {session_id}. End it first.")
        
        del self.sessions[session_id]
        self._persist()
    
    def _persist(self):
        """(DISABLED) Session persistence to JSON files is disabled."""
        pass
    
    def _load_sessions(self):
        """Load sessions from disk."""
        try:
            if os.path.exists(self.SESSIONS_FILE):
                with open(self.SESSIONS_FILE, "r") as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", {}).values():
                        session = Session.from_dict(session_data)
                        self.sessions[session.session_id] = session
        except Exception as e:
            pass
