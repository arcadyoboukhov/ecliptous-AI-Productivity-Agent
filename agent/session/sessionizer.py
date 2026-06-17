from datetime import datetime
from enum import Enum
import uuid


# --- Session State Machine ---
# Formal state management for session lifecycle
class SessionState(Enum):
    """
    Session state enumeration with formal state transitions.
    
    States:
    - INACTIVE: No session running (initial state)
    - ACTIVE: Session collecting data
    - FINALIZING: Ending, flushing buffers, running ML pipeline
    - FINALIZED: Persisted and analyzed
    
    Valid transitions:
    - INACTIVE → ACTIVE (start_session_if_needed)
    - ACTIVE → FINALIZING (end_session_if_active)
    - FINALIZING → FINALIZED (after ML callback completes)
    
    Non-overlapping: Only one session can be ACTIVE or FINALIZING at a time.
    """
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"


# --- Session Object ---
# Sessions represent coherent units of sustained work activity inferred from behavior.
# A session is the smallest unit of intentional engagement with stable cognitive context.
class Session:
    def __init__(self, start_time: datetime, session_id: str | None = None, device_id: str | None = None):
        self.id = session_id or str(uuid.uuid4())
        self.start = start_time
        self.end = start_time
        self.device_id = device_id or "unknown"  # Machine identifier for multi-device support
        
        # Formal state management
        self.state = SessionState.INACTIVE
        self.state_changed_at = start_time
        
        # whether this session is currently active (in-progress) - DEPRECATED, use state instead
        self.in_progress = True
        
        # Activity tracking for feature extraction
        self.apps = set()
        self.event_count = 0
        self.input_events = {"keys": 0, "clicks": 0, "mouse_distance": 0}
        self.timeline = {}  # Minute-bucketed timeline
        self.intent_breakdown = {}
        self.intent_segments = []
        
        # ML inference results
        self.inferred_task_id = None  # Set by ML pipeline during finalization
        
        # Rolling features (computed during session lifetime)
        self.rolling_features = {
            "entropy": 0.0,
            "continuity": 0.0,
            "intensity": 0.0,
            "max_sustained_minutes": 0.0
        }
        
        # Online task tracking (multiple tasks within single session)
        self.current_task_assignment = None  # Latest online classification result
        self.intra_session_tasks = []  # List of TaskSegment objects for task transitions
        self.task_classification_history = []  # Timeline of all classifications
    
    def set_state(self, new_state: SessionState, timestamp: datetime | None = None):
        """
        Transition to a new state. Validates state transitions are valid.
        
        Allowed transitions:
        - INACTIVE → ACTIVE
        - ACTIVE → FINALIZING
        - FINALIZING → FINALIZED
        - Any state → INACTIVE (emergency stop)
        """
        current = self.state
        ts = timestamp or datetime.now()
        
        # Validate transition
        if current == new_state:
            # Already in this state - no transition needed
            return
        
        valid_transitions = {
            SessionState.INACTIVE: [SessionState.ACTIVE],
            SessionState.ACTIVE: [SessionState.FINALIZING, SessionState.INACTIVE],
            SessionState.FINALIZING: [SessionState.FINALIZED, SessionState.INACTIVE],
            SessionState.FINALIZED: [SessionState.INACTIVE],  # Reset to inactive
        }
        
        if new_state not in valid_transitions.get(current, []):
            raise ValueError(
                f"Invalid state transition: {current.value} → {new_state.value}. "
                f"Valid transitions from {current.value}: {[s.value for s in valid_transitions[current]]}"
            )
        
        # Update state
        old_state = self.state
        self.state = new_state
        self.state_changed_at = ts
        
        # Maintain backward compatibility with in_progress flag
        if new_state == SessionState.ACTIVE:
            self.in_progress = True
        elif new_state in (SessionState.FINALIZING, SessionState.FINALIZED):
            self.in_progress = False
        elif new_state == SessionState.INACTIVE:
            self.in_progress = False
    
    def is_active(self) -> bool:
        """Check if session is actively collecting data."""
        return self.state == SessionState.ACTIVE
    
    def is_finalizing(self) -> bool:
        """Check if session is in finalization phase."""
        return self.state == SessionState.FINALIZING
    
    def is_finalized(self) -> bool:
        """Check if session has completed and been persisted."""
        return self.state == SessionState.FINALIZED

    def update_end(self, time: datetime):
        """Update the end timestamp of the session."""
        self.end = time
        self.in_progress = False
    
    def update_activity(self, keys: int = 0, clicks: int = 0, mouse_distance: float = 0, app: str = None):
        """Incrementally update session activity metrics."""
        self.input_events["keys"] += keys
        self.input_events["clicks"] += clicks
        self.input_events["mouse_distance"] += mouse_distance
        self.event_count += (keys + clicks)
        
        if app:
            self.apps.add(app)
    
    def finalize_features(self):
        """
        Finalize feature vector at session end. Called during session finalization.
        
        This method prepares the session for ML processing by computing basic features.
        More sophisticated features will be computed by the analytics layer.
        """
        # Calculate basic intensity metric
        duration_seconds = (self.end - self.start).total_seconds()
        if duration_seconds > 0:
            total_inputs = self.input_events["keys"] + self.input_events["clicks"]
            input_per_minute = (total_inputs / duration_seconds) * 60
            # Normalize intensity to [0, 1] range (assuming 60 inputs/min is maximum)
            self.rolling_features["intensity"] = min(input_per_minute / 60.0, 1.0)
        
        # Calculate app diversity (simple entropy-like metric)
        if self.apps:
            self.rolling_features["app_diversity"] = min(len(self.apps) / 10.0, 1.0)
        
        # Duration in minutes
        self.rolling_features["max_sustained_minutes"] = duration_seconds / 60.0


# --- Session Manager ---
# Compatibility shim that routes legacy callers to SessionManager v2 while keeping
# a minimal API surface (start_session_if_needed/end_session_if_active/current_session)
# used by the agent loop and tests.
from agent.session.manager_v2 import SessionManager as SessionManagerV2


class SessionManager(SessionManagerV2):
    """Compatibility wrapper that delegates to SessionManager v2.

    Legacy entry points (start_session_if_needed, end_session_if_active, current_session,
    completed_sessions) are preserved so existing callers keep working while the
    underlying implementation uses the newer session model and SignalBuffer path.
    """

    def __init__(self, idle_threshold_seconds: int = 300, ml_finalization_callback=None, gate=None):
        super().__init__(gate=gate)
        self.idle_threshold_seconds = idle_threshold_seconds
        self.ml_finalization_callback = ml_finalization_callback
        self.completed_sessions: list = []

    # --- Legacy compatibility helpers -------------------------------------------------
    def _attach_legacy_api(self, session):
        """Augment v2 Session objects with legacy fields/methods expected by ML/tests."""
        if session is None:
            return

        # Core identity aliases
        session.id = getattr(session, "session_id", None) or getattr(session, "id", None)

        # Legacy start/end aliases
        if not hasattr(session, "start"):
            session.start = getattr(session, "started_at", None) or getattr(session, "created_at", None)
        if not hasattr(session, "end"):
            session.end = getattr(session, "ended_at", None) or session.start

        # Rolling feature scaffolding
        if not hasattr(session, "rolling_features") or session.rolling_features is None:
            session.rolling_features = {
                "entropy": 0.0,
                "continuity": 0.0,
                "intensity": 0.0,
                "max_sustained_minutes": 0.0,
            }

        # Event/activity scaffolding
        if not hasattr(session, "apps") or session.apps is None:
            session.apps = set()
        if not hasattr(session, "event_count"):
            session.event_count = 0
        if not hasattr(session, "input_events") or session.input_events is None:
            session.input_events = {"keys": 0, "clicks": 0, "mouse_distance": 0.0}
        if not hasattr(session, "timeline") or session.timeline is None:
            session.timeline = {}
        if not hasattr(session, "intent_breakdown"):
            session.intent_breakdown = {}
        if not hasattr(session, "intent_segments"):
            session.intent_segments = []
        
        # ML task tracking (online task classification)
        if not hasattr(session, "intra_session_tasks") or session.intra_session_tasks is None:
            session.intra_session_tasks = []  # List of task segments/transitions from ML
        if not hasattr(session, "current_task_assignment"):
            session.current_task_assignment = None  # Latest online classification result
        if not hasattr(session, "task_classification_history"):
            session.task_classification_history = []  # Timeline of all classifications

        # Legacy methods (simple wrappers)
        def update_activity(keys: int = 0, clicks: int = 0, mouse_distance: float = 0, app: str = None):
            session.input_events["keys"] = session.input_events.get("keys", 0) + int(keys)
            session.input_events["clicks"] = session.input_events.get("clicks", 0) + int(clicks)
            session.input_events["mouse_distance"] = session.input_events.get("mouse_distance", 0.0) + float(mouse_distance or 0.0)
            session.event_count = getattr(session, "event_count", 0) + int(keys) + int(clicks)
            if app:
                try:
                    session.apps.add(app)
                except Exception:
                    session.apps = set([app])

        def update_end(time):
            session.end = time
            session.ended_at = time

        def finalize_features():
            duration_seconds = (session.end - session.start).total_seconds() if session.start and session.end else 0
            if duration_seconds > 0:
                total_inputs = session.input_events.get("keys", 0) + session.input_events.get("clicks", 0)
                input_per_minute = (total_inputs / duration_seconds) * 60
                session.rolling_features["intensity"] = min(input_per_minute / 60.0, 1.0)
            if session.apps:
                session.rolling_features["app_diversity"] = min(len(session.apps) / 10.0, 1.0)
            session.rolling_features["max_sustained_minutes"] = duration_seconds / 60.0 if duration_seconds else 0.0

        session.update_activity = update_activity
        session.update_end = update_end
        session.finalize_features = finalize_features

    @property
    def current_session(self):
        if not getattr(self, "gate", None):
            return None
        if not self.gate.is_active():
            return None
        try:
            session_id = self.gate.active_session_id
        except Exception:
            return None
        session = self.sessions.get(session_id)
        self._attach_legacy_api(session)
        return session

    def start_session_if_needed(self, start_time: datetime, device_id: str | None = None):
        """Start a new session if none is active; otherwise return the active one."""
        try:
            if self.current_session:
                return self.current_session

            # Create and immediately start a new session using the v2 manager
            name = f"Auto Session {start_time.isoformat(timespec='seconds')}"
            session = self.create_session(name=name)
            self._attach_legacy_api(session)
            # Optional device tag for downstream analytics
            try:
                session.device_id = device_id or getattr(session, "device_id", "unknown")
            except Exception:
                pass
            
            # Create SignalBuffer for timeline tracking
            try:
                from agent.session.signal_buffer import SignalBuffer
                session.signals = SignalBuffer(
                    session_id=session.id,
                    start_time=start_time
                )
            except Exception:
                pass

            started = super().start_session(session.session_id)
            # Align the start timestamp with the engagement detection time for consistency
            started.started_at = start_time
            started.start = start_time
            started.end = start_time
            self._attach_legacy_api(started)
            
            # Also attach signals to started session
            if hasattr(session, 'signals'):
                started.signals = session.signals
            
            try:
                self._persist()
                self._save_gate_state()
            except Exception as persist_err:
                # Log but don't crash - session is created in memory
                try:
                    from agent.session.error_handling import get_error_handler
                    get_error_handler().log_error("start_session_persist", persist_err, session.session_id)
                except:
                    pass
            
            # Print session creation message
            print(f"\n{'='*80}", flush=True)
            print(f"[SESSION CREATED] {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)
            print(f"  Session ID: {started.session_id}", flush=True)
            print(f"  Name: {name}", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            return started
        except Exception as e:
            # Log error and return None to prevent crash
            try:
                from agent.session.error_handling import get_error_handler
                get_error_handler().log_error("start_session_if_needed", e, critical=True)
            except:
                pass
            return None

    def end_session_if_active(self, end_time: datetime, reason: str = "idle_threshold"):
        """End the active session (if any) and invoke the optional ML callback."""
        try:
            session = self.current_session
            if not session:
                return None

            finished = super().end_session(session.session_id)
            # Normalize the end timestamp to the caller-provided time
            finished.ended_at = end_time
            finished.end = end_time
            if not getattr(finished, "start", None):
                finished.start = getattr(finished, "started_at", end_time)
            self._attach_legacy_api(finished)
            self.completed_sessions.append(finished)

            # Critical: persist session before ML callback
            persist_success = False
            try:
                self._persist()
                self._save_gate_state()
                persist_success = True
            except Exception as persist_err:
                try:
                    from agent.session.error_handling import get_error_handler
                    get_error_handler().log_error("end_session_persist", persist_err, session.session_id, critical=True)
                except:
                    pass

            # ML callback - errors here should not prevent session from being saved
            if self.ml_finalization_callback:
                try:
                    self.ml_finalization_callback(finished)
                except Exception as ml_err:
                    try:
                        from agent.session.error_handling import get_error_handler
                        get_error_handler().log_error("end_session_ml_callback", ml_err, session.session_id, critical=False)
                    except:
                        pass

            # Persist session + task segments to SQLite for DB visibility
            try:
                from agent.storage.db import upsert_session_record, replace_task_segments

                upsert_session_record(finished)
                segments = getattr(finished, "intra_session_tasks", [])
                replace_task_segments(finished.session_id, segments)
            except Exception as db_err:
                try:
                    from agent.session.error_handling import get_error_handler
                    get_error_handler().log_error("end_session_db_persist", db_err, session.session_id, critical=False)
                except:
                    pass
            
            # Calculate and print session end message
            session_id = finished.session_id
            duration_seconds = (finished.end - finished.start).total_seconds() if finished.start and finished.end else 0
            duration_minutes = duration_seconds / 60
            
            print(f"\n{'='*80}", flush=True)
            print(f"[SESSION ENDED] {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}", flush=True)
            print(f"  Session ID: {session_id}", flush=True)
            print(f"  Duration: {duration_minutes:.1f} minutes ({int(duration_seconds)} seconds)", flush=True)
            print(f"  Reason: {reason}", flush=True)
            print(f"  Events: {finished.event_count}", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            return finished
            
        except Exception as e:
            # Critical error in end_session - log and return None
            try:
                from agent.session.error_handling import get_error_handler
                session_id = getattr(session, 'session_id', 'unknown') if 'session' in locals() else 'unknown'
                get_error_handler().log_error("end_session_if_active", e, session_id, critical=True)
            except:
                pass
            return None

    def update_session_activity(self, keys: int = 0, clicks: int = 0, mouse_distance: float = 0, app: str = None, window_title: str = None):
        """Mirror the legacy per-event counters into the v2 SignalBuffer."""
        session = self.current_session
        if not session or not getattr(session, "signals", None):
            return

        try:
            for _ in range(int(keys)):
                session.signals.record_keyboard_press()
            for _ in range(int(clicks)):
                session.signals.record_mouse_click()
            if mouse_distance:
                session.signals.record_mouse_movement(mouse_distance)
            if app:
                session.signals.record_app_window(app, window_title or "")
                try:
                    session.apps.add(app)
                except Exception:
                    pass
                # Track last observed app/window for task labeling
                try:
                    session.last_app = app
                    session.last_window_title = window_title or ""
                except Exception:
                    pass

            # Maintain legacy counters for analytics compatibility
            session.event_count = getattr(session, "event_count", 0) + int(keys) + int(clicks)
            if not hasattr(session, "input_events") or session.input_events is None:
                session.input_events = {"keys": 0, "clicks": 0, "mouse_distance": 0.0}
            session.input_events["keys"] = session.input_events.get("keys", 0) + int(keys)
            session.input_events["clicks"] = session.input_events.get("clicks", 0) + int(clicks)
            session.input_events["mouse_distance"] = session.input_events.get("mouse_distance", 0.0) + float(mouse_distance or 0.0)
        except Exception:
            pass

    def get_active_session(self):
        return self.current_session
