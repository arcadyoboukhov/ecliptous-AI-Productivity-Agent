"""Background inference runner that periodically evaluates activity state.

Runs a lightweight loop every `interval_seconds` (default 3s) and evaluates the
current active session's signals via the `SignalBuffer.metrics_since` helper.
It then uses `InferenceEngine` to produce a candidate ActivityState which is
passed to `StateManager.update` (which persists STATE_CHANGE events).

Also performs streaming task assignment: assigns tasks to active sessions
as they accumulate sufficient signal data, without waiting for completion.

The runner is efficient: single background thread, light computations, and
only reads the signal buffers for the currently active session.
"""
from datetime import datetime, timezone, timedelta
import threading
import time
from typing import Optional
from pathlib import Path
import traceback

from agent.inference.engine import InferenceEngine, InferenceContext
from agent.inference.state import StateManager, ActivityState
from agent.intent.manager import IntentManager
from agent.session.gate import get_session_gate
from agent.task.live_predictor import LiveTaskPredictor
from agent.task.inference import FeatureVector
from agent.storage.db import log_live_prediction, get_intervals
from agent.error_handling import log_component_error, ComponentType, ErrorSeverity


# Global reference to the main SessionManager instance
_main_session_manager = None
# Global reference to task inference engine
_task_inference_engine = None


def set_main_session_manager(mgr):
    """Set the global reference to the main SessionManager used by main.py"""
    global _main_session_manager
    _main_session_manager = mgr


def set_task_inference_engine(engine):
    """Set the global reference to the task inference engine for live predictions"""
    global _task_inference_engine
    _task_inference_engine = engine


class InferenceRunner(threading.Thread):
    def __init__(self, interval_seconds: int = 3, window_seconds: int = 300):
        super().__init__(daemon=True)
        self.interval = interval_seconds
        self.window = window_seconds
        self._stop = threading.Event()

        # Crash logging
        self.crash_log = Path(".agent_crash.log")

        self.engine = InferenceEngine()
        self.state_mgr = StateManager()
        self.intent_mgr = IntentManager()
        
        # Use the main session manager instance from main.py
        # (no streaming task assignment - not compatible with old architecture)

    def _log_crash(self, prefix: str, exc: Exception):
        try:
            with self.crash_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - inference_runner:{prefix}: {exc}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass

    def stop(self):
        self._stop.set()

    def evaluate_once(self, ts: Optional[datetime] = None) -> Optional[ActivityState]:
        """Run a single evaluation cycle (useful for tests)."""
        try:
            timestamp = ts or datetime.now(timezone.utc)

            global _main_session_manager
            session_mgr = _main_session_manager
            gate = get_session_gate()

            session_active = False
            session = None
            metrics = {}

            try:
                with open(".agent_inference_debug.log", "a") as f:
                    f.write(f"{timestamp.isoformat()} - evaluate_once called\n")
                    f.write(f"  session_mgr: {session_mgr is not None}\n")
            except Exception:
                pass
        except Exception as e:
            log_component_error(
                ComponentType.INFERENCE,
                "evaluate_once_init",
                e,
                ErrorSeverity.ERROR
            )
            return None

        session = None
        session_id = None
        if session_mgr and gate.is_active():
            try:
                session_id = gate.get_active_session_id()
                session = session_mgr.get_session(session_id)
            except Exception:
                session = None

            if session and getattr(session, "signals", None):
                session_active = True
                try:
                    metrics = session.signals.metrics_since(self.window)
                except Exception:
                    metrics = {}

                try:
                    with open(".agent_inference_debug.log", "a") as f:
                        f.write(f"  Session found! ID: {session_id}\n")
                        f.write(f"  Metrics: {metrics}\n")
                except Exception:
                    pass

        # Determine is_idle from metrics
        input_activity_score = 0.0
        intensity = 0.0
        active_app = None
        if metrics:
            intensity = metrics.get("intensity", 0.0) or 0.0
            input_activity_score = float(intensity) / 100.0
            active_app = metrics.get("active_app")

        is_idle = input_activity_score < 0.1
        active_task = self.intent_mgr.get_active_task_id()
        
        # Check for live task prediction (session-based, highest priority)
        live_task = None
        if session_active and session:
            try:
                from agent.storage.db import get_latest_live_prediction
                live_prediction = get_latest_live_prediction(session.session_id)
                if live_prediction and live_prediction.get('task_id'):
                    live_task = live_prediction.get('task_id')
                    # Debug: log when live task is found
                    try:
                        with open(".agent_inference_debug.log", "a") as f:
                            f.write(f"  Live task found: {live_task} (confidence: {live_prediction.get('confidence', 0):.2f})\n")
                    except Exception:
                        pass
            except Exception as live_err:
                # Debug: log errors in fetching live task
                try:
                    with open(".agent_inference_debug.log", "a") as f:
                        f.write(f"  Error fetching live task: {live_err}\n")
                except Exception:
                    pass
        
        # Check for activity-based task (recent 60s activity, not session-tied)
        activity_task = None
        try:
            from agent.storage.db import get_latest_activity_task
            activity_prediction = get_latest_activity_task()
            if activity_prediction and activity_prediction.get('task_id'):
                activity_task = activity_prediction.get('task_id')
                print(f"[ACTIVITY_TASK] Found recent activity task: {activity_task} (confidence: {activity_prediction.get('confidence', 0):.2f})")
                # Debug: log when activity task is found
                try:
                    with open(".agent_inference_debug.log", "a") as f:
                        f.write(f"  Activity task found: {activity_task} (confidence: {activity_prediction.get('confidence', 0):.2f})\n")
                except Exception:
                    pass
        except Exception as activity_err:
            # Debug: log errors in fetching activity task
            try:
                with open(".agent_inference_debug.log", "a") as f:
                    f.write(f"  Error fetching activity task: {activity_err}\n")
            except Exception:
                pass
        # Determine candidate state
        if not session_active:
            # When no session is active, prefer a stable IDLE state over UNKNOWN
            # so the UI never shows an indeterminate state.
            candidate = ActivityState.IDLE
        else:
            ctx = InferenceContext(
                timestamp=timestamp,
                is_idle=is_idle,
                active_app=active_app,
                input_activity_score=input_activity_score,
                active_task=active_task,
                session_active=session_active,
            )
            candidate = self.engine.evaluate(ctx)
            
            # DEBUG: Log candidate state
            try:
                with open(".agent_inference_debug.log", "a") as f:
                    f.write(f"  Candidate state: {candidate.name if candidate else 'None'}\n")
            except Exception:
                pass

        # Attach extra metadata
        # Task priority: session_live > activity_based > intent_based
        display_task = live_task or activity_task or active_task
        
        # Determine task source for logging
        task_source = "none"
        if live_task:
            task_source = "session"
        elif activity_task:
            task_source = "activity"
        elif active_task:
            task_source = "intent"
        
        # Log which source we're using
        try:
            with open(".agent_inference_debug.log", "a") as f:
                if live_task:
                    f.write(f"  Display task source: session-based live task\n")
                elif activity_task:
                    f.write(f"  Display task source: activity-based task\n")
                elif active_task:
                    f.write(f"  Display task source: intent-based task\n")
                else:
                    f.write(f"  Display task source: none available\n")
        except Exception:
            pass
        
        context = {
            "active_task": display_task,  # Use highest-priority task for STATE_CHANGE events
            "active_app": active_app,
            "is_idle": is_idle,
            "session_active": session_active,
            "input_activity_score": input_activity_score,
            "intensity": intensity,
            "confidence": intensity / 100.0 if intensity is not None else 0.0,
        }

        # Print intensity to console for live monitoring
        # Show task with source indicator
        task_display = f"{display_task or 'N/A'}"
        if display_task:
            task_display += f" ({task_source})"
        print(f"[INTENSITY] {timestamp.strftime('%H:%M:%S')} - Intensity: {intensity:.1f} | State: {candidate.name if candidate else 'None'} | App: {active_app or 'N/A'} | Task: {task_display}")

        try:
            committed = self.state_mgr.update(candidate, context, ts=timestamp)
            # DEBUG: Log commit result
            try:
                with open(".agent_inference_debug.log", "a") as f:
                    f.write(f"  StateManager.update() returned: {committed}\n")
                    if not committed:
                        # Log pending state info
                        if self.state_mgr.pending_state:
                            elapsed = (timestamp - self.state_mgr.pending_since).total_seconds() if self.state_mgr.pending_since else 0
                            threshold = self.state_mgr.THRESHOLDS.get(self.state_mgr.pending_state, 0)
                            f.write(f"  Pending: {self.state_mgr.pending_state.name}, elapsed: {elapsed:.0f}s, threshold: {threshold}s\n")
            except Exception:
                pass
        except Exception as update_err:
            committed = False
            self._log_crash("state_update", update_err)
            # Log error via error handling system
            try:
                from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
                log_component_error(
                    ComponentType.INFERENCE,
                    "state_update",
                    update_err,
                    ErrorSeverity.ERROR
                )
            except:
                pass
        return candidate if committed else None

    def run(self):
        # light loop that sleeps in small increments so stop() is responsive
        while not self._stop.is_set():
            try:
                self.evaluate_once()
            except Exception as loop_err:
                self._log_crash("evaluate_once", loop_err)
            # wait for interval or stop
            self._stop.wait(self.interval)


class LivePredictionRunner(threading.Thread):
    """
    Background runner for real-time live task predictions.
    
    Runs every 60 seconds on the active session and predicts the current task
    from the last 60 seconds of signals. Only starts predictions 1 minute after
    a session has been started. Persists predictions independently
    from the session finalization pipeline.
    """
    
    def __init__(self, interval_seconds: float = 60.0, window_seconds: int = 60):
        """
        Initialize live prediction runner.
        
        Args:
            interval_seconds: How often to predict (default 60s)
            window_seconds: Signal window to analyze (default 60s)
        """
        super().__init__(daemon=True)
        self.interval = interval_seconds
        self.window = window_seconds
        self._stop = threading.Event()

        # Use a fixed session id for global live window predictions
        self._live_session_id = "live-window"
        
        # Initialize predictor (will get task engine via setter)
        self.predictor = LiveTaskPredictor(
            task_inference_engine=None,  # Will be set via setter
            window_seconds=window_seconds,
            confidence_threshold=0.50,   # More permissive for live predictions
            distance_threshold=0.35,
        )
        
        # Crash logging
        self.crash_log = Path(".live_prediction_crash.log")

    def _build_feature_vector_from_intervals(self, intervals: list, now: datetime) -> Optional[FeatureVector]:
        """Build a FeatureVector from last-minute interval signals (global, not session-specific)."""
        if not intervals:
            return None

        # Sort by start time (ascending)
        def _parse_ts(val):
            try:
                if isinstance(val, datetime):
                    return val
                return datetime.fromisoformat(val)
            except Exception:
                return None

        parsed = []
        for row in intervals:
            ts_start = _parse_ts(row.get("timestamp_start"))
            ts_end = _parse_ts(row.get("timestamp_end"))
            if not ts_start:
                continue
            if ts_end and ts_end < ts_start:
                ts_end = None
            parsed.append((ts_start, ts_end, row))

        if not parsed:
            return None

        parsed.sort(key=lambda x: x[0])

        # Durations and inputs
        total_minutes = 0.0
        total_keys = 0.0
        total_clicks = 0.0
        keyboard_intensities = []
        apps = []

        for ts_start, ts_end, row in parsed:
            duration_min = None
            try:
                if ts_end:
                    duration_min = (ts_end - ts_start).total_seconds() / 60.0
            except Exception:
                duration_min = None
            if duration_min is None or duration_min <= 0:
                duration_min = 1.0 / 60.0

            total_minutes += duration_min
            kb_intensity = float(row.get("keyboard_intensity", 0.0) or 0.0)
            mouse_clicks = float(row.get("mouse_clicks", 0.0) or 0.0)
            keyboard_intensities.append(kb_intensity)

            # kb_intensity is keys/min; estimate total keys for the interval
            total_keys += kb_intensity * duration_min
            total_clicks += mouse_clicks

            app = row.get("app") or "unknown"
            apps.append(app)

        if total_minutes <= 0:
            return None

        # App diversity and context switch entropy
        unique_apps = list(dict.fromkeys(apps))
        app_diversity = min(len(set(unique_apps)) / 20.0, 1.0)

        # Context switch entropy (normalized)
        from math import log2
        entropy = 0.0
        if apps:
            counts = {}
            for a in apps:
                counts[a] = counts.get(a, 0) + 1
            total = sum(counts.values())
            for c in counts.values():
                p = c / total if total else 0.0
                if p > 0:
                    entropy -= p * log2(p)
            # Normalize by log2(n) to get [0,1]
            if len(counts) > 1:
                entropy /= log2(len(counts))
            else:
                entropy = 0.0

        # Focus continuity score (fewer switches => higher score)
        switches = 0
        for i in range(1, len(apps)):
            if apps[i] != apps[i - 1]:
                switches += 1
        switch_rate = switches / max(len(apps) - 1, 1)
        focus_continuity = max(0.0, 1.0 - min(switch_rate, 1.0))

        # Sustained focus windows (consecutive same app)
        max_streak_min = 0.0
        total_streak_min = 0.0
        streak_count = 0
        streak_start = None
        streak_app = None

        for ts_start, ts_end, row in parsed:
            app = row.get("app") or "unknown"
            if streak_app is None:
                streak_app = app
                streak_start = ts_start
                streak_end = ts_end or ts_start
            elif app == streak_app:
                streak_end = ts_end or ts_start
            else:
                if streak_start:
                    dur = (streak_end - streak_start).total_seconds() / 60.0
                    max_streak_min = max(max_streak_min, dur)
                    total_streak_min += dur
                    streak_count += 1
                streak_app = app
                streak_start = ts_start
                streak_end = ts_end or ts_start

        if streak_start:
            dur = (streak_end - streak_start).total_seconds() / 60.0
            max_streak_min = max(max_streak_min, dur)
            total_streak_min += dur
            streak_count += 1

        avg_streak_min = (total_streak_min / streak_count) if streak_count else 0.0

        # Focus consistency from keyboard intensity variance
        if keyboard_intensities:
            mean_kb = sum(keyboard_intensities) / len(keyboard_intensities)
            variance = sum((x - mean_kb) ** 2 for x in keyboard_intensities) / len(keyboard_intensities)
            focus_consistency = 1.0 / (1.0 + variance / 1000.0)
        else:
            focus_consistency = 0.5

        # Input intensity (keys + clicks per minute), normalized by 8
        input_per_minute = (total_keys + total_clicks) / total_minutes if total_minutes > 0 else 0.0
        input_intensity = min(input_per_minute / 8.0, 1.0)

        # Normalize sustained metrics and duration
        max_sustained_norm = min(max_streak_min / 300.0, 1.0)
        avg_focus_norm = min(avg_streak_min / 120.0, 1.0)
        duration_norm = min(total_minutes / 480.0, 1.0)

        return FeatureVector(
            context_switch_entropy=min(max(entropy, 0.0), 1.0),
            focus_continuity_score=min(max(focus_continuity, 0.0), 1.0),
            max_sustained_minutes=max_sustained_norm,
            avg_focus_window_minutes=avg_focus_norm,
            focus_consistency=min(max(focus_consistency, 0.0), 1.0),
            app_diversity=app_diversity,
            session_duration_minutes=duration_norm,
            input_intensity=input_intensity,
            time_of_day_hour=int(now.hour),
        )
    
    def set_task_inference_engine(self, engine):
        """Set the task inference engine for live predictions."""
        self.predictor.engine = engine
    
    def _log_crash(self, prefix: str, exc: Exception):
        try:
            with self.crash_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - live_prediction_runner:{prefix}: {exc}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
    
    def stop(self):
        self._stop.set()
    
    def predict_once(self, ts: Optional[datetime] = None) -> Optional[dict]:
        """
        Run a single prediction cycle.
        
        ONLY runs if:
        1. A session is active
        2. At least 1 minute has passed since session started
        
        Early returns on either condition - no detection attempted.
        Uses fallback to core task recognition when no centroids exist yet.
        
        Returns:
            Prediction dict if successful, None otherwise
        """
        try:
            timestamp = ts or datetime.now(timezone.utc)
            
            # GUARD: Check if session is active first, before any work
            global _main_session_manager
            session_mgr = _main_session_manager
            gate = get_session_gate()
            
            if not session_mgr or not gate or not gate.is_active():
                # No active session - do not attempt prediction
                return None
            
            # Get the active session
            try:
                session_id = gate.get_active_session_id()
                session = session_mgr.get_session(session_id)
                if not session or not session.started_at:
                    # Invalid session - do not attempt prediction
                    return None
                    
                print(f"[LIVE_TASK] Retrieved session {session_id}, started_at: {session.started_at}")
            except Exception as sess_err:
                print(f"[LIVE_TASK] ERROR getting session: {sess_err}")
                return None
            
            # GUARD: Check if minimum time has passed (59.5s threshold for variance)
            time_since_start = (timestamp - session.started_at).total_seconds()
            if time_since_start < 59.5:
                # Not enough time - do not attempt prediction
                print(f"[LIVE_TASK] Skipped: Only {int(time_since_start)}s elapsed, need ~60s")
                return None
            
            print(f"[LIVE_TASK] Running prediction for session {session_id} ({int(time_since_start)}s elapsed)")
            
            # Use last-minute interval signals for the active session
            start_time = timestamp - timedelta(seconds=self.window)
            try:
                intervals = get_intervals(start_time=start_time, end_time=timestamp, session_id=session_id, limit=500)
                print(f"[LIVE_TASK] Found {len(intervals)} intervals for analysis")
            except Exception as interval_err:
                print(f"[LIVE_TASK] ERROR fetching intervals: {interval_err}")
                intervals = []

            if not intervals:
                print(f"[LIVE_TASK] No interval data available yet, skipping prediction")
                return None

            feature_vec = self._build_feature_vector_from_intervals(intervals, timestamp)
            if not feature_vec:
                print(f"[LIVE_TASK] Could not build feature vector from intervals")
                return None
            
            print(f"[LIVE_TASK] Built feature vector successfully")

            # Check if we have centroids - if not, use core task recognition fallback
            if not self.predictor.engine or not self.predictor.engine.task_centroids:
                print(f"[LIVE_TASK] No centroids available, using core task fallback")
                # Fallback to rule-based core task recognition
                try:
                    from agent.task.core_tasks import get_task_recommendation, _build_contextual_task_id
                    from agent.task.live_predictor import LiveTaskPrediction
                    
                    # Convert feature vector to dict for core tasks
                    rolling_features = {
                        "active_app": intervals[-1].get("app") if intervals else "unknown",
                        "active_window_title": intervals[-1].get("window_title", "") if intervals else "",
                        "continuity": feature_vec.focus_continuity_score,
                        "intensity": feature_vec.input_intensity,
                        "entropy": feature_vec.context_switch_entropy,
                        "app_diversity": feature_vec.app_diversity,
                        "event_count": len(intervals),
                    }
                    
                    print(f"[LIVE_TASK] Calling get_task_recommendation with app={rolling_features['active_app']}")
                    
                    # Get task recommendation
                    base_task_id, confidence, reason = get_task_recommendation(rolling_features)
                    
                    print(f"[LIVE_TASK] Recommendation: {base_task_id} (confidence: {confidence:.2f}, reason: {reason})")
                    
                    # Build contextual name
                    smart_task_id = _build_contextual_task_id(base_task_id, rolling_features)
                    
                    print(f"[LIVE_TASK] Smart task ID: {smart_task_id}")
                    
                    # Create prediction with fallback
                    prediction = LiveTaskPrediction(
                        timestamp=timestamp.isoformat(),
                        session_id=session_id,
                        task_id=smart_task_id,
                        confidence=confidence,
                        distance_to_centroid=1.0 - confidence,
                        reason=f"core_fallback_{reason}",
                        feature_window_seconds=self.window,
                        feature_vector=rolling_features,
                        alternative_tasks=[],
                    )
                    
                    # Persist to database
                    try:
                        log_live_prediction(prediction)
                        print(f"[LIVE_TASK] Saved prediction: {smart_task_id} (confidence: {confidence:.2f}, fallback)")
                    except Exception as db_err:
                        print(f"[LIVE_TASK] ERROR saving prediction: {db_err}")
                        self._log_crash("persist_prediction", db_err)
                    
                    return prediction.to_dict()
                    
                except Exception as fallback_err:
                    print(f"[LIVE_TASK] ERROR in core fallback: {fallback_err}")
                    self._log_crash("core_fallback", fallback_err)
                    return None
            
            # Use centroid-based prediction if centroids exist
            print(f"[LIVE_TASK] Using centroid-based prediction")
            prediction = self.predictor.predict(
                session_id=session_id,
                rolling_features=feature_vec,
                timestamp=timestamp,
            )
            
            if not prediction:
                print(f"[LIVE_TASK] Centroid predictor returned None")
                return None
            
            # Persist to database
            try:
                log_live_prediction(prediction)
                print(f"[LIVE_TASK] Saved prediction: {prediction.task_id} (confidence: {prediction.confidence:.2f}, centroid)")
            except Exception as db_err:
                self._log_crash("persist_prediction", db_err)
            
            return prediction.to_dict()
            
        except Exception as e:
            print(f"[LIVE_TASK] ERROR in predict_once: {e}")
            self._log_crash("predict_once", e)
            log_component_error(
                ComponentType.ML,
                "live_predict_once",
                e,
                ErrorSeverity.WARNING
            )
            return None
    
    def run(self):
        """Main loop: predict every N seconds.
        
        Only calls predict_once() if:
        1. An active session exists
        2. At least 1 minute has elapsed since session started
        
        Otherwise, skips the call entirely to avoid unnecessary function overhead.
        """
        while not self._stop.is_set():
            try:
                # Pre-check: only attempt prediction if conditions are met
                gate = get_session_gate()
                if gate and gate.is_active():
                    # Session is active, now check if enough time has passed
                    global _main_session_manager
                    session_mgr = _main_session_manager
                    if session_mgr:
                        try:
                            session_id = gate.get_active_session_id()
                            session = session_mgr.get_session(session_id)
                            if session and session.started_at:
                                time_elapsed = (datetime.now(timezone.utc) - session.started_at).total_seconds()
                                if time_elapsed >= 59.5:
                                    # Both conditions met: call prediction
                                    self.predict_once()
                        except Exception:
                            # If we can't get session info, skip this cycle
                            pass
            except Exception as loop_err:
                print(f"[LIVE_TASK] ERROR in run loop: {loop_err}")
                self._log_crash("predict_loop", loop_err)
            
            # Wait for interval or stop signal
            self._stop.wait(self.interval)

