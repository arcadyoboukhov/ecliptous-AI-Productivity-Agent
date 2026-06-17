"""
Recent Activity Task Runner - Detects current task from last 60 seconds of activity.

Independent of sessions - runs continuously and tracks what the user is currently doing
in real-time based on the last minute of interval signals.

This complements LivePredictionRunner which is session-aware.
- Session-based tasks: tied to active session timeline
- Activity-based tasks: tied to the last 60 seconds of user activity, regardless of session

Both run independently and the UI can display either or both.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading
import time
from typing import Optional, Dict
import traceback

from agent.task.inference import FeatureVector
from agent.task.live_predictor import LiveTaskPrediction
from agent.storage.db import log_live_prediction, get_intervals
from agent.error_handling import log_component_error, ComponentType, ErrorSeverity


class RecentActivityTaskRunner(threading.Thread):
    """
    Background runner for detecting tasks from recent user activity.
    
    Key differences from LivePredictionRunner:
    - Does NOT require an active session
    - Analyzes the last 60 seconds of activity regardless of session state
    - Runs every 60 seconds
    - Persists with source='activity' instead of being session-tied
    
    Use cases:
    - Detect task during idle time before session starts
    - Detect task after session ends
    - Track parallel activity (multiple apps, multiple tasks)
    """
    
    def __init__(self, interval_seconds: float = 60.0, window_seconds: int = 60):
        """
        Initialize recent activity task runner.
        
        Args:
            interval_seconds: How often to detect activity tasks (default 60s)
            window_seconds: Activity window to analyze (default 60s)
        """
        super().__init__(daemon=True)
        self.interval = interval_seconds
        self.window = window_seconds
        self._stop = threading.Event()
        
        # Initialize predictor (will get task engine via setter)
        self.task_inference_engine = None
        
        # Crash logging
        self.crash_log = Path(".recent_activity_crash.log")
    
    def set_task_inference_engine(self, engine):
        """Set the task inference engine after initialization."""
        self.task_inference_engine = engine
    
    def _log_crash(self, prefix: str, exc: Exception):
        try:
            with self.crash_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - recent_activity:{prefix}: {exc}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
    
    def stop(self):
        self._stop.set()
    
    def _build_feature_vector_from_intervals(self, intervals: list, now: datetime) -> Optional[FeatureVector]:
        """
        Build a feature vector from interval signals.
        Same logic as LivePredictionRunner but for generic activity.
        """
        if not intervals:
            return None
        
        # Parse timestamps safely
        parsed = []
        for row in intervals:
            try:
                ts_start = row.get("timestamp_start")
                ts_end = row.get("timestamp_end")
                if isinstance(ts_start, str):
                    ts_start = datetime.fromisoformat(ts_start.replace('Z', '+00:00'))
                if isinstance(ts_end, str):
                    ts_end = datetime.fromisoformat(ts_end.replace('Z', '+00:00'))
                parsed.append((ts_start, ts_end, row))
            except Exception:
                continue
        
        if not parsed:
            return None
        
        # Calculate duration in minutes
        first_ts = parsed[0][0]
        last_ts = parsed[-1][1] or parsed[-1][0]
        total_minutes = (last_ts - first_ts).total_seconds() / 60.0
        if total_minutes == 0:
            total_minutes = 1.0
        
        # Aggregate metrics
        total_keys = sum(row.get("keyboard_keys", 0) for _, _, row in parsed)
        total_clicks = sum(row.get("mouse_clicks", 0) for _, _, row in parsed)
        keyboard_intensities = [row.get("keyboard_intensity", 0.0) for _, _, row in parsed if row.get("keyboard_intensity")]
        
        # Context switches (distinct apps)
        apps = set()
        for _, _, row in parsed:
            app = row.get("app")
            if app:
                apps.add(app)
        
        context_switches = len(apps) - 1 if len(apps) > 0 else 0
        context_switch_entropy = min(context_switches / 5.0, 1.0)
        
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
        
        # Focus consistency
        if keyboard_intensities:
            mean_kb = sum(keyboard_intensities) / len(keyboard_intensities)
            variance = sum((x - mean_kb) ** 2 for x in keyboard_intensities) / len(keyboard_intensities)
            focus_consistency = 1.0 / (1.0 + variance / 1000.0)
        else:
            focus_consistency = 0.5
        
        # Input intensity
        input_per_minute = (total_keys + total_clicks) / total_minutes if total_minutes > 0 else 0.0
        input_intensity = min(input_per_minute / 8.0, 1.0)
        
        # Normalize sustained metrics
        max_sustained_norm = min(max_streak_min / 300.0, 1.0)
        avg_focus_norm = min(avg_streak_min / 120.0, 1.0)
        duration_norm = min(total_minutes / 480.0, 1.0)
        
        # Get time of day
        time_of_day = first_ts.hour if first_ts else 12
        
        # App diversity (normalized by max typical apps)
        app_diversity_norm = min(len(apps) / 10.0, 1.0)
        
        # Build feature vector with correct parameter names
        feature_vec = FeatureVector(
            context_switch_entropy=context_switch_entropy,
            focus_continuity_score=focus_consistency,
            max_sustained_minutes=max_sustained_norm,
            avg_focus_window_minutes=avg_focus_norm,
            focus_consistency=focus_consistency,
            app_diversity=app_diversity_norm,
            session_duration_minutes=duration_norm,
            input_intensity=input_intensity,
            time_of_day_hour=time_of_day,
        )
        
        return feature_vec
    
    def detect_once(self, ts: Optional[datetime] = None) -> Optional[dict]:
        """
        Run a single activity detection cycle.
        
        Does NOT require a session - analyzes the last minute of activity.
        
        Returns:
            Prediction dict if successful, None otherwise
        """
        try:
            timestamp = ts or datetime.now(timezone.utc)
            
            # Fetch last minute of interval signals (session-independent)
            start_time = timestamp - timedelta(seconds=self.window)
            end_time = timestamp
            
            print(f"[RECENT_ACTIVITY] Detecting activity from {self.window}s window")
            
            try:
                # Get intervals without filtering by session_id
                intervals = get_intervals(start_time=start_time, end_time=end_time, session_id=None, limit=500)
                print(f"[RECENT_ACTIVITY] Query range: {start_time} to {end_time}")
            except Exception as interval_err:
                print(f"[RECENT_ACTIVITY] ERROR fetching intervals: {interval_err}")
                return None
            
            if not intervals or len(intervals) == 0:
                # No activity in the last minute
                print(f"[RECENT_ACTIVITY] No interval data in last {self.window}s")
                return None
            
            print(f"[RECENT_ACTIVITY] Found {len(intervals)} intervals in last {self.window}s")
            
            # Build feature vector from activity
            feature_vec = self._build_feature_vector_from_intervals(intervals, timestamp)
            if not feature_vec:
                print(f"[RECENT_ACTIVITY] Could not build feature vector from intervals")
                return None
            
            # Get active app and window title from latest interval
            active_app = intervals[-1].get("app", "unknown") if intervals else "unknown"
            active_window_title = intervals[-1].get("window_title", "") if intervals else ""
            
            print(f"[RECENT_ACTIVITY] Built feature vector: app={active_app}, intensity={feature_vec.input_intensity:.2f}")
            
            # Check if we have centroids
            if not self.task_inference_engine or not self.task_inference_engine.task_centroids:
                print(f"[RECENT_ACTIVITY] No centroids available, using core task fallback")
                
                # Fallback to rule-based core task recognition
                try:
                    from agent.task.core_tasks import get_task_recommendation, _build_contextual_task_id
                    
                    rolling_features = {
                        "active_app": active_app,
                        "active_window_title": active_window_title,
                        "continuity": feature_vec.focus_continuity_score,
                        "intensity": feature_vec.input_intensity,
                        "entropy": feature_vec.context_switch_entropy,
                    }
                    
                    # Get core task recommendation (returns tuple: task_id, confidence, reason)
                    task_id, confidence, reason = get_task_recommendation(rolling_features)
                    
                    # Create smart task ID using the features dict
                    smart_task_id = _build_contextual_task_id(base_task_id=task_id, features=rolling_features)
                    
                    prediction = LiveTaskPrediction(
                        timestamp=timestamp.isoformat(),
                        session_id="activity",  # Special marker for activity-based (not session-tied)
                        task_id=smart_task_id,
                        confidence=confidence,
                        distance_to_centroid=1.0 - confidence,
                        reason=f"activity_fallback_{reason}",
                        feature_window_seconds=self.window,
                        feature_vector=rolling_features,
                        alternative_tasks=[],
                    )
                    
                    # Persist to database
                    try:
                        log_live_prediction(prediction)
                        print(f"[RECENT_ACTIVITY] Saved: {smart_task_id} (confidence: {confidence:.2f})")
                    except Exception as db_err:
                        print(f"[RECENT_ACTIVITY] ERROR saving: {db_err}")
                        self._log_crash("persist_prediction", db_err)
                    
                    return prediction.to_dict()
                    
                except Exception as fallback_err:
                    print(f"[RECENT_ACTIVITY] ERROR in fallback: {fallback_err}")
                    self._log_crash("fallback", fallback_err)
                    return None
            
            # Use centroid-based prediction if available
            print(f"[RECENT_ACTIVITY] Using centroid-based prediction")
            try:
                task_id, confidence = self.task_inference_engine.infer_task(
                    session_id="activity",
                    feature_vector=feature_vec,
                    extra_metadata={"source": "recent_activity"},
                )
                
                if not task_id:
                    print(f"[RECENT_ACTIVITY] Inference returned no task")
                    return None
                
                # Create prediction object
                prediction = LiveTaskPrediction(
                    timestamp=timestamp.isoformat(),
                    session_id="activity",
                    task_id=task_id,
                    confidence=confidence,
                    distance_to_centroid=1.0 - confidence,
                    reason="centroid_match",
                    feature_window_seconds=self.window,
                    feature_vector=None,  # Don't store full vector
                )
                
                # Persist to database
                try:
                    log_live_prediction(prediction)
                    print(f"[RECENT_ACTIVITY] Saved: {task_id} (confidence: {confidence:.2f})")
                except Exception as db_err:
                    self._log_crash("persist_prediction", db_err)
                
                return prediction.to_dict()
            except Exception as predict_err:
                print(f"[RECENT_ACTIVITY] ERROR in centroid prediction: {predict_err}")
                self._log_crash("centroid_predict", predict_err)
                # Fall back to core tasks if centroid fails
                return None
            
        except Exception as e:
            print(f"[RECENT_ACTIVITY] ERROR: {e}")
            self._log_crash("detect_once", e)
            return None
    
    def run(self):
        """Main loop: detect every N seconds."""
        print(f"[RECENT_ACTIVITY] Runner thread started (interval={self.interval}s, window={self.window}s)")
        iteration = 0
        while not self._stop.is_set():
            try:
                iteration += 1
                print(f"[RECENT_ACTIVITY] Iteration {iteration} starting...")
                self.detect_once()
            except Exception as loop_err:
                print(f"[RECENT_ACTIVITY] ERROR in loop: {loop_err}")
                self._log_crash("loop", loop_err)
            
            # Wait for interval or stop signal
            self._stop.wait(self.interval)
        
        print(f"[RECENT_ACTIVITY] Runner thread stopped")
