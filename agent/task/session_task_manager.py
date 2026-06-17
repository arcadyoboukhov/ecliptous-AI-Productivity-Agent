"""
Integration of Online Task Classification into SessionManager

This module provides the glue code between:
1. SessionManager (session lifecycle management)
2. OnlineTaskClassifier (real-time task assignment)
3. TaskInferenceEngine (task clustering)
4. Main agent loop (periodic updates)

Key responsibilities:
- Initialize online classifier when session starts
- Update task assignments periodically while session is active
- Finalize task segments when session ends
- Maintain task transition history
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
import logging
import sys

logger = logging.getLogger(__name__)

# Debug file for task classification
_debug_file = None
def _write_debug(msg):
    global _debug_file
    if _debug_file is None:
        try:
            import os
            log_dir = os.path.expanduser("~/.productivity_agent")
            os.makedirs(log_dir, exist_ok=True)
            _debug_file = open(os.path.join(log_dir, "task_debug.log"), "a")
        except:
            _debug_file = False  # Mark as failed
    
    if _debug_file and _debug_file is not False:
        try:
            _debug_file.write(f"{datetime.now().isoformat()} {msg}\n")
            _debug_file.flush()
        except:
            pass


class SessionTaskManager:
    """
    Manages online task classification for a session.
    
    Coordinates between:
    - SessionManager (provides sessions to track)
    - OnlineTaskClassifier (performs classifications)
    - TaskInferenceEngine (provides task centroids)
    """
    
    def __init__(self, session_manager, task_inference_engine, online_classifier=None):
        """
        Initialize task manager.
        
        Args:
            session_manager: SessionManager instance
            task_inference_engine: TaskInferenceEngine with trained centroids
            online_classifier: OnlineTaskClassifier instance (created if None)
        """
        self.session_manager = session_manager
        self.task_engine = task_inference_engine
        
        # Lazy import to avoid circular dependencies
        if online_classifier is None:
            try:
                from agent.task.online_classification import OnlineTaskClassifier
                online_classifier = OnlineTaskClassifier(task_inference_engine)
            except ImportError:
                logger.warning("OnlineTaskClassifier not available")
                online_classifier = None
        
        self.classifier = online_classifier
        self._last_classification_time: Dict[str, datetime] = {}  # Track per-session
        self._classification_interval = timedelta(seconds=10)  # Classify every 10 seconds
    
    def update_active_session_task_assignment(self, now: Optional[datetime] = None) -> Optional[Dict]:
        """
        Perform online task classification for the currently active session.
        
        This is called periodically (every 10 seconds) while a session is active.
        
        Args:
            now: Current timestamp (default: now)
            
        Returns:
            Classification result dict or None if no active session
        """
        if not self.classifier:
            _write_debug("[TASK] No classifier available")
            return None
        
        now = now or datetime.now(timezone.utc)
        session = self.session_manager.current_session
        
        if not session:
            _write_debug("[TASK] No active session")
            return None
        
        session_id = getattr(session, 'id', None) or getattr(session, 'session_id', None)
        if not session_id:
            _write_debug("[TASK] No session_id")
            return None
        
        _write_debug(f"[TASK] Processing session {session_id}")
        
        # Check if enough time has passed since last classification
        last_time = self._last_classification_time.get(session_id)
        if last_time and (now - last_time) < self._classification_interval:
            _write_debug(f"[TASK] Too soon to classify (last: {last_time})")
            return None
        
        _write_debug(f"[TASK] Need to classify session")
        
        # Initialize classifier for this session if needed
        if session_id not in self.classifier.session_states:
            start_time = getattr(session, 'start', None) or getattr(session, 'started_at', now)
            self.classifier.initialize_session(session_id, start_time)
        
        # Compute rolling features from session data
        rolling_features = self._extract_rolling_features(session, now)
        
        if not rolling_features:
            _write_debug(f"[TASK] No rolling features extracted!")
            return None
        
        _write_debug(f"[TASK] Got rolling features: {list(rolling_features.keys())}")
        
        try:
            # Perform online classification
            _write_debug(f"[TASK] Calling classifier.classify_active_session()")
            result = self.classifier.classify_active_session(
                session_id, 
                rolling_features,
                timestamp=now
            )
            
            _write_debug(f"[TASK] Classification result: {result}")
            
            # Update session with result
            if result:
                _write_debug(f"[TASK] Updating session with classification result")
                self._update_session_with_classification(session, result, rolling_features)
            else:
                _write_debug(f"[TASK] No result from classifier")
            
            self._last_classification_time[session_id] = now
            
            return {
                "task_id": result.assigned_task_id if result else None,
                "confidence": result.confidence if result else 0,
                "distance": result.distance_to_centroid if result else None,
                "transition": result.has_transition if result else False,
                "reason": result.reason,
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during online task classification: {e}", exc_info=True)
            return None
    
    def finalize_session_tasks(self, session_id: str, end_time: datetime) -> list:
        """
        Finalize task segments when session ends.
        
        Called by session manager when a session completes.
        
        Args:
            session_id: ID of session that ended
            end_time: When session ended
            
        Returns:
            List of TaskSegment objects
        """
        if not self.classifier:
            return []
        
        try:
            segments = self.classifier.finalize_session(session_id, end_time)
            return segments
        except Exception as e:
            logger.error(f"Error finalizing session tasks: {e}", exc_info=True)
            return []
    
    def get_session_task_summary(self, session_id: str) -> Dict:
        """Get current task assignment summary for a session."""
        if not self.classifier:
            return {}
        
        try:
            return self.classifier.get_session_task_summary(session_id)
        except Exception as e:
            logger.error(f"Error getting session task summary: {e}", exc_info=True)
            return {}
    
    def _extract_rolling_features(self, session: Any, now: datetime) -> Dict:
        """
        Extract rolling (cumulative) features from session data.
        
        Features are based on activity accumulated in the session so far.
        """
        try:
            # Get timeline data - first try session.timeline
            timeline = getattr(session, 'timeline', {})
            
            # If session.timeline is empty, try to get it from SignalBuffer
            if not timeline:
                _write_debug(f"[TASK] _extract_rolling_features: session.timeline empty, checking SignalBuffer")
                signal_buffer = getattr(session, 'signals', None)
                if signal_buffer:
                    activity_timeline = getattr(signal_buffer, 'activity_timeline', {})
                    _write_debug(f"[TASK] _extract_rolling_features: SignalBuffer.activity_timeline size={len(activity_timeline)}")
                    if activity_timeline:
                        timeline = activity_timeline
                        # Populate session.timeline for future access
                        session.timeline = timeline
            
            _write_debug(f"[TASK] _extract_rolling_features: timeline type={type(timeline)}, size={len(timeline) if timeline else 0}")
            
            if not timeline:
                _write_debug(f"[TASK] _extract_rolling_features: No timeline data (checked both session.timeline and SignalBuffer)")
                return {}
            
            # Collect recent timeline entries (last 10 minutes)
            recent_cutoff = now - timedelta(minutes=10)
            recent_entries = []
            
            for key, value in timeline.items():
                try:
                    # Handle both datetime objects and ISO strings
                    if isinstance(key, datetime):
                        entry_time = key
                    elif isinstance(key, str):
                        entry_time = datetime.fromisoformat(key)
                    else:
                        continue
                    
                    if entry_time >= recent_cutoff:
                        recent_entries.append(value)
                except (ValueError, TypeError) as e:
                    _write_debug(f"[TASK] _extract_rolling_features: Error parsing timeline key {key}: {e}")
                    continue
            
            # If no recent entries, use last 10 timeline entries
            if not recent_entries and timeline:
                recent_entries = list(timeline.values())[-10:]
                _write_debug(f"[TASK] _extract_rolling_features: No recent entries, using last 10")
            
            if not recent_entries:
                _write_debug(f"[TASK] _extract_rolling_features: No recent entries found")
                return {}
            
            _write_debug(f"[TASK] _extract_rolling_features: Found {len(recent_entries)} recent entries")
            
            # Calculate aggregate features
            total_keys = sum(v.get("keys", 0) for v in recent_entries if isinstance(v, dict))
            total_clicks = sum(v.get("clicks", 0) for v in recent_entries if isinstance(v, dict))
            total_mouse = sum(v.get("mouse_distance", 0) for v in recent_entries if isinstance(v, dict))
            
            # Intensity: input events per minute
            num_buckets = len(recent_entries)
            total_input = total_keys + total_clicks
            intensity = min(total_input / (num_buckets * 60) if num_buckets else 0, 1.0)
            
            # Duration since session start
            start_time = getattr(session, 'start', None) or getattr(session, 'started_at', now)
            duration_minutes = (now - start_time).total_seconds() / 60
            duration_norm = min(duration_minutes / 60, 1.0)  # Normalize to 1 hour
            
            # App diversity (number of distinct apps used)
            apps = getattr(session, 'apps', set())
            app_diversity = min(len(apps) / 5, 1.0)  # Normalize to 5 apps
            
            # Continuity: how steady the activity is
            if num_buckets > 0:
                avg_input = total_input / num_buckets
                variance = sum(
                    ((v.get("keys", 0) + v.get("clicks", 0)) - avg_input) ** 2 
                    for v in recent_entries if isinstance(v, dict)
                ) / num_buckets
                # Lower variance = higher continuity
                continuity = 1.0 / (1.0 + variance / 1000)
            else:
                continuity = 0.5

            # Active app/window (last observed)
            active_app = getattr(session, "last_app", None)
            active_window_title = getattr(session, "last_window_title", None)
            
            return {
                "intensity": intensity,
                "continuity": continuity,
                "app_diversity": app_diversity,
                "duration": duration_norm,
                "total_input": total_input,
                "num_windows": num_buckets,
                "timestamp": now.isoformat(),
                "active_app": active_app,
                "active_window_title": active_window_title
            }
            
        except Exception as e:
            logger.error(f"Error extracting rolling features: {e}", exc_info=True)
            return {}
    
    def _update_session_with_classification(self, session: Any, result: Any, features: Dict):
        """
        Update session object with classification result.
        
        Stores current task assignment and historical data.
        """
        try:
            _write_debug(f"[TASK] _update_session_with_classification called for task {result.assigned_task_id}")
            
            # Update current assignment
            session.current_task_assignment = {
                "task_id": result.assigned_task_id,
                "confidence": result.confidence,
                "distance": result.distance_to_centroid,
                "reason": result.reason,
                "features": features,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            _write_debug(f"[TASK] Set current_task_assignment to {result.assigned_task_id} (conf={result.confidence})")
            
            # Store in history
            if not hasattr(session, 'task_classification_history'):
                session.task_classification_history = []
            
            session.task_classification_history.append({
                "task_id": result.assigned_task_id,
                "confidence": result.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            _write_debug(f"[TASK] Added to history, now {len(session.task_classification_history)} items")
            
            # Update intra-session task segments from classifier
            session_id = getattr(session, 'id', None) or getattr(session, 'session_id', None)
            if session_id and session_id in self.classifier.session_states:
                state = self.classifier.session_states[session_id]
                segments = state.get("task_segments", [])
                
                # Convert TaskSegment objects to dicts for serialization
                if not hasattr(session, 'intra_session_tasks'):
                    session.intra_session_tasks = []
                
                # Update with latest segments
                session.intra_session_tasks = segments
                _write_debug(f"[TASK] Updated intra_session_tasks with {len(segments)} segments, state has current_task_id={state.get('current_task_id')}")
            else:
                _write_debug(f"[TASK] No classifier state found for session {session_id} (in_states={session_id in self.classifier.session_states if self.classifier else 'no classifier'})")

            
        except Exception as e:
            logger.error(f"Error updating session with classification: {e}", exc_info=True)
            _write_debug(f"[TASK] ERROR in _update_session_with_classification: {e}")


class TaskTransitionTracker:
    """
    Tracks transitions between tasks within a session.
    
    Provides insights like:
    - When did you switch from Task A to Task B?
    - How confident was the transition?
    - Why did the transition happen?
    """
    
    def __init__(self, confidence_threshold: float = 0.60):
        """
        Initialize tracker.
        
        Args:
            confidence_threshold: Minimum confidence to consider assignment valid
        """
        self.confidence_threshold = confidence_threshold
        self.transitions: Dict[str, list] = {}  # session_id -> [(from_task, to_task, time, reason)]
    
    def detect_transition(self, session_id: str, prev_task_id: Optional[str], 
                         curr_task_id: str, confidence: float, now: datetime) -> bool:
        """
        Detect if a task transition occurred.
        
        Returns True if this is a significant task change.
        """
        # No previous task = not a transition
        if not prev_task_id:
            return False
        
        # Same task = no transition
        if prev_task_id == curr_task_id:
            return False
        
        # Low confidence = not significant
        if confidence < self.confidence_threshold:
            return False
        
        # This is a transition
        self._record_transition(session_id, prev_task_id, curr_task_id, now)
        return True
    
    def _record_transition(self, session_id: str, from_task: str, to_task: str, now: datetime):
        """Record a task transition."""
        if session_id not in self.transitions:
            self.transitions[session_id] = []
        
        self.transitions[session_id].append({
            "from": from_task,
            "to": to_task,
            "timestamp": now,
            "iso_timestamp": now.isoformat()
        })
    
    def get_transitions(self, session_id: str) -> list:
        """Get all transitions for a session."""
        return self.transitions.get(session_id, [])
    
    def get_transition_count(self, session_id: str) -> int:
        """Get number of transitions in a session."""
        return len(self.get_transitions(session_id))
