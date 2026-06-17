"""
Real-Time Task Classification for Active Sessions

Implements online soft task assignment and task segmentation within sessions.
Allows tracking of task transitions and intra-session behavioral changes.

Key concepts:
- Session: Temporal container (bounded by idle gaps)
- Task: Behavioral pattern (coding, meetings, etc.)
- Multiple tasks can occur within one session
- Online classification: Assign task while session is still running
- Soft assignment: Confidence-tracked partial assignment
- Task segments: Record when and why task changed
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict
import math

from agent.task.segmentation_config import get_segmentation_config
from agent.task.inference import _normalize_extra_metadata


@dataclass
class TaskSegment:
    """
    A period of time in a session where a specific task was detected.
    
    Represents an intra-session task assignment with full metadata for ML.
    """
    task_id: str                    # Smart contextual name (for display)
    start_time: datetime            # When this segment started
    end_time: Optional[datetime]    # When it ended (None if still active)
    confidence: float               # [0.0, 1.0] confidence in assignment
    feature_vector: Optional[dict]  # Snapshot of features at assignment time
    reason: str                     # Why this task was assigned (e.g., "distance_match", "transition")
    distance_to_centroid: float    # How far from centroid (0.0 = perfect match)
    
    # Step 2: Baseline classification metadata (for ML bootstrapping)
    base_category: Optional[str] = None      # Generic category (e.g., "administrative_work")
    app: Optional[str] = None                # Raw app/process name (e.g., "firefox.exe")
    window_title: Optional[str] = None       # Raw window title (before normalization)
    normalized_title: Optional[str] = None   # Cleaned window title (after normalization)
    
    @property
    def duration_seconds(self) -> float:
        """Duration of this task segment in seconds."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """True if this segment is still ongoing."""
        return self.end_time is None
    
    def to_baseline_json(self) -> dict:
        """Export segment as baseline JSON format for ML training."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "app": self.app,
            "window_title": self.window_title,
            "normalized_title": self.normalized_title,
            "generic_task": self.base_category,
            "confidence": self.confidence,
            "feature_snapshot": self.feature_vector,
            "smart_name": self.task_id,
        }


@dataclass
class OnlineClassificationResult:
    """Result of online task classification."""
    assigned_task_id: str           # Current best-match task
    confidence: float               # Confidence in assignment [0.0, 1.0]
    distance_to_centroid: float    # Distance to best centroid
    alternative_tasks: List[Tuple[str, float]]  # Other candidate tasks with distances
    has_transition: bool            # True if task changed from previous
    reason: str                     # Why this task was assigned
    
    def __repr__(self):
        return (f"OnlineClassification(task={self.assigned_task_id[:12]}, "
                f"conf={self.confidence:.2f}, distance={self.distance_to_centroid:.3f})")


class OnlineTaskClassifier:
    """
    Real-time task classification for active sessions.
    
    Maintains task centroids and performs online soft assignment.
    Tracks task transitions and confidence evolution.
    """
    
    def __init__(self, task_engine, distance_threshold: float = 0.35, 
                 confidence_threshold: float = 0.60, transition_threshold: float = 0.15):
        """
        Initialize online classifier.
        
        Args:
            task_engine: TaskInferenceEngine with trained centroids
            distance_threshold: Max distance for assignment
            confidence_threshold: Min confidence to use existing task
            transition_threshold: Distance delta to trigger transition alert
        """
        self.task_engine = task_engine
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.transition_threshold = transition_threshold
        
        # Track per-session state
        self.session_states: Dict[str, dict] = {}  # session_id -> state dict
    
    def initialize_session(self, session_id: str, start_time: datetime):
        """Initialize tracking for a new session."""
        self.session_states[session_id] = {
            "start_time": start_time,
            "task_segments": [],           # List of TaskSegment objects
            "current_task_id": None,       # Currently assigned task
            "current_confidence": 0.0,     # Current confidence
            "previous_task_id": None,      # Previous task (for transition detection)
            "last_classification_time": start_time,
            "classification_history": [],  # All classifications with timestamps
        }
    
    def classify_active_session(self, session_id: str, rolling_features: dict,
                               timestamp: Optional[datetime] = None) -> OnlineClassificationResult:
        """
        Perform online classification on an active session.
        
        This is called while the session is still running.
        Returns soft assignment with confidence.
        
        Args:
            session_id: Session being classified
            rolling_features: Current rolling feature vector (partial data)
            timestamp: When classification occurred (default: now)
            
        Returns:
            OnlineClassificationResult with assignment and confidence
        """
        if session_id not in self.session_states:
            raise ValueError(f"Session {session_id} not initialized for online classification")
        
        timestamp = timestamp or datetime.now(timezone.utc)
        state = self.session_states[session_id]
        
        # Convert rolling features to comparable format
        # (This would normally use the actual feature vector from task_engine)
        if not rolling_features:
            return OnlineClassificationResult(
                assigned_task_id="unknown",
                confidence=0.0,
                distance_to_centroid=1.0,
                alternative_tasks=[],
                has_transition=False,
                reason="no_features_yet"
            )
        
        # Find nearest centroid
        if not self.task_engine.task_centroids:
            # Use core task recognition system
            from agent.task.core_tasks import (
                get_task_recommendation, 
                should_transition,
                extract_task_features
            )
            
            # Get recommended task based on behavioral patterns
            recommended_task_id, recommended_confidence, base_reason = get_task_recommendation(rolling_features)
            
            # Build smart contextual name
            from agent.task.core_tasks import _build_contextual_task_id
            from agent.task.smart_naming import normalize_window_title
            
            smart_task_id = _build_contextual_task_id(recommended_task_id, rolling_features)
            
            # Extract raw metadata for baseline
            raw_app = rolling_features.get("active_app", "")
            raw_window = rolling_features.get("active_window_title", "")
            normalized_window = normalize_window_title(raw_window) if raw_window else ""
            
            # Check if we should transition from current task
            has_transition = False
            result_task_id = smart_task_id
            result_confidence = recommended_confidence
            base_category = recommended_task_id  # Keep base generic category
            
            # Get segmentation config
            seg_config = get_segmentation_config()
            
            if state["task_segments"] and state["current_task_id"]:
                # Calculate how long we've been in current task
                last_segment_start = state["task_segments"][-1].start_time
                current_duration_seconds = (timestamp - last_segment_start).total_seconds()
                current_duration_minutes = current_duration_seconds / 60
                
                # Get context from last segment
                last_segment = state["task_segments"][-1]
                last_base_category = last_segment.base_category
                last_app = last_segment.app or ""
                last_window = last_segment.window_title or ""
                
                # Check for context change (app/window change)
                context_changed, context_reason = seg_config.should_split_on_context_change(
                    old_app=last_app,
                    new_app=raw_app,
                    old_window=last_window,
                    new_window=raw_window,
                    segment_duration_seconds=current_duration_seconds,
                    new_confidence=recommended_confidence
                )
                
                # Check behavioral category transition
                behavioral_transition = should_transition(
                    last_base_category or state["current_task_id"],
                    recommended_task_id,
                    state.get("current_confidence", 0.0),
                    recommended_confidence,
                    current_duration_minutes
                )
                
                # Split on either behavioral transition OR context change
                has_transition = behavioral_transition or context_changed
                
                if has_transition:
                    # Determine reason for split
                    if behavioral_transition and context_changed:
                        split_reason = f"behavioral_and_context_change"
                    elif behavioral_transition:
                        split_reason = f"behavioral_transition_{last_base_category}_to_{recommended_task_id}"
                    else:
                        split_reason = context_reason
                    
                    reason = split_reason
                    result_task_id = smart_task_id
                    result_confidence = recommended_confidence
                    base_category = recommended_task_id
                else:
                    # No transition - keep current task
                    # Update confidence with moving average to reflect sustained confidence
                    reason = f"continuing_{last_base_category}"
                    result_task_id = state["current_task_id"]
                    base_category = last_base_category
                    # Blend current and observed confidence for the same task
                    current_conf = state.get("current_confidence", 0.5)
                    # If recommended task matches current, use higher confidence
                    if recommended_task_id == last_base_category:
                        result_confidence = max(current_conf, recommended_confidence)
                    else:
                        # Different recommendation but not transitioning - keep current confidence
                        result_confidence = current_conf
            else:
                # First classification
                reason = f"initial_{recommended_task_id}"
            
            result = OnlineClassificationResult(
                assigned_task_id=result_task_id,
                confidence=result_confidence,
                distance_to_centroid=1.0 - result_confidence,
                alternative_tasks=[],
                has_transition=has_transition,
                reason=reason
            )
            
            # Update state - handle both new segments and transitions
            if has_transition:
                # End previous segment
                if state["task_segments"] and state["task_segments"][-1].is_active:
                    state["task_segments"][-1].end_time = timestamp
                
                # Start new segment with full metadata
                segment = TaskSegment(
                    task_id=result_task_id,
                    start_time=timestamp,
                    end_time=None,
                    confidence=result_confidence,
                    feature_vector=rolling_features,
                    reason=reason,
                    distance_to_centroid=1.0 - result_confidence,
                    base_category=base_category,
                    app=raw_app,
                    window_title=raw_window,
                    normalized_title=normalized_window
                )
                # Attach normalized metadata derived from rolling features
                try:
                    extra = {
                        'metrics_snapshot': rolling_features,
                        'apps': [rolling_features.get('active_app')] if isinstance(rolling_features, dict) and rolling_features.get('active_app') else None,
                        'top_apps': None,
                        'domains': None,
                        'window_titles': [raw_window] if raw_window else None,
                        'feature_count': rolling_features.get('event_count') if isinstance(rolling_features, dict) else None,
                        'session_id': session_id,
                    }
                    segment.metadata = extra
                    segment.metadata['normalized'] = _normalize_extra_metadata(extra)
                except Exception:
                    segment.metadata = {}
                state["task_segments"].append(segment)
            elif not state["task_segments"]:
                # First segment with full metadata
                segment = TaskSegment(
                    task_id=result_task_id,
                    start_time=timestamp,
                    end_time=None,
                    confidence=result_confidence,
                    feature_vector=rolling_features,
                    reason=reason,
                    distance_to_centroid=1.0 - result_confidence,
                    base_category=base_category,
                    app=raw_app,
                    window_title=raw_window,
                    normalized_title=normalized_window
                )
                try:
                    extra = {
                        'metrics_snapshot': rolling_features,
                        'apps': [rolling_features.get('active_app')] if isinstance(rolling_features, dict) and rolling_features.get('active_app') else None,
                        'top_apps': None,
                        'domains': None,
                        'window_titles': [raw_window] if raw_window else None,
                        'feature_count': rolling_features.get('event_count') if isinstance(rolling_features, dict) else None,
                        'session_id': session_id,
                    }
                    segment.metadata = extra
                    segment.metadata['normalized'] = _normalize_extra_metadata(extra)
                except Exception:
                    segment.metadata = {}
                state["task_segments"].append(segment)
            else:
                # Continue current segment - update the latest feature vector and metadata
                state["task_segments"][-1].feature_vector = rolling_features
                state["task_segments"][-1].confidence = result_confidence
                # Update metadata if changed (e.g., switching tabs/windows within same task)
                state["task_segments"][-1].app = raw_app
                state["task_segments"][-1].window_title = raw_window
                state["task_segments"][-1].normalized_title = normalized_window
                try:
                    extra = {
                        'metrics_snapshot': rolling_features,
                        'apps': [rolling_features.get('active_app')] if isinstance(rolling_features, dict) and rolling_features.get('active_app') else None,
                        'top_apps': None,
                        'domains': None,
                        'window_titles': [normalized_window] if normalized_window else None,
                        'feature_count': rolling_features.get('event_count') if isinstance(rolling_features, dict) else None,
                        'session_id': session_id,
                    }
                    state["task_segments"][-1].metadata = extra
                    state["task_segments"][-1].metadata['normalized'] = _normalize_extra_metadata(extra)
                except Exception:
                    pass
            
            state["current_task_id"] = result_task_id
            state["current_confidence"] = result_confidence
            state["last_classification_time"] = timestamp
            state["classification_history"].append({
                "task_id": result_task_id,
                "confidence": result_confidence,
                "timestamp": timestamp.isoformat(),
                "reason": reason,
                "features": rolling_features
            })
            
            return result
        
        # Compute distances to all centroids
        distances = {}
        for task_id, centroid in self.task_engine.task_centroids.items():
            # Would compute actual distance if we had proper feature vector
            # For now, use mock calculation
            distance = self._estimate_distance(rolling_features)
            distances[task_id] = distance
        
        # Find best match
        best_task_id = min(distances, key=distances.get)
        best_distance = distances[best_task_id]
        best_centroid = self.task_engine.task_centroids[best_task_id]
        
        # Compute confidence
        if best_distance <= self.distance_threshold:
            confidence = max(0.5, 1.0 - best_distance)
        else:
            confidence = max(0.1, 0.5 - best_distance)
        
        # Detect transition
        previous_task = state["current_task_id"]
        has_transition = (previous_task and previous_task != best_task_id and
                         best_distance <= self.distance_threshold)
        
        # Get alternative tasks
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        alternatives = [(tid, 1.0 - d) for tid, d in sorted_distances[1:4]]
        
        # Determine reason
        if has_transition:
            reason = f"transition_from_{previous_task[:8]}"
        elif best_distance <= self.distance_threshold:
            reason = "distance_match"
        else:
            reason = "distance_exceeded"
        
        # Record classification
        result = OnlineClassificationResult(
            assigned_task_id=best_task_id,
            confidence=confidence,
            distance_to_centroid=best_distance,
            alternative_tasks=alternatives,
            has_transition=has_transition,
            reason=reason
        )
        
        # Update state
        if has_transition:
            # End previous segment
            if state["task_segments"] and state["task_segments"][-1].is_active:
                state["task_segments"][-1].end_time = timestamp
            
            # Start new segment
            segment = TaskSegment(
                task_id=best_task_id,
                start_time=timestamp,
                end_time=None,
                confidence=confidence,
                feature_vector=rolling_features,
                reason=reason,
                distance_to_centroid=best_distance
            )
            state["task_segments"].append(segment)
        elif not state["task_segments"]:
            # First classification - create initial segment
            segment = TaskSegment(
                task_id=best_task_id,
                start_time=timestamp,
                end_time=None,
                confidence=confidence,
                feature_vector=rolling_features,
                reason="initial",
                distance_to_centroid=best_distance
            )
            state["task_segments"].append(segment)
        
        # Update current task
        state["current_task_id"] = best_task_id
        state["current_confidence"] = confidence
        state["previous_task_id"] = previous_task
        state["last_classification_time"] = timestamp
        state["classification_history"].append({
            "task_id": best_task_id,
            "confidence": confidence,
            "timestamp": timestamp.isoformat(),
            "reason": reason
        })
        
        return result
    
    def finalize_session(self, session_id: str, end_time: datetime) -> List[TaskSegment]:
        """
        Finalize session classification when session ends.
        
        Returns all task segments that occurred during the session.
        """
        if session_id not in self.session_states:
            return []
        
        state = self.session_states[session_id]
        
        # Close any open segment
        if state["task_segments"] and state["task_segments"][-1].is_active:
            state["task_segments"][-1].end_time = end_time
        
        segments = state["task_segments"]
        
        # Clean up
        del self.session_states[session_id]
        
        return segments
    
    def get_session_task_summary(self, session_id: str) -> Dict:
        """Get current task summary for an active session."""
        if session_id not in self.session_states:
            return {}
        
        state = self.session_states[session_id]
        
        # Calculate total time per task
        task_durations = {}
        for segment in state["task_segments"]:
            task_id = segment.task_id
            if task_id not in task_durations:
                task_durations[task_id] = 0.0
            task_durations[task_id] += segment.duration_seconds
        
        # Calculate transitions
        num_transitions = len(state["task_segments"]) - 1
        
        return {
            "current_task_id": state["current_task_id"],
            "current_confidence": state["current_confidence"],
            "num_segments": len(state["task_segments"]),
            "num_transitions": num_transitions,
            "task_durations": task_durations,
            "last_classification": state["last_classification_time"].isoformat(),
            "classification_count": len(state["classification_history"])
        }
    
    def _estimate_distance(self, features: dict) -> float:
        """
        Estimate distance from features dict.
        
        Mock implementation - would use actual feature vector distance.
        """
        # Placeholder: random value between 0 and 1
        # In real use, would compute Euclidean distance
        intensity = features.get("intensity", 0.5)
        entropy = features.get("entropy", 0.5)
        
        # Mock distance calculation
        distance = math.sqrt((intensity - 0.5) ** 2 + (entropy - 0.5) ** 2)
        return min(distance, 1.0)


def integrate_online_classification_with_session(session, task_classifier, task_engine):
    """
    Helper to integrate online classification into SessionManager workflow.
    
    Should be called periodically while session is active.
    """
    if not hasattr(session, '_online_classifier'):
        # First time - initialize
        task_classifier.initialize_session(session.id, session.start)
        session._online_classifier = task_classifier
        session._task_engine = task_engine
    
    # Compute rolling features from timeline
    rolling_features = compute_rolling_features(session)
    
    # Perform online classification
    result = task_classifier.classify_active_session(session.id, rolling_features)
    
    # Update session with current task
    session.current_task_assignment = {
        "task_id": result.assigned_task_id,
        "confidence": result.confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": result.reason
    }
    
    return result


def compute_rolling_features(session) -> dict:
    """
    Compute rolling (cumulative) feature vector from active session.
    
    Used for online classification while session is still running.
    Features are based on data accumulated so far.
    """
    timeline = session.timeline
    if not timeline:
        return {}
    
    # Get recent timeline entries (last 10 minutes for rolling window)
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(minutes=10)
    
    recent_entries = {
        k: v for k, v in timeline.items()
        if isinstance(k, datetime) and k >= recent_cutoff
    }
    
    if not recent_entries:
        recent_entries = dict(list(timeline.items())[-10:])
    
    # Calculate features from recent data
    total_keys = sum(v.get("keys", 0) for v in recent_entries.values())
    total_clicks = sum(v.get("clicks", 0) for v in recent_entries.values())
    num_buckets = len(recent_entries)
    
    # App entropy from recent activity
    app_counts = {}
    for v in recent_entries.values():
        # Would extract app info from bucket
        pass
    
    return {
        "intensity": min(1.0, (total_keys + total_clicks) / 1000),
        "entropy": 0.5,  # Placeholder
        "continuity": 0.7,  # Placeholder
        "duration": min(1.0, (datetime.now(timezone.utc) - session.start).total_seconds() / 3600)
    }
