"""
Streaming Task Assignment: Real-Time Task Inference During Active Sessions

Enables automatic task assignment as sessions accumulate feature data,
without waiting for session completion.

Design:
- Check accumulated signals at configurable intervals
- Extract intermediate feature vectors when sufficient data available
- Compare to task centroids and assign/merge if confidence threshold met
- Track assignments and confidence over session lifetime
- Dynamically adjust confidence thresholds based on data quality

Features:
1. Windowed feature extraction (last N minutes)
2. Dynamic confidence thresholds (higher for more data)
3. Task clustering with exponential moving average updates
4. Assignment tracking with timestamp and confidence
5. Merge/revert logic for contradictory assignments
"""

from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import uuid

from agent.task.inference import TaskInferenceEngine, FeatureVector


@dataclass
class StreamingTaskAssignment:
    """Record of a task assignment during active session."""
    assignment_id: str
    session_id: str
    task_id: str
    assigned_at: datetime
    confidence: float                # [0.0, 1.0]
    feature_count: int              # Number of signals accumulated
    assignment_reason: str          # "sufficient_data", "high_confidence", "merge_triggered"
    intermediate_features: dict     # Snapshot of features at assignment time


@dataclass
class StreamingTaskState:
    """
    Tracks streaming task assignments for a single active session.
    
    Allows real-time task detection as data accumulates.
    """
    session_id: str
    assignments: List[StreamingTaskAssignment] = field(default_factory=list)
    current_task_id: Optional[str] = None
    current_confidence: float = 0.0
    last_inference_time: Optional[datetime] = None
    min_events_for_inference: int = 30  # Need ≥30 events to attempt inference
    inference_interval_seconds: int = 60  # Check at most once per minute
    dynamic_confidence_threshold: float = 0.60  # Baseline, may adjust upward
    assignment_history: Dict[str, float] = field(default_factory=dict)  # task_id → max_confidence
    
    def should_attempt_inference(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last inference."""
        if self.last_inference_time is None:
            return True
        elapsed = (current_time - self.last_inference_time).total_seconds()
        return elapsed >= self.inference_interval_seconds
    
    def update_confidence_threshold(self, feature_count: int) -> float:
        """
        Dynamically adjust confidence threshold based on data quality.
        
        More data → can be more selective (higher threshold)
        Less data → be more permissive (lower threshold)
        """
        if feature_count < 30:
            # Very little data: lower threshold
            return max(0.40, self.dynamic_confidence_threshold - 0.15)
        elif feature_count < 100:
            # Moderate data: baseline threshold
            return self.dynamic_confidence_threshold
        else:
            # Substantial data: higher threshold (only accept good matches)
            return min(0.75, self.dynamic_confidence_threshold + 0.10)
    
    def record_assignment(
        self,
        task_id: str,
        assigned_at: datetime,
        confidence: float,
        feature_count: int,
        reason: str,
        features: dict
    ) -> StreamingTaskAssignment:
        """Record a new task assignment."""
        assignment = StreamingTaskAssignment(
            assignment_id=str(uuid.uuid4()),
            session_id=self.session_id,
            task_id=task_id,
            assigned_at=assigned_at,
            confidence=confidence,
            feature_count=feature_count,
            assignment_reason=reason,
            intermediate_features=features
        )
        
        self.assignments.append(assignment)
        self.current_task_id = task_id
        self.current_confidence = confidence
        self.last_inference_time = assigned_at
        
        # Track max confidence for this task
        if task_id not in self.assignment_history:
            self.assignment_history[task_id] = confidence
        else:
            self.assignment_history[task_id] = max(self.assignment_history[task_id], confidence)
        
        return assignment
    
    def get_assignment_summary(self) -> dict:
        """Summary of assignments for this session."""
        return {
            "session_id": self.session_id,
            "current_task_id": self.current_task_id,
            "current_confidence": self.current_confidence,
            "total_assignments": len(self.assignments),
            "assignment_history": self.assignment_history,
            "last_inference": self.last_inference_time.isoformat() if self.last_inference_time else None
        }


class StreamingTaskAssignmentEngine:
    """
    Real-time task inference for active sessions.
    
    Periodically:
    1. Extract intermediate feature vector from session signals
    2. Check against task centroids
    3. Assign task if confidence exceeds dynamic threshold
    4. Update centroids incrementally
    5. Handle conflicts/merges
    """
    
    def __init__(self, task_inference_engine: TaskInferenceEngine):
        """
        Initialize streaming assignment engine.
        
        Args:
            task_inference_engine: TaskInferenceEngine for clustering
        """
        self.task_engine = task_inference_engine
        self.streaming_states: Dict[str, StreamingTaskState] = {}
    
    def get_or_create_streaming_state(self, session_id: str) -> StreamingTaskState:
        """Get or create streaming task state for session."""
        if session_id not in self.streaming_states:
            self.streaming_states[session_id] = StreamingTaskState(session_id=session_id)
        return self.streaming_states[session_id]
    
    def attempt_streaming_assignment(
        self,
        session_id: str,
        session: "Session",  # Session object with signals buffer
        current_time: Optional[datetime] = None
    ) -> Optional[StreamingTaskAssignment]:
        """
        Attempt to assign a task to an active session based on current signals.
        
        Returns:
            StreamingTaskAssignment if successful assignment, None otherwise
        
        Raises:
            ValueError if session has no signal buffer
        """
        current_time = current_time or datetime.now(timezone.utc)
        
        if not session or not hasattr(session, 'signals') or session.signals is None:
            raise ValueError(f"Session {session_id} has no signal buffer")
        
        # Get streaming state for this session
        state = self.get_or_create_streaming_state(session_id)
        
        # Check if enough time has passed since last inference
        if not state.should_attempt_inference(current_time):
            return None
        
        # Extract metrics from signal buffer
        metrics = session.signals.metrics_since(300)  # Last 5 minutes
        if not metrics:
            return None
        
        feature_count = metrics.get("event_count", 0)
        
        # Need minimum events to attempt inference
        if feature_count < state.min_events_for_inference:
            return None
        
        # Extract intermediate feature vector
        try:
            # Create temporary session-like object for feature extraction
            feature_vector = self._extract_intermediate_features(session, metrics)
        except Exception:
            return None
        
        if feature_vector is None:
            return None
        
        # Get dynamic confidence threshold
        confidence_threshold = state.update_confidence_threshold(feature_count)
        
        # Attempt task inference
        try:
            # Build rich extra metadata to improve task creation
            extra_metadata = {
                'apps': getattr(session, 'apps', None),
                'top_apps': getattr(session, 'top_apps', None) if hasattr(session, 'top_apps') else None,
                'domains': getattr(session, 'domains', None) if hasattr(session, 'domains') else None,
                'window_titles': getattr(session, 'window_titles', None) if hasattr(session, 'window_titles') else None,
                'session_id': session_id,
                'feature_count': feature_count,
                'metrics_snapshot': metrics,
            }
            task_id, confidence = self.task_engine.infer_task(session_id, feature_vector, extra_metadata=extra_metadata)
        except Exception:
            return None
        
        # Check if confidence meets threshold
        if confidence < confidence_threshold:
            return None
        
        # Record assignment
        assignment = state.record_assignment(
            task_id=task_id,
            assigned_at=current_time,
            confidence=confidence,
            feature_count=feature_count,
            reason=self._get_assignment_reason(confidence, confidence_threshold),
            features={
                "context_switch_entropy": feature_vector.context_switch_entropy,
                "focus_continuity_score": feature_vector.focus_continuity_score,
                "sustained_minutes": feature_vector.max_sustained_minutes,
                "app_diversity": feature_vector.app_diversity,
                "intensity": feature_vector.input_intensity,
            }
        )
        
        return assignment
    
    def _extract_intermediate_features(self, session: "Session", metrics: dict) -> Optional[FeatureVector]:
        """
        Extract intermediate feature vector from session signals.
        
        Uses metrics aggregated from signal buffer to create a temporary feature vector
        without needing a completed session.
        """
        try:
            # Map metrics to feature dimensions
            context_switch_entropy = metrics.get("context_switch_entropy", 0.5)
            focus_continuity_score = metrics.get("focus_continuity_score", 0.5)
            max_sustained_minutes = metrics.get("max_sustained_minutes", 0.0)
            avg_focus_window_minutes = metrics.get("avg_focus_window_minutes", 0.0)
            focus_consistency = metrics.get("focus_consistency", 0.5)
            app_diversity = min(1.0, len(session.apps) / 10.0) if hasattr(session, 'apps') else 0.1
            input_intensity = metrics.get("intensity", 0.0) / 100.0  # Normalize to [0, 1]
            
            # Normalize durations to [0, 1]
            sustained_normalized = min(1.0, max_sustained_minutes / 60.0)
            
            return FeatureVector(
                context_switch_entropy=min(1.0, context_switch_entropy),
                focus_continuity_score=min(1.0, focus_continuity_score),
                max_sustained_minutes=sustained_normalized,
                avg_focus_window_minutes=min(1.0, avg_focus_window_minutes / 30.0),
                focus_consistency=min(1.0, focus_consistency),
                app_diversity=app_diversity,
                session_duration_minutes=min(1.0, metrics.get("session_duration_minutes", 0.0) / 120.0),
                input_intensity=input_intensity,
                time_of_day_hour=datetime.now(timezone.utc).hour / 24.0
            )
        except Exception:
            return None
    
    def _get_assignment_reason(self, confidence: float, threshold: float) -> str:
        """Describe why assignment was made."""
        if confidence >= 0.85:
            return "high_confidence"
        elif confidence >= 0.70:
            return "good_match"
        else:
            return "sufficient_data"
    
    def get_streaming_summary(self, session_id: str) -> Optional[dict]:
        """Get summary of streaming assignments for a session."""
        if session_id not in self.streaming_states:
            return None
        return self.streaming_states[session_id].get_assignment_summary()
    
    def finalize_session(self, session_id: str) -> List[StreamingTaskAssignment]:
        """
        Finalize streaming state when session ends.
        
        Returns list of all assignments made during session.
        """
        if session_id not in self.streaming_states:
            return []
        
        state = self.streaming_states[session_id]
        assignments = state.assignments[:]
        
        # Clean up state
        del self.streaming_states[session_id]
        
        return assignments
