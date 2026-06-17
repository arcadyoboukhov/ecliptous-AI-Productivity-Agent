"""
Live Task Prediction from Recent Signal Window

Real-time task prediction from the last 60 seconds of activity.
Runs independently from session finalization pipeline.
Predictions stored separately and updated every 1-2 seconds.

Key differences from finalized task inference:
- Uses rolling 60-second signal window instead of full session
- Computes predictions continuously even for active sessions
- Dynamic confidence thresholds (more permissive) to handle noisy short windows
- Separate persistence layer (live_task_predictions table)
- Supports confidence evolution tracking across predictions

Configuration: agent/task/live_prediction_config.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import math

from agent.task.inference import FeatureVector, TaskCentroid
from agent.error_handling import log_component_error, ComponentType, ErrorSeverity

# Load configuration
try:
    from agent.task.live_prediction_config import (
        LIVE_PREDICTION_CONFIDENCE_THRESHOLD,
        LIVE_PREDICTION_DISTANCE_THRESHOLD,
        USE_DYNAMIC_CONFIDENCE_THRESHOLDS,
        LIVE_PREDICTION_HISTORY_SIZE,
        MIN_EVENTS_FOR_PREDICTION,
    )
except ImportError:
    # Fallback defaults if config not found
    LIVE_PREDICTION_CONFIDENCE_THRESHOLD = 0.50
    LIVE_PREDICTION_DISTANCE_THRESHOLD = 0.35
    USE_DYNAMIC_CONFIDENCE_THRESHOLDS = True
    LIVE_PREDICTION_HISTORY_SIZE = 3
    MIN_EVENTS_FOR_PREDICTION = 10


@dataclass
class LiveTaskPrediction:
    """A single live task prediction from recent signals."""
    timestamp: str                           # ISO timestamp when prediction was made
    session_id: str                          # Active session ID
    task_id: str                             # Predicted task ID
    confidence: float                        # [0.0, 1.0] confidence
    distance_to_centroid: float             # Distance from centroid
    reason: str                              # Why this task was assigned
    feature_window_seconds: int             # Signal window used (60, 120, etc.)
    feature_vector: Optional[Dict] = None   # Snapshot of features used
    alternative_tasks: List[Tuple[str, float]] = field(default_factory=list)  # Other candidates
    
    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "distance_to_centroid": self.distance_to_centroid,
            "reason": self.reason,
            "feature_window_seconds": self.feature_window_seconds,
            "feature_vector": self.feature_vector,
            "alternative_tasks": [(task, float(conf)) for task, conf in self.alternative_tasks],
        }


class LiveTaskPredictor:
    """
    Real-time task prediction from rolling signal windows.
    
    Continuously predicts current task from last N seconds of activity
    without waiting for session completion.
    
    Configuration: agent.task.live_prediction_config
    """
    
    def __init__(self, 
                 task_inference_engine=None,
                 window_seconds: int = 60,
                 confidence_threshold: Optional[float] = None,
                 distance_threshold: Optional[float] = None):
        """
        Initialize live task predictor.
        
        Args:
            task_inference_engine: TaskInferenceEngine with task centroids
            window_seconds: Signal window to analyze (default 60s)
            confidence_threshold: Min confidence to make prediction
                                (default from config: 0.50, permissive)
            distance_threshold: Max distance for centroid match
                              (default from config: 0.35)
        """
        self.engine = task_inference_engine
        self.window_seconds = window_seconds
        
        # Use config values or provided overrides
        self.confidence_threshold = (
            confidence_threshold 
            if confidence_threshold is not None 
            else LIVE_PREDICTION_CONFIDENCE_THRESHOLD
        )
        self.distance_threshold = (
            distance_threshold 
            if distance_threshold is not None 
            else LIVE_PREDICTION_DISTANCE_THRESHOLD
        )
        
        # Track prediction history per session for smoothing
        self.prediction_history: Dict[str, List[LiveTaskPrediction]] = {}
        
    def predict(self, 
                session_id: str,
                rolling_features: Dict,
                timestamp: Optional[datetime] = None,
                max_history: Optional[int] = None) -> Optional[LiveTaskPrediction]:
        """
        Predict current task from rolling features.
        
        Args:
            session_id: Active session ID
            rolling_features: Feature dict from session signal buffer (contains entries like
                            context_switch_entropy, focus_continuity_score, etc.)
            timestamp: When prediction is made (default: now)
            max_history: Keep last N predictions per session for smoothing
                        (default from config: LIVE_PREDICTION_HISTORY_SIZE)
            
        Returns:
            LiveTaskPrediction if high enough confidence, else None
        """
        if not self.engine or not rolling_features:
            return None
        
        if max_history is None:
            max_history = LIVE_PREDICTION_HISTORY_SIZE
        
        try:
            timestamp = timestamp or datetime.now(timezone.utc)
            
            # Build feature vector from rolling features
            # (Handle both old dict-based and new FeatureVector-based formats)
            try:
                if isinstance(rolling_features, FeatureVector):
                    feature_vec = rolling_features
                else:
                    # Convert dict to FeatureVector
                    feature_vec = FeatureVector(
                        context_switch_entropy=float(rolling_features.get('context_switch_entropy', 0.0)),
                        focus_continuity_score=float(rolling_features.get('focus_continuity_score', 0.0)),
                        max_sustained_minutes=float(rolling_features.get('max_sustained_minutes', 0.0)),
                        avg_focus_window_minutes=float(rolling_features.get('avg_focus_window_minutes', 0.0)),
                        focus_consistency=float(rolling_features.get('focus_consistency', 0.0)),
                        app_diversity=float(rolling_features.get('app_diversity', 0.0)),
                        session_duration_minutes=float(rolling_features.get('session_duration_minutes', 0.0)),
                        input_intensity=float(rolling_features.get('input_intensity', 0.0)),
                        time_of_day_hour=int(rolling_features.get('time_of_day_hour', 0)),
                    )
            except (KeyError, ValueError, TypeError) as e:
                # Features incomplete or malformed
                return None
            
            # Find nearest centroid in task engine
            best_match = self.engine.find_nearest_centroid(feature_vec)
            
            if best_match is None:
                return None
            
            centroid, distance = best_match
            
            # Compute confidence (more permissive than finalized sessions)
            # Dynamic threshold based on distance
            base_confidence = 1.0 - min(distance / self.distance_threshold, 1.0)
            
            # If no members yet, be more conservative
            member_count = len(centroid.member_session_ids)
            stability_boost = 0.0
            if member_count >= 1:
                stability_boost = min(0.15 * math.log(member_count + 1), 0.25)
            
            confidence = min(base_confidence + stability_boost, 1.0)
            
            # Apply dynamic threshold if configured
            effective_threshold = self.confidence_threshold
            if USE_DYNAMIC_CONFIDENCE_THRESHOLDS:
                # Adjust threshold based on amount of accumulated data
                # This is approximate - in a real system would track event count
                if confidence < 0.5:
                    effective_threshold = 0.45  # More permissive when uncertain
                elif confidence >= 0.8:
                    effective_threshold = 0.65  # Stricter when lots of data
                # else use default threshold
            
            if confidence < effective_threshold:
                return None
            
            # Get alternative tasks for context
            alternatives = self.engine.find_nearest_centroids_k(feature_vec, k=3)
            alt_tasks = [(c.task_id, 1.0 - min(d / self.distance_threshold, 1.0)) 
                        for c, d in alternatives[1:]]  # Skip best match
            
            # Create prediction
            prediction = LiveTaskPrediction(
                timestamp=timestamp.isoformat(),
                session_id=session_id,
                task_id=centroid.task_id,
                confidence=confidence,
                distance_to_centroid=distance,
                reason="centroid_match",
                feature_window_seconds=self.window_seconds,
                feature_vector=rolling_features if isinstance(rolling_features, dict) else None,
                alternative_tasks=alt_tasks,
            )
            
            # Store in history for smoothing
            if session_id not in self.prediction_history:
                self.prediction_history[session_id] = []
            
            self.prediction_history[session_id].append(prediction)
            
            # Keep only recent predictions
            if len(self.prediction_history[session_id]) > max_history:
                self.prediction_history[session_id] = self.prediction_history[session_id][-max_history:]
            
            return prediction
            
        except Exception as e:
            log_component_error(
                ComponentType.ML,
                "live_predict",
                e,
                ErrorSeverity.WARNING
            )
            return None
    
    def get_smoothed_prediction(self, session_id: str) -> Optional[LiveTaskPrediction]:
        """
        Get smoothed prediction from recent history.
        
        Returns the most confident recent prediction, or the most frequent
        task if confidence is similar.
        """
        if session_id not in self.prediction_history:
            return None
        
        history = self.prediction_history[session_id]
        if not history:
            return None
        
        # Return highest confidence prediction from recent window
        return max(history[-3:], key=lambda p: p.confidence) if history else None
    
    def clear_history(self, session_id: Optional[str] = None):
        """Clear prediction history (on session end or manual reset)."""
        if session_id:
            self.prediction_history.pop(session_id, None)
        else:
            self.prediction_history.clear()
