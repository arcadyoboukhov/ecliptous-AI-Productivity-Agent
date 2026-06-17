"""
Prediction Hooks — Orchestration layer for automatic prediction updates.

Provides hooks that are called automatically when:
1. Sessions complete
2. Tasks are assigned  
3. Periodic updates occur

This layer bridges SessionManager/InferenceRunner with PredictiveIntelligence
and BehavioralModel, ensuring predictions are always current.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable
import logging

try:
    from agent.analytics.predictor import PredictiveIntelligence, CompletionEstimate, RiskSignal, Bottleneck
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.task.inference import TaskInferenceEngine
except ImportError:
    PredictiveIntelligence = None
    BehavioralModel = None
    TaskInferenceEngine = None

logger = logging.getLogger(__name__)


class PredictionHooks:
    """
    Orchestrates automatic prediction updates.
    
    Called at key lifecycle points:
    - on_session_complete(session) - After session ends + task assigned
    - on_periodic_update() - Periodic check (e.g., every minute)
    - on_task_assignment(session_id, task_id) - When task assigned to session
    """
    
    def __init__(self, behavioral_model: Optional[BehavioralModel] = None,
                 task_inference: Optional[TaskInferenceEngine] = None):
        """
        Initialize hooks.
        
        Args:
            behavioral_model: BehavioralModel instance (loads data from sessions)
            task_inference: TaskInferenceEngine instance (for task clustering)
        """
        self.behavioral_model = behavioral_model
        self.task_inference = task_inference
        self.predictor = None
        
        if behavioral_model:
            self.predictor = PredictiveIntelligence(
                behavioral_model=behavioral_model,
                task_inference=task_inference
            )
        
        self.prediction_history: Dict[str, List[CompletionEstimate]] = {}
        self.risk_history: Dict[str, List[RiskSignal]] = {}
        self.last_update = None
        
    def on_session_complete(self, session_id: str, task_id: Optional[str] = None,
                           session_metrics: Optional[Dict] = None) -> Dict:
        """
        Called when a session completes.
        
        Triggers:
        1. Update completion estimates for the completed task
        2. Detect risks for the completed session
        3. Check for bottlenecks (stuck tasks)
        4. Update behavioral baselines
        
        Args:
            session_id: Session that completed
            task_id: Task ID (if assigned)
            session_metrics: Session metrics (duration, continuity, apps, etc.)
            
        Returns:
            Dict with predictions, risks, and alerts
        """
        if not self.predictor:
            return {}
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': session_id,
            'task_id': task_id,
            'completion_estimate': None,
            'task_risks': [],
            'schedule_risks': [],
            'bottlenecks': [],
            'alerts': [],
        }
        
        # 1. Update completion estimate if task assigned
        if task_id:
            try:
                # Get current session duration (if available)
                current_duration = session_metrics.get('duration_minutes', 0) if session_metrics else 0
                current_hour = datetime.now(timezone.utc).hour
                
                estimate = self.predictor.estimate_completion(
                    task_id=task_id,
                    current_duration_minutes=current_duration,
                    current_hour=current_hour
                )
                
                if estimate:
                    result['completion_estimate'] = {
                        'task_id': estimate.task_id,
                        'estimated_minutes_remaining': estimate.estimated_minutes_remaining,
                        'estimated_completion_time': estimate.estimated_completion_time,
                        'confidence': estimate.confidence,
                        'reason': estimate.reason,
                    }
                    self._record_prediction(task_id, estimate)
            except Exception as e:
                logger.warning(f"Error estimating completion for {task_id}: {e}")
        
        # 2. Detect task-specific risks
        if task_id and session_metrics:
            try:
                risks = self.predictor.detect_task_risks(task_id, session_metrics)
                result['task_risks'] = [
                    {
                        'task_id': r.task_id,
                        'risk_level': r.risk_level.value,
                        'category': r.category,
                        'message': r.message,
                        'recommendation': r.recommendation,
                    }
                    for r in risks
                ]
                self._record_risks(task_id, risks)
                
                # Create alerts for HIGH/CRITICAL risks
                for risk in risks:
                    if risk.risk_level.value in ['high', 'critical']:
                        result['alerts'].append({
                            'type': 'risk',
                            'level': risk.risk_level.value,
                            'task_id': task_id,
                            'message': risk.message,
                            'recommendation': risk.recommendation,
                        })
            except Exception as e:
                logger.warning(f"Error detecting task risks for {task_id}: {e}")
        
        # 3. Detect schedule risks
        try:
            current_hour = datetime.now(timezone.utc).hour
            current_continuity = session_metrics.get('focus_continuity', 0.5) if session_metrics else 0.5
            
            schedule_risks = self.predictor.detect_schedule_risks(current_hour, current_continuity)
            result['schedule_risks'] = [
                {
                    'risk_level': r.risk_level.value,
                    'category': r.category,
                    'message': r.message,
                    'recommendation': r.recommendation,
                }
                for r in schedule_risks
            ]
            
            # Create alerts for schedule risks
            for risk in schedule_risks:
                result['alerts'].append({
                    'type': 'schedule',
                    'level': risk.risk_level.value,
                    'message': risk.message,
                    'recommendation': risk.recommendation,
                })
        except Exception as e:
            logger.warning(f"Error detecting schedule risks: {e}")
        
        # 4. Check for bottlenecks (periodically, not every session)
        if self.last_update is None or \
           (datetime.now(timezone.utc) - self.last_update).total_seconds() > 300:  # Every 5 minutes
            try:
                if task_id:
                    bottleneck = self.predictor.detect_stuck_task(
                        task_id=task_id,
                        last_activity=datetime.now(timezone.utc).isoformat(),
                        task_state='COMPLETED'
                    )
                    
                    if bottleneck:
                        result['bottlenecks'].append({
                            'bottleneck_id': bottleneck.bottleneck_id,
                            'type': bottleneck.type,
                            'severity': bottleneck.severity.value,
                            'message': bottleneck.message,
                            'recommendation': bottleneck.recommendation,
                        })
                        result['alerts'].append({
                            'type': 'bottleneck',
                            'level': bottleneck.severity.value,
                            'task_id': task_id,
                            'message': bottleneck.message,
                            'recommendation': bottleneck.recommendation,
                        })
            except Exception as e:
                logger.warning(f"Error detecting bottlenecks: {e}")
            
            self.last_update = datetime.now(timezone.utc)
        
        return result
    
    def on_task_assignment(self, session_id: str, task_id: str) -> Dict:
        """
        Called when a task is assigned to a session.
        
        Triggers:
        1. Initial completion estimate
        2. Risk assessment
        
        Args:
            session_id: Session ID
            task_id: Assigned task ID
            
        Returns:
            Dict with initial predictions
        """
        if not self.predictor:
            return {}
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': session_id,
            'task_id': task_id,
            'completion_estimate': None,
            'alerts': [],
        }
        
        try:
            # Get initial estimate (starting duration = 0)
            estimate = self.predictor.estimate_completion(
                task_id=task_id,
                current_duration_minutes=0.0,
                current_hour=datetime.now(timezone.utc).hour
            )
            
            if estimate:
                result['completion_estimate'] = {
                    'task_id': estimate.task_id,
                    'estimated_minutes_remaining': estimate.estimated_minutes_remaining,
                    'estimated_completion_time': estimate.estimated_completion_time,
                    'confidence': estimate.confidence,
                    'reason': estimate.reason,
                }
                
                # Log for tracking
                logger.info(f"Task assigned: {task_id} in session {session_id}. "
                           f"Estimated {estimate.estimated_minutes_remaining:.0f}m remaining "
                           f"(confidence: {estimate.confidence:.0%})")
        except Exception as e:
            logger.warning(f"Error on task assignment for {task_id}: {e}")
        
        return result
    
    def on_periodic_update(self, active_task_id: Optional[str] = None,
                          active_duration_minutes: float = 0.0) -> Dict:
        """
        Called periodically (e.g., every minute) to refresh predictions.
        
        Triggers:
        1. Update completion estimate for active task
        2. Check for emerging risks
        
        Args:
            active_task_id: Currently active task ID
            active_duration_minutes: How long task has been running
            
        Returns:
            Dict with updated predictions
        """
        if not self.predictor or not active_task_id:
            return {}
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task_id': active_task_id,
            'completion_estimate': None,
            'alerts': [],
        }
        
        try:
            estimate = self.predictor.estimate_completion(
                task_id=active_task_id,
                current_duration_minutes=active_duration_minutes,
                current_hour=datetime.now(timezone.utc).hour
            )
            
            if estimate:
                result['completion_estimate'] = {
                    'task_id': estimate.task_id,
                    'estimated_minutes_remaining': estimate.estimated_minutes_remaining,
                    'estimated_completion_time': estimate.estimated_completion_time,
                    'confidence': estimate.confidence,
                }
        except Exception as e:
            logger.warning(f"Error in periodic update for {active_task_id}: {e}")
        
        return result
    
    def get_task_productivity_trend(self, task_id: str, days: int = 7) -> Optional[Dict]:
        """
        Get productivity trend for a task over time.
        
        Returns:
            Dict with productivity metrics per day
        """
        if not self.behavioral_model:
            return None
        
        baseline = self.behavioral_model.get_task_baseline(task_id)
        if not baseline:
            return None
        
        return {
            'task_id': task_id,
            'sample_count': baseline.get('session_count', 0),
            'avg_duration_minutes': baseline.get('duration_mean_minutes', 0),
            'std_duration_minutes': baseline.get('duration_std_minutes', 0),
            'avg_focus_continuity': baseline.get('continuity_mean', 0),
            'typical_apps': baseline.get('app_signatures', []),
        }
    
    def get_task_lifecycle_statistics(self, task_id: str) -> Optional[Dict]:
        """
        Get task lifecycle statistics (from creation to completion).
        
        Returns:
            Dict with lifecycle metrics
        """
        if not self.behavioral_model:
            return None
        
        baseline = self.behavioral_model.get_task_baseline(task_id)
        if not baseline:
            return None
        
        return {
            'task_id': task_id,
            'total_sessions': baseline.get('session_count', 0),
            'avg_session_duration_minutes': baseline.get('duration_mean_minutes', 0),
            'total_time_minutes': baseline.get('duration_mean_minutes', 0) * baseline.get('session_count', 1),
            'completion_rate': 1.0,  # Assuming all tracked sessions are completed
            'focus_continuity': baseline.get('continuity_mean', 0),
        }
    
    def get_time_of_day_patterns(self) -> Optional[Dict]:
        """
        Get productivity patterns by hour of day.
        
        Returns:
            Dict with hourly productivity metrics
        """
        if not self.behavioral_model:
            return None
        
        patterns = {}
        for hour in range(24):
            baseline = self.behavioral_model.get_time_of_day_baseline(hour)
            if baseline and baseline.get('session_count', 0) > 0:
                patterns[f"{hour:02d}:00"] = {
                    'avg_continuity': baseline.get('avg_focus_continuity', 0),
                    'session_count': baseline.get('session_count', 0),
                    'productivity_level': 'high' if baseline.get('avg_focus_continuity', 0) > 0.7 else 
                                         'medium' if baseline.get('avg_focus_continuity', 0) > 0.5 else 'low',
                }
        
        return patterns if patterns else None
    
    def _record_prediction(self, task_id: str, estimate: CompletionEstimate):
        """Record completion estimate in history."""
        if task_id not in self.prediction_history:
            self.prediction_history[task_id] = []
        
        # Keep last 30 predictions per task
        self.prediction_history[task_id].append(estimate)
        if len(self.prediction_history[task_id]) > 30:
            self.prediction_history[task_id] = self.prediction_history[task_id][-30:]
    
    def _record_risks(self, task_id: str, risks: List[RiskSignal]):
        """Record risk signals in history."""
        if task_id not in self.risk_history:
            self.risk_history[task_id] = []
        
        self.risk_history[task_id].extend(risks)
        # Keep last 50 risks per task
        if len(self.risk_history[task_id]) > 50:
            self.risk_history[task_id] = self.risk_history[task_id][-50:]
    
    def get_prediction_history(self, task_id: str) -> List[Dict]:
        """Get prediction history for a task."""
        if task_id not in self.prediction_history:
            return []
        
        return [
            {
                'estimated_minutes': p.estimated_minutes_remaining,
                'confidence': p.confidence,
                'computed_at': p.computed_at,
            }
            for p in self.prediction_history[task_id]
        ]
    
    def get_risk_history(self, task_id: str) -> List[Dict]:
        """Get risk signal history for a task."""
        if task_id not in self.risk_history:
            return []
        
        return [
            {
                'level': r.risk_level.value,
                'category': r.category,
                'message': r.message,
                'computed_at': r.computed_at,
            }
            for r in self.risk_history[task_id]
        ]
