"""
Predictive Intelligence Layer — Completion Estimates, Risk Signals, Bottleneck Detection

Generates predictions when sufficient history exists. All predictions are derived
from behavioral baselines and task inference, NOT raw extrapolations.

Three core capabilities:
1. Completion-time estimates: Predict when tasks will finish based on historical patterns
2. Risk signals: Identify concerning patterns (delays, degradation, anomalies)
3. Bottleneck detection: Flag stuck/blocked tasks and systemic issues

Design principles:
- Derived artifacts: Uses behavioral baselines + task inference (no raw extrapolations)
- Confidence-aware: All predictions include confidence scores
- History-gated: Only predict when sufficient data exists (min thresholds)
- Actionable: Each prediction includes reason and recommendation
- Local-only: No external dependencies
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import json

# Import from other layers
try:
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.task.inference import TaskInferenceEngine
except ImportError:
    # For testing without full project
    BehavioralModel = None
    TaskInferenceEngine = None


# ============================================================================
# Data Models
# ============================================================================

class RiskLevel(Enum):
    """Risk severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CompletionEstimate:
    """
    Predicted completion time for a task.
    
    Derived from:
    - Historical task duration baselines
    - Current progress patterns
    - Time-of-day productivity baselines
    """
    task_id: str
    estimated_minutes_remaining: float
    estimated_completion_time: Optional[str] = None  # ISO timestamp
    confidence: float = 0.0  # [0.0, 1.0]
    reason: str = ""
    method: str = "baseline"  # 'baseline', 'trend', 'regression'
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RiskSignal:
    """
    Warning about concerning patterns.
    
    Derived from:
    - Anomaly detection (current vs. baseline)
    - Trend analysis (degrading patterns)
    - Task state analysis (stuck, delayed)
    """
    task_id: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.NONE
    category: str = ""  # 'delay', 'anomaly', 'degradation', 'stall'
    message: str = ""
    evidence: Dict = field(default_factory=dict)
    recommendation: str = ""
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Bottleneck:
    """
    Flagged bottleneck (stuck task or systemic issue).
    
    Derived from:
    - Task state analysis (no progress for extended period)
    - Pattern analysis (consistent underperformance)
    - Systemic analysis (repeated issues at specific times/tasks)
    """
    bottleneck_id: str
    type: str  # 'stuck_task', 'systemic_time', 'systemic_task'
    severity: RiskLevel = RiskLevel.MEDIUM
    description: str = ""
    affected_tasks: List[str] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)
    recommendation: str = ""
    first_detected: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# Predictive Intelligence Engine
# ============================================================================

class PredictiveIntelligence:
    """
    Main predictor engine.
    
    Generates predictions only when sufficient history exists (gated by thresholds).
    All predictions derived from behavioral baselines and task inference.
    """
    
    # Minimum data requirements
    MIN_TASK_SAMPLES = 3  # Need at least 3 sessions to predict
    MIN_HOUR_SAMPLES = 2  # Need at least 2 sessions in hour to use time-of-day baseline
    MIN_HISTORY_DAYS = 2  # Need at least 2 days of data
    
    # Anomaly thresholds
    ANOMALY_THRESHOLD_STDEV = 2.0  # 2 standard deviations = anomaly
    DEGRADATION_THRESHOLD = 0.3  # 30% worse than baseline = degradation
    STALL_THRESHOLD_HOURS = 24  # No progress for 24h = stalled
    
    def __init__(self, behavioral_model: Optional['BehavioralModel'] = None,
                 task_inference: Optional['TaskInferenceEngine'] = None):
        """
        Initialize predictor with behavioral model and task inference.
        
        Args:
            behavioral_model: BehavioralModel instance (for baselines)
            task_inference: TaskInferenceEngine instance (for task patterns)
        """
        self.behavioral_model = behavioral_model
        self.task_inference = task_inference
        self._detected_bottlenecks: Dict[str, Bottleneck] = {}
    
    # ========================================================================
    # 1. Completion-Time Estimates
    # ========================================================================
    
    def estimate_completion(self, task_id: str, current_duration_minutes: float,
                           current_hour: Optional[int] = None) -> Optional[CompletionEstimate]:
        """
        Estimate remaining time for a task.
        
        Uses task baseline + time-of-day adjustments. Only predicts if sufficient
        history exists.
        
        Args:
            task_id: Task identifier
            current_duration_minutes: How long task has been running
            current_hour: Hour of day (0-23) for time-of-day adjustment
            
        Returns:
            CompletionEstimate if sufficient history, else None
        """
        if not self.behavioral_model:
            return None
        
        # Get task baseline
        baseline = self.behavioral_model.get_task_baseline(task_id)
        if not baseline or baseline['session_count'] < self.MIN_TASK_SAMPLES:
            return None
        
        # Extract baseline duration
        mean_duration = baseline['duration_mean_minutes']
        std_duration = baseline.get('duration_std_minutes', 0.0)
        
        # Adjust for time-of-day if available
        time_factor = 1.0
        if current_hour is not None:
            hour_baseline = self.behavioral_model.get_time_of_day_baseline(current_hour)
            if hour_baseline and hour_baseline['session_count'] >= self.MIN_HOUR_SAMPLES:
                # Use continuity as productivity proxy
                hour_continuity = hour_baseline.get('avg_focus_continuity', 0.5)
                task_continuity = baseline.get('continuity_mean', 0.5)
                
                if task_continuity > 0:
                    # If current hour is less productive, extend estimate
                    time_factor = task_continuity / max(hour_continuity, 0.1)
        
        # Compute remaining time
        adjusted_mean = mean_duration * time_factor
        remaining = max(0.0, adjusted_mean - current_duration_minutes)
        
        # Compute confidence (higher with more samples and lower variance)
        sample_confidence = min(baseline['session_count'] / 10.0, 1.0)
        variance_confidence = 1.0 / (1.0 + std_duration / max(mean_duration, 1.0))
        confidence = sample_confidence * variance_confidence
        
        # Build estimate
        reason = f"Based on {baseline['session_count']} historical sessions (avg {mean_duration:.0f}m)"
        if time_factor != 1.0:
            reason += f", adjusted for time-of-day productivity"
        
        estimated_completion = None
        if remaining > 0:
            completion_time = datetime.now(timezone.utc) + timedelta(minutes=remaining)
            estimated_completion = completion_time.isoformat()
        
        return CompletionEstimate(
            task_id=task_id,
            estimated_minutes_remaining=remaining,
            estimated_completion_time=estimated_completion,
            confidence=confidence,
            reason=reason,
            method="baseline"
        )
    
    def estimate_daily_workload(self, planned_tasks: List[str],
                               current_hour: int = 9) -> Optional[Dict]:
        """
        Estimate total time needed for a list of tasks.
        
        Useful for planning: "Can I finish these 3 tasks today?"
        
        Args:
            planned_tasks: List of task IDs to complete
            current_hour: Starting hour for workload
            
        Returns:
            Dict with total_minutes, task_estimates, feasibility
        """
        if not self.behavioral_model:
            return None
        
        estimates = []
        total_minutes = 0.0
        insufficient_data = []
        
        for task_id in planned_tasks:
            baseline = self.behavioral_model.get_task_baseline(task_id)
            if not baseline or baseline['session_count'] < self.MIN_TASK_SAMPLES:
                insufficient_data.append(task_id)
                continue
            
            mean_duration = baseline['duration_mean_minutes']
            total_minutes += mean_duration
            estimates.append({
                'task_id': task_id,
                'estimated_minutes': mean_duration,
                'confidence': min(baseline['session_count'] / 10.0, 1.0)
            })
        
        # Check feasibility (assume 8-hour workday with 80% productive time)
        available_minutes = 8 * 60 * 0.8  # 384 minutes
        feasible = total_minutes <= available_minutes
        
        return {
            'total_minutes': total_minutes,
            'total_hours': total_minutes / 60.0,
            'task_estimates': estimates,
            'insufficient_data': insufficient_data,
            'feasible': feasible,
            'utilization': total_minutes / available_minutes if available_minutes > 0 else 0.0,
            'computed_at': datetime.now(timezone.utc).isoformat()
        }
    
    # ========================================================================
    # 2. Risk Signals
    # ========================================================================
    
    def detect_task_risks(self, task_id: str, current_session_metrics: Dict) -> List[RiskSignal]:
        """
        Detect risks for a specific task.
        
        Checks for:
        - Duration anomalies (much longer/shorter than baseline)
        - Focus anomalies (much worse continuity than baseline)
        - App anomalies (using unusual apps for task)
        
        Args:
            task_id: Task identifier
            current_session_metrics: Current session metrics (duration, continuity, apps, etc.)
            
        Returns:
            List of RiskSignal objects
        """
        if not self.behavioral_model:
            return []
        
        baseline = self.behavioral_model.get_task_baseline(task_id)
        if not baseline or baseline['session_count'] < self.MIN_TASK_SAMPLES:
            return []  # Not enough history
        
        risks = []
        
        # Check duration anomaly
        current_duration = current_session_metrics.get('duration_minutes', 0)
        mean_duration = baseline['duration_mean_minutes']
        std_duration = baseline.get('duration_std_minutes', 0.0)
        
        if std_duration > 0 and current_duration > 0:
            z_score = abs(current_duration - mean_duration) / std_duration
            if z_score > self.ANOMALY_THRESHOLD_STDEV:
                level = RiskLevel.MEDIUM if z_score < 3.0 else RiskLevel.HIGH
                direction = "longer" if current_duration > mean_duration else "shorter"
                
                risks.append(RiskSignal(
                    task_id=task_id,
                    risk_level=level,
                    category="anomaly",
                    message=f"Task duration {direction} than usual ({current_duration:.0f}m vs. {mean_duration:.0f}m avg)",
                    evidence={
                        'current_duration_minutes': current_duration,
                        'baseline_mean_minutes': mean_duration,
                        'baseline_std_minutes': std_duration,
                        'z_score': z_score
                    },
                    recommendation="Review task scope or conditions that may be causing deviation"
                ))
        
        # Check focus continuity degradation
        current_continuity = current_session_metrics.get('focus_continuity', 0.0)
        mean_continuity = baseline.get('continuity_mean', 0.0)
        
        if mean_continuity > 0 and current_continuity < mean_continuity * (1 - self.DEGRADATION_THRESHOLD):
            risks.append(RiskSignal(
                task_id=task_id,
                risk_level=RiskLevel.MEDIUM,
                category="degradation",
                message=f"Focus continuity degraded ({current_continuity:.2f} vs. {mean_continuity:.2f} avg)",
                evidence={
                    'current_continuity': current_continuity,
                    'baseline_continuity': mean_continuity,
                    'degradation_percent': (1 - current_continuity / mean_continuity) * 100
                },
                recommendation="Consider reducing distractions or taking a break"
            ))
        
        # Check app anomaly
        current_apps = set(current_session_metrics.get('apps', []))
        typical_apps = baseline.get('typical_apps', [])
        
        if typical_apps and current_apps:
            typical_set = set(typical_apps)
            overlap = len(current_apps & typical_set)
            overlap_ratio = overlap / max(len(current_apps), 1)
            
            if overlap_ratio < 0.3:  # Less than 30% overlap
                risks.append(RiskSignal(
                    task_id=task_id,
                    risk_level=RiskLevel.LOW,
                    category="anomaly",
                    message=f"Using unusual apps for this task (only {overlap_ratio*100:.0f}% overlap with typical)",
                    evidence={
                        'current_apps': list(current_apps),
                        'typical_apps': typical_apps,
                        'overlap_ratio': overlap_ratio
                    },
                    recommendation="Verify you're working on the intended task"
                ))
        
        return risks
    
    def detect_schedule_risks(self, current_hour: int, current_continuity: float) -> List[RiskSignal]:
        """
        Detect schedule/time-of-day risks.
        
        Checks if current time is historically unproductive for focused work.
        
        Args:
            current_hour: Current hour (0-23)
            current_continuity: Current focus continuity score
            
        Returns:
            List of RiskSignal objects
        """
        if not self.behavioral_model:
            return []
        
        hour_baseline = self.behavioral_model.get_time_of_day_baseline(current_hour)
        if not hour_baseline or hour_baseline['session_count'] < self.MIN_HOUR_SAMPLES:
            return []
        
        risks = []
        
        # Check if current hour is historically low-productivity
        avg_continuity = hour_baseline.get('avg_focus_continuity', 0.0)
        if avg_continuity < 0.5:  # Historically low focus
            risks.append(RiskSignal(
                task_id=None,
                risk_level=RiskLevel.LOW,
                category="schedule",
                message=f"Hour {current_hour} historically has low focus (avg {avg_continuity:.2f})",
                evidence={
                    'hour': current_hour,
                    'historical_continuity': avg_continuity,
                    'session_count': hour_baseline['session_count']
                },
                recommendation="Consider scheduling deep work during higher-focus hours"
            ))
        
        return risks
    
    # ========================================================================
    # 3. Bottleneck Detection
    # ========================================================================
    
    def detect_stuck_task(self, task_id: str, last_activity: str,
                         task_state: str = "ACTIVE") -> Optional[Bottleneck]:
        """
        Detect if a task is stuck (no progress for extended period).
        
        Args:
            task_id: Task identifier
            last_activity: ISO timestamp of last activity
            task_state: Current task state
            
        Returns:
            Bottleneck if task is stuck, else None
        """
        if task_state not in ["ACTIVE", "IN_PROGRESS"]:
            return None  # Only check active tasks
        
        last_time = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        
        hours_since = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
        
        if hours_since > self.STALL_THRESHOLD_HOURS:
            severity = RiskLevel.MEDIUM if hours_since < 48 else RiskLevel.HIGH
            
            bottleneck = Bottleneck(
                bottleneck_id=f"stuck-{task_id}",
                type="stuck_task",
                severity=severity,
                description=f"No activity for {hours_since:.1f} hours",
                affected_tasks=[task_id],
                evidence={
                    'last_activity': last_activity,
                    'hours_since_activity': hours_since,
                    'task_state': task_state
                },
                recommendation="Review task blockers or consider abandoning if no longer relevant",
                first_detected=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat()
            )
            
            # Track bottleneck
            self._detected_bottlenecks[bottleneck.bottleneck_id] = bottleneck
            return bottleneck
        
        return None
    
    def detect_systemic_bottlenecks(self, recent_sessions: List[Dict],
                                   lookback_days: int = 7) -> List[Bottleneck]:
        """
        Detect systemic bottlenecks (patterns across multiple tasks/times).
        
        Checks for:
        - Consistently poor performance at specific hours
        - Multiple tasks showing similar issues
        
        Args:
            recent_sessions: List of recent session dicts
            lookback_days: How far back to analyze
            
        Returns:
            List of Bottleneck objects
        """
        if not recent_sessions or not self.behavioral_model:
            return []
        
        bottlenecks = []
        
        # Group sessions by hour
        sessions_by_hour = {}
        for session in recent_sessions:
            hour = session.get('hour')
            if hour is not None:
                if hour not in sessions_by_hour:
                    sessions_by_hour[hour] = []
                sessions_by_hour[hour].append(session)
        
        # Check each hour for consistent underperformance
        for hour, sessions in sessions_by_hour.items():
            if len(sessions) < 3:
                continue
            
            avg_continuity = sum(s.get('focus_continuity', 0) for s in sessions) / len(sessions)
            hour_baseline = self.behavioral_model.get_time_of_day_baseline(hour)
            
            if hour_baseline and hour_baseline['session_count'] >= self.MIN_HOUR_SAMPLES:
                baseline_continuity = hour_baseline.get('avg_focus_continuity', 0.5)
                
                # If consistently underperforming
                if avg_continuity < baseline_continuity * (1 - self.DEGRADATION_THRESHOLD):
                    bottleneck = Bottleneck(
                        bottleneck_id=f"systemic-hour-{hour}",
                        type="systemic_time",
                        severity=RiskLevel.MEDIUM,
                        description=f"Hour {hour} consistently underperforming ({avg_continuity:.2f} vs. {baseline_continuity:.2f} baseline)",
                        affected_tasks=[s.get('task_id', 'unknown') for s in sessions],
                        evidence={
                            'hour': hour,
                            'recent_avg_continuity': avg_continuity,
                            'baseline_continuity': baseline_continuity,
                            'session_count': len(sessions)
                        },
                        recommendation=f"Consider avoiding focused work during hour {hour}, or investigate environmental factors"
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def get_all_bottlenecks(self) -> List[Bottleneck]:
        """Get all currently tracked bottlenecks."""
        return list(self._detected_bottlenecks.values())
    
    def clear_resolved_bottlenecks(self, resolved_ids: List[str]) -> None:
        """Remove resolved bottlenecks from tracking."""
        for bid in resolved_ids:
            self._detected_bottlenecks.pop(bid, None)
    
    def serialize(self) -> Dict:
        """Serialize predictor state (currently tracked bottlenecks)."""
        return {
            'version': '1.0',
            'bottlenecks': [asdict(b) for b in self._detected_bottlenecks.values()],
            'serialized_at': datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def deserialize(data: Dict) -> 'PredictiveIntelligence':
        """Deserialize predictor state."""
        predictor = PredictiveIntelligence()
        
        for b_data in data.get('bottlenecks', []):
            # Convert risk_level and severity back to enum
            if 'risk_level' in b_data and isinstance(b_data['risk_level'], str):
                b_data['risk_level'] = RiskLevel(b_data['risk_level'])
            if 'severity' in b_data and isinstance(b_data['severity'], str):
                b_data['severity'] = RiskLevel(b_data['severity'])
            
            bottleneck = Bottleneck(**b_data)
            predictor._detected_bottlenecks[bottleneck.bottleneck_id] = bottleneck
        
        return predictor
    
    def get_summary(self) -> Dict:
        """Get summary of all predictions and risks."""
        return {
            'active_bottlenecks': len(self._detected_bottlenecks),
            'bottleneck_types': {
                'stuck_task': sum(1 for b in self._detected_bottlenecks.values() if b.type == 'stuck_task'),
                'systemic_time': sum(1 for b in self._detected_bottlenecks.values() if b.type == 'systemic_time'),
                'systemic_task': sum(1 for b in self._detected_bottlenecks.values() if b.type == 'systemic_task')
            },
            'summary_at': datetime.now(timezone.utc).isoformat()
        }
