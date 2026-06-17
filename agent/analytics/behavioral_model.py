"""
Behavioral Modeling Layer — Incremental Baseline Learning

Builds behavioral baselines over time without requiring external services.
Models are online/incremental (update with each new session), local-only,
and robust to missing data.

Three core model types:
1. Baselines per task: Expected metrics when working on each task
2. Baselines per time-of-day: Circadian patterns in productivity
3. Session quality distributions: Statistical profiles of session characteristics

Design principles:
- Online/incremental: Update with streaming data (no batch reprocessing)
- Local-only: No external APIs or cloud services
- Robust to missing data: Graceful degradation with partial data
- Versioned: Track model schema for reproducibility
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Tuple, ClassVar
from collections import defaultdict
import math
import json
import random


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_mean(values: List[float]) -> float:
    """Compute mean, handles empty list gracefully."""
    return sum(values) / len(values) if values else 0.0


def compute_std(values: List[float]) -> float:
    """Compute standard deviation, handles edge cases."""
    if len(values) < 2:
        return 0.0
    mean = compute_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile from a list, tolerant of empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * (percentile / 100.0))
    idx = min(max(idx, 0), len(ordered) - 1)
    return ordered[idx]


def update_incremental_stats(count: int, mean: float, m2: float, new_value: float) -> Tuple[int, float, float]:
    """
    Update mean and M2 (for variance) incrementally using Welford's algorithm.
    
    Returns: (new_count, new_mean, new_m2)
    
    Variance = M2 / (count - 1)
    StdDev = sqrt(Variance)
    """
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return count, mean, m2


# ============================================================================
# Per-Task Baselines
# ============================================================================

@dataclass
class TaskBaseline:
    """
    Statistical baseline for a specific task/intent.
    
    Tracks typical behavior when working on this task:
    - Duration: How long sessions typically last
    - Input intensity: Typical keys+clicks per minute
    - Focus metrics: Context switch entropy, continuity
    - App usage: Which apps are typically used
    """
    task_id: str
    INTENSITY_MAX_SAMPLES: ClassVar[int] = 50  # Reservoir cap (memory-bounded)
    
    # Incremental statistics (Welford's algorithm)
    session_count: int = 0
    duration_mean: float = 0.0
    duration_m2: float = 0.0  # For variance calculation
    input_intensity_mean: float = 0.0
    input_intensity_m2: float = 0.0
    entropy_mean: float = 0.0
    entropy_m2: float = 0.0
    continuity_mean: float = 0.0
    continuity_m2: float = 0.0

    # Memory-bounded intensity distribution samples
    intensity_samples: List[float] = field(default_factory=list)
    
    # App frequency (not incremental, tracked separately)
    app_counts: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    first_seen: Optional[str] = None
    last_updated: Optional[str] = None
    
    def update(self, session_duration_minutes: float, input_per_minute: float, 
               entropy: float, continuity: float, apps: List[str]) -> None:
        """Update baseline with new session data (incremental)."""
        now = datetime.now(timezone.utc).isoformat()
        
        if self.first_seen is None:
            self.first_seen = now
        self.last_updated = now
        
        # Update duration stats
        self.session_count, self.duration_mean, self.duration_m2 = update_incremental_stats(
            self.session_count, self.duration_mean, self.duration_m2, session_duration_minutes
        )
        
        # Update input intensity (recompute from current count)
        prev_intensity_mean = self.input_intensity_mean
        self.input_intensity_mean = (
            (self.input_intensity_mean * (self.session_count - 1) + input_per_minute) / self.session_count
        )
        intensity_delta = input_per_minute - prev_intensity_mean
        self.input_intensity_m2 += intensity_delta * (input_per_minute - self.input_intensity_mean)
        # Memory-bounded reservoir for distribution (percentiles/anomalies)
        if len(self.intensity_samples) < self.INTENSITY_MAX_SAMPLES:
            self.intensity_samples.append(input_per_minute)
        else:
            idx = random.randint(0, self.session_count - 1)
            if idx < self.INTENSITY_MAX_SAMPLES:
                self.intensity_samples[idx] = input_per_minute
        
        # Update entropy and continuity similarly
        self.entropy_mean = (
            (self.entropy_mean * (self.session_count - 1) + entropy) / self.session_count
        )
        
        self.continuity_mean = (
            (self.continuity_mean * (self.session_count - 1) + continuity) / self.session_count
        )
        
        # Update app frequencies
        for app in apps:
            self.app_counts[app] = self.app_counts.get(app, 0) + 1
    
    def get_duration_std(self) -> float:
        """Compute duration standard deviation from M2."""
        if self.session_count < 2:
            return 0.0
        return math.sqrt(self.duration_m2 / (self.session_count - 1))
    
    def get_typical_apps(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get top N most-used apps for this task."""
        sorted_apps = sorted(self.app_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_apps[:top_n]

    def get_intensity_distribution(self) -> Dict[str, float]:
        """Return memory-bounded intensity percentiles for anomaly/risk checks."""
        return {
            'p50': round(compute_percentile(self.intensity_samples, 50), 2),
            'p90': round(compute_percentile(self.intensity_samples, 90), 2),
            'max_samples': self.INTENSITY_MAX_SAMPLES,
            'sample_size': len(self.intensity_samples),
        }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'task_id': self.task_id,
            'session_count': self.session_count,
            'duration_mean_minutes': round(self.duration_mean, 2),
            'duration_std_minutes': round(self.get_duration_std(), 2),
            'input_intensity_mean': round(self.input_intensity_mean, 2),
            'entropy_mean': round(self.entropy_mean, 3),
            'continuity_mean': round(self.continuity_mean, 3),
            'intensity_distribution': self.get_intensity_distribution(),
            'intensity_samples': [round(x, 3) for x in self.intensity_samples],
            'top_apps': self.get_typical_apps(3),
            'first_seen': self.first_seen,
            'last_updated': self.last_updated,
        }


# ============================================================================
# Per-Time-of-Day Baselines
# ============================================================================

@dataclass
class TimeOfDayBaseline:
    """
    Statistical baseline for a specific hour of day (0-23).
    
    Tracks typical productivity patterns during this hour:
    - Session frequency: How often sessions start in this hour
    - Session quality: Typical focus/continuity during this hour
    - Activity level: Typical input intensity
    """
    hour: int  # 0-23
    
    # Incremental statistics
    session_count: int = 0
    session_duration_mean: float = 0.0
    session_duration_m2: float = 0.0
    input_intensity_mean: float = 0.0
    input_intensity_m2: float = 0.0
    focus_continuity_mean: float = 0.0
    focus_continuity_m2: float = 0.0
    
    # Classification breakdown (not incremental)
    classification_counts: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    first_seen: Optional[str] = None
    last_updated: Optional[str] = None
    
    def update(self, session_duration_minutes: float, input_per_minute: float,
               focus_continuity: float, classification: str) -> None:
        """Update baseline with new session data."""
        now = datetime.now(timezone.utc).isoformat()
        
        if self.first_seen is None:
            self.first_seen = now
        self.last_updated = now
        
        # Update duration
        self.session_count, self.session_duration_mean, self.session_duration_m2 = update_incremental_stats(
            self.session_count, self.session_duration_mean, self.session_duration_m2, session_duration_minutes
        )
        
        # Update input intensity
        _, self.input_intensity_mean, self.input_intensity_m2 = update_incremental_stats(
            self.session_count - 1, self.input_intensity_mean, self.input_intensity_m2, input_per_minute
        )
        
        # Update focus continuity
        _, self.focus_continuity_mean, self.focus_continuity_m2 = update_incremental_stats(
            self.session_count - 1, self.focus_continuity_mean, self.focus_continuity_m2, focus_continuity
        )
        
        # Update classification counts
        self.classification_counts[classification] = self.classification_counts.get(classification, 0) + 1
    
    def get_duration_std(self) -> float:
        """Compute duration standard deviation."""
        if self.session_count < 2:
            return 0.0
        return math.sqrt(self.session_duration_m2 / (self.session_count - 1))
    
    def get_dominant_classification(self) -> str:
        """Get most common session type for this hour."""
        if not self.classification_counts:
            return "unknown"
        return max(self.classification_counts.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'hour': self.hour,
            'session_count': self.session_count,
            'avg_duration_minutes': round(self.session_duration_mean, 2),
            'duration_std_minutes': round(self.get_duration_std(), 2),
            'avg_input_intensity': round(self.input_intensity_mean, 2),
            'avg_focus_continuity': round(self.focus_continuity_mean, 3),
            'dominant_classification': self.get_dominant_classification(),
            'classification_breakdown': dict(self.classification_counts),
            'first_seen': self.first_seen,
            'last_updated': self.last_updated,
        }


# ============================================================================
# Session Quality Distribution
# ============================================================================

@dataclass
class SessionQualityDistribution:
    """
    Statistical distribution of session quality metrics.
    
    Tracks overall session characteristics across all tasks/times:
    - Duration distribution (percentiles)
    - Input intensity distribution
    - Focus metrics distribution
    - Classification frequencies
    """
    
    # Sample tracking (for percentile computation)
    duration_samples: List[float] = field(default_factory=list)
    input_intensity_samples: List[float] = field(default_factory=list)
    entropy_samples: List[float] = field(default_factory=list)
    continuity_samples: List[float] = field(default_factory=list)
    
    # Incremental statistics
    total_sessions: int = 0
    duration_mean: float = 0.0
    duration_m2: float = 0.0
    
    # Classification counts
    classification_counts: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    model_version: str = "1.0"
    last_updated: Optional[str] = None
    
    # Sample limit (to prevent unbounded memory growth)
    MAX_SAMPLES: int = 1000
    
    def update(self, session_duration_minutes: float, input_per_minute: float,
               entropy: float, continuity: float, classification: str) -> None:
        """Update distribution with new session."""
        now = datetime.now(timezone.utc).isoformat()
        self.last_updated = now
        
        # Incremental mean/variance
        self.total_sessions, self.duration_mean, self.duration_m2 = update_incremental_stats(
            self.total_sessions, self.duration_mean, self.duration_m2, session_duration_minutes
        )
        
        # Sample tracking (with reservoir sampling for bounded memory)
        if len(self.duration_samples) < self.MAX_SAMPLES:
            self.duration_samples.append(session_duration_minutes)
            self.input_intensity_samples.append(input_per_minute)
            self.entropy_samples.append(entropy)
            self.continuity_samples.append(continuity)
        else:
            # Reservoir sampling: replace random sample with probability k/n
            import random
            idx = random.randint(0, self.total_sessions - 1)
            if idx < self.MAX_SAMPLES:
                self.duration_samples[idx] = session_duration_minutes
                self.input_intensity_samples[idx] = input_per_minute
                self.entropy_samples[idx] = entropy
                self.continuity_samples[idx] = continuity
        
        # Classification counts
        self.classification_counts[classification] = self.classification_counts.get(classification, 0) + 1
    
    def get_percentile(self, samples: List[float], percentile: float) -> float:
        """Compute percentile from samples (0-100)."""
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * (percentile / 100.0))
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]
    
    def get_duration_percentiles(self) -> Dict[str, float]:
        """Get duration percentiles (25th, 50th, 75th, 95th)."""
        return {
            'p25': self.get_percentile(self.duration_samples, 25),
            'p50': self.get_percentile(self.duration_samples, 50),
            'p75': self.get_percentile(self.duration_samples, 75),
            'p95': self.get_percentile(self.duration_samples, 95),
        }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'total_sessions': self.total_sessions,
            'duration_mean_minutes': round(self.duration_mean, 2),
            'duration_std_minutes': round(math.sqrt(self.duration_m2 / (self.total_sessions - 1)) if self.total_sessions > 1 else 0.0, 2),
            'duration_percentiles': {k: round(v, 2) for k, v in self.get_duration_percentiles().items()},
            'avg_input_intensity': round(compute_mean(self.input_intensity_samples), 2),
            'avg_entropy': round(compute_mean(self.entropy_samples), 3),
            'avg_continuity': round(compute_mean(self.continuity_samples), 3),
            'classification_breakdown': dict(self.classification_counts),
            'model_version': self.model_version,
            'last_updated': self.last_updated,
        }


# ============================================================================
# Behavioral Model Manager
# ============================================================================

class BehavioralModel:
    """
    Unified behavioral modeling engine.
    
    Manages three types of baselines:
    1. Per-task baselines
    2. Per-time-of-day baselines
    3. Overall session quality distribution
    
    All models are incremental (online updates) and local-only.
    """
    
    def __init__(self):
        self.task_baselines: Dict[str, TaskBaseline] = {}
        self.time_of_day_baselines: Dict[int, TimeOfDayBaseline] = {}
        self.session_quality: SessionQualityDistribution = SessionQualityDistribution()
        
        # Initialize all 24 hour slots
        for hour in range(24):
            self.time_of_day_baselines[hour] = TimeOfDayBaseline(hour=hour)
    
    def update_from_session(self, session, task_id: Optional[str] = None) -> None:
        """
        Update all baselines with data from a new session.
        
        Args:
            session: Session object with computed features
            task_id: Optional task ID (from inference or explicit assignment)
        """
        # Extract metrics from session
        duration_seconds = (session.end - session.start).total_seconds()
        duration_minutes = duration_seconds / 60.0
        
        # Input intensity
        input_events = getattr(session, 'input_events', {})
        total_input = input_events.get('keys', 0) + input_events.get('clicks', 0)
        input_per_minute = (total_input / duration_minutes) if duration_minutes > 0 else 0.0
        
        # Feature extraction
        entropy = getattr(session, 'context_switch_entropy', 0.5)
        continuity = getattr(session, 'focus_continuity_score', 0.5)
        
        # Classification
        from agent.analytics.daily import classify_session
        classification = classify_session(session)
        
        # Apps
        apps = getattr(session, 'apps', [])
        
        # Time of day
        hour = session.start.hour
        
        # Update per-task baseline (if task specified)
        if task_id:
            if task_id not in self.task_baselines:
                self.task_baselines[task_id] = TaskBaseline(task_id=task_id)
            
            self.task_baselines[task_id].update(
                duration_minutes, input_per_minute, entropy, continuity, apps
            )
        
        # Update time-of-day baseline
        self.time_of_day_baselines[hour].update(
            duration_minutes, input_per_minute, continuity, classification
        )
        
        # Update overall session quality distribution
        self.session_quality.update(
            duration_minutes, input_per_minute, entropy, continuity, classification
        )
    
    def get_task_baseline(self, task_id: str) -> Optional[Dict]:
        """Get baseline for a specific task."""
        baseline = self.task_baselines.get(task_id)
        return baseline.to_dict() if baseline else None
    
    def get_time_of_day_baseline(self, hour: int) -> Dict:
        """Get baseline for a specific hour (0-23)."""
        baseline = self.time_of_day_baselines.get(hour)
        return baseline.to_dict() if baseline else {}
    
    def get_session_quality_summary(self) -> Dict:
        """Get overall session quality distribution summary."""
        return self.session_quality.to_dict()
    
    def get_productivity_by_hour(self) -> List[Dict]:
        """Get productivity profile across all hours."""
        return [
            {
                'hour': hour,
                'session_count': baseline.session_count,
                'avg_continuity': round(baseline.focus_continuity_mean, 3),
                'dominant_type': baseline.get_dominant_classification(),
            }
            for hour, baseline in sorted(self.time_of_day_baselines.items())
            if baseline.session_count > 0
        ]
    
    def get_best_hours_for_focus(self, top_n: int = 3) -> List[int]:
        """Get top N hours with highest focus continuity."""
        hours_with_data = [
            (hour, baseline.focus_continuity_mean)
            for hour, baseline in self.time_of_day_baselines.items()
            if baseline.session_count > 0
        ]
        sorted_hours = sorted(hours_with_data, key=lambda x: x[1], reverse=True)
        return [hour for hour, _ in sorted_hours[:top_n]]
    
    def serialize(self) -> Dict:
        """Serialize entire model to dictionary."""
        return {
            'version': '1.0',
            'task_baselines': {
                task_id: baseline.to_dict()
                for task_id, baseline in self.task_baselines.items()
            },
            'time_of_day_baselines': {
                hour: baseline.to_dict()
                for hour, baseline in self.time_of_day_baselines.items()
                if baseline.session_count > 0
            },
            'session_quality': self.session_quality.to_dict(),
            'summary': {
                'total_tasks_tracked': len(self.task_baselines),
                'total_hours_with_data': sum(1 for b in self.time_of_day_baselines.values() if b.session_count > 0),
                'total_sessions_analyzed': self.session_quality.total_sessions,
            }
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'BehavioralModel':
        """Reconstruct model from serialized state."""
        model = cls()
        
        # Restore task baselines
        for task_id, baseline_data in data.get('task_baselines', {}).items():
            baseline = TaskBaseline(task_id=task_id)
            baseline.session_count = baseline_data.get('session_count', 0)
            baseline.duration_mean = baseline_data.get('duration_mean_minutes', 0.0)
            baseline.input_intensity_mean = baseline_data.get('input_intensity_mean', 0.0)
            baseline.entropy_mean = baseline_data.get('entropy_mean', 0.0)
            baseline.continuity_mean = baseline_data.get('continuity_mean', 0.0)
            baseline.intensity_samples = baseline_data.get('intensity_samples', [])
            baseline.first_seen = baseline_data.get('first_seen')
            baseline.last_updated = baseline_data.get('last_updated')
            model.task_baselines[task_id] = baseline
        
        # Restore time-of-day baselines (simplified - full restoration would need more fields)
        for hour_str, baseline_data in data.get('time_of_day_baselines', {}).items():
            hour = int(hour_str)
            baseline = model.time_of_day_baselines[hour]
            baseline.session_count = baseline_data.get('session_count', 0)
            baseline.session_duration_mean = baseline_data.get('avg_duration_minutes', 0.0)
            baseline.input_intensity_mean = baseline_data.get('avg_input_intensity', 0.0)
            baseline.focus_continuity_mean = baseline_data.get('avg_focus_continuity', 0.0)
            baseline.classification_counts = baseline_data.get('classification_breakdown', {})
            baseline.first_seen = baseline_data.get('first_seen')
            baseline.last_updated = baseline_data.get('last_updated')
        
        # Restore session quality (simplified)
        quality_data = data.get('session_quality', {})
        model.session_quality.total_sessions = quality_data.get('total_sessions', 0)
        model.session_quality.duration_mean = quality_data.get('duration_mean_minutes', 0.0)
        model.session_quality.classification_counts = quality_data.get('classification_breakdown', {})
        model.session_quality.last_updated = quality_data.get('last_updated')
        
        return model
