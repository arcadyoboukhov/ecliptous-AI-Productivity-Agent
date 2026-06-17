"""
Task Inference Layer — Latent Task Discovery via Feature Vectors

Automatically infers tasks from accumulated behavioral patterns when no explicit
task exists. Uses feature vectors to match activity to existing task centroids,
with support for creating new latent tasks when confidence is insufficient.

Design principles:
- Tasks are latent: automatically discovered from features, not explicitly required
- Incremental: centroids updated as new data arrives
- Label-optional: tasks can exist without user-provided names
- Confidence-tracked: each inference includes a confidence score
- Drift-aware: tracks how task patterns evolve over time

Feature vectors (from sessions):
- Context switch entropy: app-switching predictability [0.0, 1.0]
- Focus continuity score: sustained work periods [0.0, 1.0]
- Sustained activity metrics: detailed activity breakdown
- Duration: session length in minutes
- App signature: set of apps used (normalized)
- Time of day: hourly slot (0-23)

Task centroids store:
- Mean feature vector (weighted average of constituent sessions)
- Confidence: agreement between member sessions (0.0=uncertain, 1.0=highly consistent)
- Drift: how much the task pattern is changing over time
- Member count: sessions attributed to this task
- Creation timestamp: when task was first inferred
- Last seen: most recent session included
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import math
import uuid

from agent.error_handling import log_component_error, ComponentType, ErrorSeverity


# ============================================================================
# Feature Vector & Centroid Models
# ============================================================================

@dataclass
class FeatureVector:
    """
    A normalized feature vector derived from a session's computed features.
    
    All values normalized to [0.0, 1.0] for comparison and distance metrics.
    """
    context_switch_entropy: float        # [0.0, 1.0] from DailySummary
    focus_continuity_score: float        # [0.0, 1.0] from DailySummary
    max_sustained_minutes: float         # normalized by typical max (300 min = 5 hours)
    avg_focus_window_minutes: float      # normalized by typical max (120 min)
    focus_consistency: float             # [0.0, 1.0] from sustained metrics
    app_diversity: float                 # norm(unique_apps) / max_apps, [0.0, 1.0]
    session_duration_minutes: float      # normalized by typical max (480 min = 8 hours)
    input_intensity: float               # normalized keys+clicks per minute
    time_of_day_hour: int                # [0, 23] hour slot (not normalized)
    
    def distance_to(self, other: 'FeatureVector') -> float:
        """
        Compute Euclidean distance to another feature vector.
        
        Returns a normalized distance [0.0 = identical, 1.0 = completely different].
        Excludes time_of_day_hour from distance (allows flexible matching across time).
        """
        if not isinstance(other, FeatureVector):
            return 1.0
        
        # Euclidean distance for normalized features
        sq_sum = 0.0
        fields = [
            ('context_switch_entropy', self.context_switch_entropy, other.context_switch_entropy),
            ('focus_continuity_score', self.focus_continuity_score, other.focus_continuity_score),
            ('max_sustained_minutes', self.max_sustained_minutes, other.max_sustained_minutes),
            ('avg_focus_window_minutes', self.avg_focus_window_minutes, other.avg_focus_window_minutes),
            ('focus_consistency', self.focus_consistency, other.focus_consistency),
            ('app_diversity', self.app_diversity, other.app_diversity),
            ('session_duration_minutes', self.session_duration_minutes, other.session_duration_minutes),
            ('input_intensity', self.input_intensity, other.input_intensity),
        ]
        
        for name, val1, val2 in fields:
            diff = val1 - val2
            sq_sum += diff * diff
        
        # Normalize by number of dimensions for interpretability
        distance = math.sqrt(sq_sum / len([f for f, _, _ in fields]))
        return min(distance, 1.0)  # Cap at 1.0


@dataclass
class TaskCentroid:
    """
    Centroid (mean) of a cluster of similar behavioral sessions.
    
    Represents a latent task inferred from accumulated sessions.
    Tracks both feature means and per-feature variances to quantify stability.
    """
    task_id: str                                          # Unique identifier
    feature_vector: FeatureVector                         # Mean of member features
    feature_variance: FeatureVector                       # EMA variance of features
    feature_drift: FeatureVector                          # EMA per-feature drift magnitude
    drift_cumulative: float = 0.0                         # Cumulative centroid movement
    member_session_ids: List[str] = field(default_factory=list)  # Sessions in this cluster
    confidence: float = 0.5                               # [0.0, 1.0] consistency
    drift: float = 0.0                                    # [0.0, 1.0] change rate
    label: Optional[str] = None                           # Optional user-provided label
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)          # Extra representative data (apps, titles, domains)
    
    @property
    def is_unstable(self) -> bool:
        """
        Returns True if task drift exceeds threshold.
        
        Indicates pattern instability and recommends human review.
        Threshold is 0.15 (15% of max drift range).
        """
        feature_drift_avg = (
            self.feature_drift.context_switch_entropy +
            self.feature_drift.focus_continuity_score +
            self.feature_drift.max_sustained_minutes +
            self.feature_drift.avg_focus_window_minutes +
            self.feature_drift.focus_consistency +
            self.feature_drift.app_diversity +
            self.feature_drift.session_duration_minutes +
            self.feature_drift.input_intensity
        ) / 8.0
        return self.drift > 0.15 or self.drift_cumulative > 1.0 or feature_drift_avg > 0.08
    
    def __repr__(self):
        label_str = f" ({self.label})" if self.label else ""
        stability = "⚠️ UNSTABLE" if self.is_unstable else "stable"
        return f"TaskCentroid({self.task_id[:8]}{label_str}, conf={self.confidence:.2f}, drift={self.drift:.2f} ({stability}), n={len(self.member_session_ids)})"


# ============================================================================
# Feature Extraction from Sessions
# ============================================================================

def extract_feature_vector(session, session_summary: Optional[dict] = None) -> FeatureVector:
    """
    Extract a normalized feature vector from a session.
    
    Args:
        session: Session object with feature data (from DailySummary aggregation)
        session_summary: Optional pre-computed summary dict
    
    Returns:
        FeatureVector with all fields normalized to [0.0, 1.0]
    """
    try:
        # Get feature summary (either from session object or pre-computed dict)
        if session_summary:
            entropy = session_summary.get('context_switch_entropy', 0.5)
            continuity = session_summary.get('focus_continuity_score', 0.5)
            sustained = session_summary.get('sustained_activity_metrics', {})
        else:
            entropy = getattr(session, 'context_switch_entropy', 0.5)
            continuity = getattr(session, 'focus_continuity_score', 0.5)
            sustained = getattr(session, 'sustained_activity_metrics', {})
        
        # Extract sustained metrics
        max_sustained = sustained.get('max_sustained_minutes', 30.0) if isinstance(sustained, dict) else 30.0
        avg_focus = sustained.get('avg_focus_window_minutes', 25.0) if isinstance(sustained, dict) else 25.0
        consistency = sustained.get('focus_consistency', 0.5) if isinstance(sustained, dict) else 0.5
        
        # Calculate app diversity (normalized)
        apps = getattr(session, 'apps', [])
        max_apps = 20  # Typical max unique apps per day
        app_diversity = min(len(apps) / max_apps, 1.0) if apps else 0.0
        
        # Session duration (normalized by 8 hours = 480 minutes)
        duration_seconds = (session.end - session.start).total_seconds() if hasattr(session, 'end') and hasattr(session, 'start') else 1800
        duration_minutes = duration_seconds / 60.0
        duration_norm = min(duration_minutes / 480.0, 1.0)
        
        # Input intensity (keys + clicks per minute, normalized by typical 8 per minute)
        input_events = getattr(session, 'input_events', {})
        total_input = input_events.get('keys', 0) + input_events.get('clicks', 0) if isinstance(input_events, dict) else 0
        input_per_minute = (total_input / duration_minutes) if duration_minutes > 0 else 0.0
        input_intensity = min(input_per_minute / 8.0, 1.0)
        
        # Time of day (hour slot, not normalized)
        time_slot = 12  # Default midday
        if hasattr(session, 'start') and session.start:
            time_slot = session.start.hour
        
        return FeatureVector(
            context_switch_entropy=min(entropy, 1.0),
            focus_continuity_score=min(continuity, 1.0),
            max_sustained_minutes=min(max_sustained / 300.0, 1.0),  # Normalize by 5 hours
            avg_focus_window_minutes=min(avg_focus / 120.0, 1.0),  # Normalize by 2 hours
            focus_consistency=min(consistency, 1.0),
            app_diversity=app_diversity,
            session_duration_minutes=duration_norm,
            input_intensity=input_intensity,
            time_of_day_hour=time_slot,
        )
    except Exception as e:
        log_component_error(
            ComponentType.ML,
            "extract_feature_vector",
            e,
            ErrorSeverity.ERROR,
            session_id=getattr(session, 'id', 'unknown')
        )
        # Return default safe feature vector
        return FeatureVector(
            context_switch_entropy=0.5,
            focus_continuity_score=0.5,
            max_sustained_minutes=0.5,
            avg_focus_window_minutes=0.5,
            focus_consistency=0.5,
            app_diversity=0.5,
            session_duration_minutes=0.5,
            input_intensity=0.5,
            time_of_day_hour=12,
        )


def _normalize_extra_metadata(extra: Optional[dict]) -> dict:
    """
    Normalize extra metadata into a compact, numeric-first representation.

    Expected source keys (best-effort):
      - metrics_snapshot: dict with interval-level metrics (cpu_usage, ram_usage, gpu_usage, event_count, session_duration_minutes)
      - apps/top_apps: list of app names
      - domains: list of domains
      - window_titles: list of titles

    Returns a dict with normalized numeric fields in [0.0, 1.0], plus cleaned arrays.
    """
    if not extra:
        return {}

    out = {}
    metrics = extra.get('metrics_snapshot') or {}

    # CPU / RAM / GPU: expect percentage values 0..100
    try:
        cpu = metrics.get('cpu_usage') if 'cpu_usage' in metrics else metrics.get('cpu_percent') if 'cpu_percent' in metrics else None
        ram = metrics.get('ram_usage') if 'ram_usage' in metrics else metrics.get('memory_percent') if 'memory_percent' in metrics else None
        gpu = metrics.get('gpu_usage') if 'gpu_usage' in metrics else metrics.get('gpu_percent') if 'gpu_percent' in metrics else None

        # Try parse to float when present
        cpu = float(cpu) if cpu is not None else None
        ram = float(ram) if ram is not None else None
        gpu = float(gpu) if gpu is not None else None

        # Fallback to aggregated fields if interval metrics missing
        if cpu is None and extra.get('avg_cpu') is not None:
            try:
                cpu = float(extra.get('avg_cpu'))
            except Exception:
                cpu = None
        if ram is None and extra.get('avg_ram') is not None:
            try:
                ram = float(extra.get('avg_ram'))
            except Exception:
                ram = None
        if gpu is None and extra.get('avg_gpu') is not None:
            try:
                gpu = float(extra.get('avg_gpu'))
            except Exception:
                gpu = None
    except Exception:
        cpu = ram = gpu = None

    # Some collectors store cpu/ram/gpu as fractions (0.0285 == 2.85%).
    # Normalize to percentage space expected by downstream logic.
    try:
        if cpu is not None and cpu <= 1.0:
            cpu = cpu * 100.0
    except Exception:
        pass
    try:
        if ram is not None and ram <= 1.0:
            ram = ram * 100.0
    except Exception:
        pass
    try:
        if gpu is not None and gpu <= 1.0:
            gpu = gpu * 100.0
    except Exception:
        pass

    out['cpu_normalized'] = (max(0.0, min(1.0, cpu / 100.0)) if cpu is not None else None)
    out['ram_normalized'] = (max(0.0, min(1.0, ram / 100.0)) if ram is not None else None)
    out['gpu_normalized'] = (max(0.0, min(1.0, gpu / 100.0)) if gpu is not None else None)

    # Input intensity: metrics may include 'intensity' or keys/clicks counts
    intensity = None
    if 'intensity' in metrics:
        try:
            intensity = float(metrics.get('intensity'))
        except Exception:
            intensity = None
    elif metrics.get('input_intensity') is not None:
        try:
            intensity = float(metrics.get('input_intensity'))
        except Exception:
            intensity = None
    else:
        try:
            keys = float(metrics.get('keys', 0) or 0)
            clicks = float(metrics.get('clicks', 0) or 0)
            # duration may be in metrics (minutes) or fallback to feature_count-based estimate
            duration_min = None
            if metrics.get('session_duration_minutes') is not None:
                duration_min = float(metrics.get('session_duration_minutes'))
            elif extra.get('feature_count') is not None:
                # assume feature_count roughly equals total input events over the segment; avoid division by tiny numbers
                duration_min = max(1.0, float(extra.get('feature_count')) / 60.0)
            if duration_min is None or duration_min <= 0:
                duration_min = 1.0
            intensity = (keys + clicks) / duration_min
        except Exception:
            intensity = None

    out['input_intensity_normalized'] = (max(0.0, min(1.0, float(intensity) / 8.0)) if intensity is not None else None)

    # Signal quality: events relative to a reasonable cap (e.g., 300)
    ev = None
    if 'event_count' in metrics:
        ev = metrics.get('event_count')
    elif metrics.get('total_input') is not None:
        ev = metrics.get('total_input')
    elif extra.get('feature_count') is not None:
        ev = extra.get('feature_count')
    if ev is not None:
        try:
            ev = float(ev)
        except Exception:
            ev = None
    out['signal_quality'] = (max(0.0, min(1.0, ev / 300.0)) if ev is not None else None)

    # Top apps/domains/window_titles: normalize strings and truncate
    def _clean_list(x):
        if not x:
            return []
        try:
            return [str(i).strip().lower() for i in x][:10]
        except Exception:
            return []

    out['top_apps'] = _clean_list(extra.get('top_apps') or extra.get('apps') or [])
    out['domains'] = _clean_list(extra.get('domains') or [])
    out['window_titles'] = _clean_list(extra.get('window_titles') or [])

    # Feature count and session id
    out['feature_count'] = int(extra.get('feature_count') or metrics.get('event_count') or 0)
    out['session_id'] = extra.get('session_id')

    return out


# ============================================================================
# Task Inference Engine
# ============================================================================

class TaskInferenceEngine:
    """
    Infers latent tasks from accumulated feature vectors.
    
    Core algorithm:
    1. For each new session, extract feature vector
    2. Compare to existing task centroids
    3. If nearest match exceeds confidence threshold: assign to that task
    4. Otherwise: create new latent task
    5. Update centroid with new session data
    6. Track confidence and drift
    """
    
    # Configuration
    CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to assign to existing task
    DISTANCE_THRESHOLD = 0.35    # Maximum Euclidean distance to match (lower=stricter)
    DRIFT_THRESHOLD = 0.15       # Max acceptable drift before warning
    MIN_MEMBERS_FOR_CONFIDENCE = 3  # Minimum sessions needed for stable centroid
    EMA_ALPHA = 0.20             # Exponential moving average factor for centroid updates
    
    def __init__(self):
        self.task_centroids: Dict[str, TaskCentroid] = {}
        self.session_to_task: Dict[str, str] = {}  # session_id -> task_id
    
    def infer_task(self, session_id: str, feature_vector: FeatureVector, extra_metadata: Optional[dict] = None) -> Tuple[str, float]:
        """
        Infer a task for a given session based on its feature vector.
        
        Returns:
            (task_id, confidence) tuple
        """
        try:
            if not self.task_centroids:
                # First task: create initial centroid
                task_id = self._create_new_task(session_id, feature_vector, extra_metadata=extra_metadata)
                return task_id, 1.0
            
            # Find nearest centroid
            best_task_id, best_distance = self._find_nearest_centroid(feature_vector)
            
            # Compute confidence based on distance and centroid stability
            if best_distance <= self.DISTANCE_THRESHOLD:
                centroid = self.task_centroids[best_task_id]
                # Confidence decreases with distance
                base_confidence = 1.0 - (best_distance / self.DISTANCE_THRESHOLD)
                # Boost confidence if centroid is stable (many members, low drift)
                stability_boost = (len(centroid.member_session_ids) / self.MIN_MEMBERS_FOR_CONFIDENCE) * 0.2
                confidence = min(base_confidence + stability_boost, 1.0)
                
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    # Assign to existing task
                    self.session_to_task[session_id] = best_task_id
                    centroid.member_session_ids.append(session_id)
                    self._update_centroid(best_task_id, feature_vector)
                    return best_task_id, confidence
            
            # Create new latent task (confidence too low or distance too high)
            task_id = self._create_new_task(session_id, feature_vector, extra_metadata=extra_metadata)
            return task_id, 0.5  # New tasks get medium confidence
        except Exception as e:
            log_component_error(
                ComponentType.ML,
                "infer_task",
                e,
                ErrorSeverity.ERROR,
                session_id=session_id
            )
            # Return a default task on failure
            fallback_task_id = f"error-fallback-{str(uuid.uuid4())[:8]}"
            return fallback_task_id, 0.0
    
    def _find_nearest_centroid(self, feature_vector: FeatureVector) -> Tuple[str, float]:
        """Find the nearest task centroid and return (task_id, distance)."""
        best_task_id = None
        best_distance = float('inf')
        
        for task_id, centroid in self.task_centroids.items():
            distance = feature_vector.distance_to(centroid.feature_vector)
            if distance < best_distance:
                best_distance = distance
                best_task_id = task_id
        
        return best_task_id, best_distance
    
    def _create_new_task(self, session_id: str, feature_vector: FeatureVector, extra_metadata: Optional[dict] = None) -> str:
        """Create a new latent task centroid."""
        task_id = f"latent-{str(uuid.uuid4())[:8]}"
        centroid = TaskCentroid(
            task_id=task_id,
            feature_vector=feature_vector,
            feature_variance=FeatureVector(
                context_switch_entropy=0.0,
                focus_continuity_score=0.0,
                max_sustained_minutes=0.0,
                avg_focus_window_minutes=0.0,
                focus_consistency=0.0,
                app_diversity=0.0,
                session_duration_minutes=0.0,
                input_intensity=0.0,
                time_of_day_hour=feature_vector.time_of_day_hour,
            ),
            feature_drift=FeatureVector(
                context_switch_entropy=0.0,
                focus_continuity_score=0.0,
                max_sustained_minutes=0.0,
                avg_focus_window_minutes=0.0,
                focus_consistency=0.0,
                app_diversity=0.0,
                session_duration_minutes=0.0,
                input_intensity=0.0,
                time_of_day_hour=feature_vector.time_of_day_hour,
            ),
            member_session_ids=[session_id],
            confidence=0.5,  # New tasks start with medium confidence
            drift=0.0,
        )
        # Attach additional metadata (apps, top windows, categories, signal quality)
        centroid.metadata = extra_metadata or {}
        # Also add a normalized view for numeric analysis and similarity features
        try:
            centroid.metadata['normalized'] = _normalize_extra_metadata(extra_metadata or {})
        except Exception:
            centroid.metadata['normalized'] = {}
        self.task_centroids[task_id] = centroid
        self.session_to_task[session_id] = task_id
        return task_id
    
    def _update_centroid(self, task_id: str, feature_vector: FeatureVector) -> None:
        """
        Update a centroid to include a new session's feature vector.
        
        - Recompute mean of all member features
        - Update EMA variance for stability
        - Update confidence based on member consistency, variance, drift
        - Track drift as change from previous centroid
        """
        centroid = self.task_centroids[task_id]
        n = len(centroid.member_session_ids)
        alpha = min(self.EMA_ALPHA, 1.0 / max(n, 1))  # smooth more as n grows

        # Store old centroid for drift calculation
        old_feature = centroid.feature_vector

        def _ema(old_val: float, new_val: float) -> float:
            return old_val * (1 - alpha) + new_val * alpha

        # Update mean (EMA)
        new_vector = FeatureVector(
            context_switch_entropy=_ema(old_feature.context_switch_entropy, feature_vector.context_switch_entropy),
            focus_continuity_score=_ema(old_feature.focus_continuity_score, feature_vector.focus_continuity_score),
            max_sustained_minutes=_ema(old_feature.max_sustained_minutes, feature_vector.max_sustained_minutes),
            avg_focus_window_minutes=_ema(old_feature.avg_focus_window_minutes, feature_vector.avg_focus_window_minutes),
            focus_consistency=_ema(old_feature.focus_consistency, feature_vector.focus_consistency),
            app_diversity=_ema(old_feature.app_diversity, feature_vector.app_diversity),
            session_duration_minutes=_ema(old_feature.session_duration_minutes, feature_vector.session_duration_minutes),
            input_intensity=_ema(old_feature.input_intensity, feature_vector.input_intensity),
            time_of_day_hour=centroid.feature_vector.time_of_day_hour,  # Keep mode time
        )

        # Update EMA variance per feature (variance of residuals)
        var = centroid.feature_variance
        def _ema_var(old_mean: float, old_var: float, new_val: float) -> float:
            delta = new_val - old_mean
            return (1 - alpha) * (old_var + alpha * (delta * delta))

        centroid.feature_variance = FeatureVector(
            context_switch_entropy=_ema_var(old_feature.context_switch_entropy, var.context_switch_entropy, feature_vector.context_switch_entropy),
            focus_continuity_score=_ema_var(old_feature.focus_continuity_score, var.focus_continuity_score, feature_vector.focus_continuity_score),
            max_sustained_minutes=_ema_var(old_feature.max_sustained_minutes, var.max_sustained_minutes, feature_vector.max_sustained_minutes),
            avg_focus_window_minutes=_ema_var(old_feature.avg_focus_window_minutes, var.avg_focus_window_minutes, feature_vector.avg_focus_window_minutes),
            focus_consistency=_ema_var(old_feature.focus_consistency, var.focus_consistency, feature_vector.focus_consistency),
            app_diversity=_ema_var(old_feature.app_diversity, var.app_diversity, feature_vector.app_diversity),
            session_duration_minutes=_ema_var(old_feature.session_duration_minutes, var.session_duration_minutes, feature_vector.session_duration_minutes),
            input_intensity=_ema_var(old_feature.input_intensity, var.input_intensity, feature_vector.input_intensity),
            time_of_day_hour=var.time_of_day_hour,
        )

        # Track per-feature drift (EMA of absolute deltas)
        drift_vec = centroid.feature_drift
        centroid.feature_drift = FeatureVector(
            context_switch_entropy=_ema(drift_vec.context_switch_entropy, abs(feature_vector.context_switch_entropy - old_feature.context_switch_entropy)),
            focus_continuity_score=_ema(drift_vec.focus_continuity_score, abs(feature_vector.focus_continuity_score - old_feature.focus_continuity_score)),
            max_sustained_minutes=_ema(drift_vec.max_sustained_minutes, abs(feature_vector.max_sustained_minutes - old_feature.max_sustained_minutes)),
            avg_focus_window_minutes=_ema(drift_vec.avg_focus_window_minutes, abs(feature_vector.avg_focus_window_minutes - old_feature.avg_focus_window_minutes)),
            focus_consistency=_ema(drift_vec.focus_consistency, abs(feature_vector.focus_consistency - old_feature.focus_consistency)),
            app_diversity=_ema(drift_vec.app_diversity, abs(feature_vector.app_diversity - old_feature.app_diversity)),
            session_duration_minutes=_ema(drift_vec.session_duration_minutes, abs(feature_vector.session_duration_minutes - old_feature.session_duration_minutes)),
            input_intensity=_ema(drift_vec.input_intensity, abs(feature_vector.input_intensity - old_feature.input_intensity)),
            time_of_day_hour=drift_vec.time_of_day_hour,
        )

        centroid.feature_vector = new_vector

        # Track drift: centroid shift plus per-feature divergence
        drift_shift = old_feature.distance_to(new_vector)
        feature_drift_values = [
            centroid.feature_drift.context_switch_entropy,
            centroid.feature_drift.focus_continuity_score,
            centroid.feature_drift.max_sustained_minutes,
            centroid.feature_drift.avg_focus_window_minutes,
            centroid.feature_drift.focus_consistency,
            centroid.feature_drift.app_diversity,
            centroid.feature_drift.session_duration_minutes,
            centroid.feature_drift.input_intensity,
        ]
        feature_drift_avg = sum(feature_drift_values) / len(feature_drift_values)

        # Accumulate total movement for downstream splitting/relabeling heuristics
        centroid.drift_cumulative = min(centroid.drift_cumulative + drift_shift, 10.0)

        if n > 1:
            # Combine recent shift and feature-level divergence; keep bounded [0,1]
            combined_drift = centroid.drift * 0.5 + drift_shift * 0.3 + feature_drift_avg * 0.2
            centroid.drift = max(0.0, min(combined_drift, 1.0))
        else:
            centroid.drift = min(drift_shift, 1.0)

        # Confidence combines: membership count, variance (stability), drift
        member_factor = min(n / self.MIN_MEMBERS_FOR_CONFIDENCE, 1.5)  # up to 1.5 boost
        # Average variance across features (lower = more stable)
        var_values = [
            centroid.feature_variance.context_switch_entropy,
            centroid.feature_variance.focus_continuity_score,
            centroid.feature_variance.max_sustained_minutes,
            centroid.feature_variance.avg_focus_window_minutes,
            centroid.feature_variance.focus_consistency,
            centroid.feature_variance.app_diversity,
            centroid.feature_variance.session_duration_minutes,
            centroid.feature_variance.input_intensity,
        ]
        avg_variance = sum(var_values) / len(var_values)
        variance_factor = max(0.0, 1.0 - avg_variance)  # high variance lowers confidence
        drift_factor = max(0.0, 1.0 - centroid.drift * 2)  # drift 0.5 -> 0

        raw_conf = 0.4 * min(member_factor, 1.5) + 0.35 * variance_factor + 0.25 * drift_factor
        centroid.confidence = max(0.0, min(raw_conf, 1.0))

        centroid.last_updated = datetime.now(timezone.utc).isoformat()
    
    def assign_label(self, task_id: str, label: str) -> bool:
        """Assign a human-provided label to a latent task."""
        if task_id not in self.task_centroids:
            return False
        self.task_centroids[task_id].label = label
        return True
    
    def get_task_summary(self, task_id: str) -> Optional[Dict]:
        """Get a summary of a task's characteristics."""
        if task_id not in self.task_centroids:
            return None
        
        centroid = self.task_centroids[task_id]
        feature_drift_avg = (
            centroid.feature_drift.context_switch_entropy +
            centroid.feature_drift.focus_continuity_score +
            centroid.feature_drift.max_sustained_minutes +
            centroid.feature_drift.avg_focus_window_minutes +
            centroid.feature_drift.focus_consistency +
            centroid.feature_drift.app_diversity +
            centroid.feature_drift.session_duration_minutes +
            centroid.feature_drift.input_intensity
        ) / 8.0
        meta = getattr(centroid, 'metadata', {}) or {}
        return {
            'task_id': task_id,
            'label': centroid.label,
            'members': len(centroid.member_session_ids),
            'confidence': round(centroid.confidence, 3),
            'drift': round(centroid.drift, 3),
            'drift_cumulative': round(centroid.drift_cumulative, 3),
            'feature_drift_avg': round(feature_drift_avg, 4),
            'is_unstable': centroid.is_unstable,
            'characteristics': {
                'entropy': round(centroid.feature_vector.context_switch_entropy, 3),
                'continuity': round(centroid.feature_vector.focus_continuity_score, 3),
                'max_sustained_minutes': round(centroid.feature_vector.max_sustained_minutes * 300, 1),
                'app_diversity': round(centroid.feature_vector.app_diversity, 3),
                'variance_entropy': round(centroid.feature_variance.context_switch_entropy, 4),
                'variance_continuity': round(centroid.feature_variance.focus_continuity_score, 4),
            },
            'metadata': meta,
            'created_at': centroid.created_at,
            'last_updated': centroid.last_updated,
        }
    
    def get_unstable_tasks(self) -> List[Dict]:
        """
        Get all tasks marked as unstable (drift > 0.15).
        
        Returns:
            List of unstable task summaries sorted by drift (highest first)
        """
        unstable = [
            {
                'task_id': task_id,
                'label': centroid.label,
                'members': len(centroid.member_session_ids),
                'confidence': round(centroid.confidence, 3),
                'drift': round(centroid.drift, 3),
                'drift_cumulative': round(centroid.drift_cumulative, 3),
                'severity': 'critical' if centroid.drift > 0.35 else 'high',
                'recommendation': self._get_stability_recommendation(centroid),
                'created_at': centroid.created_at,
                'last_updated': centroid.last_updated,
            }
            for task_id, centroid in self.task_centroids.items()
            if centroid.is_unstable
        ]
        
        # Sort by drift descending (highest drift first)
        return sorted(unstable, key=lambda x: x['drift'], reverse=True)
    
    def _get_stability_recommendation(self, centroid: TaskCentroid) -> str:
        """Generate human-readable recommendation based on instability."""
        if centroid.drift > 0.35:
            return "Task pattern highly unstable. Consider splitting into subtasks or reviewing scope."
        elif centroid.drift > 0.25:
            return "Task pattern moderately unstable. Monitor evolution; may need scope review."
        else:
            return "Task pattern shows variability. Verify consistency across sessions."
    
    def serialize(self) -> Dict:
        """Serialize state for persistence."""
        return {
            'version': '1.0',
            'task_centroids': {
                task_id: {
                    'task_id': centroid.task_id,
                    'feature_vector': asdict(centroid.feature_vector),
                    'feature_variance': asdict(centroid.feature_variance),
                    'feature_drift': asdict(centroid.feature_drift),
                    'member_session_ids': centroid.member_session_ids,
                    'confidence': centroid.confidence,
                    'drift': centroid.drift,
                    'drift_cumulative': centroid.drift_cumulative,
                    'label': centroid.label,
                    'metadata': getattr(centroid, 'metadata', {}),
                    'created_at': centroid.created_at,
                    'last_updated': centroid.last_updated,
                }
                for task_id, centroid in self.task_centroids.items()
            },
            'session_to_task': self.session_to_task,
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'TaskInferenceEngine':
        """Reconstruct from serialized state."""
        engine = cls()
        
        for task_id, centroid_data in data.get('task_centroids', {}).items():
            fv_data = centroid_data.get('feature_vector', {})
            fv_var_data = centroid_data.get('feature_variance', {}) or {
                'context_switch_entropy': 0.0,
                'focus_continuity_score': 0.0,
                'max_sustained_minutes': 0.0,
                'avg_focus_window_minutes': 0.0,
                'focus_consistency': 0.0,
                'app_diversity': 0.0,
                'session_duration_minutes': 0.0,
                'input_intensity': 0.0,
                'time_of_day_hour': fv_data.get('time_of_day_hour', 0),
            }
            fv_drift_data = centroid_data.get('feature_drift', {}) or {
                'context_switch_entropy': 0.0,
                'focus_continuity_score': 0.0,
                'max_sustained_minutes': 0.0,
                'avg_focus_window_minutes': 0.0,
                'focus_consistency': 0.0,
                'app_diversity': 0.0,
                'session_duration_minutes': 0.0,
                'input_intensity': 0.0,
                'time_of_day_hour': fv_data.get('time_of_day_hour', 0),
            }
            feature_vector = FeatureVector(**fv_data)
            feature_variance = FeatureVector(**fv_var_data)
            feature_drift = FeatureVector(**fv_drift_data)
            
            centroid = TaskCentroid(
                task_id=task_id,
                feature_vector=feature_vector,
                feature_variance=feature_variance,
                feature_drift=feature_drift,
                member_session_ids=centroid_data.get('member_session_ids', []),
                confidence=centroid_data.get('confidence', 0.5),
                drift=centroid_data.get('drift', 0.0),
                drift_cumulative=centroid_data.get('drift_cumulative', 0.0),
                label=centroid_data.get('label'),
                created_at=centroid_data.get('created_at', datetime.now(timezone.utc).isoformat()),
                last_updated=centroid_data.get('last_updated', datetime.now(timezone.utc).isoformat()),
            )
            # Restore metadata if present
            if 'metadata' in centroid_data:
                setattr(centroid, 'metadata', centroid_data.get('metadata') or {})
            engine.task_centroids[task_id] = centroid
        
        engine.session_to_task = data.get('session_to_task', {})
        return engine
