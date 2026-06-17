"""
Day 5: Session Analytics & Aggregation Layer

Transforms completed sessions into daily summaries with:
- Session classification (focused, passive, mixed, etc.)
- Daily aggregation metrics
- Productivity signals (non-judgmental indicators)
- Deterministic output format
"""

from datetime import datetime, timezone, date, timedelta
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter


# ============================================================================
# Session Classification
# ============================================================================

def classify_session(session) -> str:
    """
    Classify a session into behavioral categories using deterministic rules.
    
    Returns one of: "focused", "passive", "mixed", "idle-heavy"
    """
    duration_seconds = (session.end - session.start).total_seconds()
    
    # Calculate input density
    total_input = session.input_events["keys"] + session.input_events["clicks"]
    input_per_minute = total_input / (duration_seconds / 60) if duration_seconds > 0 else 0
    
    # Calculate app diversity
    app_count = len(session.apps)
    
    # Classification rules (deterministic, order matters)
    
    # Idle-heavy: very few input events relative to duration
    if total_input < 5 and duration_seconds > 300:  # <5 inputs in 5+ minutes
        return "idle-heavy"
    
    # Focused: single app, high input density, sustained activity
    if app_count == 1 and input_per_minute >= 4:
        return "focused"
    
    # Passive: sustained single app, low input (reading, watching)
    if app_count == 1 and input_per_minute < 4:
        return "passive"
    
    # Mixed: multiple apps, variable input
    if app_count > 1:
        return "mixed"
    
    # Default
    return "passive"


# ============================================================================
# Daily Summary Data Structure
# ============================================================================

@dataclass
class DailySummary:
    """Aggregates all sessions within a single calendar day."""
    
    date: date
    sessions: List = field(default_factory=list)
    
    # Time-based metrics
    total_active_seconds: float = 0.0
    total_idle_seconds: float = 0.0
    session_count: int = 0
    avg_session_duration_seconds: float = 0.0
    longest_session_seconds: float = 0.0
    
    # Input-based metrics
    total_keys: int = 0
    total_clicks: int = 0
    total_mouse_distance: int = 0
    
    # App-based metrics
    app_time: Dict[str, float] = field(default_factory=dict)  # app -> seconds
    app_percentages: Dict[str, float] = field(default_factory=dict)  # app -> %
    most_used_app: Optional[str] = None
    app_switch_count: int = 0
    
    # Classification breakdown
    activity_breakdown: Dict[str, int] = field(default_factory=dict)  # type -> count
    
    # Derived signals
    focus_ratio: float = 0.0  # focused_time / total_active_time
    fragmentation_index: float = 0.0  # sessions_per_hour
    input_density: float = 0.0  # (keys + clicks) / active_minutes
    avg_app_switches_per_session: float = 0.0
    
    # Feature extraction (deterministic, versioned)
    context_switch_entropy: float = 0.0  # Shannon entropy of app-switch sequence
    focus_continuity_score: float = 0.0  # Weighted duration of focus periods
    sustained_activity_metrics: Dict[str, float] = field(default_factory=dict)  # breakdown of activity windows
    
    # Provenance & Versioning
    feature_schema_version: str = "1.0"  # Feature computation schema version
    computed_at: Optional[datetime] = None  # ISO timestamp of computation
    
    # Task/intent metrics
    task_time: Dict[Optional[str], float] = field(default_factory=dict)  # intent_id -> seconds
    task_switch_count: int = 0


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def compute_context_switch_entropy(sessions: List) -> float:
    """
    Compute Shannon entropy of the app-switch sequence.
    
    Entropy measures the "surprise" or unpredictability of app switches.
    - High entropy (closer to 1.0): Many different apps, uniform distribution
    - Low entropy (closer to 0.0): Few apps, predictable switching pattern
    
    Returns:
        Entropy value in range [0.0, 1.0] normalized by log2(num_apps)
    """
    if not sessions:
        return 0.0
    
    # Build sequence of apps visited across all sessions
    app_sequence = []
    for session in sessions:
        if session.apps:
            app_sequence.extend(session.apps)
    
    if len(app_sequence) < 2:
        return 0.0
    
    # Count transitions (app_i -> app_{i+1})
    transitions = Counter()
    for i in range(len(app_sequence) - 1):
        if app_sequence[i] != app_sequence[i + 1]:  # Only count actual switches
            key = (app_sequence[i], app_sequence[i + 1])
            transitions[key] += 1
    
    if not transitions:
        return 0.0
    
    # Compute Shannon entropy
    total_switches = sum(transitions.values())
    entropy = 0.0
    for count in transitions.values():
        if count > 0:
            p = count / total_switches
            entropy -= p * math.log2(p)
    
    # Normalize by maximum entropy (log2 of number of unique transitions)
    unique_transitions = len(transitions)
    max_entropy = math.log2(unique_transitions) if unique_transitions > 1 else 1.0
    
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return min(1.0, normalized_entropy)  # Cap at 1.0


def compute_focus_continuity_score(sessions: List) -> float:
    """
    Compute a continuity score for focus periods.
    
    Measures how sustained and unbroken focus sessions are:
    - High score (closer to 1.0): Long, uninterrupted focus sessions
    - Low score (closer to 0.0): Fragmented focus with many interruptions
    
    Calculation:
    - For each "focused" session, accumulate weighted duration (longer = higher weight)
    - Divide by theoretical maximum (all time in one long focused session)
    
    Returns:
        Score in range [0.0, 1.0]
    """
    if not sessions:
        return 0.0
    
    # Filter focused sessions
    focused_sessions = [
        s for s in sessions 
        if classify_session(s) == "focused"
    ]
    
    if not focused_sessions:
        return 0.0
    
    # Calculate weighted continuity
    # Longer sessions get higher weight (quadratic to emphasize sustained focus)
    total_weighted_duration = 0.0
    for session in focused_sessions:
        duration = (session.end - session.start).total_seconds()
        # Weight by duration (longer = exponentially more valuable)
        weighted = duration ** 1.5  # Exponential weighting
        total_weighted_duration += weighted
    
    # Maximum possible score: all time in single long session
    max_duration = max(
        (s.end - s.start).total_seconds() 
        for s in focused_sessions
    )
    theoretical_max = sum(
        (s.end - s.start).total_seconds() 
        for s in focused_sessions
    ) ** 1.5
    
    if theoretical_max == 0:
        return 0.0
    
    # Normalize and return
    continuity_score = total_weighted_duration / theoretical_max
    return min(1.0, continuity_score)


def compute_sustained_activity_metrics(sessions: List) -> Dict[str, float]:
    """
    Compute breakdown of sustained activity windows.
    
    Analyzes how activity is distributed across sessions:
    - max_sustained_minutes: Duration of longest unbroken focus period
    - avg_focus_window_minutes: Average length of focus periods
    - focus_window_count: Number of distinct focus periods
    - longest_gap_minutes: Longest idle/break between focus sessions
    - focus_consistency: Std dev of focus durations (lower = more consistent)
    
    Returns:
        Dict with metrics summarizing sustained activity patterns
    """
    if not sessions:
        return {
            "max_sustained_minutes": 0.0,
            "avg_focus_window_minutes": 0.0,
            "focus_window_count": 0,
            "longest_gap_minutes": 0.0,
            "focus_consistency": 0.0,
        }
    
    # Extract focused sessions and their durations
    focused_sessions = [
        s for s in sessions 
        if classify_session(s) == "focused"
    ]
    
    if not focused_sessions:
        return {
            "max_sustained_minutes": 0.0,
            "avg_focus_window_minutes": 0.0,
            "focus_window_count": 0,
            "longest_gap_minutes": 0.0,
            "focus_consistency": 0.0,
        }
    
    # Sort by start time
    focused_sessions.sort(key=lambda s: s.start)
    
    # Durations in minutes
    durations_minutes = [
        (s.end - s.start).total_seconds() / 60 
        for s in focused_sessions
    ]
    
    # Calculate gaps between focus sessions
    gaps_minutes = []
    for i in range(len(focused_sessions) - 1):
        gap = (focused_sessions[i + 1].start - focused_sessions[i].end).total_seconds() / 60
        if gap > 0:
            gaps_minutes.append(gap)
    
    # Calculate consistency (lower std dev = more consistent)
    if len(durations_minutes) > 1:
        mean_duration = sum(durations_minutes) / len(durations_minutes)
        variance = sum((d - mean_duration) ** 2 for d in durations_minutes) / len(durations_minutes)
        std_dev = math.sqrt(variance)
        # Normalize by mean to get coefficient of variation
        consistency = 1.0 / (1.0 + std_dev / mean_duration) if mean_duration > 0 else 0.0
    else:
        consistency = 1.0  # Single window = perfect consistency
    
    return {
        "max_sustained_minutes": max(durations_minutes),
        "avg_focus_window_minutes": sum(durations_minutes) / len(durations_minutes),
        "focus_window_count": len(focused_sessions),
        "longest_gap_minutes": max(gaps_minutes) if gaps_minutes else 0.0,
        "focus_consistency": min(1.0, consistency),
    }


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_daily_summary(sessions: List, target_date: date = None) -> DailySummary:
    """
    Aggregate all sessions from a given day into a DailySummary.
    
    Args:
        sessions: List of completed Session objects
        target_date: Date to aggregate for. If None, uses today.
    
    Returns:
        DailySummary with all computed metrics
    """
    if target_date is None:
        target_date = date.today()
    
    # Filter sessions for the target date
    day_sessions = [
        s for s in sessions
        if s.start.astimezone().date() == target_date
    ]
    
    summary = DailySummary(date=target_date, sessions=day_sessions)
    summary.computed_at = datetime.now(timezone.utc)  # Record computation time
    
    if not day_sessions:
        return summary
    
    # --- Time-based metrics ---
    summary.session_count = len(day_sessions)
    
    for session in day_sessions:
        duration = (session.end - session.start).total_seconds()
        summary.total_active_seconds += duration
        summary.longest_session_seconds = max(summary.longest_session_seconds, duration)
    
    if summary.session_count > 0:
        summary.avg_session_duration_seconds = summary.total_active_seconds / summary.session_count
    
    # --- Input metrics ---
    for session in day_sessions:
        summary.total_keys += session.input_events["keys"]
        summary.total_clicks += session.input_events["clicks"]
        summary.total_mouse_distance += session.input_events["mouse_distance"]

    # --- Task attribution (if available) ---
    # Use attribution resolver if session contains segments or intents file exists
    try:
        from agent.intent.attribution import resolve_session_attribution
        INTENTS_FILE = os.path.join("agent", "intent", "intents.json")
    except Exception:
        resolve_session_attribution = None
        INTENTS_FILE = None

    task_time_acc = defaultdict(float)
    task_switches = 0
    for session in day_sessions:
        # Prefer persisted segments on the session
        if getattr(session, "intent_segments", None):
            # sum durations per intent
            segments = session.intent_segments
            last_intent = None
            for intent_id, s_ts, e_ts in segments:
                if s_ts is None:
                    continue
                e = e_ts or session.end
                dur = (e - s_ts).total_seconds()
                task_time_acc[intent_id] += dur
                if last_intent is not None and intent_id != last_intent:
                    task_switches += 1
                last_intent = intent_id
        elif resolve_session_attribution:
            breakdown = resolve_session_attribution(session, INTENTS_FILE)
            for intent_id, seconds in breakdown.items():
                task_time_acc[intent_id] += seconds
            # count switches as number of distinct intents - 1
            if breakdown:
                task_switches += max(0, len(breakdown) - 1)

    summary.task_time = dict(task_time_acc)
    summary.task_switch_count = task_switches
    
    # --- App usage metrics ---
    app_time = defaultdict(float)
    last_app_set = set()
    
    for session in day_sessions:
        # Time per app (distribute session time equally across apps)
        if session.apps:
            time_per_app = (session.end - session.start).total_seconds() / len(session.apps)
            for app in session.apps:
                app_time[app] += time_per_app
        
        # Count app switches
        current_apps = session.apps
        if last_app_set and current_apps != last_app_set:
            summary.app_switch_count += 1
        last_app_set = current_apps
    
    summary.app_time = dict(app_time)
    
    # Calculate app percentages
    if summary.total_active_seconds > 0 and summary.app_time:
        for app, time_seconds in summary.app_time.items():
            summary.app_percentages[app] = (time_seconds / summary.total_active_seconds) * 100

        # Most used app
        summary.most_used_app = max(summary.app_time.keys(), key=lambda x: summary.app_time[x])
    else:
        summary.most_used_app = None
    
    # --- Classification breakdown ---
    activity_breakdown = defaultdict(int)
    focused_time = 0.0
    
    for session in day_sessions:
        activity_type = classify_session(session)
        activity_breakdown[activity_type] += 1
        
        if activity_type == "focused":
            focused_time += (session.end - session.start).total_seconds()
    
    summary.activity_breakdown = dict(activity_breakdown)
    
    # --- Derived signals ---
    active_minutes = summary.total_active_seconds / 60 if summary.total_active_seconds > 0 else 1
    
    # Focus ratio
    if summary.total_active_seconds > 0:
        summary.focus_ratio = focused_time / summary.total_active_seconds
    
    # Fragmentation index (sessions per hour)
    hours = summary.total_active_seconds / 3600 if summary.total_active_seconds > 0 else 1
    summary.fragmentation_index = summary.session_count / hours
    
    # Input density
    summary.input_density = (summary.total_keys + summary.total_clicks) / active_minutes
    
    # Avg app switches per session
    if summary.session_count > 0:
        summary.avg_app_switches_per_session = summary.app_switch_count / summary.session_count
    
    # --- Feature Extraction (deterministic, versioned) ---
    # Compute entropy-based metrics
    summary.context_switch_entropy = compute_context_switch_entropy(day_sessions)
    
    # Compute focus continuity
    summary.focus_continuity_score = compute_focus_continuity_score(day_sessions)
    
    # Compute sustained activity metrics
    summary.sustained_activity_metrics = compute_sustained_activity_metrics(day_sessions)
    
    return summary


# ============================================================================
# Reporting
# ============================================================================

def format_daily_report(summary: DailySummary) -> str:
    """
    Generate a human-readable daily report from a DailySummary.
    """
    report = []
    report.append("=" * 75)
    report.append(f"DAILY SUMMARY — {summary.date}")
    report.append("=" * 75)
    
    # Overview
    report.append("\n[OVERVIEW]")
    report.append(f"  Sessions: {summary.session_count}")
    hours = summary.total_active_seconds / 3600
    minutes = (summary.total_active_seconds % 3600) / 60
    report.append(f"  Total Active Time: {hours:.0f}h {minutes:.0f}m ({summary.total_active_seconds:.0f}s)")
    if summary.session_count > 0:
        avg_min = summary.avg_session_duration_seconds / 60
        longest_min = summary.longest_session_seconds / 60
        report.append(f"  Avg Session: {avg_min:.1f}m | Longest: {longest_min:.1f}m")
    
    # Session Breakdown
    if summary.activity_breakdown:
        report.append("\n[SESSION BREAKDOWN BY TYPE]")
        for activity_type, count in sorted(summary.activity_breakdown.items()):
            report.append(f"  {activity_type.upper()}: {count}")
    
    # App Usage
    if summary.app_time:
        report.append("\n[APP USAGE]")
        report.append(f"  Most Used: {summary.most_used_app}")
        for app in sorted(summary.app_time.keys(), key=lambda x: summary.app_time[x], reverse=True):
            time_sec = summary.app_time[app]
            time_min = time_sec / 60
            pct = summary.app_percentages.get(app, 0)
            report.append(f"    {app}: {time_min:.0f}m ({pct:.1f}%)")
        report.append(f"  App Switches: {summary.app_switch_count}")
    
    # Input Statistics
    report.append("\n[INPUT STATISTICS]")
    report.append(f"  Keys: {summary.total_keys}")
    report.append(f"  Clicks: {summary.total_clicks}")
    report.append(f"  Mouse Distance: {summary.total_mouse_distance:,} px")
    active_min = summary.total_active_seconds / 60 if summary.total_active_seconds > 0 else 1
    report.append(f"  Input Density: {summary.input_density:.1f} events/min")
    
    # Derived Signals
    report.append("\n[PRODUCTIVITY SIGNALS]")
    report.append(f"  Focus Ratio: {summary.focus_ratio:.1%} (focused time / total active)")
    report.append(f"  Fragmentation: {summary.fragmentation_index:.1f} sessions/hour")
    report.append(f"  Avg Switches/Session: {summary.avg_app_switches_per_session:.1f}")
    
    # Feature Extraction Results
    report.append("\n[ADVANCED FEATURES]")
    report.append(f"  Context Switch Entropy: {summary.context_switch_entropy:.3f}")
    report.append(f"    (0.0=predictable, 1.0=chaotic)")
    report.append(f"  Focus Continuity Score: {summary.focus_continuity_score:.3f}")
    report.append(f"    (0.0=fragmented, 1.0=sustained)")
    if summary.sustained_activity_metrics:
        report.append("  Sustained Activity:")
        metrics = summary.sustained_activity_metrics
        report.append(f"    Max Sustained: {metrics.get('max_sustained_minutes', 0):.0f}m")
        report.append(f"    Avg Focus Window: {metrics.get('avg_focus_window_minutes', 0):.0f}m")
        report.append(f"    Focus Window Count: {metrics.get('focus_window_count', 0)}")
        report.append(f"    Longest Gap: {metrics.get('longest_gap_minutes', 0):.0f}m")
        report.append(f"    Focus Consistency: {metrics.get('focus_consistency', 0):.3f}")
    
    # Metadata
    if summary.computed_at:
        report.append(f"\n[METADATA]")
        report.append(f"  Feature Schema Version: {summary.feature_schema_version}")
        report.append(f"  Computed At: {summary.computed_at.isoformat()}")
    
    report.append("\n" + "=" * 75)

    # Task / Intent Breakdown
    if summary.task_time:
        report.append("\n[TASK BREAKDOWN]")
        # Format: intent_id or Unattributed
        for intent_id, seconds in sorted(summary.task_time.items(), key=lambda x: x[1], reverse=True):
            label = "Unattributed" if intent_id is None else intent_id
            mins = seconds / 60
            report.append(f"  {label}: {mins:.0f}m ({int(seconds)}s)")
        report.append(f"  Task Switches: {summary.task_switch_count}")
    
    return "\n".join(report)


def print_daily_report(summary: DailySummary) -> None:
    """Print a formatted daily report to console."""
    print(format_daily_report(summary))
