"""
Day 6: Trend & Comparative Analysis Layer

Operates entirely on Day 5 daily summaries to show:
- Trend metrics (deltas and slopes vs prior days)
- Baseline comparisons (vs rolling averages)
- Volatility and stability signals
- Pattern detection (pure pattern flags, not interpretations)

Fully deterministic. No scoring. No judgment.
"""

from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from statistics import mean, stdev, median
import math


# ============================================================================
# Trend Data Structure
# ============================================================================

@dataclass
class TrendMetrics:
    """Cross-day comparison metrics for a given day."""
    
    date: date
    
    # Today's values (from DailySummary)
    focus_ratio: float
    total_active_seconds: float
    fragmentation: float
    input_density: float
    session_count: int
    avg_session_duration: float
    
    # Day-over-day deltas
    focus_ratio_delta: Optional[float] = None  # points, e.g., +0.042 means +4.2%
    active_time_delta: Optional[float] = None  # seconds
    fragmentation_delta: Optional[float] = None  # sessions/hour
    input_density_delta: Optional[float] = None  # events/min
    
    # Rolling baseline comparisons (7-day)
    focus_ratio_vs_7day_pct: Optional[float] = None
    active_time_vs_7day_pct: Optional[float] = None
    fragmentation_vs_7day_pct: Optional[float] = None
    input_density_vs_7day_pct: Optional[float] = None
    
    # Volatility indicators
    session_duration_volatility: str = "unknown"  # "low", "moderate", "high"
    focus_volatility: str = "unknown"
    active_time_volatility: str = "unknown"
    
    # Detected patterns
    patterns: List[str] = field(default_factory=list)


# ============================================================================
# Calculation Functions
# ============================================================================

def calculate_day_over_day_deltas(
    today_summary,
    yesterday_summary: Optional = None
) -> Dict[str, Optional[float]]:
    """
    Calculate day-over-day deltas.
    
    Returns:
        Dict with keys: focus_ratio_delta, active_time_delta, etc.
    """
    deltas = {
        "focus_ratio_delta": None,
        "active_time_delta": None,
        "fragmentation_delta": None,
        "input_density_delta": None,
    }
    
    if yesterday_summary is None:
        return deltas
    
    # Focus ratio delta (in percentage points)
    if yesterday_summary.focus_ratio is not None:
        deltas["focus_ratio_delta"] = today_summary.focus_ratio - yesterday_summary.focus_ratio
    
    # Active time delta (in seconds)
    deltas["active_time_delta"] = today_summary.total_active_seconds - yesterday_summary.total_active_seconds
    
    # Fragmentation delta
    if yesterday_summary.fragmentation_index > 0:
        deltas["fragmentation_delta"] = today_summary.fragmentation_index - yesterday_summary.fragmentation_index
    
    # Input density delta
    if yesterday_summary.input_density > 0:
        deltas["input_density_delta"] = today_summary.input_density - yesterday_summary.input_density
    
    return deltas


def calculate_baseline_comparisons(
    today_summary,
    daily_summaries: List
) -> Dict[str, Optional[float]]:
    """
    Compare today's metrics to rolling 7-day baseline.
    
    Returns:
        Dict with _vs_7day_pct keys (percentage difference)
    """
    comparisons = {
        "focus_ratio_vs_7day_pct": None,
        "active_time_vs_7day_pct": None,
        "fragmentation_vs_7day_pct": None,
        "input_density_vs_7day_pct": None,
    }
    
    # Get 7-day window (excluding today)
    seven_day_window = [
        s for s in daily_summaries
        if s.date < today_summary.date and (today_summary.date - s.date).days < 7
    ]
    
    if not seven_day_window:
        return comparisons
    
    # Focus ratio comparison (vs mean)
    if seven_day_window:
        focus_values = [s.focus_ratio for s in seven_day_window if s.focus_ratio is not None]
        if focus_values:
            baseline = mean(focus_values)
            if baseline > 0:
                comparisons["focus_ratio_vs_7day_pct"] = ((today_summary.focus_ratio - baseline) / baseline) * 100
    
    # Active time comparison (vs mean)
    active_values = [s.total_active_seconds for s in seven_day_window]
    if active_values:
        baseline = mean(active_values)
        if baseline > 0:
            comparisons["active_time_vs_7day_pct"] = ((today_summary.total_active_seconds - baseline) / baseline) * 100
    
    # Fragmentation comparison (vs mean)
    frag_values = [s.fragmentation_index for s in seven_day_window]
    if frag_values:
        baseline = mean(frag_values)
        if baseline > 0:
            comparisons["fragmentation_vs_7day_pct"] = ((today_summary.fragmentation_index - baseline) / baseline) * 100
    
    # Input density comparison (vs mean)
    input_values = [s.input_density for s in seven_day_window]
    if input_values:
        baseline = mean(input_values)
        if baseline > 0:
            comparisons["input_density_vs_7day_pct"] = ((today_summary.input_density - baseline) / baseline) * 100
    
    return comparisons


def calculate_volatility(
    daily_summaries: List,
    window_days: int = 7
) -> Dict[str, str]:
    """
    Calculate volatility indicators for the past N days.
    
    Returns deterministic labels: "low", "moderate", "high"
    """
    volatility = {
        "session_duration_volatility": "unknown",
        "focus_volatility": "unknown",
        "active_time_volatility": "unknown",
    }
    
    if not daily_summaries:
        return volatility
    
    # Session duration coefficient of variation
    durations = [s.avg_session_duration_seconds for s in daily_summaries[-window_days:] if s.avg_session_duration_seconds > 0]
    if len(durations) > 1:
        cv = (stdev(durations) / mean(durations)) * 100 if mean(durations) > 0 else 0
        volatility["session_duration_volatility"] = _classify_volatility(cv)
    
    # Focus ratio coefficient of variation
    focus_values = [s.focus_ratio for s in daily_summaries[-window_days:] if s.focus_ratio is not None]
    if len(focus_values) > 1:
        cv = (stdev(focus_values) / mean(focus_values)) * 100 if mean(focus_values) > 0 else 0
        volatility["focus_volatility"] = _classify_volatility(cv)
    
    # Active time coefficient of variation
    active_values = [s.total_active_seconds for s in daily_summaries[-window_days:]]
    if len(active_values) > 1:
        cv = (stdev(active_values) / mean(active_values)) * 100 if mean(active_values) > 0 else 0
        volatility["active_time_volatility"] = _classify_volatility(cv)
    
    return volatility


def _classify_volatility(cv: float) -> str:
    """
    Classify coefficient of variation into deterministic buckets.
    
    CV = (std dev / mean) * 100
    - CV < 15%: Low
    - CV 15-35%: Moderate
    - CV > 35%: High
    """
    if cv < 15:
        return "low"
    elif cv < 35:
        return "moderate"
    else:
        return "high"


def detect_patterns(
    daily_summaries: List,
    today_summary
) -> List[str]:
    """
    Detect pure pattern flags from recent daily summaries.
    
    Returns list of pattern descriptions (no interpretation).
    """
    patterns = []
    
    if not daily_summaries:
        return patterns
    
    # Get last 7 days (excluding today)
    recent = daily_summaries[-7:]
    
    if not recent:
        return patterns
    
    # Pattern 1: Fragmentation trend
    frag_values = [s.fragmentation_index for s in recent]
    if len(frag_values) >= 3:
        last_3 = frag_values[-3:]
        if all(last_3[i] < last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Fragmentation increasing for 3+ consecutive days")
        elif all(last_3[i] > last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Fragmentation decreasing for 3+ consecutive days")
    
    # Pattern 2: Focus ratio trend
    focus_values = [s.focus_ratio for s in recent if s.focus_ratio is not None]
    if len(focus_values) >= 3:
        last_3 = focus_values[-3:]
        if all(last_3[i] < last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Focus ratio improving for 3+ consecutive days")
        elif all(last_3[i] > last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Focus ratio declining for 3+ consecutive days")
    
    # Pattern 3: Session count trend
    session_counts = [s.session_count for s in recent]
    if len(session_counts) >= 3:
        last_3 = session_counts[-3:]
        avg_recent = mean(last_3)
        if all(c < 4 for c in last_3):
            patterns.append("Consistently few sessions (low fragmentation)")
        elif all(c > 8 for c in last_3):
            patterns.append("Consistently many sessions (high fragmentation)")
    
    # Pattern 4: Input density trend
    input_values = [s.input_density for s in recent]
    if len(input_values) >= 3:
        last_3 = input_values[-3:]
        if all(last_3[i] < last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Input density increasing for 3+ days")
        elif all(last_3[i] > last_3[i+1] for i in range(len(last_3)-1)):
            patterns.append("Input density decreasing for 3+ days")
    
    # Pattern 5: Active time variance
    active_values = [s.total_active_seconds for s in recent]
    if len(active_values) >= 5:
        low_days = sum(1 for v in active_values if v < 3600)  # < 1 hour
        high_days = sum(1 for v in active_values if v > 28800)  # > 8 hours
        if low_days >= 2:
            patterns.append("Multiple very low-activity days")
        if high_days >= 2:
            patterns.append("Multiple high-activity days")
    
    return patterns


# ============================================================================
# Analysis Aggregator
# ============================================================================

def compute_trend_metrics(
    today_summary,
    all_daily_summaries: List
) -> TrendMetrics:
    """
    Compute all trend metrics for a given day.
    
    Args:
        today_summary: DailySummary for today
        all_daily_summaries: List of all historical DailySummary objects
    
    Returns:
        TrendMetrics with all comparisons and patterns
    """
    # Find yesterday's summary
    yesterday = today_summary.date - timedelta(days=1)
    yesterday_summary = next(
        (s for s in all_daily_summaries if s.date == yesterday),
        None
    )
    
    # Calculate deltas
    deltas = calculate_day_over_day_deltas(today_summary, yesterday_summary)
    
    # Calculate baseline comparisons
    baselines = calculate_baseline_comparisons(today_summary, all_daily_summaries)
    
    # Calculate volatility (7-day window)
    volatility = calculate_volatility(all_daily_summaries)
    
    # Detect patterns
    patterns = detect_patterns(all_daily_summaries, today_summary)
    
    # Assemble trend metrics
    metrics = TrendMetrics(
        date=today_summary.date,
        focus_ratio=today_summary.focus_ratio,
        total_active_seconds=today_summary.total_active_seconds,
        fragmentation=today_summary.fragmentation_index,
        input_density=today_summary.input_density,
        session_count=today_summary.session_count,
        avg_session_duration=today_summary.avg_session_duration_seconds,
        **deltas,
        **baselines,
        **volatility,
        patterns=patterns,
    )
    
    return metrics


# ============================================================================
# Reporting
# ============================================================================

def format_trend_report(metrics: TrendMetrics) -> str:
    """Generate a human-readable trend report."""
    report = []
    report.append("=" * 75)
    report.append(f"TREND ANALYSIS — {metrics.date} (Last 7 Days)")
    report.append("=" * 75)
    
    # Trend Metrics (vs yesterday)
    report.append("\n[TREND METRICS (vs Previous Day)]")
    
    focus_arrow = "↑" if metrics.focus_ratio_delta and metrics.focus_ratio_delta > 0 else "↓" if metrics.focus_ratio_delta and metrics.focus_ratio_delta < 0 else "→"
    report.append(f"  Focus Ratio: {metrics.focus_ratio:.1%} {focus_arrow} {metrics.focus_ratio_delta:+.1%}" if metrics.focus_ratio_delta else f"  Focus Ratio: {metrics.focus_ratio:.1%} (no prior day)")
    
    if metrics.active_time_delta is not None:
        hours = metrics.active_time_delta / 3600
        minutes = (abs(metrics.active_time_delta) % 3600) / 60
        arrow = "↑" if metrics.active_time_delta > 0 else "↓" if metrics.active_time_delta < 0 else "→"
        sign = "+" if metrics.active_time_delta > 0 else ""
        report.append(f"  Total Active Time: {metrics.total_active_seconds/3600:.1f}h {arrow} {sign}{hours:.1f}h ({sign}{minutes:.0f}m)")
    else:
        report.append(f"  Total Active Time: {metrics.total_active_seconds/3600:.1f}h (no prior day)")
    
    if metrics.fragmentation_delta is not None:
        arrow = "↑" if metrics.fragmentation_delta > 0 else "↓" if metrics.fragmentation_delta < 0 else "→"
        report.append(f"  Fragmentation: {metrics.fragmentation:.1f} {arrow} {metrics.fragmentation_delta:+.1f}")
    else:
        report.append(f"  Fragmentation: {metrics.fragmentation:.1f} (no prior day)")
    
    if metrics.input_density_delta is not None:
        arrow = "↑" if metrics.input_density_delta > 0 else "↓" if metrics.input_density_delta < 0 else "→"
        report.append(f"  Input Density: {metrics.input_density:.1f}/min {arrow} {metrics.input_density_delta:+.1f}/min")
    else:
        report.append(f"  Input Density: {metrics.input_density:.1f}/min (no prior day)")
    
    # Baseline Comparisons (7-day)
    report.append("\n[BASELINE COMPARISONS (vs 7-Day Average)]")
    if metrics.focus_ratio_vs_7day_pct is not None:
        report.append(f"  Focus Ratio: {metrics.focus_ratio:.1%} ({metrics.focus_ratio_vs_7day_pct:+.1f}%)")
    if metrics.active_time_vs_7day_pct is not None:
        report.append(f"  Active Time: {metrics.total_active_seconds/3600:.1f}h ({metrics.active_time_vs_7day_pct:+.1f}%)")
    if metrics.fragmentation_vs_7day_pct is not None:
        report.append(f"  Fragmentation: {metrics.fragmentation:.1f} ({metrics.fragmentation_vs_7day_pct:+.1f}%)")
    if metrics.input_density_vs_7day_pct is not None:
        report.append(f"  Input Density: {metrics.input_density:.1f}/min ({metrics.input_density_vs_7day_pct:+.1f}%)")
    
    # Volatility/Stability
    report.append("\n[STABILITY INDICATORS (7-Day)]")
    report.append(f"  Session Duration: {metrics.session_duration_volatility.upper()}")
    report.append(f"  Focus Ratio: {metrics.focus_volatility.upper()}")
    report.append(f"  Active Time: {metrics.active_time_volatility.upper()}")
    
    # Patterns
    if metrics.patterns:
        report.append("\n[DETECTED PATTERNS]")
        for pattern in metrics.patterns:
            report.append(f"  • {pattern}")
    else:
        report.append("\n[DETECTED PATTERNS]")
        report.append("  None")
    
    report.append("\n" + "=" * 75)
    
    return "\n".join(report)


def print_trend_report(metrics: TrendMetrics) -> None:
    """Print a formatted trend report to console."""
    print(format_trend_report(metrics))
