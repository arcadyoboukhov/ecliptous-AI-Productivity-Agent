"""
Day 7: Weekly Baselines, Deviation Detection, and Pattern Persistence

Computes 7-day rolling baselines, flags deviations (sigma), and
identifies stable patterns (persistence over >=4 of last 7 days).

Inputs: List[DailySummary]
Outputs: Weekly report (string) and structured results
"""
from datetime import date, timedelta
from statistics import mean, stdev
from typing import List, Dict, Any, Optional


def _classify_volatility_cv(cv: float) -> str:
    if cv < 15:
        return "low"
    elif cv < 35:
        return "moderate"
    else:
        return "high"


def compute_7day_baseline(daily_summaries: List, target_date: date) -> Dict[str, Any]:
    """
    Compute 7-day rolling baseline (mean ± std) for core metrics.
    Uses days strictly before target_date (recent up to 7 days).
    Gracefully degrades if fewer than 7 days available.
    Returns dict with mean/std and count.
    """
    window_days = 7
    window = [
        s for s in daily_summaries
        if s.date < target_date and (target_date - s.date).days <= window_days
    ]
    window = sorted(window, key=lambda s: s.date)

    result = {
        "count": len(window),
        "active_time_mean": None,
        "active_time_std": None,
        "avg_session_mean": None,
        "avg_session_std": None,
        "focus_ratio_mean": None,
        "focus_ratio_std": None,
        "fragmentation_mean": None,
        "fragmentation_std": None,
        "input_density_mean": None,
        "input_density_std": None,
    }

    if not window:
        return result

    def _stats(values):
        if not values:
            return (None, None)
        if len(values) == 1:
            return (mean(values), 0.0)
        return (mean(values), stdev(values))

    active_vals = [s.total_active_seconds for s in window]
    avg_session_vals = [s.avg_session_duration_seconds for s in window]
    focus_vals = [s.focus_ratio for s in window if s.focus_ratio is not None]
    frag_vals = [s.fragmentation_index for s in window]
    input_vals = [s.input_density for s in window]

    result["active_time_mean"], result["active_time_std"] = _stats(active_vals)
    result["avg_session_mean"], result["avg_session_std"] = _stats(avg_session_vals)
    result["focus_ratio_mean"], result["focus_ratio_std"] = _stats(focus_vals)
    result["fragmentation_mean"], result["fragmentation_std"] = _stats(frag_vals)
    result["input_density_mean"], result["input_density_std"] = _stats(input_vals)

    return result


def detect_deviations(today_summary, baseline: Dict[str, Any]) -> List[str]:
    """
    Flag deviations of today's metrics against the baseline using sigma.
    Returns list of human-readable deviation strings.
    """
    notes = []

    def _z(val, mean_v, std_v):
        if mean_v is None or std_v is None or std_v == 0:
            return None
        return (val - mean_v) / std_v

    # Active time
    z_active = _z(today_summary.total_active_seconds, baseline.get("active_time_mean"), baseline.get("active_time_std"))
    if z_active is not None and abs(z_active) >= 1.8:
        direction = "above" if z_active > 0 else "below"
        notes.append(f"Active time is {abs(z_active):.1f}σ {direction} baseline")

    # Focus ratio
    z_focus = _z(today_summary.focus_ratio, baseline.get("focus_ratio_mean"), baseline.get("focus_ratio_std"))
    if z_focus is not None and abs(z_focus) >= 1.8:
        direction = "above" if z_focus > 0 else "below"
        notes.append(f"Focus ratio is {abs(z_focus):.1f}σ {direction} baseline")

    # Fragmentation
    z_frag = _z(today_summary.fragmentation_index, baseline.get("fragmentation_mean"), baseline.get("fragmentation_std"))
    if z_frag is not None and abs(z_frag) >= 1.8:
        direction = "above" if z_frag > 0 else "below"
        notes.append(f"Fragmentation is {abs(z_frag):.1f}σ {direction} baseline")

    # Input density
    z_input = _z(today_summary.input_density, baseline.get("input_density_mean"), baseline.get("input_density_std"))
    if z_input is not None and abs(z_input) >= 1.8:
        direction = "above" if z_input > 0 else "below"
        notes.append(f"Input density is {abs(z_input):.1f}σ {direction} baseline")

    # Cross-metric anomaly: input spike without active time increase
    if z_input is not None and z_active is not None:
        if z_input > 1.8 and z_active < 1.0:
            notes.append("Input density spike without corresponding active time increase")

    return notes


def identify_stable_patterns(daily_summaries: List, target_date: date) -> List[str]:
    """
    Identify patterns that persist across the last 7 days (including today if present).
    Patterns must occur in >=4 of the last 7 days.
    Uses available DailySummary fields.
    """
    patterns = []
    window_days = 7
    recent = [s for s in daily_summaries if (target_date - s.date).days < window_days and (target_date - s.date).days >= 0]
    recent = sorted(recent, key=lambda s: s.date)
    if not recent:
        return patterns

    # Pattern: App dominance across recent days
    app_totals: Dict[str, float] = {}
    for s in recent:
        for app, secs in (s.app_time or {}).items():
            app_totals[app] = app_totals.get(app, 0.0) + secs
    total_time = sum(app_totals.values()) if app_totals else 0.0
    if total_time > 0:
        for app, secs in app_totals.items():
            pct = (secs / total_time) * 100
            if pct >= 50.0:
                patterns.append(f"{app} accounts for {pct:.0f}% of active time over last {len(recent)} days")

    # Pattern: Consistently shorter sessions (avg session shorter than baseline) for >=4 days
    baseline = compute_7day_baseline(daily_summaries, target_date)
    avg_mean = baseline.get("avg_session_mean")
    avg_std = baseline.get("avg_session_std")
    shorter_count = 0
    longer_count = 0
    for s in recent:
        if avg_mean is not None and avg_std is not None:
            if s.avg_session_duration_seconds < (avg_mean - avg_std):
                shorter_count += 1
            if s.avg_session_duration_seconds > (avg_mean + avg_std):
                longer_count += 1
    if shorter_count >= 4:
        patterns.append(f"Consistently shorter sessions ({shorter_count}/{len(recent)} days) compared to baseline")
    if longer_count >= 4:
        patterns.append(f"Consistently longer sessions ({longer_count}/{len(recent)} days) compared to baseline")

    # Pattern: Focus ratio persistent high/low
    focus_high_count = 0
    focus_low_count = 0
    fr_mean = baseline.get("focus_ratio_mean")
    fr_std = baseline.get("focus_ratio_std")
    if fr_mean is not None and fr_std is not None:
        for s in recent:
            if s.focus_ratio is None:
                continue
            if s.focus_ratio > (fr_mean + fr_std):
                focus_high_count += 1
            if s.focus_ratio < (fr_mean - fr_std):
                focus_low_count += 1
    if focus_high_count >= 4:
        patterns.append(f"Consistently higher focus ratio ({focus_high_count}/{len(recent)} days)")
    if focus_low_count >= 4:
        patterns.append(f"Consistently lower focus ratio ({focus_low_count}/{len(recent)} days)")

    # Pattern: Rising fragmentation for >=4 consecutive days
    frag_values = [s.fragmentation_index for s in recent if s.fragmentation_index is not None]
    if len(frag_values) >= 4:
        # check for any run of 4 consecutive increasing days
        for i in range(len(frag_values) - 3):
            window = frag_values[i:i+4]
            if all(window[j] < window[j+1] for j in range(3)):
                patterns.append(f"Fragmentation increasing for 4 consecutive days (recent)")
                break

    return patterns


def format_weekly_report(daily_summaries: List, target_date: date) -> str:
    """
    Produce a weekly report for target_date summarizing baselines, stable patterns,
    and notable deviations for target_date.
    """
    baseline = compute_7day_baseline(daily_summaries, target_date)
    # find today's summary
    today = next((s for s in daily_summaries if s.date == target_date), None)

    lines = []
    lines.append("="*75)
    lines.append(f"WEEKLY SUMMARY — week ending {target_date}")
    lines.append("="*75)

    lines.append("\n[BASELINES]")
    if baseline["count"] == 0:
        lines.append("  Not enough data to compute 7-day baseline")
    else:
        def _fmt_time(seconds):
            if seconds is None:
                return "n/a"
            h = seconds / 3600
            return f"{h:.1f}h"
        lines.append(f"  Avg Active Time: {_fmt_time(baseline['active_time_mean'])} ± {_fmt_time(baseline['active_time_std'])}")
        lines.append(f"  Avg Session Duration: {baseline['avg_session_mean'] / 60:.0f}m ± {baseline['avg_session_std'] / 60:.0f}m" if baseline['avg_session_mean'] is not None else "  Avg Session Duration: n/a")
        lines.append(f"  Avg Focus Ratio: {baseline['focus_ratio_mean']:.0%} ± {baseline['focus_ratio_std']:.0%}" if baseline['focus_ratio_mean'] is not None else "  Avg Focus Ratio: n/a")
        lines.append(f"  Avg Fragmentation: {baseline['fragmentation_mean']:.2f} ± {baseline['fragmentation_std']:.2f}")

    lines.append("\n[CONSISTENT PATTERNS]")
    patterns = identify_stable_patterns(daily_summaries, target_date)
    if patterns:
        for p in patterns:
            lines.append(f"  • {p}")
    else:
        lines.append("  None detected")

    lines.append("\n[NOTABLE DEVIATIONS]")
    if today is None or baseline["count"] == 0:
        lines.append("  Not available (insufficient history)")
    else:
        deviations = detect_deviations(today, baseline)
        if deviations:
            for d in deviations:
                lines.append(f"  • {d}")
        else:
            lines.append("  None")

    lines.append("\n[STABILITY]")
    # volatility classification for the 7-day window
    # compute CVs
    def _cv_label(values):
        if not values or len(values) <= 1:
            return "unknown"
        m = mean(values)
        if m == 0:
            return "unknown"
        cv = (stdev(values) / m) * 100
        return _classify_volatility_cv(cv)

    if baseline['count'] > 0:
        active_vals = [s.total_active_seconds for s in daily_summaries if (target_date - s.date).days < 7 and (target_date - s.date).days >= 0]
        avg_session_vals = [s.avg_session_duration_seconds for s in daily_summaries if (target_date - s.date).days < 7 and (target_date - s.date).days >= 0]
        focus_vals = [s.focus_ratio for s in daily_summaries if (target_date - s.date).days < 7 and (target_date - s.date).days >= 0 and s.focus_ratio is not None]

        lines.append(f"  Focus Ratio: {_cv_label(focus_vals)}")
        lines.append(f"  Session Duration: {_cv_label(avg_session_vals)}")
        lines.append(f"  Active Time: {_cv_label(active_vals)}")
    else:
        lines.append("  Not available (insufficient history)")

    lines.append("\n" + "="*75)
    return "\n".join(lines)
