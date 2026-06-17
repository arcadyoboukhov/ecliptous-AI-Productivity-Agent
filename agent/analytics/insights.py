"""
Analytics & Insights Layer (Step 5)

Generates analytics from enriched intervals, segments, and tasks:
- Productivity & efficiency metrics
- Focus periods
- Context switching frequency
- Peak productivity hours
- Interaction analysis
- System resource analytics
- Audio/video context insights
- Visualization-ready data (heatmaps, timelines, series)
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import math

from agent.storage.db import get_intervals
from agent.task.preprocessing import PreprocessingPipeline
from agent.task.clustering import TaskRecognitionPipeline
from agent.task.enriched_storage import persist_enriched_tasks
from agent.storage.db import save_analytics_snapshot


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clamp(value: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, value))


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = _safe_mean(x)
    mean_y = _safe_mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _interval_duration_minutes(interval: Dict) -> float:
    start = _parse_ts(interval['timestamp_start'])
    end = _parse_ts(interval['timestamp_end'])
    return max(0.001, (end - start).total_seconds() / 60.0)


def _compute_input_score(interval: Dict) -> float:
    kb = interval.get('keyboard_intensity', 0.0)  # keys/min
    clicks = interval.get('mouse_clicks', 0)
    duration_min = _interval_duration_minutes(interval)
    clicks_per_min = clicks / duration_min if duration_min > 0 else 0.0

    kb_norm = min(kb, 60.0) / 60.0
    mouse_norm = min(clicks_per_min, 10.0) / 10.0
    return _clamp((kb_norm + mouse_norm) / 2.0)


def _compute_system_score(interval: Dict) -> float:
    cpu = interval.get('cpu_usage', 0.0) or 0.0
    ram = interval.get('ram_usage', 0.0) or 0.0
    gpu = interval.get('gpu_usage')
    values = [cpu, ram]
    if gpu is not None:
        values.append(gpu)
    return _clamp(_safe_mean(values))


def _compute_audio_context_score(interval: Dict) -> float:
    mic = 1.0 if interval.get('mic_active') else 0.0
    cam = 1.0 if interval.get('camera_active') else 0.0
    vol = interval.get('audio_volume', 0.0) or 0.0

    distraction = _clamp(0.4 * mic + 0.4 * cam + 0.2 * min(vol, 1.0))
    return _clamp(1.0 - distraction)


def _compute_focus_score(interval: Dict) -> float:
    input_score = _compute_input_score(interval)
    system_score = _compute_system_score(interval)
    audio_score = _compute_audio_context_score(interval)
    return _clamp(0.5 * input_score + 0.3 * system_score + 0.2 * audio_score)


def _bucket_hour(dt: datetime) -> int:
    return dt.astimezone().hour


def _ensure_sorted(intervals: List[Dict]) -> List[Dict]:
    return sorted(intervals, key=lambda i: i['timestamp_start'])


@dataclass
class FocusPeriod:
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    avg_focus_score: float
    primary_app: Optional[str]
    apps_used: List[str]

    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_minutes': self.duration_minutes,
            'avg_focus_score': self.avg_focus_score,
            'primary_app': self.primary_app,
            'apps_used': self.apps_used,
        }


def detect_focus_periods(
    intervals: List[Dict],
    focus_threshold: float = 0.6,
    max_gap_minutes: float = 1.5,
) -> List[FocusPeriod]:
    """Detect focus periods based on focus score and continuity."""
    if not intervals:
        return []

    intervals = _ensure_sorted(intervals)
    periods = []
    current = []

    for interval in intervals:
        score = _compute_focus_score(interval)
        start = _parse_ts(interval['timestamp_start'])
        end = _parse_ts(interval['timestamp_end'])

        if score >= focus_threshold:
            if not current:
                current.append(interval)
            else:
                last_end = _parse_ts(current[-1]['timestamp_end'])
                gap = (start - last_end).total_seconds() / 60.0
                if gap <= max_gap_minutes:
                    current.append(interval)
                else:
                    period = _build_focus_period(current)
                    if period:
                        periods.append(period)
                    current = [interval]
        else:
            if current:
                period = _build_focus_period(current)
                if period:
                    periods.append(period)
                current = []

    if current:
        period = _build_focus_period(current)
        if period:
            periods.append(period)

    return periods


def _build_focus_period(intervals: List[Dict]) -> Optional[FocusPeriod]:
    if not intervals:
        return None

    start = _parse_ts(intervals[0]['timestamp_start'])
    end = _parse_ts(intervals[-1]['timestamp_end'])
    duration = (end - start).total_seconds() / 60.0

    scores = [_compute_focus_score(i) for i in intervals]
    app_counts = Counter(i.get('app') for i in intervals if i.get('app'))
    primary_app = app_counts.most_common(1)[0][0] if app_counts else None

    return FocusPeriod(
        start_time=start,
        end_time=end,
        duration_minutes=duration,
        avg_focus_score=_safe_mean(scores),
        primary_app=primary_app,
        apps_used=[a for a, _ in app_counts.most_common()],
    )


def _group_by_hour(intervals: List[Dict]) -> Dict[int, List[Dict]]:
    buckets = defaultdict(list)
    for interval in intervals:
        ts = _parse_ts(interval['timestamp_start'])
        hour = _bucket_hour(ts)
        buckets[hour].append(interval)
    return buckets


def _compute_context_switches(intervals: List[Dict]) -> Tuple[int, Dict[int, int]]:
    if not intervals:
        return 0, {}
    intervals = _ensure_sorted(intervals)
    switches = 0
    per_hour = defaultdict(int)
    last_app = None
    for interval in intervals:
        app = interval.get('app')
        if last_app is not None and app and app != last_app:
            switches += 1
            hour = _bucket_hour(_parse_ts(interval['timestamp_start']))
            per_hour[hour] += 1
        if app:
            last_app = app
    return switches, dict(per_hour)


def _compute_peak_hours(intervals: List[Dict]) -> List[Dict]:
    hourly = _group_by_hour(intervals)
    hour_scores = []
    for hour, items in hourly.items():
        scores = [_compute_focus_score(i) for i in items]
        hour_scores.append({
            'hour': hour,
            'avg_focus_score': _safe_mean(scores),
            'interval_count': len(items),
        })
    hour_scores.sort(key=lambda h: h['avg_focus_score'], reverse=True)
    return hour_scores


def _compute_interaction_trends(intervals: List[Dict]) -> List[Dict]:
    hourly = _group_by_hour(intervals)
    trend = []
    for hour in range(24):
        items = hourly.get(hour, [])
        if not items:
            trend.append({
                'hour': hour,
                'keys_per_min': 0.0,
                'clicks_per_min': 0.0,
                'copy_paste_density': 0.0,
            })
            continue

        total_duration = sum(_interval_duration_minutes(i) for i in items)
        total_keys = sum(int(i.get('keyboard_intensity', 0.0) * _interval_duration_minutes(i)) for i in items)
        total_clicks = sum(i.get('mouse_clicks', 0) for i in items)
        total_copy_paste = sum(i.get('copy_count', 0) + i.get('paste_count', 0) for i in items)

        keys_per_min = total_keys / total_duration if total_duration else 0.0
        clicks_per_min = total_clicks / total_duration if total_duration else 0.0
        copy_paste_density = total_copy_paste / total_duration if total_duration else 0.0

        trend.append({
            'hour': hour,
            'keys_per_min': keys_per_min,
            'clicks_per_min': clicks_per_min,
            'copy_paste_density': copy_paste_density,
        })

    return trend


def _compute_audio_video_series(intervals: List[Dict]) -> List[Dict]:
    hourly = _group_by_hour(intervals)
    series = []
    for hour in range(24):
        items = hourly.get(hour, [])
        if not items:
            series.append({
                'hour': hour,
                'mic_ratio': 0.0,
                'camera_ratio': 0.0,
                'audio_avg': 0.0,
            })
            continue
        mic_ratio = _safe_mean([1.0 if i.get('mic_active') else 0.0 for i in items])
        camera_ratio = _safe_mean([1.0 if i.get('camera_active') else 0.0 for i in items])
        audio_avg = _safe_mean([i.get('audio_volume', 0.0) or 0.0 for i in items])
        series.append({
            'hour': hour,
            'mic_ratio': mic_ratio,
            'camera_ratio': camera_ratio,
            'audio_avg': audio_avg,
        })
    return series


def _compute_system_series(intervals: List[Dict]) -> List[Dict]:
    hourly = _group_by_hour(intervals)
    series = []
    for hour in range(24):
        items = hourly.get(hour, [])
        if not items:
            series.append({
                'hour': hour,
                'cpu_avg': 0.0,
                'ram_avg': 0.0,
                'gpu_avg': 0.0,
            })
            continue
        cpu_avg = _safe_mean([i.get('cpu_usage', 0.0) or 0.0 for i in items])
        ram_avg = _safe_mean([i.get('ram_usage', 0.0) or 0.0 for i in items])
        gpu_values = [i.get('gpu_usage') for i in items if i.get('gpu_usage') is not None]
        gpu_avg = _safe_mean(gpu_values) if gpu_values else 0.0
        series.append({
            'hour': hour,
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
        })
    return series


def _compute_heatmap(intervals: List[Dict]) -> List[Dict]:
    hourly = _group_by_hour(intervals)
    heatmap = []
    for hour in range(24):
        items = hourly.get(hour, [])
        if not items:
            heatmap.append({
                'hour': hour,
                'focus_avg': 0.0,
                'input_avg': 0.0,
                'system_avg': 0.0,
                'audio_avg': 0.0,
            })
            continue
        focus_avg = _safe_mean([_compute_focus_score(i) for i in items])
        input_avg = _safe_mean([_compute_input_score(i) for i in items])
        system_avg = _safe_mean([_compute_system_score(i) for i in items])
        audio_avg = _safe_mean([i.get('audio_volume', 0.0) or 0.0 for i in items])
        heatmap.append({
            'hour': hour,
            'focus_avg': focus_avg,
            'input_avg': input_avg,
            'system_avg': system_avg,
            'audio_avg': audio_avg,
        })
    return heatmap


def _compute_audio_engagement(intervals: List[Dict], threshold: float) -> Dict:
    engaged = [i for i in intervals if (i.get('audio_volume', 0.0) or 0.0) >= threshold]
    total_minutes = sum(_interval_duration_minutes(i) for i in intervals)
    engaged_minutes = sum(_interval_duration_minutes(i) for i in engaged)
    return {
        'threshold': threshold,
        'engaged_minutes': engaged_minutes,
        'engaged_ratio': engaged_minutes / total_minutes if total_minutes else 0.0,
    }


def _compute_meeting_periods(segments: List) -> List[Dict]:
    meetings = []
    for seg in segments:
        if seg.mic_active_ratio >= 0.6 or seg.camera_active_ratio >= 0.4:
            meetings.append({
                'segment_id': seg.segment_id,
                'start_time': seg.start_time.isoformat() if seg.start_time else None,
                'end_time': seg.end_time.isoformat() if seg.end_time else None,
                'duration_minutes': seg.duration_minutes,
                'primary_app': seg.primary_app,
                'mic_ratio': seg.mic_active_ratio,
                'camera_ratio': seg.camera_active_ratio,
            })
    return meetings


def _compute_system_by_session(intervals: List[Dict]) -> List[Dict]:
    sessions = defaultdict(list)
    for interval in intervals:
        sid = interval.get('session_id') or 'unassigned'
        sessions[sid].append(interval)

    results = []
    for sid, items in sessions.items():
        cpu_avg = _safe_mean([i.get('cpu_usage', 0.0) or 0.0 for i in items])
        ram_avg = _safe_mean([i.get('ram_usage', 0.0) or 0.0 for i in items])
        gpu_values = [i.get('gpu_usage') for i in items if i.get('gpu_usage') is not None]
        gpu_avg = _safe_mean(gpu_values) if gpu_values else None
        results.append({
            'session_id': sid,
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
            'interval_count': len(items),
        })
    return results


def _compute_task_resource_usage(tasks: List) -> List[Dict]:
    results = []
    for task in tasks:
        if not task.segments:
            continue
        cpu_avg = _safe_mean([s.mean_cpu_usage for s in task.segments])
        ram_avg = _safe_mean([s.mean_ram_usage for s in task.segments])
        gpu_values = [s.mean_gpu_usage for s in task.segments if s.mean_gpu_usage is not None]
        gpu_avg = _safe_mean(gpu_values) if gpu_values else None
        results.append({
            'task_id': task.task_id,
            'task_name': task.task_name,
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
            'duration_minutes': task.total_duration_minutes,
        })
    return results


def _detect_heavy_computation(tasks: List, segments: List) -> Dict:
    heavy_tasks = [
        t for t in tasks
        if (t.avg_cpu_usage >= 0.7) or (t.avg_ram_usage >= 0.8)
    ]
    heavy_segments = [
        s for s in segments
        if (s.mean_cpu_usage >= 0.7) or (s.mean_ram_usage >= 0.8) or ((s.mean_gpu_usage or 0.0) >= 0.6)
    ]
    return {
        'task_count': len(heavy_tasks),
        'segment_count': len(heavy_segments),
        'tasks': [
            {
                'task_id': t.task_id,
                'task_name': t.task_name,
                'cpu_avg': t.avg_cpu_usage,
                'ram_avg': t.avg_ram_usage,
            } for t in heavy_tasks
        ],
        'segments': [
            {
                'segment_id': s.segment_id,
                'primary_app': s.primary_app,
                'cpu_avg': s.mean_cpu_usage,
                'ram_avg': s.mean_ram_usage,
                'gpu_avg': s.mean_gpu_usage,
            } for s in heavy_segments
        ]
    }


def generate_insights_report(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 5000,
    clustering_method: str = 'dbscan',
    focus_threshold: float = 0.6,
    audio_engagement_threshold: float = 0.5,
    persist_enriched: bool = True,
    persist_snapshot: bool = True,
) -> Dict:
    """Generate analytics and insights over interval data."""
    intervals = get_intervals(start_time=start_time, end_time=end_time, limit=limit)

    if not intervals:
        return {
            'version': '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'data': {
                'message': 'No intervals found for the requested range.'
            }
        }

    intervals = _ensure_sorted(intervals)

    # Preprocess to segments
    preprocessing = PreprocessingPipeline(min_segment_minutes=10.0)
    preprocessing.fit(intervals)
    segments = preprocessing.process(intervals)

    # Cluster to tasks
    recognition = TaskRecognitionPipeline(clustering_method=clustering_method)
    tasks = recognition.recognize_tasks(segments)
    task_summary = recognition.get_task_summary(tasks)

    stored_task_ids = []
    if persist_enriched and tasks:
        try:
            stored_task_ids = persist_enriched_tasks(tasks)
        except Exception:
            stored_task_ids = []

    # Productivity metrics
    focus_periods = detect_focus_periods(intervals, focus_threshold=focus_threshold)
    total_active_minutes = sum(_interval_duration_minutes(i) for i in intervals)
    focus_minutes = sum(p.duration_minutes for p in focus_periods)
    focus_ratio = focus_minutes / total_active_minutes if total_active_minutes else 0.0
    avg_focus_score = _safe_mean([_compute_focus_score(i) for i in intervals])

    # Context switching
    total_switches, switches_per_hour = _compute_context_switches(intervals)

    # Peak hours
    peak_hours = _compute_peak_hours(intervals)

    # Interaction analysis
    interaction_trends = _compute_interaction_trends(intervals)
    total_copy_paste = sum(i.get('copy_count', 0) + i.get('paste_count', 0) for i in intervals)
    copy_paste_density = total_copy_paste / total_active_minutes if total_active_minutes else 0.0

    # Correlations
    focus_scores = [_compute_focus_score(i) for i in intervals]
    mic_series = [1.0 if i.get('mic_active') else 0.0 for i in intervals]
    cam_series = [1.0 if i.get('camera_active') else 0.0 for i in intervals]
    audio_series = [i.get('audio_volume', 0.0) or 0.0 for i in intervals]

    mic_focus_corr = _pearson_corr(mic_series, focus_scores)
    cam_focus_corr = _pearson_corr(cam_series, focus_scores)
    audio_focus_corr = _pearson_corr(audio_series, focus_scores)

    # System analytics
    system_by_session = _compute_system_by_session(intervals)
    system_by_task = _compute_task_resource_usage(tasks)
    heavy_compute = _detect_heavy_computation(tasks, segments)

    # Audio/video insights
    meeting_periods = _compute_meeting_periods(segments)
    audio_engagement = _compute_audio_engagement(intervals, threshold=audio_engagement_threshold)

    # Visualization data
    heatmap = _compute_heatmap(intervals)
    system_series = _compute_system_series(intervals)
    audio_video_series = _compute_audio_video_series(intervals)

    task_timeline = [
        {
            'task_id': t.task_id,
            'task_name': t.task_name,
            'start_time': t.start_time.isoformat() if t.start_time else None,
            'end_time': t.end_time.isoformat() if t.end_time else None,
            'duration_minutes': t.total_duration_minutes,
            'primary_app': t.primary_app,
            'window_pattern': t.primary_window_pattern,
            'avg_activity_score': t.avg_activity_score,
            'confidence': t.cluster_confidence,
        }
        for t in tasks
    ]

    report = {
        'version': '1.0',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'data': {
            'range': {
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None,
                'interval_count': len(intervals),
                'segment_count': len(segments),
                'task_count': len(tasks),
            },
            'productivity': {
                'focus_minutes': focus_minutes,
                'active_minutes': total_active_minutes,
                'focus_ratio': focus_ratio,
                'avg_focus_score': avg_focus_score,
                'focus_periods': [p.to_dict() for p in focus_periods],
                'peak_hours': peak_hours,
            },
            'context_switching': {
                'total_switches': total_switches,
                'switches_per_hour': switches_per_hour,
            },
            'interaction': {
                'copy_paste_density': copy_paste_density,
                'total_copy_paste_events': total_copy_paste,
                'trends_by_hour': interaction_trends,
            },
            'correlations': {
                'mic_focus_correlation': mic_focus_corr,
                'camera_focus_correlation': cam_focus_corr,
                'audio_focus_correlation': audio_focus_corr,
            },
            'system_resources': {
                'by_session': system_by_session,
                'by_task': system_by_task,
                'heavy_computation': heavy_compute,
            },
            'audio_video': {
                'meeting_periods': meeting_periods,
                'audio_engagement': audio_engagement,
                'series_by_hour': audio_video_series,
            },
            'tasks': {
                'summary': task_summary,
                'timeline': task_timeline,
            },
            'storage': {
                'enriched_tasks_saved': len(stored_task_ids),
                'enriched_task_ids': stored_task_ids,
            },
            'visualization': {
                'heatmap': heatmap,
                'system_series': system_series,
                'interaction_series': interaction_trends,
                'audio_video_series': audio_video_series,
                'task_timeline': task_timeline,
            }
        }
    }

    if persist_snapshot:
        try:
            save_analytics_snapshot(report, start_time=start_time, end_time=end_time)
        except Exception:
            pass

    return report
