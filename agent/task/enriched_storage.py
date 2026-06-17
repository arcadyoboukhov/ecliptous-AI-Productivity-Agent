"""
Enriched Task Storage Helpers

Builds normalized task records from clustered Task objects and persists them.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import Counter

from agent.storage.db import save_enriched_tasks


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _majority_bool(values: List[float], threshold: float = 0.5) -> bool:
    if not values:
        return False
    return _safe_mean(values) >= threshold


def build_enriched_task_record(task) -> Dict:
    """Build a normalized enriched task record from a Task object."""
    segments = getattr(task, 'segments', []) or []

    # Determine window title
    window_title = getattr(task, 'primary_window_pattern', None)
    if not window_title and segments:
        titles = [s.primary_window_title for s in segments if getattr(s, 'primary_window_title', None)]
        if titles:
            window_title = Counter(titles).most_common(1)[0][0]

    # Aggregate segment-based metrics
    avg_gpu = _safe_mean([s.mean_gpu_usage for s in segments if s.mean_gpu_usage is not None])
    audio_volume = _safe_mean([s.max_audio_volume for s in segments]) if segments else 0.0

    mic_on = _majority_bool([s.mic_active_ratio for s in segments])
    camera_on = _majority_bool([s.camera_active_ratio for s in segments])

    copy_count = sum(s.total_copy_events for s in segments) if segments else 0
    paste_count = sum(s.total_paste_events for s in segments) if segments else 0

    keyboard_intensity = _safe_mean([s.keyboard_intensity_avg for s in segments]) or 0.0
    mouse_activity = sum(s.total_mouse_clicks for s in segments) if segments else 0

    # Derive session_id (most common if available)
    session_id = None
    if segments:
        session_ids = [s.session_id for s in segments if s.session_id]
        if session_ids:
            session_id = Counter(session_ids).most_common(1)[0][0]

    record = {
        "task_id": getattr(task, 'task_id', None),
        "session_id": session_id,
        "task_name": getattr(task, 'task_name', 'Unknown Task'),
        "app": getattr(task, 'primary_app', None),
        "window_title": window_title,
        "avg_cpu": getattr(task, 'avg_cpu_usage', 0.0),
        "avg_ram": getattr(task, 'avg_ram_usage', 0.0),
        "avg_gpu": avg_gpu,
        "mic_on": mic_on,
        "camera_on": camera_on,
        "audio_volume": audio_volume,
        "copy_count": copy_count,
        "paste_count": paste_count,
        "keyboard_intensity": keyboard_intensity,
        "mouse_activity": mouse_activity,
        "start_time": task.start_time.isoformat() if getattr(task, 'start_time', None) else None,
        "end_time": task.end_time.isoformat() if getattr(task, 'end_time', None) else None,
        "duration_minutes": getattr(task, 'total_duration_minutes', 0.0),
    }

    return record


def persist_enriched_tasks(tasks: List) -> List[int]:
    """Persist enriched task records to storage and return row IDs."""
    records = [build_enriched_task_record(task) for task in tasks]
    return save_enriched_tasks(records)
