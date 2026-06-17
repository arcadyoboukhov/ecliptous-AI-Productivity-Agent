"""
Session persistence layer.

Handles saving and loading sessions from disk in JSON format.
Provides a simple, versioned approach without database complexity.
"""

import json
import os
import time
import errno
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import timedelta


# Configuration
SESSIONS_FILE = "sessions.json"
SESSIONS_BACKUP_FILE = "sessions.backup.json"


def _merge_overlapping_sessions(sessions: List) -> List:
    """
    Merge sessions that overlap or are contiguous so that there is never
    more than one session spanning the same time range.
    """
    if not sessions:
        return []

    # Sort sessions by start time
    try:
        sessions_sorted = sorted(sessions, key=lambda s: s.start)
    except Exception:
        return sessions

    merged = []
    cur = sessions_sorted[0]

    for nxt in sessions_sorted[1:]:
        # If next starts before or at current end, merge them
        if nxt.start <= cur.end + timedelta(seconds=0):
            # extend end
            cur.end = max(cur.end, nxt.end)
            # merge apps
            try:
                cur.apps.update(getattr(nxt, "apps", set()))
            except Exception:
                pass
            # sum event counts
            try:
                cur.event_count += getattr(nxt, "event_count", 0)
            except Exception:
                pass
            # merge input_events
            try:
                for k in ("keys", "clicks", "mouse_distance"):
                    cur.input_events[k] = cur.input_events.get(k, 0) + nxt.input_events.get(k, 0)
            except Exception:
                pass
            # merge timeline by minute bucket
            try:
                for bucket, counts in getattr(nxt, "timeline", {}).items():
                    if bucket in cur.timeline:
                        for kk, vv in counts.items():
                            cur.timeline[bucket][kk] = cur.timeline[bucket].get(kk, 0) + vv
                    else:
                        cur.timeline[bucket] = counts.copy()
            except Exception:
                pass
            # merge intent breakdown
            try:
                for ik, iv in getattr(nxt, "intent_breakdown", {}).items():
                    cur.intent_breakdown[ik] = cur.intent_breakdown.get(ik, 0) + iv
            except Exception:
                pass
            # concatenate intent segments
            try:
                cur.intent_segments = getattr(cur, "intent_segments", []) + getattr(nxt, "intent_segments", [])
            except Exception:
                pass
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged


def session_to_dict(session) -> Dict[str, Any]:
    """Convert a Session object to a JSON-serializable dictionary.
    
    Enhanced format: includes duration and feature vector for ML/analytics.
    """
    # Extract timeline from SignalBuffer if available, fallback to direct timeline
    timeline_data = {}
    if hasattr(session, "signals") and session.signals:
        # Use activity_timeline from signal buffer (minute-bucketed)
        try:
            timeline_data = session.signals.activity_timeline.copy() if hasattr(session.signals, 'activity_timeline') else {}
        except Exception:
            pass
    
    # Fallback to legacy timeline field if signals didn't provide data
    if not timeline_data and hasattr(session, "timeline") and session.timeline:
        timeline_data = session.timeline
    
    # Update end time based on timeline data (most recent bucket) or current time if in progress
    from datetime import datetime, timezone
    if timeline_data:
        # Get the most recent timeline bucket as the effective end time
        bucket_times = [datetime.fromisoformat(b) for b in timeline_data.keys()]
        session.end = max(bucket_times)
    elif getattr(session, "in_progress", False):
        # No timeline yet but session is in progress - use current time
        session.end = datetime.now(timezone.utc)
    
    duration_seconds = (session.end - session.start).total_seconds()
    
    result = {
        "id": getattr(session, "id", None),
        "session_id": getattr(session, "id", None),  # Backward compatibility
        "start": session.start.isoformat(),
        "end": session.end.isoformat(),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
        "device_id": getattr(session, "device_id", "unknown"),
        "in_progress": getattr(session, "in_progress", False),
    }
    
    # Include ML inference results if present
    if hasattr(session, "inferred_task_id") and session.inferred_task_id:
        result["inferred_task_id"] = session.inferred_task_id
    
    # Include activity metrics if present
    if hasattr(session, "input_events"):
        result["input_events"] = session.input_events
    
    if hasattr(session, "apps") and session.apps:
        result["apps"] = list(session.apps)
    
    if hasattr(session, "event_count"):
        result["event_count"] = session.event_count
    
    # Only include timeline if we have data
    if timeline_data:
        result["timeline"] = timeline_data
    
    # Include intent breakdown/segments if present (legacy fields)
    if hasattr(session, "intent_breakdown") and session.intent_breakdown:
        result["intent_breakdown"] = session.intent_breakdown
    
    if hasattr(session, "intent_segments") and session.intent_segments:
        result["intent_segments"] = session.intent_segments
    
    # Include online task classification data if present
    if hasattr(session, "current_task_assignment") and session.current_task_assignment:
        result["current_task_assignment"] = session.current_task_assignment
    
    if hasattr(session, "task_classification_history") and session.task_classification_history:
        result["task_classification_history"] = session.task_classification_history
    
    if hasattr(session, "intra_session_tasks") and session.intra_session_tasks:
        # Convert TaskSegment objects to dicts for JSON serialization
        segments_as_dicts = []
        for seg in session.intra_session_tasks:
            if hasattr(seg, 'task_id'):  # Check if it's a TaskSegment object
                seg_dict = {
                    "task_id": seg.task_id,
                    "start_time": seg.start_time.isoformat() if seg.start_time else None,
                    "end_time": seg.end_time.isoformat() if seg.end_time else None,
                    "confidence": seg.confidence,
                    "feature_vector": seg.feature_vector,
                    "reason": seg.reason,
                    "distance_to_centroid": seg.distance_to_centroid
                }
                segments_as_dicts.append(seg_dict)
            else:
                # Already a dict
                segments_as_dicts.append(seg)
        result["intra_session_tasks"] = segments_as_dicts
    
    # Include feature vector if computed
    if hasattr(session, "rolling_features") and session.rolling_features:
        result["features"] = session.rolling_features
    
    return result


def dict_to_session(data: Dict[str, Any]):
    """Convert a dictionary back to a Session object.
    
    Backward compatible: loads old full-featured session data,
    but only restores core fields (id, start, end, device_id).
    Old fields (apps, timeline, intent_segments) are discarded.
    """
    # Import here to avoid circular imports
    from agent.session.sessionizer import Session
    from agent.task.online_classification import TaskSegment
    
    session = Session(
        start_time=datetime.fromisoformat(data["start"]),
        session_id=data.get("id"),
        device_id=data.get("device_id"),
    )
    session.end = datetime.fromisoformat(data["end"])
    # restore in_progress flag if present
    try:
        session.in_progress = bool(data.get("in_progress", False))
    except Exception:
        session.in_progress = False
    
    # Restore ML task tracking fields
    session.intra_session_tasks = []
    if "intra_session_tasks" in data and data["intra_session_tasks"]:
        for task_data in data["intra_session_tasks"]:
            try:
                task_seg = TaskSegment(
                    task_id=task_data.get("task_id"),
                    start_time=datetime.fromisoformat(task_data["start_time"]) if task_data.get("start_time") else None,
                    end_time=datetime.fromisoformat(task_data["end_time"]) if task_data.get("end_time") else None,
                    confidence=task_data.get("confidence", 0.0),
                    feature_vector=task_data.get("feature_vector"),
                    reason=task_data.get("reason", ""),
                    distance_to_centroid=task_data.get("distance_to_centroid")
                )
                session.intra_session_tasks.append(task_seg)
            except Exception as e:
                print(f"Warning: Could not restore task segment: {e}")
    
    session.current_task_assignment = data.get("current_task_assignment")
    session.task_classification_history = data.get("task_classification_history", [])
    
    # Old fields are intentionally NOT restored
    # (apps, event_count, input_events, timeline, intent_breakdown, intent_segments)
    # These now live in Task and SignalWindow objects
    
    return session


def _split_long_sessions(sessions: List) -> List:
    # The concept of splitting long sessions was removed — keep original sessions unchanged.
    return sessions


# Simple cross-process lock & per-process cache to avoid races and spurious reloads
_LOCK_RETRY = 0.05
_LOCK_TIMEOUT = 5.0  # Increased from 2.0 to 5.0 for more patience with I/O
_CACHE = None
_CACHE_MTIME = None


def _lock_path(filepath: str) -> str:
    return f"{filepath}.lock"


def _acquire_lock(filepath: str, timeout: float = _LOCK_TIMEOUT, retry: float = _LOCK_RETRY) -> None:
    lock = _lock_path(filepath)
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return
        except OSError as e:
            if getattr(e, 'errno', None) not in (errno.EEXIST, errno.EACCES):
                raise
            # If lock file appears stale (older than timeout + small slack), remove it
            try:
                if os.path.exists(lock):
                    m = os.path.getmtime(lock)
                    if time.time() - m > (timeout + 1.0):
                        try:
                            os.remove(lock)
                        except Exception:
                            pass
            except Exception:
                pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout acquiring lock for {filepath}")
        time.sleep(retry)


def _release_lock(filepath: str) -> None:
    try:
        os.remove(_lock_path(filepath))
    except FileNotFoundError:
        pass


def save_sessions(sessions: List, filepath: str = SESSIONS_FILE) -> None:
    """
    (DISABLED) Session persistence to JSON is disabled.
    This is a no-op function for backwards compatibility.
    """
    pass


def load_sessions(filepath: str = SESSIONS_FILE) -> List:
    """
    Load sessions from the SQLite database only (NO JSON).
    Uses DB-only storage for all session persistence.
    
    Args:
        filepath: Ignored (kept for API compatibility, all data in DB)
    
    Returns:
        List of Session objects from database
    """
    from agent.storage.db import load_sessions_from_db
    return load_sessions_from_db(days_back=90)  # Default 90 days of sessions


def append_session(session, filepath: str = SESSIONS_FILE) -> None:
    """
    Append a single session to the sessions file with proper locking.
    """
    global _CACHE, _CACHE_MTIME
    acquired = False
def append_session(session, filepath: str = SESSIONS_FILE) -> None:
    """
    Append a single session to the database (NO JSON).
    All session persistence is now database-only.
    """
    try:
        from agent.storage.db import save_session
        save_session(session)
    except Exception as e:
        print(f"Error appending session to DB: {e}")
        raise


def upsert_session(session, filepath: str = SESSIONS_FILE) -> None:
    """
    Insert or update a session record in the database (NO JSON).
    Uses database upsert semantics.
    """
    try:
        from agent.storage.db import save_session
        save_session(session)
    except Exception as e:
        print(f"Error upserting session to DB: {e}")
        raise


# ============================================================================
# Behavioral Model Persistence
# ============================================================================

BEHAVIORAL_MODEL_FILE = "behavioral_model.json"


def save_behavioral_model_state(data: Dict) -> bool:
    """
    Save behavioral model state to disk.
    
    Args:
        data: Serialized behavioral model data
        
    Returns:
        True if successful, False otherwise
    """
    # JSON file saving disabled
    return True


def load_behavioral_model_state() -> Optional[Dict]:
    """
    Load behavioral model state from disk.
    
    Returns:
        Dict with model state, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(BEHAVIORAL_MODEL_FILE):
        return None
    
    try:
        with open(BEHAVIORAL_MODEL_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading behavioral model: {e}")
        return None
