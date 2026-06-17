"""
Test classification on the most recent session.
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter


def get_db_path() -> Path:
    return Path(__file__).parent.parent / "storage" / "events.db"


def test_last_session():
    """Test classification on segments from the most recent session."""
    from agent.task.feature_extraction import FeatureExtractor
    from agent.task.core_tasks import get_task_recommendation
    
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get the last session
    cursor.execute("""
        SELECT session_id, start_time, end_time
        FROM sessions
        ORDER BY start_time DESC
        LIMIT 1
    """)
    
    session_row = cursor.fetchone()
    if not session_row:
        print("No sessions found in database.")
        return
    
    session_id = session_row["session_id"]
    session_start = session_row["start_time"]
    session_end = session_row["end_time"] or "(active)"
    
    print(f"\n{'='*80}")
    print(f"Testing Classification on Last Session")
    print(f"{'='*80}")
    print(f"Session ID: {session_id}")
    print(f"Start: {session_start}")
    print(f"End: {session_end}")
    print(f"{'='*80}\n")
    
    # Get segments for this session
    cursor.execute("""
        SELECT id, session_id, task_id, start_time, end_time, confidence, reason
        FROM task_segments
        WHERE session_id = ?
        ORDER BY start_time
    """, (session_id,))
    
    segments = [dict(row) for row in cursor.fetchall()]
    
    if not segments:
        print(f"No segments found for session {session_id}.")
        return
    
    print(f"Found {len(segments)} segments in this session\n")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(db_path)
    
    # Classify each segment
    stats = {
        "total": len(segments),
        "changed": 0,
        "unknown_before": 0,
        "unknown_after": 0
    }
    
    old_distribution = Counter()
    new_distribution = Counter()
    reason_distribution = Counter()
    
    for i, segment in enumerate(segments, 1):
        # Reconstruct segment context from events
        cursor.execute("""
            SELECT event_type, payload, timestamp
            FROM events
            WHERE session_id = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
        """, (segment["session_id"], segment["start_time"], segment["end_time"]))
        
        events = cursor.fetchall()
        
        # Parse events to find app and window
        apps = Counter()
        windows = Counter()
        input_count = 0
        
        for event_type, payload_str, ts in events:
            try:
                payload = json.loads(payload_str) if payload_str else {}
            except:
                payload = {}
            
            if event_type == "ACTIVE_WINDOW":
                app = payload.get("app", "unknown")
                window = payload.get("title", "")
                apps[app] += 1
                if window:
                    windows[window] += 1
            elif event_type in ("INPUT", "KEYBOARD", "MOUSE"):
                input_count += 1
        
        # Get dominant app and window
        dominant_app = apps.most_common(1)[0][0] if apps else "unknown"
        dominant_window = windows.most_common(1)[0][0] if windows else ""
        
        # Create segment context
        from datetime import datetime, timezone
        start_dt = datetime.fromisoformat(segment["start_time"])
        if segment["end_time"]:
            end_dt = datetime.fromisoformat(segment["end_time"])
        else:
            end_dt = datetime.now(timezone.utc)
        
        # Ensure both are timezone-aware
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        duration_seconds = (end_dt - start_dt).total_seconds()
        
        segment_context = {
            "app": dominant_app,
            "window_title": dominant_window,
            "normalized_title": dominant_window,
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration_seconds": duration_seconds,
            "input_count": input_count
        }
        
        # Extract features and classify
        features = extractor.extract_features(segment_context)
        new_task_id, new_confidence, new_reason = get_task_recommendation(features)
        
        old_task_id = segment["task_id"]
        old_distribution[old_task_id] += 1
        new_distribution[new_task_id] += 1
        reason_distribution[new_reason] += 1
        
        if "unknown" in old_task_id.lower():
            stats["unknown_before"] += 1
        if new_task_id == "unknown":
            stats["unknown_after"] += 1
        
        changed = old_task_id != new_task_id
        if changed:
            stats["changed"] += 1
        
        # Print each segment
        change_marker = " → CHANGED" if changed else ""
        unknown_marker = " ⚠ UNKNOWN" if new_task_id == "unknown" else ""
        
        print(f"Segment {i:2d} (ID={segment['id']:4d}){change_marker}{unknown_marker}")
        print(f"  Duration: {duration_seconds/60:.1f} min")
        print(f"  App: {dominant_app[:40]:<40} Events: {len(events)}")
        print(f"  Window: {dominant_window[:50] if dominant_window else '(none)'}")
        print(f"  Old: {old_task_id:<30} (conf={segment.get('confidence', 0):.2f})")
        print(f"  New: {new_task_id:<30} (conf={new_confidence:.2f}, {new_reason})")
        print()
    
    conn.close()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total segments: {stats['total']}")
    print(f"Changed: {stats['changed']} ({stats['changed']/stats['total']*100:.1f}%)")
    print(f"Unknown before: {stats['unknown_before']}")
    print(f"Unknown after: {stats['unknown_after']}")
    
    if stats['unknown_after'] == 0:
        print(f"\n{'✓'*40}")
        print("SUCCESS! Zero unknown classifications!")
        print(f"{'✓'*40}\n")
    else:
        print(f"\n⚠ Warning: {stats['unknown_after']} segments still 'unknown'\n")
    
    print("\nOld Task Distribution:")
    for task, count in old_distribution.most_common():
        print(f"  {task:35s}: {count:2d} ({count/stats['total']*100:.1f}%)")
    
    print("\nNew Task Distribution:")
    for task, count in new_distribution.most_common():
        print(f"  {task:35s}: {count:2d} ({count/stats['total']*100:.1f}%)")
    
    print("\nClassification Methods:")
    for reason, count in reason_distribution.most_common():
        print(f"  {reason:35s}: {count:2d} ({count/stats['total']*100:.1f}%)")


if __name__ == "__main__":
    test_last_session()
