"""
Re-classify existing task segments using v2 classification system.

This updates segments with:
- New task_id based on expanded 23-category system
- Multi-layer classification (app → window → behavioral)
- Confidence and reason tracking
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter


def get_db_path() -> Path:
    """Get the path to the events database."""
    return Path(__file__).parent.parent / "storage" / "events.db"


def get_segment_context_from_events(segment_id: int, session_id: str, start_time: str, end_time: str, conn) -> dict:
    """
    Reconstruct segment context from events within the segment time range.
    Returns dict with app, window_title, and activity metrics.
    """
    cursor = conn.cursor()
    
    # Get all events in this segment's time range
    cursor.execute("""
        SELECT event_type, payload, timestamp
        FROM events
        WHERE session_id = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (session_id, start_time, end_time))
    
    events = cursor.fetchall()
    
    # Parse events to find dominant app and window
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
    
    # Calculate duration
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    duration_seconds = (end_dt - start_dt).total_seconds()
    
    return {
        "app": dominant_app,
        "window_title": dominant_window,
        "normalized_title": dominant_window,  # Will be normalized by feature extractor
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "input_count": input_count
    }


def reclassify_all_segments(dry_run: bool = True):
    """
    Re-classify all segments using v2 classification system.
    
    Args:
        dry_run: If True, only preview changes without updating database
    """
    from agent.task.feature_extraction import FeatureExtractor
    from agent.task.core_tasks import get_task_recommendation
    
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all segments
    cursor.execute("""
        SELECT id, session_id, task_id, start_time, end_time, confidence, reason
        FROM task_segments
        WHERE end_time IS NOT NULL
        ORDER BY start_time DESC
    """)
    
    segments = [dict(row) for row in cursor.fetchall()]
    
    print(f"\n{'='*80}")
    print(f"Reclassification - {len(segments)} segments")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE UPDATE'}")
    print(f"{'='*80}\n")
    
    if not segments:
        print("No segments found to reclassify.")
        return
    
    # Initialize feature extractor
    extractor = FeatureExtractor(db_path)
    
    # Statistics
    stats = {
        "total": len(segments),
        "changed": 0,
        "unchanged": 0,
        "errors": 0
    }
    
    old_distribution = Counter()
    new_distribution = Counter()
    reason_distribution = Counter()
    
    updates = []  # Store updates for batch execution
    
    for i, segment in enumerate(segments, 1):
        try:
            # Reconstruct segment context from events
            segment_context = get_segment_context_from_events(
                segment["id"],
                segment["session_id"],
                segment["start_time"],
                segment["end_time"],
                conn
            )
            
            # Extract features
            features = extractor.extract_features(segment_context)
            
            # Classify
            new_task_id, new_confidence, new_reason = get_task_recommendation(features)
            
            old_task_id = segment["task_id"]
            old_distribution[old_task_id] += 1
            new_distribution[new_task_id] += 1
            reason_distribution[new_reason] += 1
            
            changed = old_task_id != new_task_id
            if changed:
                stats["changed"] += 1
            else:
                stats["unchanged"] += 1
            
            # Store update
            updates.append({
                "segment_id": segment["id"],
                "old_task": old_task_id,
                "new_task": new_task_id,
                "new_confidence": new_confidence,
                "new_reason": new_reason,
                "app": segment_context.get("app", "unknown"),
                "window": segment_context.get("window_title", "")[:40],
                "changed": changed
            })
            
            # Print progress for changed segments or first 10
            if changed or i <= 10:
                change_marker = "→ CHANGED" if changed else ""
                print(f"Segment {segment['id']:4d} {change_marker}")
                print(f"  App: {segment_context.get('app', 'unknown')[:30]:<30}")
                print(f"  Window: {segment_context.get('window_title', 'N/A')[:50]}")
                print(f"  Old: {old_task_id:<25} (conf={segment.get('confidence', 0):.2f})")
                print(f"  New: {new_task_id:<25} (conf={new_confidence:.2f}, {new_reason})")
                print()
        
        except Exception as e:
            print(f"ERROR on segment {segment['id']}: {e}")
            stats["errors"] += 1
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("RECLASSIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total segments: {stats['total']}")
    print(f"Changed: {stats['changed']} ({stats['changed']/stats['total']*100:.1f}%)")
    print(f"Unchanged: {stats['unchanged']} ({stats['unchanged']/stats['total']*100:.1f}%)")
    print(f"Errors: {stats['errors']}")
    
    print("\nOld Task Distribution:")
    for task, count in old_distribution.most_common():
        print(f"  {task:30s}: {count:3d} ({count/stats['total']*100:.1f}%)")
    
    print("\nNew Task Distribution:")
    for task, count in new_distribution.most_common():
        print(f"  {task:30s}: {count:3d} ({count/stats['total']*100:.1f}%)")
    
    print("\nClassification Methods:")
    for reason, count in reason_distribution.most_common():
        print(f"  {reason:30s}: {count:3d} ({count/stats['total']*100:.1f}%)")
    
    # Check for unknowns
    unknown_count = new_distribution.get("unknown", 0)
    if unknown_count == 0:
        print(f"\n{'✓'*40}")
        print("SUCCESS! Zero unknown classifications!")
        print(f"{'✓'*40}\n")
    else:
        print(f"\n⚠ Warning: {unknown_count} segments still classified as 'unknown'\n")
    
    # Apply updates if not dry run
    if not dry_run:
        print(f"\nApplying {stats['changed']} updates to database...")
        
        update_cursor = conn.cursor()
        for update in updates:
            if update["changed"]:
                update_cursor.execute("""
                    UPDATE task_segments
                    SET task_id = ?,
                        confidence = ?,
                        reason = ?
                    WHERE id = ?
                """, (
                    update["new_task"],
                    update["new_confidence"],
                    update["new_reason"],
                    update["segment_id"]
                ))
        
        conn.commit()
        print(f"✓ Database updated successfully!")
    else:
        print("\n⚠ DRY RUN: No changes were made to the database.")
        print("Run with --apply to actually update the database:")
        print("  python -m agent.task.reclassify_segments --apply")
    
    conn.close()
    
    return stats, updates


if __name__ == "__main__":
    import sys
    
    dry_run = "--apply" not in sys.argv
    
    if not dry_run:
        print("\n⚠ WARNING: This will modify the database!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    
    stats, updates = reclassify_all_segments(dry_run=dry_run)
