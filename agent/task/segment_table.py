"""
Task Segment Table Display

Utility to query and display task segments in a table format for analysis.
Shows how sessions are split into granular task segments based on context changes.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


def simple_table(data: List[List], headers: List[str]) -> str:
    """Simple table formatter without external dependencies."""
    if not data:
        return "No data"
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Build table
    lines = []
    
    # Header separator
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines.append(sep)
    
    # Header row
    header_row = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines.append(header_row)
    lines.append(sep)
    
    # Data rows
    for row in data:
        data_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        lines.append(data_row)
    
    lines.append(sep)
    return "\n".join(lines)


def get_db_path() -> Path:
    """Get the path to the events database."""
    return Path(__file__).parent.parent / "storage" / "events.db"


def get_task_segments(
    session_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_duration_seconds: int = 0,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Query task segments from database.
    
    Args:
        session_id: Filter by specific session ID
        start_date: Filter by start date (YYYY-MM-DD)
        end_date: Filter by end date (YYYY-MM-DD)
        min_duration_seconds: Minimum segment duration
        limit: Maximum number of segments to return
    
    Returns:
        List of segment dictionaries
    """
    db_path = get_db_path()
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return []
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build query
    query = """
        SELECT 
            ts.id as segment_id,
            ts.session_id,
            ts.task_id,
            ts.start_time,
            ts.end_time,
            ts.confidence,
            ts.reason,
            ts.feature_vector,
            s.start_time as session_start,
            s.end_time as session_end
        FROM task_segments ts
        LEFT JOIN sessions s ON ts.session_id = s.session_id
        WHERE 1=1
    """
    params = []
    
    if session_id:
        query += " AND ts.session_id = ?"
        params.append(session_id)
    
    if start_date:
        query += " AND date(ts.start_time) >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date(ts.start_time) <= ?"
        params.append(end_date)
    
    query += " ORDER BY ts.start_time DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    segments = []
    for row in rows:
        # Calculate duration
        start_time = datetime.fromisoformat(row["start_time"])
        end_time_str = row["end_time"]
        
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str)
            duration_seconds = (end_time - start_time).total_seconds()
        else:
            duration_seconds = 0  # Active segment
        
        # Skip if below minimum duration
        if duration_seconds < min_duration_seconds:
            continue
        
        # Parse feature vector to extract app/window
        import json
        features = {}
        try:
            features = json.loads(row["feature_vector"]) if row["feature_vector"] else {}
        except json.JSONDecodeError:
            features = {}
        
        app = features.get("active_app", "unknown")
        window_title = features.get("active_window_title", "")
        
        # Normalize window title
        from agent.task.smart_naming import normalize_window_title
        normalized_title = normalize_window_title(window_title) if window_title else ""
        
        # Extract generic task from reason field
        reason = row["reason"] or ""
        generic_task = "unknown"
        
        if "_" in reason:
            parts = reason.split("_")
            # Patterns: "initial_administrative_work", "continuing_deep_development", "window_change"
            if parts[0] in ["initial", "continuing", "behavioral"]:
                generic_task = "_".join(parts[1:]) if len(parts) > 1 else parts[-1]
            elif "transition" in parts[0]:
                # "behavioral_transition_admin_to_coding"
                if "to" in parts:
                    to_idx = parts.index("to")
                    generic_task = "_".join(parts[to_idx + 1:]) if to_idx < len(parts) - 1 else "unknown"
            elif parts[0] in ["window", "app", "context"]:
                # Context change - try to extract from previous context
                generic_task = "context_change"
        
        # Fallback: extract from task_id if available
        if generic_task in ["unknown", "context_change"]:
            task_id = row["task_id"]
            # Try to match known categories
            known_categories = [
                "administrative_work", "deep_development", "technical_research",
                "context_switching", "team_meeting", "strategic_planning"
            ]
            for cat in known_categories:
                if cat.replace("_", " ").lower() in task_id.lower():
                    generic_task = cat
                    break
        
        segment = {
            "segment_id": row["segment_id"],
            "session_id": row["session_id"],
            "start_time": start_time,
            "end_time": end_time if end_time_str else None,
            "duration_seconds": duration_seconds,
            "duration_minutes": round(duration_seconds / 60, 1),
            "app": app,
            "window_title": window_title,
            "normalized_title": normalized_title,
            "generic_task": generic_task,
            "smart_name": row["task_id"],
            "confidence": row["confidence"],
            "reason": reason
        }
        
        segments.append(segment)
    
    conn.close()
    return segments


def display_segments_table(
    segments: List[Dict],
    show_full_title: bool = False,
    max_title_length: int = 40
) -> str:
    """
    Format segments as a table.
    
    Args:
        segments: List of segment dictionaries
        show_full_title: Whether to show full window title
        max_title_length: Maximum length for window title display
    
    Returns:
        Formatted table string
    """
    if not segments:
        return "No segments found."
    
    # Prepare table data
    table_data = []
    for seg in segments:
        # Format times
        start_str = seg["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        end_str = seg["end_time"].strftime("%H:%M:%S") if seg["end_time"] else "active"
        
        # Truncate window title if needed
        title = seg["normalized_title"] or seg["window_title"]
        if not show_full_title and len(title) > max_title_length:
            title = title[:max_title_length - 3] + "..."
        
        # Format duration
        duration_str = f"{seg['duration_minutes']}m" if seg["duration_minutes"] > 0 else "-"
        
        # Format segment ID (it's an integer)
        seg_id = str(seg["segment_id"]) if seg["segment_id"] else "-"
        
        table_data.append([
            seg_id,
            start_str,
            end_str,
            duration_str,
            seg["app"],
            title,
            seg["generic_task"],
            f"{seg['confidence']:.2f}"
        ])
    
    headers = ["Segment ID", "Start", "End", "Duration", "App", "Window Title", "Generic Task", "Conf"]
    
    return simple_table(table_data, headers)


def display_session_segments(session_id: str) -> str:
    """Display all segments for a specific session in table format."""
    segments = get_task_segments(session_id=session_id)
    
    if not segments:
        return f"No segments found for session {session_id}"
    
    # Calculate session summary
    total_duration = sum(s["duration_seconds"] for s in segments)
    total_minutes = total_duration / 60
    
    # Count unique contexts
    unique_apps = len(set(s["app"] for s in segments))
    unique_tasks = len(set(s["generic_task"] for s in segments))
    
    summary = f"\nSession: {session_id}\n"
    summary += f"Total segments: {len(segments)}\n"
    summary += f"Total duration: {total_minutes:.1f} minutes\n"
    summary += f"Unique apps: {unique_apps}\n"
    summary += f"Unique tasks: {unique_tasks}\n\n"
    
    return summary + display_segments_table(segments)


def display_recent_segments(limit: int = 20) -> str:
    """Display most recent segments."""
    segments = get_task_segments(limit=limit, min_duration_seconds=30)
    
    header = f"\nMost Recent Segments (limit: {limit}, min duration: 30s)\n"
    header += "=" * 80 + "\n"
    
    return header + display_segments_table(segments)


def get_segment_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    Calculate statistics about task segmentation.
    
    Returns:
        Dictionary with segmentation statistics
    """
    segments = get_task_segments(start_date=start_date, end_date=end_date, min_duration_seconds=30)
    
    if not segments:
        return {"total_segments": 0}
    
    # Group by session
    sessions = {}
    for seg in segments:
        sid = seg["session_id"]
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(seg)
    
    # Calculate stats
    total_duration = sum(s["duration_seconds"] for s in segments)
    avg_segment_duration = total_duration / len(segments)
    
    # Segments per session
    segments_per_session = [len(segs) for segs in sessions.values()]
    avg_segments_per_session = sum(segments_per_session) / len(sessions)
    
    # Task distribution
    task_counts = {}
    task_durations = {}
    for seg in segments:
        task = seg["generic_task"]
        task_counts[task] = task_counts.get(task, 0) + 1
        task_durations[task] = task_durations.get(task, 0) + seg["duration_seconds"]
    
    # App distribution
    app_counts = {}
    for seg in segments:
        app = seg["app"]
        app_counts[app] = app_counts.get(app, 0) + 1
    
    return {
        "total_segments": len(segments),
        "total_sessions": len(sessions),
        "avg_segments_per_session": round(avg_segments_per_session, 1),
        "total_duration_minutes": round(total_duration / 60, 1),
        "avg_segment_duration_minutes": round(avg_segment_duration / 60, 1),
        "task_distribution": {
            task: {
                "count": count,
                "total_minutes": round(task_durations.get(task, 0) / 60, 1)
            }
            for task, count in task_counts.items()
        },
        "app_distribution": dict(sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    }


if __name__ == "__main__":
    import sys
    
    # CLI interface
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "recent":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            print(display_recent_segments(limit=limit))
        
        elif command == "session":
            if len(sys.argv) < 3:
                print("Usage: python segment_table.py session <session_id>")
            else:
                session_id = sys.argv[2]
                print(display_session_segments(session_id))
        
        elif command == "stats":
            stats = get_segment_statistics()
            print("\nTask Segmentation Statistics")
            print("=" * 80)
            print(f"Total segments: {stats['total_segments']}")
            print(f"Total sessions: {stats['total_sessions']}")
            print(f"Average segments per session: {stats['avg_segments_per_session']}")
            print(f"Average segment duration: {stats['avg_segment_duration_minutes']} minutes")
            print(f"\nTask Distribution:")
            for task, data in stats['task_distribution'].items():
                print(f"  {task}: {data['count']} segments ({data['total_minutes']} minutes)")
            print(f"\nTop Apps:")
            for app, count in list(stats['app_distribution'].items())[:10]:
                print(f"  {app}: {count} segments")
    
    else:
        # Default: show recent segments
        print(display_recent_segments(limit=15))
