"""
Baseline data export for ML training.

This module provides functions to export task segments with their baseline metadata
for use in training ML models. The baseline format preserves:
- Raw app and window title data
- Generic task category (base_category)
- Smart contextual name
- Behavioral features
- Confidence scores

Usage:
    from agent.task.baseline_export import export_baseline_segments
    
    # Export all segments
    segments = export_baseline_segments()
    
    # Export segments from specific date range
    segments = export_baseline_segments(
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    # Save to file
    import json
    with open("training_data.json", "w") as f:
        json.dump(segments, f, indent=2)
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


def get_db_path() -> Path:
    """Get the path to the events database."""
    return Path(__file__).parent.parent / "storage" / "events.db"


def export_baseline_segments(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_duration_seconds: int = 30,
    include_low_confidence: bool = False,
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Export task segments in baseline format for ML training.
    
    Args:
        start_date: Optional ISO date string (YYYY-MM-DD) for filtering
        end_date: Optional ISO date string (YYYY-MM-DD) for filtering
        min_duration_seconds: Minimum segment duration to include
        include_low_confidence: Whether to include segments below confidence threshold
        confidence_threshold: Minimum confidence score (if include_low_confidence=False)
    
    Returns:
        List of segment dictionaries in baseline format
    """
    db_path = get_db_path()
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return []
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build query with optional filters
    query = """
        SELECT 
            session_id,
            task_id,
            start_time,
            end_time,
            confidence,
            feature_vector,
            reason
        FROM task_segments
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND date(start_time) >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date(start_time) <= ?"
        params.append(end_date)
    
    if not include_low_confidence:
        query += " AND confidence >= ?"
        params.append(confidence_threshold)
    
    query += " ORDER BY start_time"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    segments = []
    for row in rows:
        # Calculate duration
        start_time = datetime.fromisoformat(row["start_time"])
        end_time_str = row["end_time"]
        
        if not end_time_str:
            # Segment still active - skip for training data
            continue
        
        end_time = datetime.fromisoformat(end_time_str)
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Filter by minimum duration
        if duration_seconds < min_duration_seconds:
            continue
        
        # Parse feature vector
        features = {}
        try:
            features = json.loads(row["feature_vector"]) if row["feature_vector"] else {}
        except json.JSONDecodeError:
            features = {}
        
        # Extract metadata from features
        raw_app = features.get("active_app", "")
        raw_window = features.get("active_window_title", "")
        
        # Normalize window title
        from agent.task.smart_naming import normalize_window_title
        normalized_window = normalize_window_title(raw_window) if raw_window else ""
        
        # Parse task_id to extract base category if available
        # Format could be: "Administrative Work - ChatGPT" or just "administrative_work"
        task_id = row["task_id"]
        base_category = task_id  # Default to full task_id
        
        # Try to extract base category from reason field
        # Reason format: "initial_administrative_work", "continuing_deep_development", etc.
        reason = row["reason"] or ""
        if "_" in reason:
            parts = reason.split("_", 1)
            if len(parts) == 2:
                action, category = parts
                if action in ["initial", "continuing", "transition"]:
                    base_category = category
        
        # Build baseline segment
        segment = {
            "session_id": row["session_id"],
            "start_time": row["start_time"],
            "end_time": end_time_str,
            "duration_seconds": duration_seconds,
            "app": raw_app,
            "window_title": raw_window,
            "normalized_title": normalized_window,
            "generic_task": base_category,
            "smart_name": task_id,
            "confidence": row["confidence"],
            "reason": reason,
            "feature_snapshot": {
                # Key behavioral features for ML
                "input_rate": features.get("input_rate_per_minute", 0.0),
                "typing_ratio": features.get("typing_ratio", 0.0),
                "mouse_ratio": features.get("mouse_ratio", 0.0),
                "window_switches": features.get("window_switches_per_minute", 0.0),
                "time_in_window": features.get("time_in_current_window_seconds", 0.0),
                "clipboard_events": features.get("clipboard_events_per_minute", 0.0),
                "has_coding_app": features.get("has_coding_app", False),
                "has_browser": features.get("has_browser", False),
                "has_communication_app": features.get("has_communication_app", False),
            }
        }
        
        segments.append(segment)
    
    conn.close()
    
    print(f"Exported {len(segments)} baseline segments")
    return segments


def get_baseline_stats(segments: List[Dict]) -> Dict:
    """
    Calculate statistics about baseline segments for data quality assessment.
    
    Args:
        segments: List of baseline segment dictionaries
    
    Returns:
        Dictionary with statistics
    """
    if not segments:
        return {"total_segments": 0}
    
    # Count by generic task category
    category_counts = {}
    total_duration = 0
    confidence_sum = 0
    apps = set()
    
    for seg in segments:
        category = seg.get("generic_task", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        total_duration += seg.get("duration_seconds", 0)
        confidence_sum += seg.get("confidence", 0)
        if seg.get("app"):
            apps.add(seg["app"])
    
    avg_duration = total_duration / len(segments)
    avg_confidence = confidence_sum / len(segments)
    
    return {
        "total_segments": len(segments),
        "categories": category_counts,
        "avg_duration_seconds": round(avg_duration, 1),
        "avg_confidence": round(avg_confidence, 3),
        "unique_apps": len(apps),
        "apps": sorted(list(apps))
    }


if __name__ == "__main__":
    # Example usage
    print("Exporting baseline segments...")
    segments = export_baseline_segments(min_duration_seconds=60)
    
    if segments:
        stats = get_baseline_stats(segments)
        print("\nBaseline Data Statistics:")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Average duration: {stats['avg_duration_seconds']}s")
        print(f"Average confidence: {stats['avg_confidence']}")
        print(f"\nCategories:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count}")
        print(f"\nUnique apps: {stats['unique_apps']}")
        
        # Save to file
        output_file = Path(__file__).parent / "baseline_training_data.json"
        with open(output_file, "w") as f:
            json.dump({
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "total_segments": len(segments),
                    "statistics": stats
                },
                "segments": segments
            }, f, indent=2)
        print(f"\nSaved to: {output_file}")
    else:
        print("No segments found to export")
