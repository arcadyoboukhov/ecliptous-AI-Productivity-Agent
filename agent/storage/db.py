"""
SQLite storage layer for events persistence.

This module provides database initialization and access patterns for the productivity agent.
Events are persisted to events.db which lives next to this module.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Tuple


def get_db_path() -> str:
    """Return the path to the events database."""
    db_dir = Path(__file__).parent
    return str(db_dir / "events.db")


def get_connection() -> sqlite3.Connection:
    """Get a connection to the events database."""
    try:
        db_path = get_db_path()
        conn = sqlite3.connect(db_path, timeout=10.0)  # 10 second timeout
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(ComponentType.DATABASE, "get_connection", e, ErrorSeverity.CRITICAL)
        raise


def init_db():
    """Initialize the events database schema if needed."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create events table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                session_id TEXT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for efficient session_id lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_session_id 
            ON events(session_id)
        """)
        
        # Create intensity_scores table for session intensity tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intensity_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                intensity_score REAL NOT NULL,
                window_seconds INTEGER DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create index for efficient session_id lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_intensity_session_id 
            ON intensity_scores(session_id, timestamp DESC)
        """)
        
        # Create state_history table for tracking UI state changes (ACTIVE_ALIGNED, PAUSED, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for efficient state history lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_state_history_timestamp 
            ON state_history(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_state_history_session 
            ON state_history(session_id, timestamp DESC)
        """)
        
        # Create sessions table to store session metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                device_id TEXT DEFAULT 'unknown',
                in_progress INTEGER DEFAULT 0,
                current_task_id TEXT,
                current_task_confidence REAL DEFAULT 0.0,
                total_events INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_id 
            ON sessions(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_time 
            ON sessions(start_time DESC)
        """)
        
        # Create task_segments table to store ML-classified task segments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                confidence REAL DEFAULT 0.0,
                reason TEXT,
                distance_to_centroid REAL,
                feature_vector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_segments_session 
            ON task_segments(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_segments_time 
            ON task_segments(start_time DESC)
        """)

        # Create enriched_tasks table to store normalized task objects
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enriched_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                session_id TEXT,
                task_name TEXT NOT NULL,
                app TEXT,
                window_title TEXT,
                avg_cpu REAL,
                avg_ram REAL,
                avg_gpu REAL,
                mic_on INTEGER,
                camera_on INTEGER,
                audio_volume REAL,
                copy_count INTEGER,
                paste_count INTEGER,
                keyboard_intensity REAL,
                mouse_activity REAL,
                start_time TEXT,
                end_time TEXT,
                duration_minutes REAL,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_enriched_tasks_time
            ON enriched_tasks(start_time DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_enriched_tasks_session
            ON enriched_tasks(session_id, start_time DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_enriched_tasks_task
            ON enriched_tasks(task_id, start_time DESC)
        """)

        # Create analytics_snapshots table for realtime analytics payloads
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_snapshots_time
            ON analytics_snapshots(timestamp DESC)
        """)
        
        # Create interval_signals table for storing normalized interval data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interval_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_start TEXT NOT NULL,
                timestamp_end TEXT NOT NULL,
                
                -- User Activity
                app TEXT,
                window_title TEXT,
                
                -- Keyboard/Mouse/Copy-Paste
                keyboard_intensity REAL DEFAULT 0.0,
                mouse_clicks INTEGER DEFAULT 0,
                mouse_distance REAL DEFAULT 0.0,
                copy_count INTEGER DEFAULT 0,
                paste_count INTEGER DEFAULT 0,
                cut_count INTEGER DEFAULT 0,
                
                -- Audio/Video
                mic_active INTEGER DEFAULT 0,
                camera_active INTEGER DEFAULT 0,
                audio_volume REAL DEFAULT 0.0,
                
                -- System Resources (normalized 0-1)
                cpu_usage REAL DEFAULT 0.0,
                ram_usage REAL DEFAULT 0.0,
                gpu_usage REAL,
                disk_read_mbps REAL DEFAULT 0.0,
                disk_write_mbps REAL DEFAULT 0.0,
                
                -- Contextual
                time_of_day TEXT,
                day_of_week INTEGER,
                is_weekend INTEGER DEFAULT 0,
                is_work_hours INTEGER DEFAULT 0,
                
                -- Session context
                session_id TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create indexes for efficient interval queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interval_signals_time 
            ON interval_signals(timestamp_start DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interval_signals_session 
            ON interval_signals(session_id, timestamp_start DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interval_signals_app 
            ON interval_signals(app, timestamp_start DESC)
        """)
        
        # Create live_task_predictions table for real-time task predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_task_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                distance_to_centroid REAL,
                reason TEXT,
                feature_window_seconds INTEGER DEFAULT 60,
                feature_vector TEXT,
                alternative_tasks TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create indexes for efficient live prediction queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_predictions_session 
            ON live_task_predictions(session_id, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_live_predictions_time 
            ON live_task_predictions(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        from agent.error_handling import handle_critical_failure, ComponentType
        handle_critical_failure(ComponentType.DATABASE, "init_db", e)
        raise


def log_event(event_type: str, payload: Optional[dict] = None, ts: Optional[datetime] = None, session_id: Optional[str] = None) -> int:
    """
    Log an event to the database with optional session association.
    
    Args:
        event_type: Type of event (e.g., 'INPUT_ACTIVITY', 'CONTEXT_SWITCH')
        payload: Optional dict payload
        ts: Optional timestamp (defaults to now in UTC)
        session_id: Optional session ID for association. If None, event is unassigned (background)
    
    Returns:
        The row ID of the inserted event
    
    Design (Decoupled Session Interaction):
    - If session_id is provided: event is associated with that session
    - If session_id is None: event is stored as background/unassigned
    - Background events can later be inferred into sessions during post-processing
    - Data collection does NOT depend on session state
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if ts is None:
            ts = datetime.now(timezone.utc)
        
        payload_str = None
        if payload is not None:
            try:
                payload_str = json.dumps(payload)
            except Exception:
                payload_str = str(payload)
        
        cursor.execute(
            "INSERT INTO events (timestamp, event_type, session_id, payload) VALUES (?, ?, ?, ?)",
            (ts.isoformat(), event_type, session_id, payload_str)
        )
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return row_id
        
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(
            ComponentType.DATABASE, 
            "log_event", 
            e, 
            ErrorSeverity.ERROR,
            event_type=event_type,
            session_id=session_id
        )
        return -1  # Return invalid ID on error


def get_events(event_type: Optional[str] = None, limit: Optional[int] = None) -> List[Tuple[str, str, Optional[dict]]]:
    """
    Retrieve events from the database.
    
    Args:
        event_type: Optional filter by event type
        limit: Optional limit on number of results
    
    Returns:
        List of (timestamp, event_type, payload) tuples
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if event_type:
        query = "SELECT timestamp, event_type, payload FROM events WHERE event_type = ? ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query, (event_type,))
    else:
        query = "SELECT timestamp, event_type, payload FROM events ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
    
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        ts, evt_type, payload_str = row[0], row[1], row[2]
        payload = None
        if payload_str:
            try:
                payload = json.loads(payload_str)
            except Exception:
                payload = payload_str
        result.append((ts, evt_type, payload))
    
    return result


def clear_events():
    """Clear all events from the database (for testing)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()


def log_intensity_score(session_id: str, intensity_score: float, timestamp: Optional[datetime] = None, window_seconds: int = 10) -> int:
    """
    Log an intensity score to the database.
    
    Args:
        session_id: The session this intensity score belongs to
        intensity_score: The intensity value (0-100)
        timestamp: Optional timestamp (defaults to now in UTC)
        window_seconds: Time window used to calculate the score (default 10)
    
    Returns:
        The row ID of the inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    cursor.execute(
        "INSERT INTO intensity_scores (session_id, timestamp, intensity_score, window_seconds) VALUES (?, ?, ?, ?)",
        (session_id, timestamp.isoformat(), intensity_score, window_seconds)
    )
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return row_id


def log_state_change(state: str, session_id: Optional[str] = None, timestamp: Optional[datetime] = None) -> int:
    """
    Log a state change to the database.
    
    Args:
        state: The state (ACTIVE_ALIGNED, ACTIVE_UNALIGNED, PAUSED, IDLE, UNKNOWN, etc.)
        session_id: Optional session ID this state is associated with
        timestamp: Optional timestamp (defaults to now in UTC)
    
    Returns:
        The row ID of the inserted record
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    cursor.execute(
        "INSERT INTO state_history (session_id, timestamp, state) VALUES (?, ?, ?)",
        (session_id, timestamp.isoformat(), state)
    )
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return row_id


def get_state_history(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, session_id: Optional[str] = None) -> List[Tuple[datetime, str, Optional[str]]]:
    """
    Retrieve state history within a time range.
    
    Args:
        start_time: Optional start of time range
        end_time: Optional end of time range
        session_id: Optional filter by session ID
    
    Returns:
        List of (timestamp, state, session_id) tuples
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    conditions = []
    params = []
    
    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time.isoformat())
    
    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time.isoformat())
    
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    
    query = "SELECT timestamp, state, session_id FROM state_history"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp ASC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row[0])
            result.append((ts, row[1], row[2]))
        except Exception:
            pass
    
    return result


def get_intensity_scores(session_id: str, limit: Optional[int] = None) -> List[Tuple[datetime, float]]:
    """
    Retrieve intensity scores for a session.
    
    Args:
        session_id: The session ID to query
        limit: Optional limit on number of results
    
    Returns:
        List of (timestamp, intensity_score) tuples
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT timestamp, intensity_score FROM intensity_scores WHERE session_id = ? ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        ts_str, score = row[0], row[1]
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.now(timezone.utc)
        result.append((ts, score))
    
    return result


def get_intensity_stats(session_id: str) -> dict:
    """
    Get intensity statistics for a session from the database.
    
    Args:
        session_id: The session ID to query
    
    Returns:
        Dict with count, average, min, max statistics
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            AVG(intensity_score) as average,
            MIN(intensity_score) as min,
            MAX(intensity_score) as max
        FROM intensity_scores
        WHERE session_id = ?
    """, (session_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] > 0:
        return {
            "count": row[0],
            "average": row[1],
            "min": row[2],
            "max": row[3]
        }
    else:
        return {
            "count": 0,
            "average": None,
            "min": None,
            "max": None
        }


def clear_intensity_scores(session_id: Optional[str] = None):
    """Clear intensity scores from the database (for testing).
    
    Args:
        session_id: Optional session ID to clear. If None, clears all.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("DELETE FROM intensity_scores WHERE session_id = ?", (session_id,))
    else:
        cursor.execute("DELETE FROM intensity_scores")
    
    conn.commit()
    conn.close()


# ============================================================================
# Decoupled Session Interaction Queries
# ============================================================================

def get_events_for_session(session_id: str) -> List[Tuple]:
    """
    Get all events associated with a specific session.
    
    Args:
        session_id: Session ID to query
    
    Returns:
        List of event tuples: (id, timestamp, event_type, session_id, payload)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, event_type, session_id, payload
        FROM events
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    
    results = cursor.fetchall()
    conn.close()
    return results


def upsert_session_record(session) -> bool:
    """Insert or update a session record in the sessions table."""
    try:
        session_id = getattr(session, "session_id", None) or getattr(session, "id", None)
        if not session_id:
            return False

        start_time = getattr(session, "start", None) or getattr(session, "started_at", None)
        end_time = getattr(session, "end", None) or getattr(session, "ended_at", None)
        device_id = getattr(session, "device_id", "unknown")
        in_progress = 1 if getattr(session, "in_progress", False) else 0
        total_events = getattr(session, "event_count", 0)

        current_task_id = None
        current_task_confidence = 0.0
        assignment = getattr(session, "current_task_assignment", None)
        if isinstance(assignment, dict) and assignment.get("task_id"):
            current_task_id = assignment.get("task_id")
            current_task_confidence = assignment.get("confidence", 0.0) or 0.0
        else:
            inferred_task_id = getattr(session, "inferred_task_id", None)
            if inferred_task_id:
                current_task_id = inferred_task_id
                current_task_confidence = 1.0

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (
                session_id, start_time, end_time, device_id, in_progress,
                current_task_id, current_task_confidence, total_events
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                device_id=excluded.device_id,
                in_progress=excluded.in_progress,
                current_task_id=excluded.current_task_id,
                current_task_confidence=excluded.current_task_confidence,
                total_events=excluded.total_events,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                session_id,
                start_time.isoformat() if start_time else None,
                end_time.isoformat() if end_time else None,
                device_id,
                in_progress,
                current_task_id,
                float(current_task_confidence),
                int(total_events) if total_events is not None else 0,
            ),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(ComponentType.DATABASE, "upsert_session_record", e, ErrorSeverity.ERROR)
        return False


def replace_task_segments(session_id: str, segments: list) -> bool:
    """Replace task_segments for a session with the provided list."""
    try:
        if not session_id:
            return False

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM task_segments WHERE session_id = ?", (session_id,))

        rows = []
        for seg in segments or []:
            if isinstance(seg, dict):
                task_id = seg.get("task_id")
                start_time = seg.get("start_time")
                end_time = seg.get("end_time")
                confidence = seg.get("confidence", 0.0)
                reason = seg.get("reason")
                distance = seg.get("distance_to_centroid")
                feature_vector = seg.get("feature_vector")
            else:
                task_id = getattr(seg, "task_id", None)
                start_time = getattr(seg, "start_time", None)
                end_time = getattr(seg, "end_time", None)
                confidence = getattr(seg, "confidence", 0.0)
                reason = getattr(seg, "reason", None)
                distance = getattr(seg, "distance_to_centroid", None)
                feature_vector = getattr(seg, "feature_vector", None)

            if not task_id or not start_time:
                continue

            if isinstance(start_time, str):
                start_iso = start_time
            else:
                start_iso = start_time.isoformat()

            if end_time:
                end_iso = end_time if isinstance(end_time, str) else end_time.isoformat()
            else:
                end_iso = None

            fv = None
            if feature_vector is not None:
                try:
                    fv = json.dumps(feature_vector)
                except Exception:
                    fv = str(feature_vector)

            rows.append(
                (
                    session_id,
                    task_id,
                    start_iso,
                    end_iso,
                    float(confidence or 0.0),
                    reason,
                    distance,
                    fv,
                )
            )

        if rows:
            cursor.executemany(
                """
                INSERT INTO task_segments (
                    session_id, task_id, start_time, end_time, confidence,
                    reason, distance_to_centroid, feature_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(ComponentType.DATABASE, "replace_task_segments", e, ErrorSeverity.ERROR)
        return False


def get_background_events() -> List[Tuple]:
    """
    Get all background/unassigned events (session_id IS NULL).
    
    These are events collected when no explicit session was active.
    Can later be inferred into sessions during post-processing.
    
    Returns:
        List of event tuples: (id, timestamp, event_type, session_id, payload)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, event_type, session_id, payload
        FROM events
        WHERE session_id IS NULL
        ORDER BY timestamp ASC
    """)
    
    results = cursor.fetchall()
    conn.close()
    return results


def count_events_by_session() -> dict:
    """
    Count events grouped by session (including background).
    
    Returns:
        Dict: {session_id: count, ...} where None key is background events
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT session_id, COUNT(*) as count
        FROM events
        GROUP BY session_id
        ORDER BY count DESC
    """)
    
    results = {}
    for row in cursor.fetchall():
        session_id = row[0]
        count = row[1]
        # Convert NULL to None for background
        results[session_id if session_id is not None else "background"] = count
    
    conn.close()
    return results


def infer_background_events_to_sessions(session_mapping: dict) -> dict:
    """
    Infer background events into sessions based on timestamp proximity.
    
    Args:
        session_mapping: Dict of {session_id: (start_time, end_time, ...)}
    
    Returns:
        Dict: {event_id: assigned_session_id} for events that were inferred
    """
    background_events = get_background_events()
    assignments = {}
    
    for event in background_events:
        event_id = event[0]
        event_timestamp_str = event[1]
        
        # Parse timestamp string to datetime
        event_timestamp = datetime.fromisoformat(event_timestamp_str)
        
        # Find the best matching session based on timestamp
        best_session = None
        smallest_gap = float('inf')
        
        for session_id, session_info in session_mapping.items():
            start_time = session_info[0]
            end_time = session_info[1]
            
            # If event is within session bounds, assign immediately
            if start_time <= event_timestamp <= end_time:
                assignments[event_id] = session_id
                best_session = session_id
                break
            
            # Otherwise, track closest session
            gap = min(abs((start_time - event_timestamp).total_seconds()),
                     abs((end_time - event_timestamp).total_seconds()))
            if gap < smallest_gap:
                smallest_gap = gap
                best_session = session_id
        
        # If no exact match, assign to closest session (if within 5 minutes)
        if best_session and smallest_gap <= 300:
            assignments[event_id] = best_session
    
    return assignments


# ============================================================================
# SESSION PERSISTENCE (DB-only, no JSON)
# ============================================================================

def save_session(session) -> None:
    """
    Save a session object to the database.
    Stores session metadata and all task segments.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Session uses 'id' attribute, not 'session_id'
        session_id = getattr(session, 'id', None) or getattr(session, 'session_id', None) or str(session.start)[:8]
        start_time = session.start.isoformat() if session.start else datetime.now(timezone.utc).isoformat()
        end_time = session.end.isoformat() if session.end else None
        device_id = getattr(session, 'device_id', 'unknown')
        in_progress = 1 if getattr(session, 'in_progress', False) else 0
        
        current_task_id = None
        current_task_confidence = 0.0
        if hasattr(session, 'current_task_assignment') and session.current_task_assignment:
            current_task_id = session.current_task_assignment.get('task_id')
            current_task_confidence = session.current_task_assignment.get('confidence', 0.0)
        
        # Insert or update session
        cursor.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, start_time, end_time, device_id, in_progress, current_task_id, current_task_confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, start_time, end_time, device_id, in_progress, current_task_id, current_task_confidence, datetime.now(timezone.utc).isoformat()))
        
        # Save task segments
        if hasattr(session, 'intra_session_tasks') and session.intra_session_tasks:
            for segment in session.intra_session_tasks:
                task_id = getattr(segment, 'task_id', 'unknown')
                segment_start = getattr(segment, 'start_time', None)
                segment_end = getattr(segment, 'end_time', None)
                confidence = getattr(segment, 'confidence', 0.0)
                reason = getattr(segment, 'reason', '')
                distance = getattr(segment, 'distance_to_centroid', None)
                features = getattr(segment, 'feature_vector', None)
                
                feature_str = json.dumps(features) if features else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO task_segments
                    (session_id, task_id, start_time, end_time, confidence, reason, distance_to_centroid, feature_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    task_id,
                    segment_start.isoformat() if segment_start else None,
                    segment_end.isoformat() if segment_end else None,
                    confidence,
                    reason,
                    distance,
                    feature_str
                ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving session: {e}")
        raise


def load_sessions_from_db(days_back: int = 30) -> List:
    """
    Load sessions from the database for the last N days.
    Returns Session objects reconstructed from DB data.
    """
    from agent.session.sessionizer import Session
    from agent.task.online_classification import TaskSegment
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=days_back)).isoformat()
        
        # Load sessions
        cursor.execute("""
            SELECT session_id, start_time, end_time, device_id, in_progress, current_task_id, current_task_confidence
            FROM sessions
            WHERE start_time >= ?
            ORDER BY start_time DESC
        """, (cutoff,))
        
        sessions = []
        for row in cursor.fetchall():
            session_id, start_time_str, end_time_str, device_id, in_progress, current_task_id, current_task_confidence = row
            
            session = Session(
                start_time=datetime.fromisoformat(start_time_str),
                session_id=session_id,
                device_id=device_id
            )
            session.end = datetime.fromisoformat(end_time_str) if end_time_str else None
            session.in_progress = bool(in_progress)
            
            # Restore current task assignment
            if current_task_id:
                session.current_task_assignment = {
                    'task_id': current_task_id,
                    'confidence': current_task_confidence,
                    'reason': 'restored_from_db'
                }
            
            # Load task segments
            cursor.execute("""
                SELECT task_id, start_time, end_time, confidence, reason, distance_to_centroid, feature_vector
                FROM task_segments
                WHERE session_id = ?
                ORDER BY start_time ASC
            """, (session_id,))
            
            session.intra_session_tasks = []
            for seg_row in cursor.fetchall():
                task_id, seg_start_str, seg_end_str, confidence, reason, distance, feature_str = seg_row
                
                feature_vector = None
                if feature_str:
                    try:
                        feature_vector = json.loads(feature_str)
                    except:
                        pass
                
                segment = TaskSegment(
                    task_id=task_id,
                    start_time=datetime.fromisoformat(seg_start_str) if seg_start_str else None,
                    end_time=datetime.fromisoformat(seg_end_str) if seg_end_str else None,
                    confidence=confidence,
                    reason=reason,
                    distance_to_centroid=distance,
                    feature_vector=feature_vector
                )
                # Attempt to attach normalized metadata derived from stored feature_vector
                try:
                    from agent.task.inference import _normalize_extra_metadata

                    # Build base extra from stored feature_vector
                    extra = {
                        'metrics_snapshot': feature_vector or {},
                        'apps': [feature_vector.get('active_app')] if isinstance(feature_vector, dict) and feature_vector.get('active_app') else None,
                        'top_apps': None,
                        'domains': None,
                        'window_titles': [feature_vector.get('active_window_title')] if isinstance(feature_vector, dict) and feature_vector.get('active_window_title') else None,
                        'feature_count': int(feature_vector.get('total_input')) if isinstance(feature_vector, dict) and feature_vector.get('total_input') is not None else None,
                        'session_id': session_id,
                    }

                    # Augment metrics from interval_signals if available for this segment
                    try:
                        seg_start = datetime.fromisoformat(seg_start_str) if seg_start_str else None
                        seg_end = datetime.fromisoformat(seg_end_str) if seg_end_str else None
                        if seg_start and seg_end:
                            intervals = get_intervals(start_time=seg_start, end_time=seg_end, session_id=session_id, limit=1000)
                        else:
                            intervals = get_intervals(session_id=session_id, limit=200)

                        if intervals:
                            cpu_vals = [float(i.get('cpu_usage') or i.get('cpu_percent') or 0.0) for i in intervals]
                            ram_vals = [float(i.get('ram_usage') or i.get('memory_percent') or 0.0) for i in intervals]
                            gpu_vals = [float(i.get('gpu_usage') or i.get('gpu_percent') or 0.0) for i in intervals if (i.get('gpu_usage') is not None or i.get('gpu_percent') is not None)]
                            audio_vals = [float(i.get('audio_volume') or 0.0) for i in intervals]
                            key_vals = [float(i.get('keyboard_intensity') or 0.0) for i in intervals]
                            clicks = [float(i.get('mouse_clicks') or 0.0) for i in intervals]

                            avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
                            avg_ram = sum(ram_vals) / len(ram_vals) if ram_vals else 0.0
                            avg_gpu = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0
                            avg_audio = sum(audio_vals) / len(audio_vals) if audio_vals else 0.0
                            total_input = sum(key_vals) + sum(clicks)
                            # If the averages are all zero, try expanding the window +/- 2 minutes
                            if avg_cpu == 0.0 and avg_ram == 0.0 and avg_gpu == 0.0 and seg_start and seg_end:
                                try:
                                    expand_start = seg_start - timedelta(seconds=120)
                                    expand_end = seg_end + timedelta(seconds=120)
                                    alt_intervals = get_intervals(start_time=expand_start, end_time=expand_end, session_id=session_id, limit=1000)
                                    if alt_intervals:
                                        cpu_vals2 = [float(i.get('cpu_usage') or i.get('cpu_percent') or 0.0) for i in alt_intervals]
                                        ram_vals2 = [float(i.get('ram_usage') or i.get('memory_percent') or 0.0) for i in alt_intervals]
                                        gpu_vals2 = [float(i.get('gpu_usage') or i.get('gpu_percent') or 0.0) for i in alt_intervals if (i.get('gpu_usage') is not None or i.get('gpu_percent') is not None)]
                                        key_vals2 = [float(i.get('keyboard_intensity') or 0.0) for i in alt_intervals]
                                        clicks2 = [float(i.get('mouse_clicks') or 0.0) for i in alt_intervals]
                                        if cpu_vals2:
                                            avg_cpu = sum(cpu_vals2) / len(cpu_vals2)
                                        if ram_vals2:
                                            avg_ram = sum(ram_vals2) / len(ram_vals2)
                                        if gpu_vals2:
                                            avg_gpu = sum(gpu_vals2) / len(gpu_vals2)
                                        total_input = sum(key_vals2) + sum(clicks2)
                                except Exception:
                                    pass

                            # Attach aggregated metrics into metrics_snapshot so normalizer can pick them
                            try:
                                ms = dict(extra.get('metrics_snapshot') or {})
                                ms['cpu_usage'] = avg_cpu
                                ms['ram_usage'] = avg_ram
                                ms['gpu_usage'] = avg_gpu
                                ms['audio_volume'] = avg_audio
                                ms['event_count'] = int(total_input)
                                extra['metrics_snapshot'] = ms
                                # Also expose some top-level fields
                                extra['avg_cpu'] = avg_cpu
                                extra['avg_ram'] = avg_ram
                                extra['avg_gpu'] = avg_gpu
                            except Exception:
                                pass
                    except Exception:
                        # Ignore interval augmentation failures
                        pass

                    segment.metadata = extra
                    segment.metadata['normalized'] = _normalize_extra_metadata(extra)
                except Exception:
                    # Fail silently; metadata is optional for older DB rows
                    try:
                        segment.metadata = {}
                    except Exception:
                        pass
                session.intra_session_tasks.append(segment)
            
            sessions.append(session)
        
        conn.close()
        return sessions
    
    except Exception as e:
        print(f"Error loading sessions from DB: {e}")
        return []


def delete_session(session_id: str) -> None:
    """Delete a session and all its task segments from the database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM task_segments WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting session: {e}")
        raise


# ============================================================================
# Interval Signals Storage (Step 2: Storage & Data Management Layer)
# ============================================================================

def save_interval_signal(interval_data: dict) -> int:
    """
    Save a normalized interval signal to the database.
    
    Args:
        interval_data: Dictionary from IntervalSignals.to_dict()
        
    Returns:
        The row ID of the inserted interval
    """
    print(f"[DB] save_interval_signal called with session_id={interval_data.get('session_id')}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interval_signals (
                timestamp_start, timestamp_end,
                app, window_title,
                keyboard_intensity, mouse_clicks, mouse_distance,
                copy_count, paste_count, cut_count,
                mic_active, camera_active, audio_volume,
                cpu_usage, ram_usage, gpu_usage,
                disk_read_mbps, disk_write_mbps,
                time_of_day, day_of_week, is_weekend, is_work_hours,
                session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interval_data.get('timestamp_start'),
            interval_data.get('timestamp_end'),
            interval_data.get('app'),
            interval_data.get('window_title'),
            interval_data.get('keyboard_intensity', 0.0),
            interval_data.get('mouse_clicks', 0),
            interval_data.get('mouse_distance', 0.0),
            interval_data.get('copy_count', 0),
            interval_data.get('paste_count', 0),
            interval_data.get('cut_count', 0),
            1 if interval_data.get('mic_active') else 0,
            1 if interval_data.get('camera_active') else 0,
            interval_data.get('audio_volume', 0.0),
            interval_data.get('cpu_usage', 0.0),
            interval_data.get('ram_usage', 0.0),
            interval_data.get('gpu_usage'),
            interval_data.get('disk_read_mbps', 0.0),
            interval_data.get('disk_write_mbps', 0.0),
            interval_data.get('time_of_day', 'unknown'),
            interval_data.get('day_of_week', 0),
            1 if interval_data.get('is_weekend') else 0,
            1 if interval_data.get('is_work_hours') else 0,
            interval_data.get('session_id')
        ))
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return row_id
        
    except Exception as e:
        print(f"Error saving interval signal: {e}")
        raise


def get_intervals(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, 
                  session_id: Optional[str] = None, limit: int = 1000) -> List[dict]:
    """
    Query interval signals from the database.
    
    Args:
        start_time: Optional start time filter
        end_time: Optional end time filter
        session_id: Optional session ID filter
        limit: Maximum number of intervals to return (default 1000)
        
    Returns:
        List of interval data dictionaries
    """
    print(f"[DB] get_intervals called: session_id={session_id}, start={start_time}, end={end_time}, limit={limit}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM interval_signals WHERE 1=1"
        params = []
        
        if start_time and end_time:
            # Use proper interval overlap logic: intervals that overlap with [start_time, end_time]
            query += " AND timestamp_start < ? AND timestamp_end > ?"
            params.append(end_time.isoformat())
            params.append(start_time.isoformat())
        elif start_time:
            # Just start time filter
            query += " AND timestamp_end > ?"
            params.append(start_time.isoformat())
        elif end_time:
            # Just end time filter
            query += " AND timestamp_start < ?"
            params.append(end_time.isoformat())
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY timestamp_start DESC LIMIT ?"
        params.append(limit)
        
        print(f"[DB] Executing query: {query}")
        print(f"[DB] With params: {params}")
        cursor.execute(query, params)
        
        intervals = []
        for row in cursor.fetchall():
            intervals.append(dict(row))
        
        print(f"[DB] Found {len(intervals)} intervals")
        conn.close()
        return intervals
        
    except Exception as e:
        print(f"Error querying intervals: {e}")
        raise


def save_enriched_task(task_data: dict) -> int:
    """Save a single enriched task record."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        raw_json = json.dumps(task_data, default=str)

        cursor.execute("""
            INSERT INTO enriched_tasks (
                task_id, session_id, task_name, app, window_title,
                avg_cpu, avg_ram, avg_gpu,
                mic_on, camera_on, audio_volume,
                copy_count, paste_count,
                keyboard_intensity, mouse_activity,
                start_time, end_time, duration_minutes,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_data.get('task_id'),
            task_data.get('session_id'),
            task_data.get('task_name'),
            task_data.get('app'),
            task_data.get('window_title'),
            task_data.get('avg_cpu'),
            task_data.get('avg_ram'),
            task_data.get('avg_gpu'),
            1 if task_data.get('mic_on') else 0,
            1 if task_data.get('camera_on') else 0,
            task_data.get('audio_volume'),
            task_data.get('copy_count'),
            task_data.get('paste_count'),
            task_data.get('keyboard_intensity'),
            task_data.get('mouse_activity'),
            task_data.get('start_time'),
            task_data.get('end_time'),
            task_data.get('duration_minutes'),
            raw_json,
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    except Exception as e:
        print(f"Error saving enriched task: {e}")
        raise


def save_enriched_tasks(tasks: List[dict]) -> List[int]:
    """Save multiple enriched task records."""
    ids = []
    for task in tasks:
        ids.append(save_enriched_task(task))
    return ids


def get_enriched_tasks(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                       session_id: Optional[str] = None, limit: int = 500) -> List[dict]:
    """Query enriched task records."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM enriched_tasks WHERE 1=1"
        params = []

        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND end_time <= ?"
            params.append(end_time.isoformat())

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    except Exception as e:
        print(f"Error querying enriched tasks: {e}")
        raise


def save_analytics_snapshot(payload: dict, start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            ts: Optional[datetime] = None) -> int:
    """Persist a realtime analytics snapshot payload."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        timestamp = ts or datetime.now(timezone.utc)
        payload_str = json.dumps(payload, default=str)

        cursor.execute("""
            INSERT INTO analytics_snapshots (timestamp, start_time, end_time, payload)
            VALUES (?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            start_time.isoformat() if start_time else None,
            end_time.isoformat() if end_time else None,
            payload_str,
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    except Exception as e:
        print(f"Error saving analytics snapshot: {e}")
        raise


def get_latest_analytics_snapshot(start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Optional[dict]:
    """Get the latest analytics snapshot within an optional time range."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT timestamp, payload FROM analytics_snapshots WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT 1"

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        ts_str, payload_str = row
        try:
            payload = json.loads(payload_str) if payload_str else {}
        except Exception:
            payload = {"raw": payload_str}

        return {
            "timestamp": ts_str,
            "payload": payload,
        }

    except Exception as e:
        print(f"Error loading analytics snapshot: {e}")
        return None


def cleanup_old_intervals(days_to_keep: int = 90) -> int:
    """
    Delete interval signals older than the specified number of days.
    
    Default retention: 90 days for privacy.
    
    Args:
        days_to_keep: Number of days to keep (default 90)
        
    Returns:
        Number of intervals deleted
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()
        
        cursor.execute("""
            DELETE FROM interval_signals 
            WHERE timestamp_start < ?
        """, (cutoff_str,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        print(f"Error cleaning up old intervals: {e}")
        raise


def cleanup_old_enriched_tasks(days_to_keep: int = 90) -> int:
    """Delete enriched task records older than the specified number of days."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()

        cursor.execute("""
            DELETE FROM enriched_tasks
            WHERE COALESCE(end_time, start_time) < ?
        """, (cutoff_str,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    except Exception as e:
        print(f"Error cleaning up old enriched tasks: {e}")
        raise


def cleanup_old_analytics_snapshots(days_to_keep: int = 30) -> int:
    """Delete analytics snapshots older than the specified number of days."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()

        cursor.execute("""
            DELETE FROM analytics_snapshots
            WHERE timestamp < ?
        """, (cutoff_str,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    except Exception as e:
        print(f"Error cleaning up old analytics snapshots: {e}")
        raise


def cleanup_old_events(days_to_keep: int = 90) -> int:
    """
    Delete events older than the specified number of days.
    
    Default retention: 90 days for privacy.
    
    Args:
        days_to_keep: Number of days to keep (default 90)
        
    Returns:
        Number of events deleted
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()
        
        cursor.execute("""
            DELETE FROM events 
            WHERE timestamp < ?
        """, (cutoff_str,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        print(f"Error cleaning up old events: {e}")
        raise


def get_storage_stats() -> dict:
    """
    Get statistics about stored data.
    
    Returns:
        Dictionary with counts and storage info
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Count intervals
        cursor.execute("SELECT COUNT(*) FROM interval_signals")
        stats['interval_count'] = cursor.fetchone()[0]
        
        # Count events
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['event_count'] = cursor.fetchone()[0]
        
        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        stats['session_count'] = cursor.fetchone()[0]

        # Count enriched tasks
        cursor.execute("SELECT COUNT(*) FROM enriched_tasks")
        stats['enriched_task_count'] = cursor.fetchone()[0]

        # Count analytics snapshots
        cursor.execute("SELECT COUNT(*) FROM analytics_snapshots")
        stats['analytics_snapshot_count'] = cursor.fetchone()[0]
        
        # Oldest interval
        cursor.execute("SELECT MIN(timestamp_start) FROM interval_signals")
        oldest = cursor.fetchone()[0]
        stats['oldest_interval'] = oldest
        
        # Newest interval
        cursor.execute("SELECT MAX(timestamp_start) FROM interval_signals")
        newest = cursor.fetchone()[0]
        stats['newest_interval'] = newest
        
        # Database file size
        db_path = get_db_path()
        if os.path.exists(db_path):
            stats['db_size_bytes'] = os.path.getsize(db_path)
            stats['db_size_mb'] = round(stats['db_size_bytes'] / (1024 * 1024), 2)
        
        conn.close()
        return stats
        
    except Exception as e:
        print(f"Error getting storage stats: {e}")
        return {}


def log_live_prediction(prediction) -> int:
    """
    Log a live task prediction to the database.
    
    Args:
        prediction: LiveTaskPrediction object with attributes:
                   timestamp, session_id, task_id, confidence, distance_to_centroid,
                   reason, feature_window_seconds, feature_vector, alternative_tasks
    
    Returns:
        The row ID of the inserted record
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        feature_vec_json = None
        if prediction.feature_vector:
            try:
                feature_vec_json = json.dumps(prediction.feature_vector)
            except Exception:
                feature_vec_json = str(prediction.feature_vector)
        
        alt_tasks_json = None
        if prediction.alternative_tasks:
            try:
                alt_tasks_json = json.dumps(prediction.alternative_tasks)
            except Exception:
                alt_tasks_json = str(prediction.alternative_tasks)
        
        cursor.execute("""
            INSERT INTO live_task_predictions 
            (timestamp, session_id, task_id, confidence, distance_to_centroid, 
             reason, feature_window_seconds, feature_vector, alternative_tasks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.timestamp,
            prediction.session_id,
            prediction.task_id,
            prediction.confidence,
            prediction.distance_to_centroid,
            prediction.reason,
            prediction.feature_window_seconds,
            feature_vec_json,
            alt_tasks_json,
        ))
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return row_id
        
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(
            ComponentType.DATABASE,
            "log_live_prediction",
            e,
            ErrorSeverity.WARNING,
            session_id=prediction.session_id
        )
        return -1


def get_latest_live_prediction(session_id: str) -> Optional[dict]:
    """
    Get the most recent live task prediction for a session.
    
    Args:
        session_id: Session ID to query
    
    Returns:
        Dict with prediction data, or None if no predictions found
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, task_id, confidence, distance_to_centroid, reason,
                   feature_vector, alternative_tasks
            FROM live_task_predictions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        result = {
            'timestamp': row[0],
            'task_id': row[1],
            'confidence': row[2],
            'distance_to_centroid': row[3],
            'reason': row[4],
            'feature_vector': None,
            'alternative_tasks': None,
        }
        
        if row[5]:  # feature_vector
            try:
                result['feature_vector'] = json.loads(row[5])
            except Exception:
                result['feature_vector'] = row[5]
        
        if row[6]:  # alternative_tasks
            try:
                result['alternative_tasks'] = json.loads(row[6])
            except Exception:
                result['alternative_tasks'] = row[6]
        
        return result
        
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(
            ComponentType.DATABASE,
            "get_latest_live_prediction",
            e,
            ErrorSeverity.WARNING
        )
        return None


def get_latest_activity_task() -> Optional[dict]:
    """
    Get the most recent activity-based task prediction (not session-tied).
    
    Activity tasks are predictions from the last 60 seconds of user activity,
    independent of any session. Use session_id='activity' as a marker.
    
    Returns:
        Dict with prediction data, or None if no predictions found
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, task_id, confidence, distance_to_centroid, reason,
                   feature_vector, alternative_tasks
            FROM live_task_predictions
            WHERE session_id = 'activity'
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        result = {
            'timestamp': row[0],
            'task_id': row[1],
            'confidence': row[2],
            'distance_to_centroid': row[3],
            'reason': row[4],
            'feature_vector': None,
            'alternative_tasks': None,
        }
        
        if row[5]:  # feature_vector
            try:
                result['feature_vector'] = json.loads(row[5])
            except Exception:
                result['feature_vector'] = row[5]
        
        if row[6]:  # alternative_tasks
            try:
                result['alternative_tasks'] = json.loads(row[6])
            except Exception:
                result['alternative_tasks'] = row[6]
        
        return result
        
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(
            ComponentType.DATABASE,
            "get_latest_activity_task",
            e,
            ErrorSeverity.WARNING
        )
        return None


def get_live_predictions_for_session(session_id: str, limit: int = 10) -> List[dict]:
    """
    Get recent live task predictions for a session.
    
    Args:
        session_id: Session ID to query
        limit: Max number of predictions to return
    
    Returns:
        List of prediction dicts ordered by timestamp (newest first)
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, task_id, confidence, distance_to_centroid, reason,
                   feature_vector, alternative_tasks
            FROM live_task_predictions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            pred = {
                'timestamp': row[0],
                'task_id': row[1],
                'confidence': row[2],
                'distance_to_centroid': row[3],
                'reason': row[4],
                'feature_vector': None,
                'alternative_tasks': None,
            }
            
            if row[5]:  # feature_vector
                try:
                    pred['feature_vector'] = json.loads(row[5])
                except Exception:
                    pred['feature_vector'] = row[5]
            
            if row[6]:  # alternative_tasks
                try:
                    pred['alternative_tasks'] = json.loads(row[6])
                except Exception:
                    pred['alternative_tasks'] = row[6]
            
            result.append(pred)
        
        return result
        
    except Exception as e:
        from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
        log_component_error(
            ComponentType.DATABASE,
            "get_live_predictions_for_session",
            e,
            ErrorSeverity.WARNING
        )
        return []


def clear_old_live_predictions(days_to_keep: int = 7) -> int:
    """
    Clear live predictions older than N days.
    
    Args:
        days_to_keep: Number of days to keep (default 7)
    
    Returns:
        Number of predictions deleted
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.isoformat()
        
        cursor.execute("""
            DELETE FROM live_task_predictions 
            WHERE timestamp < ?
        """, (cutoff_str,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        print(f"Error cleaning up old live predictions: {e}")
        return 0
