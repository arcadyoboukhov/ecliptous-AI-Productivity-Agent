"""
Centralized logging module with proper session_id support.
This avoids any import caching issues by being in a separate module.
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json as _json

from agent.storage.db import get_connection


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> int:
    """
    Log an event to the database.
    
    Args:
        event_type: The type of event
        payload: Optional payload dict
        session_id: Optional session ID
    
    Returns:
        The row ID of the inserted event
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Convert dict payloads to JSON
    if payload is not None and not isinstance(payload, str):
        payload_str = _json.dumps(payload)
    else:
        payload_str = payload
    
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO events (timestamp, event_type, session_id, payload) VALUES (?, ?, ?, ?)",
        (ts, event_type, session_id, payload_str)
    )
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return row_id
