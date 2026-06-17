"""
Session lifecycle utilities for agent startup/shutdown.

Handles:
- Ending all active sessions on agent start
- Starting a new session on agent start
- Ending all sessions on agent stop
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add parent directory to path to import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def end_all_active_sessions(reason: str = "agent_lifecycle"):
    """
    End all currently active sessions.
    
    Args:
        reason: Reason for ending sessions (for logging)
    
    Returns:
        int: Number of sessions ended
    """
    try:
        from agent.session.sessionizer import SessionManager
        
        # Initialize session manager
        session_manager = SessionManager(idle_threshold_seconds=300)
        
        ended_count = 0
        now = datetime.now(timezone.utc)
        
        # End current active session if any
        if session_manager.current_session:
            try:
                session_id = session_manager.current_session.id or session_manager.current_session.session_id
                session_manager.end_session(session_id)
                ended_count += 1
                print(f"[LIFECYCLE] Ended active session: {session_id}", flush=True)
            except Exception as e:
                print(f"[LIFECYCLE] Error ending active session: {e}", flush=True)
        
        # Also check for any sessions that are started but not properly ended
        for session_id, session in session_manager.sessions.items():
            try:
                # Skip if already ended
                if session.ended_at is not None:
                    continue
                    
                # If session is started (has started_at), end it
                if hasattr(session, 'started_at') and session.started_at is not None:
                    session_manager.end_session(session_id)
                    ended_count += 1
                    print(f"[LIFECYCLE] Ended lingering session: {session_id}", flush=True)
            except Exception as e:
                print(f"[LIFECYCLE] Error ending session {session_id}: {e}", flush=True)
        
        return ended_count
        
    except Exception as e:
        print(f"[LIFECYCLE] Error in end_all_active_sessions: {e}", flush=True)
        return 0


def start_new_session(name: str = None) -> str:
    """
    Start a new session.
    
    Args:
        name: Optional session name. If not provided, generates default name.
    
    Returns:
        str: Session ID of the newly created session
    """
    try:
        from agent.session.sessionizer import SessionManager
        
        # Initialize session manager
        session_manager = SessionManager(idle_threshold_seconds=300)
        
        # Generate session name if not provided
        if name is None:
            now = datetime.now(timezone.utc)
            name = f"Agent Session {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create new session
        session = session_manager.create_session(name=name)
        
        # Start the session
        started_session = session_manager.start_session(session.session_id)
        
        session_id = started_session.session_id
        print(f"[LIFECYCLE] Started new session: {session_id} ({name})", flush=True)
        
        return session_id
        
    except Exception as e:
        print(f"[LIFECYCLE] Error starting new session: {e}", flush=True)
        raise


def on_agent_start():
    """
    Called when agent starts.
    Ends all active sessions and starts a new one.
    
    Returns:
        str: Session ID of the newly created session
    """
    print("[LIFECYCLE] Agent starting - cleaning up sessions...", flush=True)
    
    # End all active sessions
    ended_count = end_all_active_sessions(reason="agent_start")
    if ended_count > 0:
        print(f"[LIFECYCLE] Ended {ended_count} session(s)", flush=True)
    
    # Start a new session
    session_id = start_new_session()
    
    print(f"[LIFECYCLE] Agent started with session: {session_id}", flush=True)
    return session_id


def on_agent_stop():
    """
    Called when agent stops.
    Ends all active sessions.
    
    Returns:
        int: Number of sessions ended
    """
    print("[LIFECYCLE] Agent stopping - ending sessions...", flush=True)
    
    # End all active sessions
    ended_count = end_all_active_sessions(reason="agent_stop")
    
    if ended_count > 0:
        print(f"[LIFECYCLE] Ended {ended_count} session(s)", flush=True)
    else:
        print("[LIFECYCLE] No active sessions to end", flush=True)
    
    return ended_count
