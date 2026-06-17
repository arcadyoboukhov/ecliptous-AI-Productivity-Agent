"""
SessionGate: Single source of truth for tracking state authorization.

The SessionGate enforces the core invariant:
- No session is active → no tracking
- At most one active session at a time
- All signals are only collected when a session is active
"""


class SessionAlreadyActive(Exception):
    """Raised when trying to start a session while another is active."""
    pass


class NoActiveSession(Exception):
    """Raised when trying to operate on a session when none is active."""
    pass


class SessionGate:
    """
    Global gatekeeper for tracking authorization.
    
    Enforces:
    - Exactly one active session at a time
    - All signal collection is guarded by this gate
    - Explicit manual control over tracking
    """
    
    def __init__(self):
        self.active_session_id: str | None = None
    
    def is_active(self) -> bool:
        """Check if tracking is currently enabled."""
        return self.active_session_id is not None
    
    def get_active_session_id(self) -> str:
        """Get the currently active session ID, or raise NoActiveSession."""
        if self.active_session_id is None:
            raise NoActiveSession("No session is currently active")
        return self.active_session_id
    
    def start(self, session_id: str):
        """
        Start a new session. Fails if another session is already active.
        
        Args:
            session_id: The session to activate
        
        Raises:
            SessionAlreadyActive: If a session is already active
        """
        if self.active_session_id is not None:
            raise SessionAlreadyActive(
                f"Another session is already active: {self.active_session_id}. "
                f"End it before starting a new one."
            )
        self.active_session_id = session_id
    
    def stop(self):
        """
        Stop the active session (if any).
        
        Resumes the inactive state; tracking is disabled.
        """
        self.active_session_id = None


# Global singleton gate instance
_gate: SessionGate | None = None


def get_session_gate() -> SessionGate:
    """Get the global SessionGate singleton."""
    global _gate
    if _gate is None:
        _gate = SessionGate()
    return _gate
