"""
Signal tracking: input events and OS signals attributed to tasks or sessions.

Signals are ALWAYS collected while a session is open.
They are attributed to:
  - Active task (if one exists), OR
  - Session (task_id = None)

No signal creates or ends sessions/tasks.
"""
from datetime import datetime
from typing import Optional
import uuid


class SignalWindow:
    """
    Aggregated signal data over a time window.
    
    Attributes:
      id: Unique identifier
      session_id: Parent session (required)
      task_id: Parent task (optional, None = unattributed)
      start_time: Window start
      end_time: Window end
      keys: Total keyboard events
      clicks: Total mouse clicks
      mouse_distance: Total mouse pixels moved
      active_app: Last observed active app name
    """
    
    def __init__(
        self,
        signal_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        keys: int = 0,
        clicks: int = 0,
        mouse_distance: int = 0,
        active_app: str | None = None,
    ):
        self.id = signal_id or str(uuid.uuid4())
        self.session_id = session_id or ""  # FK to session
        self.task_id = task_id  # Optional FK to task
        self.start_time = start_time
        self.end_time = end_time
        self.keys = keys
        self.clicks = clicks
        self.mouse_distance = mouse_distance
        self.active_app = active_app
    
    def has_activity(self) -> bool:
        """True if any signal was recorded."""
        return self.keys > 0 or self.clicks > 0 or self.mouse_distance > 0


class SignalCollector:
    """
    Accumulates OS signals during a session and batches them into windows.
    
    Always active when a session is active.
    Routes signals to active task if one exists, else to session.
    """
    
    def __init__(self):
        self.current_window: SignalWindow | None = None
        self.windows: list[SignalWindow] = []
    
    def start_window(self, session_id: str, task_id: str | None, start_time: datetime) -> SignalWindow:
        """Begin a new signal window (e.g., on task change or session start)."""
        if self.current_window and not self.current_window.end_time:
            # Close previous window
            self.current_window.end_time = start_time
            self.windows.append(self.current_window)
        
        self.current_window = SignalWindow(
            session_id=session_id,
            task_id=task_id,
            start_time=start_time,
        )
        return self.current_window
    
    def record_key(self) -> None:
        """Record a keyboard event."""
        if self.current_window:
            self.current_window.keys += 1
    
    def record_click(self) -> None:
        """Record a mouse click."""
        if self.current_window:
            self.current_window.clicks += 1
    
    def record_mouse_movement(self, distance: int) -> None:
        """Record mouse movement (pixels)."""
        if self.current_window:
            self.current_window.mouse_distance += distance
    
    def record_active_app(self, app_name: str) -> None:
        """Record the currently active application."""
        if self.current_window:
            self.current_window.active_app = app_name
    
    def close_window(self, end_time: datetime) -> SignalWindow | None:
        """Close current window and return it."""
        if not self.current_window:
            return None
        
        self.current_window.end_time = end_time
        closed = self.current_window
        self.windows.append(closed)
        self.current_window = None
        return closed
    
    def finalize(self, end_time: datetime) -> list[SignalWindow]:
        """Close all windows and return list of all collected windows."""
        self.close_window(end_time)
        return self.windows.copy()
