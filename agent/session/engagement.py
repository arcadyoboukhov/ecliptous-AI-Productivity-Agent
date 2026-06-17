"""
Engagement Detection - determines when to activate signal collection.

Signal collection becomes active only after engagement conditions are met:
- Sustained input activity (keyboard/mouse)
- Stable active window (not system processes)
- Time threshold after agent start
"""
from datetime import datetime, timedelta
from collections import deque


class EngagementDetector:
    """
    Detects user engagement to trigger signal collection.
    
    Engagement is detected when:
    1. Time threshold since agent start is met (warmup period)
    2. Sustained input activity is observed
    3. Active window is stable and not a system process
    """
    
    # System processes to ignore
    SYSTEM_PROCESSES = {
        "explorer.exe", "dwm.exe", "taskmgr.exe", "systemsettings.exe",
        "searchhost.exe", "startmenuexperiencehost.exe", "shellexperiencehost.exe"
    }
    
    def __init__(self, warmup_seconds=30, input_threshold=5, window_stability_seconds=10):
        """
        Initialize engagement detector.
        
        Args:
            warmup_seconds: Minimum seconds after agent start before engagement
            input_threshold: Minimum input events needed to detect engagement
            window_stability_seconds: Seconds window must be stable
        """
        self.start_time = datetime.now()
        self.warmup_seconds = warmup_seconds
        self.input_threshold = input_threshold
        self.window_stability_seconds = window_stability_seconds
        
        # Track input activity
        self.input_events = deque(maxlen=100)  # Recent input events
        
        # Track window stability
        self.current_window = None
        self.window_change_time = None
        
        # Engagement state
        self.is_engaged = False
        self.engagement_time = None
    
    def record_input_event(self, event_type: str):
        """Record an input event (keyboard, mouse click, mouse move)."""
        self.input_events.append((datetime.now(), event_type))
    
    def record_window_change(self, process_name: str, window_title: str):
        """Record an active window change."""
        window_id = (process_name, window_title)
        
        if window_id != self.current_window:
            self.current_window = window_id
            self.window_change_time = datetime.now()
    
    def check_engagement(self) -> bool:
        """
        Check if user is engaged based on current signals.
        
        Returns:
            True if engaged, False otherwise
        """
        now = datetime.now()
        
        # Already engaged - stay engaged
        if self.is_engaged:
            return True
        
        # Check warmup period
        if (now - self.start_time).total_seconds() < self.warmup_seconds:
            return False
        
        # Check sustained input activity
        recent_inputs = [t for t, _ in self.input_events if (now - t).total_seconds() < 30]
        if len(recent_inputs) < self.input_threshold:
            return False
        
        # Check window stability (not changing rapidly, not system process)
        if self.current_window is None:
            return False
        
        process_name, _ = self.current_window
        
        # Ignore system processes
        if process_name and process_name.lower() in self.SYSTEM_PROCESSES:
            return False
        
        # Check window has been stable
        if self.window_change_time is None:
            return False
        
        window_stable_duration = (now - self.window_change_time).total_seconds()
        if window_stable_duration < self.window_stability_seconds:
            return False
        
        # All conditions met - mark as engaged
        self.is_engaged = True
        self.engagement_time = now
        return True
    
    def is_user_engaged(self) -> bool:
        """Quick check if user is currently engaged."""
        return self.is_engaged
    
    def reset(self):
        """Reset engagement state (e.g., after long idle period)."""
        self.is_engaged = False
        self.engagement_time = None
        self.input_events.clear()
