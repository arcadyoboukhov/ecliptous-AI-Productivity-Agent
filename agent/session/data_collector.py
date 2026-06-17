"""
Independent Data Collector - converts raw signals to normalized events.

ARCHITECTURE:
  1. Raw signals (keyboard, mouse, window) arrive from OS hooks (ephemeral, in-memory)
  2. DataCollector normalizes them into schema-compliant events
  3. Only normalized events are persisted to database
  
Raw signals NEVER written directly to disk - this is a hard rule.
"""
from datetime import datetime, timezone
from typing import Dict, Any
from collections import deque


class DataCollector:
    """
    Converts raw signals to normalized events.
    
    Raw signals (ephemeral in-memory) -> Normalized events -> Database persistence
    
    Raw signals exist only in memory and are processed immediately.
    Only normalized events may be persisted.
    
    DECOUPLED SESSION INTERACTION:
    - Data collection operates independently of session state
    - Events can be associated with a session IF one is provided
    - Events can exist as background/unassigned (session_id=None)
    - Later, background events can be inferred into sessions
    """
    
    def __init__(self, db_logger, session_id_provider=None, activity_callback=None):
        """
        Initialize data collector.
        
        Args:
            db_logger: Function to log normalized events to database (must accept session_id parameter)
            session_id_provider: Optional callable that returns current session_id (or None for background)
            activity_callback: Optional callable(keys, clicks, mouse_distance) to notify of activity
        """
        self.log_event = db_logger
        self.session_id_provider = session_id_provider  # Callable to get current session_id
        self.activity_callback = activity_callback  # Callable to notify of activity
        self.is_collecting = False
        
        # Collection metadata
        self.collection_start_time = None
        self.normalized_events_persisted = 0
        self.background_events_persisted = 0  # Track unassigned events
        
        # Raw signal buffers (ephemeral, in-memory only)
        # These are never written to disk directly
        self._raw_keyboard_buffer = deque(maxlen=100)  # Recent keyboard timestamps
        self._raw_mouse_buffer = deque(maxlen=100)    # Recent mouse events
        self._last_window_state = None                # Track for change detection
        self._last_idle_state = False                 # Track idle transitions
        # Latest sampled device state (sampled at raw-hook time)
        self._last_device_state = None  # Tuple (timestamp, state_dict)
    
    def set_session_id_provider(self, provider):
        """Set or update the session_id provider (callable that returns current session_id or None)."""
        self.session_id_provider = provider
    
    def _get_current_session_id(self):
        """Get current session_id, or None if no session active (background)."""
        if self.session_id_provider is None:
            return None
        try:
            return self.session_id_provider()
        except Exception:
            return None
    
    def start_collection(self):
        """Activate data collection and persist normalized COLLECTION_START event."""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_start_time = datetime.now(timezone.utc)
            
            # Persist normalized event (NOT a raw signal)
            # Note: Can be associated with a session if one is active, or background if not
            session_id = self._get_current_session_id()
            self.log_event("COLLECTION_START", {
                "timestamp": self.collection_start_time.isoformat(),
                "source": "engagement_detector"
            }, session_id=session_id)
            self.normalized_events_persisted += 1
            if session_id is None:
                self.background_events_persisted += 1
    
    def stop_collection(self):
        """Deactivate collection and persist normalized COLLECTION_STOP event."""
        if self.is_collecting:
            self.is_collecting = False
            now = datetime.now(timezone.utc)
            
            # Persist normalized event (NOT a raw signal)
            session_id = self._get_current_session_id()
            self.log_event("COLLECTION_STOP", {
                "timestamp": now.isoformat(),
                "normalized_events_persisted": self.normalized_events_persisted,
                "background_events": self.background_events_persisted,
                "collection_duration_seconds": (now - self.collection_start_time).total_seconds()
            }, session_id=session_id)
            if session_id is None:
                self.background_events_persisted += 1
    
    def is_active(self) -> bool:
        """Check if collector is active."""
        return self.is_collecting
    
    def collect_keyboard_event(self):
        """
        Accept raw keyboard input and normalize to INPUT_ACTIVITY event.
        
        FLOW:
          Raw signal (keyboard press timestamp) -> memory buffer -> normalized INPUT_ACTIVITY -> persist
        """
        if not self.is_collecting:
            return
        
        # RAW SIGNAL: Sample device APIs at hook time to capture mic/cam state
        # aligned with the raw input signal. This snapshot is stored in-memory
        # and later attached to the normalized event emitted by _normalize_and_persist_input_activity.
        try:
            self._sample_device_state()
        except Exception:
            pass

        # RAW SIGNAL: Store timestamp in ephemeral buffer (never persisted directly)
        ts = datetime.now(timezone.utc)
        self._raw_keyboard_buffer.append(ts)
        
        # Notify session manager of activity
        if self.activity_callback:
            self.activity_callback(keys=1, clicks=0, mouse_distance=0)
        
        # NORMALIZATION: Compute input activity from recent signals
        self._normalize_and_persist_input_activity()
    
    def collect_mouse_click(self, x: int, y: int, button: str):
        """
        Accept raw mouse click and normalize to INPUT_ACTIVITY event.
        
        FLOW:
          Raw signal (click coords, button) -> memory buffer -> normalized INPUT_ACTIVITY -> persist
        """
        if not self.is_collecting:
            return
        
        # RAW SIGNAL: Sample device APIs at hook time to capture mic/cam state
        try:
            self._sample_device_state()
        except Exception:
            pass

        # RAW SIGNAL: Store in ephemeral buffer (never persisted directly)
        ts = datetime.now(timezone.utc)
        self._raw_mouse_buffer.append({
            "timestamp": ts,
            "type": "click",
            "x": x,
            "y": y,
            "button": button
        })
        
        # Notify session manager of activity
        if self.activity_callback:
            self.activity_callback(keys=0, clicks=1, mouse_distance=0)
        
        # NORMALIZATION: Compute input activity from recent signals
        self._normalize_and_persist_input_activity()
    
    def collect_mouse_move(self, x: int, y: int, distance: float):
        """
        Accept raw mouse movement and normalize to INPUT_ACTIVITY event.
        
        FLOW:
          Raw signal (move distance delta) -> memory buffer -> normalized INPUT_ACTIVITY -> persist
        """
        if not self.is_collecting:
            return
        
        # Only process significant movements (raw signal filter)
        if distance <= 50:
            return
        
        # RAW SIGNAL: Store in ephemeral buffer (never persisted directly)
        try:
            self._sample_device_state()
        except Exception:
            pass

        ts = datetime.now(timezone.utc)
        self._raw_mouse_buffer.append({
            "timestamp": ts,
            "type": "move",
            "distance": distance
        })
        
        # Notify session manager of activity
        if self.activity_callback:
            self.activity_callback(keys=0, clicks=0, mouse_distance=distance)
        
        # NORMALIZATION: Compute input activity from recent signals
        self._normalize_and_persist_input_activity()
    
    def _normalize_and_persist_input_activity(self):
        """
        Normalize accumulated raw input signals into a single INPUT_ACTIVITY event.
        
        NORMALIZATION PHASE:
          - Strips raw coordinates/details
          - Reduces to activity count and time delta
          - Assigns single normalized timestamp
          - Enforces schema
          
        This batches multiple raw signals into a single normalized event
        to avoid 1:1 persistence (which would defeat the purpose of normalization).
        """
        now = datetime.now(timezone.utc)
        
        # Count recent input events (keyboard + mouse) in a rolling window
        # Use a longer window to allow batching of multiple raw signals
        recent_keyboard = [t for t in self._raw_keyboard_buffer if (now - t).total_seconds() < 5]
        recent_mouse = [e for e in self._raw_mouse_buffer if (now - e["timestamp"]).total_seconds() < 5]
        
        total_input_count = len(recent_keyboard) + len(recent_mouse)
        
        # Only persist if we have accumulated a reasonable batch (at least 2 events)
        # This prevents 1:1 mapping between raw signals and persisted events
        if total_input_count < 2:
            return
        
        # NORMALIZED EVENT: Aggregate into schema-compliant structure
        # Key insight: we're persisting a count/summary, not individual raw events
        session_id = self._get_current_session_id()
        self.log_event("INPUT_ACTIVITY", {
            "timestamp": now.isoformat(),
            "input_count": total_input_count,
            "keyboard_events": len(recent_keyboard),
            "mouse_events": len(recent_mouse),
            "time_window_seconds": 5,
            "source": "input_hooks"
        }, session_id=session_id)
        self.normalized_events_persisted += 1
        if session_id is None:
            self.background_events_persisted += 1
        
        # Clear the buffers after normalization to avoid double-counting
        self._raw_keyboard_buffer.clear()
        self._raw_mouse_buffer.clear()
        
        # Additionally sample OS device APIs (audio/video state, battery) at the
        # same time we persist an input activity summary. This ensures device
        # metadata is captured temporally aligned with keyboard/mouse activity
        # and is persisted with the same session association when available.
        try:
            # Prefer device snapshot sampled at hook time for best alignment.
            device_state = None
            if self._last_device_state and (now - self._last_device_state[0]).total_seconds() <= 10:
                device_state = self._last_device_state[1]
            else:
                # Lazy import and fallback to on-demand sampling
                try:
                    from agent.signals.audio_video import get_audio_video_state
                    device_state = get_audio_video_state()
                except Exception:
                    device_state = None

            # Battery information via psutil when available
            battery_info = None
            try:
                import psutil
                if hasattr(psutil, "sensors_battery"):
                    batt = psutil.sensors_battery()
                    if batt is not None:
                        battery_info = {"percent": float(batt.percent), "power_plugged": bool(batt.power_plugged)}
            except Exception:
                battery_info = None

            # Compose device payload and persist as DEVICE_STATE
            device_payload = {
                "timestamp": now.isoformat(),
                "audio_video": device_state,
                "battery": battery_info,
                "source": "device_api_on_input"
            }
            self.log_event("DEVICE_STATE", device_payload, session_id=session_id)
            self.normalized_events_persisted += 1
            if session_id is None:
                self.background_events_persisted += 1
        except Exception:
            # Non-fatal: device sampling should never break input processing
            pass

    def _sample_device_state(self):
        """Sample lightweight device APIs and cache the result with timestamp.

        This method is intended to be called from raw-hook handlers (fast path).
        It performs lazy imports and keeps the snapshot in-memory for later
        normalization.
        """
        try:
            from agent.signals.audio_video import get_audio_video_state
        except Exception:
            get_audio_video_state = None

        state = None
        if get_audio_video_state:
            try:
                state = get_audio_video_state()
            except Exception:
                state = None

        # Battery sampling is slightly heavier; include if available but don't fail
        battery_info = None
        try:
            import psutil
            if hasattr(psutil, "sensors_battery"):
                batt = psutil.sensors_battery()
                if batt is not None:
                    battery_info = {"percent": float(batt.percent), "power_plugged": bool(batt.power_plugged)}
        except Exception:
            battery_info = None

        now = datetime.now(timezone.utc)
        device_snapshot = {"audio_video": state, "battery": battery_info, "sampled_at": now.isoformat()}
        self._last_device_state = (now, device_snapshot)
    
    def collect_window_change(self, process_name: str, window_title: str):
        """
        Accept raw window change and normalize to CONTEXT_SWITCH event.
        
        FLOW:
          Raw signal (window title, PID) -> detect change -> normalized CONTEXT_SWITCH -> persist
        """
        if not self.is_collecting:
            return
        
        # RAW SIGNAL: Detect window state transition (ephemeral comparison)
        current_window = (process_name, window_title)
        if current_window == self._last_window_state:
            return  # No actual change
        
        self._last_window_state = current_window
        
        # NORMALIZED EVENT: Persist normalized context switch
        session_id = self._get_current_session_id()
        self.log_event("CONTEXT_SWITCH", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process": process_name,
            "window_title": window_title,
            "source": "active_window_detector"
        }, session_id=session_id)
        self.normalized_events_persisted += 1
        if session_id is None:
            self.background_events_persisted += 1
    
    def collect_idle_start(self, idle_seconds: float):
        """
        Accept raw idle detection and normalize to IDLE_ENTER event.
        
        FLOW:
          Raw signal (idle duration threshold exceeded) -> detect transition -> normalized IDLE_ENTER -> persist
        """
        if not self.is_collecting:
            return
        
        # RAW SIGNAL: Track state transition (ephemeral)
        if self._last_idle_state:
            return  # Already idle
        
        self._last_idle_state = True
        
        # NORMALIZED EVENT: Persist idle entrance
        session_id = self._get_current_session_id()
        self.log_event("IDLE_ENTER", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold_seconds": idle_seconds,
            "source": "idle_detector"
        }, session_id=session_id)
        self.normalized_events_persisted += 1
        if session_id is None:
            self.background_events_persisted += 1
    
    def collect_idle_end(self, idle_seconds: float):
        """
        Accept raw idle recovery and normalize to IDLE_EXIT event.
        
        FLOW:
          Raw signal (idle duration below threshold) -> detect transition -> normalized IDLE_EXIT -> persist
        """
        if not self.is_collecting:
            return
        
        # RAW SIGNAL: Track state transition (ephemeral)
        if not self._last_idle_state:
            return  # Not idle
        
        self._last_idle_state = False
        
        # NORMALIZED EVENT: Persist idle exit
        session_id = self._get_current_session_id()
        self.log_event("IDLE_EXIT", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "idle_duration_seconds": idle_seconds,
            "source": "idle_detector"
        }, session_id=session_id)
        self.normalized_events_persisted += 1
        if session_id is None:
            self.background_events_persisted += 1
