"""
SignalBuffer: Per-session collection of input signals.

Signals are owned by sessions, not tasks. The buffer is created when a session
starts and destroyed when the session ends. Tasks only reference time ranges
within the session; signal attribution happens post-processing.

Signals collected:
- Mouse movement distance
- Mouse clicks
- Keyboard presses
- Active window/app transitions (timeline)
"""
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Deque, Tuple
from collections import deque

# Intensity presets define normalization baselines and weights.
# DEFAULT matches the legacy behavior. Other presets tweak baselines to reflect
# expected activity levels for different task types.
INTENSITY_PRESETS = {
    "DEFAULT": {
        "baselines": {
            "keys": 35.0,
            "clicks": 6.0,
            "mouse": 300.0,
            "app_changes": 1.5,
        },
        "weights": {
            "keys": 0.45,
            "clicks": 0.20,
            "mouse": 0.25,
            "app_changes": 0.10,
        },
    },

    "ACTIVE": {
        "baselines": {
            "keys": 55.0,
            "clicks": 10.0,
            "mouse": 500.0,
            "app_changes": 2.0,
        },
        "weights": {
            "keys": 0.50,
            "clicks": 0.20,
            "mouse": 0.20,
            "app_changes": 0.10,
        },
    },

    "HYBRID": {
        "baselines": {
            "keys": 30.0,
            "clicks": 6.0,
            "mouse": 350.0,
            "app_changes": 1.5,
        },
        "weights": {
            "keys": 0.40,
            "clicks": 0.20,
            "mouse": 0.30,
            "app_changes": 0.10,
        },
    },

    "PASSIVE": {
        "baselines": {
            "keys": 12.0,
            "clicks": 3.0,
            "mouse": 120.0,
            "app_changes": 0.8,
        },
        "weights": {
            "keys": 0.25,
            "clicks": 0.20,
            "mouse": 0.35,
            "app_changes": 0.20,
        },
    },
}



def _resolve_intensity_preset(name: str):
    key = (name or "DEFAULT").upper()
    return INTENSITY_PRESETS.get(key, INTENSITY_PRESETS["DEFAULT"])


@dataclass
class AppWindow:
    """Record of an app window being active."""
    app_name: str
    window_title: str
    timestamp: datetime
    duration_seconds: float = 0.0  # Filled in after next window change


@dataclass
class SignalBuffer:
    """
    Per-session signal buffer.
    
    Created on session start, destroyed on session end.
    All signals belong to the session and are indexed by timestamp.
    """
    session_id: str
    start_time: datetime
    
    # Aggregated signal counters (for legacy/export)
    mouse_distance: float = 0.0
    mouse_clicks: int = 0
    keyboard_presses: int = 0
    copy_paste_count: int = 0
    
    # Minute-bucketed timeline for persistence
    # Format: {"2024-01-28T14:35:00": {"keys": 10, "clicks": 2, "mouse": 150.5, "copy_paste": 0}}
    activity_timeline: dict = field(default_factory=dict)
    
    # Current intensity preset (changes with active task)
    current_preset: str = "DEFAULT"
    
    # Recent-event buffers for inference (timestamps)
    _recent_clicks: Deque[datetime] = field(default_factory=lambda: deque())
    _recent_keys: Deque[datetime] = field(default_factory=lambda: deque())
    # movements: deque of (timestamp, distance)
    _recent_movements: Deque[Tuple[datetime, float]] = field(default_factory=lambda: deque())
    # app_timeline remains as a timeline, but track app change timestamps separately
    app_timeline: List[AppWindow] = field(default_factory=list)
    _recent_app_changes: Deque[datetime] = field(default_factory=lambda: deque())

    def set_preset(self, preset_name: str):
        """Change the intensity preset and print notification.
        
        Args:
            preset_name: One of DEFAULT, ACTIVE, PASSIVE, HYBRID
        """
        preset_name = (preset_name or "DEFAULT").upper()
        
        # Validate preset exists
        if preset_name not in INTENSITY_PRESETS:
            preset_name = "DEFAULT"
        
        # Only print and update if actually changing
        if preset_name != self.current_preset:
            print(f"[PRESET CHANGE] {self.current_preset} → {preset_name}")
            self.current_preset = preset_name
    
    def ensure_recent_activity_baseline(self):
        """Initialize baseline - no longer seeds fake data.
        
        Kept for backward compatibility but now just ensures buffers exist.
        Real activity will be collected by hooks.
        """
        pass
    
    def record_mouse_movement(self, distance: float):
        """Record mouse movement distance."""
        now = datetime.now(timezone.utc)
        self.mouse_distance += distance
        self._recent_movements.append((now, float(distance)))
        # keep movement history bounded (10 minutes max)
        cutoff = now - timedelta(minutes=10)
        while self._recent_movements and self._recent_movements[0][0] < cutoff:
            self._recent_movements.popleft()
        
        # Update minute-bucketed timeline
        bucket = now.replace(second=0, microsecond=0).isoformat()
        if bucket not in self.activity_timeline:
            self.activity_timeline[bucket] = {"keys": 0, "clicks": 0, "mouse": 0.0, "copy_paste": 0}
        self.activity_timeline[bucket]["mouse"] += distance
    
    def record_mouse_click(self):
        """Record a mouse click."""
        now = datetime.now(timezone.utc)
        self.mouse_clicks += 1
        self._recent_clicks.append(now)
        cutoff = now - timedelta(minutes=10)
        while self._recent_clicks and self._recent_clicks[0] < cutoff:
            self._recent_clicks.popleft()
        
        # Update minute-bucketed timeline
        bucket = now.replace(second=0, microsecond=0).isoformat()
        if bucket not in self.activity_timeline:
            self.activity_timeline[bucket] = {"keys": 0, "clicks": 0, "mouse": 0.0, "copy_paste": 0}
        self.activity_timeline[bucket]["clicks"] += 1
    
    def record_keyboard_press(self):
        """Record a keyboard press."""
        now = datetime.now(timezone.utc)
        self.keyboard_presses += 1
        self._recent_keys.append(now)
        cutoff = now - timedelta(minutes=10)
        while self._recent_keys and self._recent_keys[0] < cutoff:
            self._recent_keys.popleft()
        
        # Update minute-bucketed timeline
        bucket = now.replace(second=0, microsecond=0).isoformat()
        if bucket not in self.activity_timeline:
            self.activity_timeline[bucket] = {"keys": 0, "clicks": 0, "mouse": 0.0, "copy_paste": 0}
        self.activity_timeline[bucket]["keys"] += 1
    
    def record_copy_paste(self):
        """Record a copy/paste operation (Ctrl+C, Ctrl+V, Ctrl+X)."""
        now = datetime.now(timezone.utc)
        self.copy_paste_count += 1
        
        # Update minute-bucketed timeline
        bucket = now.replace(second=0, microsecond=0).isoformat()
        if bucket not in self.activity_timeline:
            self.activity_timeline[bucket] = {"keys": 0, "clicks": 0, "mouse": 0.0, "copy_paste": 0}
        self.activity_timeline[bucket]["copy_paste"] += 1
    
    def record_app_window(self, app_name: str, window_title: str, timestamp: datetime | None = None):
        """
        Record an app window becoming active.
        
        Args:
            app_name: Name of the application
            window_title: Window title
            timestamp: When the window became active (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Close out the previous window with duration
        if self.app_timeline:
            prev = self.app_timeline[-1]
            prev.duration_seconds = (timestamp - prev.timestamp).total_seconds()
        
        # Add new window
        self.app_timeline.append(AppWindow(
            app_name=app_name,
            window_title=window_title,
            timestamp=timestamp
        ))
        # track app change timestamp for inference
        self._recent_app_changes.append(timestamp)
        cutoff = timestamp - timedelta(minutes=10)
        while self._recent_app_changes and self._recent_app_changes[0] < cutoff:
            self._recent_app_changes.popleft()
    
    def to_dict(self) -> dict:
        """Serialize the signal buffer to a dict for storage."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "mouse_distance": self.mouse_distance,
            "mouse_clicks": self.mouse_clicks,
            "keyboard_presses": self.keyboard_presses,
            "activity_timeline": self.activity_timeline,
            "app_timeline": [
                {
                    "app_name": w.app_name,
                    "window_title": w.window_title,
                    "timestamp": w.timestamp.isoformat(),
                    "duration_seconds": w.duration_seconds
                }
                for w in self.app_timeline
            ],
            # do not persist recent buffers (runtime-only)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SignalBuffer":
        """Deserialize a signal buffer from a dict."""
        buffer = cls(
            session_id=data["session_id"],
            start_time=datetime.fromisoformat(data["start_time"])
        )
        buffer.mouse_distance = data.get("mouse_distance", 0.0)
        buffer.mouse_clicks = data.get("mouse_clicks", 0)
        buffer.keyboard_presses = data.get("keyboard_presses", 0)
        buffer.activity_timeline = data.get("activity_timeline", {})
        
        for w_data in data.get("app_timeline", []):
            w = AppWindow(
                app_name=w_data["app_name"],
                window_title=w_data["window_title"],
                timestamp=datetime.fromisoformat(w_data["timestamp"]),
                duration_seconds=w_data.get("duration_seconds", 0.0)
            )
            buffer.app_timeline.append(w)
        
        return buffer

    # --- Inference helpers ---
    def _prune_since(self, dq: Deque[datetime], cutoff: datetime):
        while dq and dq[0] < cutoff:
            dq.popleft()

    def metrics_since(self, seconds: int = 300, preset: str | None = None) -> dict:
        """Return activity metrics over the last `seconds` seconds (default 5m).

        Includes counts of keys, clicks, total movement distance, app changes, and
        the last active app name. Also computes an `intensity` score (0-100)
        reflecting how intensely the machine is being used in the given window.
        Presets tweak normalization baselines for different activity expectations.
        
        Args:
            seconds: Time window to analyze
            preset: Intensity preset to use (if None, uses current_preset)
        """
        # Use current preset if not explicitly provided
        if preset is None:
            preset = self.current_preset
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=seconds)

        # Prune deques
        self._prune_since(self._recent_clicks, cutoff)
        self._prune_since(self._recent_keys, cutoff)
        self._prune_since(self._recent_app_changes, cutoff)
        # Prune movements by timestamp
        while self._recent_movements and self._recent_movements[0][0] < cutoff:
            self._recent_movements.popleft()

        clicks = len(self._recent_clicks)
        keys = len(self._recent_keys)
        app_changes = len(self._recent_app_changes)
        mouse_dist = sum(d for ts, d in self._recent_movements)

        active_app = None
        if self.app_timeline:
            active_app = self.app_timeline[-1].app_name

        # compute a raw input activity score (legacy compatibility)
        raw_score = keys + clicks + (mouse_dist / 100.0) + (app_changes * 2)

        # Resolve preset config (DEFAULT preserves legacy behavior)
        cfg = _resolve_intensity_preset(preset)
        baselines = cfg["baselines"]
        weights = cfg["weights"]

        # Compute normalized components over the window (relative to 5-minute baselines)
        window_minutes = max(1.0, seconds / 60.0)
        scale = window_minutes / 5.0

        keys_norm = min(1.0, keys / (baselines["keys"] * scale))
        clicks_norm = min(1.0, clicks / (baselines["clicks"] * scale))
        move_norm = min(1.0, mouse_dist / (baselines["mouse"] * scale))
        appchange_norm = min(1.0, app_changes / (baselines["app_changes"] * scale))

        # Weighted intensity (0.0 - 1.0)
        intensity_norm = (
            keys_norm * weights["keys"]
            + clicks_norm * weights["clicks"]
            + move_norm * weights["mouse"]
            + appchange_norm * weights["app_changes"]
        )
        intensity = round(float(intensity_norm * 100.0), 1)  # 0.0 - 100.0 (one decimal)

        # Keep a backward-compatible small confidence float [0,1]
        confidence = min(1.0, intensity_norm)

        return {
            "keys": keys,
            "clicks": clicks,
            "mouse_distance": mouse_dist,
            "app_changes": app_changes,
            "active_app": active_app,
            "input_activity_score": raw_score,
            "confidence": confidence,
            "intensity": intensity,
            "window_seconds": seconds,
            "last_ts": now.isoformat(),
            "preset": (preset or "DEFAULT").upper(),
        }
