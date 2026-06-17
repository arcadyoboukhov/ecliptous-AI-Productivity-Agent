"""
Real-Time Signal Collection Layer - Interval Aggregation

Collects all signals in configurable intervals (1-minute or 5-second default)
and produces structured interval objects with normalized signals.

Architecture:
- Individual signal monitors run continuously
- IntervalAggregator collects snapshots at interval boundaries
- Signals normalized at collection time
- Output: structured interval objects with all signals
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
import threading
import time
import re


@dataclass
class IntervalSignals:
    """
    Structured interval object with all normalized signals.
    
    One of these is created every interval (1-min or 5-sec).
    """
    # Timing
    timestamp_start: str  # UTC ISO format
    timestamp_end: str    # UTC ISO format
    
    # User Activity
    app: Optional[str] = None                # e.g., "firefox.exe"
    window_title: Optional[str] = None       # Normalized
    
    # Keyboard/Mouse/Copy-Paste
    keyboard_intensity: float = 0.0          # Keys per minute
    mouse_clicks: int = 0                    # Count
    mouse_distance: float = 0.0              # Pixels
    copy_count: int = 0                      # Ctrl+C
    paste_count: int = 0                     # Ctrl+V
    cut_count: int = 0                       # Ctrl+X
    
    # Audio/Video
    mic_active: bool = False                 # Boolean
    camera_active: bool = False              # Boolean
    audio_volume: float = 0.0                # 0.0-1.0
    
    # System Resources
    cpu_usage: float = 0.0                   # 0.0-1.0 (normalized from %)
    ram_usage: float = 0.0                   # 0.0-1.0 (normalized from %)
    gpu_usage: Optional[float] = None        # 0.0-1.0 or None if unavailable
    disk_read_mbps: float = 0.0              # MB/s
    disk_write_mbps: float = 0.0             # MB/s
    
    # Contextual
    time_of_day: str = "unknown"             # morning/afternoon/evening/night
    day_of_week: int = 0                     # 0=Monday, 6=Sunday
    is_weekend: bool = False
    is_work_hours: bool = False
    
    # Session context (optional)
    session_id: Optional[str] = None         # If available
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_normalized_dict(self) -> Dict:
        """Convert to dictionary with normalized keys (snake_case)."""
        return self.to_dict()


class IntervalAggregator:
    """
    Real-time signal aggregation in configurable intervals.
    
    Collects all signals at regular intervals and produces
    structured IntervalSignals objects.
    """
    
    def __init__(self, interval_seconds: float = 60.0, session_id_provider=None):
        """
        Initialize interval aggregator.
        
        Args:
            interval_seconds: Interval length (default 60s for 1-minute)
                            Common values: 5.0 (5-second), 60.0 (1-minute)
            session_id_provider: Optional callable to get current session_id
        """
        self.interval_seconds = interval_seconds
        self.session_id_provider = session_id_provider
        
        self.is_running = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Signal buffers for current interval
        self._keyboard_presses = 0
        self._mouse_clicks = 0
        self._mouse_distance = 0.0
        self._copy_count = 0
        self._paste_count = 0
        self._cut_count = 0
        
        self._current_app = None
        self._current_window = None
        
        # Interval history
        self.intervals: List[IntervalSignals] = []
        self._max_history = 100  # Keep last 100 intervals
        
        # Callbacks
        self._on_interval_complete = None
    
    def start(self):
        """Start collecting signals in intervals."""
        import sys
        if self.is_running:
            sys.stderr.write(f"[INTERVAL] Already running, skipping start\n")
            sys.stderr.flush()
            return
        
        sys.stderr.write(f"[INTERVAL] Starting IntervalAggregator with {self.interval_seconds}s intervals\n")
        sys.stderr.flush()
        self.is_running = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(target=self._aggregation_loop, daemon=False)  # Non-daemon for testing
        self._monitor_thread.start()
        sys.stderr.write(f"[INTERVAL] Thread started successfully\n")
        sys.stderr.flush()
    
    def stop(self):
        """Stop collecting signals."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def record_keyboard_press(self):
        """Record a keyboard press in current interval."""
        self._keyboard_presses += 1
    
    def record_mouse_click(self):
        """Record a mouse click in current interval."""
        self._mouse_clicks += 1
    
    def record_mouse_movement(self, distance: float):
        """Record mouse movement distance in current interval."""
        self._mouse_distance += distance
    
    def record_copy(self):
        """Record a copy operation (Ctrl+C)."""
        self._copy_count += 1
    
    def record_paste(self):
        """Record a paste operation (Ctrl+V)."""
        self._paste_count += 1
    
    def record_cut(self):
        """Record a cut operation (Ctrl+X)."""
        self._cut_count += 1
    
    def set_active_window(self, app_name: Optional[str], window_title: Optional[str]):
        """Set the currently active window."""
        self._current_app = app_name
        self._current_window = window_title
    
    def on_interval_complete(self, callback):
        """Set callback to be called when interval completes with IntervalSignals object."""
        self._on_interval_complete = callback
    
    def get_last_interval(self) -> Optional[IntervalSignals]:
        """Get the most recent completed interval."""
        return self.intervals[-1] if self.intervals else None
    
    def get_intervals(self, count: int = 10) -> List[IntervalSignals]:
        """Get the last N completed intervals."""
        return self.intervals[-count:]
    
    def _aggregation_loop(self):
        """Main aggregation loop that runs at interval boundaries."""
        import sys
        from pathlib import Path
        
        log_file = Path(".interval_debug.log")
        
        now_start = datetime.now(timezone.utc)
        next_interval_time = now_start + timedelta(seconds=self.interval_seconds)
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"[{now_start.isoformat()}] Loop started, first interval at {next_interval_time.isoformat()}\n")
        
        sys.stderr.write(f"[INTERVAL] Loop started at {now_start.isoformat()}, first interval at {next_interval_time.isoformat()}\n")
        sys.stderr.flush()
        
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            wait_time = (next_interval_time - now).total_seconds()
            
            if wait_time > 0:
                self._stop_event.wait(wait_time)
            
            if not self.is_running:
                break
            
            # Collect and aggregate signals for this interval
            try:
                import sys
                sys.stderr.write(f"[INTERVAL] Collecting interval ending at {next_interval_time.isoformat()}\n")
                sys.stderr.flush()
                interval = self._collect_interval(next_interval_time)
                self.intervals.append(interval)
                
                # Trim history
                if len(self.intervals) > self._max_history:
                    self.intervals.pop(0)
                
                # Persist to database (90-day retention)
                try:
                    import sys
                    from agent.storage.db import save_interval_signal
                    from pathlib import Path
                    log_file = Path(".interval_debug.log")
                    
                    interval_dict = interval.to_dict()
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now(timezone.utc).isoformat()}] Saving interval: session_id={interval_dict.get('session_id')}, app={interval_dict.get('app')}\n")
                    
                    row_id = save_interval_signal(interval_dict)
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now(timezone.utc).isoformat()}] Saved successfully, row_id={row_id}\n")
                except Exception as e:
                    from pathlib import Path
                    log_file = Path(".interval_debug.log")
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now(timezone.utc).isoformat()}] ERROR saving interval: {e}\n")
                    log_file.write_text(log_file.read_text() + msg if log_file.exists() else msg, encoding='utf-8')
                
                # Call callback if registered
                if self._on_interval_complete:
                    try:
                        self._on_interval_complete(interval)
                    except Exception:
                        pass
                
                # Reset buffers for next interval
                self._reset_buffers()
                
            except Exception as e:
                import sys
                sys.stderr.write(f"[INTERVAL] ERROR in aggregation loop: {e}\n")
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            
            # Schedule next interval
            next_interval_time += timedelta(seconds=self.interval_seconds)
    
    def _collect_interval(self, interval_end_time: datetime) -> IntervalSignals:
        """Collect and aggregate all signals for current interval."""
        interval_start_time = interval_end_time - timedelta(seconds=self.interval_seconds)
        
        # Get all current signals
        from agent.signals.collector import get_all_signals
        
        signals = get_all_signals()
        
        # Get audio volume from dedicated monitor (more reliable)
        try:
            from agent.signals.audio_volume import get_audio_volume_monitor
            audio_monitor = get_audio_volume_monitor()
            if not audio_monitor.is_monitoring:
                audio_monitor.start()
            audio_volume = audio_monitor.get_current_volume()
        except Exception:
            audio_volume = signals.get("audio_volume", 0.0)
        
        # Normalize keyboard intensity (keys per minute)
        duration_minutes = self.interval_seconds / 60.0
        keyboard_intensity = self._keyboard_presses / duration_minutes if duration_minutes > 0 else 0.0
        
        # Normalize resource usage (convert % to 0-1)
        cpu_usage = signals.get("cpu_percent", 0.0) / 100.0
        ram_usage = signals.get("ram_percent", 0.0) / 100.0
        gpu_usage = signals.get("gpu_percent")
        if gpu_usage is not None:
            gpu_usage = gpu_usage / 100.0
        
        # Normalize window title
        normalized_title = self._normalize_window_title(self._current_window) if self._current_window else None
        
        # Get session ID if provider available
        session_id = None
        from pathlib import Path
        log_file = Path(".interval_debug.log")
        
        if self.session_id_provider:
            try:
                session_id = self.session_id_provider()
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now(timezone.utc).isoformat()}] session_id_provider returned: {session_id}\n")
            except Exception as e:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now(timezone.utc).isoformat()}] ERROR session_id_provider: {e}\n")
        else:
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"[{datetime.now(timezone.utc).isoformat()}] No session_id_provider configured\n")
        
        
        # Create interval object
        interval = IntervalSignals(
            timestamp_start=interval_start_time.isoformat(),
            timestamp_end=interval_end_time.isoformat(),
            
            # User Activity
            app=self._current_app,
            window_title=normalized_title,
            
            # Keyboard/Mouse/Copy-Paste
            keyboard_intensity=keyboard_intensity,
            mouse_clicks=self._mouse_clicks,
            mouse_distance=self._mouse_distance,
            copy_count=self._copy_count,
            paste_count=self._paste_count,
            cut_count=self._cut_count,
            
            # Audio/Video
            mic_active=signals.get("microphone_active", False),
            camera_active=signals.get("camera_active", False),
            audio_volume=audio_volume,  # Use dedicated monitor result
            
            # System Resources
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
            gpu_usage=gpu_usage,
            disk_read_mbps=signals.get("disk_read_mbps", 0.0),
            disk_write_mbps=signals.get("disk_write_mbps", 0.0),
            
            # Contextual
            time_of_day=signals.get("time_of_day", "unknown"),
            day_of_week=signals.get("day_of_week", 0),
            is_weekend=signals.get("is_weekend", False),
            is_work_hours=signals.get("is_work_hours", False),
            
            # Session
            session_id=session_id
        )
        
        return interval
    
    def _reset_buffers(self):
        """Reset signal buffers for next interval."""
        self._keyboard_presses = 0
        self._mouse_clicks = 0
        self._mouse_distance = 0.0
        self._copy_count = 0
        self._paste_count = 0
        self._cut_count = 0
    
    def _normalize_window_title(self, title: str) -> str:
        """
        Normalize window title.
        
        - Lowercase
        - Strip common prefixes (browser tabs, etc.)
        - Remove special characters
        """
        if not title:
            return ""
        
        # Lowercase
        title = title.lower()
        
        # Strip browser prefixes
        prefixes = [
            "— mozilla firefox",
            "- google chrome",
            "- microsoft edge",
            "- safari",
            "[",  # Discord, Slack prefixes
        ]
        for prefix in prefixes:
            if prefix in title:
                title = title.replace(prefix, "").strip()
        
        # Remove special characters but keep spaces
        title = re.sub(r'[^a-z0-9\s]', '', title)
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        return title


# Global singleton
_interval_aggregator = None


def get_interval_aggregator(interval_seconds: float = 60.0, session_id_provider=None) -> IntervalAggregator:
    """Get or create the global interval aggregator."""
    global _interval_aggregator
    if _interval_aggregator is None:
        _interval_aggregator = IntervalAggregator(interval_seconds, session_id_provider=session_id_provider)
    elif session_id_provider is not None:
        _interval_aggregator.session_id_provider = session_id_provider
    return _interval_aggregator


def start_interval_collection(interval_seconds: float = 60.0, on_interval_callback=None, session_id_provider=None) -> IntervalAggregator:
    """
    Start real-time interval-based signal collection.
    
    Args:
        interval_seconds: Interval length (default 60s)
        on_interval_callback: Optional callback(IntervalSignals) called when interval completes
    
    Returns:
        IntervalAggregator instance
    """
    aggregator = get_interval_aggregator(interval_seconds, session_id_provider=session_id_provider)
    
    if on_interval_callback:
        aggregator.on_interval_complete(on_interval_callback)
    
    aggregator.start()
    return aggregator


if __name__ == "__main__":
    import json
    
    print("="*80)
    print("INTERVAL SIGNAL AGGREGATION - TEST")
    print("="*80 + "\n")
    
    # Start collection with 5-second intervals for testing
    print("Starting 5-second interval collection...")
    
    def on_interval(interval: IntervalSignals):
        print(f"\n[INTERVAL] {interval.timestamp_start[:19]} → {interval.timestamp_end[:19]}")
        print(f"  App: {interval.app} | Window: {interval.window_title}")
        print(f"  Keyboard: {interval.keyboard_intensity:.1f} keys/min | Mouse: {interval.mouse_clicks} clicks, {interval.mouse_distance:.0f}px")
        print(f"  Copy: {interval.copy_count} | Paste: {interval.paste_count} | Cut: {interval.cut_count}")
        print(f"  Mic: {interval.mic_active} | Camera: {interval.camera_active} | Audio vol: {interval.audio_volume:.2f}")
        print(f"  CPU: {interval.cpu_usage*100:.1f}% | RAM: {interval.ram_usage*100:.1f}% | GPU: {interval.gpu_usage*100 if interval.gpu_usage else 'N/A'}")
        print(f"  Time: {interval.time_of_day} | Weekday: {interval.day_of_week} | Work hours: {interval.is_work_hours}")
    
    aggregator = start_interval_collection(interval_seconds=5.0, on_interval_callback=on_interval)
    
    print("Collecting signals for 30 seconds...\n")
    
    try:
        # Simulate some signal activity
        for i in range(6):  # 6 * 5 seconds = 30 seconds
            time.sleep(5)
            
            # Simulate some input
            if i % 2 == 0:
                aggregator.record_keyboard_press()
                aggregator.record_keyboard_press()
            
            aggregator.record_mouse_click()
            aggregator.record_mouse_movement(50)
            
            if i == 2:
                aggregator.record_copy()
            
            aggregator.set_active_window("firefox.exe", "ChatGPT - OpenAI Browser")
    
    except KeyboardInterrupt:
        pass
    finally:
        aggregator.stop()
    
    # Print summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80 + "\n")
    
    intervals = aggregator.get_intervals(10)
    print(f"Collected {len(intervals)} intervals\n")
    
    for interval in intervals:
        print(f"Interval: {interval.timestamp_start[:19]}")
        print(f"  Data: {json.dumps(interval.to_dict(), indent=2, default=str)}\n")
