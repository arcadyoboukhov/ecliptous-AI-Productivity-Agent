"""
Comprehensive Signal Collector - integrates all signal sources.

Collects signals from:
- User activity (keyboard, mouse, copy/paste, app/window)
- Audio/video (mic, camera, audio volume)
- System resources (CPU, RAM, GPU, Disk I/O)
- Contextual metadata (time, day, historical patterns)

All signals are normalized and timestamped before persistence.
"""

from datetime import datetime, timezone
from typing import Dict, Optional
import threading
import time


class ComprehensiveSignalCollector:
    """
    Unified collector for all productivity signals.
    
    Orchestrates multiple signal sources and provides
    normalized, timestamped signal data.
    """
    
    def __init__(self, poll_interval: float = 2.0):
        """
        Initialize the comprehensive signal collector.
        
        Args:
            poll_interval: How often to poll resource signals (seconds)
        """
        self.poll_interval = poll_interval
        self.is_collecting = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Initialize signal monitors (lazy-loaded)
        self._audio_video_monitor = None
        self._resource_monitor = None
        self._audio_volume_monitor = None
        
        # Current signal snapshot
        self._current_signals = {}
        self._last_update = None
    
    def start(self):
        """Start collecting all signals."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self._stop_event.clear()
        
        # Start audio/video monitoring
        try:
            from agent.signals.audio_video import get_audio_video_monitor
            self._audio_video_monitor = get_audio_video_monitor()
            self._audio_video_monitor.start()
        except Exception:
            pass
        
        # Start dedicated audio volume monitoring
        try:
            from agent.signals.audio_volume import get_audio_volume_monitor
            self._audio_volume_monitor = get_audio_volume_monitor()
            self._audio_volume_monitor.start()
        except Exception:
            pass
        
        # Start resource monitoring
        try:
            from agent.signals.system_resources import get_resource_monitor
            self._resource_monitor = get_resource_monitor()
            self._resource_monitor.start()
        except Exception:
            pass
        
        # Start polling thread
        self._monitor_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop collecting all signals."""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        self._stop_event.set()
        
        # Stop monitors
        if self._audio_video_monitor:
            self._audio_video_monitor.stop()
        if self._audio_volume_monitor:
            self._audio_volume_monitor.stop()
        if self._resource_monitor:
            self._resource_monitor.stop()
        
        # Wait for polling thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def get_signals(self) -> Dict:
        """
        Get current signal snapshot.
        
        Returns dict with all available signals:
            User Activity:
            - app_name, window_title (from active_window hooks)
            - keyboard_intensity, mouse_activity (from input hooks)
            - copy_paste_count (from input hooks)
            
            Audio/Video:
            - microphone_active: bool
            - camera_active: bool
            - audio_volume: float (0.0-1.0)
            
            System Resources:
            - cpu_percent: float (0-100)
            - ram_percent: float (0-100)
            - gpu_percent: float (0-100) or None
            - disk_read_mbps: float
            - disk_write_mbps: float
            
            Contextual:
            - timestamp: ISO string
            - time_of_day: str (morning/afternoon/evening/night)
            - day_of_week: int (0-6)
            - is_weekend: bool
            - is_work_hours: bool
        """
        return self._current_signals.copy()
    
    def _polling_loop(self):
        """Main polling loop to update signal snapshot."""
        while not self._stop_event.is_set():
            try:
                self._update_signal_snapshot()
            except Exception:
                pass
            
            self._stop_event.wait(self.poll_interval)
    
    def _update_signal_snapshot(self):
        """Update the current signal snapshot from all sources."""
        now = datetime.now(timezone.utc)
        signals = {}
        
        # Contextual metadata (always available)
        signals["timestamp"] = now.isoformat()
        signals["time_of_day"] = self._get_time_of_day(now.hour)
        signals["day_of_week"] = now.weekday()
        signals["is_weekend"] = now.weekday() >= 5
        signals["is_work_hours"] = 9 <= now.hour < 17
        
        # Audio/video signals
        if self._audio_video_monitor:
            try:
                av_state = self._audio_video_monitor.get_state()
                signals["microphone_active"] = av_state["microphone_active"]
                signals["camera_active"] = av_state["camera_active"]
            except Exception:
                pass
        
        # Use dedicated audio volume monitor if available (more reliable)
        if self._audio_volume_monitor:
            try:
                volume = self._audio_volume_monitor.get_current_volume()
                signals["audio_volume"] = volume
            except Exception:
                signals["audio_volume"] = 0.0
        elif self._audio_video_monitor:
            try:
                av_state = self._audio_video_monitor.get_state()
                signals["audio_volume"] = av_state.get("audio_volume", 0.0)
            except Exception:
                signals["audio_volume"] = 0.0
        else:
            signals["audio_volume"] = 0.0
        
        # System resource signals
        if self._resource_monitor:
            try:
                metrics = self._resource_monitor.get_metrics()
                signals["cpu_percent"] = metrics["cpu_percent"]
                signals["ram_percent"] = metrics["ram_percent"]
                signals["gpu_percent"] = metrics["gpu_percent"]
                signals["disk_read_mbps"] = metrics["disk_read_mbps"]
                signals["disk_write_mbps"] = metrics["disk_write_mbps"]
            except Exception:
                pass
        
        # User activity signals (from session SignalBuffer)
        # These are tracked by input hooks and stored in session.signals
        # We don't duplicate them here, just note they're available
        signals["_note"] = "keyboard, mouse, copy/paste tracked in session.signals"
        
        self._current_signals = signals
        self._last_update = now
    
    def _get_time_of_day(self, hour: int) -> str:
        """Convert hour to time of day category."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"


# Global singleton
_signal_collector = None


def get_signal_collector() -> ComprehensiveSignalCollector:
    """Get or create the global comprehensive signal collector."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = ComprehensiveSignalCollector()
    return _signal_collector


def start_comprehensive_collection():
    """Start collecting all available signals."""
    collector = get_signal_collector()
    collector.start()
    return collector


def stop_comprehensive_collection():
    """Stop collecting all signals."""
    collector = get_signal_collector()
    collector.stop()


def get_all_signals() -> Dict:
    """Get current snapshot of all signals."""
    collector = get_signal_collector()
    if not collector.is_collecting:
        collector.start()
    return collector.get_signals()


if __name__ == "__main__":
    import json
    
    print("Starting comprehensive signal collection...")
    collector = start_comprehensive_collection()
    
    print("Collecting for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        signals = collector.get_signals()
        
        # Print formatted output
        print(f"\n[{i+1}/10] Signal Snapshot:")
        print(f"  Time: {signals.get('time_of_day', 'N/A')}, Weekday: {signals.get('day_of_week', 'N/A')}")
        print(f"  Mic: {signals.get('microphone_active', False)} | Camera: {signals.get('camera_active', False)} | Volume: {signals.get('audio_volume', 0.0):.2f}")
        print(f"  CPU: {signals.get('cpu_percent', 0):.1f}% | RAM: {signals.get('ram_percent', 0):.1f}% | GPU: {signals.get('gpu_percent', 'N/A')}")
        print(f"  Disk R: {signals.get('disk_read_mbps', 0):.2f} MB/s | W: {signals.get('disk_write_mbps', 0):.2f} MB/s")
    
    print("\n\nFinal signal snapshot:")
    print(json.dumps(signals, indent=2, default=str))
    
    stop_comprehensive_collection()
    print("\nCollection stopped.")
