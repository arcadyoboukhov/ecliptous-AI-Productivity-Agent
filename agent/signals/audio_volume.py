"""
Audio Volume Monitor - Multiple Implementation Approaches

Primary: Session enumeration using pycaw
Fallback: Audio activity detection via process monitoring
"""

import threading
import time
from datetime import datetime, timezone
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class AudioVolumeMonitor:
    """
    Monitor system audio volume by detecting audio sessions and process activity.
    This detects when audio is being produced on the system.
    """
    
    def __init__(self, poll_interval: float = 0.5):
        """
        Initialize audio volume monitor.
        
        Args:
            poll_interval: How often to check audio (seconds)
        """
        self.poll_interval = poll_interval
        self.is_monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._current_volume = 0.0
        self._last_update = None
        
        # Try to initialize pycaw
        self._use_pycaw = False
        self.AudioUtilities = None
        try:
            from pycaw.pycaw import AudioUtilities
            self.AudioUtilities = AudioUtilities
            self._use_pycaw = True
            logger.debug("AudioVolumeMonitor: pycaw available")
        except ImportError:
            logger.debug("AudioVolumeMonitor: pycaw not available, will use fallback")
    
    def start(self):
        """Start monitoring audio volume."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("AudioVolumeMonitor started")
    
    def stop(self):
        """Stop monitoring audio volume."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        
        logger.debug("AudioVolumeMonitor stopped")
    
    def get_current_volume(self) -> float:
        """
        Get current system audio volume.
        
        Returns:
            Volume level 0.0-1.0, or 0.0 if no audio detected
        """
        return self._current_volume
    
    def _monitoring_loop(self):
        """Main loop to monitor audio volume."""
        # CRITICAL: Initialize COM on this thread
        try:
            import ctypes
            ole32 = ctypes.windll.ole32
            
            # CoInitializeEx(NULL, COINIT_MULTITHREADED)
            hr = ole32.CoInitializeEx(None, 0)
            if hr < 0 and hr != -2147221008:  # S_FALSE = already initialized
                logger.debug(f"CoInitializeEx failed: {hr}")
                return
            
            com_initialized = True
        except Exception as e:
            logger.debug(f"Error initializing COM: {e}")
            com_initialized = False
        
        try:
            while not self._stop_event.is_set():
                try:
                    if self._use_pycaw and self.AudioUtilities:
                        # Try master volume first (more accurate)
                        volume = self._get_master_volume()
                        # If master volume not available, fall back to sessions
                        if volume == 0.0:
                            volume = self._get_volume_from_sessions()
                    else:
                        volume = self._get_volume_from_processes()
                    
                    self._current_volume = max(0.0, min(1.0, volume))
                    self._last_update = datetime.now(timezone.utc)
                except Exception as e:
                    logger.debug(f"Error updating volume: {e}")
                    self._current_volume = 0.0
                
                self._stop_event.wait(self.poll_interval)
        finally:
            # Clean up COM
            if com_initialized:
                try:
                    ole32.CoUninitialize()
                except:
                    pass
    
    def _get_master_volume(self) -> float:
        """
        Get the system master volume (device volume, not application volume).
        This is what the user sees in the volume mixer / system tray.
        
        Returns:
            Master volume level 0.0-1.0, or 0.0 if unable to get
        """
        try:
            # Get the default audio device (speakers)
            devices = self.AudioUtilities.GetSpeakers()
            
            # Use EndpointVolume property
            endpoint_volume = devices.EndpointVolume
            
            if endpoint_volume is not None:
                # Get master volume level scalar (0.0-1.0)
                level = endpoint_volume.GetMasterVolumeLevelScalar()
                
                if level is not None:
                    return float(level)
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting master volume: {e}")
            return 0.0
    
    def _get_volume_from_sessions(self) -> float:
        """
        Get volume by checking all active audio sessions.
        Returns max volume from any session producing audio.
        """
        try:
            # GetAllSessions returns the sessions from the default audio device
            sessions = self.AudioUtilities.GetAllSessions()
            
            max_volume = 0.0
            for session in sessions:
                try:
                    # Get the simple audio volume interface
                    volume_interface = session.SimpleAudioVolume
                    if volume_interface is not None:
                        try:
                            # GetMasterVolume returns 0.0-1.0
                            session_volume = volume_interface.GetMasterVolume()
                            if session_volume is not None and session_volume > 0.0:
                                # Get process to verify it's a real audio app
                                try:
                                    process = session.Process
                                    if process:
                                        proc_name = process.name().lower()
                                        # Don't count silent background processes
                                        if not proc_name.endswith('explorer.exe'):
                                            max_volume = max(max_volume, session_volume)
                                except:
                                    # Count it anyway
                                    max_volume = max(max_volume, session_volume)
                        except:
                            pass
                except:
                    continue
            
            return max_volume
        except Exception as e:
            logger.debug(f"Error getting volume from sessions: {e}")
            return 0.0
    
    def _get_volume_from_processes(self) -> float:
        """
        Fallback: Detect audio by checking for common audio applications.
        Returns 0.5 if audio app found running, 0.0 otherwise.
        """
        try:
            import psutil
            
            # Check for common audio applications
            audio_apps = {
                'svchost.exe',      # Windows audio service
                'wmplayer.exe',     # Windows Media Player
                'spotify.exe',      # Spotify
                'vlc.exe',          # VLC
                'mpv.exe',          # mpv player
                'foobar2000.exe',   # Foobar2000
                'chrome.exe',       # Chrome (web audio)
                'firefox.exe',      # Firefox (web audio)
                'edge.exe',         # Edge (web audio)
                'discord.exe',      # Discord (audio calls)
                'skype.exe',        # Skype (audio calls)
                'teams.exe',        # Teams (audio calls)
            }
            
            # Check running processes
            for proc in psutil.process_iter(['name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if proc_name in audio_apps:
                        # Audio application is running
                        return 0.5  # Indicate audio is likely playing
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting volume from processes: {e}")
            return 0.0


# Singleton instance
_monitor_instance: Optional[AudioVolumeMonitor] = None
_monitor_lock = threading.Lock()


def get_audio_volume_monitor() -> AudioVolumeMonitor:
    """Get or create the audio volume monitor singleton."""
    global _monitor_instance
    
    if _monitor_instance is None:
        with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = AudioVolumeMonitor()
    
    return _monitor_instance

if __name__ == "__main__":
    # Test the audio volume monitor
    monitor = get_audio_volume_monitor()
    monitor.start()
    
    print("Audio Volume Monitor Test")
    print("=" * 50)
    print("Play some audio and watch the volume levels...")
    print("(Monitor for 30 seconds)\n")
    
    try:
        for i in range(30):
            time.sleep(1)
            volume = monitor.get_current_volume()
            bar = '#' * int(volume * 20)
            print(f"[{i+1:2d}s] Volume: {volume:.3f} {bar}")
    finally:
        monitor.stop()
        print("\nMonitor stopped")
