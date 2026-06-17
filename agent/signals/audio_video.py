"""
Audio/Video Signal Detection

Detects microphone, camera, and audio playback activity.
Windows-specific implementation using Win32 APIs and AudioSession.
"""

import threading
import time
from typing import Dict, Optional
from datetime import datetime, timezone

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioSessionManager2
    HAS_PYCAW = True
except ImportError:
    HAS_PYCAW = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Windows audio endpoint volume API (fallback)
try:
    from ctypes import *
    from ctypes.wintypes import *
    
    # GUID for IAudioEndpointVolume
    IID_IAUDIOENDPOINTVOL = GUID(0x5CDF2C82, 0x841E, 0x4546, c_ubyte * 8)
    IID_IAUDIOENDPOINTVOL.contents_extra = (0xA5, 0x0B, 0xB8, 0x20, 0x4B, 0x95, 0xDC, 0xEF)
    
    HAS_CTYPES_AUDIO = True
except:
    HAS_CTYPES_AUDIO = False


class AudioVideoMonitor:
    """
    Monitors audio/video device usage.
    
    Signals detected:
    - Microphone active (process using audio input)
    - Camera active (process using video input)
    - Audio playback volume level (0.0 to 1.0)
    """
    
    def __init__(self):
        self.is_monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Current state
        self._microphone_active = False
        self._camera_active = False
        self._audio_volume = 0.0
        self._last_update = None
        
        # Privacy: do not store process names, only on/off metadata
        self._mic_processes = set()
        self._camera_processes = set()
    
    def start(self):
        """Start monitoring audio/video signals."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def get_state(self, include_processes: bool = False) -> Dict:
        """
        Get current audio/video state.
        
        Returns:
            dict with keys:
                - microphone_active: bool
                - camera_active: bool
                - audio_volume: float (0.0 to 1.0)
                - last_update: ISO timestamp
        """
        state = {
            "microphone_active": self._microphone_active,
            "camera_active": self._camera_active,
            "audio_volume": self._audio_volume,
            "last_update": self._last_update.isoformat() if self._last_update else None
        }
        if include_processes:
            state["mic_processes"] = list(self._mic_processes)
            state["camera_processes"] = list(self._camera_processes)
        return state
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._update_audio_state()
                self._update_camera_state()
                self._last_update = datetime.now(timezone.utc)
            except Exception:
                pass
            
            # Poll every 2 seconds
            self._stop_event.wait(2.0)
    
    def _update_audio_state(self):
        """Update microphone and audio playback state."""
        if not HAS_PYCAW:
            return
        
        try:
            # Get system master volume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioSessionManager2._iid_, CLSCTX_ALL, None)
            manager = interface.QueryInterface(IAudioSessionManager2)
            
            sessions = manager.GetSessionEnumerator()
            count = sessions.GetCount()
            
            # Track active audio processes and volumes
            active_mic = False
            max_volume = 0.0
            session_volumes = []
            
            for i in range(count):
                try:
                    session = sessions.GetSession(i)
                    
                    # Get process info
                    process_id = session.GetProcessId()
                    if process_id:
                        if HAS_PSUTIL:
                            try:
                                proc = psutil.Process(process_id)
                                proc_name = proc.name()
                                
                                # Check if process is using microphone
                                if self._is_mic_process(proc_name):
                                    active_mic = True
                            except:
                                pass
                        
                        # Get volume level from this session
                        try:
                            # Get the audio volume control interface
                            volume_interface = session.QueryInterface(IAudioSessionManager2._iid_)
                            if volume_interface:
                                # Try to get SimpleAudioVolume
                                simple_volume = session.SimpleAudioVolume
                                if simple_volume:
                                    try:
                                        vol_level = simple_volume.GetMasterVolume()
                                        if vol_level is not None and vol_level > 0:
                                            session_volumes.append((process_id, vol_level))
                                            max_volume = max(max_volume, vol_level)
                                    except:
                                        pass
                        except:
                            pass
                except:
                    pass
            
            # If we got session volumes, use the max
            if session_volumes:
                self._audio_volume = max_volume
            else:
                # Fallback: try to get master volume directly
                try:
                    devices = AudioUtilities.GetSpeakers()
                    volume = devices.GetVolumeObject()
                    if volume:
                        master_vol = volume.GetMasterVolumeLevelScalar()
                        if master_vol is not None:
                            self._audio_volume = float(master_vol)
                except:
                    pass
            
            # Privacy: only store on/off state, do not retain process names
            self._mic_processes = set()
            self._microphone_active = active_mic
            
        except Exception as e:
            pass
    
    def _update_camera_state(self):
        """Update camera active state."""
        if not HAS_PSUTIL:
            return
        
        try:
            # Check for processes known to use camera
            active_camera = False
            
            for proc in psutil.process_iter(['name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if self._is_camera_process(proc_name):
                        active_camera = True
                except:
                    pass
            
            # Privacy: only store on/off state, do not retain process names
            self._camera_processes = set()
            self._camera_active = active_camera
            
        except Exception:
            pass
    
    def _is_mic_process(self, process_name: str) -> bool:
        """Check if process typically uses microphone."""
        proc_lower = process_name.lower()
        
        mic_indicators = [
            'zoom', 'teams', 'skype', 'discord', 'slack',
            'webex', 'meet', 'obs', 'streamlabs',
            'audacity', 'voicemeeter', 'voicemod'
        ]
        
        return any(indicator in proc_lower for indicator in mic_indicators)
    
    def _is_camera_process(self, process_name: str) -> bool:
        """Check if process typically uses camera."""
        camera_indicators = [
            'zoom', 'teams', 'skype', 'discord',
            'webex', 'meet', 'obs', 'streamlabs',
            'facetime', 'camera', 'snap'
        ]
        
        return any(indicator in process_name for indicator in camera_indicators)


# Global singleton
_audio_video_monitor = None


def get_audio_video_monitor() -> AudioVideoMonitor:
    """Get or create the global audio/video monitor singleton."""
    global _audio_video_monitor
    if _audio_video_monitor is None:
        _audio_video_monitor = AudioVideoMonitor()
    return _audio_video_monitor


def get_audio_video_state() -> Dict:
    """
    Get current audio/video state.
    
    Returns dict with microphone_active, camera_active, audio_volume.
    """
    monitor = get_audio_video_monitor()
    if not monitor.is_monitoring:
        monitor.start()
    return monitor.get_state()


if __name__ == "__main__":
    print("Testing Audio/Video Monitor...")
    print(f"pycaw available: {HAS_PYCAW}")
    print(f"psutil available: {HAS_PSUTIL}")
    print()
    
    monitor = get_audio_video_monitor()
    monitor.start()
    
    print("Monitoring for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        state = monitor.get_state()
        print(f"\r[{i+1}/10] Mic: {state['microphone_active']} | "
              f"Camera: {state['camera_active']} | "
              f"Volume: {state['audio_volume']:.2f}", end='')
    
    print("\n\nFinal state:")
    print(state)
    
    monitor.stop()
