import time
import ctypes
from ctypes import Structure, c_uint, byref

def get_idle_seconds():
    """Return seconds since last keyboard or mouse input (Windows).

    Uses the Win32 GetLastInputInfo / GetTickCount pair via ctypes which
    is robust across environments and does not depend on pywin32.
    Returns 0.0 on failure or non-Windows platforms.
    """
    try:
        class LASTINPUTINFO(Structure):
            _fields_ = [("cbSize", c_uint), ("dwTime", c_uint)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        if not user32.GetLastInputInfo(byref(lii)):
            return 0.0

        last_tick = lii.dwTime
        current_tick = kernel32.GetTickCount()

        # Handle tick count wraparound safely by using unsigned arithmetic
        idle_ms = (current_tick - last_tick) & 0xFFFFFFFF
        return float(idle_ms) / 1000.0
    except Exception:
        return 0.0

