try:
    import win32gui
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_active_window():
    if not HAS_WIN32:
        # Fallback when win32gui is not available
        return {
            "process_name": None,
            "window_title": None
        }
    
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return None

    # Window title
    window_title = win32gui.GetWindowText(hwnd)

    # Process ID
    _, pid = win32process.GetWindowThreadProcessId(hwnd)

    process_name = None
    if HAS_PSUTIL:
        try:
            process = psutil.Process(pid)
            process_name = process.name()
        except (psutil.Error, Exception):
            process_name = None

    return {
        "process_name": process_name,
        "window_title": window_title
    }
