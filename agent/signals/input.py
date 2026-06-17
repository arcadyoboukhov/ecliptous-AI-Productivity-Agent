"""
Input event hooks for capturing keyboard and mouse activity.
Runs in parallel daemon threads without blocking the main loop.

Signals are gated by SessionGate: they're only collected when a session is active.
"""
from datetime import datetime, timezone
import time

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    keyboard = None
    mouse = None
    PYNPUT_AVAILABLE = False


# Module-level singletons so hooks share state and persist signal updates.
_SESSION_MANAGER_SINGLETON = None
_ACTIVE_WINDOW_THREAD_STARTED = False
_HOOKS_ATTACHED = False


def _get_session_manager(candidate=None):
    """Return a shared SessionManager v2 instance for hook handlers.

    If a candidate with `get_session` is provided, use it; otherwise keep
    a module-level singleton so keyboard/mouse threads update the same
    in-memory sessions and we can persist counters reliably.
    """
    global _SESSION_MANAGER_SINGLETON
    if candidate is not None and hasattr(candidate, "get_session"):
        return candidate
    if _SESSION_MANAGER_SINGLETON is None:
        from agent.session.manager_v2 import SessionManager
        _SESSION_MANAGER_SINGLETON = SessionManager()
    return _SESSION_MANAGER_SINGLETON


def _persist_manager(mgr):
    try:
        mgr._persist()
        mgr._save_gate_state()
    except Exception:
        pass


def _start_active_window_watcher(session_gate, session_manager, debug=False):
    """Poll active window and record app changes for the active session."""
    global _ACTIVE_WINDOW_THREAD_STARTED
    if _ACTIVE_WINDOW_THREAD_STARTED:
        return
    _ACTIVE_WINDOW_THREAD_STARTED = True

    import threading
    from agent.signals.active_window import get_active_window

    def _loop():
        last = None
        while True:
            try:
                if not session_gate.is_active():
                    time.sleep(1)
                    continue
                session_id = session_gate.active_session_id
                session = session_manager.get_session(session_id)
                if not session or not session.signals:
                    time.sleep(1)
                    continue
                aw = get_active_window()
                if aw and aw != last:
                    proc = aw.get("process_name") or "unknown"
                    title = aw.get("window_title") or ""
                    session.signals.record_app_window(proc, title)
                    
                    # Also update interval aggregator
                    try:
                        from agent.signals.interval_aggregator import get_interval_aggregator
                        aggregator = get_interval_aggregator()
                        if aggregator and aggregator.is_running:
                            aggregator.set_active_window(proc, title)
                    except:
                        pass
                    
                    # Batch persistence to reduce I/O overhead
                    last = aw
            except Exception as e:
                pass
            finally:
                time.sleep(2)  # Reduced polling frequency for better performance

    threading.Thread(target=_loop, daemon=True, name="active-window-watcher").start()


def attach_input_hooks(signal_manager=None, session_gate=None, debug=False):
    """
    Attach keyboard and mouse listeners to capture input events.

    Events are only recorded when a session is active (checked via SessionGate).
    Listeners run as daemon threads and don't block the main loop.

    If `pynput` is not available, this becomes a no-op and a warning is printed.

    Args:
        signal_manager: Optional SignalBuffer manager (legacy, kept for compatibility)
        session_gate: SessionGate instance (checks if tracking is enabled)
        debug: If True, print debug messages when hooks fire.
    """
    from agent.session.gate import get_session_gate
    import time
    
    if session_gate is None:
        session_gate = get_session_gate()

    # Reuse a shared SessionManager so counters persist across hook callbacks.
    session_manager = _get_session_manager(signal_manager)

    # Avoid attaching duplicate hooks if called multiple times in-process.
    global _HOOKS_ATTACHED
    if _HOOKS_ATTACHED:
        return  # Hooks already attached, nothing to do

    if not PYNPUT_AVAILABLE:
        _start_active_window_watcher(session_gate, session_manager, debug=debug)
        _HOOKS_ATTACHED = True
        return

    # Track last mouse position for distance calculation
    last_position = [None]  # Use list to allow modification in nested function
    
    # Track copy/paste events
    copy_paste_count = [0]  # Counter for copy/paste operations
    last_keys = []  # Track recent key combination

    def on_key_press(key):
        """Record keyboard press event (if session is active)."""
        # Gate: only record if session is active
        if not session_gate.is_active():
            return

        timestamp = datetime.now(timezone.utc)

        # Find the active session's signal buffer and record
        try:
            session_id = session_gate.active_session_id
            if session_id:
                session = session_manager.get_session(session_id)
                if session and session.signals:
                    session.signals.record_keyboard_press()
                    
                    # Also record to interval aggregator
                    try:
                        from agent.signals.interval_aggregator import get_interval_aggregator
                        aggregator = get_interval_aggregator()
                        if aggregator and aggregator.is_running:
                            aggregator.record_keyboard_press()
                    except:
                        pass
                    
                    # Detect copy/paste operations (Ctrl+C, Ctrl+V, Ctrl+X)
                    try:
                        # Check for control key combinations
                        key_str = str(key).lower()
                        
                        # Store recent keys for combination detection
                        last_keys.append(key_str)
                        if len(last_keys) > 3:
                            last_keys.pop(0)
                        
                        # Detect Ctrl+C, Ctrl+V, Ctrl+X combinations
                        if any('ctrl' in k or 'control' in k for k in last_keys):
                            if "'c'" in key_str or "'v'" in key_str or "'x'" in key_str:
                                if hasattr(session.signals, 'record_copy_paste'):
                                    session.signals.record_copy_paste()
                                copy_paste_count[0] += 1
                    except:
                        pass
                    
                    # Batch persistence to reduce I/O overhead
                    # _persist_manager(session_manager)
        except Exception as e:
            pass

    def on_click(x, y, button, pressed):
        """Record mouse click event (only on press, not release)."""
        # Gate: only record if session is active
        if not pressed or not session_gate.is_active():
            return

        timestamp = datetime.now(timezone.utc)

        try:
            session_id = session_gate.active_session_id
            if session_id:
                session = session_manager.get_session(session_id)
                if session and session.signals:
                    session.signals.record_mouse_click()
                    
                    # Also record to interval aggregator
                    try:
                        from agent.signals.interval_aggregator import get_interval_aggregator
                        aggregator = get_interval_aggregator()
                        if aggregator and aggregator.is_running:
                            aggregator.record_mouse_click()
                    except:
                        pass
                    
                    # Batch persistence to reduce I/O overhead
                    # _persist_manager(session_manager)
        except Exception as e:
            pass

    def on_move(x, y):
        """Record mouse movement distance (if session is active)."""
        # Gate: only record if session is active
        if not session_gate.is_active():
            last_position[0] = (x, y)  # type: ignore  # Still track position
            return

        try:
            session_id = session_gate.active_session_id
            if session_id:
                session = session_manager.get_session(session_id)

                if session and session.signals:
                    if last_position[0] is not None:
                        dx = x - last_position[0][0]
                        dy = y - last_position[0][1]
                        dist = int((dx**2 + dy**2)**0.5)  # Euclidean distance
                        if dist > 0:
                            session.signals.record_mouse_movement(dist)
                            
                            # Also record to interval aggregator
                            try:
                                from agent.signals.interval_aggregator import get_interval_aggregator
                                aggregator = get_interval_aggregator()
                                if aggregator and aggregator.is_running:
                                    aggregator.record_mouse_movement(float(dist))
                            except:
                                pass
                            
                            # Batch persistence to reduce I/O overhead
                            # _persist_manager(session_manager)
        except Exception as e:
            pass
        finally:
            last_position[0] = (x, y)  # type: ignore

    # Start keyboard listener as daemon thread
    try:
        keyboard.Listener(on_press=on_key_press, daemon=True).start()  # type: ignore
    except Exception as e:
        pass

    # Start mouse listener as daemon thread for clicks
    try:
        mouse.Listener(on_click=on_click, daemon=True).start()  # type: ignore
    except Exception as e:
        pass

    # Start mouse listener as daemon thread for movement
    try:
        mouse.Listener(on_move=on_move, daemon=True).start()  # type: ignore
    except Exception as e:
        pass

    # Start active window watcher thread (records app changes)
    _start_active_window_watcher(session_gate, session_manager, debug=debug)

    _HOOKS_ATTACHED = True
    print("Start a session and interact with your computer to see events below:")
    print("  - Type on keyboard -> [KEYBOARD] messages")
    print("  - Click mouse -> [MOUSE CLICK] messages")
    print("  - Move mouse -> [MOUSE MOVE] messages")
    print("  - Switch apps -> [APP CHANGE] messages")
    print("="*80 + "\n")

