"""
Input hooks with engagement detection - only collect when user is engaged.
"""
from datetime import datetime, timezone
import time
import threading

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    keyboard = None
    mouse = None
    PYNPUT_AVAILABLE = False


_HOOKS_ATTACHED = False


def attach_input_hooks_with_engagement(engagement_detector, data_collector, debug=False):
    """
    Attach keyboard and mouse listeners with engagement detection.
    
    All input events notify the engagement detector.
    Only collect data when engaged.
    """
    global _HOOKS_ATTACHED
    
    try:
        if _HOOKS_ATTACHED:
            return  # Already attached
        
        if not PYNPUT_AVAILABLE:
            from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
            log_component_error(
                ComponentType.SIGNALS,
                "attach_input_hooks",
                Exception("pynput not available"),
                ErrorSeverity.CRITICAL
            )
            _HOOKS_ATTACHED = True
            return
        
        # Track last mouse position and movement time
        last_position = [None]
        last_move_time = [0]
        
        def on_key_press(key):
            """Record keyboard event - minimal overhead."""
            try:
                # Always notify engagement detector (very fast)
                engagement_detector.record_input_event("keyboard")
                
                # Only collect if engaged (quick flag check)
                if data_collector.is_active():
                    data_collector.collect_keyboard_event()
            except Exception as e:
                # Silently log but don't block input
                try:
                    from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
                    log_component_error(ComponentType.SIGNALS, "on_key_press", e, ErrorSeverity.WARNING)
                except:
                    pass
        
        def on_click(x, y, button, pressed):
            """Record mouse click - minimal overhead."""
            if not pressed:
                return
            
            try:
                # Always notify engagement detector
                engagement_detector.record_input_event("mouse_click")
                
                # Only collect if engaged
                if data_collector.is_active():
                    data_collector.collect_mouse_click(x, y, str(button))
            except Exception as e:
                # Silently log but don't block input
                try:
                    from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
                    log_component_error(ComponentType.SIGNALS, "on_click", e, ErrorSeverity.WARNING)
                except:
                    pass
        
        def on_move(x, y):
            """Record mouse movement with throttling - minimal overhead."""
            now_ns = time.time_ns()  # Faster than time.time() for throttling
            
            # Throttle mouse move events to every 100ms to reduce overhead
            if now_ns - last_move_time[0] < 100_000_000:  # 100ms in nanoseconds
                return
            
            last_move_time[0] = now_ns
            
            try:
                # Always notify engagement detector
                engagement_detector.record_input_event("mouse_move")
                
                # Calculate distance
                distance = 0
                if last_position[0] is not None:
                    dx = x - last_position[0][0]
                    dy = y - last_position[0][1]
                    distance = (dx**2 + dy**2)**0.5
                
                last_position[0] = (x, y)
                
                # Only collect significant movements when engaged (skip debug logging)
                if data_collector.is_active() and distance > 50:
                    data_collector.collect_mouse_move(x, y, distance)
            except Exception as e:
                # Silently log but don't block input
                try:
                    from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
                    log_component_error(ComponentType.SIGNALS, "on_move", e, ErrorSeverity.WARNING, x=x, y=y)
                except:
                    pass
        
        # Helper to start listener in a background thread with timeout protection
        def start_listener_with_timeout(listener_factory, listener_name, timeout=3):
            """Start a listener in a background thread to avoid blocking."""
            def _start():
                try:
                    listener = listener_factory()
                    listener.start()
                except Exception as e:
                    pass
            
            thread = threading.Thread(target=_start, daemon=True)
            thread.start()
            thread.join(timeout=timeout)
        
        # Start listeners in background threads to avoid blocking
        start_listener_with_timeout(
            lambda: keyboard.Listener(on_press=on_key_press, daemon=True),
            "Keyboard listener"
        )
        
        start_listener_with_timeout(
            lambda: mouse.Listener(on_click=on_click, daemon=True),
            "Mouse click listener"
        )
        
        start_listener_with_timeout(
            lambda: mouse.Listener(on_move=on_move, daemon=True),
            "Mouse movement listener"
        )
        _HOOKS_ATTACHED = True
    except Exception as e:
        # Log failure to attach hooks
        try:
            from agent.error_handling import log_component_error, ComponentType, ErrorSeverity
            log_component_error(
                ComponentType.SIGNALS,
                "attach_hooks_general",
                e,
                ErrorSeverity.CRITICAL
            )
        except:
            pass
        # Set flag anyway to prevent repeated attempts
        _HOOKS_ATTACHED = True
