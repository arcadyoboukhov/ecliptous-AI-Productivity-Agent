"""
Comprehensive error handling for session management.

Provides robust error handling wrappers for all session operations:
- Creation failures
- Persistence failures
- ML callback failures
- State corruption recovery
"""

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Any
import json


class SessionErrorHandler:
    """Centralized error handling for session operations."""
    
    def __init__(self, error_log_path: str = ".session_errors.log"):
        self.error_log_path = Path(error_log_path)
        self.error_count = 0
        self.last_error_time = None
    
    def log_error(self, operation: str, error: Exception, session_id: Optional[str] = None, critical: bool = False):
        """Log session error with context and stack trace."""
        self.error_count += 1
        self.last_error_time = datetime.now(timezone.utc)
        
        severity = "CRITICAL" if critical else "ERROR"
        
        try:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 70}\n")
                f.write(f"[{severity}] {self.last_error_time.isoformat()}\n")
                f.write(f"Operation: {operation}\n")
                if session_id:
                    f.write(f"Session ID: {session_id}\n")
                f.write(f"Error: {type(error).__name__}: {error}\n")
                f.write(f"Stack trace:\n")
                traceback.print_exc(file=f)
                f.write(f"{'=' * 70}\n")
        except Exception as log_err:
            # If we can't even log, print to console
            print(f"FATAL: Could not log session error: {log_err}")
    
    def safe_execute(self, operation: str, func: Callable, *args, 
                     default_return: Any = None, 
                     session_id: Optional[str] = None,
                     critical: bool = False,
                     **kwargs):
        """
        Execute a function with comprehensive error handling.
        
        Args:
            operation: Description of the operation
            func: Function to execute
            *args: Positional arguments for func
            default_return: Value to return on error
            session_id: Optional session ID for context
            critical: Whether this is a critical operation
            **kwargs: Keyword arguments for func
        
        Returns:
            Function result on success, default_return on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_error(operation, e, session_id, critical)
            return default_return


# Global error handler instance
_error_handler = SessionErrorHandler()


def get_error_handler() -> SessionErrorHandler:
    """Get the global session error handler."""
    return _error_handler


def safe_session_operation(operation: str, critical: bool = False):
    """
    Decorator for session operations that need error handling.
    
    Usage:
        @safe_session_operation("create_session", critical=True)
        def create_session(self, name):
            # ... session creation logic
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id') or (args[1] if len(args) > 1 else None)
            return _error_handler.safe_execute(
                operation, 
                func, 
                *args, 
                session_id=session_id,
                critical=critical,
                **kwargs
            )
        return wrapper
    return decorator


def safe_persist_session(session, file_path: str = "sessions.json"):
    """
    (DISABLED) Session persistence to JSON files is disabled.
    This is a no-op function for backwards compatibility.
    """
    return True


def safe_ml_finalization(session, ml_callback: Callable):
    """
    Safely execute ML finalization callback with error isolation.
    
    Ensures ML errors don't crash session management.
    """
    handler = get_error_handler()
    session_id = getattr(session, 'id', 'unknown')
    
    try:
        ml_callback(session)
        return True
    except Exception as e:
        handler.log_error("ml_finalization_callback", e, session_id, critical=False)
        # ML failure should not prevent session from being saved
        return False


def recover_corrupted_session_file(file_path: str = "sessions.json"):
    """
    (DISABLED) Session file recovery is disabled.
    All session data is stored in the database only.
    """
    return True
