"""
Comprehensive error handling system for the entire Productivity Agent.

Provides unified error handling, logging, and recovery mechanisms
for all components: sessions, database, signals, inference, analytics, etc.
"""

import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Any, Dict
from enum import Enum
import threading


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "DEBUG"         # Informational, no action needed
    INFO = "INFO"           # Normal operation info
    WARNING = "WARNING"     # Potential issue, operation continues
    ERROR = "ERROR"         # Error occurred, operation failed but system continues
    CRITICAL = "CRITICAL"   # Critical failure, component may be unavailable


class ComponentType(Enum):
    """System components for error categorization."""
    SESSION = "SESSION"
    DATABASE = "DATABASE"
    SIGNALS = "SIGNALS"
    INFERENCE = "INFERENCE"
    ML = "ML"
    ANALYTICS = "ANALYTICS"
    INTENT = "INTENT"
    PERSISTENCE = "PERSISTENCE"
    PROCESS = "PROCESS"
    MAIN_LOOP = "MAIN_LOOP"
    UNKNOWN = "UNKNOWN"


class GlobalErrorHandler:
    """Centralized error handling for the entire application."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.error_log_path = Path(".agent_errors.log")
        self.error_counts: Dict[ComponentType, int] = {comp: 0 for comp in ComponentType}
        self.last_error_time: Dict[ComponentType, Optional[datetime]] = {comp: None for comp in ComponentType}
        self.total_errors = 0
        self.critical_errors = 0
        self._initialized = True
    
    def log(self, 
            component: ComponentType,
            operation: str, 
            error: Exception, 
            severity: ErrorSeverity = ErrorSeverity.ERROR,
            context: Optional[Dict[str, Any]] = None):
        """
        Log an error with full context and stack trace.
        
        Args:
            component: Which component the error occurred in
            operation: Description of what was being attempted
            error: The exception that occurred
            severity: How severe this error is
            context: Additional context (session_id, user_id, etc.)
        """
        self.total_errors += 1
        self.error_counts[component] += 1
        self.last_error_time[component] = datetime.now(timezone.utc)
        
        if severity == ErrorSeverity.CRITICAL:
            self.critical_errors += 1
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"[{severity.value}] {timestamp}\n")
                f.write(f"Component: {component.value}\n")
                f.write(f"Operation: {operation}\n")
                
                if context:
                    f.write(f"Context:\n")
                    for key, value in context.items():
                        f.write(f"  {key}: {value}\n")
                
                f.write(f"Error: {type(error).__name__}: {error}\n")
                f.write(f"Stack trace:\n")
                traceback.print_exc(file=f)
                f.write(f"{'=' * 80}\n")
                
        except Exception as log_err:
            # Last resort: print to stderr if logging fails
            print(f"FATAL: Could not log error: {log_err}", file=sys.stderr)
            print(f"Original error: {error}", file=sys.stderr)
    
    def safe_execute(self, 
                     component: ComponentType,
                     operation: str, 
                     func: Callable, 
                     *args,
                     default_return: Any = None,
                     severity: ErrorSeverity = ErrorSeverity.ERROR,
                     context: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Any:
        """
        Execute a function with comprehensive error handling.
        
        Args:
            component: Which component is calling this
            operation: Description of the operation
            func: Function to execute
            *args: Positional arguments for func
            default_return: Value to return on error
            severity: How severe errors should be treated
            context: Additional context for logging
            **kwargs: Keyword arguments for func
        
        Returns:
            Function result on success, default_return on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log(component, operation, e, severity, context)
            return default_return
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": self.total_errors,
            "critical_errors": self.critical_errors,
            "errors_by_component": {
                comp.value: count 
                for comp, count in self.error_counts.items() 
                if count > 0
            },
            "last_errors": {
                comp.value: time.isoformat() if time else None
                for comp, time in self.last_error_time.items()
                if time is not None
            }
        }


# Global singleton instance
_error_handler = GlobalErrorHandler()


def get_error_handler() -> GlobalErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


def safe_db_operation(operation: str, context: Optional[Dict] = None):
    """Decorator for database operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_handler.safe_execute(
                ComponentType.DATABASE,
                operation,
                func,
                *args,
                context=context,
                severity=ErrorSeverity.ERROR,
                **kwargs
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def safe_signal_operation(operation: str):
    """Decorator for signal collection operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_handler.safe_execute(
                ComponentType.SIGNALS,
                operation,
                func,
                *args,
                severity=ErrorSeverity.WARNING,  # Signals are non-critical
                **kwargs
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def safe_inference_operation(operation: str):
    """Decorator for inference engine operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_handler.safe_execute(
                ComponentType.INFERENCE,
                operation,
                func,
                *args,
                severity=ErrorSeverity.ERROR,
                **kwargs
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def safe_ml_operation(operation: str):
    """Decorator for ML operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_handler.safe_execute(
                ComponentType.ML,
                operation,
                func,
                *args,
                severity=ErrorSeverity.WARNING,  # ML failures shouldn't crash the system
                **kwargs
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def safe_analytics_operation(operation: str):
    """Decorator for analytics operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_handler.safe_execute(
                ComponentType.ANALYTICS,
                operation,
                func,
                *args,
                severity=ErrorSeverity.WARNING,
                **kwargs
            )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def log_component_error(component: ComponentType, operation: str, error: Exception, 
                        severity: ErrorSeverity = ErrorSeverity.ERROR, **context):
    """Quick helper to log an error from any component."""
    _error_handler.log(component, operation, error, severity, context)


def handle_critical_failure(component: ComponentType, operation: str, error: Exception, **context):
    """Handle a critical failure that may require shutdown or major recovery."""
    _error_handler.log(component, operation, error, ErrorSeverity.CRITICAL, context)
    # Could add notifications, restart logic, etc. here
