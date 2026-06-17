"""
Comprehensive Signal Collection Specification

This module defines all signals to collect for the productivity agent,
organized by category with implementation status and priority.

ARCHITECTURE:
- Raw signals (in-memory) -> DataCollector -> Normalized events -> Database
- Signals are gated by SessionGate (only collected during active sessions)
- All timestamps are UTC ISO format
- Resource metrics normalized to 0-1 range
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class SignalStatus(Enum):
    """Implementation status of a signal."""
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    PLANNED = "planned"
    NOT_STARTED = "not_started"


class SignalPriority(Enum):
    """Priority level for signal implementation."""
    CRITICAL = "critical"  # Core functionality
    HIGH = "high"         # Important for accuracy
    MEDIUM = "medium"     # Nice to have
    LOW = "low"           # Future enhancement


@dataclass
class SignalDefinition:
    """Definition of a signal to collect."""
    name: str
    category: str
    description: str
    data_type: str
    unit: Optional[str]
    normalization: Optional[str]
    status: SignalStatus
    priority: SignalPriority
    current_implementation: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "data_type": self.data_type,
            "unit": self.unit,
            "normalization": self.normalization,
            "status": self.status.value,
            "priority": self.priority.value,
            "implementation": self.current_implementation
        }


# =============================================================================
# CATEGORY 0a: USER ACTIVITY SIGNALS (CRITICAL)
# =============================================================================

USER_ACTIVITY_SIGNALS = [
    SignalDefinition(
        name="app_process_name",
        category="user_activity",
        description="Name of active application/process (e.g., firefox.exe, Code.exe)",
        data_type="string",
        unit=None,
        normalization=None,
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.CRITICAL,
        current_implementation="agent/signals/active_window.py::get_active_window()"
    ),
    
    SignalDefinition(
        name="window_title",
        category="user_activity",
        description="Title of active window/tab",
        data_type="string",
        unit=None,
        normalization="Normalized via smart_naming.py::normalize_window_title()",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.CRITICAL,
        current_implementation="agent/signals/active_window.py::get_active_window()"
    ),
    
    SignalDefinition(
        name="start_timestamp",
        category="user_activity",
        description="Session/segment start time (UTC ISO format)",
        data_type="datetime",
        unit="UTC ISO string",
        normalization=None,
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.CRITICAL,
        current_implementation="agent/session/sessionizer.py - auto-captured on session start"
    ),
    
    SignalDefinition(
        name="end_timestamp",
        category="user_activity",
        description="Session/segment end time (UTC ISO format)",
        data_type="datetime",
        unit="UTC ISO string",
        normalization=None,
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.CRITICAL,
        current_implementation="agent/session/sessionizer.py - auto-captured on session end"
    ),
    
    SignalDefinition(
        name="duration_seconds",
        category="user_activity",
        description="Duration of active interaction in seconds",
        data_type="float",
        unit="seconds",
        normalization="Computed as (end_time - start_time).total_seconds()",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.CRITICAL,
        current_implementation="Computed field in task segments and sessions"
    ),
    
    SignalDefinition(
        name="keyboard_intensity",
        category="user_activity",
        description="Keystroke rate (keys per minute)",
        data_type="float",
        unit="keys/minute",
        normalization="Raw count / duration_minutes",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/signals/input.py - pynput keyboard listener, buffered counts"
    ),
    
    SignalDefinition(
        name="mouse_activity",
        category="user_activity",
        description="Mouse movement distance and click count",
        data_type="dict",
        unit="pixels, clicks",
        normalization="Distance in pixels, click count",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/signals/input.py - pynput mouse listener, tracking movement/clicks"
    ),
    
    SignalDefinition(
        name="copy_paste_events",
        category="user_activity",
        description="Copy/paste operations (normalized, no content capture)",
        data_type="int",
        unit="count",
        normalization="Count only, content never stored",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/signals/input.py - Ctrl+C/V/X detection in keyboard hook"
    ),
]


# =============================================================================
# CATEGORY 0b: AUDIO/VIDEO SIGNALS (HIGH PRIORITY)
# =============================================================================

AUDIO_VIDEO_SIGNALS = [
    SignalDefinition(
        name="microphone_active",
        category="audio_video",
        description="Microphone on/off state",
        data_type="bool",
        unit=None,
        normalization="Boolean: True if mic in use, False otherwise",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/signals/audio_video.py::AudioVideoMonitor - tracks processes using audio input"
    ),
    
    SignalDefinition(
        name="camera_active",
        category="audio_video",
        description="Camera on/off state",
        data_type="bool",
        unit=None,
        normalization="Boolean: True if camera in use, False otherwise",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/signals/audio_video.py::AudioVideoMonitor - tracks processes using camera"
    ),
    
    SignalDefinition(
        name="audio_playback_volume",
        category="audio_video",
        description="System audio output volume level",
        data_type="float",
        unit="normalized 0-1",
        normalization="0.0 (mute) to 1.0 (max volume)",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/signals/audio_video.py::AudioVideoMonitor - pycaw audio session monitoring"
    ),
]


# =============================================================================
# CATEGORY 0c: SYSTEM RESOURCE SIGNALS (MEDIUM PRIORITY)
# =============================================================================

SYSTEM_RESOURCE_SIGNALS = [
    SignalDefinition(
        name="cpu_usage",
        category="system_resources",
        description="CPU utilization percentage",
        data_type="float",
        unit="percentage",
        normalization="0.0 to 100.0 (% of total CPU)",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/signals/system_resources.py::SystemResourceMonitor - psutil.cpu_percent()"
    ),
    
    SignalDefinition(
        name="ram_usage",
        category="system_resources",
        description="RAM utilization percentage",
        data_type="float",
        unit="percentage",
        normalization="0.0 to 100.0 (% of total RAM)",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/signals/system_resources.py::SystemResourceMonitor - psutil.virtual_memory()"
    ),
    
    SignalDefinition(
        name="gpu_usage",
        category="system_resources",
        description="GPU utilization percentage",
        data_type="float",
        unit="percentage",
        normalization="0.0 to 100.0 (% of total GPU)",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.LOW,
        current_implementation="agent/signals/system_resources.py::SystemResourceMonitor - GPUtil.getGPUs()"
    ),
    
    SignalDefinition(
        name="disk_io",
        category="system_resources",
        description="Disk read/write activity",
        data_type="float",
        unit="MB/s",
        normalization="Megabytes per second",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.LOW,
        current_implementation="agent/signals/system_resources.py::SystemResourceMonitor - psutil.disk_io_counters()"
    ),
]


# =============================================================================
# CATEGORY 0d: CONTEXTUAL METADATA (IMPLEMENTED)
# =============================================================================

CONTEXTUAL_METADATA = [
    SignalDefinition(
        name="time_of_day",
        category="contextual",
        description="Time of day category (morning/afternoon/evening/night)",
        data_type="string",
        unit=None,
        normalization="4 categories based on hour: 5-12 morning, 12-17 afternoon, 17-21 evening, 21-5 night",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/task/feature_extraction.py::_extract_contextual_features()"
    ),
    
    SignalDefinition(
        name="day_of_week",
        category="contextual",
        description="Day of week (0=Monday, 6=Sunday)",
        data_type="int",
        unit="0-6",
        normalization="ISO weekday: 0=Monday, 6=Sunday",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/task/feature_extraction.py::_extract_contextual_features()"
    ),
    
    SignalDefinition(
        name="is_weekend",
        category="contextual",
        description="Boolean flag for weekend (Saturday/Sunday)",
        data_type="bool",
        unit=None,
        normalization="True if weekday >= 5, False otherwise",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.HIGH,
        current_implementation="agent/task/feature_extraction.py::_extract_contextual_features()"
    ),
    
    SignalDefinition(
        name="is_work_hours",
        category="contextual",
        description="Boolean flag for typical work hours (9-17)",
        data_type="bool",
        unit=None,
        normalization="True if hour 9-17, False otherwise",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/task/feature_extraction.py::_extract_contextual_features()"
    ),
    
    SignalDefinition(
        name="location",
        category="contextual",
        description="User location (if allowed/available)",
        data_type="string",
        unit=None,
        normalization="City/region or 'unknown'",
        status=SignalStatus.NOT_STARTED,
        priority=SignalPriority.LOW,
        current_implementation=None
    ),
    
    SignalDefinition(
        name="historical_app_patterns",
        category="contextual",
        description="Historical usage patterns for app sequences and durations",
        data_type="dict",
        unit=None,
        normalization="JSON with app frequencies, typical durations, transition patterns",
        status=SignalStatus.IMPLEMENTED,
        priority=SignalPriority.MEDIUM,
        current_implementation="agent/task/feature_extraction.py::_load_historical_patterns() - complete implementation with duration stats and task sequences"
    ),
]


# =============================================================================
# SUMMARY & IMPLEMENTATION PLAN
# =============================================================================

ALL_SIGNALS = (
    USER_ACTIVITY_SIGNALS +
    AUDIO_VIDEO_SIGNALS +
    SYSTEM_RESOURCE_SIGNALS +
    CONTEXTUAL_METADATA
)


def get_implementation_summary() -> Dict[str, Any]:
    """Get summary of signal implementation status."""
    total = len(ALL_SIGNALS)
    by_status = {}
    by_priority = {}
    by_category = {}
    
    for signal in ALL_SIGNALS:
        # Count by status
        status_key = signal.status.value
        by_status[status_key] = by_status.get(status_key, 0) + 1
        
        # Count by priority
        priority_key = signal.priority.value
        by_priority[priority_key] = by_priority.get(priority_key, 0) + 1
        
        # Count by category
        cat_key = signal.category
        by_category[cat_key] = by_category.get(cat_key, 0) + 1
    
    return {
        "total_signals": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "by_category": by_category
    }


def get_not_implemented_signals():
    """Get list of signals that need implementation."""
    return [s for s in ALL_SIGNALS if s.status in (SignalStatus.NOT_STARTED, SignalStatus.PARTIAL)]


def get_next_priority_signals(max_count: int = 5):
    """Get next signals to implement, sorted by priority."""
    not_impl = get_not_implemented_signals()
    
    # Sort by priority (critical first) then category
    priority_order = {
        SignalPriority.CRITICAL: 0,
        SignalPriority.HIGH: 1,
        SignalPriority.MEDIUM: 2,
        SignalPriority.LOW: 3
    }
    
    sorted_signals = sorted(not_impl, key=lambda s: priority_order[s.priority])
    return sorted_signals[:max_count]


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIGNAL COLLECTION SPECIFICATION")
    print("="*80 + "\n")
    
    summary = get_implementation_summary()
    
    print(f"Total Signals: {summary['total_signals']}\n")
    
    print("Implementation Status:")
    for status, count in summary['by_status'].items():
        pct = (count / summary['total_signals']) * 100
        print(f"  {status:15s}: {count:2d} ({pct:5.1f}%)")
    
    print("\nPriority Distribution:")
    for priority, count in summary['by_priority'].items():
        print(f"  {priority:10s}: {count:2d}")
    
    print("\nCategory Distribution:")
    for category, count in summary['by_category'].items():
        print(f"  {category:20s}: {count:2d}")
    
    print("\n" + "="*80)
    print("NEXT PRIORITY SIGNALS TO IMPLEMENT")
    print("="*80 + "\n")
    
    next_signals = get_next_priority_signals(10)
    for i, signal in enumerate(next_signals, 1):
        print(f"{i}. {signal.name} ({signal.priority.value})")
        print(f"   Category: {signal.category}")
        print(f"   Description: {signal.description}")
        print(f"   Status: {signal.status.value}")
        print()
    
    print("="*80)
