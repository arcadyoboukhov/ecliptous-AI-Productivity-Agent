"""
Task Segmentation Configuration

Defines thresholds and rules for splitting sessions into task segments.
These parameters control when new segments are created based on behavioral
and contextual changes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentationConfig:
    """Configuration for task segmentation rules."""
    
    # Session-level thresholds
    session_start_minutes: int = 10  # Min active work to consider a session
    session_end_idle_minutes: int = 10  # Idle time to end session (currently 5 in code)
    
    # Segment-level thresholds
    min_segment_duration_seconds: int = 30  # Minimum segment duration to keep
    context_change_threshold_seconds: int = 60  # Min duration before allowing context split
    
    # Context change detection
    split_on_app_change: bool = True  # Split when app changes
    split_on_window_change: bool = True  # Split when window title changes
    split_on_category_change: bool = True  # Split when behavioral category changes
    
    # Window title change sensitivity
    window_similarity_threshold: float = 0.7  # How similar titles must be to avoid split (0-1)
    ignore_minor_title_changes: bool = True  # Ignore suffix changes (tab switches in same app)
    
    # Confidence thresholds
    min_confidence_for_split: float = 0.6  # Only split if confident in new task
    
    # Duration-based rules
    force_split_after_minutes: Optional[int] = 30  # Force split after long segment (None = disabled)
    
    def should_split_on_context_change(
        self,
        old_app: str,
        new_app: str,
        old_window: str,
        new_window: str,
        segment_duration_seconds: float,
        new_confidence: float
    ) -> tuple[bool, str]:
        """
        Determine if context change warrants a new segment.
        
        Returns:
            (should_split, reason)
        """
        # Check minimum duration
        if segment_duration_seconds < self.context_change_threshold_seconds:
            return False, "duration_too_short"
        
        # Check confidence
        if new_confidence < self.min_confidence_for_split:
            return False, "confidence_too_low"
        
        # Check app change
        if self.split_on_app_change and old_app != new_app:
            return True, f"app_change_{old_app}_to_{new_app}"
        
        # Check window change
        if self.split_on_window_change:
            if self.ignore_minor_title_changes:
                # Normalize and compare base titles
                from agent.task.smart_naming import normalize_window_title
                old_normalized = normalize_window_title(old_window)
                new_normalized = normalize_window_title(new_window)
                
                if old_normalized != new_normalized:
                    return True, f"window_change"
            else:
                if old_window != new_window:
                    return True, f"window_change"
        
        return False, "no_context_change"


# Default global configuration
DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()


def get_segmentation_config() -> SegmentationConfig:
    """Get the current segmentation configuration."""
    return DEFAULT_SEGMENTATION_CONFIG


def update_segmentation_config(**kwargs) -> SegmentationConfig:
    """
    Update segmentation configuration.
    
    Example:
        update_segmentation_config(
            min_segment_duration_seconds=60,
            split_on_window_change=True
        )
    """
    global DEFAULT_SEGMENTATION_CONFIG
    
    for key, value in kwargs.items():
        if hasattr(DEFAULT_SEGMENTATION_CONFIG, key):
            setattr(DEFAULT_SEGMENTATION_CONFIG, key, value)
    
    return DEFAULT_SEGMENTATION_CONFIG
