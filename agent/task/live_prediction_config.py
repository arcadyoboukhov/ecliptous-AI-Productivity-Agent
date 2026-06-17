"""
Configuration for live task prediction system.

These settings control the real-time task prediction feature that runs
independently from session finalization, predicting the current task from
the last N seconds of activity.
"""

# ============================================================================
# Live Prediction Window & Frequency
# ============================================================================

# Time window to analyze for live prediction (seconds)
# Default: 60 seconds - analyzes activity from last minute
LIVE_PREDICTION_WINDOW_SECONDS = 60

# How often to perform predictions (seconds)
# Default: 1.5 seconds - frequent updates without excessive CPU
LIVE_PREDICTION_INTERVAL_SECONDS = 1.5

# ============================================================================
# Confidence Thresholds
# ============================================================================

# Minimum confidence to emit a live prediction
# Live predictions use more permissive thresholds than finalized sessions
# because shorter time windows are noisier
# Default: 0.50 (50%) - permissive for responsiveness
LIVE_PREDICTION_CONFIDENCE_THRESHOLD = 0.50

# Maximum Euclidean distance to centroid for task match
# Lower = stricter matching, higher = more permissive
# Default: 0.35 (same as finalized sessions, normalized 0-1 distance)
LIVE_PREDICTION_DISTANCE_THRESHOLD = 0.35

# ============================================================================
# Dynamic Confidence Adjustment
# ============================================================================

# Whether to use dynamic confidence thresholds based on event count
# When True: stricter requirements (0.70+) for lots of data, more permissive 
#            (0.45+) when data is limited
# Default: True
USE_DYNAMIC_CONFIDENCE_THRESHOLDS = True

# ============================================================================
# Smoothing & History
# ============================================================================

# Number of recent predictions to keep in memory per session
# Used for smoothing/deduplicating rapid updates
# Default: 3 - last 3 predictions
LIVE_PREDICTION_HISTORY_SIZE = 3

# ============================================================================
# Persistence
# ============================================================================

# Whether to persist live predictions to database
# When True: stored in live_task_predictions table for analytics
# Default: True
PERSIST_LIVE_PREDICTIONS = True

# Days to keep live predictions in database
# Older predictions are automatically cleaned up
# Default: 7 days
LIVE_PREDICTIONS_RETENTION_DAYS = 7

# ============================================================================
# Integration with Centroid Updates
# ============================================================================

# Whether live predictions should update centroids
# When False (default): only finalized sessions update centroids
#                       live predictions just read existing centroids
# When True: live predictions with high confidence can update centroids incrementally
# Default: False - conservative approach
LIVE_PREDICTIONS_UPDATE_CENTROIDS = False

# If live predictions update centroids, the confidence threshold required
# Default: 0.75 - only very confident live predictions update
LIVE_PREDICTION_CENTROID_UPDATE_THRESHOLD = 0.75

# ============================================================================
# Feature Extraction for Live Prediction
# ============================================================================

# Whether to downsample from 1-minute aggregation to 5-second aggregation
# for finer-grained live prediction
# Default: False - use 1-minute intervals (simpler, less CPU)
USE_FINE_GRAINED_FEATURES = False

# If using fine-grained features, the aggregation interval (seconds)
FINE_GRAINED_WINDOW_SECONDS = 5

# ============================================================================
# Filtering & Validation
# ============================================================================

# Minimum number of input events before making first prediction
# Prevents noisy early predictions
# Default: 10 - need at least 10 keyboard/mouse events
MIN_EVENTS_FOR_PREDICTION = 10

# Whether to validate predictions for coherence
# When True: reject predictions that contradict recent history
# Default: True
VALIDATE_PREDICTION_COHERENCE = True

# Maximum distance jump for transitions (prevents spurious jumps)
# If new prediction distance > current + threshold, flag as potential noise
# Default: 0.30
MAX_TRANSITION_DISTANCE_JUMP = 0.30
