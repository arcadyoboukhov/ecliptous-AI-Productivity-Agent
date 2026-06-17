"""
Quick Reference: Preprocessing & Feature Engineering

Common operations and recipes for working with the preprocessing pipeline.
"""

# ============================================================================
# BASIC OPERATIONS
# ============================================================================

# 1. Load and process intervals
from agent.task.preprocessing import PreprocessingPipeline
from agent.storage.db import get_intervals

intervals = get_intervals(limit=1000)
pipeline = PreprocessingPipeline()
pipeline.fit(intervals)
segments = pipeline.process(intervals)

# 2. Filter active work
active_segments = pipeline.process_and_filter(
    intervals,
    min_activity_score=0.3
)

# 3. Access segment features
for seg in segments:
    print(f"App: {seg.primary_app}")
    print(f"Duration: {seg.duration_minutes:.1f} min")
    print(f"Activity: {seg.keyboard_intensity_avg:.0f} keys/min")
    print(f"Active: {seg.is_active_work}")
    print(f"Multitasking: {seg.is_multitasking}")

# ============================================================================
# FEATURE ACCESS
# ============================================================================

segment = segments[0]

# Activity features
segment.keyboard_intensity_avg      # Keys per minute
segment.mouse_intensity_avg         # Clicks per minute
segment.total_copy_events           # Copy count
segment.total_paste_events          # Paste count

# System features
segment.mean_cpu_usage              # 0.0-1.0
segment.mean_ram_usage              # 0.0-1.0
segment.max_cpu_usage               # Peak CPU

# Audio/video features
segment.max_audio_volume            # 0.0-1.0
segment.mic_active_ratio            # 0.0-1.0 (% of time)
segment.camera_active_ratio         # 0.0-1.0 (% of time)

# Categorical features
segment.primary_app                 # "vscode.exe"
segment.unique_apps                 # Set of apps
segment.app_switches                # Count of switches
segment.app_time_distribution       # {app: minutes}

# Behavioral features
segment.multitasking_score          # 0.0-1.0
segment.is_multitasking             # Boolean
segment.activity_score              # 0.0-1.0
segment.is_active_work              # Boolean
segment.is_passive_work             # Boolean

# Encoded features
segment.encoded_features['app_onehot']      # [1,0,0,...]
segment.encoded_features['window_hash']     # [0.1,0.2,...]
segment.encoded_features['time_numeric']    # 0.5

# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Pattern 1: Find longest work session
longest = max(segments, key=lambda s: s.duration_minutes)
print(f"Longest: {longest.primary_app} - {longest.duration_minutes:.0f}m")

# Pattern 2: Calculate total time per app
from collections import defaultdict
app_times = defaultdict(float)
for seg in segments:
    for app, duration in seg.app_time_distribution.items():
        app_times[app] += duration

top_apps = sorted(app_times.items(), key=lambda x: x[1], reverse=True)
print(f"Top app: {top_apps[0][0]} - {top_apps[0][1]/60:.1f}h")

# Pattern 3: Separate focused vs multitasking work
focused = [s for s in segments if not s.is_multitasking]
multitasking = [s for s in segments if s.is_multitasking]
print(f"Focused: {len(focused)}, Multitasking: {len(multitasking)}")

# Pattern 4: Group by time of day
from collections import defaultdict
time_groups = defaultdict(list)
for seg in segments:
    time_groups[seg.time_of_day].append(seg)

for time_period, segs in time_groups.items():
    avg_activity = sum(s.activity_score for s in segs) / len(segs)
    print(f"{time_period}: {len(segs)} segments, activity {avg_activity:.2f}")

# Pattern 5: Export to pandas DataFrame
import pandas as pd

data = []
for seg in segments:
    data.append({
        'app': seg.primary_app,
        'duration': seg.duration_minutes,
        'keyboard': seg.keyboard_intensity_avg,
        'cpu': seg.mean_cpu_usage,
        'activity_score': seg.activity_score,
        'is_multitasking': seg.is_multitasking,
    })

df = pd.DataFrame(data)
print(df.describe())

# Pattern 6: Calculate productivity metrics
total_active_time = sum(s.duration_minutes for s in segments if s.is_active_work)
total_passive_time = sum(s.duration_minutes for s in segments if s.is_passive_work)
productivity_ratio = total_active_time / (total_active_time + total_passive_time)
print(f"Productivity ratio: {productivity_ratio:.1%}")

# ============================================================================
# FILTERING & ANALYSIS
# ============================================================================

# Filter by app
vscode_segments = [s for s in segments if s.primary_app == 'Code.exe']

# Filter by activity level
high_activity = [s for s in segments if s.activity_score > 0.7]
low_activity = [s for s in segments if s.activity_score < 0.3]

# Filter by duration
long_sessions = [s for s in segments if s.duration_minutes > 30]
short_sessions = [s for s in segments if s.duration_minutes < 15]

# Filter by work hours
work_segments = [s for s in segments if s.is_work_hours]
off_hours = [s for s in segments if not s.is_work_hours]

# Filter by system usage
cpu_intensive = [s for s in segments if s.mean_cpu_usage > 0.5]

# Combined filters
coding_sessions = [
    s for s in segments
    if s.primary_app in ['Code.exe', 'pycharm.exe']
    and s.is_active_work
    and s.duration_minutes > 20
]

# ============================================================================
# STATISTICS & INSIGHTS
# ============================================================================

def segment_statistics(segments):
    """Calculate comprehensive statistics."""
    if not segments:
        return {}
    
    return {
        'count': len(segments),
        'total_duration': sum(s.duration_minutes for s in segments),
        'avg_duration': sum(s.duration_minutes for s in segments) / len(segments),
        'avg_activity': sum(s.activity_score for s in segments) / len(segments),
        'avg_keyboard': sum(s.keyboard_intensity_avg for s in segments) / len(segments),
        'avg_cpu': sum(s.mean_cpu_usage for s in segments) / len(segments),
        'multitasking_ratio': sum(1 for s in segments if s.is_multitasking) / len(segments),
        'active_work_ratio': sum(1 for s in segments if s.is_active_work) / len(segments),
    }

stats = segment_statistics(segments)
print(f"Average duration: {stats['avg_duration']:.1f} minutes")
print(f"Average activity: {stats['avg_activity']:.2f}")
print(f"Multitasking: {stats['multitasking_ratio']:.1%}")

# ============================================================================
# EXPORT FORMATS
# ============================================================================

# Export to JSON
import json

export_data = {
    'segments': [seg.to_dict() for seg in segments],
    'summary': segment_statistics(segments),
}

with open('segments.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# Export to CSV
import csv

with open('segments.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'segment_id', 'app', 'duration', 'keyboard', 'cpu',
        'activity_score', 'is_multitasking', 'is_active_work'
    ])
    writer.writeheader()
    for seg in segments:
        writer.writerow({
            'segment_id': seg.segment_id,
            'app': seg.primary_app,
            'duration': seg.duration_minutes,
            'keyboard': seg.keyboard_intensity_avg,
            'cpu': seg.mean_cpu_usage,
            'activity_score': seg.activity_score,
            'is_multitasking': seg.is_multitasking,
            'is_active_work': seg.is_active_work,
        })

# ============================================================================
# CUSTOM PIPELINE CONFIGURATION
# ============================================================================

# Custom segment duration
pipeline = PreprocessingPipeline(
    min_segment_minutes=15.0,  # Require 15+ minutes
)

# Custom idle threshold
pipeline = PreprocessingPipeline(
    idle_split_threshold_minutes=10.0,  # Split on 10+ min gaps
)

# More apps for encoding
pipeline = PreprocessingPipeline(
    top_n_apps=50,  # Encode top 50 apps
)

# All custom settings
pipeline = PreprocessingPipeline(
    min_segment_minutes=12.0,
    idle_split_threshold_minutes=7.0,
    top_n_apps=30,
)

# ============================================================================
# REAL-TIME PROCESSING
# ============================================================================

# Process incrementally as new intervals arrive
from agent.storage.db import get_intervals
from datetime import datetime, timezone, timedelta

# Get intervals from last hour
one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
recent_intervals = get_intervals(
    start_time=one_hour_ago,
    limit=1000
)

# Process recent data
recent_segments = pipeline.process(recent_intervals)
print(f"Recent activity: {len(recent_segments)} segments")

# ============================================================================
# DEBUGGING & INSPECTION
# ============================================================================

# Inspect segment intervals
segment = segments[0]
print(f"\nSegment {segment.segment_id}")
print(f"Contains {len(segment.intervals)} intervals:")
for i, interval in enumerate(segment.intervals[:3], 1):
    print(f"  {i}. {interval['app']} - {interval['keyboard_intensity']:.0f} keys/min")

# Check encoding status
if pipeline.is_fitted:
    print(f"Encoder knows {len(pipeline.encoder.known_apps)} apps")
    print(f"App mapping: {list(pipeline.encoder.app_to_index.keys())[:5]}...")
else:
    print("Pipeline not fitted yet - call pipeline.fit() first")

# Validate segment features
def validate_segment(seg):
    """Check if segment has valid features."""
    assert seg.duration_minutes > 0, "Duration must be positive"
    assert 0 <= seg.activity_score <= 1, "Activity score out of range"
    assert 0 <= seg.mean_cpu_usage <= 1, "CPU usage out of range"
    assert seg.primary_app is not None, "Missing primary app"
    print(f"✓ Segment {seg.segment_id} is valid")

for seg in segments:
    validate_segment(seg)

print("\n✅ All patterns demonstrated!")
