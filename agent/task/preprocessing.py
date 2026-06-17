"""
Preprocessing & Feature Engineering Pipeline

Transforms raw interval signals into enriched task segments with derived features.

Architecture:
1. Segment Aggregation: Group intervals into 10+ min active work segments
2. Feature Computation: Derive statistics from intervals
3. Categorical Encoding: Transform apps, titles, time into numeric features
4. Behavioral Detection: Identify multitasking and passive/active work patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
import statistics
import math


@dataclass
class TaskSegment:
    """
    A candidate task segment aggregated from intervals.
    
    Represents 10+ minutes of continuous active work, potentially across
    multiple apps/windows within the same task context.
    """
    # Identity
    segment_id: str
    session_id: Optional[str] = None
    
    # Timing
    start_time: datetime = None
    end_time: datetime = None
    duration_minutes: float = 0.0
    
    # Interval data
    interval_count: int = 0
    intervals: List[Dict] = field(default_factory=list)
    
    # Derived features - Activity
    total_copy_events: int = 0
    total_paste_events: int = 0
    total_cut_events: int = 0
    total_keyboard_keys: int = 0
    total_mouse_clicks: int = 0
    total_mouse_distance: float = 0.0
    
    # Derived features - System
    mean_cpu_usage: float = 0.0
    mean_ram_usage: float = 0.0
    mean_gpu_usage: Optional[float] = None
    max_cpu_usage: float = 0.0
    max_ram_usage: float = 0.0
    
    # Derived features - Audio/Video
    max_audio_volume: float = 0.0
    mic_active_ratio: float = 0.0     # % of intervals with mic on
    camera_active_ratio: float = 0.0  # % of intervals with camera on
    
    # Derived features - Intensity
    keyboard_intensity_avg: float = 0.0  # keys/min average
    mouse_intensity_avg: float = 0.0     # clicks/min average
    
    # Categorical features
    primary_app: Optional[str] = None
    app_switches: int = 0
    unique_apps: Set[str] = field(default_factory=set)
    app_time_distribution: Dict[str, float] = field(default_factory=dict)  # app -> minutes
    
    primary_window_title: Optional[str] = None
    unique_window_titles: Set[str] = field(default_factory=set)
    
    time_of_day: str = "unknown"  # morning/afternoon/evening/night
    is_weekend: bool = False
    is_work_hours: bool = False
    
    # Behavioral features
    is_multitasking: bool = False
    multitasking_score: float = 0.0   # 0.0-1.0, higher = more app switching
    
    is_passive_work: bool = False
    is_active_work: bool = True
    activity_score: float = 0.0        # 0.0-1.0, higher = more active
    
    # Encoded features (populated later)
    encoded_features: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/analysis."""
        return {
            'segment_id': self.segment_id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_minutes': self.duration_minutes,
            'interval_count': self.interval_count,
            
            # Activity
            'total_copy_events': self.total_copy_events,
            'total_paste_events': self.total_paste_events,
            'total_cut_events': self.total_cut_events,
            'total_keyboard_keys': self.total_keyboard_keys,
            'total_mouse_clicks': self.total_mouse_clicks,
            'total_mouse_distance': self.total_mouse_distance,
            
            # System
            'mean_cpu_usage': self.mean_cpu_usage,
            'mean_ram_usage': self.mean_ram_usage,
            'mean_gpu_usage': self.mean_gpu_usage,
            'max_cpu_usage': self.max_cpu_usage,
            'max_ram_usage': self.max_ram_usage,
            
            # Audio/Video
            'max_audio_volume': self.max_audio_volume,
            'mic_active_ratio': self.mic_active_ratio,
            'camera_active_ratio': self.camera_active_ratio,
            
            # Intensity
            'keyboard_intensity_avg': self.keyboard_intensity_avg,
            'mouse_intensity_avg': self.mouse_intensity_avg,
            
            # Categorical
            'primary_app': self.primary_app,
            'app_switches': self.app_switches,
            'unique_apps': list(self.unique_apps),
            'app_time_distribution': self.app_time_distribution,
            'primary_window_title': self.primary_window_title,
            'unique_window_titles': list(self.unique_window_titles),
            'time_of_day': self.time_of_day,
            'is_weekend': self.is_weekend,
            'is_work_hours': self.is_work_hours,
            
            # Behavioral
            'is_multitasking': self.is_multitasking,
            'multitasking_score': self.multitasking_score,
            'is_passive_work': self.is_passive_work,
            'is_active_work': self.is_active_work,
            'activity_score': self.activity_score,
            
            # Encoded
            'encoded_features': self.encoded_features,
        }


class SegmentAggregator:
    """
    Aggregates intervals into candidate task segments.
    
    Strategy:
    - Combine consecutive intervals with similar app/context
    - Require 10+ minutes of active work
    - Split on long idle gaps (5+ minutes)
    - Split on major context switches (different app domains)
    """
    
    def __init__(
        self,
        min_segment_minutes: float = 10.0,
        idle_split_threshold_minutes: float = 5.0,
        context_switch_threshold: float = 0.7,  # Similarity threshold
    ):
        self.min_segment_minutes = min_segment_minutes
        self.idle_split_threshold = timedelta(minutes=idle_split_threshold_minutes)
        self.context_switch_threshold = context_switch_threshold
    
    def aggregate_intervals(self, intervals: List[Dict]) -> List[TaskSegment]:
        """
        Aggregate intervals into task segments.
        
        Args:
            intervals: List of interval dicts from storage (sorted by time)
        
        Returns:
            List of TaskSegment objects
        """
        if not intervals:
            return []
        
        segments = []
        current_segment = None
        segment_counter = 0
        
        for interval in intervals:
            # Parse timestamps
            start_time = self._parse_timestamp(interval['timestamp_start'])
            end_time = self._parse_timestamp(interval['timestamp_end'])
            
            # Check if we should start a new segment
            should_split = self._should_split_segment(current_segment, interval, start_time)
            
            if should_split or current_segment is None:
                # Save previous segment if it meets minimum duration
                if current_segment and self._meets_minimum_duration(current_segment):
                    segments.append(current_segment)
                
                # Start new segment
                segment_counter += 1
                current_segment = TaskSegment(
                    segment_id=f"seg_{start_time.strftime('%Y%m%d_%H%M%S')}_{segment_counter}",
                    session_id=interval.get('session_id'),
                    start_time=start_time,
                    end_time=end_time,
                    intervals=[interval],
                    interval_count=1,
                )
            else:
                # Extend current segment
                current_segment.intervals.append(interval)
                current_segment.interval_count += 1
                current_segment.end_time = end_time
        
        # Save last segment if valid
        if current_segment and self._meets_minimum_duration(current_segment):
            segments.append(current_segment)
        
        return segments
    
    def _should_split_segment(
        self,
        current_segment: Optional[TaskSegment],
        interval: Dict,
        interval_start: datetime
    ) -> bool:
        """Determine if we should start a new segment."""
        if current_segment is None:
            return False
        
        # Check for idle gap
        time_gap = interval_start - current_segment.end_time
        if time_gap > self.idle_split_threshold:
            return True
        
        # Check for major context switch (different app domain)
        current_app = current_segment.intervals[-1].get('app')
        new_app = interval.get('app')
        
        if current_app and new_app and current_app != new_app:
            # Calculate app similarity
            similarity = self._calculate_app_similarity(current_app, new_app)
            if similarity < self.context_switch_threshold:
                return True
        
        return False
    
    def _calculate_app_similarity(self, app1: str, app2: str) -> float:
        """
        Calculate similarity between two apps (0.0-1.0).
        
        Higher score = more similar (same domain/category).
        """
        if app1 == app2:
            return 1.0
        
        # Simple heuristic: check if they share common words
        # (e.g., "chrome.exe" vs "msedge.exe" are both browsers)
        words1 = set(app1.lower().replace('.exe', '').split('_'))
        words2 = set(app2.lower().replace('.exe', '').split('_'))
        
        if words1 & words2:  # Common words
            return 0.8
        
        # Check browser family
        browsers = {'chrome', 'firefox', 'edge', 'safari', 'msedge', 'opera'}
        if (words1 & browsers) and (words2 & browsers):
            return 0.9
        
        # Check IDE family
        ides = {'vscode', 'code', 'pycharm', 'idea', 'eclipse', 'sublime'}
        if (words1 & ides) and (words2 & ides):
            return 0.9
        
        # Different domains
        return 0.3
    
    def _meets_minimum_duration(self, segment: TaskSegment) -> bool:
        """Check if segment meets minimum duration requirement."""
        if not segment.intervals:
            return False
        
        duration = (segment.end_time - segment.start_time).total_seconds() / 60.0
        return duration >= self.min_segment_minutes
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO timestamp string to datetime."""
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


class FeatureComputer:
    """
    Computes derived features for task segments.
    
    Takes raw intervals and produces statistical summaries:
    - Aggregations (sum, mean, max)
    - Ratios and percentages
    - Categorical distributions
    """
    
    def compute_features(self, segment: TaskSegment) -> TaskSegment:
        """
        Compute all derived features for a segment.
        
        Modifies segment in place and returns it.
        """
        if not segment.intervals:
            return segment
        
        # Calculate duration
        segment.duration_minutes = (
            segment.end_time - segment.start_time
        ).total_seconds() / 60.0
        
        # Compute activity features
        self._compute_activity_features(segment)
        
        # Compute system features
        self._compute_system_features(segment)
        
        # Compute audio/video features
        self._compute_audio_video_features(segment)
        
        # Compute categorical features
        self._compute_categorical_features(segment)
        
        # Compute behavioral features
        self._compute_behavioral_features(segment)
        
        return segment
    
    def _compute_activity_features(self, segment: TaskSegment):
        """Compute activity-related features (keyboard, mouse, copy/paste)."""
        total_copy = 0
        total_paste = 0
        total_cut = 0
        total_keys = 0
        total_clicks = 0
        total_distance = 0.0
        keyboard_intensities = []
        
        for interval in segment.intervals:
            total_copy += interval.get('copy_count', 0)
            total_paste += interval.get('paste_count', 0)
            total_cut += interval.get('cut_count', 0)
            total_clicks += interval.get('mouse_clicks', 0)
            total_distance += interval.get('mouse_distance', 0.0)
            
            # Keyboard intensity and total keys
            kb_intensity = interval.get('keyboard_intensity', 0.0)
            keyboard_intensities.append(kb_intensity)
            
            # Estimate total keys from intensity (keys/min * interval_duration_min)
            interval_duration_min = self._get_interval_duration_minutes(interval)
            total_keys += int(kb_intensity * interval_duration_min)
        
        segment.total_copy_events = total_copy
        segment.total_paste_events = total_paste
        segment.total_cut_events = total_cut
        segment.total_keyboard_keys = total_keys
        segment.total_mouse_clicks = total_clicks
        segment.total_mouse_distance = total_distance
        
        # Averages
        segment.keyboard_intensity_avg = statistics.mean(keyboard_intensities) if keyboard_intensities else 0.0
        
        # Mouse intensity (clicks per minute)
        if segment.duration_minutes > 0:
            segment.mouse_intensity_avg = total_clicks / segment.duration_minutes
        else:
            segment.mouse_intensity_avg = 0.0
    
    def _compute_system_features(self, segment: TaskSegment):
        """Compute system resource features (CPU, RAM, GPU)."""
        cpu_values = []
        ram_values = []
        gpu_values = []
        
        for interval in segment.intervals:
            cpu = interval.get('cpu_usage', 0.0)
            ram = interval.get('ram_usage', 0.0)
            gpu = interval.get('gpu_usage')
            
            cpu_values.append(cpu)
            ram_values.append(ram)
            if gpu is not None:
                gpu_values.append(gpu)
        
        # CPU stats
        if cpu_values:
            segment.mean_cpu_usage = statistics.mean(cpu_values)
            segment.max_cpu_usage = max(cpu_values)
        
        # RAM stats
        if ram_values:
            segment.mean_ram_usage = statistics.mean(ram_values)
            segment.max_ram_usage = max(ram_values)
        
        # GPU stats
        if gpu_values:
            segment.mean_gpu_usage = statistics.mean(gpu_values)
    
    def _compute_audio_video_features(self, segment: TaskSegment):
        """Compute audio/video features (mic, camera, volume)."""
        audio_volumes = []
        mic_active_count = 0
        camera_active_count = 0
        
        for interval in segment.intervals:
            audio = interval.get('audio_volume', 0.0)
            mic = interval.get('mic_active', False)
            camera = interval.get('camera_active', False)
            
            audio_volumes.append(audio)
            if mic:
                mic_active_count += 1
            if camera:
                camera_active_count += 1
        
        # Max audio volume
        segment.max_audio_volume = max(audio_volumes) if audio_volumes else 0.0
        
        # Ratios
        total_intervals = len(segment.intervals)
        if total_intervals > 0:
            segment.mic_active_ratio = mic_active_count / total_intervals
            segment.camera_active_ratio = camera_active_count / total_intervals
    
    def _compute_categorical_features(self, segment: TaskSegment):
        """Compute categorical features (apps, windows, time context)."""
        # App statistics
        app_durations = defaultdict(float)
        app_sequence = []
        window_titles = []
        
        for interval in segment.intervals:
            app = interval.get('app')
            window = interval.get('window_title')
            
            if app:
                app_sequence.append(app)
                duration_min = self._get_interval_duration_minutes(interval)
                app_durations[app] += duration_min
                segment.unique_apps.add(app)
            
            if window:
                window_titles.append(window)
                segment.unique_window_titles.add(window)
        
        # Primary app (most time spent)
        if app_durations:
            segment.primary_app = max(app_durations.items(), key=lambda x: x[1])[0]
            segment.app_time_distribution = dict(app_durations)
        
        # App switches
        switches = 0
        for i in range(1, len(app_sequence)):
            if app_sequence[i] != app_sequence[i-1]:
                switches += 1
        segment.app_switches = switches
        
        # Primary window title (most common)
        if window_titles:
            window_counter = Counter(window_titles)
            segment.primary_window_title = window_counter.most_common(1)[0][0]
        
        # Time context (use first interval as representative)
        if segment.intervals:
            first_interval = segment.intervals[0]
            segment.time_of_day = first_interval.get('time_of_day', 'unknown')
            segment.is_weekend = first_interval.get('is_weekend', False)
            segment.is_work_hours = first_interval.get('is_work_hours', False)
    
    def _compute_behavioral_features(self, segment: TaskSegment):
        """Compute behavioral features (multitasking, active/passive work)."""
        # Multitasking detection
        self._detect_multitasking(segment)
        
        # Active vs passive work detection
        self._detect_activity_level(segment)
    
    def _detect_multitasking(self, segment: TaskSegment):
        """
        Detect multitasking based on app switching frequency.
        
        High app switch rate + multiple unique apps = multitasking
        """
        unique_app_count = len(segment.unique_apps)
        switch_rate = segment.app_switches / segment.duration_minutes if segment.duration_minutes > 0 else 0
        
        # Multitasking score (0.0-1.0)
        # Factors: unique apps (normalized to 5+) + switch rate (normalized to 1/min)
        app_score = min(1.0, unique_app_count / 5.0)
        switch_score = min(1.0, switch_rate / 1.0)  # 1+ switches per minute
        
        segment.multitasking_score = (app_score + switch_score) / 2.0
        
        # Classify as multitasking if score > 0.5
        segment.is_multitasking = segment.multitasking_score > 0.5
    
    def _detect_activity_level(self, segment: TaskSegment):
        """
        Detect passive vs active work.
        
        Active work: High keyboard/mouse + moderate audio/system usage
        Passive work: Low keyboard/mouse + high audio (e.g., watching videos, meetings)
        """
        # Keyboard/mouse activity (normalized)
        kb_score = min(1.0, segment.keyboard_intensity_avg / 60.0)  # Normalize to 60 keys/min
        mouse_score = min(1.0, segment.mouse_intensity_avg / 10.0)  # Normalize to 10 clicks/min
        input_activity = (kb_score + mouse_score) / 2.0
        
        # Audio presence
        audio_score = segment.max_audio_volume
        
        # Activity score: prioritize input activity
        # High input = active work
        # Low input + high audio = passive work (meetings, videos)
        segment.activity_score = input_activity
        
        # Classification
        if input_activity > 0.3:
            segment.is_active_work = True
            segment.is_passive_work = False
        elif audio_score > 0.5 and input_activity < 0.2:
            # Low input but audio present = passive (watching/listening)
            segment.is_passive_work = True
            segment.is_active_work = False
        else:
            # Low everything = idle/passive
            segment.is_passive_work = True
            segment.is_active_work = False
    
    def _get_interval_duration_minutes(self, interval: Dict) -> float:
        """Calculate interval duration in minutes."""
        start = self._parse_timestamp(interval['timestamp_start'])
        end = self._parse_timestamp(interval['timestamp_end'])
        return (end - start).total_seconds() / 60.0
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO timestamp string to datetime."""
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


class CategoricalEncoder:
    """
    Encodes categorical features into numeric representations.
    
    Supports:
    - One-hot encoding for apps
    - Simple embedding/hashing for window titles
    - Time of day numeric encoding
    """
    
    def __init__(self, top_n_apps: int = 20):
        """
        Initialize encoder.
        
        Args:
            top_n_apps: Number of top apps to one-hot encode (others = 'other')
        """
        self.top_n_apps = top_n_apps
        self.known_apps: Set[str] = set()
        self.app_to_index: Dict[str, int] = {}
    
    def fit(self, segments: List[TaskSegment]):
        """
        Learn encoding from segments (find top apps).
        
        Args:
            segments: Training segments to learn from
        """
        # Count app frequencies
        app_counts = Counter()
        for segment in segments:
            for app in segment.unique_apps:
                app_counts[app] += 1
        
        # Get top N apps
        top_apps = [app for app, _ in app_counts.most_common(self.top_n_apps)]
        self.known_apps = set(top_apps)
        
        # Create app -> index mapping
        self.app_to_index = {app: i for i, app in enumerate(sorted(top_apps))}
        self.app_to_index['__OTHER__'] = len(self.app_to_index)
    
    def encode_segment(self, segment: TaskSegment) -> TaskSegment:
        """
        Encode categorical features for a segment.
        
        Adds encoded features to segment.encoded_features dict.
        """
        # One-hot encode primary app
        app_encoding = self._encode_app(segment.primary_app)
        
        # Encode window title (simple hash to fixed dimensions)
        window_encoding = self._encode_window_title(segment.primary_window_title)
        
        # Encode time of day
        time_encoding = self._encode_time_of_day(segment.time_of_day)
        
        # Store encoded features
        segment.encoded_features = {
            'app_onehot': app_encoding,
            'window_hash': window_encoding,
            'time_numeric': time_encoding,
            'is_weekend': 1 if segment.is_weekend else 0,
            'is_work_hours': 1 if segment.is_work_hours else 0,
        }
        
        return segment
    
    def _encode_app(self, app: Optional[str]) -> List[int]:
        """One-hot encode app name."""
        if not self.app_to_index:
            # Not fitted yet, return empty
            return [0] * (self.top_n_apps + 1)
        
        # Create one-hot vector
        encoding = [0] * len(self.app_to_index)
        
        if app and app in self.known_apps:
            index = self.app_to_index[app]
            encoding[index] = 1
        else:
            # Unknown app -> __OTHER__
            other_index = self.app_to_index['__OTHER__']
            encoding[other_index] = 1
        
        return encoding
    
    def _encode_window_title(self, title: Optional[str], dimensions: int = 10) -> List[float]:
        """
        Simple hash-based encoding for window titles.
        
        Maps title to fixed-dimension vector using hash function.
        """
        if not title:
            return [0.0] * dimensions
        
        # Simple hash to dimensions
        encoding = [0.0] * dimensions
        for i, char in enumerate(title):
            hash_val = (ord(char) + i) % dimensions
            encoding[hash_val] += 1.0
        
        # Normalize
        total = sum(encoding)
        if total > 0:
            encoding = [v / total for v in encoding]
        
        return encoding
    
    def _encode_time_of_day(self, time_of_day: str) -> float:
        """
        Encode time of day as numeric value (0.0-1.0).
        
        Maps to circular encoding: morning=0.25, afternoon=0.5, evening=0.75, night=0.0
        """
        time_map = {
            'morning': 0.25,
            'afternoon': 0.5,
            'evening': 0.75,
            'night': 0.0,
            'unknown': 0.5,  # Default to middle
        }
        return time_map.get(time_of_day, 0.5)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline: intervals → enriched segments.
    
    Usage:
        pipeline = PreprocessingPipeline()
        segments = pipeline.process(intervals)
    """
    
    def __init__(
        self,
        min_segment_minutes: float = 10.0,
        idle_split_threshold_minutes: float = 5.0,
        top_n_apps: int = 20,
    ):
        self.aggregator = SegmentAggregator(
            min_segment_minutes=min_segment_minutes,
            idle_split_threshold_minutes=idle_split_threshold_minutes,
        )
        self.feature_computer = FeatureComputer()
        self.encoder = CategoricalEncoder(top_n_apps=top_n_apps)
        self.is_fitted = False
    
    def fit(self, intervals: List[Dict]):
        """
        Fit the pipeline on training data (learn app encodings).
        
        Args:
            intervals: Training interval data
        """
        # Aggregate to segments
        segments = self.aggregator.aggregate_intervals(intervals)
        
        # Compute features
        for segment in segments:
            self.feature_computer.compute_features(segment)
        
        # Fit encoder
        self.encoder.fit(segments)
        self.is_fitted = True
    
    def process(self, intervals: List[Dict]) -> List[TaskSegment]:
        """
        Full preprocessing pipeline: intervals → enriched segments.
        
        Args:
            intervals: Raw interval dicts from storage
        
        Returns:
            List of enriched TaskSegment objects
        """
        # Step 1: Aggregate intervals into segments
        segments = self.aggregator.aggregate_intervals(intervals)
        
        # Step 2: Compute derived features
        for segment in segments:
            self.feature_computer.compute_features(segment)
        
        # Step 3: Encode categorical features (if fitted)
        if self.is_fitted:
            for segment in segments:
                self.encoder.encode_segment(segment)
        
        return segments
    
    def process_and_filter(
        self,
        intervals: List[Dict],
        min_activity_score: float = 0.1,
    ) -> List[TaskSegment]:
        """
        Process intervals and filter out low-activity segments.
        
        Args:
            intervals: Raw interval data
            min_activity_score: Minimum activity score to keep segment
        
        Returns:
            Filtered list of active segments
        """
        segments = self.process(intervals)
        
        # Filter by activity
        active_segments = [
            seg for seg in segments
            if seg.activity_score >= min_activity_score
        ]
        
        return active_segments
