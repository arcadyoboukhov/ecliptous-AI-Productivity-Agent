"""
Task Recognition & Clustering

Enhanced clustering using enriched segment features from preprocessing pipeline.
Groups similar segments into coherent tasks using multi-dimensional similarity.

Features considered:
- App/window similarity
- Temporal proximity
- User interaction patterns (keyboard, mouse, copy/paste)
- Audio/video context (mic, camera, volume)
- System load patterns (CPU, RAM, GPU)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
import math
import json


@dataclass
class Task:
    """
    A recognized task composed of one or more segments.
    
    Tasks represent coherent work activities identified through clustering.
    """
    task_id: str
    task_name: str                      # Inferred or user-provided name
    segments: List = field(default_factory=list)  # TaskSegment objects
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_minutes: float = 0.0
    
    # Context (from segments)
    primary_app: Optional[str] = None
    apps_used: Set[str] = field(default_factory=set)
    primary_window_pattern: Optional[str] = None
    
    # Aggregated features
    avg_activity_score: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_ram_usage: float = 0.0
    total_keyboard_keys: int = 0
    total_copy_paste_events: int = 0
    
    # Behavioral characteristics
    is_multitasking_task: bool = False
    is_active_task: bool = True
    
    # Confidence
    cluster_confidence: float = 0.0     # How well segments cluster together
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_minutes': self.total_duration_minutes,
            'segment_count': len(self.segments),
            'primary_app': self.primary_app,
            'apps_used': list(self.apps_used),
            'primary_window_pattern': self.primary_window_pattern,
            'avg_activity_score': self.avg_activity_score,
            'avg_cpu_usage': self.avg_cpu_usage,
            'avg_ram_usage': self.avg_ram_usage,
            'total_keyboard_keys': self.total_keyboard_keys,
            'total_copy_paste_events': self.total_copy_paste_events,
            'is_multitasking_task': self.is_multitasking_task,
            'is_active_task': self.is_active_task,
            'cluster_confidence': self.cluster_confidence,
        }


class EnrichedSimilarityMetrics:
    """
    Enhanced similarity metrics using enriched features.
    
    Computes multi-dimensional similarity between segments considering:
    - App/window context
    - Temporal proximity
    - Interaction patterns
    - Audio/video context
    - System resource patterns
    """
    
    def __init__(
        self,
        app_weight: float = 0.25,
        window_weight: float = 0.15,
        temporal_weight: float = 0.15,
        interaction_weight: float = 0.15,
        audio_video_weight: float = 0.10,
        system_weight: float = 0.10,
        behavioral_weight: float = 0.10,
    ):
        """
        Initialize with feature weights (must sum to 1.0).
        
        Args:
            app_weight: Weight for app similarity
            window_weight: Weight for window title similarity
            temporal_weight: Weight for temporal proximity
            interaction_weight: Weight for interaction patterns
            audio_video_weight: Weight for audio/video context
            system_weight: Weight for system resource patterns
            behavioral_weight: Weight for behavioral features
        """
        self.weights = {
            'app': app_weight,
            'window': window_weight,
            'temporal': temporal_weight,
            'interaction': interaction_weight,
            'audio_video': audio_video_weight,
            'system': system_weight,
            'behavioral': behavioral_weight,
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def calculate_similarity(self, seg1, seg2) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall similarity between two segments.
        
        Args:
            seg1, seg2: TaskSegment objects
        
        Returns:
            Tuple of (overall_similarity, component_scores)
            - overall_similarity: 0.0-1.0 (1.0 = identical)
            - component_scores: Dict with individual similarity scores
        """
        components = {}
        
        # 1. App similarity
        components['app'] = self._app_similarity(seg1, seg2)
        
        # 2. Window title similarity
        components['window'] = self._window_similarity(seg1, seg2)
        
        # 3. Temporal proximity
        components['temporal'] = self._temporal_similarity(seg1, seg2)
        
        # 4. Interaction pattern similarity
        components['interaction'] = self._interaction_similarity(seg1, seg2)
        
        # 5. Audio/video context similarity
        components['audio_video'] = self._audio_video_similarity(seg1, seg2)
        
        # 6. System resource pattern similarity
        components['system'] = self._system_similarity(seg1, seg2)
        
        # 7. Behavioral similarity
        components['behavioral'] = self._behavioral_similarity(seg1, seg2)
        
        # Calculate weighted overall similarity
        overall = sum(
            self.weights[key] * score
            for key, score in components.items()
        )
        
        return overall, components
    
    def calculate_distance(self, seg1, seg2) -> float:
        """
        Calculate distance (inverse of similarity).
        
        Returns:
            Distance in range [0.0, 1.0] (0.0 = identical)
        """
        similarity, _ = self.calculate_similarity(seg1, seg2)
        return 1.0 - similarity
    
    def _app_similarity(self, seg1, seg2) -> float:
        """
        Calculate app similarity.
        
        Strategy:
        - Exact match: 1.0
        - Same category (browsers, IDEs): 0.7
        - Different: 0.0
        """
        app1 = seg1.primary_app
        app2 = seg2.primary_app
        
        if not app1 or not app2:
            return 0.0
        
        if app1 == app2:
            return 1.0
        
        # Check category similarity
        category_sim = self._app_category_similarity(app1, app2)
        return category_sim
    
    def _app_category_similarity(self, app1: str, app2: str) -> float:
        """Check if apps are in the same category."""
        app1_lower = app1.lower()
        app2_lower = app2.lower()
        
        # Browser family
        browsers = {'chrome', 'firefox', 'edge', 'safari', 'msedge', 'opera', 'brave'}
        if any(b in app1_lower for b in browsers) and any(b in app2_lower for b in browsers):
            return 0.7
        
        # IDE family
        ides = {'vscode', 'code', 'pycharm', 'idea', 'eclipse', 'sublime', 'atom', 'vim'}
        if any(ide in app1_lower for ide in ides) and any(ide in app2_lower for ide in ides):
            return 0.7
        
        # Office suite
        office = {'word', 'excel', 'powerpoint', 'outlook', 'onenote'}
        if any(o in app1_lower for o in office) and any(o in app2_lower for o in office):
            return 0.7
        
        # Communication
        comm = {'slack', 'teams', 'zoom', 'discord', 'skype'}
        if any(c in app1_lower for c in comm) and any(c in app2_lower for c in comm):
            return 0.8  # Higher since context switches within meetings are common
        
        return 0.0
    
    def _window_similarity(self, seg1, seg2) -> float:
        """
        Calculate window title similarity using token-based matching.
        """
        title1 = seg1.primary_window_title
        title2 = seg2.primary_window_title
        
        if not title1 or not title2:
            return 0.5  # Neutral if missing
        
        # Tokenize and normalize
        tokens1 = set(self._tokenize_title(title1))
        tokens2 = set(self._tokenize_title(title2))
        
        if not tokens1 or not tokens2:
            return 0.5
        
        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard
    
    def _tokenize_title(self, title: str) -> List[str]:
        """Tokenize window title (lowercase, split, filter)."""
        if not title:
            return []
        
        # Common stopwords to ignore
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Tokenize
        tokens = title.lower().split()
        tokens = [t.strip('[](){}:;,."\'') for t in tokens]
        tokens = [t for t in tokens if t and t not in stopwords and len(t) > 2]
        
        return tokens
    
    def _temporal_similarity(self, seg1, seg2) -> float:
        """
        Calculate temporal proximity.
        
        Strategy:
        - Overlapping or adjacent: 1.0
        - Within 5 minutes: decay exponentially
        - Beyond 30 minutes: 0.0
        """
        if not seg1.start_time or not seg2.start_time:
            return 0.5  # Neutral
        
        # Calculate time gap
        if seg1.start_time <= seg2.start_time:
            gap = (seg2.start_time - seg1.end_time).total_seconds() / 60.0  # Minutes
        else:
            gap = (seg1.start_time - seg2.end_time).total_seconds() / 60.0
        
        # Overlapping or touching
        if gap <= 0:
            return 1.0
        
        # Exponential decay (half-life = 10 minutes)
        # At 10 min: 0.5, at 20 min: 0.25, at 30 min: 0.125
        if gap > 30:
            return 0.0
        
        similarity = math.exp(-gap / 10.0 * math.log(2))
        return similarity
    
    def _interaction_similarity(self, seg1, seg2) -> float:
        """
        Calculate interaction pattern similarity.
        
        Compare keyboard/mouse intensity and copy/paste behavior.
        """
        # Keyboard intensity similarity
        kb1 = seg1.keyboard_intensity_avg
        kb2 = seg2.keyboard_intensity_avg
        
        # Normalize to [0, 100] keys/min
        kb1_norm = min(kb1, 100.0) / 100.0
        kb2_norm = min(kb2, 100.0) / 100.0
        kb_sim = 1.0 - abs(kb1_norm - kb2_norm)
        
        # Mouse intensity similarity
        mouse1 = seg1.mouse_intensity_avg
        mouse2 = seg2.mouse_intensity_avg
        
        # Normalize to [0, 20] clicks/min
        mouse1_norm = min(mouse1, 20.0) / 20.0
        mouse2_norm = min(mouse2, 20.0) / 20.0
        mouse_sim = 1.0 - abs(mouse1_norm - mouse2_norm)
        
        # Copy/paste ratio similarity
        cp1 = seg1.total_copy_events + seg1.total_paste_events
        cp2 = seg2.total_copy_events + seg2.total_paste_events
        
        # Normalize to [0, 50] events
        cp1_norm = min(cp1, 50.0) / 50.0
        cp2_norm = min(cp2, 50.0) / 50.0
        cp_sim = 1.0 - abs(cp1_norm - cp2_norm)
        
        # Average
        return (kb_sim + mouse_sim + cp_sim) / 3.0
    
    def _audio_video_similarity(self, seg1, seg2) -> float:
        """
        Calculate audio/video context similarity.
        
        Compare mic/camera usage and audio volume.
        """
        # Mic active similarity (both on/off = similar)
        mic1 = seg1.mic_active_ratio
        mic2 = seg2.mic_active_ratio
        mic_sim = 1.0 - abs(mic1 - mic2)
        
        # Camera active similarity
        cam1 = seg1.camera_active_ratio
        cam2 = seg2.camera_active_ratio
        cam_sim = 1.0 - abs(cam1 - cam2)
        
        # Audio volume similarity
        vol1 = seg1.max_audio_volume
        vol2 = seg2.max_audio_volume
        vol_sim = 1.0 - abs(vol1 - vol2)
        
        # Average (weighted towards mic/camera since they're stronger signals)
        return (mic_sim * 0.4 + cam_sim * 0.4 + vol_sim * 0.2)
    
    def _system_similarity(self, seg1, seg2) -> float:
        """
        Calculate system resource pattern similarity.
        
        Compare CPU, RAM, GPU usage patterns.
        """
        # CPU similarity
        cpu_sim = 1.0 - abs(seg1.mean_cpu_usage - seg2.mean_cpu_usage)
        
        # RAM similarity
        ram_sim = 1.0 - abs(seg1.mean_ram_usage - seg2.mean_ram_usage)
        
        # GPU similarity (if available)
        if seg1.mean_gpu_usage is not None and seg2.mean_gpu_usage is not None:
            gpu_sim = 1.0 - abs(seg1.mean_gpu_usage - seg2.mean_gpu_usage)
            return (cpu_sim + ram_sim + gpu_sim) / 3.0
        else:
            return (cpu_sim + ram_sim) / 2.0
    
    def _behavioral_similarity(self, seg1, seg2) -> float:
        """
        Calculate behavioral characteristic similarity.
        
        Compare multitasking and activity patterns.
        """
        # Multitasking similarity
        mt1 = seg1.multitasking_score
        mt2 = seg2.multitasking_score
        mt_sim = 1.0 - abs(mt1 - mt2)
        
        # Activity level similarity
        act1 = seg1.activity_score
        act2 = seg2.activity_score
        act_sim = 1.0 - abs(act1 - act2)
        
        # Work type similarity (active vs passive)
        work_sim = 1.0 if seg1.is_active_work == seg2.is_active_work else 0.3
        
        return (mt_sim + act_sim + work_sim) / 3.0


class TaskClusterer:
    """
    Clusters segments into tasks using enriched similarity metrics.
    
    Supports multiple clustering algorithms:
    - DBSCAN: Density-based clustering
    - Hierarchical: Agglomerative clustering with linkage
    - Time-windowed: Simple temporal grouping
    """
    
    def __init__(
        self,
        similarity_metrics: Optional[EnrichedSimilarityMetrics] = None,
        min_task_duration_minutes: float = 10.0,
    ):
        """
        Initialize clusterer.
        
        Args:
            similarity_metrics: Custom similarity metrics (or default)
            min_task_duration_minutes: Minimum duration for a valid task
        """
        self.metrics = similarity_metrics or EnrichedSimilarityMetrics()
        self.min_task_duration = min_task_duration_minutes
    
    def cluster_dbscan(
        self,
        segments: List,
        eps: float = 0.3,
        min_samples: int = 1,
    ) -> List[Task]:
        """
        Cluster segments using DBSCAN algorithm.
        
        Args:
            segments: List of TaskSegment objects
            eps: Maximum distance for neighborhood (0.0-1.0)
            min_samples: Minimum segments to form a cluster
        
        Returns:
            List of Task objects
        """
        if not segments:
            return []
        
        # Calculate distance matrix
        n = len(segments)
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.metrics.calculate_distance(segments[i], segments[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Simple DBSCAN implementation
        labels = [-1] * n  # -1 = noise, >= 0 = cluster ID
        cluster_id = 0
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            
            # Find neighbors
            neighbors = [j for j in range(n) if distances[i][j] <= eps]
            
            if len(neighbors) < min_samples:
                # Mark as noise (for now)
                labels[i] = -1
            else:
                # Start new cluster
                self._expand_cluster(i, neighbors, cluster_id, labels, visited, 
                                    distances, eps, min_samples)
                cluster_id += 1
        
        # Convert clusters to tasks
        tasks = self._labels_to_tasks(segments, labels)
        return tasks
    
    def cluster_hierarchical(
        self,
        segments: List,
        distance_threshold: float = 0.4,
    ) -> List[Task]:
        """
        Cluster segments using hierarchical clustering.
        
        Args:
            segments: List of TaskSegment objects
            distance_threshold: Distance threshold for merging
        
        Returns:
            List of Task objects
        """
        if not segments:
            return []
        
        # Start with each segment in its own cluster
        clusters = [[i] for i in range(len(segments))]
        
        # Calculate initial distance matrix
        n = len(segments)
        distances = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.metrics.calculate_distance(segments[i], segments[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Agglomerative clustering (complete linkage)
        while len(clusters) > 1:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Complete linkage: max distance between any pair
                    max_dist = max(
                        distances[si][sj]
                        for si in clusters[i]
                        for sj in clusters[j]
                    )
                    
                    if max_dist < min_dist:
                        min_dist = max_dist
                        merge_i, merge_j = i, j
            
            # Stop if minimum distance exceeds threshold
            if min_dist > distance_threshold:
                break
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Convert clusters to labels
        labels = [-1] * len(segments)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                labels[idx] = cluster_id
        
        # Convert to tasks
        tasks = self._labels_to_tasks(segments, labels)
        return tasks
    
    def cluster_time_windowed(
        self,
        segments: List,
        max_gap_minutes: float = 30.0,
        min_similarity: float = 0.5,
    ) -> List[Task]:
        """
        Simple time-windowed clustering with similarity check.
        
        Groups temporally close segments if they're similar enough.
        
        Args:
            segments: List of TaskSegment objects
            max_gap_minutes: Maximum time gap between segments
            min_similarity: Minimum similarity to group together
        
        Returns:
            List of Task objects
        """
        if not segments:
            return []
        
        # Sort by time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Group into tasks
        labels = []
        current_cluster = 0
        labels.append(current_cluster)
        
        for i in range(1, len(sorted_segments)):
            prev_seg = sorted_segments[i - 1]
            curr_seg = sorted_segments[i]
            
            # Check time gap
            gap_minutes = (curr_seg.start_time - prev_seg.end_time).total_seconds() / 60.0
            
            # Check similarity
            similarity, _ = self.metrics.calculate_similarity(prev_seg, curr_seg)
            
            # Decide if same cluster
            if gap_minutes <= max_gap_minutes and similarity >= min_similarity:
                labels.append(current_cluster)
            else:
                current_cluster += 1
                labels.append(current_cluster)
        
        # Convert to tasks
        tasks = self._labels_to_tasks(sorted_segments, labels)
        return tasks
    
    def _expand_cluster(self, point_idx, neighbors, cluster_id, labels, visited,
                       distances, eps, min_samples):
        """Expand DBSCAN cluster."""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                
                # Find neighbors of neighbor
                neighbor_neighbors = [
                    j for j in range(len(distances))
                    if distances[neighbor_idx][j] <= eps
                ]
                
                if len(neighbor_neighbors) >= min_samples:
                    neighbors.extend([n for n in neighbor_neighbors if n not in neighbors])
            
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            i += 1
    
    def _labels_to_tasks(self, segments: List, labels: List[int]) -> List[Task]:
        """Convert cluster labels to Task objects."""
        # Group segments by cluster
        clusters = defaultdict(list)
        for seg, label in zip(segments, labels):
            if label >= 0:  # Skip noise (-1)
                clusters[label].append(seg)
        
        # Create tasks
        tasks = []
        for cluster_id, cluster_segments in clusters.items():
            task = self._create_task_from_segments(cluster_segments, cluster_id)
            
            # Filter by minimum duration
            if task.total_duration_minutes >= self.min_task_duration:
                tasks.append(task)
        
        return tasks
    
    def _create_task_from_segments(self, segments: List, cluster_id: int) -> Task:
        """Create a Task object from clustered segments."""
        if not segments:
            return None
        
        # Sort by time
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        
        # Aggregate timing
        start_time = sorted_segs[0].start_time
        end_time = sorted_segs[-1].end_time
        total_duration = sum(s.duration_minutes for s in sorted_segs)
        
        # Aggregate apps
        all_apps = set()
        app_counts = Counter()
        for seg in sorted_segs:
            if seg.primary_app:
                all_apps.add(seg.primary_app)
                app_counts[seg.primary_app] += seg.duration_minutes
        
        primary_app = app_counts.most_common(1)[0][0] if app_counts else None
        
        # Aggregate window titles (find common pattern)
        window_pattern = self._extract_window_pattern(sorted_segs)
        
        # Aggregate features
        avg_activity = sum(s.activity_score for s in sorted_segs) / len(sorted_segs)
        avg_cpu = sum(s.mean_cpu_usage for s in sorted_segs) / len(sorted_segs)
        avg_ram = sum(s.mean_ram_usage for s in sorted_segs) / len(sorted_segs)
        
        total_keys = sum(s.total_keyboard_keys for s in sorted_segs)
        total_cp = sum(s.total_copy_events + s.total_paste_events for s in sorted_segs)
        
        # Behavioral
        multitasking_count = sum(1 for s in sorted_segs if s.is_multitasking)
        is_multitasking = multitasking_count > len(sorted_segs) / 2
        
        active_count = sum(1 for s in sorted_segs if s.is_active_work)
        is_active = active_count > len(sorted_segs) / 2
        
        # Calculate cluster confidence (average pairwise similarity)
        confidence = self._calculate_cluster_confidence(sorted_segs)
        
        # Generate task name
        task_name = self._generate_task_name(
            primary_app, window_pattern, is_active, is_multitasking
        )
        
        # Create task
        task = Task(
            task_id=f"task_{start_time.strftime('%Y%m%d_%H%M%S')}_{cluster_id}",
            task_name=task_name,
            segments=sorted_segs,
            start_time=start_time,
            end_time=end_time,
            total_duration_minutes=total_duration,
            primary_app=primary_app,
            apps_used=all_apps,
            primary_window_pattern=window_pattern,
            avg_activity_score=avg_activity,
            avg_cpu_usage=avg_cpu,
            avg_ram_usage=avg_ram,
            total_keyboard_keys=total_keys,
            total_copy_paste_events=total_cp,
            is_multitasking_task=is_multitasking,
            is_active_task=is_active,
            cluster_confidence=confidence,
        )
        
        return task
    
    def _extract_window_pattern(self, segments: List) -> Optional[str]:
        """Extract common window title pattern from segments."""
        if not segments:
            return None
        
        # Get all window titles
        titles = [s.primary_window_title for s in segments if s.primary_window_title]
        
        if not titles:
            return None
        
        # Find most common tokens
        all_tokens = []
        for title in titles:
            tokens = title.lower().split()
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        # Get top 3 most common tokens
        top_tokens = [t for t, _ in token_counts.most_common(3)]
        
        if top_tokens:
            return " ".join(top_tokens)
        
        return None
    
    def _calculate_cluster_confidence(self, segments: List) -> float:
        """
        Calculate cluster confidence as average pairwise similarity.
        
        Higher = segments are more similar to each other.
        """
        if len(segments) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                sim, _ = self.metrics.calculate_similarity(segments[i], segments[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _generate_task_name(
        self,
        primary_app: Optional[str],
        window_pattern: Optional[str],
        is_active: bool,
        is_multitasking: bool,
    ) -> str:
        """Generate a descriptive task name."""
        # Start with app
        if not primary_app:
            name = "Unknown Task"
        else:
            # Clean app name
            app_clean = primary_app.replace('.exe', '').replace('_', ' ').title()
            name = app_clean
        
        # Add window pattern if available
        if window_pattern and len(window_pattern) > 3:
            name = f"{name}: {window_pattern.title()}"
        
        # Add type indicators
        if is_multitasking:
            name = f"{name} (Multitasking)"
        elif is_active:
            name = f"{name} (Active)"
        else:
            name = f"{name} (Passive)"
        
        return name


class TaskRecognitionPipeline:
    """
    Complete pipeline: segments → clustered tasks.
    
    Integrates preprocessing and clustering.
    """
    
    def __init__(
        self,
        clustering_method: str = 'dbscan',  # 'dbscan', 'hierarchical', 'time_windowed'
        **clusterer_kwargs
    ):
        """
        Initialize pipeline.
        
        Args:
            clustering_method: Algorithm to use
            **clusterer_kwargs: Arguments for TaskClusterer
        """
        self.clustering_method = clustering_method
        self.clusterer = TaskClusterer(**clusterer_kwargs)
    
    def recognize_tasks(self, segments: List) -> List[Task]:
        """
        Recognize tasks from segments.
        
        Args:
            segments: List of TaskSegment objects from preprocessing
        
        Returns:
            List of Task objects
        """
        if not segments:
            return []
        
        # Cluster based on method
        if self.clustering_method == 'dbscan':
            tasks = self.clusterer.cluster_dbscan(segments)
        elif self.clustering_method == 'hierarchical':
            tasks = self.clusterer.cluster_hierarchical(segments)
        elif self.clustering_method == 'time_windowed':
            tasks = self.clusterer.cluster_time_windowed(segments)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Sort by start time
        tasks = sorted(tasks, key=lambda t: t.start_time)
        
        return tasks
    
    def get_task_summary(self, tasks: List[Task]) -> Dict:
        """Generate summary statistics for tasks."""
        if not tasks:
            return {
                'task_count': 0,
                'total_duration_hours': 0.0,
                'avg_task_duration_minutes': 0.0,
                'top_apps': [],
                'active_task_ratio': 0.0,
                'multitasking_ratio': 0.0,
            }
        
        total_duration = sum(t.total_duration_minutes for t in tasks)
        avg_duration = total_duration / len(tasks)
        
        # Top apps
        app_times = Counter()
        for task in tasks:
            app_times[task.primary_app] += task.total_duration_minutes
        
        top_apps = [
            {'app': app, 'duration_hours': dur / 60.0}
            for app, dur in app_times.most_common(5)
        ]
        
        # Ratios
        active_count = sum(1 for t in tasks if t.is_active_task)
        multitasking_count = sum(1 for t in tasks if t.is_multitasking_task)
        
        return {
            'task_count': len(tasks),
            'total_duration_hours': total_duration / 60.0,
            'avg_task_duration_minutes': avg_duration,
            'avg_confidence': sum(t.cluster_confidence for t in tasks) / len(tasks),
            'top_apps': top_apps,
            'active_task_ratio': active_count / len(tasks),
            'multitasking_ratio': multitasking_count / len(tasks),
        }
