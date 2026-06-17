"""
Segment Clustering

Unsupervised clustering algorithms to merge fragmented segments into unified tasks.
Reduces fragmentation caused by tab switching and context changes.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json

from agent.task.segment_distances import calculate_pairwise_distances, analyze_distance_distribution


class SegmentClusterer:
    """
    Clusters related segments into unified tasks using various algorithms.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize clusterer with database connection."""
        self.db_path = db_path or self._get_db_path()
    
    def _get_db_path(self) -> Path:
        """Get the path to the events database."""
        return Path(__file__).parent.parent / "storage" / "events.db"
    
    def load_segments_for_session(self, session_id: str) -> List[Dict]:
        """Load all segments for a specific session."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, session_id, task_id, start_time, end_time,
                   confidence, reason, feature_vector
            FROM task_segments
            WHERE session_id = ?
            ORDER BY start_time
        """, (session_id,))
        
        rows = cursor.fetchall()
        segments = []
        
        for row in rows:
            # Parse times
            start_time = datetime.fromisoformat(row["start_time"])
            end_time = datetime.fromisoformat(row["end_time"]) if row["end_time"] else None
            
            if not end_time:
                continue  # Skip active segments
            
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Parse feature vector
            features = {}
            try:
                features = json.loads(row["feature_vector"]) if row["feature_vector"] else {}
            except:
                features = {}
            
            app = features.get("active_app", "unknown")
            window_title = features.get("active_window_title", "")
            
            # Normalize title
            from agent.task.smart_naming import normalize_window_title
            normalized_title = normalize_window_title(window_title) if window_title else ""
            
            # Extract generic task
            reason = row["reason"] or ""
            generic_task = "unknown"
            if "_" in reason:
                parts = reason.split("_")
                if parts[0] in ["initial", "continuing"]:
                    generic_task = "_".join(parts[1:])
            
            segment = {
                "id": row["id"],
                "session_id": session_id,
                "task_id": row["task_id"],
                "generic_task": generic_task,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "app": app,
                "window_title": window_title,
                "normalized_title": normalized_title,
                "confidence": row["confidence"]
            }
            
            segments.append(segment)
        
        conn.close()
        return segments
    
    def cluster_dbscan(self, segments: List[Dict], eps: float = 0.4, 
                       min_samples: int = 1) -> List[int]:
        """
        Cluster segments using DBSCAN (Density-Based Spatial Clustering).
        
        Good for finding clusters of varying shapes and handling noise.
        
        Args:
            segments: List of segment dictionaries
            eps: Maximum distance between samples in same cluster
            min_samples: Minimum samples in neighborhood to form core point
        
        Returns:
            List of cluster labels (-1 = noise, 0+ = cluster id)
        """
        n = len(segments)
        
        if n == 0:
            return []
        
        # Calculate distance matrix
        distances = calculate_pairwise_distances(segments)
        
        # Initialize labels (-1 = unvisited)
        labels = [-1] * n
        cluster_id = 0
        
        # DBSCAN algorithm
        for i in range(n):
            if labels[i] != -1:
                continue  # Already processed
            
            # Find neighbors
            neighbors = self._find_neighbors(i, distances, eps)
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Mark as noise
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            
            # Expand cluster
            seed_set = neighbors.copy()
            while seed_set:
                j = seed_set.pop()
                
                if labels[j] == -1:
                    labels[j] = cluster_id
                elif labels[j] != -1:
                    continue  # Already assigned
                
                labels[j] = cluster_id
                
                # Find neighbors of j
                j_neighbors = self._find_neighbors(j, distances, eps)
                if len(j_neighbors) >= min_samples:
                    seed_set.extend(j_neighbors)
            
            cluster_id += 1
        
        return labels
    
    def cluster_agglomerative(self, segments: List[Dict], 
                             n_clusters: Optional[int] = None,
                             distance_threshold: float = 0.5) -> List[int]:
        """
        Cluster segments using Agglomerative (Hierarchical) Clustering.
        
        Good for creating hierarchical task structures.
        
        Args:
            segments: List of segment dictionaries
            n_clusters: Target number of clusters (if None, uses distance_threshold)
            distance_threshold: Maximum distance for merging
        
        Returns:
            List of cluster labels
        """
        n = len(segments)
        
        if n == 0:
            return []
        
        if n == 1:
            return [0]
        
        # Calculate distance matrix
        distances = calculate_pairwise_distances(segments)
        
        # Initialize: each segment is its own cluster
        clusters = [[i] for i in range(n)]
        
        # Merge clusters until stopping criterion
        while True:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate distance between clusters (average linkage)
                    dist = self._cluster_distance(clusters[i], clusters[j], distances)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Check stopping criterion
            if n_clusters is not None:
                if len(clusters) <= n_clusters:
                    break
            elif min_dist > distance_threshold:
                break
            
            if merge_i == -1:
                break
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Convert to labels
        labels = [0] * n
        for cluster_id, cluster_members in enumerate(clusters):
            for member_idx in cluster_members:
                labels[member_idx] = cluster_id
        
        return labels
    
    def cluster_hdbscan(self, segments: List[Dict], min_cluster_size: int = 2,
                       min_samples: int = 1) -> List[int]:
        """
        Cluster segments using HDBSCAN (Hierarchical DBSCAN).
        
        Better at handling varying densities and noise.
        This is a simplified version - full HDBSCAN requires external library.
        
        Args:
            segments: List of segment dictionaries
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core points
        
        Returns:
            List of cluster labels (-1 = noise)
        """
        # Simplified HDBSCAN: use DBSCAN with adaptive epsilon
        # Calculate distance distribution to choose good eps
        stats = analyze_distance_distribution(segments)
        
        if not stats or "quartiles" not in stats:
            # Fall back to regular DBSCAN
            return self.cluster_dbscan(segments, eps=0.4, min_samples=min_samples)
        
        # Use median distance as eps
        eps = stats["quartiles"]["q2"]
        
        return self.cluster_dbscan(segments, eps=eps, min_samples=min_samples)
    
    def _find_neighbors(self, idx: int, distances: List[List[float]], 
                       eps: float) -> List[int]:
        """Find all points within eps distance of idx."""
        neighbors = []
        for j in range(len(distances[idx])):
            if j != idx and distances[idx][j] <= eps:
                neighbors.append(j)
        return neighbors
    
    def _cluster_distance(self, cluster1: List[int], cluster2: List[int],
                         distances: List[List[float]]) -> float:
        """
        Calculate distance between two clusters using average linkage.
        """
        total_dist = 0.0
        count = 0
        
        for i in cluster1:
            for j in cluster2:
                total_dist += distances[i][j]
                count += 1
        
        return total_dist / count if count > 0 else float('inf')
    
    def merge_clustered_segments(self, segments: List[Dict], 
                                 labels: List[int]) -> List[Dict]:
        """
        Merge segments with same cluster label into unified tasks.
        
        Args:
            segments: Original segments
            labels: Cluster labels from clustering algorithm
        
        Returns:
            List of merged task dictionaries
        """
        # Group segments by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise
                clusters[label].append(segments[i])
        
        # Merge each cluster into a task
        merged_tasks = []
        
        for cluster_id, cluster_segments in clusters.items():
            if not cluster_segments:
                continue
            
            # Sort by time
            cluster_segments.sort(key=lambda s: s["start_time"])
            
            # Calculate merged properties
            start_time = cluster_segments[0]["start_time"]
            end_time = cluster_segments[-1]["end_time"]
            total_duration = sum(s["duration_seconds"] for s in cluster_segments)
            
            # Most common app
            apps = [s["app"] for s in cluster_segments]
            most_common_app = max(set(apps), key=apps.count)
            
            # Merge window titles (take most common words)
            all_titles = [s["normalized_title"] for s in cluster_segments if s["normalized_title"]]
            merged_title = self._merge_titles(all_titles)
            
            # Most common generic task
            tasks = [s["generic_task"] for s in cluster_segments]
            most_common_task = max(set(tasks), key=tasks.count)
            
            # Average confidence
            avg_confidence = sum(s["confidence"] for s in cluster_segments) / len(cluster_segments)
            
            # Create merged task
            merged_task = {
                "cluster_id": cluster_id,
                "task_name": f"{most_common_task.replace('_', ' ').title()} - {merged_title}",
                "generic_task": most_common_task,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": total_duration / 60,
                "app": most_common_app,
                "window_context": merged_title,
                "segment_count": len(cluster_segments),
                "segment_ids": [s["id"] for s in cluster_segments],
                "confidence": avg_confidence,
                "segments": cluster_segments
            }
            
            merged_tasks.append(merged_task)
        
        return merged_tasks
    
    def _merge_titles(self, titles: List[str]) -> str:
        """
        Merge multiple window titles into a representative title.
        
        Takes the most common meaningful words.
        """
        if not titles:
            return "Unknown"
        
        if len(titles) == 1:
            return titles[0]
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for title in titles:
            words = title.lower().split()
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    word_counts[word] += 1
        
        # Get top 3 words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if not top_words:
            return titles[0]  # Fall back to first title
        
        # Capitalize and join
        merged = " ".join(word.capitalize() for word, _ in top_words)
        
        return merged


def cluster_session_segments(session_id: str, algorithm: str = "dbscan",
                             **kwargs) -> Dict:
    """
    Cluster segments from a session into unified tasks.
    
    Args:
        session_id: Session ID to cluster
        algorithm: "dbscan", "agglomerative", or "hdbscan"
        **kwargs: Algorithm-specific parameters
    
    Returns:
        Dictionary with clustering results
    """
    clusterer = SegmentClusterer()
    
    # Load segments
    segments = clusterer.load_segments_for_session(session_id)
    
    if not segments:
        return {"error": "No segments found for session"}
    
    # Analyze distances
    dist_stats = analyze_distance_distribution(segments)
    
    # Perform clustering
    if algorithm == "dbscan":
        eps = kwargs.get("eps", 0.4)
        min_samples = kwargs.get("min_samples", 1)
        labels = clusterer.cluster_dbscan(segments, eps, min_samples)
    
    elif algorithm == "agglomerative":
        n_clusters = kwargs.get("n_clusters", None)
        distance_threshold = kwargs.get("distance_threshold", 0.5)
        labels = clusterer.cluster_agglomerative(segments, n_clusters, distance_threshold)
    
    elif algorithm == "hdbscan":
        min_cluster_size = kwargs.get("min_cluster_size", 2)
        min_samples = kwargs.get("min_samples", 1)
        labels = clusterer.cluster_hdbscan(segments, min_cluster_size, min_samples)
    
    else:
        return {"error": f"Unknown algorithm: {algorithm}"}
    
    # Count clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    
    # Merge segments into tasks
    merged_tasks = clusterer.merge_clustered_segments(segments, labels)
    
    return {
        "session_id": session_id,
        "algorithm": algorithm,
        "parameters": kwargs,
        "original_segments": len(segments),
        "clusters_found": n_clusters,
        "noise_segments": n_noise,
        "merged_tasks": merged_tasks,
        "distance_stats": dist_stats,
        "cluster_labels": labels
    }


if __name__ == "__main__":
    # Example usage
    print("Testing segment clustering...")
    
    # Get a session with multiple segments
    clusterer = SegmentClusterer()
    
    conn = sqlite3.connect(str(clusterer.db_path))
    cursor = conn.cursor()
    
    # Find session with most segments
    cursor.execute("""
        SELECT session_id, COUNT(*) as seg_count
        FROM task_segments
        WHERE end_time IS NOT NULL
        GROUP BY session_id
        ORDER BY seg_count DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        session_id = row[0]
        seg_count = row[1]
        
        print(f"\nTesting on session {session_id} with {seg_count} segments")
        
        # Try DBSCAN
        print("\n=== DBSCAN ===")
        result = cluster_session_segments(session_id, "dbscan", eps=0.4, min_samples=1)
        print(f"Clusters found: {result['clusters_found']}")
        print(f"Noise segments: {result['noise_segments']}")
        print(f"Merged tasks: {len(result['merged_tasks'])}")
        
        for task in result['merged_tasks']:
            print(f"\n  Task: {task['task_name']}")
            print(f"    Duration: {task['duration_minutes']:.1f} minutes")
            print(f"    Segments: {task['segment_count']}")
            print(f"    App: {task['app']}")
        
        # Try Agglomerative
        print("\n=== Agglomerative ===")
        result = cluster_session_segments(session_id, "agglomerative", distance_threshold=0.5)
        print(f"Clusters found: {result['clusters_found']}")
        print(f"Merged tasks: {len(result['merged_tasks'])}")
    
    else:
        print("No segments found in database")
