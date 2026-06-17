"""
Task Consolidation Analysis

Tools to analyze and visualize the effect of clustering on task fragmentation.
Shows before/after comparison of fragmented segments vs. consolidated tasks.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from agent.task.segment_clustering import cluster_session_segments, SegmentClusterer
from agent.task.segment_distances import analyze_distance_distribution


def analyze_session_fragmentation(session_id: str) -> Dict:
    """
    Analyze fragmentation in a session before clustering.
    
    Shows how many context switches and small segments exist.
    """
    clusterer = SegmentClusterer()
    segments = clusterer.load_segments_for_session(session_id)
    
    if not segments:
        return {"error": "No segments found"}
    
    # Sort by time
    segments.sort(key=lambda s: s["start_time"])
    
    # Count fragmentation metrics
    app_switches = 0
    window_switches = 0
    task_switches = 0
    short_segments = 0  # < 2 minutes
    
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        
        if prev["app"] != curr["app"]:
            app_switches += 1
        
        if prev["normalized_title"] != curr["normalized_title"]:
            window_switches += 1
        
        if prev["generic_task"] != curr["generic_task"]:
            task_switches += 1
        
        if curr["duration_seconds"] < 120:
            short_segments += 1
    
    # Calculate total duration
    total_duration = sum(s["duration_seconds"] for s in segments) / 60  # minutes
    
    # Group by app
    apps = defaultdict(int)
    for s in segments:
        apps[s["app"]] += 1
    
    # Group by task
    tasks = defaultdict(int)
    for s in segments:
        tasks[s["generic_task"]] += 1
    
    return {
        "session_id": session_id,
        "total_segments": len(segments),
        "total_duration_minutes": total_duration,
        "avg_segment_duration": total_duration / len(segments),
        "app_switches": app_switches,
        "window_switches": window_switches,
        "task_switches": task_switches,
        "short_segments": short_segments,
        "short_segment_ratio": short_segments / len(segments),
        "apps_used": len(apps),
        "tasks_detected": len(tasks),
        "app_distribution": dict(apps),
        "task_distribution": dict(tasks)
    }


def compare_clustering_algorithms(session_id: str) -> Dict:
    """
    Compare results from different clustering algorithms on same session.
    """
    algorithms = [
        ("dbscan", {"eps": 0.4, "min_samples": 1}),
        ("dbscan", {"eps": 0.5, "min_samples": 1}),
        ("agglomerative", {"distance_threshold": 0.4}),
        ("agglomerative", {"distance_threshold": 0.5}),
        ("hdbscan", {"min_cluster_size": 2})
    ]
    
    results = []
    
    for algo, params in algorithms:
        result = cluster_session_segments(session_id, algo, **params)
        
        if "error" in result:
            continue
        
        results.append({
            "algorithm": f"{algo} ({params})",
            "clusters": result["clusters_found"],
            "noise": result["noise_segments"],
            "merged_tasks": len(result["merged_tasks"]),
            "reduction": 1 - (len(result["merged_tasks"]) / result["original_segments"]) if result["original_segments"] > 0 else 0
        })
    
    return {
        "session_id": session_id,
        "comparisons": results
    }


def generate_consolidation_report(session_id: str, 
                                  algorithm: str = "agglomerative",
                                  **kwargs) -> str:
    """
    Generate human-readable report showing consolidation results.
    """
    # Get fragmentation analysis
    frag = analyze_session_fragmentation(session_id)
    
    if "error" in frag:
        return f"Error: {frag['error']}"
    
    # Run clustering
    result = cluster_session_segments(session_id, algorithm, **kwargs)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("TASK CONSOLIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Before consolidation
    lines.append("BEFORE CONSOLIDATION:")
    lines.append(f"  Session: {session_id}")
    lines.append(f"  Total segments: {frag['total_segments']}")
    lines.append(f"  Total duration: {frag['total_duration_minutes']:.1f} minutes")
    lines.append(f"  Average segment: {frag['avg_segment_duration']:.1f} minutes")
    lines.append("")
    lines.append(f"  Fragmentation metrics:")
    lines.append(f"    App switches: {frag['app_switches']}")
    lines.append(f"    Window switches: {frag['window_switches']}")
    lines.append(f"    Task switches: {frag['task_switches']}")
    lines.append(f"    Short segments (< 2 min): {frag['short_segments']} ({frag['short_segment_ratio']:.0%})")
    lines.append("")
    lines.append(f"  Apps used: {frag['apps_used']}")
    for app, count in sorted(frag['app_distribution'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"    {app}: {count} segments")
    lines.append("")
    
    # After consolidation
    lines.append("AFTER CONSOLIDATION:")
    lines.append(f"  Algorithm: {algorithm.upper()}")
    lines.append(f"  Parameters: {kwargs}")
    lines.append("")
    lines.append(f"  Clusters found: {result['clusters_found']}")
    lines.append(f"  Noise segments: {result['noise_segments']}")
    lines.append(f"  Merged tasks: {len(result['merged_tasks'])}")
    
    if result['original_segments'] > 0:
        reduction = (1 - len(result['merged_tasks']) / result['original_segments']) * 100
        lines.append(f"  Fragmentation reduction: {reduction:.1f}%")
    lines.append("")
    
    # Show merged tasks
    if result['merged_tasks']:
        lines.append("CONSOLIDATED TASKS:")
        lines.append("")
        
        for i, task in enumerate(result['merged_tasks'], 1):
            lines.append(f"  Task {i}: {task['task_name']}")
            lines.append(f"    Duration: {task['duration_minutes']:.1f} minutes")
            lines.append(f"    App: {task['app']}")
            lines.append(f"    Context: {task['window_context']}")
            lines.append(f"    Merged from: {task['segment_count']} segments")
            lines.append(f"    Confidence: {task['confidence']:.2f}")
            
            # Show time range
            start = task['start_time'].strftime("%H:%M:%S")
            end = task['end_time'].strftime("%H:%M:%S")
            lines.append(f"    Time range: {start} - {end}")
            lines.append("")
    
    # Distance statistics
    if "distance_stats" in result and result["distance_stats"]:
        stats = result["distance_stats"]
        lines.append("DISTANCE ANALYSIS:")
        lines.append(f"  Segment pairs analyzed: {stats.get('total_pairs', 0)}")
        lines.append(f"  Mean distance: {stats.get('mean_distance', 0):.3f}")
        lines.append(f"  Median distance: {stats.get('median_distance', 0):.3f}")
        lines.append("")
        lines.append(f"  Similarity distribution:")
        lines.append(f"    Very similar (< 0.3): {stats.get('very_similar_pairs', 0)} pairs")
        lines.append(f"    Similar (0.3 - 0.5): {stats.get('similar_pairs', 0)} pairs")
        lines.append(f"    Different (>= 0.5): {stats.get('different_pairs', 0)} pairs")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def find_best_clustering_params(session_id: str) -> Dict:
    """
    Find optimal clustering parameters for a session.
    
    Tests different parameter combinations and recommends best one.
    """
    # Get segments
    clusterer = SegmentClusterer()
    segments = clusterer.load_segments_for_session(session_id)
    
    if not segments:
        return {"error": "No segments found"}
    
    # Analyze distance distribution
    dist_stats = analyze_distance_distribution(segments)
    
    # Test different parameters
    candidates = []
    
    # DBSCAN with varying eps
    for eps in [0.3, 0.4, 0.5, 0.6]:
        result = cluster_session_segments(session_id, "dbscan", eps=eps, min_samples=1)
        if "error" not in result and result["clusters_found"] > 0:
            score = result["clusters_found"] - result["noise_segments"] * 0.5
            candidates.append({
                "algorithm": "dbscan",
                "params": {"eps": eps},
                "clusters": result["clusters_found"],
                "noise": result["noise_segments"],
                "score": score
            })
    
    # Agglomerative with varying threshold
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        result = cluster_session_segments(session_id, "agglomerative", distance_threshold=threshold)
        if "error" not in result and result["clusters_found"] > 0:
            # Prefer fewer clusters (more consolidation)
            ideal_clusters = max(2, len(segments) // 3)
            score = 10 - abs(result["clusters_found"] - ideal_clusters)
            candidates.append({
                "algorithm": "agglomerative",
                "params": {"distance_threshold": threshold},
                "clusters": result["clusters_found"],
                "noise": 0,
                "score": score
            })
    
    if not candidates:
        return {"error": "No valid clustering found"}
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "session_id": session_id,
        "original_segments": len(segments),
        "distance_stats": dist_stats,
        "recommended": candidates[0],
        "all_candidates": candidates[:5]
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze" and len(sys.argv) > 2:
            session_id = sys.argv[2]
            frag = analyze_session_fragmentation(session_id)
            
            print(f"\nFragmentation Analysis for {session_id}:")
            print(f"Total segments: {frag['total_segments']}")
            print(f"Duration: {frag['total_duration_minutes']:.1f} minutes")
            print(f"App switches: {frag['app_switches']}")
            print(f"Window switches: {frag['window_switches']}")
            print(f"Short segments: {frag['short_segments']}")
        
        elif command == "compare" and len(sys.argv) > 2:
            session_id = sys.argv[2]
            comp = compare_clustering_algorithms(session_id)
            
            print(f"\nClustering Algorithm Comparison for {session_id}:")
            for result in comp["comparisons"]:
                print(f"\n{result['algorithm']}:")
                print(f"  Clusters: {result['clusters']}")
                print(f"  Noise: {result['noise']}")
                print(f"  Merged tasks: {result['merged_tasks']}")
                print(f"  Reduction: {result['reduction']:.1%}")
        
        elif command == "report" and len(sys.argv) > 2:
            session_id = sys.argv[2]
            report = generate_consolidation_report(session_id, "agglomerative", distance_threshold=0.5)
            print(report)
        
        elif command == "optimize" and len(sys.argv) > 2:
            session_id = sys.argv[2]
            result = find_best_clustering_params(session_id)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nOptimal Clustering Parameters for {session_id}:")
                print(f"Original segments: {result['original_segments']}")
                print(f"\nRecommended:")
                rec = result["recommended"]
                print(f"  Algorithm: {rec['algorithm']}")
                print(f"  Parameters: {rec['params']}")
                print(f"  Clusters: {rec['clusters']}")
                print(f"  Score: {rec['score']:.2f}")
    
    else:
        # Default: find a session and generate report
        clusterer = SegmentClusterer()
        conn = sqlite3.connect(str(clusterer.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, COUNT(*) as seg_count
            FROM task_segments
            WHERE end_time IS NOT NULL
            GROUP BY session_id
            HAVING seg_count >= 3
            ORDER BY seg_count DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            session_id = row[0]
            report = generate_consolidation_report(session_id, "agglomerative", distance_threshold=0.5)
            print(report)
        else:
            print("No sessions with multiple segments found")
