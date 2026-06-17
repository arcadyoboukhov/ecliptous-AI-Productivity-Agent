"""
Segment Distance Metrics

Custom distance functions for measuring similarity between task segments.
Used by clustering algorithms to group related segments into unified tasks.
"""

import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import Counter


def calculate_segment_distance(seg1: Dict, seg2: Dict, 
                               weights: Dict[str, float] = None) -> float:
    """
    Calculate composite distance between two segments.
    
    Combines multiple distance metrics with configurable weights:
    - Text similarity (window titles)
    - App similarity
    - Temporal proximity
    - Behavioral similarity (generic task)
    
    Args:
        seg1, seg2: Segment dictionaries
        weights: Dict with keys: text, app, temporal, behavioral
    
    Returns:
        Distance value (0 = identical, 1+ = different)
    """
    if weights is None:
        weights = {
            "text": 0.4,       # Window title similarity
            "app": 0.3,        # Application context
            "temporal": 0.2,   # Time proximity
            "behavioral": 0.1  # Task category
        }
    
    # Calculate individual distances
    text_dist = text_distance(seg1, seg2)
    app_dist = app_distance(seg1, seg2)
    temporal_dist = temporal_distance(seg1, seg2)
    behavioral_dist = behavioral_distance(seg1, seg2)
    
    # Weighted sum
    total_distance = (
        weights["text"] * text_dist +
        weights["app"] * app_dist +
        weights["temporal"] * temporal_dist +
        weights["behavioral"] * behavioral_dist
    )
    
    return total_distance


def text_distance(seg1: Dict, seg2: Dict) -> float:
    """
    Calculate text similarity distance between window titles.
    
    Uses simple cosine similarity of word vectors (bag of words).
    Returns 0 for identical, 1 for completely different.
    """
    # Get normalized titles
    title1 = seg1.get("normalized_title", "") or seg1.get("window_title", "")
    title2 = seg2.get("normalized_title", "") or seg2.get("window_title", "")
    
    if not title1 or not title2:
        return 1.0  # Max distance if missing
    
    # Tokenize
    tokens1 = _tokenize_for_similarity(title1)
    tokens2 = _tokenize_for_similarity(title2)
    
    if not tokens1 or not tokens2:
        return 1.0
    
    # Calculate cosine similarity
    similarity = _cosine_similarity(tokens1, tokens2)
    
    # Convert to distance (0 = identical, 1 = different)
    distance = 1.0 - similarity
    
    return distance


def app_distance(seg1: Dict, seg2: Dict) -> float:
    """
    Calculate application similarity distance.
    
    Returns 0 if same app, 1 if different apps.
    Can be refined with app categories (browsers, IDEs, etc.)
    """
    app1 = seg1.get("app", "").lower()
    app2 = seg2.get("app", "").lower()
    
    if not app1 or not app2:
        return 0.5  # Neutral if unknown
    
    # Exact match
    if app1 == app2:
        return 0.0
    
    # Check if same category (browsers, IDEs, etc.)
    category1 = _get_app_category(app1)
    category2 = _get_app_category(app2)
    
    if category1 == category2:
        return 0.3  # Same category but different app
    
    return 1.0  # Different categories


def temporal_distance(seg1: Dict, seg2: Dict, max_gap_minutes: int = 60) -> float:
    """
    Calculate temporal proximity distance.
    
    Segments close in time are more likely to be part of same task.
    
    Args:
        max_gap_minutes: Maximum time gap to consider (normalized to 1.0)
    
    Returns:
        0 = same time, 1+ = far apart
    """
    time1 = seg1.get("start_time")
    time2 = seg2.get("start_time")
    
    # Parse if string
    if isinstance(time1, str):
        time1 = datetime.fromisoformat(time1)
    if isinstance(time2, str):
        time2 = datetime.fromisoformat(time2)
    
    if not time1 or not time2:
        return 1.0
    
    # Calculate time gap
    time_gap = abs((time1 - time2).total_seconds() / 60)  # minutes
    
    # Normalize to [0, 1] range
    normalized_distance = min(time_gap / max_gap_minutes, 1.0)
    
    return normalized_distance


def behavioral_distance(seg1: Dict, seg2: Dict) -> float:
    """
    Calculate behavioral similarity distance based on generic task category.
    
    Same task category = lower distance.
    """
    task1 = seg1.get("generic_task", "unknown")
    task2 = seg2.get("generic_task", "unknown")
    
    if task1 == "unknown" or task2 == "unknown":
        return 0.5  # Neutral
    
    if task1 == task2:
        return 0.0  # Same task
    
    # Check if related tasks
    related_pairs = [
        {"deep_development", "technical_research"},
        {"team_meeting", "administrative_work"},
    ]
    
    for pair in related_pairs:
        if task1 in pair and task2 in pair:
            return 0.4  # Related tasks
    
    return 1.0  # Different tasks


def _tokenize_for_similarity(text: str) -> List[str]:
    """Tokenize text for similarity calculation."""
    # Lowercase
    text = text.lower()
    
    # Remove special chars
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Remove stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
    }
    
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    return tokens


def _cosine_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Calculate cosine similarity between two token lists.
    
    Returns value in [0, 1] where 1 = identical, 0 = no overlap.
    """
    # Create frequency vectors
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Get all unique tokens
    all_tokens = set(counter1.keys()) | set(counter2.keys())
    
    if not all_tokens:
        return 0.0
    
    # Build vectors
    vec1 = [counter1.get(token, 0) for token in all_tokens]
    vec2 = [counter2.get(token, 0) for token in all_tokens]
    
    # Calculate dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    
    # Calculate magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1))
    mag2 = math.sqrt(sum(v * v for v in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = dot_product / (mag1 * mag2)
    
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


def _get_app_category(app: str) -> str:
    """Categorize app into broad categories."""
    app_lower = app.lower()
    
    # Browsers
    if any(browser in app_lower for browser in ['firefox', 'chrome', 'edge', 'safari', 'brave']):
        return "browser"
    
    # IDEs and code editors
    if any(ide in app_lower for ide in ['code', 'vscode', 'pycharm', 'intellij', 'sublime', 'atom', 'vim', 'emacs']):
        return "ide"
    
    # Communication
    if any(comm in app_lower for comm in ['slack', 'teams', 'discord', 'telegram', 'zoom', 'meet']):
        return "communication"
    
    # Office
    if any(office in app_lower for office in ['word', 'excel', 'powerpoint', 'outlook', 'onenote']):
        return "office"
    
    # Terminal
    if any(term in app_lower for term in ['terminal', 'cmd', 'powershell', 'bash', 'iterm']):
        return "terminal"
    
    return "other"


def calculate_pairwise_distances(segments: List[Dict], weights: Dict[str, float] = None) -> List[List[float]]:
    """
    Calculate pairwise distance matrix for all segments.
    
    Returns:
        NxN distance matrix where N = len(segments)
    """
    n = len(segments)
    distances = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_segment_distance(segments[i], segments[j], weights)
            distances[i][j] = dist
            distances[j][i] = dist  # Symmetric
    
    return distances


def analyze_distance_distribution(segments: List[Dict]) -> Dict:
    """
    Analyze distance distribution to help choose clustering parameters.
    
    Returns statistics about segment similarities.
    """
    if len(segments) < 2:
        return {"message": "Need at least 2 segments"}
    
    # Calculate all pairwise distances
    distances = calculate_pairwise_distances(segments)
    
    # Flatten upper triangle (excluding diagonal)
    all_distances = []
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            all_distances.append(distances[i][j])
    
    if not all_distances:
        return {}
    
    # Calculate statistics
    all_distances.sort()
    
    stats = {
        "total_pairs": len(all_distances),
        "min_distance": min(all_distances),
        "max_distance": max(all_distances),
        "mean_distance": sum(all_distances) / len(all_distances),
        "median_distance": all_distances[len(all_distances) // 2],
        "quartiles": {
            "q1": all_distances[len(all_distances) // 4],
            "q2": all_distances[len(all_distances) // 2],
            "q3": all_distances[3 * len(all_distances) // 4]
        },
        "very_similar_pairs": sum(1 for d in all_distances if d < 0.3),
        "similar_pairs": sum(1 for d in all_distances if 0.3 <= d < 0.5),
        "different_pairs": sum(1 for d in all_distances if d >= 0.5)
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    seg1 = {
        "app": "firefox.exe",
        "window_title": "ChatGPT – Mozilla Firefox",
        "normalized_title": "ChatGPT",
        "generic_task": "administrative_work",
        "start_time": "2026-01-29T22:10:00"
    }
    
    seg2 = {
        "app": "firefox.exe",
        "window_title": "ChatGPT - Writing Code – Mozilla Firefox",
        "normalized_title": "ChatGPT - Writing Code",
        "generic_task": "administrative_work",
        "start_time": "2026-01-29T22:12:00"
    }
    
    seg3 = {
        "app": "code.exe",
        "window_title": "main.py - VSCode",
        "normalized_title": "main.py - VSCode",
        "generic_task": "deep_development",
        "start_time": "2026-01-29T22:15:00"
    }
    
    print("Distance between ChatGPT segments:")
    dist = calculate_segment_distance(seg1, seg2)
    print(f"  Total: {dist:.3f}")
    print(f"  Text: {text_distance(seg1, seg2):.3f}")
    print(f"  App: {app_distance(seg1, seg2):.3f}")
    print(f"  Temporal: {temporal_distance(seg1, seg2):.3f}")
    print(f"  Behavioral: {behavioral_distance(seg1, seg2):.3f}")
    
    print("\nDistance between ChatGPT and VSCode:")
    dist = calculate_segment_distance(seg1, seg3)
    print(f"  Total: {dist:.3f}")
    print(f"  Text: {text_distance(seg1, seg3):.3f}")
    print(f"  App: {app_distance(seg1, seg3):.3f}")
    print(f"  Temporal: {temporal_distance(seg1, seg3):.3f}")
    print(f"  Behavioral: {behavioral_distance(seg1, seg3):.3f}")
