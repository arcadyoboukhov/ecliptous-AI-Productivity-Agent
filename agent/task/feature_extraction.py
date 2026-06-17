"""
Feature Extraction for Task Recognition

Extracts rich feature vectors from task segments for ML-based classification.
Combines contextual, behavioral, and semantic features into unified vectors.

Feature Categories:
- Contextual: app, window, time of day, duration
- Behavioral: switching patterns, sequence history, duration patterns
- Semantic: NLP features from window titles (embeddings, keywords)
"""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import json
import math


class FeatureExtractor:
    """
    Extracts feature vectors from task segments for ML classification.
    
    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract_features(segment)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize feature extractor.
        
        Args:
            db_path: Path to database (default: auto-detect)
        """
        self.db_path = db_path or self._get_db_path()
        
        # Cache for app encoding
        self._app_vocabulary = {}
        self._window_vocabulary = {}
        
        # Load historical data for behavioral features
        self._load_historical_patterns()
    
    def _get_db_path(self) -> Path:
        """Get the path to the events database."""
        return Path(__file__).parent.parent / "storage" / "events.db"
    
    def _load_historical_patterns(self):
        """Load historical patterns from database for behavioral features."""
        self._task_duration_stats = defaultdict(list)
        self._app_switch_patterns = defaultdict(int)
        self._task_sequence_patterns = []
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load historical task durations
            cursor.execute("""
                SELECT task_id, 
                       (julianday(end_time) - julianday(start_time)) * 24 * 60 as duration_minutes
                FROM task_segments
                WHERE end_time IS NOT NULL
                ORDER BY start_time DESC
                LIMIT 500
            """)
            
            for row in cursor.fetchall():
                task_id = row["task_id"]
                duration = row["duration_minutes"]
                if duration > 0:
                    self._task_duration_stats[task_id].append(duration)
            
            # Load task sequences for pattern analysis
            cursor.execute("""
                SELECT session_id, task_id, start_time
                FROM task_segments
                ORDER BY start_time
            """)
            
            current_session = None
            session_tasks = []
            
            for row in cursor.fetchall():
                if row["session_id"] != current_session:
                    if session_tasks:
                        self._task_sequence_patterns.append(session_tasks)
                    current_session = row["session_id"]
                    session_tasks = []
                session_tasks.append(row["task_id"])
            
            if session_tasks:
                self._task_sequence_patterns.append(session_tasks)
            
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not load historical patterns: {e}")
    
    def extract_features(self, segment: Dict, session_context: Optional[Dict] = None) -> Dict:
        """
        Extract complete feature vector from a segment.
        
        Args:
            segment: Segment dictionary with task_id, app, window_title, etc.
            session_context: Optional session context (previous segments, etc.)
        
        Returns:
            Dictionary with feature categories and values
        """
        features = {}
        
        # Extract all feature categories
        features.update(self._extract_contextual_features(segment))
        features.update(self._extract_behavioral_features(segment, session_context))
        features.update(self._extract_semantic_features(segment))
        
        return features
    
    def _extract_contextual_features(self, segment: Dict) -> Dict:
        """
        Extract contextual features: app, window, time, duration.
        
        Returns dict with:
        - app_name: Raw app name (for classification)
        - window_title: Raw window title (for classification)
        - app_encoded: One-hot encoded app name
        - time_of_day: morning/afternoon/evening/night (encoded)
        - duration_minutes: Segment duration
        - weekday: Day of week (0-6)
        - is_weekend: Boolean
        """
        features = {}
        
        # App encoding (categorical)
        app = segment.get("app", "unknown")
        features["app_name"] = app
        features["app_encoded"] = self._encode_app(app)
        
        # Window title (raw for classification)
        window_title = segment.get("normalized_title", "") or segment.get("window_title", "")
        features["window_title"] = window_title
        
        # Time of day features
        start_time = segment.get("start_time")
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        if start_time:
            hour = start_time.hour
            
            # Time of day (4 categories)
            if 5 <= hour < 12:
                time_category = "morning"
            elif 12 <= hour < 17:
                time_category = "afternoon"
            elif 17 <= hour < 21:
                time_category = "evening"
            else:
                time_category = "night"
            
            features["time_of_day"] = time_category
            features["time_of_day_encoded"] = self._encode_time_of_day(time_category)
            features["hour"] = hour
            
            # Weekday features
            weekday = start_time.weekday()  # 0=Monday, 6=Sunday
            features["weekday"] = weekday
            features["is_weekend"] = 1 if weekday >= 5 else 0
            features["is_work_hours"] = 1 if 9 <= hour < 17 else 0
        
        # Duration
        duration_seconds = segment.get("duration_seconds", 0)
        duration_minutes = duration_seconds / 60
        features["duration_minutes"] = duration_minutes
        features["duration_log"] = math.log1p(duration_minutes)  # Log-scaled for ML
        
        # Duration category (binned)
        if duration_minutes < 2:
            duration_cat = "very_short"
        elif duration_minutes < 10:
            duration_cat = "short"
        elif duration_minutes < 30:
            duration_cat = "medium"
        else:
            duration_cat = "long"
        
        features["duration_category"] = duration_cat
        features["duration_category_encoded"] = self._encode_duration_category(duration_cat)
        
        return features
    
    def _extract_behavioral_features(self, segment: Dict, session_context: Optional[Dict]) -> Dict:
        """
        Extract behavioral features: switching patterns, sequence, history.
        
        Returns dict with:
        - previous_task: Previous task in session
        - task_switch_count: Number of switches in session
        - avg_task_duration: Historical average for this task
        - task_frequency: How often this task appears historically
        """
        features = {}
        
        task_id = segment.get("generic_task", "unknown")
        
        # Previous task in sequence (if available)
        if session_context and "previous_segments" in session_context:
            prev_segments = session_context["previous_segments"]
            if prev_segments:
                features["previous_task"] = prev_segments[-1].get("generic_task", "none")
                features["previous_task_encoded"] = self._encode_task(features["previous_task"])
                
                # Count task switches in current session
                features["task_switch_count"] = len(prev_segments)
                
                # Check if returning to previous task (task switching pattern)
                if len(prev_segments) >= 2:
                    two_tasks_ago = prev_segments[-2].get("generic_task", "")
                    features["is_task_return"] = 1 if two_tasks_ago == task_id else 0
                else:
                    features["is_task_return"] = 0
            else:
                features["previous_task"] = "none"
                features["previous_task_encoded"] = [0] * 6  # Zero vector
                features["task_switch_count"] = 0
                features["is_task_return"] = 0
        else:
            features["previous_task"] = "none"
            features["previous_task_encoded"] = [0] * 6
            features["task_switch_count"] = 0
            features["is_task_return"] = 0
        
        # Historical duration patterns for this task
        if task_id in self._task_duration_stats:
            durations = self._task_duration_stats[task_id]
            features["avg_task_duration"] = sum(durations) / len(durations)
            features["min_task_duration"] = min(durations)
            features["max_task_duration"] = max(durations)
            features["task_frequency"] = len(durations)
        else:
            features["avg_task_duration"] = 0
            features["min_task_duration"] = 0
            features["max_task_duration"] = 0
            features["task_frequency"] = 0
        
        # Duration deviation from historical average
        current_duration = segment.get("duration_minutes", 0)
        if features["avg_task_duration"] > 0:
            features["duration_deviation"] = current_duration - features["avg_task_duration"]
            features["duration_ratio"] = current_duration / features["avg_task_duration"]
        else:
            features["duration_deviation"] = 0
            features["duration_ratio"] = 1.0
        
        # Sequence position features
        if session_context and "session_position" in session_context:
            features["session_position"] = session_context["session_position"]
            features["is_session_start"] = 1 if features["session_position"] == 0 else 0
        else:
            features["session_position"] = 0
            features["is_session_start"] = 1
        
        return features
    
    def _extract_semantic_features(self, segment: Dict) -> Dict:
        """
        Extract semantic/NLP features from window title.
        
        Returns dict with:
        - title_tokens: Tokenized and normalized words
        - title_keywords: Extracted keywords
        - title_embedding: Text embedding vector (if available)
        - domain_indicators: Domain-specific keyword presence
        """
        features = {}
        
        window_title = segment.get("normalized_title", "") or segment.get("window_title", "")
        
        # Tokenize and normalize
        tokens = self._tokenize_title(window_title)
        features["title_tokens"] = tokens
        features["title_length"] = len(tokens)
        
        # Extract keywords (remove stopwords)
        keywords = self._extract_keywords(tokens)
        features["title_keywords"] = keywords
        features["keyword_count"] = len(keywords)
        
        # Domain indicators (coding, documentation, communication, etc.)
        domain_features = self._extract_domain_indicators(window_title, tokens)
        features.update(domain_features)
        
        # Simple text features (without heavy embeddings)
        features["has_code_indicators"] = self._has_code_indicators(window_title)
        features["has_doc_indicators"] = self._has_documentation_indicators(window_title)
        features["has_comm_indicators"] = self._has_communication_indicators(window_title)
        
        # Keyword-based simple embedding (bag of words)
        features["title_bow"] = self._create_bow_vector(keywords)
        
        return features
    
    def _tokenize_title(self, title: str) -> List[str]:
        """Tokenize window title into normalized words."""
        if not title:
            return []
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        title = re.sub(r'[^\w\s-]', ' ', title)
        
        # Split into tokens
        tokens = title.split()
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """Extract keywords by removing stopwords."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'there', 'here'
        }
        
        keywords = [t for t in tokens if t not in stopwords]
        return keywords
    
    def _extract_domain_indicators(self, title: str, tokens: List[str]) -> Dict:
        """Extract domain-specific indicator features."""
        features = {}
        
        # Coding domain
        coding_keywords = {'code', 'git', 'github', 'vscode', 'python', 'javascript',
                          'programming', 'develop', 'debug', 'commit', 'branch', 'repo'}
        features["domain_coding"] = 1 if any(k in tokens for k in coding_keywords) else 0
        
        # Documentation domain
        doc_keywords = {'doc', 'documentation', 'readme', 'wiki', 'guide', 'tutorial',
                       'manual', 'help', 'reference'}
        features["domain_documentation"] = 1 if any(k in tokens for k in doc_keywords) else 0
        
        # Communication domain
        comm_keywords = {'chat', 'slack', 'teams', 'email', 'message', 'meet', 'zoom',
                        'call', 'discord', 'telegram'}
        features["domain_communication"] = 1 if any(k in tokens for k in comm_keywords) else 0
        
        # Research domain
        research_keywords = {'search', 'google', 'stackoverflow', 'research', 'article',
                            'blog', 'news', 'reddit'}
        features["domain_research"] = 1 if any(k in tokens for k in research_keywords) else 0
        
        # Productivity domain
        prod_keywords = {'calendar', 'todo', 'task', 'notion', 'trello', 'jira',
                        'confluence', 'spreadsheet', 'excel'}
        features["domain_productivity"] = 1 if any(k in tokens for k in prod_keywords) else 0
        
        # Content domain
        content_keywords = {'youtube', 'video', 'music', 'spotify', 'netflix',
                           'twitter', 'facebook', 'instagram', 'social'}
        features["domain_content"] = 1 if any(k in tokens for k in content_keywords) else 0
        
        return features
    
    def _has_code_indicators(self, title: str) -> int:
        """Check for coding-related indicators."""
        indicators = ['.py', '.js', '.ts', '.java', '.cpp', 'github', 'gitlab',
                     'vscode', 'visual studio', 'pycharm', 'intellij']
        return 1 if any(ind in title.lower() for ind in indicators) else 0
    
    def _has_documentation_indicators(self, title: str) -> int:
        """Check for documentation indicators."""
        indicators = ['docs', 'readme', 'wiki', 'documentation', 'api reference']
        return 1 if any(ind in title.lower() for ind in indicators) else 0
    
    def _has_communication_indicators(self, title: str) -> int:
        """Check for communication app indicators."""
        indicators = ['slack', 'teams', 'discord', 'telegram', 'whatsapp',
                     'messenger', 'chat', 'zoom', 'meet']
        return 1 if any(ind in title.lower() for ind in indicators) else 0
    
    def _create_bow_vector(self, keywords: List[str], max_features: int = 50) -> List[float]:
        """
        Create simple bag-of-words vector from keywords.
        
        This is a lightweight alternative to embeddings.
        """
        # Get top keywords from vocabulary
        if not hasattr(self, '_keyword_vocabulary'):
            self._build_keyword_vocabulary()
        
        # Create zero vector
        vector = [0.0] * max_features
        
        # Fill in presence indicators
        for keyword in keywords:
            if keyword in self._keyword_vocabulary:
                idx = self._keyword_vocabulary[keyword]
                if idx < max_features:
                    vector[idx] = 1.0
        
        return vector
    
    def _build_keyword_vocabulary(self, max_keywords: int = 50):
        """Build keyword vocabulary from historical data."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT feature_vector
                FROM task_segments
                WHERE feature_vector IS NOT NULL
                LIMIT 1000
            """)
            
            keyword_counter = Counter()
            
            for row in cursor.fetchall():
                try:
                    features = json.loads(row[0])
                    window_title = features.get("active_window_title", "")
                    tokens = self._tokenize_title(window_title)
                    keywords = self._extract_keywords(tokens)
                    keyword_counter.update(keywords)
                except:
                    continue
            
            # Get top keywords
            top_keywords = [k for k, _ in keyword_counter.most_common(max_keywords)]
            self._keyword_vocabulary = {k: i for i, k in enumerate(top_keywords)}
            
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not build keyword vocabulary: {e}")
            self._keyword_vocabulary = {}
    
    def _encode_app(self, app: str) -> List[float]:
        """One-hot encode app name."""
        # Common apps (can be expanded)
        common_apps = [
            'firefox.exe', 'chrome.exe', 'code.exe', 'notepad.exe',
            'slack.exe', 'teams.exe', 'outlook.exe', 'excel.exe',
            'word.exe', 'pycharm64.exe', 'sublime_text.exe', 'unknown'
        ]
        
        # Normalize app name
        app_normalized = app.lower() if app else 'unknown'
        
        # Create one-hot vector
        vector = [0.0] * len(common_apps)
        if app_normalized in common_apps:
            idx = common_apps.index(app_normalized)
            vector[idx] = 1.0
        else:
            # Unknown app
            vector[-1] = 1.0
        
        return vector
    
    def _encode_time_of_day(self, time_category: str) -> List[float]:
        """One-hot encode time of day."""
        categories = ['morning', 'afternoon', 'evening', 'night']
        vector = [0.0] * len(categories)
        if time_category in categories:
            idx = categories.index(time_category)
            vector[idx] = 1.0
        return vector
    
    def _encode_duration_category(self, duration_cat: str) -> List[float]:
        """One-hot encode duration category."""
        categories = ['very_short', 'short', 'medium', 'long']
        vector = [0.0] * len(categories)
        if duration_cat in categories:
            idx = categories.index(duration_cat)
            vector[idx] = 1.0
        return vector
    
    def _encode_task(self, task_id: str) -> List[float]:
        """One-hot encode task category."""
        # Core task categories
        tasks = [
            'administrative_work', 'deep_development', 'technical_research',
            'context_switching', 'team_meeting', 'strategic_planning'
        ]
        
        vector = [0.0] * len(tasks)
        
        # Extract base category from task_id
        task_lower = task_id.lower()
        for i, task in enumerate(tasks):
            if task.replace('_', ' ') in task_lower or task in task_lower:
                vector[i] = 1.0
                break
        
        return vector
    
    def extract_feature_vector_flat(self, segment: Dict, session_context: Optional[Dict] = None) -> List[float]:
        """
        Extract flattened feature vector suitable for ML models.
        
        Returns a single list of numeric features.
        """
        features = self.extract_features(segment, session_context)
        
        # Flatten all numeric features into single vector
        flat_vector = []
        
        # Contextual features
        flat_vector.extend(features.get("app_encoded", [0] * 12))
        flat_vector.extend(features.get("time_of_day_encoded", [0] * 4))
        flat_vector.extend(features.get("duration_category_encoded", [0] * 4))
        flat_vector.append(features.get("duration_log", 0))
        flat_vector.append(features.get("hour", 0) / 24.0)  # Normalized
        flat_vector.append(features.get("weekday", 0) / 7.0)  # Normalized
        flat_vector.append(features.get("is_weekend", 0))
        flat_vector.append(features.get("is_work_hours", 0))
        
        # Behavioral features
        flat_vector.extend(features.get("previous_task_encoded", [0] * 6))
        flat_vector.append(min(features.get("task_switch_count", 0) / 10.0, 1.0))  # Normalized
        flat_vector.append(features.get("is_task_return", 0))
        flat_vector.append(min(features.get("task_frequency", 0) / 100.0, 1.0))  # Normalized
        flat_vector.append(features.get("duration_ratio", 1.0))
        flat_vector.append(features.get("is_session_start", 0))
        
        # Semantic features
        flat_vector.append(min(features.get("title_length", 0) / 20.0, 1.0))  # Normalized
        flat_vector.append(min(features.get("keyword_count", 0) / 10.0, 1.0))  # Normalized
        flat_vector.append(features.get("domain_coding", 0))
        flat_vector.append(features.get("domain_documentation", 0))
        flat_vector.append(features.get("domain_communication", 0))
        flat_vector.append(features.get("domain_research", 0))
        flat_vector.append(features.get("domain_productivity", 0))
        flat_vector.append(features.get("domain_content", 0))
        flat_vector.append(features.get("has_code_indicators", 0))
        flat_vector.append(features.get("has_doc_indicators", 0))
        flat_vector.append(features.get("has_comm_indicators", 0))
        
        # BOW features (top 50 keywords)
        flat_vector.extend(features.get("title_bow", [0] * 50))
        
        return flat_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in flat vector."""
        names = []
        
        # App encoding (12 features)
        apps = ['firefox', 'chrome', 'code', 'notepad', 'slack', 'teams',
                'outlook', 'excel', 'word', 'pycharm', 'sublime', 'unknown']
        names.extend([f"app_{a}" for a in apps])
        
        # Time of day (4 features)
        names.extend(['time_morning', 'time_afternoon', 'time_evening', 'time_night'])
        
        # Duration category (4 features)
        names.extend(['dur_very_short', 'dur_short', 'dur_medium', 'dur_long'])
        
        # Other contextual (5 features)
        names.extend(['duration_log', 'hour_norm', 'weekday_norm', 'is_weekend', 'is_work_hours'])
        
        # Previous task (6 features)
        tasks = ['admin', 'development', 'research', 'switching', 'meeting', 'planning']
        names.extend([f"prev_{t}" for t in tasks])
        
        # Behavioral (5 features)
        names.extend(['switch_count_norm', 'is_task_return', 'task_freq_norm',
                     'duration_ratio', 'is_session_start'])
        
        # Semantic (11 features)
        names.extend(['title_len_norm', 'keyword_count_norm',
                     'domain_coding', 'domain_docs', 'domain_comm',
                     'domain_research', 'domain_prod', 'domain_content',
                     'has_code', 'has_docs', 'has_comm'])
        
        # BOW (50 features)
        names.extend([f"bow_{i}" for i in range(50)])
        
        return names


def extract_features_for_segments(segment_ids: Optional[List[int]] = None, 
                                  limit: int = 100) -> List[Tuple[int, List[float], Dict]]:
    """
    Extract feature vectors for segments from database.
    
    Args:
        segment_ids: Optional list of specific segment IDs
        limit: Maximum number of segments to process
    
    Returns:
        List of (segment_id, feature_vector, metadata) tuples
    """
    extractor = FeatureExtractor()
    
    db_path = extractor._get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build query
    if segment_ids:
        placeholders = ','.join('?' * len(segment_ids))
        query = f"""
            SELECT id, session_id, task_id, start_time, end_time,
                   confidence, reason, feature_vector
            FROM task_segments
            WHERE id IN ({placeholders})
            ORDER BY start_time
        """
        cursor.execute(query, segment_ids)
    else:
        query = """
            SELECT id, session_id, task_id, start_time, end_time,
                   confidence, reason, feature_vector
            FROM task_segments
            WHERE end_time IS NOT NULL
            ORDER BY start_time DESC
            LIMIT ?
        """
        cursor.execute(query, (limit,))
    
    rows = cursor.fetchall()
    
    results = []
    session_segments_cache = {}
    
    for row in rows:
        segment_id = row["id"]
        session_id = row["session_id"]
        
        # Parse segment data
        start_time = datetime.fromisoformat(row["start_time"])
        end_time = datetime.fromisoformat(row["end_time"]) if row["end_time"] else None
        
        if not end_time:
            continue
        
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Parse feature vector for app/window
        try:
            stored_features = json.loads(row["feature_vector"]) if row["feature_vector"] else {}
        except:
            stored_features = {}
        
        app = stored_features.get("active_app", "unknown")
        window_title = stored_features.get("active_window_title", "")
        
        # Normalize window title
        from agent.task.smart_naming import normalize_window_title
        normalized_title = normalize_window_title(window_title) if window_title else ""
        
        # Extract generic task from reason
        reason = row["reason"] or ""
        generic_task = "unknown"
        if "_" in reason:
            parts = reason.split("_")
            if parts[0] in ["initial", "continuing"]:
                generic_task = "_".join(parts[1:])
        
        # Build segment dict
        segment = {
            "id": segment_id,
            "session_id": session_id,
            "task_id": row["task_id"],
            "generic_task": generic_task,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60,
            "app": app,
            "window_title": window_title,
            "normalized_title": normalized_title,
            "confidence": row["confidence"]
        }
        
        # Get session context (previous segments in same session)
        if session_id not in session_segments_cache:
            session_segments_cache[session_id] = []
        
        previous_segments = session_segments_cache[session_id].copy()
        session_context = {
            "previous_segments": previous_segments,
            "session_position": len(previous_segments)
        }
        
        # Extract features
        feature_vector = extractor.extract_feature_vector_flat(segment, session_context)
        
        # Store metadata
        metadata = {
            "segment_id": segment_id,
            "session_id": session_id,
            "task_id": row["task_id"],
            "generic_task": generic_task,
            "start_time": start_time.isoformat(),
            "confidence": row["confidence"]
        }
        
        results.append((segment_id, feature_vector, metadata))
        
        # Update session cache
        session_segments_cache[session_id].append(segment)
    
    conn.close()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing feature extraction...")
    
    extractor = FeatureExtractor()
    
    # Get sample segment
    results = extract_features_for_segments(limit=5)
    
    if results:
        print(f"\nExtracted features for {len(results)} segments")
        
        # Show first segment
        seg_id, features, metadata = results[0]
        print(f"\nSegment ID: {seg_id}")
        print(f"Task: {metadata['task_id']}")
        print(f"Generic: {metadata['generic_task']}")
        print(f"Feature vector length: {len(features)}")
        print(f"Sample features: {features[:20]}")
        
        # Show feature names
        names = extractor.get_feature_names()
        print(f"\nTotal features: {len(names)}")
        print(f"Feature categories:")
        print(f"  - Contextual: app (12), time (4), duration (5), other (5)")
        print(f"  - Behavioral: previous_task (6), patterns (5)")
        print(f"  - Semantic: domain (11), bow (50)")
        print(f"  Total: {len(names)} features")
    else:
        print("No segments found in database")
