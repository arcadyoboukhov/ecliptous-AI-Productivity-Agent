"""
Test v2 classification system with expanded generic names and multi-layer logic.

Tests:
1. App-based classification
2. Window title classification
3. Behavioral fallback
4. No "unknown" results
"""

import sqlite3
from pathlib import Path
from collections import Counter
from agent.task.feature_extraction import FeatureExtractor
from agent.task.core_tasks import get_task_recommendation


def get_db_path() -> Path:
    """Get the path to the events database."""
    return Path(__file__).parent.parent / "storage" / "events.db"


def test_classification_on_segments():
    """Test classification on all existing segments."""
    db_path = get_db_path()
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Load all segments with feature vectors
    cursor.execute("""
        SELECT 
            id,
            task_id,
            start_time,
            end_time,
            confidence as old_confidence,
            reason as old_reason,
            feature_vector
        FROM task_segments
        WHERE end_time IS NOT NULL AND feature_vector IS NOT NULL
        ORDER BY start_time DESC
        LIMIT 50
    """)
    
    segments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    print(f"\n{'='*80}")
    print(f"Classification Test - {len(segments)} segments")
    print(f"{'='*80}\n")
    
    if not segments:
        print("No segments found with feature vectors. Run feature extraction first:")
        print("  python -m agent.task.feature_extraction")
        return []
    
    # Parse feature vectors and classify
    extractor = FeatureExtractor(db_path)
    
    results = []
    classification_reasons = Counter()
    task_distribution = Counter()
    unknown_count = 0
    
    for i, segment in enumerate(segments, 1):
        # Parse feature vector
        import json
        try:
            features = json.loads(segment["feature_vector"])
        except:
            print(f"Warning: Could not parse feature vector for segment {segment['id']}")
            continue
        
        # Classify
        task_id, confidence, reason = get_task_recommendation(features)
        
        app = features.get("app_name", "unknown")
        window_title = features.get("window_title", "")
        
        results.append({
            "segment_id": segment["id"],
            "app": app,
            "window_title": window_title[:50] if window_title else "",
            "old_task": segment["task_id"],
            "old_confidence": segment["old_confidence"],
            "new_task": task_id,
            "new_confidence": confidence,
            "reason": reason
        })
        
        classification_reasons[reason] += 1
        task_distribution[task_id] += 1
        
        if task_id == "unknown":
            unknown_count += 1
        
        # Print progress - show first 20 and any unknowns
        if i <= 20 or task_id == "unknown":
            print(f"Segment {segment['id']:4d}:")
            print(f"  App: {app[:30]:<30} Window: {window_title[:40]}")
            print(f"  Old: {segment['task_id']:<25} (conf={segment['old_confidence']:.2f})")
            print(f"  New: {task_id:<25} (conf={confidence:.2f}, {reason})")
            if segment['task_id'] != task_id:
                print(f"  → CHANGED!")
            print()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total segments classified: {len(segments)}")
    print(f"Unknown classifications: {unknown_count} ({unknown_count/len(segments)*100:.1f}%)")
    print(f"Successfully classified: {len(segments) - unknown_count} ({(len(segments)-unknown_count)/len(segments)*100:.1f}%)\n")
    
    print("Classification Methods:")
    for reason, count in classification_reasons.most_common():
        print(f"  {reason:30s}: {count:3d} ({count/len(segments)*100:.1f}%)")
    
    print("\nTask Distribution:")
    for task, count in task_distribution.most_common():
        print(f"  {task:30s}: {count:3d} ({count/len(segments)*100:.1f}%)")
    
    # Check for improvement
    if unknown_count == 0:
        print(f"\n{'✓'*40}")
        print("SUCCESS! Zero unknown classifications!")
        print(f"{'✓'*40}\n")
    else:
        print(f"\n{'!'*40}")
        print(f"WARNING: Still have {unknown_count} unknown classifications")
        print(f"{'!'*40}\n")
        
        # Show unknown segments
        print("Unknown segments:")
        for result in results:
            if result["new_task"] == "unknown":
                print(f"  ID {result['segment_id']:4d}: {result['app']:30s} | {result['window_title']}")
    
    return results


def test_specific_apps():
    """Test classification for specific known apps."""
    print(f"\n{'='*80}")
    print("Specific App Tests")
    print(f"{'='*80}\n")
    
    test_cases = [
        {"app": "Code.exe", "window_title": "main.py - Visual Studio Code", "expected": "deep_development"},
        {"app": "firefox.exe", "window_title": "GitHub - Pull Request #123", "expected": "code_review"},
        {"app": "firefox.exe", "window_title": "YouTube - How to Python", "expected": "content_consumption"},
        {"app": "firefox.exe", "window_title": "Stack Overflow - Python question", "expected": "technical_research"},
        {"app": "OUTLOOK.EXE", "window_title": "Inbox - Microsoft Outlook", "expected": "email_communication"},
        {"app": "Teams.exe", "window_title": "Chat - John Doe", "expected": "chat_messaging"},
        {"app": "Teams.exe", "window_title": "Call - Team Meeting", "expected": "video_conferencing"},
        {"app": "explorer.exe", "window_title": "Documents - File Explorer", "expected": "file_management"},
        {"app": "powershell.exe", "window_title": "Windows PowerShell", "expected": "system_maintenance"},
        {"app": "unknown_app.exe", "window_title": "Some random window", "expected": "general_productivity"},
    ]
    
    extractor = FeatureExtractor()
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        # Create minimal segment
        segment = {
            "app": test["app"],
            "window_title": test["window_title"],
            "normalized_title": test["window_title"],
            "start_time": "2024-01-01T10:00:00+00:00",
            "end_time": "2024-01-01T10:15:00+00:00",
            "duration_seconds": 900
        }
        
        features = extractor.extract_features(segment)
        task_id, confidence, reason = get_task_recommendation(features)
        
        status = "✓" if task_id == test["expected"] else "✗"
        if task_id == test["expected"]:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {test['app']:20s} → {task_id:25s} (conf={confidence:.2f}, {reason})")
        if task_id != test["expected"]:
            print(f"    Expected: {test['expected']}")
    
    print(f"\nTest Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.0f}%)")
    
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")


if __name__ == "__main__":
    # Test on actual segments
    results = test_classification_on_segments()
    
    # Test specific scenarios
    print()
    test_specific_apps()
