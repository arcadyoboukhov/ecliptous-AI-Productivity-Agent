"""
Feature Vector Analysis and Export

Tools for analyzing extracted feature vectors and exporting them for ML training.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from agent.task.feature_extraction import FeatureExtractor, extract_features_for_segments


def export_feature_dataset(
    output_path: Optional[Path] = None,
    limit: int = 500,
    include_feature_names: bool = True
) -> Dict:
    """
    Export complete feature dataset for ML training.
    
    Args:
        output_path: Path to save JSON file (default: agent/task/feature_dataset.json)
        limit: Maximum segments to export
        include_feature_names: Include feature names in export
    
    Returns:
        Dataset dictionary
    """
    print(f"Extracting features for up to {limit} segments...")
    
    # Extract features
    results = extract_features_for_segments(limit=limit)
    
    if not results:
        print("No segments found")
        return {"segments": [], "feature_count": 0}
    
    print(f"Extracted features for {len(results)} segments")
    
    # Get feature names
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # Build dataset
    dataset = {
        "version": "1.0",
        "feature_count": len(feature_names),
        "segment_count": len(results),
        "feature_names": feature_names if include_feature_names else [],
        "segments": []
    }
    
    for seg_id, features, metadata in results:
        segment_data = {
            "segment_id": seg_id,
            "task_id": metadata["task_id"],
            "generic_task": metadata["generic_task"],
            "confidence": metadata["confidence"],
            "start_time": metadata["start_time"],
            "features": features
        }
        dataset["segments"].append(segment_data)
    
    # Save to file
    if output_path is None:
        output_path = Path(__file__).parent / "feature_dataset.json"
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Total segments: {len(results)}")
    print(f"Features per segment: {len(feature_names)}")
    
    return dataset


def analyze_feature_importance(dataset: Optional[Dict] = None, top_n: int = 20) -> Dict:
    """
    Analyze feature importance using simple variance-based ranking.
    
    High variance features are more likely to be discriminative.
    """
    if dataset is None:
        print("Loading dataset...")
        dataset_path = Path(__file__).parent / "feature_dataset.json"
        if not dataset_path.exists():
            print("No dataset found. Run export_feature_dataset first.")
            return {}
        
        with open(dataset_path) as f:
            dataset = json.load(f)
    
    segments = dataset["segments"]
    feature_names = dataset["feature_names"]
    
    if not segments:
        return {}
    
    # Calculate variance for each feature
    n_features = len(feature_names)
    n_segments = len(segments)
    
    variances = []
    means = []
    
    for feat_idx in range(n_features):
        # Get all values for this feature
        values = [seg["features"][feat_idx] for seg in segments]
        
        # Calculate mean
        mean_val = sum(values) / len(values)
        means.append(mean_val)
        
        # Calculate variance
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        variances.append(variance)
    
    # Rank features by variance
    ranked_indices = sorted(range(n_features), key=lambda i: variances[i], reverse=True)
    
    # Get top N features
    top_features = []
    for i in range(min(top_n, len(ranked_indices))):
        idx = ranked_indices[i]
        values = [seg["features"][idx] for seg in segments]
        std = math.sqrt(variances[idx])
        
        top_features.append({
            "feature_name": feature_names[idx],
            "variance": variances[idx],
            "mean": means[idx],
            "std": std
        })
    
    return {
        "top_features": top_features,
        "total_features": len(feature_names)
    }


def get_feature_statistics(dataset: Optional[Dict] = None) -> Dict:
    """
    Calculate statistics about features across all segments.
    """
    if dataset is None:
        dataset_path = Path(__file__).parent / "feature_dataset.json"
        if not dataset_path.exists():
            return {}
        
        with open(dataset_path) as f:
            dataset = json.load(f)
    
    segments = dataset["segments"]
    feature_names = dataset["feature_names"]
    
    if not segments:
        return {}
    
    # Collect all feature values
    all_values = []
    for seg in segments:
        all_values.extend(seg["features"])
    
    # Calculate statistics
    mean_val = sum(all_values) / len(all_values)
    variance = sum((v - mean_val) ** 2 for v in all_values) / len(all_values)
    std_val = math.sqrt(variance)
    
    zero_count = sum(1 for v in all_values if v == 0)
    sparsity = zero_count / len(all_values)
    
    stats = {
        "feature_count": len(feature_names),
        "segment_count": len(segments),
        "feature_stats": {
            "mean": mean_val,
            "std": std_val,
            "min": min(all_values),
            "max": max(all_values)
        },
        "sparsity": sparsity,
        "task_distribution": defaultdict(int)
    }
    
    # Count tasks
    for seg in segments:
        task = seg["generic_task"]
        stats["task_distribution"][task] += 1
    
    stats["task_distribution"] = dict(stats["task_distribution"])
    
    return stats


def visualize_feature_correlation(dataset: Optional[Dict] = None, save_path: Optional[Path] = None):
    """
    Calculate feature correlation matrix.
    
    Useful for identifying redundant features.
    Note: Simplified version without numpy - only finds high correlations.
    """
    if dataset is None:
        dataset_path = Path(__file__).parent / "feature_dataset.json"
        if not dataset_path.exists():
            print("No dataset found")
            return None
        
        with open(dataset_path) as f:
            dataset = json.load(f)
    
    segments = dataset["segments"]
    feature_names = dataset["feature_names"]
    
    if not segments:
        return None
    
    print("Correlation analysis requires numpy. Skipping for now.")
    return {"high_correlation_pairs": []}


def create_feature_summary_report(output_path: Optional[Path] = None) -> str:
    """
    Create comprehensive feature summary report.
    """
    print("Creating feature summary report...")
    
    # Export dataset if not exists
    dataset_path = Path(__file__).parent / "feature_dataset.json"
    if not dataset_path.exists():
        dataset = export_feature_dataset()
    else:
        with open(dataset_path) as f:
            dataset = json.load(f)
    
    # Get statistics
    stats = get_feature_statistics(dataset)
    importance = analyze_feature_importance(dataset, top_n=15)
    
    # Build report
    report = []
    report.append("=" * 80)
    report.append("FEATURE EXTRACTION SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("Dataset Overview:")
    report.append(f"  Total segments: {stats['segment_count']}")
    report.append(f"  Total features: {stats['feature_count']}")
    report.append(f"  Feature sparsity: {stats['sparsity']:.1%}")
    report.append("")
    
    report.append("Feature Statistics:")
    report.append(f"  Mean value: {stats['feature_stats']['mean']:.4f}")
    report.append(f"  Std deviation: {stats['feature_stats']['std']:.4f}")
    report.append(f"  Range: [{stats['feature_stats']['min']:.2f}, {stats['feature_stats']['max']:.2f}]")
    report.append("")
    
    report.append("Task Distribution:")
    for task, count in sorted(stats['task_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['segment_count'] * 100
        report.append(f"  {task}: {count} segments ({pct:.1f}%)")
    report.append("")
    
    report.append("Top Features by Variance:")
    for i, feat in enumerate(importance['top_features'][:15], 1):
        report.append(f"  {i}. {feat['feature_name']:<30} "
                     f"var={feat['variance']:.4f} "
                     f"mean={feat['mean']:.3f}")
    report.append("")
    
    report.append("Feature Categories:")
    report.append("  Contextual features:")
    report.append("    - App encoding: 12 features (one-hot)")
    report.append("    - Time of day: 4 features (morning/afternoon/evening/night)")
    report.append("    - Duration: 5 features (category + log-scaled + ratios)")
    report.append("    - Temporal: 5 features (hour, weekday, weekend, work hours)")
    report.append("")
    report.append("  Behavioral features:")
    report.append("    - Previous task: 6 features (one-hot encoded)")
    report.append("    - Switching patterns: 5 features (count, return, frequency)")
    report.append("")
    report.append("  Semantic features:")
    report.append("    - Domain indicators: 6 features (coding, docs, comm, research, prod, content)")
    report.append("    - Text indicators: 3 features (has_code, has_docs, has_comm)")
    report.append("    - Title stats: 2 features (length, keyword count)")
    report.append("    - Bag of words: 50 features (top keywords)")
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save report
    if output_path is None:
        output_path = Path(__file__).parent / "feature_summary.txt"
    
    with open(output_path, "w") as f:
        f.write(report_text)
    
    print(f"Report saved to: {output_path}")
    
    return report_text


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "export":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
            export_feature_dataset(limit=limit)
        
        elif command == "analyze":
            importance = analyze_feature_importance()
            print("\nTop 15 Features by Variance:")
            for i, feat in enumerate(importance['top_features'], 1):
                print(f"{i:2d}. {feat['feature_name']:<35} "
                      f"var={feat['variance']:.4f} "
                      f"mean={feat['mean']:.3f}")
        
        elif command == "stats":
            stats = get_feature_statistics()
            print("\nFeature Statistics:")
            print(f"Segments: {stats['segment_count']}")
            print(f"Features: {stats['feature_count']}")
            print(f"Sparsity: {stats['sparsity']:.1%}")
            print(f"\nTask Distribution:")
            for task, count in stats['task_distribution'].items():
                print(f"  {task}: {count}")
        
        elif command == "report":
            report = create_feature_summary_report()
            print(report)
        
        elif command == "correlation":
            visualize_feature_correlation()
    
    else:
        # Default: create full report
        report = create_feature_summary_report()
        print(report)
