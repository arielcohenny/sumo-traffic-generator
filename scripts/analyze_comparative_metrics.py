#!/usr/bin/env python3
"""
Comparative Metrics Analysis Script

Analyzes collected traffic metrics to identify which metrics best discriminate
between good performance (Tree Method) and bad performance (Fixed timing).

This empirical analysis will guide reward function design for RL training.

Usage:
    python scripts/analyze_comparative_metrics.py [--data-dir PATH]

Output:
    - Statistical analysis of all metrics
    - Identification of discriminative metrics
    - Correlation analysis
    - Recommendations for reward function design
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_all_metrics(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all metric CSV files and combine into tree_method and fixed DataFrames.

    Args:
        data_dir: Directory containing metric CSV files

    Returns:
        (tree_method_df, fixed_df): DataFrames with all episodes combined
    """
    tree_method_dfs = []
    fixed_dfs = []

    print("Loading metric files...")
    for csv_file in sorted(data_dir.glob("episode_*_metrics.csv")):
        filename = csv_file.name
        episode_num = filename.split('_')[1]
        method = filename.split('_')[2]

        df = pd.read_csv(csv_file)
        df['episode'] = int(episode_num)

        if method == 'tree':
            tree_method_dfs.append(df)
        elif method == 'fixed':
            fixed_dfs.append(df)

    if not tree_method_dfs or not fixed_dfs:
        raise ValueError(f"No metric files found in {data_dir}")

    tree_method_all = pd.concat(tree_method_dfs, ignore_index=True)
    fixed_all = pd.concat(fixed_dfs, ignore_index=True)

    print(f"  Tree Method: {len(tree_method_dfs)} episodes, {len(tree_method_all)} data points")
    print(f"  Fixed: {len(fixed_dfs)} episodes, {len(fixed_all)} data points")
    print()

    return tree_method_all, fixed_all


def compute_metric_statistics(tree_df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive statistics for all metrics.

    Args:
        tree_df: Tree Method metrics
        fixed_df: Fixed timing metrics

    Returns:
        DataFrame with statistical comparison
    """
    # Metrics to analyze (exclude episode and timestamp)
    metric_columns = [col for col in tree_df.columns if col not in ['episode', 'timestamp']]

    results = []

    for metric in metric_columns:
        tree_values = tree_df[metric].values
        fixed_values = fixed_df[metric].values

        # Basic statistics
        tree_mean = np.mean(tree_values)
        tree_std = np.std(tree_values)
        fixed_mean = np.mean(fixed_values)
        fixed_std = np.std(fixed_values)

        # Difference and improvement
        diff = tree_mean - fixed_mean
        if fixed_mean != 0:
            improvement_pct = (diff / abs(fixed_mean)) * 100
        else:
            improvement_pct = 0

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((tree_std**2 + fixed_std**2) / 2)
        if pooled_std != 0:
            cohens_d = abs(diff) / pooled_std
        else:
            cohens_d = 0

        # Normalized separation (how well separated are the distributions?)
        if (tree_std + fixed_std) != 0:
            separation = abs(diff) / (tree_std + fixed_std)
        else:
            separation = 0

        results.append({
            'metric': metric,
            'tree_mean': tree_mean,
            'tree_std': tree_std,
            'fixed_mean': fixed_mean,
            'fixed_std': fixed_std,
            'difference': diff,
            'improvement_pct': improvement_pct,
            'cohens_d': cohens_d,
            'separation': separation
        })

    df = pd.DataFrame(results)

    # Sort by effect size (Cohen's d) to identify most discriminative metrics
    df = df.sort_values('cohens_d', ascending=False)

    return df


def identify_discriminative_metrics(stats_df: pd.DataFrame, threshold: float = 0.5) -> List[str]:
    """
    Identify metrics with strong discriminative power.

    Cohen's d interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect

    Args:
        stats_df: Statistics DataFrame
        threshold: Minimum Cohen's d for inclusion

    Returns:
        List of discriminative metric names
    """
    discriminative = stats_df[stats_df['cohens_d'] >= threshold]['metric'].tolist()
    return discriminative


def analyze_temporal_patterns(tree_df: pd.DataFrame, fixed_df: pd.DataFrame) -> None:
    """
    Analyze how metrics evolve over time during simulation.

    Args:
        tree_df: Tree Method metrics
        fixed_df: Fixed timing metrics
    """
    print("\n" + "="*80)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*80)
    print()

    # Group by timestamp and compute means
    tree_temporal = tree_df.groupby('timestamp').mean()
    fixed_temporal = fixed_df.groupby('timestamp').mean()

    # Analyze key metrics over time
    key_metrics = ['avg_speed_kmh', 'avg_occupancy', 'num_bottleneck_edges', 'total_waiting_time']

    print("Average values at start, middle, and end of simulation:")
    print()

    timestamps = sorted(tree_temporal.index.tolist())
    start_time = timestamps[0]
    mid_time = timestamps[len(timestamps)//2]
    end_time = timestamps[-1]

    for metric in key_metrics:
        if metric not in tree_temporal.columns:
            continue

        print(f"{metric}:")
        print(f"  Start ({start_time}s):")
        print(f"    Tree Method: {tree_temporal.loc[start_time, metric]:.2f}")
        print(f"    Fixed:       {fixed_temporal.loc[start_time, metric]:.2f}")
        print(f"  Middle ({mid_time}s):")
        print(f"    Tree Method: {tree_temporal.loc[mid_time, metric]:.2f}")
        print(f"    Fixed:       {fixed_temporal.loc[mid_time, metric]:.2f}")
        print(f"  End ({end_time}s):")
        print(f"    Tree Method: {tree_temporal.loc[end_time, metric]:.2f}")
        print(f"    Fixed:       {fixed_temporal.loc[end_time, metric]:.2f}")
        print()


def print_analysis_report(stats_df: pd.DataFrame, data_dir: Path) -> None:
    """
    Print comprehensive analysis report.

    Args:
        stats_df: Statistics DataFrame
        data_dir: Data directory path
    """
    print("="*80)
    print("COMPARATIVE METRICS ANALYSIS REPORT")
    print("="*80)
    print()

    # Load summary to show overall performance
    summary_file = data_dir / "summary.csv"
    if summary_file.exists():
        summary = pd.read_csv(summary_file)

        tree_summary = summary[summary['method'] == 'tree_method']
        fixed_summary = summary[summary['method'] == 'fixed']

        print("OVERALL PERFORMANCE SUMMARY")
        print("-" * 80)
        print()
        print("Tree Method:")
        print(f"  Avg Throughput: {tree_summary['throughput'].mean():.0f} vehicles")
        print(f"  Avg Duration:   {tree_summary['avg_duration'].mean():.1f}s")
        print()
        print("Fixed:")
        print(f"  Avg Throughput: {fixed_summary['throughput'].mean():.0f} vehicles")
        print(f"  Avg Duration:   {fixed_summary['avg_duration'].mean():.1f}s")
        print()

        throughput_improvement = (
            (tree_summary['throughput'].mean() - fixed_summary['throughput'].mean()) /
            fixed_summary['throughput'].mean() * 100
        )
        duration_improvement = (
            (fixed_summary['avg_duration'].mean() - tree_summary['avg_duration'].mean()) /
            fixed_summary['avg_duration'].mean() * 100
        )

        print(f"Tree Method Improvement:")
        print(f"  Throughput: +{throughput_improvement:.1f}%")
        print(f"  Duration:   -{duration_improvement:.1f}% (lower is better)")
        print()

    # Show top discriminative metrics
    print("="*80)
    print("TOP DISCRIMINATIVE METRICS (by Cohen's d)")
    print("="*80)
    print()

    print("Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large")
    print()

    # Show top 10 metrics
    top_metrics = stats_df.head(10)
    print(f"{'Rank':<6} {'Metric':<30} {'Cohens_d':<12} {'Improvement':<15} {'Direction':<10}")
    print("-" * 80)

    for idx, row in enumerate(top_metrics.itertuples(), 1):
        direction = "↑ Higher" if row.difference > 0 else "↓ Lower"
        print(f"{idx:<6} {row.metric:<30} {row.cohens_d:<12.3f} {row.improvement_pct:>+12.1f}% {direction:<10}")

    print()

    # Categorize metrics
    print("="*80)
    print("METRIC CATEGORIZATION")
    print("="*80)
    print()

    large_effect = stats_df[stats_df['cohens_d'] >= 0.8]['metric'].tolist()
    medium_effect = stats_df[(stats_df['cohens_d'] >= 0.5) & (stats_df['cohens_d'] < 0.8)]['metric'].tolist()
    small_effect = stats_df[(stats_df['cohens_d'] >= 0.2) & (stats_df['cohens_d'] < 0.5)]['metric'].tolist()

    print(f"Large Effect (d ≥ 0.8): {len(large_effect)} metrics")
    for metric in large_effect:
        print(f"  - {metric}")
    print()

    print(f"Medium Effect (0.5 ≤ d < 0.8): {len(medium_effect)} metrics")
    for metric in medium_effect:
        print(f"  - {metric}")
    print()

    print(f"Small Effect (0.2 ≤ d < 0.5): {len(small_effect)} metrics")
    for metric in small_effect:
        print(f"  - {metric}")
    print()

    # Recommendations
    print("="*80)
    print("REWARD FUNCTION DESIGN RECOMMENDATIONS")
    print("="*80)
    print()

    discriminative = large_effect + medium_effect

    if discriminative:
        print("Based on the analysis, consider including these metrics in the reward function:")
        print()

        for metric in discriminative[:5]:  # Top 5 recommendations
            row = stats_df[stats_df['metric'] == metric].iloc[0]
            direction = "maximize" if row['difference'] > 0 else "minimize"
            print(f"  {metric}:")
            print(f"    - Effect size: {row['cohens_d']:.3f} (strong discriminator)")
            print(f"    - Goal: {direction}")
            print(f"    - Tree Method achieves {row['improvement_pct']:+.1f}% vs Fixed")
            print()

        print("Reward function design strategy:")
        print("  1. Combine top discriminative metrics with appropriate weights")
        print("  2. Normalize metrics to similar scales")
        print("  3. Use positive rewards for desirable outcomes (not just penalties)")
        print("  4. Consider temporal weighting (early vs late simulation)")
        print()
    else:
        print("WARNING: No strong discriminative metrics found!")
        print("This suggests Tree Method and Fixed may perform similarly.")
        print("Review the summary.csv to verify performance differences.")
        print()


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze comparative metrics to identify discriminative features"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="evaluation/comparative_analysis",
        help="Directory containing metric CSV files"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run collect_comparative_data.sh first to generate data.")
        return 1

    # Load data
    tree_df, fixed_df = load_all_metrics(data_dir)

    # Compute statistics
    print("Computing metric statistics...")
    stats_df = compute_metric_statistics(tree_df, fixed_df)

    # Save statistics to CSV
    stats_file = data_dir / "metric_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics to: {stats_file}")
    print()

    # Analyze temporal patterns
    analyze_temporal_patterns(tree_df, fixed_df)

    # Print report
    print_analysis_report(stats_df, data_dir)

    # Identify discriminative metrics
    discriminative = identify_discriminative_metrics(stats_df, threshold=0.5)

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print(f"Identified {len(discriminative)} discriminative metrics (Cohen's d ≥ 0.5)")
    print(f"Full statistics saved to: {stats_file}")
    print()
    print("Next steps:")
    print("  1. Review metric_statistics.csv for detailed comparisons")
    print("  2. Design reward function based on top discriminative metrics")
    print("  3. Implement reward function in src/rl/environment.py")
    print("  4. Retrain RL model and evaluate performance")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
