#!/usr/bin/env python3
"""
Reward Validation Analysis Script.

Analyzes reward logs from 3 training episodes to validate that the reward function
correctly correlates with good traffic states.

Computes correlations between:
- Total reward ↔ Network metrics (speed, bottlenecks, waiting)
- Individual reward components ↔ Their target metrics

Usage:
    python scripts/analyze_reward_validation.py
"""

import pandas as pd
import numpy as np
import os
import glob


def load_episode_logs():
    """Load all reward analysis CSV files.

    Returns:
        Dictionary mapping episode number to DataFrame
    """
    episodes = {}
    log_files = sorted(glob.glob("reward_analysis_episode_*.csv"))

    if not log_files:
        print("❌ No reward analysis log files found!")
        print("   Expected files: reward_analysis_episode_1.csv, reward_analysis_episode_2.csv, etc.")
        print("   Run: python scripts/run_reward_validation_training.py first")
        return episodes

    print(f"\n{'='*80}")
    print("LOADING EPISODE LOGS")
    print(f"{'='*80}\n")

    for log_file in log_files:
        # Extract episode number from filename
        episode_num = int(log_file.split('_')[-1].replace('.csv', ''))
        df = pd.read_csv(log_file)

        print(f"  Episode {episode_num}: {len(df)} rows ({log_file})")
        episodes[episode_num] = df

    print(f"\nTotal episodes loaded: {len(episodes)}")

    return episodes


def compute_correlations(df, episode_num):
    """Compute correlations between reward and metrics for one episode.

    Args:
        df: DataFrame with reward log data
        episode_num: Episode number for reporting

    Returns:
        Dictionary of correlation results
    """
    correlations = {}

    # Total reward correlations (these tell us if reward aligns with good states)
    correlations['reward_vs_speed'] = df['total_reward'].corr(df['avg_speed'])
    correlations['reward_vs_bottlenecks'] = df['total_reward'].corr(df['bottleneck_count'])
    correlations['reward_vs_excessive_waiting'] = df['total_reward'].corr(df['vehicles_with_excessive_waiting'])

    # Component-specific correlations (validate individual components)
    correlations['throughput_component_vs_completions'] = df['throughput_reward'].corr(df['vehicles_completed_this_step'])
    correlations['speed_component_vs_speed'] = df['speed_reward'].corr(df['avg_speed'])
    correlations['bottleneck_component_vs_bottlenecks'] = df['bottleneck_penalty'].corr(df['bottleneck_count'])

    return correlations


def analyze_episode(df, episode_num):
    """Analyze a single episode's reward behavior.

    Args:
        df: DataFrame with reward log data
        episode_num: Episode number
    """
    print(f"\n{'='*80}")
    print(f"EPISODE {episode_num} ANALYSIS")
    print(f"{'='*80}\n")

    # Summary statistics
    print(f"Summary Statistics:")
    print(f"  Total steps logged:       {len(df)}")
    print(f"  Avg total reward:         {df['total_reward'].mean():+.2f}")
    print(f"  Avg speed:                {df['avg_speed'].mean():.2f} km/h")
    print(f"  Avg bottlenecks:          {df['bottleneck_count'].mean():.1f}")
    print(f"  Avg active vehicles:      {df['active_vehicles'].mean():.0f}")
    print(f"  Total vehicles completed: {df['vehicles_completed_this_step'].sum():.0f}")

    # Compute correlations
    corr = compute_correlations(df, episode_num)

    print(f"\n{'-'*80}")
    print(f"REWARD FUNCTION VALIDATION (Correlation Analysis)")
    print(f"{'-'*80}\n")

    print("Total Reward Correlations:")
    print(f"  Reward ↔ Speed:                {corr['reward_vs_speed']:+.3f}  "
          f"{'✓ GOOD' if corr['reward_vs_speed'] > 0.3 else '⚠ WEAK' if corr['reward_vs_speed'] > 0 else '✗ BAD'}")
    print(f"  Reward ↔ Bottlenecks:          {corr['reward_vs_bottlenecks']:+.3f}  "
          f"{'✓ GOOD' if corr['reward_vs_bottlenecks'] < -0.3 else '⚠ WEAK' if corr['reward_vs_bottlenecks'] < 0 else '✗ BAD'}")
    print(f"  Reward ↔ Excessive Waiting:    {corr['reward_vs_excessive_waiting']:+.3f}  "
          f"{'✓ GOOD' if corr['reward_vs_excessive_waiting'] < -0.3 else '⚠ WEAK' if corr['reward_vs_excessive_waiting'] < 0 else '✗ BAD'}")

    print("\nComponent-Specific Correlations:")
    print(f"  Throughput Reward ↔ Completions:     {corr['throughput_component_vs_completions']:+.3f}  "
          f"{'✓ GOOD' if corr['throughput_component_vs_completions'] > 0.8 else '⚠ WEAK'}")
    print(f"  Speed Reward ↔ Speed:                {corr['speed_component_vs_speed']:+.3f}  "
          f"{'✓ GOOD' if corr['speed_component_vs_speed'] > 0.8 else '⚠ WEAK'}")
    print(f"  Bottleneck Penalty ↔ Bottlenecks:    {corr['bottleneck_component_vs_bottlenecks']:+.3f}  "
          f"{'✓ GOOD' if corr['bottleneck_component_vs_bottlenecks'] < -0.8 else '⚠ WEAK'}")

    return corr


def aggregate_analysis(all_correlations):
    """Aggregate correlation analysis across all episodes.

    Args:
        all_correlations: Dictionary mapping episode number to correlation dict
    """
    print(f"\n{'='*80}")
    print("AGGREGATE ANALYSIS (All Episodes)")
    print(f"{'='*80}\n")

    # Average correlations across episodes
    avg_corr = {}
    for key in all_correlations[list(all_correlations.keys())[0]].keys():
        values = [corr[key] for corr in all_correlations.values()]
        avg_corr[key] = np.mean(values)
        std_corr = np.std(values)

        print(f"{key}:")
        print(f"  Mean:  {avg_corr[key]:+.3f}")
        print(f"  Std:   {std_corr:.3f}")
        print()

    # Overall validation
    print(f"{'='*80}")
    print("REWARD FUNCTION VALIDITY ASSESSMENT")
    print(f"{'='*80}\n")

    checks = []

    # Check 1: Positive correlation with speed
    if avg_corr['reward_vs_speed'] > 0.3:
        print("✓ PASS: Reward positively correlated with speed")
        checks.append(True)
    else:
        print("✗ FAIL: Reward NOT positively correlated with speed")
        print(f"  → Increase REWARD_SPEED_REWARD_FACTOR (currently 2.0)")
        checks.append(False)

    # Check 2: Negative correlation with bottlenecks
    if avg_corr['reward_vs_bottlenecks'] < -0.3:
        print("✓ PASS: Reward negatively correlated with bottlenecks")
        checks.append(True)
    else:
        print("✗ FAIL: Reward NOT negatively correlated with bottlenecks")
        print(f"  → Increase REWARD_BOTTLENECK_PENALTY_PER_EDGE (currently 0.5)")
        checks.append(False)

    # Check 3: Negative correlation with excessive waiting
    if avg_corr['reward_vs_excessive_waiting'] < -0.3:
        print("✓ PASS: Reward negatively correlated with excessive waiting")
        checks.append(True)
    else:
        print("✗ FAIL: Reward NOT negatively correlated with excessive waiting")
        print(f"  → Increase REWARD_EXCESSIVE_WAITING_PENALTY (currently 0.5)")
        checks.append(False)

    # Check 4: Components behave correctly
    component_checks = [
        avg_corr['throughput_component_vs_completions'] > 0.8,
        avg_corr['speed_component_vs_speed'] > 0.8,
        avg_corr['bottleneck_component_vs_bottlenecks'] < -0.8
    ]

    if all(component_checks):
        print("✓ PASS: All reward components correctly correlated with their targets")
        checks.append(True)
    else:
        print("⚠ PARTIAL: Some reward components have weak correlations")
        checks.append(False)

    # Final verdict
    print(f"\n{'='*80}")
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✓✓✓ REWARD FUNCTION VALIDATED ({passed}/{total} checks passed)")
        print("    The reward function correctly identifies good traffic states!")
        print("    Proceed with full training.")
    elif passed >= total * 0.75:
        print(f"⚠⚠ REWARD FUNCTION PARTIALLY VALIDATED ({passed}/{total} checks passed)")
        print("    Consider tuning weights as suggested above.")
        print("    Can proceed with training but monitor performance.")
    else:
        print(f"✗✗✗ REWARD FUNCTION NEEDS TUNING ({passed}/{total} checks passed)")
        print("    Adjust weights in src/rl/constants.py as suggested above.")
        print("    Re-run validation before full training.")

    print(f"{'='*80}\n")


def main():
    """Main analysis workflow."""
    print("\n" + "="*80)
    print("REWARD VALIDATION ANALYSIS")
    print("="*80)

    # Load episode logs
    episodes = load_episode_logs()

    if not episodes:
        return

    # Analyze each episode
    all_correlations = {}
    for episode_num in sorted(episodes.keys()):
        df = episodes[episode_num]
        corr = analyze_episode(df, episode_num)
        all_correlations[episode_num] = corr

    # Aggregate analysis
    if len(all_correlations) > 1:
        aggregate_analysis(all_correlations)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
