#!/usr/bin/env python3
"""
Test all combinations of Tree Method M and L parameters.

This script runs experiments with different M and L parameter values
to analyze their impact on traffic flow performance.

Usage:
    python scripts/test_m_l_combinations.py
"""

from src.utils.statistics import parse_sumo_statistics_file
import subprocess
import csv
import os
from datetime import datetime
from pathlib import Path
import sys

# Import statistics parser
sys.path.insert(0, str(Path(__file__).parent.parent))

# Parameter ranges
L_MIN = 2.0
L_MAX = 3.2
L_STEP = 0.2

M_MIN = 0.0
M_MAX = 1.8
M_STEP = 0.2

# Sample network path
SAMPLE_PATH = "evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1"

# Simulation parameters
END_TIME = 7300
TIMEOUT_SECONDS = 6000  # 10 minutes max per experiment


def generate_parameter_combinations():
    """Generate all combinations of M and L parameters."""
    combinations = []

    # Generate L values
    l_value = L_MIN
    while l_value <= L_MAX + 0.001:  # Small epsilon for float comparison
        # Generate M values
        m_value = M_MIN
        while m_value <= M_MAX + 0.001:
            combinations.append((round(m_value, 1), round(l_value, 1)))
            m_value += M_STEP
        l_value += L_STEP

    return combinations


def run_experiment(exp_num, m_value, l_value):
    """
    Run a single experiment with given M and L values.

    Returns:
        tuple: (throughput, avg_duration) or (None, None) if failed
    """
    print(f"\n{'='*80}")
    print(f"Experiment {exp_num}: M={m_value}, L={l_value}")
    print(f"{'='*80}")

    # Build command
    cmd = [
        "python", "-m", "src.cli",
        "--tree_method_sample", SAMPLE_PATH,
        "--traffic_control", "tree_method",
        "--end-time", str(END_TIME),
        "--tree-method-m", str(m_value),
        "--tree-method-l", str(l_value)
    ]

    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )

        # Parse statistics from sumo_statistics.xml file
        stats_file = Path("workspace") / "sumo_statistics.xml"

        if not stats_file.exists():
            print(f"❌ Statistics file not found: {stats_file}")
            return None, None

        stats = parse_sumo_statistics_file(str(stats_file))

        if stats is None:
            print(f"❌ Failed to parse statistics file")
            return None, None

        # Extract the values we need
        throughput = stats['arrived']  # Total vehicles that completed
        avg_duration = stats['avg_duration']  # Average duration in seconds

        print(
            f"✅ Success: Throughput={throughput}, Avg Duration={avg_duration:.2f}s")
        return throughput, avg_duration

    except subprocess.TimeoutExpired:
        print(f"❌ Experiment timed out after {TIMEOUT_SECONDS} seconds")
        return None, None

    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return None, None


def main():
    """Main execution function."""
    print("="*80)
    print("Tree Method M/L Parameter Sweep")
    print("="*80)
    print(f"L range: {L_MIN} to {L_MAX} (step {L_STEP})")
    print(f"M range: {M_MIN} to {M_MAX} (step {M_STEP})")
    print(f"Sample: {SAMPLE_PATH}")
    print(f"End time: {END_TIME} seconds")
    print("="*80)

    # Generate parameter combinations
    combinations = generate_parameter_combinations()
    total_experiments = len(combinations)

    print(f"\nTotal experiments to run: {total_experiments}")
    print(
        f"Estimated time: {total_experiments * 5 / 60:.1f} - {total_experiments * 10 / 60:.1f} minutes")

    # Create results directory
    results_dir = Path("evaluation/exp_tree_method_m_l_combinations")
    results_dir.mkdir(exist_ok=True)

    # Create output CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"m_l_parameter_sweep_{timestamp}.csv"

    print(f"\nResults will be saved to: {output_file}")

    # Confirm before starting
    response = input("\nPress Enter to start, or Ctrl+C to cancel...")

    # Run experiments and log results
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(
            ['Exp_Num', 'M', 'L', 'Throughput', 'Avg_Duration_Seconds'])

        # Run each experiment
        for exp_num, (m_value, l_value) in enumerate(combinations, start=1):
            throughput, avg_duration = run_experiment(
                exp_num, m_value, l_value)

            # Write result
            writer.writerow([
                exp_num,
                m_value,
                l_value,
                throughput if throughput is not None else 'FAILED',
                f'{avg_duration:.2f}' if avg_duration is not None else 'FAILED'
            ])

            # Flush to disk after each experiment
            csvfile.flush()

            # Progress update
            print(
                f"\nProgress: {exp_num}/{total_experiments} ({exp_num/total_experiments*100:.1f}%)")

    print("\n" + "="*80)
    print("Parameter sweep complete!")
    print(f"Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
