#!/usr/bin/env python3
"""
Compare RL model checkpoints by running each through the same simulation scenario.
Outputs results to compare_checkpoints_result with Throughput and Average duration.
"""

import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


CHECKPOINT_DIR = Path(
    '.worktrees/rl-training/models/rl_20260105_090955/checkpoint')
RESULTS_FILE = Path('compare_checkpoints_result')

# Simulation parameters (matching the user's command)
BASE_CMD = [
    'env', 'PYTHONUNBUFFERED=1',
    'python', '-m', 'src.cli',
    '--network-seed', '24208',
    '--grid_dimension', '6',
    '--junctions_to_remove', '2',
    '--block_size_m', '280',
    '--lane_count', 'realistic',
    '--step-length', '1.0',
    '--land_use_block_size_m', '25.0',
    '--attractiveness', 'land_use',
    '--traffic_light_strategy', 'partial_opposites',
    '--routing_strategy', 'realtime 100',
    '--vehicle_types', 'passenger 100',
    '--passenger-routes', 'in 0 out 0 inner 100 pass 0',
    '--departure_pattern', 'uniform',
    '--private-traffic-seed', '72632',
    '--public-traffic-seed', '27031',
    '--start_time_hour', '8.0',
    '--num_vehicles', '22000',
    '--end-time', '7300',
    '--traffic_control', 'rl',
    '--rl-cycle-lengths', '90'
]


def get_checkpoints() -> list[Path]:
    """Get all checkpoint files sorted by step number."""
    if not CHECKPOINT_DIR.exists():
        print(f"Error: Checkpoint directory not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    checkpoints = list(CHECKPOINT_DIR.glob('rl_traffic_model_*_steps.zip'))

    # Sort by step number
    def extract_steps(p: Path) -> int:
        match = re.search(r'_(\d+)_steps\.zip$', p.name)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=extract_steps)
    return checkpoints


def get_completed_checkpoints() -> set[str]:
    """Read existing results file and return set of already-processed checkpoint names."""
    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header and empty lines
                if not line or line.startswith('checkpoint'):
                    continue
                # Extract checkpoint name (first column)
                parts = line.split(',')
                if parts:
                    completed.add(parts[0].strip())
    return completed


def parse_statistics(output: str) -> Optional[Tuple[float, float]]:
    """Extract Throughput and Average duration from simulation output."""
    throughput_match = re.search(
        r'Throughput:\s*([0-9.]+)\s*veh/h', output, re.IGNORECASE)
    duration_match = re.search(
        r'Average\s+duration:\s*([0-9.]+)s', output, re.IGNORECASE)

    if throughput_match and duration_match:
        return float(throughput_match.group(1)), float(duration_match.group(1))
    return None


def run_simulation(checkpoint_path: Path) -> Optional[Tuple[float, float]]:
    """Run simulation with given checkpoint and return (throughput, avg_duration)."""
    cmd = BASE_CMD + ['--rl_model_path', str(checkpoint_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15 minute timeout
        )

        output = result.stdout + result.stderr
        return parse_statistics(output)

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    checkpoints = get_checkpoints()
    completed = get_completed_checkpoints()

    # Filter out already-completed checkpoints
    remaining = [cp for cp in checkpoints if cp.name not in completed]

    print(f"Found {len(checkpoints)} total checkpoints")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining to evaluate: {len(remaining)}")
    print(f"Results will be written to: {RESULTS_FILE}")
    print(f"{'='*70}")

    # Write header only if file doesn't exist
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'w') as f:
            f.write("checkpoint, Throughput, Average duration\n")

    if not remaining:
        print("\nAll checkpoints already evaluated!")
        return

    for i, checkpoint in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Testing: {checkpoint.name}")
        print(f"  Running simulation... ", end='', flush=True)

        result = run_simulation(checkpoint)

        if result:
            throughput, avg_duration = result
            print(f"SUCCESS")
            print(
                f"  Throughput: {throughput:.2f} veh/h, Avg Duration: {avg_duration:.2f}s")

            # Append result
            with open(RESULTS_FILE, 'a') as f:
                f.write(
                    f"{checkpoint.name}, {throughput:.2f}, {avg_duration:.2f}\n")
        else:
            print(f"FAILED (could not parse statistics)")
            # Write failure entry
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{checkpoint.name}, FAILED, FAILED\n")

    print(f"\n{'='*70}")
    print(f"Completed! Results saved to: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
