#!/usr/bin/env python3
"""
Compare RL model checkpoints by running each through the same simulation scenario.

This script takes ZERO arguments. It reads config.yaml from the current
working directory (an experiment folder) and evaluates all checkpoints
in ./checkpoint/, writing results to ./compare_checkpoints_result.csv.

Already-evaluated checkpoints are skipped, so the script is resumable.

Usage:
    cd rl/experiments/my_exp/
    python ../../server/compare_checkpoints.py
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMPY_DISABLE_THREADING'] = '1'
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Set, Tuple, List

# Add project root to path so imports work from any directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rl.server.config_loader import load_single_config, build_env_params_string


CHECKPOINT_DIR_NAME = "checkpoint"
RESULTS_FILENAME = "compare_checkpoints_result.csv"


def get_all_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Get all checkpoint files sorted by step number (ascending).

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Returns:
        List of checkpoint Paths sorted by step count.
    """
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    checkpoints = list(checkpoint_dir.glob('rl_traffic_model_*_steps.zip'))

    def extract_steps(p: Path) -> int:
        match = re.search(r'_(\d+)_steps\.zip$', p.name)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=extract_steps)
    return checkpoints


def get_completed_checkpoints(results_file: Path) -> Set[str]:
    """Read existing results CSV and return set of already-evaluated checkpoint names.

    Args:
        results_file: Path to the results CSV file.

    Returns:
        Set of checkpoint filenames that have already been evaluated.
    """
    completed = set()
    if results_file.exists():
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('checkpoint'):
                    continue
                parts = line.split(',')
                if parts:
                    completed.add(parts[0].strip())
    return completed


def parse_statistics(output: str) -> Optional[Tuple[float, float]]:
    """Extract Throughput and Average duration from simulation output.

    Args:
        output: Combined stdout+stderr from the simulation process.

    Returns:
        Tuple of (throughput, avg_duration) or None if parsing fails.
    """
    throughput_match = re.search(
        r'Throughput:\s*([0-9.]+)\s*veh/h', output, re.IGNORECASE)
    duration_match = re.search(
        r'Average\s+duration:\s*([0-9.]+)s', output, re.IGNORECASE)

    if throughput_match and duration_match:
        return float(throughput_match.group(1)), float(duration_match.group(1))
    return None


def run_simulation(
    checkpoint_path: Path,
    env_params: str,
    cycle_lengths: List[int],
    project_root: str,
) -> Optional[Tuple[float, float]]:
    """Run simulation with given checkpoint and return metrics.

    Args:
        checkpoint_path: Path to the checkpoint .zip file.
        env_params: Environment parameters string from build_env_params_string.
        cycle_lengths: List of cycle lengths from config.
        project_root: Absolute path to the project root directory.

    Returns:
        Tuple of (throughput, avg_duration) or None on failure.
    """
    cycle_str = ' '.join(str(c) for c in cycle_lengths)

    cmd = (
        ['env', 'PYTHONUNBUFFERED=1', 'python', '-m', 'src.cli']
        + shlex.split(env_params)
        + ['--rl_model_path', str(checkpoint_path),
           '--rl-cycle-lengths', cycle_str]
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
            cwd=project_root,
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
    experiment_dir = os.getcwd()

    # ── Validate config.yaml exists ───────────────────────────────────
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found in {experiment_dir}", file=sys.stderr)
        print("This script must be run from an experiment folder containing config.yaml.",
              file=sys.stderr)
        sys.exit(1)

    # ── Load config and build env params ──────────────────────────────
    config, scenario_dict, traffic_control = load_single_config(config_path)
    env_params = build_env_params_string(config, scenario_dict, traffic_control)

    cycle_lengths = config.cycle_lengths
    cycle_str = ' '.join(str(c) for c in cycle_lengths)

    print(f"Experiment dir: {experiment_dir}")
    print(f"Traffic control: {traffic_control}")
    print(f"Cycle lengths: {cycle_lengths}")
    print(f"Env params: {env_params}")

    # ── Discover checkpoints ──────────────────────────────────────────
    checkpoint_dir = Path(experiment_dir) / CHECKPOINT_DIR_NAME
    checkpoints = get_all_checkpoints(checkpoint_dir)

    # ── Load already-completed results ────────────────────────────────
    results_file = Path(experiment_dir) / RESULTS_FILENAME
    completed = get_completed_checkpoints(results_file)

    remaining = [cp for cp in checkpoints if cp.name not in completed]

    print(f"\nFound {len(checkpoints)} total checkpoints")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining to evaluate: {len(remaining)}")
    print(f"Results file: {results_file}")
    print(f"{'=' * 70}")

    # ── Write CSV header if new file ──────────────────────────────────
    if not results_file.exists():
        with open(results_file, 'w') as f:
            f.write("checkpoint, Throughput, Average duration\n")

    if not remaining:
        print("\nAll checkpoints already evaluated!")
        return

    # ── Evaluate each checkpoint ──────────────────────────────────────
    for i, checkpoint in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Testing: {checkpoint.name}")
        print(f"  Running simulation... ", end='', flush=True)

        result = run_simulation(checkpoint, env_params, cycle_lengths, _project_root)

        if result:
            throughput, avg_duration = result
            print("SUCCESS")
            print(f"  Throughput: {throughput:.2f} veh/h, Avg Duration: {avg_duration:.2f}s")

            with open(results_file, 'a') as f:
                f.write(f"{checkpoint.name}, {throughput:.2f}, {avg_duration:.2f}\n")
        else:
            print("FAILED (could not parse statistics)")
            with open(results_file, 'a') as f:
                f.write(f"{checkpoint.name}, FAILED, FAILED\n")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Completed! Results saved to: {results_file}")


if __name__ == '__main__':
    main()
