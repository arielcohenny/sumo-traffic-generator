#!/usr/bin/env python3
"""
Collect Tree Method Demonstrations using Multi-Process Isolation.

This script orchestrates multiple episode collection processes, running each
episode in a completely separate Python process to guarantee zero state leakage.

Usage:
    python scripts/collect_demonstrations_multiprocess.py \
        --scenarios 20 \
        --base-seed 42 \
        --config configs/tree_method_demonstrations_1.json \
        --output models/demonstrations/demo.npz
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_episodes_multiprocess(
    num_scenarios: int,
    base_seed: int,
    config_file: str,
    cycle_length: int,
    output_file: str
) -> bool:
    """
    Collect demonstrations by running each episode in a separate process.

    Args:
        num_scenarios: Number of episodes to collect
        base_seed: Base random seed
        config_file: Path to configuration JSON
        cycle_length: Tree Method cycle length
        output_file: Final output file path

    Returns:
        True if all episodes succeeded
    """
    logger.info("="*70)
    logger.info("MULTI-PROCESS DEMONSTRATION COLLECTION")
    logger.info("="*70)
    logger.info(f"Scenarios: {num_scenarios}")
    logger.info(f"Base seed: {base_seed}")
    logger.info(f"Config: {config_file}")
    logger.info(f"Cycle length: {cycle_length}s")
    logger.info(f"Output: {output_file}")
    logger.info("="*70)
    logger.info("")

    # Load config
    with open(config_file) as f:
        config = json.load(f)

    # Create temporary directory for episode files
    temp_dir = Path("demonstration_temp_episodes")
    temp_dir.mkdir(exist_ok=True)

    # Collect each episode in a separate process
    episode_files = []
    failed_episodes = []

    for scenario_idx in range(num_scenarios):
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"EPISODE {scenario_idx + 1}/{num_scenarios}")
        logger.info(f"{'='*70}")

        episode_file = temp_dir / f"episode_{scenario_idx}.npz"
        episode_files.append(episode_file)

        # Run episode in separate process
        cmd = [
            sys.executable,
            "scripts/collect_single_episode.py",
            "--scenario-idx", str(scenario_idx),
            "--base-seed", str(base_seed),
            "--config", config_file,
            "--cycle-length", str(cycle_length),
            "--output", str(episode_file)
        ]

        logger.info(f"Starting subprocess...")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per episode
            )

            if result.returncode == 0:
                logger.info(f"✓ Episode {scenario_idx} completed successfully")

                # Show last 10 lines of output
                output_lines = result.stdout.strip().split('\n')
                logger.info("Last 10 lines of output:")
                for line in output_lines[-10:]:
                    logger.info(f"  {line}")
            else:
                logger.error(f"✗ Episode {scenario_idx} FAILED (exit code {result.returncode})")
                logger.error("STDOUT:")
                for line in result.stdout.strip().split('\n')[-20:]:
                    logger.error(f"  {line}")
                logger.error("STDERR:")
                for line in result.stderr.strip().split('\n')[-20:]:
                    logger.error(f"  {line}")
                failed_episodes.append(scenario_idx)

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Episode {scenario_idx} TIMEOUT (exceeded 2 hours)")
            failed_episodes.append(scenario_idx)
        except Exception as e:
            logger.error(f"✗ Episode {scenario_idx} ERROR: {e}")
            failed_episodes.append(scenario_idx)

    # Report results
    logger.info("")
    logger.info("="*70)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total episodes: {num_scenarios}")
    logger.info(f"Successful: {num_scenarios - len(failed_episodes)}")
    logger.info(f"Failed: {len(failed_episodes)}")

    if failed_episodes:
        logger.warning(f"Failed episodes: {failed_episodes}")

    # Combine episode files
    if len(failed_episodes) < num_scenarios:
        logger.info("")
        logger.info("Combining episode files...")

        all_states = []
        all_actions = []
        episode_lengths = []
        per_scenario_params = []

        for scenario_idx, episode_file in enumerate(episode_files):
            if scenario_idx in failed_episodes:
                continue

            if not episode_file.exists():
                logger.warning(f"Episode {scenario_idx} file not found: {episode_file}")
                continue

            data = np.load(episode_file, allow_pickle=True)
            metadata = data['metadata'].item()

            states = data['states']
            actions = data['actions']

            all_states.append(states)
            all_actions.append(actions)
            episode_lengths.append(len(states))
            per_scenario_params.append({
                'scenario_idx': scenario_idx,
                'seed': base_seed + scenario_idx,
                'sampled_params': metadata['sampled_params']
            })

            logger.info(f"  Episode {scenario_idx}: {len(states)} pairs")

        # Concatenate all data
        combined_states = np.vstack(all_states)
        combined_actions = np.vstack(all_actions)

        logger.info(f"Total demonstration pairs: {len(combined_states)}")

        # Save combined file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            states=combined_states,
            actions=combined_actions,
            episode_lengths=np.array(episode_lengths),
            metadata={
                'num_scenarios': num_scenarios - len(failed_episodes),
                'base_seed': base_seed,
                'fixed_params': config['fixed_params'],
                'varying_params_options': config['varying_params'],
                'per_scenario_params': per_scenario_params,
                'collection_date': datetime.now().isoformat(),
                'collection_method': 'multiprocess_isolated'
            }
        )

        logger.info(f"✓ Saved combined demonstrations to: {output_file}")

        # Cleanup episode files
        logger.info("Cleaning up temporary episode files...")
        for episode_file in episode_files:
            if episode_file.exists():
                episode_file.unlink()

        logger.info("✓ Cleanup complete")

        return len(failed_episodes) == 0
    else:
        logger.error("All episodes failed!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Collect Tree Method demonstrations using multi-process isolation'
    )
    parser.add_argument('--scenarios', type=int, default=20,
                        help='Number of scenarios to collect')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--config', type=str,
                        default='configs/tree_method_demonstrations_1.json',
                        help='Configuration JSON file')
    parser.add_argument('--cycle-lengths', type=int, nargs='+', default=[90],
                        help='Tree Method cycle lengths')
    parser.add_argument('--output', type=str,
                        help='Output NPZ file (default: auto-generated)')

    args = parser.parse_args()

    # Use first cycle length (for now, multi-cycle not supported in multiprocess)
    cycle_length = args.cycle_lengths[0]

    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("models/demonstrations")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"rl_demonstration_{timestamp}/demo_{timestamp}.npz"
        output_file.parent.mkdir(exist_ok=True)

    # Run collection
    success = collect_episodes_multiprocess(
        num_scenarios=args.scenarios,
        base_seed=args.base_seed,
        config_file=args.config,
        cycle_length=cycle_length,
        output_file=str(output_file)
    )

    if success:
        logger.info("")
        logger.info("="*70)
        logger.info("✓✓✓ COLLECTION COMPLETE - All episodes successful!")
        logger.info("="*70)
        sys.exit(0)
    else:
        logger.error("")
        logger.error("="*70)
        logger.error("✗✗✗ COLLECTION INCOMPLETE - Some episodes failed")
        logger.error("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
