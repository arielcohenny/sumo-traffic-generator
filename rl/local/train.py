#!/usr/bin/env python3
"""
Local RL Training Script for development and debugging.

Designed for laptop use: single env by default, quick iteration,
verbose logging. Uses modular config files (5 separate YAMLs).

Usage:
    python rl/local/train.py \
        --network rl/configs/network/grid6_realistic.yaml \
        --scenarios rl/configs/scenarios/heavy_load.yaml \
        --algorithm rl/configs/algorithm/ppo_default.yaml \
        --reward rl/configs/reward/empirical.yaml \
        --execution rl/configs/execution/quick_test.yaml
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMPY_DISABLE_THREADING'] = '1'
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import argparse
import logging
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rl.training import train_rl_policy
from src.rl.experiment_config import (
    load_modular_config, build_env_params_list, save_experiment
)


def setup_logging(log_level: str = "INFO"):
    """Setup verbose logging for local development."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RL training for development and debugging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Modular config files (5 concerns)
    parser.add_argument('--network', type=str, required=True,
                        help="Path to network config YAML")
    parser.add_argument('--scenarios', type=str, required=True,
                        help="Path to scenarios config YAML")
    parser.add_argument('--algorithm', type=str, required=True,
                        help="Path to algorithm config YAML")
    parser.add_argument('--reward', type=str, required=True,
                        help="Path to reward config YAML")
    parser.add_argument('--execution', type=str, required=True,
                        help="Path to execution config YAML")

    # Overrides
    parser.add_argument('--timesteps', type=int, default=None,
                        help="Override total_timesteps from execution config")
    parser.add_argument('--all-scenarios', action='store_true',
                        help="Use all scenarios (default: first scenario only)")

    # Model management
    parser.add_argument('--resume-from', type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument('--models-dir', type=str, default='rl/models',
                        help="Directory to save models")

    # Cycle parameters
    parser.add_argument('--cycle-lengths', type=int, nargs='+', default=None,
                        help="Override cycle lengths from config")
    parser.add_argument('--cycle-strategy', type=str, default=None,
                        choices=['fixed', 'random', 'sequential'],
                        help="Override cycle strategy from config")

    # Debug
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'],
                        default='INFO', help="Logging level")

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load modular config
    logger.info("Loading modular config files...")
    config = load_modular_config(
        network_path=args.network,
        scenarios_path=args.scenarios,
        algorithm_path=args.algorithm,
        reward_path=args.reward,
        execution_path=args.execution,
    )

    # Apply CLI overrides
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps

    # Build env-params list
    env_params_list = build_env_params_list(config)

    if not args.all_scenarios and len(env_params_list) > 1:
        logger.info(f"Using first scenario only (of {len(env_params_list)}). "
                     f"Use --all-scenarios to use all.")
        env_params_list = env_params_list[:1]

    logger.info(f"=== LOCAL RL TRAINING ===")
    logger.info(f"Scenarios: {len(env_params_list)}")
    logger.info(f"Timesteps: {config.total_timesteps}")
    logger.info(f"Reward: {config.reward_function}")
    logger.info(f"Grid: {config.grid_dimension}x{config.grid_dimension}")

    # Resolve cycle params
    cycle_lengths = args.cycle_lengths or config.cycle_lengths
    cycle_strategy = args.cycle_strategy or config.cycle_strategy

    try:
        start_time = time.time()
        model = train_rl_policy(
            env_params_string=env_params_list[0],
            total_timesteps=config.total_timesteps,
            checkpoint_freq=config.checkpoint_freq,
            use_parallel=len(env_params_list) > 1,
            n_envs=len(env_params_list) if len(env_params_list) > 1 else 1,
            resume_from_model=args.resume_from,
            cycle_lengths=cycle_lengths,
            cycle_strategy=cycle_strategy,
            experiment_config=config,
            env_params_list=env_params_list if len(env_params_list) > 1 else None,
            models_dir=args.models_dir,
        )

        elapsed = time.time() - start_time
        logger.info(f"=== LOCAL TRAINING COMPLETE ({elapsed:.1f}s) ===")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
