#!/usr/bin/env python3
"""
Server (HPC) RL Training Script for production parallel training.

Designed for HPC clusters: multi-scenario parallel environments,
batch orchestration, resume support, production logging.

Usage:
    python rl/server/train.py \
        --network rl/configs/network/grid6_realistic.yaml \
        --scenarios rl/configs/scenarios/heavy_load.yaml \
        --algorithm rl/configs/algorithm/ppo_default.yaml \
        --reward rl/configs/reward/empirical.yaml \
        --execution rl/configs/execution/long_run.yaml

    # Resume with different scenarios (batch chaining):
    python rl/server/train.py \
        --network rl/configs/network/grid6_realistic.yaml \
        --scenarios rl/configs/scenarios/heavy_load_batch2.yaml \
        --algorithm rl/configs/algorithm/ppo_default.yaml \
        --reward rl/configs/reward/empirical.yaml \
        --execution rl/configs/execution/long_run.yaml \
        --resume-from rl/models/heavy_load_20260201/final_model.zip
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


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup production logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Suppress verbose RL logging for cleaner production output
    logging.getLogger('src.rl.controller').setLevel(logging.WARNING)
    logging.getLogger('src.rl.environment').setLevel(logging.WARNING)
    logging.getLogger('TrafficControlEnv').setLevel(logging.WARNING)
    logging.getLogger('src.rl.training').setLevel(logging.INFO)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Server RL training with multi-scenario parallel environments",
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
    parser.add_argument('--checkpoint-freq', type=int, default=None,
                        help="Override checkpoint_freq from execution config")

    # Model management
    parser.add_argument('--resume-from', type=str, default=None,
                        help="Path to model checkpoint to resume from (batch chaining)")
    parser.add_argument('--pretrain-from', type=str, default=None,
                        help="Path to pre-trained model to initialize from")
    parser.add_argument('--model-name', type=str, default=None,
                        help="Custom name prefix for model directory")
    parser.add_argument('--models-dir', type=str, default='rl/models',
                        help="Directory to save models")

    # Cycle parameters
    parser.add_argument('--cycle-lengths', type=int, nargs='+', default=None,
                        help="Override cycle lengths from config")
    parser.add_argument('--cycle-strategy', type=str, default=None,
                        choices=['fixed', 'random', 'sequential'],
                        help="Override cycle strategy from config")

    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help="Logging level")
    parser.add_argument('--quiet', action='store_true',
                        help="Minimal console output")

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.quiet:
        args.log_level = "WARNING"

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
    if args.checkpoint_freq is not None:
        config.checkpoint_freq = args.checkpoint_freq

    # Build env-params list (one per scenario)
    env_params_list = build_env_params_list(config)
    n_scenarios = len(env_params_list)

    logger.info(f"=== SERVER RL TRAINING ===")
    logger.info(f"Scenarios: {n_scenarios}")
    logger.info(f"Timesteps: {config.total_timesteps}")
    logger.info(f"Checkpoint freq: {config.checkpoint_freq}")
    logger.info(f"Reward: {config.reward_function}")
    logger.info(f"Grid: {config.grid_dimension}x{config.grid_dimension}")
    logger.info(f"Network seed: {config.network_seed}")

    for i, params in enumerate(env_params_list):
        scenario_name = config.scenarios[i].get("name", f"scenario_{i}") if i < len(config.scenarios) else f"scenario_{i}"
        logger.info(f"  Scenario {i} ({scenario_name}): {params[:100]}...")

    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")

    # Resolve cycle params
    cycle_lengths = args.cycle_lengths or config.cycle_lengths
    cycle_strategy = args.cycle_strategy or config.cycle_strategy

    try:
        start_time = time.time()

        # Multi-scenario parallel training
        use_parallel = n_scenarios > 1
        model = train_rl_policy(
            env_params_string=env_params_list[0],
            total_timesteps=config.total_timesteps,
            checkpoint_freq=config.checkpoint_freq,
            use_parallel=use_parallel,
            n_envs=n_scenarios,
            resume_from_model=args.resume_from,
            pretrain_from_model=args.pretrain_from,
            cycle_lengths=cycle_lengths,
            cycle_strategy=cycle_strategy,
            experiment_config=config,
            env_params_list=env_params_list if use_parallel else None,
            models_dir=args.models_dir,
        )

        elapsed = time.time() - start_time
        logger.info(f"=== SERVER TRAINING COMPLETE ===")
        logger.info(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Speed: {config.total_timesteps/elapsed:.1f} timesteps/sec")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except (BrokenPipeError, ConnectionError, OSError) as e:
        logger.error(f"SUMO connection error: {e}")
        logger.info("This is common during long training runs.")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
