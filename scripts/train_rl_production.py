#!/usr/bin/env python3
"""
Production RL Training Script for Traffic Signal Control.

This script provides production-scale RL training with proper workspace isolation,
parallel environment support, and independent execution capabilities.
"""

# Fix NumPy + Python 3.13 multiprocessing compatibility issue
# Must be set before any imports that use NumPy/PyTorch
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
# Alternative fix for NumPy 2.3.1 multiprocessing recursion bug
os.environ['NUMPY_DISABLE_THREADING'] = '1'
# Critical fix for NumPy 2.3.1 + Python 3.13 longdouble multiprocessing error
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rl.training import train_rl_policy
from src.rl.constants import (
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_MODEL_SAVE_PATH, DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_N_PARALLEL_ENVS, MIN_PARALLEL_ENVS, MAX_PARALLEL_ENVS,
    PARALLEL_WORKSPACE_PREFIX, SINGLE_ENV_THRESHOLD, DEFAULT_CYCLE_LENGTH
)
from src.rl.experiment_config import load_experiment, save_experiment, validate_experiment


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging for production training."""
    level = getattr(logging, log_level.upper())

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Suppress verbose RL logging for cleaner production output
    # Set RL modules to WARNING level (only show important messages)
    logging.getLogger('src.rl.controller').setLevel(logging.WARNING)
    logging.getLogger('src.rl.environment').setLevel(logging.WARNING)
    logging.getLogger('TrafficControlEnv').setLevel(logging.WARNING)

    # Keep training.py at INFO level for progress monitoring
    logging.getLogger('src.rl.training').setLevel(logging.INFO)


def create_unique_workspace(base_name: str = "rl_production") -> str:
    """Create unique workspace with timestamp and process ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    return f"{base_name}_{timestamp}_{process_id}"


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser for production training."""
    parser = argparse.ArgumentParser(
        description="Production RL training for traffic signal control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment config (single source of truth)
    parser.add_argument(
        '--experiment', type=str, default=None,
        help="Path to experiment YAML config file (replaces hardcoded constants)"
    )

    # Training scale parameters
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help="Total training timesteps - overrides experiment YAML if provided"
    )
    parser.add_argument(
        '--parallel-envs', type=int, default=4,
        help=f"Number of parallel environments ({MIN_PARALLEL_ENVS}-{MAX_PARALLEL_ENVS})"
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=10000,
        help="Frequency of model checkpointing"
    )

    # Workspace and model management
    parser.add_argument(
        '--workspace-base', type=str, default="rl_production",
        help="Base name for unique workspace (will add timestamp and PID)"
    )
    parser.add_argument(
        '--model-name', type=str, default="rl_traffic_production",
        help="Base name for saved model"
    )
    parser.add_argument(
        '--models-dir', type=str, default="models",
        help="Directory to save models"
    )

    # Training configuration
    parser.add_argument(
        '--single-env', action='store_true',
        help="Use single environment instead of parallel (for debugging)"
    )
    parser.add_argument(
        '--resume-from', type=str,
        help="Path to model to resume training from"
    )
    parser.add_argument(
        '--pretrain-from', type=str,
        help="Path to pre-trained model (from imitation learning) to initialize from"
    )

    # Logging and monitoring
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help="Console logging level (file logs always in model directory)"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Minimal console output (overrides log-level)"
    )

    # Advanced training options
    parser.add_argument(
        '--tensorboard', action='store_true',
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        '--save-every', type=int, default=50000,
        help="Save model every N timesteps"
    )

    # Environment parameters (required unless --experiment provides simulation params)
    parser.add_argument(
        '--env-params', type=str, default=None,
        help='Environment parameters string (e.g., "--network-seed 42 --grid_dimension 5 --num_vehicles 4500 ..."). '
             'If --experiment is provided, simulation params from YAML are used as defaults.'
    )

    # RL cycle parameters
    parser.add_argument(
        '--cycle-lengths',
        type=int,
        nargs='+',
        default=[DEFAULT_CYCLE_LENGTH],
        help=f'List of cycle lengths in seconds (default: [{DEFAULT_CYCLE_LENGTH}]). Example: --cycle-lengths 60 90 120'
    )
    parser.add_argument(
        '--cycle-strategy',
        type=str,
        choices=['fixed', 'random', 'sequential'],
        default='fixed',
        help='Strategy for selecting cycle length per episode (default: fixed)'
    )

    return parser


def validate_training_environment():
    """Validate training environment and dependencies."""
    logger = logging.getLogger(__name__)

    # Check required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Check dependencies
    try:
        from stable_baselines3 import PPO
        logger.info("✓ stable-baselines3 available")
    except ImportError as e:
        logger.error(f"✗ stable-baselines3 not available: {e}")
        return False

    try:
        import sumolib
        import traci
        logger.info("✓ SUMO libraries available")
    except ImportError as e:
        logger.error(f"✗ SUMO libraries not available: {e}")
        return False

    logger.info("Training environment validation passed")
    return True


def main():
    """Main production training script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup console-only logging (all file logs go to model directory)
    if args.quiet:
        args.log_level = "WARNING"

    setup_logging(args.log_level, log_file=None)  # Console only
    logger = logging.getLogger(__name__)

    # Load experiment config if provided
    experiment_config = None
    if args.experiment:
        logger.info(f"Loading experiment config: {args.experiment}")
        experiment_config = load_experiment(args.experiment)
        warnings = validate_experiment(experiment_config)
        if warnings:
            for w in warnings:
                logger.warning(f"Config warning: {w}")
        logger.info(f"Experiment: {experiment_config.name} - {experiment_config.description}")

        # Apply CLI overrides to experiment config
        if args.timesteps is not None:
            experiment_config.total_timesteps = args.timesteps
        if args.checkpoint_freq is not None:
            experiment_config.checkpoint_freq = args.checkpoint_freq

        # Build env-params from experiment config simulation params if not provided
        if args.env_params is None:
            args.env_params = (
                f"--network-seed {experiment_config.network_seed} "
                f"--grid_dimension {experiment_config.grid_dimension} "
                f"--num_vehicles {experiment_config.num_vehicles} "
                f"--end-time {experiment_config.end_time} "
                f"--routing_strategy '{experiment_config.routing_strategy}' "
                f"--vehicle_types '{experiment_config.vehicle_types}' "
                f"--departure_pattern {experiment_config.departure_pattern}"
            )
            logger.info(f"Built env-params from experiment config: {args.env_params}")

        # Use experiment values as defaults for timesteps/checkpoint
        if args.timesteps is None:
            args.timesteps = experiment_config.total_timesteps
        if args.cycle_lengths == [DEFAULT_CYCLE_LENGTH]:
            args.cycle_lengths = experiment_config.cycle_lengths

    # Validate that env-params is available
    if args.env_params is None:
        logger.error("--env-params is required when --experiment is not provided")
        return 1

    # Default timesteps if still None
    if args.timesteps is None:
        args.timesteps = 100000

    logger.info("=== PRODUCTION RL TRAINING STARTED ===")
    logger.info(f"Training timesteps: {args.timesteps:,}")
    logger.info(f"Parallel environments: {args.parallel_envs}")
    logger.info(f"Cycle lengths: {args.cycle_lengths}")
    logger.info(f"Cycle strategy: {args.cycle_strategy}")
    if experiment_config:
        logger.info(f"Experiment: {experiment_config.name}")
        logger.info(f"Reward function: {experiment_config.reward_function}")
    logger.info("All logs and files will be organized in models/rl_YYYYMMDD_HHMMSS/")

    try:
        # Validate environment
        if not validate_training_environment():
            logger.error("Training environment validation failed")
            return 1

        # Configure parallel environments
        if args.single_env:
            args.parallel_envs = SINGLE_ENV_THRESHOLD
            use_parallel = False
            logger.info("Single environment mode enabled")
        else:
            args.parallel_envs = max(MIN_PARALLEL_ENVS, min(args.parallel_envs, MAX_PARALLEL_ENVS))
            use_parallel = True
            logger.info(f"Parallel training with {args.parallel_envs} environments")

        # Validate --pretrain-from and --resume-from are mutually exclusive
        if args.pretrain_from and args.resume_from:
            logger.error("--pretrain-from and --resume-from are mutually exclusive")
            logger.error("  --pretrain-from: Start from imitation learning pre-trained model")
            logger.error("  --resume-from: Continue training existing model")
            return 1

        if args.pretrain_from:
            if not os.path.exists(args.pretrain_from):
                logger.error(f"Pre-trained model not found: {args.pretrain_from}")
                return 1
            logger.info(f"Starting from pre-trained model: {args.pretrain_from}")

        # Calculate estimated training time
        estimated_time_minutes = (args.timesteps / 1000) * 2.1  # Based on actual measurements
        logger.info(f"Estimated training time: ~{estimated_time_minutes:.1f} minutes")

        # Start training with robustness features
        logger.info("Starting production RL training...")
        start_time = time.time()

        try:
            model = train_rl_policy(
                env_params_string=args.env_params,
                total_timesteps=args.timesteps,
                model_save_path=None,  # Now auto-generated in model directory
                checkpoint_freq=args.checkpoint_freq,
                base_workspace=None,  # Now auto-generated in model directory
                use_parallel=use_parallel,
                n_envs=args.parallel_envs,
                resume_from_model=args.resume_from,
                pretrain_from_model=args.pretrain_from,
                cycle_lengths=args.cycle_lengths,
                cycle_strategy=args.cycle_strategy,
                experiment_config=experiment_config
            )
        except (BrokenPipeError, ConnectionError, OSError) as e:
            logger.warning(f"SUMO connection error during training: {e}")
            logger.info("This is common during long training runs. The training will retry...")
            # Attempt recovery with single environment if parallel failed
            if use_parallel and args.parallel_envs > 1:
                logger.info("Retrying with single environment for stability...")
                model = train_rl_policy(
                    env_params_string=args.env_params,
                    total_timesteps=args.timesteps,
                    model_save_path=None,
                    checkpoint_freq=args.checkpoint_freq,
                    base_workspace=None,
                    use_parallel=False,
                    n_envs=1,
                    resume_from_model=args.resume_from,
                    pretrain_from_model=args.pretrain_from,
                    cycle_lengths=args.cycle_lengths,
                    cycle_strategy=args.cycle_strategy,
                    experiment_config=experiment_config
                )
            else:
                raise

        training_time = time.time() - start_time
        logger.info("=== PRODUCTION TRAINING COMPLETED ===")
        logger.info(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"Training speed: {args.timesteps/training_time:.1f} timesteps/second")
        logger.info("All model files are organized in the model directory (see training logs above)")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())