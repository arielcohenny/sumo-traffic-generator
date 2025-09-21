#!/usr/bin/env python3
"""
Production RL Training Script for Traffic Signal Control.

This script provides production-scale RL training with proper workspace isolation,
parallel environment support, and independent execution capabilities.
"""

import argparse
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rl.training import train_rl_policy
from src.rl.config import get_rl_config
from src.rl.constants import (
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_MODEL_SAVE_PATH, DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_N_PARALLEL_ENVS, MIN_PARALLEL_ENVS, MAX_PARALLEL_ENVS,
    PARALLEL_WORKSPACE_PREFIX, SINGLE_ENV_THRESHOLD
)


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

    # Training scale parameters
    parser.add_argument(
        '--timesteps', type=int, default=100000,
        help="Total training timesteps (production scale: 100k-1M)"
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

    # Logging and monitoring
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help="Logging level"
    )
    parser.add_argument(
        '--log-file', type=str,
        help="Optional log file (if not specified, uses auto-generated name)"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Minimal output (overrides log-level)"
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

    # Setup unique workspace and logging
    workspace = create_unique_workspace(args.workspace_base)

    if args.log_file is None:
        args.log_file = f"logs/rl_training_{workspace}.log"

    if args.quiet:
        args.log_level = "WARNING"

    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("=== PRODUCTION RL TRAINING STARTED ===")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Training timesteps: {args.timesteps:,}")
    logger.info(f"Parallel environments: {args.parallel_envs}")
    logger.info(f"Log file: {args.log_file}")

    try:
        # Validate environment
        if not validate_training_environment():
            logger.error("Training environment validation failed")
            return 1

        # Get RL configuration
        config = get_rl_config()
        logger.info(f"RL config: {config.grid_dimension}×{config.grid_dimension} grid, {config.num_vehicles} vehicles")

        # Configure parallel environments
        if args.single_env:
            args.parallel_envs = SINGLE_ENV_THRESHOLD
            use_parallel = False
            logger.info("Single environment mode enabled")
        else:
            args.parallel_envs = max(MIN_PARALLEL_ENVS, min(args.parallel_envs, MAX_PARALLEL_ENVS))
            use_parallel = True
            logger.info(f"Parallel training with {args.parallel_envs} environments")

        # Create model save path with unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(args.models_dir, f"{args.model_name}_{timestamp}")

        # Calculate estimated training time
        estimated_time_minutes = (args.timesteps / 1000) * 0.5  # Rough estimate
        logger.info(f"Estimated training time: ~{estimated_time_minutes:.1f} minutes")

        # Start training with robustness features
        logger.info("Starting production RL training...")
        start_time = time.time()

        try:
            model = train_rl_policy(
                config=config,
                total_timesteps=args.timesteps,
                model_save_path=model_save_path,
                checkpoint_freq=args.checkpoint_freq,
                base_workspace=workspace,
                use_parallel=use_parallel,
                n_envs=args.parallel_envs,
                resume_from_model=args.resume_from
            )
        except (BrokenPipeError, ConnectionError, OSError) as e:
            logger.warning(f"SUMO connection error during training: {e}")
            logger.info("This is common during long training runs. The training will retry...")
            # Attempt recovery with single environment if parallel failed
            if use_parallel and args.parallel_envs > 1:
                logger.info("Retrying with single environment for stability...")
                model = train_rl_policy(
                    config=config,
                    total_timesteps=args.timesteps,
                    model_save_path=model_save_path,
                    checkpoint_freq=args.checkpoint_freq,
                    base_workspace=workspace + "_recovery",
                    use_parallel=False,
                    n_envs=1,
                    resume_from_model=args.resume_from
                )
            else:
                raise

        training_time = time.time() - start_time
        logger.info("=== PRODUCTION TRAINING COMPLETED ===")
        logger.info(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"Training speed: {args.timesteps/training_time:.1f} timesteps/second")
        logger.info(f"Model saved: {model_save_path}.zip")
        logger.info(f"Workspace: {workspace}")
        logger.info(f"Log file: {args.log_file}")

        # Save training summary
        summary_file = f"logs/training_summary_{workspace}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Production RL Training Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"End time: {datetime.now()}\n")
            f.write(f"Training duration: {training_time:.1f} seconds\n")
            f.write(f"Total timesteps: {args.timesteps:,}\n")
            f.write(f"Parallel environments: {args.parallel_envs}\n")
            f.write(f"Training speed: {args.timesteps/training_time:.1f} timesteps/second\n")
            f.write(f"Model saved: {model_save_path}.zip\n")
            f.write(f"Workspace: {workspace}\n")
            f.write(f"Config: {config.grid_dimension}×{config.grid_dimension} grid, {config.num_vehicles} vehicles\n")

        logger.info(f"Training summary saved: {summary_file}")
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