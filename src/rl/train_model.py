#!/usr/bin/env python3
"""
Training script for RL traffic signal control.

This script provides a command-line interface for training RL models
with configurable parameters and automatic model versioning.
"""

import argparse
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rl.training import train_rl_policy, evaluate_rl_policy
from src.rl.config import get_rl_config
from src.rl.constants import (
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_MODEL_SAVE_PATH, DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_N_PARALLEL_ENVS, MIN_PARALLEL_ENVS, MAX_PARALLEL_ENVS,
    PARALLEL_WORKSPACE_PREFIX, DEFAULT_EVAL_EPISODES, SINGLE_ENV_THRESHOLD
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rl_training.log')
        ]
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train RL policy for traffic signal control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        '--timesteps', type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help=f"Total training timesteps"
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=DEFAULT_CHECKPOINT_FREQ,
        help="Frequency of model checkpointing"
    )
    parser.add_argument(
        '--parallel-envs', type=int, default=DEFAULT_N_PARALLEL_ENVS,
        help=f"Number of parallel environments ({MIN_PARALLEL_ENVS}-{MAX_PARALLEL_ENVS})"
    )

    # Model management
    parser.add_argument(
        '--model-path', type=str, default=DEFAULT_MODEL_SAVE_PATH,
        help="Path to save trained model"
    )
    parser.add_argument(
        '--workspace', type=str, default=PARALLEL_WORKSPACE_PREFIX,
        help="Base workspace directory for training environments"
    )

    # Evaluation
    parser.add_argument(
        '--evaluate', action='store_true',
        help="Evaluate model after training"
    )
    parser.add_argument(
        '--eval-episodes', type=int, default=DEFAULT_EVAL_EPISODES,
        help="Number of episodes for evaluation"
    )

    # Training configuration
    parser.add_argument(
        '--single-env', action='store_true',
        help="Use single environment instead of parallel environments"
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Enable verbose logging"
    )

    return parser


def main():
    """Main training script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=== RL TRAINING SCRIPT STARTED ===")
    logger.info(f"Training timesteps: {args.timesteps}")
    logger.info(f"Parallel environments: {args.parallel_envs}")
    logger.info(f"Model save path: {args.model_path}")

    try:
        # Get RL configuration
        config = get_rl_config()

        # Override parallel environments if single-env mode
        if args.single_env:
            args.parallel_envs = SINGLE_ENV_THRESHOLD
            logger.info("Single environment mode enabled")

        # Validate parallel environments
        args.parallel_envs = max(MIN_PARALLEL_ENVS, min(args.parallel_envs, MAX_PARALLEL_ENVS))
        logger.info(f"Using {args.parallel_envs} parallel environments")

        # Train model (config is frozen, so we pass parallel_envs separately)
        logger.info("Starting RL model training...")

        # Override the training function to handle parallel environments
        from src.rl.training import create_vectorized_env, train_rl_policy

        # Create custom training call to handle parallel environments
        if not args.single_env and args.parallel_envs > SINGLE_ENV_THRESHOLD:
            logger.info(f"Using parallel training with {args.parallel_envs} environments")
            # We'll need to modify the training approach for parallel envs
            use_parallel = True
            n_envs = args.parallel_envs
        else:
            logger.info("Using single environment training")
            use_parallel = False
            n_envs = SINGLE_ENV_THRESHOLD

        model = train_rl_policy(
            config=config,
            total_timesteps=args.timesteps,
            model_save_path=args.model_path,
            checkpoint_freq=args.checkpoint_freq,
            base_workspace=args.workspace,
            use_parallel=use_parallel,
            n_envs=n_envs
        )

        logger.info("Training completed successfully!")

        # Evaluate model if requested
        if args.evaluate:
            logger.info("Starting model evaluation...")
            # Use the saved model path from training
            model_files = [f for f in os.listdir(os.path.dirname(args.model_path))
                          if f.startswith(os.path.basename(args.model_path)) and f.endswith('.zip')]

            if model_files:
                # Use most recent model
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(os.path.dirname(args.model_path), latest_model)

                metrics = evaluate_rl_policy(
                    model_path=model_path,
                    config=config,
                    n_episodes=args.eval_episodes
                )

                logger.info("Evaluation completed!")
                logger.info("Final metrics:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value:.3f}")
            else:
                logger.warning("No trained model found for evaluation")

        logger.info("=== RL TRAINING SCRIPT COMPLETED ===")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()