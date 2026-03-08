#!/usr/bin/env python3
"""
Fresh single-env RL training from a self-contained experiment folder.

This script takes ZERO arguments. It reads config.yaml from the current
working directory and writes all outputs (checkpoints, eval logs, best
model, training.log) into that same directory.

Usage:
    cd /path/to/experiment_folder   # must contain config.yaml
    python /path/to/train_single.py
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMPY_DISABLE_THREADING'] = '1'
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import logging
import sys
import time

# Add project root to path so src.rl imports work from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rl.server.config_loader import load_single_config, build_env_params_string, has_checkpoints
from src.rl.training import train_rl_policy
from src.rl.experiment_config import save_experiment


def setup_logging(experiment_dir: str):
    """Setup logging to both console and training.log in the experiment dir."""
    log_file = os.path.join(experiment_dir, "training.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

    # Suppress verbose RL logging for cleaner output
    logging.getLogger('src.rl.controller').setLevel(logging.WARNING)
    logging.getLogger('src.rl.environment').setLevel(logging.WARNING)
    logging.getLogger('TrafficControlEnv').setLevel(logging.WARNING)
    logging.getLogger('src.rl.training').setLevel(logging.INFO)


def main():
    experiment_dir = os.getcwd()

    # ── Validate config.yaml exists ───────────────────────────────────
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found in {experiment_dir}", file=sys.stderr)
        print("This script must be run from an experiment folder containing config.yaml.", file=sys.stderr)
        sys.exit(1)

    # ── Safety: refuse to start if checkpoints already exist ──────────
    if has_checkpoints(experiment_dir):
        print(
            f"ERROR: Checkpoints already exist in {experiment_dir}/checkpoint/",
            file=sys.stderr,
        )
        print("Use resume_single.py to continue training from an existing checkpoint.", file=sys.stderr)
        sys.exit(1)

    # ── Logging ───────────────────────────────────────────────────────
    setup_logging(experiment_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"=== FRESH SINGLE-ENV TRAINING ===")
    logger.info(f"Experiment dir: {experiment_dir}")

    # ── Load config ───────────────────────────────────────────────────
    logger.info(f"Loading config from {config_path}")
    config, scenario_dict, traffic_control = load_single_config(config_path)

    # ── Build env params ──────────────────────────────────────────────
    env_params_string = build_env_params_string(config, scenario_dict, traffic_control)
    logger.info(f"Env params: {env_params_string}")

    # ── Save resolved config ──────────────────────────────────────────
    resolved_path = os.path.join(experiment_dir, "experiment.yaml")
    save_experiment(config, resolved_path)
    logger.info(f"Resolved config saved to {resolved_path}")

    # ── Log summary ───────────────────────────────────────────────────
    logger.info(f"Timesteps: {config.total_timesteps}")
    logger.info(f"Checkpoint freq: {config.checkpoint_freq}")
    logger.info(f"Reward: {config.reward_function}")
    logger.info(f"Grid: {config.grid_dimension}x{config.grid_dimension}")
    logger.info(f"Traffic control: {traffic_control}")
    logger.info(f"Cycle lengths: {config.cycle_lengths}")
    logger.info(f"Cycle strategy: {config.cycle_strategy}")

    # ── Train ─────────────────────────────────────────────────────────
    try:
        start_time = time.time()

        train_rl_policy(
            env_params_string=env_params_string,
            total_timesteps=config.total_timesteps,
            checkpoint_freq=config.checkpoint_freq,
            use_parallel=False,
            n_envs=1,
            resume_from_model=None,
            cycle_lengths=config.cycle_lengths,
            cycle_strategy=config.cycle_strategy,
            experiment_config=config,
            env_params_list=None,
            models_dir=experiment_dir,
            initial_timesteps_override=None,
        )

        elapsed = time.time() - start_time
        logger.info(f"=== TRAINING COMPLETE ===")
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
