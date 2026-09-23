"""
Utility functions for RL training pipeline.

This module provides helper functions for model management, training configuration,
and evaluation utilities.
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .constants import (
    DEFAULT_MODELS_DIRECTORY, TRAINING_MODEL_METADATA_EXTENSION,
    TRAINING_BEST_MODEL_PREFIX, TRAINING_CHECKPOINT_PREFIX,
    DEFAULT_EVAL_EPISODES, TRAINING_MIN_IMPROVEMENT_THRESHOLD,
    CLEANUP_DEFAULT_KEEP_LATEST, MODEL_SIZE_CONVERSION_FACTOR,
    TRAINING_PROGRESS_MIN_MODELS, PARALLEL_TRAINING_EFFICIENCY,
    TIME_CONVERSION_MINUTES, TIME_CONVERSION_HOURS, DEFAULT_TIME_PER_TIMESTEP,
    PARALLEL_ENV_EFFICIENCY_BASELINE
)


def find_latest_model(model_directory: str = DEFAULT_MODELS_DIRECTORY,
                     model_prefix: str = "rl_traffic") -> Optional[str]:
    """Find the most recently trained model.

    Args:
        model_directory: Directory to search for models
        model_prefix: Prefix of model files to search for

    Returns:
        Path to latest model or None if no models found
    """
    if not os.path.exists(model_directory):
        return None

    # Search for model files
    pattern = os.path.join(model_directory, f"{model_prefix}*.zip")
    model_files = glob.glob(pattern)

    if not model_files:
        return None

    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def find_best_model(model_directory: str = DEFAULT_MODELS_DIRECTORY) -> Optional[str]:
    """Find the best performing model from evaluation callbacks.

    Args:
        model_directory: Directory to search for models

    Returns:
        Path to best model or None if no best model found
    """
    best_model_path = os.path.join(model_directory, TRAINING_BEST_MODEL_PREFIX, "best_model.zip")
    return best_model_path if os.path.exists(best_model_path) else None


def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a trained model.

    Args:
        model_path: Path to the model file

    Returns:
        Model metadata dictionary or None if metadata not found
    """
    metadata_path = model_path + TRAINING_MODEL_METADATA_EXTENSION
    if not os.path.exists(metadata_path):
        # Try without .zip extension
        metadata_path = os.path.splitext(model_path)[0] + TRAINING_MODEL_METADATA_EXTENSION

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load metadata: {e}")
        return None


def list_available_models(model_directory: str = DEFAULT_MODELS_DIRECTORY) -> List[Dict[str, Any]]:
    """List all available trained models with their metadata.

    Args:
        model_directory: Directory to search for models

    Returns:
        List of model information dictionaries
    """
    if not os.path.exists(model_directory):
        return []

    models = []
    pattern = os.path.join(model_directory, "*.zip")
    model_files = glob.glob(pattern)

    for model_path in model_files:
        model_info = {
            'path': model_path,
            'name': os.path.basename(model_path),
            'size_mb': os.path.getsize(model_path) / (MODEL_SIZE_CONVERSION_FACTOR * MODEL_SIZE_CONVERSION_FACTOR),
            'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path)),
            'metadata': load_model_metadata(model_path)
        }
        models.append(model_info)

    # Sort by modification time (most recent first)
    models.sort(key=lambda x: x['modified_time'], reverse=True)
    return models


def cleanup_old_checkpoints(model_directory: str = DEFAULT_MODELS_DIRECTORY,
                           keep_latest: int = CLEANUP_DEFAULT_KEEP_LATEST) -> int:
    """Clean up old checkpoint files to save disk space.

    Args:
        model_directory: Directory containing model checkpoints
        keep_latest: Number of latest checkpoints to keep

    Returns:
        Number of files deleted
    """
    if not os.path.exists(model_directory):
        return 0

    logger = logging.getLogger(__name__)

    # Find checkpoint files
    pattern = os.path.join(model_directory, f"{TRAINING_CHECKPOINT_PREFIX}*.zip")
    checkpoint_files = glob.glob(pattern)

    if len(checkpoint_files) <= keep_latest:
        return 0

    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=os.path.getmtime)

    # Delete oldest checkpoints
    files_to_delete = checkpoint_files[:-keep_latest]
    deleted_count = 0

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            logger.info(f"Deleted old checkpoint: {os.path.basename(file_path)}")

            # Also delete metadata if it exists
            metadata_path = file_path + TRAINING_MODEL_METADATA_EXTENSION
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

    logger.info(f"Cleaned up {deleted_count} old checkpoint files")
    return deleted_count


def compare_model_performance(model_paths: List[str],
                            eval_episodes: int = DEFAULT_EVAL_EPISODES) -> Dict[str, Dict[str, float]]:
    """Compare performance of multiple models.

    Args:
        model_paths: List of paths to models to compare
        eval_episodes: Number of episodes for evaluation

    Returns:
        Dictionary mapping model names to performance metrics
    """
    from .training import evaluate_rl_policy
    logger = logging.getLogger(__name__)
    logger.warning("compare_model_performance uses legacy config path - use ExperimentConfig instead")
    results = {}

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        logger.info(f"Evaluating model: {model_name}")

        try:
            metrics = evaluate_rl_policy(
                model_path=model_path,
                config=config,
                n_episodes=eval_episodes
            )
            results[model_name] = metrics
            logger.info(f"Evaluation completed for {model_name}")

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    return results


def analyze_training_progress(model_directory: str = DEFAULT_MODELS_DIRECTORY) -> Optional[Dict[str, Any]]:
    """Analyze training progress from model metadata.

    Args:
        model_directory: Directory containing training models

    Returns:
        Training progress analysis or None if insufficient data
    """
    models = list_available_models(model_directory)

    if len(models) < TRAINING_PROGRESS_MIN_MODELS:
        return None

    # Extract training metrics from metadata
    training_times = []
    timestep_counts = []
    model_names = []

    for model in models:
        metadata = model.get('metadata')
        if metadata:
            training_times.append(metadata.get('training_time_seconds', 0))
            timestep_counts.append(metadata.get('training_timesteps', 0))
            model_names.append(model['name'])

    if not training_times:
        return None

    analysis = {
        'total_models': len(models),
        'total_training_time': sum(training_times),
        'average_training_time': np.mean(training_times),
        'total_timesteps': sum(timestep_counts),
        'average_timesteps_per_model': np.mean(timestep_counts),
        'training_efficiency': {
            'fastest_training': min(training_times),
            'slowest_training': max(training_times),
            'time_per_timestep': np.mean([t/ts for t, ts in zip(training_times, timestep_counts) if ts > 0])
        },
        'model_progression': list(zip(model_names, training_times, timestep_counts))
    }

    return analysis


def validate_training_environment() -> Dict[str, bool]:
    """Validate that the training environment is properly configured.

    Returns:
        Dictionary of validation results
    """
    logger = logging.getLogger(__name__)
    results = {}

    # Check model directory
    try:
        os.makedirs(DEFAULT_MODELS_DIRECTORY, exist_ok=True)
        results['models_directory'] = True
        logger.info(f"Models directory accessible: {DEFAULT_MODELS_DIRECTORY}")
    except Exception as e:
        results['models_directory'] = False
        logger.error(f"Models directory not accessible: {e}")

    # Check stable-baselines3 availability
    try:
        from stable_baselines3 import PPO
        results['stable_baselines3'] = True
        logger.info("stable-baselines3 available")
    except ImportError as e:
        results['stable_baselines3'] = False
        logger.error(f"stable-baselines3 not available: {e}")

    # Check TensorBoard availability
    try:
        import tensorboard
        results['tensorboard'] = True
        logger.info("TensorBoard available")
    except ImportError:
        results['tensorboard'] = False
        logger.warning("TensorBoard not available - training logs will be limited")

    # Check RL environment (basic import check only - full validation requires SUMO)
    try:
        from .environment import TrafficControlEnv
        results['rl_environment'] = True
        logger.info("RL environment import check passed")
    except Exception as e:
        results['rl_environment'] = False
        logger.error(f"RL environment import failed: {e}")

    return results


def create_training_summary(model_path: str) -> str:
    """Create a formatted summary of training results.

    Args:
        model_path: Path to the trained model

    Returns:
        Formatted training summary string
    """
    metadata = load_model_metadata(model_path)
    if not metadata:
        return f"No metadata available for model: {os.path.basename(model_path)}"

    summary = f"""
=== Training Summary ===
Model: {metadata.get('model_name', 'Unknown')}
Training Date: {metadata.get('training_date', 'Unknown')}
Training Time: {metadata.get('training_time_seconds', 0):.1f} seconds
Total Timesteps: {metadata.get('training_timesteps', 0):,}

Network Architecture: {metadata.get('network_architecture', 'Unknown')}

Hyperparameters:
"""

    hyperparams = metadata.get('hyperparameters', {})
    for key, value in hyperparams.items():
        summary += f"  {key}: {value}\n"

    env_info = metadata.get('environment_info', {})
    if env_info:
        summary += f"\nEnvironment Configuration:\n"
        for key, value in env_info.items():
            summary += f"  {key}: {value}\n"

    return summary.strip()


def estimate_training_time(timesteps: int,
                          parallel_envs: int = 1,
                          reference_time_per_timestep: float = DEFAULT_TIME_PER_TIMESTEP) -> Dict[str, float]:
    """Estimate training time based on configuration.

    Args:
        timesteps: Total training timesteps
        parallel_envs: Number of parallel environments
        reference_time_per_timestep: Reference time per timestep (seconds)

    Returns:
        Dictionary with time estimates
    """
    # Adjust for parallel efficiency
    parallel_efficiency = PARALLEL_TRAINING_EFFICIENCY if parallel_envs > 1 else PARALLEL_ENV_EFFICIENCY_BASELINE
    effective_speedup = parallel_envs * parallel_efficiency

    estimated_seconds = (timesteps * reference_time_per_timestep) / effective_speedup

    return {
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_seconds / TIME_CONVERSION_MINUTES,
        'estimated_hours': estimated_seconds / TIME_CONVERSION_HOURS,
        'parallel_envs': parallel_envs,
        'parallel_efficiency': parallel_efficiency,
        'effective_speedup': effective_speedup
    }