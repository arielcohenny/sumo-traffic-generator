"""
RL Training Pipeline for Traffic Signal Control.

This module implements the PPO training pipeline for learning
network-wide traffic coordination policies.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .environment import TrafficControlEnv
from .constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_CLIP_RANGE, DEFAULT_BATCH_SIZE,
    DEFAULT_N_STEPS, DEFAULT_N_EPOCHS, DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA,
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_CHECKPOINT_FREQ, DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_EVAL_EPISODES, DEFAULT_MODELS_DIRECTORY, SINGLE_ENV_THRESHOLD,
    DEFAULT_INITIAL_STEP, DEFAULT_INITIAL_TIME, DEFAULT_FALLBACK_VALUE, PARALLEL_WORKSPACE_PREFIX,
    STD_CALCULATION_MIN_VALUES, STD_CALCULATION_FALLBACK,
    TRAINING_NETWORK_ARCHITECTURE, TRAINING_DEVICE_AUTO, TRAINING_TENSORBOARD_LOG_DIR,
    TRAINING_BEST_MODEL_PREFIX, TRAINING_CHECKPOINT_PREFIX, TRAINING_MODEL_METADATA_EXTENSION,
    TRAINING_EVAL_EPISODES_PER_CHECKPOINT, TRAINING_PATIENCE_EPISODES,
    TRAINING_MIN_IMPROVEMENT_THRESHOLD, TRAINING_PROGRESS_LOG_INTERVAL,
    TRAINING_POLICY_TYPE, TRAINING_ACTIVATION_FUNCTION, TRAINING_VERBOSE_LEVEL,
    VARIANCE_CALCULATION_POWER
)
from .config import get_rl_config


class TrafficMetricsCallback(BaseCallback):
    """Custom callback for tracking traffic-specific metrics during training."""

    def __init__(self, log_interval: int = TRAINING_PROGRESS_LOG_INTERVAL, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.completion_rates = []
        self.last_log_step = DEFAULT_INITIAL_STEP

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Log traffic metrics periodically
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self._log_traffic_metrics()
            self.last_log_step = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        """Log traffic metrics at the end of each rollout."""
        # Extract episode information from training environment
        if hasattr(self.training_env, 'get_attr'):
            try:
                # Get episode data from vectorized environments
                env_rewards = self.training_env.get_attr('episode_rewards')
                env_lengths = self.training_env.get_attr('episode_lengths')

                # Flatten and store episode data
                for rewards in env_rewards:
                    if rewards:
                        self.episode_rewards.extend(rewards)
                for lengths in env_lengths:
                    if lengths:
                        self.episode_lengths.extend(lengths)

            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not extract episode metrics: {e}")

        # Log current metrics
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:])
            mean_length = np.mean(self.episode_lengths[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:])

            self.logger.record('traffic/mean_episode_reward', mean_reward)
            self.logger.record('traffic/mean_episode_length', mean_length)
            self.logger.record('traffic/total_episodes', len(self.episode_rewards))

    def _log_traffic_metrics(self):
        """Log detailed traffic metrics."""
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:]
            recent_lengths = self.episode_lengths[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:]

            if recent_rewards:
                self.logger.record('traffic/recent_mean_reward', np.mean(recent_rewards))
                self.logger.record('traffic/recent_std_reward', np.std(recent_rewards))
                self.logger.record('traffic/recent_min_reward', np.min(recent_rewards))
                self.logger.record('traffic/recent_max_reward', np.max(recent_rewards))

            if recent_lengths:
                self.logger.record('traffic/recent_mean_length', np.mean(recent_lengths))
                self.logger.record('traffic/recent_std_length', np.std(recent_lengths))


def make_env(config: Any, env_index: int, base_workspace: str = PARALLEL_WORKSPACE_PREFIX):
    """
    Create a single environment for vectorization.

    Args:
        config: RL training configuration
        env_index: Environment index for unique workspace
        base_workspace: Base workspace directory name

    Returns:
        callable: Environment creation function
    """
    def _init():
        # Get unique CLI arguments for this environment
        cli_args = config.get_cli_args_for_env(env_index, base_workspace)

        # Create environment with unique workspace
        env = TrafficControlEnv(cli_args)
        return env

    return _init


def create_vectorized_env(config: Any, n_envs: int = None, base_workspace: str = PARALLEL_WORKSPACE_PREFIX):
    """
    Create vectorized environment for parallel training.

    Args:
        config: RL training configuration
        n_envs: Number of parallel environments (uses config.n_parallel_envs if None)
        base_workspace: Base workspace directory name

    Returns:
        VecEnv: Vectorized environment for parallel training
    """
    logger = logging.getLogger(__name__)

    if n_envs is None:
        n_envs = config.n_parallel_envs

    logger.info(f"Creating {n_envs} parallel environments with base workspace: {base_workspace}")

    # Create list of environment creation functions
    env_fns = [make_env(config, i, base_workspace) for i in range(n_envs)]

    # Use SubprocVecEnv for true parallel execution
    # Falls back to DummyVecEnv for single environment or debugging
    if n_envs == SINGLE_ENV_THRESHOLD:
        logger.info("Using DummyVecEnv for single environment")
        vec_env = DummyVecEnv(env_fns)
    else:
        logger.info(f"Using SubprocVecEnv for {n_envs} parallel processes")
        vec_env = SubprocVecEnv(env_fns)

    logger.info(f"Vectorized environment created: {type(vec_env).__name__}")
    return vec_env


def train_rl_policy(config: Any,
                   total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
                   model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
                   checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ,
                   base_workspace: str = PARALLEL_WORKSPACE_PREFIX,
                   use_parallel: bool = True,
                   n_envs: int = None,
                   resume_from_model: str = None) -> PPO:
    """
    Train PPO policy for traffic signal control.

    Args:
        config: RL training configuration
        total_timesteps: Total training timesteps
        model_save_path: Path to save trained model
        checkpoint_freq: Frequency of model checkpointing
        base_workspace: Base workspace directory for parallel environments
        use_parallel: Whether to use parallel environments (True) or single environment (False)
        n_envs: Number of parallel environments (if None, uses config.n_parallel_envs)
        resume_from_model: Path to existing model to resume training from (if None, creates new model)

    Returns:
        PPO: Trained PPO model
    """
    logger = logging.getLogger(__name__)
    logger.info("=== RL TRAINING STARTED ===")

    # Determine number of environments to use
    if n_envs is None:
        # Fall back to config if available
        n_envs = getattr(config, 'n_parallel_envs', SINGLE_ENV_THRESHOLD)

    # Create environment (vectorized or single)
    if use_parallel and n_envs > SINGLE_ENV_THRESHOLD:
        logger.info(f"Using parallel training with {n_envs} environments")
        env = create_vectorized_env(config, n_envs=n_envs, base_workspace=base_workspace)
    else:
        logger.info("Using single environment training")
        env = TrafficControlEnv(config)
        logger.info("Validating RL environment...")
        check_env(env)
        logger.info("RL environment validation passed!")

    # Configure PPO with traffic control-specific parameters
    # Based on recommendations from RL_DISCUSSION.md

    # Create TensorBoard log directory (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = os.path.join(TRAINING_TENSORBOARD_LOG_DIR, f"rl_training_{timestamp}")
        os.makedirs(tensorboard_log, exist_ok=True)
        logger.info(f"TensorBoard logging enabled: {tensorboard_log}")
    except ImportError:
        logger.info("TensorBoard not available - training will proceed without TensorBoard logging")

    # Create or load PPO model
    if resume_from_model and os.path.exists(resume_from_model):
        logger.info(f"Resuming training from existing model: {resume_from_model}")
        try:
            model = PPO.load(resume_from_model, env=env, tensorboard_log=tensorboard_log, device=TRAINING_DEVICE_AUTO)
            logger.info(f"Successfully loaded model from {resume_from_model}")
            logger.info(f"Continuing training for {total_timesteps} additional timesteps")
        except Exception as e:
            logger.error(f"Failed to load model from {resume_from_model}: {e}")
            logger.info("Creating new model instead...")
            resume_from_model = None

    if not resume_from_model:
        # Policy network configuration
        policy_kwargs = {
            'net_arch': TRAINING_NETWORK_ARCHITECTURE,
            'activation_fn': TRAINING_ACTIVATION_FUNCTION
        }

        model = PPO(
            TRAINING_POLICY_TYPE,
            env,
            learning_rate=DEFAULT_LEARNING_RATE,
            clip_range=DEFAULT_CLIP_RANGE,
            batch_size=DEFAULT_BATCH_SIZE,
            n_steps=DEFAULT_N_STEPS,
            n_epochs=DEFAULT_N_EPOCHS,
            gamma=DEFAULT_GAMMA,
            gae_lambda=DEFAULT_GAE_LAMBDA,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=TRAINING_VERBOSE_LEVEL,
            device=TRAINING_DEVICE_AUTO
        )

        logger.info(f"PPO model initialized with policy: {model.policy}")
        logger.info(f"Training for {total_timesteps} timesteps")

    # Set up callbacks
    callbacks = []

    # Traffic metrics callback for monitoring training progress
    traffic_callback = TrafficMetricsCallback(
        log_interval=TRAINING_PROGRESS_LOG_INTERVAL,
        verbose=TRAINING_VERBOSE_LEVEL
    )
    callbacks.append(traffic_callback)

    # Checkpoint callback to save model periodically
    if checkpoint_freq > DEFAULT_INITIAL_STEP:
        os.makedirs(DEFAULT_MODELS_DIRECTORY, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(DEFAULT_MODELS_DIRECTORY, TRAINING_CHECKPOINT_PREFIX),
            name_prefix="rl_traffic_model",
            verbose=TRAINING_VERBOSE_LEVEL
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback for best model selection
    if use_parallel and n_envs > SINGLE_ENV_THRESHOLD:
        # Create single environment for evaluation
        eval_env = TrafficControlEnv(config.get_cli_args_for_env(DEFAULT_INITIAL_STEP, base_workspace, max_envs=n_envs))
    else:
        eval_env = env

    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=TRAINING_EVAL_EPISODES_PER_CHECKPOINT,
        eval_freq=checkpoint_freq,
        best_model_save_path=os.path.join(DEFAULT_MODELS_DIRECTORY, TRAINING_BEST_MODEL_PREFIX),
        log_path=os.path.join(DEFAULT_MODELS_DIRECTORY, "eval_logs"),
        deterministic=True,
        verbose=TRAINING_VERBOSE_LEVEL
    )
    callbacks.append(eval_callback)

    logger.info(f"Created {len(callbacks)} training callbacks")
    logger.info(f"TensorBoard logs: {tensorboard_log}")

    # Execute training
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        training_time = time.time() - training_start_time
        logger.info("=== RL TRAINING COMPLETED ===")
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model with versioning
        final_model_path = _save_model_with_metadata(
            model, model_save_path, total_timesteps, training_time,
            config, tensorboard_log, logger
        )
        logger.info(f"Final model saved to: {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        env.close()

    return model


def evaluate_rl_policy(model_path: str,
                      config: Any,
                      n_episodes: int = DEFAULT_EVAL_EPISODES) -> Dict[str, float]:
    """
    Evaluate trained RL policy performance.

    Args:
        model_path: Path to trained model
        config: Environment configuration
        n_episodes: Number of evaluation episodes

    Returns:
        Dict: Evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== RL EVALUATION STARTED ===")

    # Load trained model
    model = PPO.load(model_path)
    logger.info(f"Loaded model from: {model_path}")

    # Create evaluation environment
    env = TrafficControlEnv(config)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(n_episodes):
            logger.info(f"Evaluation episode {episode + 1}/{n_episodes}")

            obs, _ = env.reset()
            episode_reward = DEFAULT_FALLBACK_VALUE
            episode_length = DEFAULT_INITIAL_STEP
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += SINGLE_ENV_THRESHOLD

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            logger.info(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Length: {episode_length}")

        # Calculate evaluation metrics
        metrics = {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'std_reward': _calculate_std(episode_rewards),
            'mean_length': sum(episode_lengths) / len(episode_lengths),
            'std_length': _calculate_std(episode_lengths)
        }

        logger.info("=== RL EVALUATION RESULTS ===")
        logger.info(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(f"Mean length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    finally:
        env.close()

    return metrics


def _calculate_std(values):
    """Calculate standard deviation of values."""
    if len(values) <= STD_CALCULATION_MIN_VALUES:
        return STD_CALCULATION_FALLBACK
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return variance ** VARIANCE_CALCULATION_POWER


def _save_model_with_metadata(model: PPO,
                             model_save_path: str,
                             total_timesteps: int,
                             training_time: float,
                             config: Any,
                             tensorboard_log: str,
                             logger: logging.Logger) -> str:
    """Save model with versioning and metadata.

    Args:
        model: Trained PPO model
        model_save_path: Base path for model saving
        total_timesteps: Total training timesteps
        training_time: Training duration in seconds
        config: Training configuration
        tensorboard_log: TensorBoard log directory
        logger: Logger instance

    Returns:
        str: Path to saved model
    """
    # Generate versioned model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(model_save_path))[0]
    versioned_name = f"{base_name}_{timestamp}"
    versioned_path = os.path.join(os.path.dirname(model_save_path), versioned_name)

    # Save model
    model.save(versioned_path)

    # Create and save metadata
    metadata = {
        'model_name': versioned_name,
        'training_timesteps': total_timesteps,
        'training_time_seconds': training_time,
        'training_date': datetime.now().isoformat(),
        'network_architecture': TRAINING_NETWORK_ARCHITECTURE,
        'tensorboard_log': tensorboard_log,
        'hyperparameters': {
            'learning_rate': float(model.learning_rate) if hasattr(model.learning_rate, '__float__') else model.learning_rate(1.0),
            'clip_range': float(model.clip_range) if hasattr(model.clip_range, '__float__') else model.clip_range(1.0),
            'batch_size': model.batch_size,
            'n_steps': model.n_steps,
            'n_epochs': model.n_epochs,
            'gamma': model.gamma,
            'gae_lambda': model.gae_lambda,
            'device': TRAINING_DEVICE_AUTO
        },
        'environment_info': {
            'grid_dimension': getattr(config, 'grid_dimension', 'unknown'),
            'num_vehicles': getattr(config, 'num_vehicles', 'unknown'),
            'end_time': getattr(config, 'end_time', 'unknown')
        }
    }

    # Save metadata file
    metadata_path = versioned_path + TRAINING_MODEL_METADATA_EXTENSION
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to: {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to save metadata: {e}")

    return versioned_path


if __name__ == "__main__":
    # Example training script usage
    logging.basicConfig(level=logging.INFO)

    # Example script - Phase 2: Load configuration and execute training
    from .config import get_rl_config

    config = get_rl_config()
    print("RL training script ready.")
    print("Example usage:")
    print("  from src.rl.training import train_rl_policy")
    print("  from src.rl.config import get_rl_config")
    print("  config = get_rl_config()")
    print("  model = train_rl_policy(config)")