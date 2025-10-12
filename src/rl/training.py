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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .environment import TrafficControlEnv
from .constants import (
    DEFAULT_LEARNING_RATE, DEFAULT_CLIP_RANGE, DEFAULT_BATCH_SIZE,
    DEFAULT_N_STEPS, DEFAULT_N_EPOCHS, DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA, MAX_GRAD_NORM,
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_CHECKPOINT_FREQ, DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_EVAL_EPISODES, DEFAULT_MODELS_DIRECTORY, SINGLE_ENV_THRESHOLD,
    DEFAULT_INITIAL_STEP, DEFAULT_INITIAL_TIME, DEFAULT_FALLBACK_VALUE, PARALLEL_WORKSPACE_PREFIX,
    STD_CALCULATION_MIN_VALUES, STD_CALCULATION_FALLBACK,
    TRAINING_NETWORK_ARCHITECTURE, TRAINING_DEVICE_AUTO, TRAINING_TENSORBOARD_LOG_DIR,
    TRAINING_BEST_MODEL_PREFIX, TRAINING_CHECKPOINT_PREFIX, TRAINING_MODEL_METADATA_EXTENSION,
    TRAINING_EVAL_EPISODES_PER_CHECKPOINT, TRAINING_PATIENCE_EPISODES,
    TRAINING_MIN_IMPROVEMENT_THRESHOLD, TRAINING_PROGRESS_LOG_INTERVAL,
    TRAINING_POLICY_TYPE, TRAINING_ACTIVATION_FUNCTION, TRAINING_VERBOSE_LEVEL,
    VARIANCE_CALCULATION_POWER,
    LEARNING_RATE_SCHEDULE_ENABLED, LEARNING_RATE_INITIAL, LEARNING_RATE_FINAL, LEARNING_RATE_DECAY_RATE,
    ENTROPY_COEF_SCHEDULE_ENABLED, ENTROPY_COEF_INITIAL, ENTROPY_COEF_FINAL, ENTROPY_COEF_DECAY_STEPS,
    EARLY_STOPPING_ENABLED, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_VERBOSE
)
# Config module removed - parameters now passed via --env-params


class CumulativeCheckpointCallback(CheckpointCallback):
    """Custom checkpoint callback that tracks cumulative timesteps across training sessions."""

    def __init__(self, save_freq, save_path, name_prefix="rl_model", initial_timesteps=0, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.initial_timesteps = initial_timesteps

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Override to use cumulative timestep count in checkpoint names.

        Args:
            checkpoint_type: not used (kept for compatibility)
            extension: file extension
        """
        cumulative_steps = self.num_timesteps + self.initial_timesteps
        return os.path.join(self.save_path, f"{self.name_prefix}_{cumulative_steps}_steps")


class PersistentBestEvalCallback(EvalCallback):
    """
    Enhanced EvalCallback that persists best model tracking across training sessions.

    Standard EvalCallback resets best_mean_reward to -inf on each training session,
    causing worse models to overwrite better ones when resuming training. This class
    fixes that by loading historical evaluation data and initializing the threshold
    from the true historical best.
    """

    def __init__(self, eval_env, callback_on_new_best=None, callback_after_eval=None,
                 n_eval_episodes=5, eval_freq=10000, log_path=None,
                 best_model_save_path=None, deterministic=True, render=False,
                 verbose=1, warn=True, initial_timesteps=0):
        """
        Initialize callback with historical best tracking and evaluation log continuation.

        Args:
            eval_env: Environment for evaluation
            callback_on_new_best: Optional callback when new best is found
            callback_after_eval: Optional callback after each evaluation
            n_eval_episodes: Number of episodes per evaluation
            eval_freq: Evaluate every N steps
            log_path: Path to save evaluation logs
            best_model_save_path: Path to save best model
            deterministic: Use deterministic actions for evaluation
            render: Render environment during evaluation
            verbose: Verbosity level
            warn: Show warnings
            initial_timesteps: Cumulative timesteps from previous training sessions (for checkpoint resume)
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )

        # Store initial timesteps offset for cumulative evaluation timesteps
        self.initial_timesteps = initial_timesteps

        # Load historical evaluations and best performance
        # This must happen AFTER super().__init__() so parent class initializes instance variables
        self._load_and_inject_historical_evaluations()

        # Path for best model metadata
        self.metadata_path = None
        if best_model_save_path:
            self.metadata_path = os.path.join(
                best_model_save_path, "best_model_metadata.json")

    def _load_and_inject_historical_evaluations(self):
        """Load historical evaluation data and inject into parent class instance variables."""
        if not self.log_path:
            return

        # EvalCallback adds /evaluations to log_path, so parent directory contains the npz file
        parent_dir = os.path.dirname(self.log_path)
        eval_log_file = os.path.join(parent_dir, "evaluations.npz")

        if not os.path.exists(eval_log_file):
            if self.verbose > 0:
                print(f"No historical evaluations found at {eval_log_file}")
                print(f"Starting fresh evaluation log")
            return

        try:
            import numpy as np
            data = np.load(eval_log_file)

            if 'results' in data and len(data['results']) > 0:
                # Inject historical data directly into parent class instance variables
                # These are the lists that get saved by np.savez in parent class
                self.evaluations_timesteps = list(data['timesteps'])
                self.evaluations_results = list(data['results'])
                self.evaluations_length = list(
                    data['ep_lengths']) if 'ep_lengths' in data else []

                # Calculate mean reward for each evaluation
                mean_rewards = [np.mean(result) for result in data['results']]
                historical_best = np.max(mean_rewards)
                best_idx = np.argmax(mean_rewards)
                best_timestep = data['timesteps'][best_idx]

                # Initialize best_mean_reward from historical data instead of -inf
                self.best_mean_reward = historical_best

                if self.verbose > 0:
                    print(f"=" * 60)
                    print(f"Loaded historical evaluation data:")
                    print(
                        f"  Total evaluations: {len(self.evaluations_timesteps)}")
                    print(
                        f"  Last evaluation at timestep: {self.evaluations_timesteps[-1]}")
                    print(f"  Best mean reward: {historical_best:.2f}")
                    print(f"  Best from timestep: {best_timestep}")
                    print(f"  New evaluations will be appended to this history")
                    print(f"=" * 60)

        except Exception as e:
            if self.verbose > 0:
                print(
                    f"Warning: Could not load historical evaluations from {eval_log_file}: {e}")
                print(f"Starting with default best_mean_reward = -inf")
                import traceback
                traceback.print_exc()

    def _on_step(self) -> bool:
        """
        Override to add cumulative timestep tracking, metadata, and backup functionality.

        The parent EvalCallback records self.num_timesteps when saving evaluations.
        We need to add initial_timesteps offset to make timesteps cumulative across
        training sessions (e.g., [10000, 15000, 20000] instead of [10000, 5000, 5000]).
        """
        # Temporarily add initial_timesteps offset to num_timesteps for evaluation recording
        original_num_timesteps = self.num_timesteps
        self.num_timesteps = self.num_timesteps + self.initial_timesteps

        # Call parent _on_step which will record evaluations with adjusted num_timesteps
        result = super()._on_step()

        # Restore original num_timesteps (critical for PPO training to continue correctly)
        self.num_timesteps = original_num_timesteps

        # If we just saved a new best model, save metadata
        # Only save metadata when there's actual improvement (strict inequality)
        # Parent EvalCallback only saves model when mean_reward > best_mean_reward
        if (hasattr(self, 'last_mean_reward') and
                self.last_mean_reward > self.best_mean_reward):
            if self.metadata_path and self.best_model_save_path:
                self._save_best_model_metadata()

        return result

    def _save_best_model_metadata(self):
        """Save metadata about the current best model."""
        try:
            from datetime import datetime

            metadata = {
                'best_mean_reward': float(self.best_mean_reward),
                'timestep': int(self.num_timesteps),
                'timestamp': datetime.now().isoformat(),
                'n_eval_episodes': self.n_eval_episodes,
                'model_path': os.path.join(self.best_model_save_path, 'best_model.zip')
            }

            # Back up previous metadata if it exists
            if os.path.exists(self.metadata_path):
                backup_path = self.metadata_path.replace(
                    '.json', '_previous.json')
                import shutil
                shutil.copy2(self.metadata_path, backup_path)

            # Save new metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"✓ NEW BEST MODEL FOUND AND SAVED")
                print(f"  Reward: {metadata['best_mean_reward']:.2f}")
                print(f"  Timestep: {metadata['timestep']}")
                print(f"  Saved to: {metadata['model_path']}")
                print(f"  Metadata: {self.metadata_path}")
                print(f"{'='*60}\n")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save best model metadata: {e}")


def get_learning_rate_schedule(initial_lr: float, final_lr: float, decay_rate: float):
    """
    Create exponential learning rate schedule function.

    Args:
        initial_lr: Starting learning rate
        final_lr: Minimum learning rate
        decay_rate: Exponential decay rate per step

    Returns:
        callable: Learning rate schedule function
    """
    def lr_schedule(progress_remaining: float) -> float:
        """
        Learning rate schedule based on remaining progress.

        Args:
            progress_remaining: 1.0 at start, 0.0 at end of training

        Returns:
            float: Current learning rate
        """
        # Exponential decay: lr = initial_lr * decay_rate^steps
        # We use (1 - progress_remaining) to get progress from 0 to 1
        progress = 1.0 - progress_remaining
        lr = initial_lr * (decay_rate ** progress)
        # Clamp to minimum
        return max(lr, final_lr)

    return lr_schedule


def get_entropy_coef_schedule(initial_coef: float, final_coef: float, decay_steps: int):
    """
    Create linear entropy coefficient schedule function.

    Args:
        initial_coef: Starting entropy coefficient (high exploration)
        final_coef: Final entropy coefficient (low exploration)
        decay_steps: Number of steps to decay over

    Returns:
        callable: Entropy coefficient schedule function
    """
    def entropy_schedule(progress_remaining: float) -> float:
        """
        Entropy coefficient schedule based on remaining progress.

        Args:
            progress_remaining: 1.0 at start, 0.0 at end of training

        Returns:
            float: Current entropy coefficient
        """
        # Linear decay from initial to final
        progress = 1.0 - progress_remaining
        coef = initial_coef - (initial_coef - final_coef) * progress
        # Clamp to minimum
        return max(coef, final_coef)

    return entropy_schedule


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
            mean_reward = np.mean(
                self.episode_rewards[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:])
            mean_length = np.mean(
                self.episode_lengths[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:])

            self.logger.record('traffic/mean_episode_reward', mean_reward)
            self.logger.record('traffic/mean_episode_length', mean_length)
            self.logger.record('traffic/total_episodes',
                               len(self.episode_rewards))

    def _log_traffic_metrics(self):
        """Log detailed traffic metrics."""
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:]
            recent_lengths = self.episode_lengths[-TRAINING_EVAL_EPISODES_PER_CHECKPOINT:]

            if recent_rewards:
                self.logger.record(
                    'traffic/recent_mean_reward', np.mean(recent_rewards))
                self.logger.record(
                    'traffic/recent_std_reward', np.std(recent_rewards))
                self.logger.record(
                    'traffic/recent_min_reward', np.min(recent_rewards))
                self.logger.record(
                    'traffic/recent_max_reward', np.max(recent_rewards))

            if recent_lengths:
                self.logger.record(
                    'traffic/recent_mean_length', np.mean(recent_lengths))
                self.logger.record(
                    'traffic/recent_std_length', np.std(recent_lengths))


def make_env(env_params_string: str, env_index: int, base_workspace: str = PARALLEL_WORKSPACE_PREFIX):
    """
    Create a single environment for vectorization.

    Args:
        env_params_string: Raw parameter string for environment
        env_index: Environment index for unique workspace
        base_workspace: Base workspace directory name

    Returns:
        callable: Environment creation function
    """
    def _init():
        # Create unique workspace path for this environment
        env_workspace = f"{base_workspace}/env_{env_index:03d}"

        # Append workspace to parameter string
        env_params_with_workspace = f"{env_params_string} --workspace {env_workspace}"

        # Create environment with unique workspace
        env = TrafficControlEnv(env_params_with_workspace)
        return env

    return _init


def create_vectorized_env(env_params_string: str, n_envs: int = None, base_workspace: str = PARALLEL_WORKSPACE_PREFIX):
    """
    Create vectorized environment for parallel training.

    Args:
        env_params_string: Raw parameter string for environment creation
        n_envs: Number of parallel environments
        base_workspace: Base workspace directory name

    Returns:
        VecEnv: Vectorized environment for parallel training
    """
    logger = logging.getLogger(__name__)

    if n_envs is None:
        n_envs = DEFAULT_N_PARALLEL_ENVS

    # Create list of environment creation functions
    env_fns = [make_env(env_params_string, i, base_workspace)
               for i in range(n_envs)]

    # Use SubprocVecEnv for true parallel execution, DummyVecEnv for single environment
    if n_envs == SINGLE_ENV_THRESHOLD:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    return vec_env


def train_rl_policy(env_params_string: str,
                    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
                    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
                    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ,
                    base_workspace: str = PARALLEL_WORKSPACE_PREFIX,
                    use_parallel: bool = True,
                    n_envs: int = None,
                    resume_from_model: str = None,
                    pretrain_from_model: str = None) -> PPO:
    """
    Train PPO policy for traffic signal control.

    Args:
        env_params_string: Raw parameter string for environment creation (e.g., "--network-seed 42 --grid_dimension 5 ...")
        total_timesteps: Total training timesteps
        model_save_path: Path to save trained model (DEPRECATED - now auto-generated in model directory)
        checkpoint_freq: Frequency of model checkpointing
        base_workspace: Base workspace directory for parallel environments (DEPRECATED - now auto-generated in model directory)
        use_parallel: Whether to use parallel environments (True) or single environment (False)
        n_envs: Number of parallel environments (if None, defaults to 1)
        resume_from_model: Path to existing model to resume training from (if None, creates new model)
        pretrain_from_model: Path to pre-trained model from imitation learning to initialize from (if None, creates new model from scratch)

    Returns:
        PPO: Trained PPO model
    """
    logger = logging.getLogger(__name__)
    logger.info("=== RL TRAINING STARTED ===")

    # Determine model directory: either extract from resume path or create new unique directory
    model_dir = None
    initial_timesteps = 0

    if resume_from_model and os.path.exists(resume_from_model):
        # Extract model directory from checkpoint path
        # Expected format: models/rl_YYYYMMDD_HHMMSS/checkpoint/rl_traffic_model_XXXX_steps.zip
        import re
        match = re.search(r'models/(rl_\d{8}_\d{6})', resume_from_model)
        if match:
            model_dir = os.path.join(DEFAULT_MODELS_DIRECTORY, match.group(1))
            logger.info(
                f"Resuming training - using existing model directory: {model_dir}")
        else:
            logger.warning(
                f"Could not extract model directory from resume path: {resume_from_model}")
            logger.warning(f"Creating new model directory instead")

    if model_dir is None:
        # Create new unique model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(DEFAULT_MODELS_DIRECTORY, f"rl_{timestamp}")
        logger.info(
            f"Starting new training - created model directory: {model_dir}")

    # Create model directory structure
    workspace_dir = os.path.join(model_dir, "workspace")
    eval_logs_dir = os.path.join(model_dir, "eval_logs")
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    best_model_dir = os.path.join(model_dir, "best_model")

    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(eval_logs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    logger.info(f"Model directory structure:")
    logger.info(f"  Workspace:    {workspace_dir}")
    logger.info(f"  Eval logs:    {eval_logs_dir}")
    logger.info(f"  Checkpoints:  {checkpoint_dir}")
    logger.info(f"  Best model:   {best_model_dir}")

    # Update base_workspace to model directory
    # The config's update_workspace() will create model_dir/workspace/ automatically
    base_workspace = model_dir

    # Determine number of environments to use
    if n_envs is None:
        n_envs = SINGLE_ENV_THRESHOLD

    # Create environment (vectorized or single)
    if use_parallel and n_envs > SINGLE_ENV_THRESHOLD:
        env = create_vectorized_env(
            env_params_string, n_envs=n_envs, base_workspace=base_workspace)
        _verify_enhanced_state_space(
            env, env_params_string, is_vectorized=True)
    else:
        # For single environment, add workspace to params
        single_env_params = f"{env_params_string} --workspace {base_workspace}"
        env = TrafficControlEnv(single_env_params)
        check_env(env)
        _verify_enhanced_state_space(env, env_params_string)

    # Configure PPO with traffic control-specific parameters
    # Based on recommendations from RL_DISCUSSION.md

    # TensorBoard logging disabled (no logs directory)
    tensorboard_log = None

    # Create or load PPO model
    if resume_from_model and os.path.exists(resume_from_model):
        logger.info(
            f"Resuming training from existing model: {resume_from_model}")
        try:
            model = PPO.load(resume_from_model, env=env,
                             tensorboard_log=tensorboard_log, device=TRAINING_DEVICE_AUTO)
            logger.info(f"Successfully loaded model from {resume_from_model}")
            logger.info(
                f"Continuing training for {total_timesteps} additional timesteps")

            # Extract step count from checkpoint filename if available
            import re
            match = re.search(r'(\d+)_steps\.zip', resume_from_model)
            if match:
                initial_timesteps = int(match.group(1))
                logger.info(
                    f"Detected {initial_timesteps} initial steps from checkpoint filename")
            else:
                logger.warning(
                    f"Could not extract step count from filename: {resume_from_model}")
                logger.warning(
                    f"Checkpoint naming will restart from 0 (actual training continues normally)")

        except Exception as e:
            logger.error(f"Failed to load model from {resume_from_model}: {e}")
            logger.info("Creating new model instead...")
            resume_from_model = None

    # Load pre-trained model from imitation learning (mutually exclusive with resume)
    if not resume_from_model and pretrain_from_model and os.path.exists(pretrain_from_model):
        logger.info("=== LOADING PRE-TRAINED MODEL FROM IMITATION LEARNING ===")
        logger.info(f"Pre-trained model path: {pretrain_from_model}")
        try:
            model = PPO.load(pretrain_from_model, env=env,
                             tensorboard_log=tensorboard_log, device=TRAINING_DEVICE_AUTO)
            logger.info(
                f"✓ Successfully loaded pre-trained model from {pretrain_from_model}")
            logger.info(
                f"Starting RL fine-tuning for {total_timesteps} timesteps")
            logger.info(
                "Policy weights initialized from Tree Method imitation learning")
        except Exception as e:
            logger.error(
                f"Failed to load pre-trained model from {pretrain_from_model}: {e}")
            logger.info("Creating new model from scratch instead...")
            pretrain_from_model = None

    if not resume_from_model and not pretrain_from_model:
        # Policy network configuration
        policy_kwargs = {
            'net_arch': TRAINING_NETWORK_ARCHITECTURE,
            'activation_fn': TRAINING_ACTIVATION_FUNCTION
        }

        # Configure learning rate schedule
        if LEARNING_RATE_SCHEDULE_ENABLED:
            learning_rate = get_learning_rate_schedule(
                LEARNING_RATE_INITIAL,
                LEARNING_RATE_FINAL,
                LEARNING_RATE_DECAY_RATE
            )
        else:
            learning_rate = DEFAULT_LEARNING_RATE

        # Configure entropy coefficient
        if ENTROPY_COEF_SCHEDULE_ENABLED:
            ent_coef = ENTROPY_COEF_INITIAL
        else:
            ent_coef = 0.01  # Small default for exploration

        model = PPO(
            TRAINING_POLICY_TYPE,
            env,
            learning_rate=learning_rate,
            clip_range=DEFAULT_CLIP_RANGE,
            batch_size=DEFAULT_BATCH_SIZE,
            n_steps=DEFAULT_N_STEPS,
            n_epochs=DEFAULT_N_EPOCHS,
            gamma=DEFAULT_GAMMA,
            gae_lambda=DEFAULT_GAE_LAMBDA,
            ent_coef=ent_coef,
            max_grad_norm=MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=TRAINING_VERBOSE_LEVEL,
            device=TRAINING_DEVICE_AUTO
        )

        logger.info(f"PPO model initialized with policy: {model.policy}")
        logger.info(f"Training for {total_timesteps} timesteps")
        logger.info(f"Hyperparameters: gamma={DEFAULT_GAMMA}, clip={DEFAULT_CLIP_RANGE}, "
                    f"batch={DEFAULT_BATCH_SIZE}, steps={DEFAULT_N_STEPS}, "
                    f"epochs={DEFAULT_N_EPOCHS}, grad_clip={MAX_GRAD_NORM}")

    # Set up callbacks
    callbacks = []

    # Traffic metrics callback for monitoring training progress
    traffic_callback = TrafficMetricsCallback(
        log_interval=TRAINING_PROGRESS_LOG_INTERVAL,
        verbose=TRAINING_VERBOSE_LEVEL
    )
    callbacks.append(traffic_callback)

    # Checkpoint callback to save model periodically with cumulative step tracking
    if checkpoint_freq > DEFAULT_INITIAL_STEP:
        checkpoint_callback = CumulativeCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="rl_traffic_model",
            initial_timesteps=initial_timesteps,
            verbose=TRAINING_VERBOSE_LEVEL
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            f"Checkpoint callback configured: save_path={checkpoint_dir}, initial_timesteps={initial_timesteps}")

    # Evaluation callback for best model selection
    if use_parallel and n_envs > SINGLE_ENV_THRESHOLD:
        # Create single environment for evaluation
        eval_env = TrafficControlEnv(config.get_cli_args_for_env(
            DEFAULT_INITIAL_STEP, base_workspace, max_envs=n_envs))
    else:
        eval_env = env

    # Configure early stopping if enabled
    stop_callback = None
    if EARLY_STOPPING_ENABLED:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=EARLY_STOPPING_PATIENCE,
            min_evals=5,  # Minimum evaluations before stopping can occur
            verbose=1 if EARLY_STOPPING_VERBOSE else 0
        )
        logger.info(f"Early stopping enabled: patience={EARLY_STOPPING_PATIENCE} evaluations, "
                    f"min_delta={EARLY_STOPPING_MIN_DELTA}")

    # Use PersistentBestEvalCallback to preserve best model across training sessions
    eval_callback = PersistentBestEvalCallback(
        eval_env=eval_env,
        callback_on_new_best=stop_callback,  # Early stopping callback
        n_eval_episodes=TRAINING_EVAL_EPISODES_PER_CHECKPOINT,
        eval_freq=checkpoint_freq,
        best_model_save_path=best_model_dir,
        log_path=eval_logs_dir,
        deterministic=True,
        verbose=TRAINING_VERBOSE_LEVEL,
        # Pass cumulative timesteps for evaluation log continuation
        initial_timesteps=initial_timesteps
    )
    callbacks.append(eval_callback)
    logger.info(
        f"Evaluation callback configured: best_model={best_model_dir}, eval_logs={eval_logs_dir}, initial_timesteps={initial_timesteps}")

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

        # Save final model with versioning in model directory
        final_model_save_path = os.path.join(model_dir, "final_model")
        final_model_path = _save_model_with_metadata(
            model, final_model_save_path, total_timesteps, training_time,
            tensorboard_log, logger
        )
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Model directory: {model_dir}")

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

    # Create evaluation environment first for compatibility checking
    env = TrafficControlEnv(config)
    logger.info(f"Evaluation environment created")

    # Load trained model
    model = PPO.load(model_path)
    logger.info(f"Loaded model from: {model_path}")

    # Validate model compatibility with current environment
    _validate_model_compatibility(model, env, config, logger)

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
            logger.info(
                f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Length: {episode_length}")

        # Calculate evaluation metrics
        metrics = {
            'mean_reward': sum(episode_rewards) / len(episode_rewards),
            'std_reward': _calculate_std(episode_rewards),
            'mean_length': sum(episode_lengths) / len(episode_lengths),
            'std_length': _calculate_std(episode_lengths)
        }

        logger.info("=== RL EVALUATION RESULTS ===")
        logger.info(
            f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(
            f"Mean length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

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


def _validate_model_compatibility(model, env, config, logger):
    """
    Validate that a loaded model is compatible with the current environment configuration.

    Args:
        model: Loaded PPO model
        env: Environment instance
        config: RL training configuration
        logger: Logger instance

    Raises:
        ValueError: If model is incompatible with current environment
    """
    logger.info("=== MODEL COMPATIBILITY VALIDATION ===")

    try:
        # Get sample observation and action to test dimensions
        obs, _ = env.reset()
        current_state_size = len(obs)
        current_action_size = env.action_space.nvec if hasattr(
            env.action_space, 'nvec') else env.action_space.n

        # Test model prediction to check input/output compatibility
        try:
            action, _ = model.predict(obs, deterministic=True)
            predicted_action_size = len(action) if hasattr(
                action, '__len__') else 1
        except Exception as e:
            raise ValueError(f"Model cannot process current state vector: {e}")

        # Validate state space compatibility
        expected_state_size = config.state_vector_size_estimate
        if current_state_size != expected_state_size:
            logger.warning(
                f"State size mismatch: environment={current_state_size}, config={expected_state_size}")

        # Validate action space compatibility
        expected_action_size = config.action_vector_size
        if isinstance(current_action_size, (list, tuple)):
            total_actions = sum(current_action_size)
        else:
            total_actions = current_action_size

        if predicted_action_size != total_actions:
            raise ValueError(f"Action space incompatibility: model outputs {predicted_action_size} actions, "
                             f"environment expects {total_actions}")

        # Check for phase-only vs legacy model
        from .constants import RL_PHASE_ONLY_MODE, RL_TRAINING_ACTION_VECTOR_SIZE, RL_TRAINING_ACTION_VECTOR_SIZE_LEGACY
        if RL_PHASE_ONLY_MODE:
            if predicted_action_size == RL_TRAINING_ACTION_VECTOR_SIZE_LEGACY:
                logger.warning("⚠️  Loading legacy phase+duration model in phase-only mode - "
                               "this may cause compatibility issues")
            elif predicted_action_size == RL_TRAINING_ACTION_VECTOR_SIZE:
                logger.info(
                    "✓ Phase-only model detected - compatible with current configuration")
        else:
            if predicted_action_size == RL_TRAINING_ACTION_VECTOR_SIZE:
                logger.warning("⚠️  Loading phase-only model in legacy mode - "
                               "this may cause compatibility issues")

        # Test a full step to ensure complete compatibility
        action, _ = model.predict(obs, deterministic=True)
        obs_new, reward, terminated, truncated, info = env.step(action)

        logger.info("✓ Model compatibility validation passed")
        logger.info(f"  State space: {current_state_size} features")
        logger.info(f"  Action space: {predicted_action_size} actions")
        logger.info(
            f"  Model prediction successful, environment step successful")

    except Exception as e:
        logger.error(f"Model compatibility validation failed: {e}")
        raise ValueError(f"Model incompatible with current environment: {e}")


def _verify_enhanced_state_space(env, env_params_string, is_vectorized=False):
    """
    Verify that the enhanced state space is properly implemented and functional.

    Args:
        env: RL environment (single or vectorized)
        env_params_string: Parameter string (not used, kept for compatibility)
        is_vectorized: Whether the environment is vectorized
    """
    logger = logging.getLogger(__name__)
    logger.info("=== ENHANCED STATE SPACE VERIFICATION ===")

    try:
        if is_vectorized:
            # For vectorized environments, test one environment
            logger.info(
                "Testing enhanced state space on vectorized environment...")

            # Reset and get initial observation from first environment
            observations = env.reset()
            # observations is a tuple (obs_array, info_dict) for vectorized envs
            if isinstance(observations, tuple):
                obs_array = observations[0]
                sample_obs = obs_array[0]  # First environment's observation
            else:
                sample_obs = observations[0]

        else:
            # For single environment
            logger.info(
                "Testing enhanced state space on single environment...")
            sample_obs, _ = env.reset()

        # Verify observation dimensions
        actual_state_size = len(sample_obs)

        logger.info(f"State vector size: {actual_state_size}")

        # Verify the observation contains meaningful data
        non_zero_count = np.count_nonzero(sample_obs)
        zero_count = actual_state_size - non_zero_count
        non_zero_ratio = non_zero_count / actual_state_size

        logger.info(f"State vector composition:")
        logger.info(
            f"  - Non-zero features: {non_zero_count}/{actual_state_size} ({non_zero_ratio:.1%})")
        logger.info(f"  - Zero features: {zero_count}/{actual_state_size}")
        logger.info(
            f"  - Value range: [{np.min(sample_obs):.3f}, {np.max(sample_obs):.3f}]")

        # Verify enhanced features are present (check for Tree Method integration)
        if hasattr(env, 'traffic_analyzer') or (is_vectorized and hasattr(env.envs[0], 'traffic_analyzer')):
            logger.info("✓ Tree Method traffic analyzer detected")

            # Enable debug mode for detailed verification
            if is_vectorized:
                test_env = env.envs[0]
            else:
                test_env = env

            if hasattr(test_env, 'enable_debug_mode'):
                logger.info(
                    "Enabling debug mode for detailed state inspection...")
                # test_env.enable_debug_mode()

                # Get another observation with debug enabled
                if is_vectorized:
                    env.reset()
                else:
                    test_env.reset()

                logger.info(
                    "Debug mode enabled - detailed state logging should appear above")
        else:
            logger.warning(
                "⚠️  Tree Method traffic analyzer not found - may not be properly initialized")

        # Log state vector size (no expected size comparison since we removed config)
        logger.info(f"State vector size: {actual_state_size} features")

        # Check for proper normalization [0, 1]
        if np.all(sample_obs >= 0.0) and np.all(sample_obs <= 1.0):
            logger.info(
                "✓ All state values properly normalized to [0, 1] range")
        else:
            out_of_range = np.sum((sample_obs < 0.0) | (sample_obs > 1.0))
            logger.warning(f"⚠️  {out_of_range} features outside [0, 1] range")

        # Verify non-trivial state (not all zeros)
        if non_zero_ratio > 0.1:  # At least 10% non-zero
            logger.info("✓ State vector contains meaningful traffic data")
        else:
            logger.warning(
                f"⚠️  Very few non-zero features ({non_zero_ratio:.1%}) - check simulation activity")

        logger.info("=== STATE SPACE VERIFICATION COMPLETE ===")

    except Exception as e:
        logger.error(f"Enhanced state space verification failed: {e}")
        raise


def _save_model_with_metadata(model: PPO,
                              model_save_path: str,
                              total_timesteps: int,
                              training_time: float,
                              tensorboard_log: str,
                              logger: logging.Logger) -> str:
    """Save model with versioning and metadata.

    Args:
        model: Trained PPO model
        model_save_path: Base path for model saving
        total_timesteps: Total training timesteps
        training_time: Training duration in seconds
        tensorboard_log: TensorBoard log directory
        logger: Logger instance

    Returns:
        str: Path to saved model
    """
    # Generate versioned model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(model_save_path))[0]
    versioned_name = f"{base_name}_{timestamp}"
    versioned_path = os.path.join(
        os.path.dirname(model_save_path), versioned_name)

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
            'ent_coef': model.ent_coef,
            'max_grad_norm': MAX_GRAD_NORM,
            'device': TRAINING_DEVICE_AUTO
        },
        'environment_info': {
            'note': 'Environment parameters provided via --env-params at runtime'
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

    # Example script - Execute training with parameter string
    print("RL training script ready.")
    print("Example usage:")
    print("  from src.rl.training import train_rl_policy")
    print("  env_params = '--network-seed 42 --grid_dimension 3 --num_vehicles 150 --end-time 3600'")
    print("  model = train_rl_policy(env_params)")
