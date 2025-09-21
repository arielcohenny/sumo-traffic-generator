"""
RLController for traffic signal control using reinforcement learning.

This module implements the RL-based traffic controller that integrates with
the existing traffic control architecture.
"""

import os
import time
import logging
from typing import Any, Optional, Dict, List
import numpy as np
import traci

from src.orchestration.traffic_controller import TrafficController
from .constants import (
    RL_CONTROLLER_NAME, RL_CONTROLLER_DISPLAY_NAME, RL_CONTROLLER_DESCRIPTION,
    DEFAULT_RL_MODEL_PATH, RL_MODEL_EXTENSION, RL_MODEL_VALIDATION_TIMEOUT,
    RL_MODEL_LOAD_RETRIES, RL_ACTION_EXECUTION_TIMEOUT, RL_SAFETY_CHECK_ENABLED,
    RL_MIN_GREEN_TIME_ENFORCEMENT, RL_MAX_DURATION_ENFORCEMENT,
    RL_INFERENCE_TIME_TRACKING, RL_ACTION_DISTRIBUTION_TRACKING,
    RL_STATISTICS_COLLECTION_INTERVAL, RL_PERFORMANCE_LOGGING_ENABLED,
    RL_TRAINING_MODE, RL_INFERENCE_MODE, RL_DEFAULT_MODE,
    RL_CONTROLLER_ERROR_PREFIX, RL_MODEL_COMPATIBILITY_CHECK,
    RL_GRACEFUL_DEGRADATION, PHASE_DURATION_OPTIONS, MIN_GREEN_TIME, MAX_PHASE_DURATION,
    DEFAULT_FALLBACK_VALUE, DEFAULT_INITIAL_TIME, NUM_TRAFFIC_LIGHT_PHASES,
    RL_MODEL_LOAD_RETRY_DELAY, TRAFFIC_LIGHT_DEFINITION_INDEX,
    RL_INFERENCE_TIME_MAX_HISTORY, RL_INFERENCE_TIME_KEEP_RECENT,
    RL_ACTION_DISTRIBUTION_LOG_INTERVAL
)
from .environment import TrafficControlEnv
from .config import get_rl_config


class RLController(TrafficController):
    """Traffic controller using reinforcement learning."""

    def __init__(self, args):
        """Initialize RL controller.

        Args:
            args: Command line arguments containing RL configuration
        """
        super().__init__(args)

        # Controller identification
        self.controller_name = RL_CONTROLLER_NAME
        self.display_name = RL_CONTROLLER_DISPLAY_NAME

        # Model management
        self.model = None
        self.model_path = getattr(args, 'rl_model_path', DEFAULT_RL_MODEL_PATH)
        self.mode = RL_INFERENCE_MODE if self.model_path else RL_DEFAULT_MODE

        # RL Environment
        self.rl_env = None
        self.current_observation = None

        # Performance tracking
        self.graph = None
        self.inference_times = []
        self.action_counts = {}
        self.statistics_step_counter = DEFAULT_INITIAL_TIME

        # Traffic signal state
        self.traffic_lights = {}
        self.last_action_time = DEFAULT_INITIAL_TIME

        self.logger.info(f"=== RL CONTROLLER INITIALIZATION ===")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Model path: {self.model_path}")

    def initialize(self) -> None:
        """Initialize RL controller with model loading and environment setup."""
        try:
            self.logger.info(f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} INITIALIZATION ===")

            # Initialize Graph object for vehicle tracking (same as other controllers)
            from src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.graph import Graph
            self.graph = Graph(self.args.end_time)

            # Load RL model if in inference mode
            if self.mode == RL_INFERENCE_MODE:
                self._load_rl_model()

            # Initialize RL environment for state collection
            self._initialize_rl_environment()

            # Initialize traffic light tracking
            self._initialize_traffic_lights()

            # Initialize action distribution tracking
            if RL_ACTION_DISTRIBUTION_TRACKING:
                self._initialize_action_tracking()

            self.logger.info(f"RL Controller initialization completed successfully")

        except Exception as e:
            self.logger.error(f"{RL_CONTROLLER_ERROR_PREFIX} initialization failed: {e}")
            if not RL_GRACEFUL_DEGRADATION:
                raise

    def _load_rl_model(self) -> None:
        """Load trained RL model for inference."""
        if not self.model_path:
            self.logger.info("No model path provided - running in training mode")
            return

        try:
            from stable_baselines3 import PPO

            # Validate model path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"RL model not found: {self.model_path}")

            # Load model with retries
            for attempt in range(RL_MODEL_LOAD_RETRIES):
                try:
                    self.logger.info(f"Loading RL model from: {self.model_path} (attempt {attempt + 1})")
                    self.model = PPO.load(self.model_path)

                    # Validate model compatibility if enabled
                    if RL_MODEL_COMPATIBILITY_CHECK:
                        self._validate_model_compatibility()

                    self.logger.info("RL model loaded successfully")
                    return

                except Exception as e:
                    self.logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
                    if attempt == RL_MODEL_LOAD_RETRIES - 1:
                        raise
                    time.sleep(RL_MODEL_LOAD_RETRY_DELAY)  # Brief pause before retry

        except Exception as e:
            self.logger.error(f"Failed to load RL model: {e}")
            if not RL_GRACEFUL_DEGRADATION:
                raise

    def _validate_model_compatibility(self) -> None:
        """Validate that loaded model is compatible with current network."""
        if not self.model or not self.rl_env:
            return

        try:
            # Check observation space compatibility
            model_obs_space = self.model.observation_space
            env_obs_space = self.rl_env.observation_space

            if model_obs_space.shape != env_obs_space.shape:
                raise ValueError(f"Observation space mismatch: model={model_obs_space.shape}, env={env_obs_space.shape}")

            # Check action space compatibility
            model_action_space = self.model.action_space
            env_action_space = self.rl_env.action_space

            if hasattr(model_action_space, 'nvec') and hasattr(env_action_space, 'nvec'):
                if not np.array_equal(model_action_space.nvec, env_action_space.nvec):
                    raise ValueError(f"Action space mismatch: model={model_action_space.nvec}, env={env_action_space.nvec}")

            self.logger.info("Model compatibility validation passed")

        except Exception as e:
            self.logger.error(f"Model compatibility validation failed: {e}")
            raise

    def _initialize_rl_environment(self) -> None:
        """Initialize RL environment for state collection."""
        try:
            # Get RL configuration
            rl_config = get_rl_config()

            # Create RL environment in inference mode (no reset/step needed)
            # Just used for state collection and action space definitions
            self.rl_env = TrafficControlEnv(rl_config)

            self.logger.info(f"RL environment initialized")
            self.logger.info(f"Observation space: {self.rl_env.observation_space}")
            self.logger.info(f"Action space: {self.rl_env.action_space}")

        except Exception as e:
            self.logger.error(f"Failed to initialize RL environment: {e}")
            raise

    def _initialize_traffic_lights(self) -> None:
        """Initialize traffic light state tracking."""
        try:
            traffic_lights = traci.trafficlight.getIDList()

            for tl_id in traffic_lights:
                # Get phase information
                complete_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[TRAFFIC_LIGHT_DEFINITION_INDEX]
                phases = complete_def.phases

                self.traffic_lights[tl_id] = {
                    'phase_count': len(phases),
                    'last_phase': DEFAULT_INITIAL_TIME,
                    'last_duration': DEFAULT_INITIAL_TIME,
                    'phase_start_time': DEFAULT_INITIAL_TIME
                }

            self.logger.info(f"Initialized tracking for {len(self.traffic_lights)} traffic lights")

        except Exception as e:
            self.logger.error(f"Failed to initialize traffic lights: {e}")
            raise

    def _initialize_action_tracking(self) -> None:
        """Initialize action distribution tracking."""
        if not RL_ACTION_DISTRIBUTION_TRACKING:
            return

        # Initialize counters for each possible action
        for phase in range(NUM_TRAFFIC_LIGHT_PHASES):
            for duration_idx in range(len(PHASE_DURATION_OPTIONS)):
                action_key = f"phase_{phase}_duration_{duration_idx}"
                self.action_counts[action_key] = DEFAULT_INITIAL_TIME

    def update(self, step: int) -> None:
        """Update RL control at given simulation step.

        Args:
            step: Current simulation step
        """
        try:
            # Vehicle tracking (same as other controllers)
            if hasattr(self, 'graph') and self.graph:
                self.graph.add_vehicles_to_step()
                self.graph.close_prev_vehicle_step(step)
        except Exception as e:
            self.logger.warning(f"RL vehicle tracking failed at step {step}: {e}")

        # RL control logic
        try:
            # Collect current state
            if self.rl_env:
                self.current_observation = self.rl_env._get_observation()

            # Generate RL action
            if self.mode == RL_INFERENCE_MODE and self.model:
                action = self._get_rl_action()
            else:
                # Training mode or no model - use default/random actions
                action = self._get_default_action()

            # Execute traffic signal actions
            self._execute_actions(action, step)

            # Update statistics
            if step % RL_STATISTICS_COLLECTION_INTERVAL == 0:
                self._update_statistics()

        except Exception as e:
            self.logger.error(f"RL update failed at step {step}: {e}")
            if not RL_GRACEFUL_DEGRADATION:
                raise

    def _get_rl_action(self) -> np.ndarray:
        """Get action from trained RL model."""
        if not self.model or self.current_observation is None:
            return self._get_default_action()

        try:
            # Track inference time if enabled
            start_time = time.time() if RL_INFERENCE_TIME_TRACKING else None

            # Get action from model
            action, _ = self.model.predict(self.current_observation, deterministic=True)

            # Record inference time
            if RL_INFERENCE_TIME_TRACKING and start_time:
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)

            # Track action distribution
            if RL_ACTION_DISTRIBUTION_TRACKING:
                self._track_action(action)

            return action

        except Exception as e:
            self.logger.error(f"RL model inference failed: {e}")
            return self._get_default_action()

    def _get_default_action(self) -> np.ndarray:
        """Get default action when model is not available."""
        if not self.rl_env:
            # Fallback: simple fixed action
            num_intersections = len(self.traffic_lights)
            return np.zeros(num_intersections * 2, dtype=np.int32)

        # Sample random action from action space
        return self.rl_env.action_space.sample()

    def _execute_actions(self, action: np.ndarray, step: int) -> None:
        """Execute RL actions on traffic signals.

        Args:
            action: RL action array
            step: Current simulation step
        """
        try:
            if not self.rl_env or len(self.traffic_lights) == 0:
                return

            # Reshape action to (num_intersections, 2) format
            num_intersections = len(self.traffic_lights)
            action_pairs = action.reshape(num_intersections, 2)

            traffic_light_ids = list(self.traffic_lights.keys())

            for i, tl_id in enumerate(traffic_light_ids):
                if i >= len(action_pairs):
                    break

                phase_idx, duration_idx = action_pairs[i]

                # Apply safety constraints if enabled
                if RL_SAFETY_CHECK_ENABLED:
                    phase_idx, duration_idx = self._apply_safety_constraints(
                        tl_id, int(phase_idx), int(duration_idx), step)

                # Convert duration index to actual duration
                duration = PHASE_DURATION_OPTIONS[int(duration_idx)]

                # Execute TraCI commands (same as other controllers)
                traci.trafficlight.setPhase(tl_id, int(phase_idx))
                traci.trafficlight.setPhaseDuration(tl_id, duration)

                # Update tracking
                self.traffic_lights[tl_id]['last_phase'] = int(phase_idx)
                self.traffic_lights[tl_id]['last_duration'] = duration
                self.traffic_lights[tl_id]['phase_start_time'] = step

            self.last_action_time = step

        except Exception as e:
            self.logger.error(f"Action execution failed at step {step}: {e}")
            raise

    def _apply_safety_constraints(self, tl_id: str, phase_idx: int, duration_idx: int, step: int) -> tuple:
        """Apply safety constraints to RL actions.

        Args:
            tl_id: Traffic light ID
            phase_idx: Requested phase index
            duration_idx: Requested duration index
            step: Current step

        Returns:
            tuple: (validated_phase_idx, validated_duration_idx)
        """
        # Constrain phase to valid range
        tl_info = self.traffic_lights.get(tl_id, {})
        max_phase = tl_info.get('phase_count', NUM_TRAFFIC_LIGHT_PHASES) - 1
        validated_phase = max(0, min(phase_idx, max_phase))

        # Constrain duration to valid range
        max_duration_idx = len(PHASE_DURATION_OPTIONS) - 1
        validated_duration_idx = max(0, min(duration_idx, max_duration_idx))

        # Enforce minimum green time if needed
        if RL_MIN_GREEN_TIME_ENFORCEMENT:
            duration = PHASE_DURATION_OPTIONS[validated_duration_idx]
            if duration < MIN_GREEN_TIME:
                # Find minimum acceptable duration index
                for i, dur in enumerate(PHASE_DURATION_OPTIONS):
                    if dur >= MIN_GREEN_TIME:
                        validated_duration_idx = i
                        break

        # Enforce maximum duration if needed
        if RL_MAX_DURATION_ENFORCEMENT:
            duration = PHASE_DURATION_OPTIONS[validated_duration_idx]
            if duration > MAX_PHASE_DURATION:
                # Find maximum acceptable duration index
                for i in range(len(PHASE_DURATION_OPTIONS) - 1, -1, -1):
                    if PHASE_DURATION_OPTIONS[i] <= MAX_PHASE_DURATION:
                        validated_duration_idx = i
                        break

        return validated_phase, validated_duration_idx

    def _track_action(self, action: np.ndarray) -> None:
        """Track action distribution for analysis."""
        if not RL_ACTION_DISTRIBUTION_TRACKING:
            return

        try:
            num_intersections = len(self.traffic_lights)
            action_pairs = action.reshape(num_intersections, 2)

            for phase_idx, duration_idx in action_pairs:
                action_key = f"phase_{int(phase_idx)}_duration_{int(duration_idx)}"
                if action_key in self.action_counts:
                    self.action_counts[action_key] += 1

        except Exception as e:
            self.logger.warning(f"Action tracking failed: {e}")

    def _update_statistics(self) -> None:
        """Update performance statistics."""
        self.statistics_step_counter += 1

        if not RL_PERFORMANCE_LOGGING_ENABLED:
            return

        try:
            # Log inference time statistics
            if RL_INFERENCE_TIME_TRACKING and self.inference_times:
                avg_inference_time = np.mean(self.inference_times)
                max_inference_time = np.max(self.inference_times)
                self.logger.debug(f"RL inference time - avg: {avg_inference_time:.4f}s, max: {max_inference_time:.4f}s")

                # Clear old times to prevent memory buildup
                if len(self.inference_times) > RL_INFERENCE_TIME_MAX_HISTORY:
                    self.inference_times = self.inference_times[-RL_INFERENCE_TIME_KEEP_RECENT:]

            # Log action distribution periodically
            if RL_ACTION_DISTRIBUTION_TRACKING and self.statistics_step_counter % RL_ACTION_DISTRIBUTION_LOG_INTERVAL == 0:
                self._log_action_distribution()

        except Exception as e:
            self.logger.warning(f"Statistics update failed: {e}")

    def _log_action_distribution(self) -> None:
        """Log action distribution statistics."""
        if not self.action_counts:
            return

        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return

        self.logger.debug("RL Action Distribution:")
        for action_key, count in sorted(self.action_counts.items()):
            percentage = (count / total_actions) * 100
            self.logger.debug(f"  {action_key}: {count} ({percentage:.1f}%)")

    def cleanup(self) -> None:
        """Clean up RL controller resources and report statistics."""
        try:
            self.logger.info(f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} CLEANUP STARTED ===")

            # Report vehicle statistics using Graph object (same as other controllers)
            if hasattr(self, 'graph') and self.graph:
                self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")

                # Report RL statistics using same calculation as other methods
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    rl_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info(f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} STATISTICS ===")
                    self.logger.info(f"RL - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(f"RL - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(f"RL - Average duration: {rl_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(f"RL - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info(f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} STATISTICS ===")
                    self.logger.info("RL - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")

            # Report RL-specific statistics
            self._report_rl_statistics()

            # Clean up RL environment
            if self.rl_env:
                try:
                    self.rl_env.close()
                except Exception as e:
                    self.logger.warning(f"RL environment cleanup failed: {e}")

        except Exception as e:
            self.logger.error(f"Error in RL cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _report_rl_statistics(self) -> None:
        """Report RL-specific performance statistics."""
        try:
            self.logger.info(f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} PERFORMANCE STATISTICS ===")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Model path: {self.model_path}")

            # Inference time statistics
            if RL_INFERENCE_TIME_TRACKING and self.inference_times:
                avg_time = np.mean(self.inference_times)
                min_time = np.min(self.inference_times)
                max_time = np.max(self.inference_times)
                total_inferences = len(self.inference_times)

                self.logger.info(f"Inference Time Statistics:")
                self.logger.info(f"  Total inferences: {total_inferences}")
                self.logger.info(f"  Average time: {avg_time:.4f}s")
                self.logger.info(f"  Min time: {min_time:.4f}s")
                self.logger.info(f"  Max time: {max_time:.4f}s")

            # Action distribution statistics
            if RL_ACTION_DISTRIBUTION_TRACKING and self.action_counts:
                total_actions = sum(self.action_counts.values())
                self.logger.info(f"Action Distribution Statistics:")
                self.logger.info(f"  Total actions: {total_actions}")

                # Most common actions
                sorted_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)
                self.logger.info("  Top 5 most common actions:")
                for i, (action_key, count) in enumerate(sorted_actions[:5]):
                    percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                    self.logger.info(f"    {i+1}. {action_key}: {count} ({percentage:.1f}%)")

        except Exception as e:
            self.logger.warning(f"RL statistics reporting failed: {e}")