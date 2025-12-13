"""
RLController for traffic signal control using reinforcement learning.

This module implements the RL-based traffic controller that integrates with
the existing traffic control architecture.
"""

import logging
import os
import time
import traceback
from typing import Any, Optional, Dict, List

import numpy as np
import traci

from src.orchestration.traffic_controller import TrafficController
from .constants import (
    DEFAULT_CYCLE_LENGTH,
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
    RL_ACTION_DISTRIBUTION_LOG_INTERVAL,
    RL_PHASE_ONLY_MODE, RL_FIXED_PHASE_DURATION
)
from .environment import TrafficControlEnv


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

        # RL cycle parameters
        self.cycle_lengths = getattr(
            args, 'rl_cycle_lengths', [DEFAULT_CYCLE_LENGTH])
        self.cycle_strategy = getattr(args, 'rl_cycle_strategy', 'fixed')
        # Use first cycle length as decision interval
        self.decision_interval = self.cycle_lengths[0]

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
        self.traffic_lights_initialized = False  # Track whether traffic lights have been initialized
        self.junction_phase_counts = {}  # Store phase counts for each junction
        self.last_action_time = DEFAULT_INITIAL_TIME

        # Duration-based control state
        self.current_schedules = {}  # Store phase schedules for current cycle
        self.cycle_start_step = 0  # Track when current cycle started

        # Verify mode setting logic
        if self.model_path:
            expected_mode = RL_INFERENCE_MODE
            if self.mode != expected_mode:
                self.logger.error(
                    f"MODE MISMATCH: Expected '{expected_mode}' but got '{self.mode}'")
        else:
            expected_mode = RL_DEFAULT_MODE
            if self.mode != expected_mode:
                self.logger.error(
                    f"MODE MISMATCH: Expected '{expected_mode}' but got '{self.mode}'")

    def initialize(self) -> None:
        """Initialize RL controller with model loading and environment setup."""
        try:
            # Initialize Graph object for vehicle tracking (same as other controllers)
            from src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.graph import Graph
            self.graph = Graph(self.args.end_time, self.args.tree_method_m, self.args.tree_method_l)

            # Load RL model if in inference mode
            if self.mode == RL_INFERENCE_MODE:
                self._load_rl_model()

            # Initialize RL environment for state collection
            self._initialize_rl_environment()

            # Note: Traffic light initialization moved to first update() call
            # when SUMO is actually running (lazy initialization)

            # Initialize action distribution tracking
            if RL_ACTION_DISTRIBUTION_TRACKING:
                self._initialize_action_tracking()

        except Exception as e:
            self.logger.error(
                f"{RL_CONTROLLER_ERROR_PREFIX} initialization failed: {e}")
            if not RL_GRACEFUL_DEGRADATION:
                raise

    def _load_rl_model(self) -> None:
        """Load trained RL model for inference."""
        if not self.model_path:
            return

        try:
            from stable_baselines3 import PPO

            # Validate model path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"RL model not found: {self.model_path}")

            # Load model with retries
            for attempt in range(RL_MODEL_LOAD_RETRIES):
                try:
                    self.model = PPO.load(self.model_path)

                    # Validate model compatibility if enabled
                    if RL_MODEL_COMPATIBILITY_CHECK:
                        self._validate_model_compatibility()

                    return

                except Exception as e:
                    if attempt == RL_MODEL_LOAD_RETRIES - 1:
                        raise
                    # Brief pause before retry
                    time.sleep(RL_MODEL_LOAD_RETRY_DELAY)

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
                self.logger.error(f"OBSERVATION SPACE MISMATCH DETECTED!")
                self.logger.error(f"  Model expects: {model_obs_space.shape}")
                self.logger.error(
                    f"  Environment provides: {env_obs_space.shape}")
                raise ValueError(
                    f"Observation space mismatch: model={model_obs_space.shape}, env={env_obs_space.shape}")

            # Check observation bounds compatibility
            if hasattr(model_obs_space, 'low') and hasattr(env_obs_space, 'low'):
                model_low = model_obs_space.low
                env_low = env_obs_space.low
                model_high = model_obs_space.high
                env_high = env_obs_space.high

                self.logger.info(
                    f"Model bounds: low={model_low[0]:.3f} to {model_low[-1]:.3f}, high={model_high[0]:.3f} to {model_high[-1]:.3f}")
                self.logger.info(
                    f"Environment bounds: low={env_low[0]:.3f} to {env_low[-1]:.3f}, high={env_high[0]:.3f} to {env_high[-1]:.3f}")

                # Check if bounds are approximately equal (allow some tolerance)
                low_diff = np.abs(model_low - env_low).max()
                high_diff = np.abs(model_high - env_high).max()
                self.logger.info(
                    f"Bounds differences: low_diff={low_diff:.6f}, high_diff={high_diff:.6f}")

                if low_diff > 0.1 or high_diff > 0.1:
                    self.logger.warning(
                        f"Observation bounds mismatch (but within tolerance)")

            # Check action space compatibility
            model_action_space = self.model.action_space
            env_action_space = self.rl_env.action_space

            self.logger.info(f"Model action space: {model_action_space}")
            self.logger.info(f"Environment action space: {env_action_space}")

            if hasattr(model_action_space, 'nvec') and hasattr(env_action_space, 'nvec'):
                self.logger.info(
                    f"Model action nvec: {model_action_space.nvec}")
                self.logger.info(
                    f"Environment action nvec: {env_action_space.nvec}")

                if not np.array_equal(model_action_space.nvec, env_action_space.nvec):
                    self.logger.error(f"ACTION SPACE MISMATCH DETECTED!")
                    self.logger.error(
                        f"  Model expects: {model_action_space.nvec}")
                    self.logger.error(
                        f"  Environment provides: {env_action_space.nvec}")
                    raise ValueError(
                        f"Action space mismatch: model={model_action_space.nvec}, env={env_action_space.nvec}")

            self.logger.info("=== MODEL COMPATIBILITY VALIDATION: SUCCESS ===")

        except Exception as e:
            self.logger.error(
                f"=== MODEL COMPATIBILITY VALIDATION: FAILED ===")
            self.logger.error(f"Model compatibility validation failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_rl_environment(self) -> None:
        """Initialize RL environment for state collection."""
        try:
            self.logger.info("Creating RL environment...")
            # Create minimal environment for inference (no pipeline execution)
            # This avoids re-parsing arguments and running the full simulation pipeline
            self.rl_env = TrafficControlEnv.from_namespace(
                self.args, minimal=True)

            self.logger.info(f"RL environment created successfully")
            self.logger.info(f"Observation space: {self.rl_env.observation_space.shape}")
            self.logger.info(f"Action space: {self.rl_env.action_space.shape}")

        except Exception as e:
            self.logger.error(f"Failed to initialize RL environment: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_traffic_lights(self) -> None:
        """Initialize traffic light state tracking."""
        try:
            traffic_lights = traci.trafficlight.getIDList()

            for tl_id in traffic_lights:
                # Get phase information
                complete_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[
                    TRAFFIC_LIGHT_DEFINITION_INDEX]
                phases = complete_def.phases

                phase_count = len(phases)
                self.traffic_lights[tl_id] = {
                    'phase_count': phase_count,
                    'last_phase': DEFAULT_INITIAL_TIME,
                    'last_duration': DEFAULT_INITIAL_TIME,
                    'phase_start_time': DEFAULT_INITIAL_TIME
                }

                # Store phase count for phase clipping (used in _apply_scheduled_phases)
                self.junction_phase_counts[tl_id] = phase_count

            # self.logger.info(
            #     f"Initialized tracking for {len(self.traffic_lights)} traffic lights")

        except Exception as e:
            self.logger.error(f"Failed to initialize traffic lights: {e}")
            raise

    def _initialize_action_tracking(self) -> None:
        """Initialize action distribution tracking."""
        if not RL_ACTION_DISTRIBUTION_TRACKING:
            return

        if RL_PHASE_ONLY_MODE:
            # Initialize counters for phase-only actions
            for phase in range(NUM_TRAFFIC_LIGHT_PHASES):
                action_key = f"phase_{phase}"
                self.action_counts[action_key] = DEFAULT_INITIAL_TIME
        else:
            # Initialize counters for phase+duration actions
            for phase in range(NUM_TRAFFIC_LIGHT_PHASES):
                for duration_idx in range(len(PHASE_DURATION_OPTIONS)):
                    action_key = f"phase_{phase}_duration_{duration_idx}"
                    self.action_counts[action_key] = DEFAULT_INITIAL_TIME

    def update(self, step: int) -> None:
        """Update RL control at given simulation step.

        Args:
            step: Current simulation step
        """
        # Lazy initialization: Initialize traffic lights on first update when SUMO is running
        if not self.traffic_lights_initialized:
            try:
                self._initialize_traffic_lights()
                self.traffic_lights_initialized = True
            except Exception as e:
                self.logger.error(f"Failed to initialize traffic lights on first update: {e}")
                # Don't raise - allow simulation to continue

        try:
            # Vehicle tracking (same as other controllers)
            if hasattr(self, 'graph') and self.graph:
                self.graph.add_vehicles_to_step()
                self.graph.close_prev_vehicle_step(step)
        except Exception as e:
            self.logger.warning(
                f"RL vehicle tracking failed at step {step}: {e}")

        # RL control logic - only make decisions at specified intervals
        try:
            # For continuous duration-based control: apply scheduled phases at every step
            from .constants import RL_USE_CONTINUOUS_ACTIONS
            if RL_USE_CONTINUOUS_ACTIONS and hasattr(self, 'current_schedules') and self.current_schedules:
                # Calculate time within current cycle
                time_in_cycle = (
                    step - self.cycle_start_step) % self.decision_interval
                self._apply_scheduled_phases(time_in_cycle)

            # Only make RL decisions every decision_interval seconds
            if step % self.decision_interval == 0:
                # self.logger.info(
                #     f"=== RL DECISION STEP {step} (DECISION INTERVAL: {self.decision_interval}) ===")
                # self.logger.info(
                #     f"Mode: {self.mode}, Model available: {self.model is not None}")

                # Collect current state
                if self.rl_env:
                    try:
                        self.current_observation = self.rl_env._get_observation()
                        # self.logger.info(
                        #     f"Observation collected - shape: {self.current_observation.shape if self.current_observation is not None else 'None'}")

                        # DIAGNOSTIC: Log observation statistics to detect if observations are frozen
                        if self.current_observation is not None:
                            obs_hash = hash(self.current_observation.tobytes())
                            obs_min = np.min(self.current_observation)
                            obs_max = np.max(self.current_observation)
                            obs_mean = np.mean(self.current_observation)
                            obs_std = np.std(self.current_observation)
                            obs_first_10 = self.current_observation[:10]
                            obs_last_10 = self.current_observation[-10:]

                            self.logger.info(f"DIAGNOSTIC Step {step} - Observation stats:")
                            self.logger.info(f"  Hash: {obs_hash}, Min: {obs_min:.6f}, Max: {obs_max:.6f}, Mean: {obs_mean:.6f}, Std: {obs_std:.6f}")
                            self.logger.info(f"  First 10: {obs_first_10}")
                            self.logger.info(f"  Last 10: {obs_last_10}")

                    except Exception as e:
                        self.logger.error(
                            f"Failed to collect observation: {e}")
                        import traceback
                        self.logger.error(
                            f"Traceback: {traceback.format_exc()}")
                        self.current_observation = None
                else:
                    self.logger.warning(
                        "No RL environment available for observation collection")

                # Generate RL action
                action_source = "UNKNOWN"
                if self.mode == RL_INFERENCE_MODE and self.model:
                    # self.logger.info("Using INFERENCE MODE with trained model")
                    action = self._get_rl_action()
                    action_source = "RL_MODEL"
                else:
                    # Training mode or no model - use default/random actions
                    # self.logger.info(
                    #     f"Using DEFAULT MODE (mode: {self.mode}, model: {self.model is not None})")
                    action = self._get_default_action()
                    action_source = "DEFAULT"

                # self.logger.info(
                #     f"Action source: {action_source}, Action shape: {action.shape if action is not None else 'None'}")
                # if action is not None:
                #     self.logger.info(f"Action values: {action}")

                # Execute traffic signal actions
                self._execute_actions(action, step)
            # else:
            #     # Skip non-decision steps - only log occasionally to avoid spam
            #     # Log every 10 decision intervals
            #     if step % (self.decision_interval * 10) == 0:
            #         self.logger.debug(
            #             f"Step {step}: Skipping non-decision step (next decision at step {((step // self.decision_interval) + 1) * self.decision_interval})")

            # Update statistics
            if step % RL_STATISTICS_COLLECTION_INTERVAL == 0:
                self._update_statistics()

        except Exception as e:
            self.logger.error(f"RL update failed at step {step}: {e}")
            if not RL_GRACEFUL_DEGRADATION:
                raise

    def _get_rl_action(self) -> np.ndarray:
        """Get action from trained RL model."""
        # self.logger.info(f"=== RL MODEL INFERENCE START ===")
        # self.logger.info(f"Model available: {self.model is not None}")
        # self.logger.info(f"Observation available: {self.current_observation is not None}")

        if not self.model:
            self.logger.warning(
                "No model available - falling back to default action")
            return self._get_default_action()

        if self.current_observation is None:
            self.logger.warning(
                "No observation available - falling back to default action")
            return self._get_default_action()

        try:
            # Track inference time if enabled
            start_time = time.time() if RL_INFERENCE_TIME_TRACKING else None

            # self.logger.info(f"Calling model.predict() with observation shape: {self.current_observation.shape}")

            # Get action from model
            action, _ = self.model.predict(
                self.current_observation, deterministic=True)

            # self.logger.info(f"Model prediction successful - action type: {type(action)}, shape: {action.shape}")
            # self.logger.info(f"Predicted action values: {action}")

            # DIAGNOSTIC: Log raw model outputs to detect if actions are frozen
            action_hash = hash(action.tobytes())
            self.logger.info(f"DIAGNOSTIC - Raw model action output:")
            self.logger.info(f"  Action hash: {action_hash}")
            self.logger.info(f"  First 8 values: {action[:8]}")
            self.logger.info(f"  Last 8 values: {action[-8:]}")
            self.logger.info(f"  Action shape: {action.shape}, min: {np.min(action):.6f}, max: {np.max(action):.6f}, mean: {np.mean(action):.6f}")
            # Log B0 junction's raw values (junction index 4, action indices 16-19)
            if len(action) >= 20:
                b0_values = action[16:20]
                self.logger.info(f"  B0 junction (indices 16-19) raw values: {b0_values}")

            # Record inference time
            if RL_INFERENCE_TIME_TRACKING and start_time:
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.logger.info(f"Inference time: {inference_time:.4f}s")

            # Track action distribution
            if RL_ACTION_DISTRIBUTION_TRACKING:
                self._track_action(action)

            # self.logger.info(f"=== RL MODEL INFERENCE SUCCESS ===")
            return action

        except Exception as e:
            self.logger.error(f"=== RL MODEL INFERENCE FAILED ===")
            self.logger.error(f"RL model inference failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.error(f"Falling back to default action")
            return self._get_default_action()

    def _get_default_action(self) -> np.ndarray:
        """Get default action when model is not available."""
        self.logger.info(f"=== GENERATING DEFAULT ACTION ===")
        self.logger.info(
            f"RL environment available: {self.rl_env is not None}")
        self.logger.info(f"Traffic lights count: {len(self.traffic_lights)}")

        if not self.rl_env:
            # Fallback: simple fixed action
            num_intersections = len(self.traffic_lights)
            self.logger.info(
                f"No RL environment - using fixed action for {num_intersections} intersections")

            if RL_PHASE_ONLY_MODE:
                action = np.zeros(num_intersections,
                                  dtype=np.int32)  # Phase-only
                self.logger.info(f"Fixed action (phase-only): {action}")
                return action
            else:
                action = np.zeros(num_intersections * 2,
                                  dtype=np.int32)  # Phase + duration
                self.logger.info(f"Fixed action (phase+duration): {action}")
                return action

        # Sample random action from action space
        self.logger.info(
            f"Using RL environment action space: {self.rl_env.action_space}")
        action = self.rl_env.action_space.sample()
        self.logger.info(f"Random action sampled: {action}")
        return action

    def _execute_actions(self, action: np.ndarray, step: int) -> None:
        """Execute RL actions on traffic signals.

        Args:
            action: RL action array
            step: Current simulation step
        """

        # Utility function to convert lane ID to edge ID
        def lane_id_to_edge_id(lane_id):
            """Convert lane ID (e.g., 'A1A0_H_s_0') to edge ID (e.g., 'A1A0_H_s')"""
            if '_' in lane_id:
                # Remove the last part after underscore (lane index)
                return '_'.join(lane_id.split('_')[:-1])
            return lane_id

        try:

            if not self.rl_env or len(self.traffic_lights) == 0:
                self.logger.warning(
                    f"Cannot execute actions - RL env: {self.rl_env is not None}, Traffic lights: {len(self.traffic_lights)}")
                return

            num_intersections = len(self.traffic_lights)
            traffic_light_ids = list(self.traffic_lights.keys())
            # self.logger.info(
            #     f"Executing actions for {num_intersections} intersections: {traffic_light_ids}")

            # Import action space constants
            from .constants import RL_USE_CONTINUOUS_ACTIONS, RL_ACTIONS_PER_JUNCTION, MIN_PHASE_DURATION

            if RL_USE_CONTINUOUS_ACTIONS:
                # Duration-based control: convert raw actions to duration schedules
                self.logger.info("Using CONTINUOUS DURATION-BASED mode")

                junction_schedules = {}

                for i, junction_id in enumerate(traffic_light_ids):
                    # Extract 4 values for this junction
                    start_idx = i * RL_ACTIONS_PER_JUNCTION
                    end_idx = start_idx + RL_ACTIONS_PER_JUNCTION
                    raw_outputs = action[start_idx:end_idx]

                    # Apply softmax to get proportions
                    proportions = self._softmax(raw_outputs)

                    # Convert to durations summing to current_cycle_length
                    durations = self._proportions_to_durations(
                        proportions,
                        self.decision_interval,  # Use decision_interval as cycle_length
                        MIN_PHASE_DURATION
                    )

                    junction_schedules[junction_id] = durations

                    # Log for first junction
                    if i < 24:
                        self.logger.info(
                            f"  Example: junction {junction_id} -> durations {durations} (total: {sum(durations)}s)")

                # Store schedule for application during cycle
                self.current_schedules = junction_schedules
                self.cycle_start_step = step

                self.logger.info(
                    f"  Stored schedules for {len(junction_schedules)} junctions, cycle starts at step {step}")

            elif RL_PHASE_ONLY_MODE:
                # self.logger.info("Using PHASE-ONLY mode")
                # Phase-only mode: action is flat array of phase indices
                # Ensure correct length
                phase_actions = action.flatten()[:num_intersections]
                # self.logger.info(f"Phase actions extracted: {phase_actions}")

                # Apply safety constraints and execute actions
                for i, tl_id in enumerate(traffic_light_ids):
                    if i >= len(phase_actions):
                        self.logger.warning(
                            f"Action index {i} exceeds phase_actions length {len(phase_actions)}")
                        break

                    phase_idx = int(phase_actions[i])
                    original_phase = phase_idx

                    # Apply safety constraints if enabled
                    if RL_SAFETY_CHECK_ENABLED:
                        phase_idx = self._apply_phase_safety_constraints(
                            tl_id, phase_idx, step)
                        if phase_idx != original_phase:
                            self.logger.info(
                                f"  Safety constraint applied: {original_phase} -> {phase_idx}")

                    # Apply the RL decision with fixed duration
                    try:
                        traci.trafficlight.setPhase(tl_id, phase_idx)
                        traci.trafficlight.setPhaseDuration(
                            tl_id, RL_FIXED_PHASE_DURATION)

                    except Exception as e:
                        self.logger.error(
                            f"  Failed to apply RL action to {tl_id}: {e}")
                        import traceback
                        self.logger.error(
                            f"  Traceback: {traceback.format_exc()}")

                    # Update tracking
                    self.traffic_lights[tl_id]['last_phase'] = phase_idx
                    self.traffic_lights[tl_id]['last_duration'] = RL_FIXED_PHASE_DURATION
                    self.traffic_lights[tl_id]['phase_start_time'] = step

            else:
                self.logger.info("Using PHASE+DURATION mode")
                # Legacy phase+duration mode: reshape to (num_intersections, 2) format
                action_pairs = action.reshape(num_intersections, 2)
                self.logger.info(f"Action pairs: {action_pairs}")

                # Apply safety constraints and execute actions
                for i, tl_id in enumerate(traffic_light_ids):
                    if i >= len(action_pairs):
                        self.logger.warning(
                            f"Action index {i} exceeds action_pairs length {len(action_pairs)}")
                        break

                    phase_idx, duration_idx = action_pairs[i]
                    original_phase = int(phase_idx)
                    original_duration_idx = int(duration_idx)
                    self.logger.info(
                        f"Traffic light {tl_id} (index {i}): Requested phase {phase_idx}, duration_idx {duration_idx}")

                    # Get current phase before change
                    try:
                        current_phase = traci.trafficlight.getPhase(tl_id)
                        current_duration = traci.trafficlight.getPhaseDuration(
                            tl_id)
                        self.logger.info(
                            f"  Current state: phase {current_phase}, duration {current_duration}")
                    except Exception as e:
                        self.logger.warning(
                            f"  Could not get current state: {e}")

                    # Apply safety constraints if enabled
                    if RL_SAFETY_CHECK_ENABLED:
                        phase_idx, duration_idx = self._apply_safety_constraints(
                            tl_id, int(phase_idx), int(duration_idx), step)
                        if phase_idx != original_phase or duration_idx != original_duration_idx:
                            self.logger.info(
                                f"  Safety constraint applied: phase {original_phase} -> {phase_idx}, duration_idx {original_duration_idx} -> {duration_idx}")

                    duration = PHASE_DURATION_OPTIONS[int(duration_idx)]
                    self.logger.info(
                        f"  Final values: phase {phase_idx}, duration {duration}")

                    # Apply the RL decision
                    try:
                        self.logger.info(
                            f"  Setting phase to {phase_idx} with duration {duration}")
                        traci.trafficlight.setPhase(tl_id, int(phase_idx))
                        traci.trafficlight.setPhaseDuration(tl_id, duration)

                        # Verify the change was applied
                        new_phase = traci.trafficlight.getPhase(tl_id)
                        new_duration = traci.trafficlight.getPhaseDuration(
                            tl_id)
                        self.logger.info(
                            f"  Verification: phase {new_phase}, duration {new_duration}")

                        if new_phase != int(phase_idx):
                            self.logger.error(
                                f"  PHASE MISMATCH: Requested {phase_idx}, got {new_phase}")
                        # Allow small floating point differences
                        if abs(new_duration - duration) > 1:
                            self.logger.error(
                                f"  DURATION MISMATCH: Requested {duration}, got {new_duration}")

                    except Exception as e:
                        self.logger.error(
                            f"  Failed to apply RL action to {tl_id}: {e}")
                        import traceback
                        self.logger.error(
                            f"  Traceback: {traceback.format_exc()}")

                    # Update tracking
                    self.traffic_lights[tl_id]['last_phase'] = int(phase_idx)
                    self.traffic_lights[tl_id]['last_duration'] = duration
                    self.traffic_lights[tl_id]['phase_start_time'] = step

            self.last_action_time = step
            # self.logger.info(
            #     f"=== ACTION EXECUTION COMPLETED AT STEP {step} ===")

        except Exception as e:
            self.logger.error(
                f"=== ACTION EXECUTION FAILED AT STEP {step} ===")
            self.logger.error(f"Action execution failed at step {step}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _apply_phase_safety_constraints(self, tl_id: str, phase_idx: int, step: int) -> int:
        """Apply safety constraints to phase-only RL actions.

        Args:
            tl_id: Traffic light ID
            phase_idx: Requested phase index
            step: Current step

        Returns:
            int: Validated phase index
        """
        # Constrain phase to valid range
        tl_info = self.traffic_lights.get(tl_id, {})
        max_phase = tl_info.get('phase_count', NUM_TRAFFIC_LIGHT_PHASES) - 1
        validated_phase = max(0, min(phase_idx, max_phase))

        return validated_phase

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

        return validated_phase, validated_duration_idx

    def _softmax(self, x):
        """Numerically stable softmax.

        Args:
            x: Array of raw values

        Returns:
            Array of probabilities summing to 1.0
        """
        if len(x) == 0:
            raise ValueError("Cannot apply softmax to empty array")

        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _proportions_to_durations(self, proportions, cycle_length, min_phase_time):
        """Convert proportions to integer durations with constraints.

        Args:
            proportions: Array of 4 proportions summing to 1.0 (e.g., [0.25, 0.15, 0.40, 0.20])
            cycle_length: Total cycle time in seconds (e.g., 90)
            min_phase_time: Minimum duration per phase in seconds (e.g., 10)

        Returns:
            List of 4 integer durations summing exactly to cycle_length
        """
        num_phases = len(proportions)
        available_time = cycle_length - (num_phases * min_phase_time)

        # Calculate durations
        durations = [min_phase_time + (p * available_time)
                     for p in proportions]

        # Round to integers
        durations = [int(round(d)) for d in durations]

        # Ensure exact sum (adjust largest phase if needed)
        diff = cycle_length - sum(durations)
        if diff != 0:
            max_idx = np.argmax(durations)
            durations[max_idx] += diff

        return durations

    def _apply_scheduled_phases(self, time_in_cycle):
        """Apply correct phase based on schedule and time within cycle.

        Args:
            time_in_cycle: Time elapsed since start of cycle (0 to cycle_length-1)
        """
        if not hasattr(self, 'current_schedules') or not self.current_schedules:
            return

        for junction_id, durations in self.current_schedules.items():
            # Find which phase should be active at this time
            elapsed = 0
            for phase_idx, duration in enumerate(durations):
                if time_in_cycle < elapsed + duration:
                    # This is the active phase - apply it
                    # FIX: Clip phase_idx to actual phase count at this junction
                    actual_phase_count = self.junction_phase_counts.get(junction_id, 4)
                    safe_phase_idx = min(phase_idx, actual_phase_count - 1)

                    try:
                        traci.trafficlight.setPhase(junction_id, safe_phase_idx)
                        traci.trafficlight.setPhaseDuration(
                            junction_id, duration)
                    except Exception as e:
                        self.logger.warning(f"Failed to set phase {safe_phase_idx} for {junction_id}: {e}")
                    break
                elapsed += duration

    def _track_action(self, action: np.ndarray) -> None:
        """Track action distribution for analysis."""
        if not RL_ACTION_DISTRIBUTION_TRACKING:
            return

        try:
            num_intersections = len(self.traffic_lights)

            if RL_PHASE_ONLY_MODE:
                # Phase-only mode: track phase indices only
                phase_actions = action.flatten()[:num_intersections]
                for phase_idx in phase_actions:
                    action_key = f"phase_{int(phase_idx)}"
                    if action_key not in self.action_counts:
                        self.action_counts[action_key] = 0
                    self.action_counts[action_key] += 1
            else:
                # Legacy phase+duration mode
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
                self.logger.debug(
                    f"RL inference time - avg: {avg_inference_time:.4f}s, max: {max_inference_time:.4f}s")

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
            # self.logger.info(
            #     f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} CLEANUP STARTED ===")

            # Report vehicle statistics using Graph object (same as other controllers)
            # if hasattr(self, 'graph') and self.graph:
            #     self.logger.info(f"Graph object exists: {type(self.graph)}")
            #     self.logger.info(
            #         f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
            #     self.logger.info(
            #         f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")

            #     # Report RL statistics using same calculation as other methods
            #     if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
            #         rl_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
            #         self.logger.info(
            #             f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} STATISTICS ===")
            #         self.logger.info(
            #             f"RL - Vehicles completed: {self.graph.ended_vehicles_count}")
            #         self.logger.info(
            #             f"RL - Total driving time: {self.graph.vehicle_total_time}")
            #         self.logger.info(
            #             f"RL - Average duration: {rl_avg_duration:.2f} steps")
            #         if hasattr(self.graph, 'driving_Time_seconds'):
            #             self.logger.info(
            #                 f"RL - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
            #     else:
            #         self.logger.info(
            #             f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} STATISTICS ===")
            #         self.logger.info(
            #             "RL - No completed vehicles found or graph not properly initialized")
            # else:
            #     self.logger.info("Graph object not found or not initialized")

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
        if not RL_PERFORMANCE_LOGGING_ENABLED:
            return

        try:
            self.logger.info(
                f"=== {RL_CONTROLLER_DISPLAY_NAME.upper()} PERFORMANCE STATISTICS ===")
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
                sorted_actions = sorted(
                    self.action_counts.items(), key=lambda x: x[1], reverse=True)
                self.logger.info("  Top 5 most common actions:")
                for i, (action_key, count) in enumerate(sorted_actions[:5]):
                    percentage = (count / total_actions) * \
                        100 if total_actions > 0 else 0
                    self.logger.info(
                        f"    {i+1}. {action_key}: {count} ({percentage:.1f}%)")

        except Exception as e:
            self.logger.warning(f"RL statistics reporting failed: {e}")
