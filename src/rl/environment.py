"""
RL Environment for Traffic Signal Control.

This module implements the OpenAI Gymnasium environment that interfaces
with SUMO simulation for reinforcement learning training.
"""

import argparse
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
from typing import Dict, Tuple, Any
from pathlib import Path

import gymnasium as gym
import numpy as np
import traci

from .reward import RewardCalculator
from .constants import (
    STATE_NORMALIZATION_MIN, STATE_NORMALIZATION_MAX,
    EDGE_FEATURES_COUNT, JUNCTION_FEATURES_COUNT,
    ACTIONS_PER_INTERSECTION, PHASE_DURATION_OPTIONS,
    MIN_PHASE_DURATION, MAX_PHASE_DURATION, MIN_GREEN_TIME, MAX_GREEN_TIME,
    MEASUREMENT_INTERVAL_STEPS,
    MAX_DENSITY_VEHICLES_PER_METER, MAX_FLOW_VEHICLES_PER_SECOND,
    CONGESTION_WAITING_TIME_THRESHOLD, NUM_TRAFFIC_LIGHT_PHASES, NUM_PHASE_DURATION_OPTIONS,
    DEFAULT_INITIAL_STEP, DEFAULT_INITIAL_TIME, DEFAULT_FALLBACK_VALUE, DEFAULT_OBSERVATION_PADDING,
    RL_PHASE_ONLY_MODE, RL_FIXED_PHASE_DURATION, RL_ACTIONS_PER_INTERSECTION_PHASE_ONLY,
    PROGRESSIVE_BONUS_ENABLED,
    REWARD_THROUGHPUT_PER_VEHICLE, REWARD_WAITING_TIME_PENALTY_WEIGHT,
    REWARD_EXCESSIVE_WAITING_PENALTY, REWARD_EXCESSIVE_WAITING_THRESHOLD,
    REWARD_SPEED_REWARD_FACTOR, REWARD_SPEED_NORMALIZATION,
    REWARD_BOTTLENECK_PENALTY_PER_EDGE, REWARD_INSERTION_BONUS, REWARD_INSERTION_THRESHOLD
)
from .vehicle_tracker import VehicleTracker
from src.pipeline.pipeline_factory import PipelineFactory
from src.args.parser import create_argument_parser

# Import Tree Method constants for cost-based reward calculation
from src.traffic_control.decentralized_traffic_bottlenecks.shared.config import (
    MAX_DENSITY, MIN_VELOCITY, M, L
)


class TrafficControlEnv(gym.Env):
    """
    OpenAI Gymnasium environment for network-wide traffic signal control.

    Implements the environment design from RL_DISCUSSION.md:
    - State: Macroscopic traffic indicators (speed, density, flow, congestion)
    - Action: Phase + duration selection for all intersections
    - Reward: Individual vehicle penalties + throughput bonuses
    """

    @classmethod
    def from_args(cls, args):
        """Create environment from argparse Namespace object.

        Args:
            args: Parsed argparse Namespace with simulation parameters

        Returns:
            TrafficControlEnv: Initialized environment
        """
        # Convert args to parameter string with proper quoting
        env_params_list = []

        # Skip RL-specific and internal args
        skip_keys = {
            'traffic_control', 'rl_model_path', 'gui', 'quiet', 'verbose',
            'bottleneck_detection_interval', 'atlcs_interval', 'tree_method_interval',
            'tree_method_sample',
            # Skip internal seed cache attributes added by multi_seed_utils
            '_network_seed', '_private_traffic_seed', '_public_traffic_seed'
        }

        for key, value in vars(args).items():
            if value is not None and key not in skip_keys:
                # Convert underscores to hyphens for CLI compatibility
                cli_key = key.replace('_', '-')

                if isinstance(value, bool):
                    if value:
                        env_params_list.append(f"--{cli_key}")
                else:
                    # Convert floats to ints if they're whole numbers
                    if isinstance(value, float) and value.is_integer():
                        value_str = str(int(value))
                    else:
                        value_str = str(value)

                    # Use shlex.quote to properly handle spaces and special characters
                    env_params_list.append(
                        f"--{cli_key} {shlex.quote(value_str)}")

        env_params_string = ' '.join(env_params_list)

        return cls(env_params_string)

    @classmethod
    def from_namespace(cls, args, minimal=False):
        """Create environment from argparse Namespace without running pipeline.

        Args:
            args: Parsed argparse Namespace with simulation parameters
            minimal: If True, create minimal env for inference (no pipeline)

        Returns:
            TrafficControlEnv: Initialized environment
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        gym.Env.__init__(instance)

        # Extract key parameters directly from args
        # FIX: Account for junctions_to_remove in intersection count
        from src.network.generate_grid import parse_junctions_to_remove

        grid_dimension = int(args.grid_dimension)
        junctions_remove_input = getattr(args, 'junctions_to_remove', '0')
        is_list, junction_ids, remove_count = parse_junctions_to_remove(junctions_remove_input)
        instance.num_intersections = (grid_dimension * grid_dimension) - remove_count
        instance.num_vehicles = args.num_vehicles
        instance.end_time = args.end_time
        instance.workspace = getattr(args, 'workspace', 'workspace')
        instance.cli_args = vars(args)

        # Cycle length management
        from .constants import DEFAULT_CYCLE_LENGTH, MIN_PHASE_DURATION, MAX_CYCLE_LENGTH
        instance.cycle_lengths = getattr(args, 'rl_cycle_lengths', [DEFAULT_CYCLE_LENGTH])
        instance.cycle_strategy = getattr(args, 'rl_cycle_strategy', 'fixed')
        instance.current_cycle_length = instance.cycle_lengths[0]
        instance.min_phase_time = MIN_PHASE_DURATION
        instance.decision_count = 0
        instance.current_schedules = {}
        instance.cycle_start_step = 0

        # Network path for reuse (None for inference mode)
        instance.network_path = None

        # Calculate state vector size dynamically by getting actual network dimensions
        # This ensures observation space matches the actual edge count after edge splitting
        from .constants import (
            RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
            RL_DYNAMIC_NETWORK_FEATURES_COUNT, RL_USE_CONTINUOUS_ACTIONS, RL_ACTIONS_PER_JUNCTION
        )

        # Get actual edge and junction counts by temporarily generating the network
        actual_edge_count, actual_junction_count = instance._get_actual_network_dimensions()

        instance.state_vector_size = (actual_edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                                      actual_junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                                      RL_DYNAMIC_NETWORK_FEATURES_COUNT +
                                      1)  # +1 for normalized cycle_length

        # Define observation space
        instance.observation_space = gym.spaces.Box(
            low=STATE_NORMALIZATION_MIN,
            high=STATE_NORMALIZATION_MAX + 0.01,
            shape=(instance.state_vector_size,),
            dtype=np.float32
        )

        # Define action space
        if RL_USE_CONTINUOUS_ACTIONS:
            # Use finite bounds for PPO compatibility
            # Actions are logits that get normalized via softmax
            # Range of -10 to +10 covers practical spectrum (exp(-10) ≈ 0%, exp(+10) ≈ 100%)
            instance.action_space = gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(instance.num_intersections * RL_ACTIONS_PER_JUNCTION,),
                dtype=np.float32
            )
        elif RL_PHASE_ONLY_MODE:
            instance.action_space = gym.spaces.MultiDiscrete(
                [NUM_TRAFFIC_LIGHT_PHASES] * instance.num_intersections)
        else:
            instance.action_space = gym.spaces.MultiDiscrete(
                [NUM_TRAFFIC_LIGHT_PHASES, NUM_PHASE_DURATION_OPTIONS] * instance.num_intersections)

        # Initialize simulation state (match __init__ exactly)
        instance.current_step = DEFAULT_INITIAL_STEP
        instance.episode_start_time = DEFAULT_INITIAL_TIME
        instance.workspace_dir = None
        instance.vehicle_tracker = None
        instance.junction_ids = []
        instance.edge_ids = []

        # TraCI connection state
        instance.traci_connected = False
        instance.traci_port = None

        # Tree Method traffic analyzer
        instance.traffic_analyzer = None

        # For minimal inference mode, don't run pipeline - controller will handle TraCI
        # if minimal:
        #     import logging
        #     logger = logging.getLogger(__name__)
        #     logger.info(
        #         "Created minimal TrafficControlEnv for inference (no pipeline)")

        return instance

    def __init__(self, env_params_string: str, episode_number: int = 0,
                 cycle_lengths: list = None, cycle_strategy: str = 'fixed',
                 network_path: str = None, network_dimensions: tuple = None):
        """Initialize the traffic control environment.

        Args:
            env_params_string: Raw parameter string for environment (e.g., "--network-seed 42 --grid_dimension 5 ...")
            episode_number: Episode number for reward logging (0 = auto-increment)
            cycle_lengths: List of cycle lengths in seconds (default: [90])
            cycle_strategy: How to select cycle length ('fixed', 'random', 'sequential', 'adaptive')
            network_path: Path to pre-generated network files for reuse (if None, generates fresh network each episode)
            network_dimensions: Pre-computed (edge_count, junction_count) to skip network generation during init
        """
        super().__init__()
        from .constants import DEFAULT_CYCLE_LENGTH, MIN_PHASE_DURATION

        # Parse parameter string into CLI args
        parser = create_argument_parser()
        args_list = shlex.split(env_params_string)
        self.cli_args = vars(parser.parse_args(args_list))

        # Store network path for reuse (if provided)
        self.network_path = network_path

        # Extract key parameters from CLI args
        from src.network.generate_grid import parse_junctions_to_remove

        grid_dimension = int(self.cli_args['grid_dimension'])

        # Account for junctions_to_remove in intersection count
        junctions_remove_input = self.cli_args.get('junctions_to_remove', '0')
        is_list, junction_ids, remove_count = parse_junctions_to_remove(junctions_remove_input)
        self.num_intersections = (grid_dimension * grid_dimension) - remove_count

        self.num_vehicles = self.cli_args['num_vehicles']
        self.end_time = self.cli_args['end_time']
        self.workspace = self.cli_args.get('workspace', 'workspace')

        # Cycle length management for variable duration control
        self.cycle_lengths = cycle_lengths if cycle_lengths else [DEFAULT_CYCLE_LENGTH]
        self.cycle_strategy = cycle_strategy
        self.current_cycle_length = self.cycle_lengths[0]
        self.min_phase_time = MIN_PHASE_DURATION
        self.decision_count = 0  # Track number of decisions for sequential strategy
        self.current_schedules = {}  # Store phase schedules for current cycle
        self.cycle_start_step = 0  # Track when current cycle started

        # Calculate state vector size dynamically
        # This ensures observation space matches actual edge count after edge splitting
        from .constants import (
            RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
            RL_DYNAMIC_NETWORK_FEATURES_COUNT
        )

        # Use pre-computed dimensions if provided, otherwise generate network temporarily
        if network_dimensions:
            actual_edge_count, actual_junction_count = network_dimensions
            print(f"[ENV] Using provided dimensions: {actual_edge_count} edges, {actual_junction_count} junctions", flush=True)
        else:
            actual_edge_count, actual_junction_count = self._get_actual_network_dimensions()

        self.state_vector_size = (actual_edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                                  actual_junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                                  RL_DYNAMIC_NETWORK_FEATURES_COUNT +
                                  1)  # +1 for normalized cycle_length

        # Define observation space: normalized state vector [0, 1]
        # State includes traffic features (edges) + signal features (junctions) + cycle_length
        state_size = self.state_vector_size
        self.observation_space = gym.spaces.Box(
            low=STATE_NORMALIZATION_MIN,
            # Add tolerance for floating point precision
            high=STATE_NORMALIZATION_MAX + 0.01,
            shape=(state_size,),
            dtype=np.float32
        )

        # Define action space based on control mode
        from .constants import RL_USE_CONTINUOUS_ACTIONS, RL_ACTIONS_PER_JUNCTION

        if RL_USE_CONTINUOUS_ACTIONS:
            # Duration-based control: continuous actions for phase duration proportions
            # Each intersection outputs 4 values (one per phase) that will be converted to durations via softmax
            # Shape: (num_intersections * 4,) - continuous values from neural network
            # Use finite bounds for PPO compatibility
            # Range of -10 to +10 covers practical spectrum (exp(-10) ≈ 0%, exp(+10) ≈ 100%)
            self.action_space = gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.num_intersections * RL_ACTIONS_PER_JUNCTION,),
                dtype=np.float32
            )
        elif RL_PHASE_ONLY_MODE:
            # Phase-only control: discrete actions for phase selection per intersection
            # Each intersection: [phase_id] only
            # Temporarily use default, will be updated in _start_sumo_simulation
            self.action_space = gym.spaces.MultiDiscrete(
                [NUM_TRAFFIC_LIGHT_PHASES] * self.num_intersections)
        else:
            # Legacy phase + duration control: discrete actions for phase + duration per intersection
            # Each intersection: [phase_id, duration_index]
            # Temporarily use default, will be updated in _start_sumo_simulation
            self.action_space = gym.spaces.MultiDiscrete(
                [NUM_TRAFFIC_LIGHT_PHASES, NUM_PHASE_DURATION_OPTIONS] * self.num_intersections)

        # Initialize simulation state
        self.current_step = DEFAULT_INITIAL_STEP
        self.episode_start_time = DEFAULT_INITIAL_TIME
        self.workspace_dir = None
        self.vehicle_tracker = None
        self.junction_ids = []
        self.edge_ids = []

        # TraCI connection state
        self.traci_connected = False
        self.traci_port = None  # Will be dynamically assigned

        # Tree Method traffic analyzer (initialized after network topology is known)
        self.traffic_analyzer = None

        # Reward analysis logging
        self.reward_calculator = None
        self.episode_number = episode_number  # Track episode for log filenames

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed for episode generation (CRITICAL for RL training diversity)

        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        # CRITICAL: Use the seed to create episode variation
        # Without this, all episodes are identical → all rewards are identical → no learning!
        if seed is not None:
            self.episode_seed = seed
            # Set numpy random seed for reproducibility
            np.random.seed(seed)
        else:
            # Generate random seed if not provided
            self.episode_seed = np.random.randint(0, 2**31 - 1)
            np.random.seed(self.episode_seed)

        # Close any existing TraCI connection
        if self.traci_connected:
            try:
                traci.close()
            except:
                pass
            self.traci_connected = False

        # Create workspace directory for this episode
        if self.cli_args and 'workspace' in self.cli_args:
            # Use workspace from CLI args
            self.workspace_dir = self.cli_args['workspace']
            os.makedirs(self.workspace_dir, exist_ok=True)
        else:
            # Create temporary workspace if not specified
            self.workspace_dir = tempfile.mkdtemp(prefix="rl_episode_")

        # Generate SUMO configuration using existing pipeline
        self._generate_sumo_files()

        # Start SUMO simulation with TraCI
        self._start_sumo_simulation()

        # Initialize vehicle tracker
        self.vehicle_tracker = VehicleTracker()

        # Set total vehicles expected for progressive bonus milestones
        if PROGRESSIVE_BONUS_ENABLED:
            total_vehicles = self._get_total_vehicles_expected()
            if total_vehicles > 0:
                self.vehicle_tracker.set_total_vehicles_expected(
                    total_vehicles)

        # Reset episode state
        self.current_step = DEFAULT_INITIAL_STEP
        self.episode_start_time = DEFAULT_INITIAL_TIME

        # Initialize reward analysis CSV logging
        # Always initialize for demonstration collection tracking
        # Initialize reward calculator with CSV logging
        log_filename = f"reward_analysis_episode_{self.episode_number}.csv"
        self.reward_calculator = RewardCalculator(log_file_path=log_filename)

        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"Initialized reward analysis logging: {log_filename}")

        # Get initial observation
        observation = self._get_observation()
        info = {'episode_step': self.current_step}

        return observation, info

    def step(self, action):
        """Execute one simulation step with the given action.

        Args:
            action: RL agent's action (duration proportions for all intersections if continuous,
                    or phase indices if discrete)

        Returns:
            observation: New state after action execution
            reward: Reward signal for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        if not self.traci_connected:
            raise RuntimeError("SUMO simulation not connected")

        # Apply traffic light actions (sets schedule for continuous, or applies phase for discrete)
        self._apply_traffic_light_actions(action)

        # Advance simulation by current cycle length
        from .constants import RL_USE_CONTINUOUS_ACTIONS, MEASUREMENT_INTERVAL_STEPS

        if RL_USE_CONTINUOUS_ACTIONS:
            # Duration-based control: apply scheduled phases throughout cycle
            for t in range(self.current_cycle_length):
                if self.traci_connected:
                    # Apply correct phase based on schedule and time within cycle
                    self._apply_scheduled_phases(t)

                    # Advance SUMO simulation
                    traci.simulationStep()
                    self.current_step += 1

                    # Update vehicle tracker every measurement interval
                    if self.current_step % MEASUREMENT_INTERVAL_STEPS == 0:
                        self.vehicle_tracker.update_vehicles(self.current_step)
        else:
            # Phase-only or legacy control: advance by current cycle length
            for _ in range(self.current_cycle_length):
                if self.traci_connected:
                    traci.simulationStep()
                    self.current_step += 1

                    if self.current_step % MEASUREMENT_INTERVAL_STEPS == 0:
                        self.vehicle_tracker.update_vehicles(self.current_step)

        # Select next cycle length for next decision
        if RL_USE_CONTINUOUS_ACTIONS:
            self._select_next_cycle_length()
            self.decision_count += 1

        # Get new observation
        observation = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check episode termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            'episode_step': self.current_step,
            'simulation_time': traci.simulation.getTime() if self.traci_connected else DEFAULT_INITIAL_TIME
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Collect enhanced network state using Tree Method traffic analysis.

        Returns:
            np.array: Normalized state vector containing sophisticated traffic indicators
        """
        # Check if TraCI is already connected (inference mode)
        if not self.traci_connected:
            try:
                # Try to use existing TraCI connection
                test_time = traci.simulation.getTime()
                # If this succeeds, TraCI is connected - use existing connection
                self.traci_connected = True
                self.edge_ids = traci.edge.getIDList()
                self.junction_ids = traci.trafficlight.getIDList()

                logger = logging.getLogger(self.__class__.__name__)

            except Exception:
                # TraCI not connected and we're not in training mode
                logger = logging.getLogger(self.__class__.__name__)
                logger.error(
                    f"=== NO TRACI CONNECTION AVAILABLE ===\nReturning zero observation")
                return np.zeros(self.state_vector_size, dtype=np.float32)

        # Initialize traffic analyzer on first call
        if self.traffic_analyzer is None:
            from .traffic_analysis import RLTrafficAnalyzer
            debug_mode = hasattr(self, '_debug_state') and self._debug_state
            # Get m and l parameters from CLI args
            m = self.cli_args.get('tree_method_m', 0.8)
            l = self.cli_args.get('tree_method_l', 2.8)
            self.traffic_analyzer = RLTrafficAnalyzer(
                self.edge_ids, m, l, debug=debug_mode)

        observation = []

        # Enhanced edge features (6 features per edge using Tree Method analysis)
        edge_count = 0
        edges_processed = []
        for edge_id in self.edge_ids:
            if ':' in edge_id:  # Skip internal edges
                continue
            edge_features = self.traffic_analyzer.get_enhanced_edge_features(
                edge_id)
            observation.extend(edge_features)
            edges_processed.append(edge_id)
            edge_count += 1

        # Enhanced junction features (2 features per junction)
        junction_count = 0
        junction_features_total = []
        for junction_id in self.junction_ids:
            junction_features = self.traffic_analyzer.get_enhanced_junction_features(
                junction_id)
            junction_features_total.extend(junction_features)
            observation.extend(junction_features)
            junction_count += 1

        # Network-level features (5 features)
        network_features = self.traffic_analyzer.get_network_level_features()
        observation.extend(network_features)

        # Cycle length feature (1 feature) - normalized to [0, 1] range
        from .constants import MAX_CYCLE_LENGTH
        normalized_cycle_length = self.current_cycle_length / MAX_CYCLE_LENGTH
        observation.append(normalized_cycle_length)

        # Debug logging for feature construction (only when debug mode is enabled)
        if hasattr(self, '_debug_state') and self._debug_state and self.current_step % 10 == 0:  # Log every 10 steps
            logger = logging.getLogger(self.__class__.__name__)

        # Ensure correct size and return
        expected_size = self.state_vector_size
        if len(observation) < expected_size:
            observation.extend([DEFAULT_OBSERVATION_PADDING]
                               * (expected_size - len(observation)))
        elif len(observation) > expected_size:
            observation = observation[:expected_size]

        # Enhanced state validation and debugging
        observation_array = np.array(observation, dtype=np.float32)

        # Log detailed inspection for first few observations or if debug mode enabled
        if (hasattr(self, '_debug_state') and self._debug_state) or self.current_step < 30:
            if hasattr(self.traffic_analyzer, 'log_detailed_inspection'):
                # Pass actual counts for accurate inspection
                actual_edge_count = len(
                    [e for e in self.edge_ids if ':' not in e])
                actual_junction_count = len(self.junction_ids)

                # Debug: Log the actual counts being used
                # Update the inspection function to pass the correct counts
                inspection = self.traffic_analyzer.inspect_state_vector(
                    observation, actual_edge_count, actual_junction_count)
                if self.current_step < 3:  # Only log detailed inspection for first few steps
                    self.traffic_analyzer.log_detailed_inspection_from_dict(
                        inspection)

        # Validate observation quality
        non_zero_count = np.count_nonzero(observation_array)
        if non_zero_count == 0 and self.current_step > 10:
            logger = logging.getLogger(self.__class__.__name__)
            logger.warning(
                f"Step {self.current_step}: All {len(observation)} features are zero - possible data collection issue")

        return observation_array

    def _calc_k_by_u(self, current_speed_km_h, free_flow_speed_km_h):
        """Calculate density from speed using Greenshields model (Tree Method formula).

        Args:
            current_speed_km_h: Current speed in km/h
            free_flow_speed_km_h: Free-flow speed in km/h

        Returns:
            float: Density in vehicles/km/lane
        """
        return max(
            round(MAX_DENSITY * ((max(1 - (current_speed_km_h / free_flow_speed_km_h), 0) ** (1 - M)) ** (1 / (L - 1)))),
            0
        )

    def _calc_u_by_k(self, current_density, free_flow_speed_km_h):
        """Calculate speed from density using Greenshields model (Tree Method formula).

        Args:
            current_density: Current density in vehicles/km/lane
            free_flow_speed_km_h: Free-flow speed in km/h

        Returns:
            float: Speed in km/h
        """
        return max(
            round(free_flow_speed_km_h * ((1 - (current_density / MAX_DENSITY) ** (L - 1)) ** (1 / (1 - M)))),
            MIN_VELOCITY
        )

    def _calculate_q_max_u(self, free_flow_speed_km_h, num_lanes):
        """Calculate optimal flow speed (q_max_u) using Tree Method's algorithm.

        This is the speed at which traffic flow is maximized for this edge.

        Args:
            free_flow_speed_km_h: Free-flow speed in km/h
            num_lanes: Number of lanes

        Returns:
            float: Optimal flow speed in km/h
        """
        q_max = 0
        q_max_u = -1

        for k in range(MAX_DENSITY):
            u = self._calc_u_by_k(k, free_flow_speed_km_h)
            q = u * k * num_lanes
            if q > q_max:
                q_max = q
                q_max_u = u

        return q_max_u

    def _compute_reward(self):
        """Empirically validated reward based on comparative analysis.

        Uses z-score normalization and Cohen's d-based weights from
        Tree Method vs Fixed timing comparative study.

        Updated 2025-12-07: Removed broken travel time metric, added throughput bonus.

        Returns:
            float: Reward value (higher is better, expected range -1 to +2)
        """
        if not self.vehicle_tracker or not self.reward_calculator:
            return DEFAULT_FALLBACK_VALUE

        # Collect the 3 empirically validated metrics (travel time removed - broken)
        avg_waiting_per_vehicle = self._get_avg_waiting_per_vehicle()
        avg_speed_kmh = self._get_avg_speed_kmh()
        avg_queue_length = self._get_avg_queue_length()

        # Get throughput for bonus (vehicles that completed their trips this step)
        vehicles_arrived = len(traci.simulation.getArrivedIDList())

        # Compute empirical reward using validated function
        reward = self.reward_calculator.compute_empirical_reward(
            avg_waiting_per_vehicle=avg_waiting_per_vehicle,
            avg_speed_kmh=avg_speed_kmh,
            avg_queue_length=avg_queue_length,
            vehicles_arrived_this_step=vehicles_arrived
        )

        # Logging (every 100 steps) - Optional for analysis
        if self.current_step % 100 == 0:
            logger = logging.getLogger(self.__class__.__name__)
            logger.debug(
                f"Step {self.current_step}: "
                f"waiting={avg_waiting_per_vehicle:.2f}s, "
                f"speed={avg_speed_kmh:.2f}km/h, "
                f"queue={avg_queue_length:.2f}, "
                f"arrived={vehicles_arrived}, "
                f"reward={reward:.4f}"
            )

        return reward

    def _get_avg_waiting_per_vehicle(self) -> float:
        """Get average waiting time per vehicle (seconds).

        Returns:
            float: Average waiting time across all active vehicles
        """
        if not self.traci_connected:
            return 0.0

        try:
            active_vehicles = traci.vehicle.getIDList()
            if not active_vehicles:
                return 0.0

            total_waiting_time = 0.0
            for vehicle_id in active_vehicles:
                try:
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    total_waiting_time += waiting_time
                except Exception:
                    continue

            return total_waiting_time / len(active_vehicles)
        except Exception:
            return 0.0

    def _get_avg_speed_kmh(self) -> float:
        """Get average network speed (km/h).

        Returns:
            float: Average speed across all active vehicles
        """
        if not self.traci_connected:
            return 0.0

        try:
            active_vehicles = traci.vehicle.getIDList()
            if not active_vehicles:
                return 0.0

            total_speed = 0.0
            for vehicle_id in active_vehicles:
                try:
                    speed_ms = traci.vehicle.getSpeed(vehicle_id)
                    total_speed += speed_ms * 3.6  # Convert m/s to km/h
                except Exception:
                    continue

            return total_speed / len(active_vehicles)
        except Exception:
            return 0.0

    def _get_avg_queue_length(self) -> float:
        """Get average queue length (vehicles) across all controlled lanes.

        Matches MetricLogger's lane-level calculation for proper normalization.

        Returns:
            float: Average number of halting vehicles per lane
        """
        if not self.traci_connected:
            return 0.0

        try:
            total_halting = 0.0
            lane_count = 0

            # Iterate through controlled lanes (matching MetricLogger)
            for junction_id in self.junction_ids:
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
                    for lane_id in controlled_lanes:
                        halting_count = traci.lane.getLastStepHaltingNumber(lane_id)
                        total_halting += halting_count
                        lane_count += 1
                except Exception:
                    continue

            if lane_count == 0:
                return 0.0

            return total_halting / lane_count
        except Exception:
            return 0.0

    def _get_avg_edge_travel_time(self) -> float:
        """Get average edge travel time (seconds) across all edges.

        Uses SUMO's built-in getTraveltime() to match MetricLogger.

        Returns:
            float: Average travel time per edge
        """
        if not self.traci_connected:
            return 0.0

        try:
            total_travel_time = 0.0
            edge_count = 0

            for edge_id in self.edge_ids:
                if ':' in edge_id:  # Skip internal edges
                    continue

                try:
                    # Use SUMO's built-in travel time (matches MetricLogger)
                    travel_time = traci.edge.getTraveltime(edge_id)
                    total_travel_time += travel_time
                    edge_count += 1
                except Exception:
                    continue

            if edge_count == 0:
                return 0.0

            return total_travel_time / edge_count
        except Exception:
            return 0.0

    def _is_terminated(self):
        """Check if episode should end naturally.

        Returns:
            bool: True if simulation completed successfully
        """
        if not self.traci_connected:
            return True

        # Check if simulation time reached configured end_time
        current_time = traci.simulation.getTime()
        if current_time >= self.end_time:
            return True

        # Check if all vehicles have completed their journeys
        try:
            active_vehicles = traci.vehicle.getIDList()
            departed_vehicles = traci.simulation.getDepartedIDList()
            arrived_vehicles = traci.simulation.getArrivedIDList()

            # Episode ends when no more vehicles are active and no more will depart
            if len(active_vehicles) == 0 and len(departed_vehicles) == 0:
                return True

        except Exception:
            # If we can't get vehicle info, assume episode should end
            return True

        return False

    def _is_truncated(self):
        """Check if episode should be truncated early.

        Returns:
            bool: True if episode should be cut short (timeout, error, etc.)
        """
        if not self.traci_connected:
            return True

        # Check for SUMO error conditions
        try:
            # Test if TraCI is still responsive
            _ = traci.simulation.getTime()
        except Exception:
            # TraCI connection lost or error occurred
            return True

        # Check for excessive episode length (safety check)
        if self.current_step > self.end_time * 2:
            return True

        return False

    def get_final_statistics(self) -> Dict[str, float]:
        """Get final simulation statistics for reward function evaluation.

        Returns:
            Dict containing:
                - completion_rate: Fraction of vehicles that completed their journey
                - avg_waiting_time: Average waiting time per vehicle (seconds)
                - avg_time_loss: Average time loss per vehicle (seconds)
                - throughput: Vehicles per hour
                - vehicles_arrived: Total number of vehicles that arrived
                - vehicles_loaded: Total number of vehicles loaded into simulation
                - vehicles_inserted: Total number of vehicles that departed
                - insertion_rate: Fraction of loaded vehicles that were inserted
        """
        stats = {
            'completion_rate': 0.0,
            'avg_waiting_time': 0.0,
            'avg_time_loss': 0.0,
            'throughput': 0.0,
            'vehicles_arrived': 0,
            'vehicles_loaded': 0,
            'vehicles_inserted': 0,
            'insertion_rate': 0.0
        }

        if not self.traci_connected:
            return stats

        try:
            # Get vehicle counts
            arrived_list = traci.simulation.getArrivedIDList()
            departed_list = traci.simulation.getDepartedIDList()
            loaded_count = traci.simulation.getLoadedNumber()
            departed_count = traci.simulation.getDepartedNumber()
            arrived_count = traci.simulation.getArrivedNumber()

            # Basic counts
            stats['vehicles_loaded'] = loaded_count
            stats['vehicles_inserted'] = departed_count
            stats['vehicles_arrived'] = arrived_count

            # Completion rate: arrived / loaded
            if loaded_count > 0:
                stats['completion_rate'] = arrived_count / loaded_count

            # Insertion rate: departed / loaded
            if loaded_count > 0:
                stats['insertion_rate'] = departed_count / loaded_count

            # Throughput: vehicles per hour
            simulation_time_hours = traci.simulation.getTime() / 3600.0
            if simulation_time_hours > 0:
                stats['throughput'] = arrived_count / simulation_time_hours

            # Average waiting time and time loss (from arrived vehicles)
            total_waiting_time = 0.0
            total_time_loss = 0.0

            # Get trip info from vehicle tracker if available
            if self.vehicle_tracker and len(self.vehicle_tracker.completed_vehicles) > 0:
                for vehicle_id in self.vehicle_tracker.completed_vehicles:
                    if vehicle_id in self.vehicle_tracker.vehicle_histories:
                        # Use tracked data if available
                        vehicle_data = self.vehicle_tracker.vehicle_histories[vehicle_id]
                        total_waiting_time += vehicle_data.get(
                            'last_waiting_time', 0.0)

                # Average
                completed_count = len(self.vehicle_tracker.completed_vehicles)
                if completed_count > 0:
                    stats['avg_waiting_time'] = total_waiting_time / \
                        completed_count

            # For time loss, we'd need trip info from SUMO - use approximation
            # Time loss ≈ waiting time (rough estimate)
            stats['avg_time_loss'] = stats['avg_waiting_time']

        except Exception as e:
            logger = logging.getLogger(self.__class__.__name__)
            logger.warning(f"Error collecting final statistics: {e}")

        return stats


    def close(self):
        """Clean up environment resources."""
        # Close reward calculator (which closes the log file)
        if self.reward_calculator:
            try:
                self.reward_calculator.close()
                logger = logging.getLogger(self.__class__.__name__)
                logger.info(
                    f"Closed reward analysis log for episode {self.episode_number}")
            except:
                pass

        if self.traci_connected:
            try:
                traci.close()
            except:
                pass
            self.traci_connected = False

        # Clean up temporary workspace if created
        if self.workspace_dir and self.workspace_dir.startswith('/tmp'):
            try:
                shutil.rmtree(self.workspace_dir)
            except:
                pass

    def _generate_sumo_files(self):
        """Generate SUMO network and route files using existing pipeline.

        If network_path is set, reuses pre-generated network files and only
        generates routes (Steps 6-7). Otherwise, runs the full pipeline (Steps 1-7).
        """
        # Create args object from stored CLI args
        if self.cli_args is None:
            raise ValueError(
                "Unable to generate CLI args - no parameters provided")

        # Convert dict to Namespace object
        args = argparse.Namespace(**self.cli_args)

        # CRITICAL FIX: Override seeds with episode seed for episode variation
        # Each episode needs different traffic patterns for meaningful learning!
        # However, respect explicitly set seeds from CLI (for demonstration collection)
        if hasattr(self, 'episode_seed'):
            # Use episode seed to generate deterministic but varied traffic per episode
            # Offset each seed type so network/private/public traffic vary independently
            # ONLY override if seeds were not explicitly set (None means not set)
            if getattr(args, 'network_seed', None) is None:
                args.network_seed = self.episode_seed
            if getattr(args, 'private_traffic_seed', None) is None:
                args.private_traffic_seed = self.episode_seed + 1000
            if getattr(args, 'public_traffic_seed', None) is None:
                args.public_traffic_seed = self.episode_seed + 2000

        # DEBUG: Log actual seeds being used (use print for reliable output)
        logger = logging.getLogger(__name__)
        print(f"[SUMO FILES] Seeds: network={getattr(args, 'network_seed', 'None')}, private={getattr(args, 'private_traffic_seed', 'None')}, public={getattr(args, 'public_traffic_seed', 'None')}", flush=True)

        # Check network reuse with print() for reliable output
        print(f"[SUMO FILES] network_path='{self.network_path}'", flush=True)

        # If network_path is set, use network reuse (it MUST exist, we verified at generation time)
        if self.network_path:
            net_file = os.path.join(self.network_path, "grid.net.xml")
            dir_exists = os.path.exists(self.network_path)
            file_exists = os.path.exists(net_file)
            print(f"[SUMO FILES] dir_exists={dir_exists}, file_exists={file_exists}", flush=True)

            if file_exists:
                print(f"[SUMO FILES] ✅ Reusing network from: {self.network_path}", flush=True)
                self._generate_with_network_reuse(args)
                return
            else:
                # Fatal error - network_path set but file missing
                print(f"[SUMO FILES] ERROR: Network path set but grid.net.xml missing!", flush=True)
                if dir_exists:
                    contents = os.listdir(self.network_path)
                    print(f"[SUMO FILES] Dir contents: {contents}", flush=True)
                raise RuntimeError(f"Network path set but grid.net.xml missing: {net_file}")

        # Only run full pipeline if no network_path (shouldn't happen in normal RL training)
        print(f"[SUMO FILES] ⚠️ No network_path, running FULL pipeline (Steps 1-7)...", flush=True)
        pipeline = PipelineFactory.create_pipeline(args)
        if hasattr(pipeline, 'execute_file_generation_only'):
            pipeline.execute_file_generation_only()
        else:
            # Fallback for other pipeline types (e.g., SamplePipeline)
            pipeline.execute()

    def _generate_with_network_reuse(self, args):
        """Generate SUMO files reusing pre-generated network (Steps 6-7 only).

        Copies network files from network_path to workspace, then generates
        only vehicle routes and SUMO configuration.

        Args:
            args: Namespace with simulation parameters
        """
        from src.config import CONFIG
        from src.constants import COMPARISON_NETWORK_FILES
        from src.traffic.builder import execute_route_generation
        from src.sumo_integration.sumo_utils import execute_config_generation

        logger = logging.getLogger(__name__)

        # Update CONFIG to use current workspace
        CONFIG.update_workspace(args.workspace)

        # Create workspace directory
        actual_workspace = os.path.join(args.workspace, 'workspace')
        os.makedirs(actual_workspace, exist_ok=True)

        # Copy network files from pre-generated network path
        network_files_copied = 0
        optional_files = ["zones.geojson", "attractiveness_phases.json"]
        all_files = list(COMPARISON_NETWORK_FILES) + optional_files

        for filename in all_files:
            src_file = os.path.join(self.network_path, filename)
            dst_file = os.path.join(actual_workspace, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                network_files_copied += 1
                logger.debug(f"Copied {filename} to workspace")

        logger.info(f"📁 Copied {network_files_copied} network files to workspace")

        # Step 6: Generate vehicle routes with episode-specific seeds
        logger.info("🚗 Generating vehicle routes (Step 6)")
        execute_route_generation(args)

        # Step 7: Generate SUMO configuration
        logger.info("⚙️ Generating SUMO configuration (Step 7)")
        execute_config_generation(args)

    def _start_sumo_simulation(self):
        """Start SUMO simulation with TraCI connection."""
        # Find the generated SUMO config file
        # Note: CONFIG.update_workspace() already appends '/workspace' to the base path,
        # so self.workspace_dir already points to the correct workspace directory
        # (e.g., "models/rl_20251009_111047" becomes "models/rl_20251009_111047/workspace" via CONFIG)

        # However, self.workspace_dir still has the original value (before CONFIG added /workspace)
        # So we need to look in workspace_dir/workspace for the actual files
        actual_workspace = os.path.join(self.workspace_dir, 'workspace')

        sumo_config = None
        if os.path.exists(actual_workspace):
            for file in os.listdir(actual_workspace):
                if file.endswith('.sumocfg'):
                    sumo_config = os.path.join(actual_workspace, file)
                    break

        if not sumo_config:
            raise RuntimeError(
                f"No SUMO config file found in {actual_workspace}")

        # Get a free port for this environment
        if self.traci_port is None:
            self.traci_port = self._get_free_port()

        # Start SUMO with TraCI using unique port
        # Don't specify output files - let SUMO use defaults or config file settings
        sumo_cmd = [
            'sumo',
            '-c', sumo_config,
            '--start',
            '--quit-on-end',
            '--no-step-log',
            '--no-warnings',
        ]

        # Add SUMO random seed if episode_seed is set
        # This ensures vehicle behavior (lane changing, speed variation) differs between episodes
        if hasattr(self, 'episode_seed') and self.episode_seed is not None:
            sumo_cmd.extend(['--seed', str(self.episode_seed)])
            logger = logging.getLogger(__name__)
            logger.info(f"🎲 SUMO random seed set to: {self.episode_seed}")

        traci.start(sumo_cmd, port=self.traci_port)
        self.traci_connected = True

        # Get network topology information
        self.junction_ids = traci.trafficlight.getIDList()
        self.edge_ids = traci.edge.getIDList()

        # Cache edge properties for cost-based reward calculation
        # These are static properties that don't change during simulation
        self.edge_properties = {}
        for edge_id in self.edge_ids:
            # Get lane information (use first lane for length and max speed since all lanes on an edge share these)
            lane_id = f"{edge_id}_0"  # First lane ID
            try:
                self.edge_properties[edge_id] = {
                    'lanes': traci.edge.getLaneNumber(edge_id),
                    'length': traci.lane.getLength(lane_id),  # meters
                    'free_flow_speed': traci.lane.getMaxSpeed(lane_id) * 3.6  # m/s to km/h
                }
            except Exception as e:
                # If lane doesn't exist (shouldn't happen), skip this edge
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not cache properties for edge {edge_id}: {e}")

        # Read actual phase counts for each junction
        self.junction_phase_counts = {}
        for junction_id in self.junction_ids:
            logic = traci.trafficlight.getAllProgramLogics(junction_id)
            if logic and len(logic) > 0:
                num_phases = len(logic[0].phases)
                self.junction_phase_counts[junction_id] = num_phases

        # Update action space with actual traffic light phases
        self._update_action_space_from_sumo()

        # Observation space already set correctly during __init__ by _get_actual_network_dimensions()
        # No need to update it here

    def _apply_traffic_light_actions(self, actions):
        """Apply RL actions to traffic lights.

        For continuous actions: converts duration proportions to schedules
        For discrete actions: applies phase immediately
        """
        if not self.traci_connected:
            return

        from .constants import RL_USE_CONTINUOUS_ACTIONS, RL_ACTIONS_PER_JUNCTION

        # Debug: Log action shape for troubleshooting
        expected_size = len(self.junction_ids) * RL_ACTIONS_PER_JUNCTION
        if len(actions) != expected_size:
            raise ValueError(
                f"Action size mismatch! "
                f"Received: {len(actions)}, Expected: {expected_size} "
                f"(junctions: {len(self.junction_ids)}, actions_per_junction: {RL_ACTIONS_PER_JUNCTION}). "
                f"This indicates the action space was not properly initialized."
            )

        # Record actions for vehicle tracking
        action_dict = {}

        if RL_USE_CONTINUOUS_ACTIONS:
            # Duration-based control: convert raw actions to duration schedules
            junction_schedules = {}

            for i, junction_id in enumerate(self.junction_ids):
                # Extract 4 values for this junction
                start_idx = i * RL_ACTIONS_PER_JUNCTION
                end_idx = start_idx + RL_ACTIONS_PER_JUNCTION
                raw_outputs = actions[start_idx:end_idx]

                # Apply softmax to get proportions
                proportions = self._softmax(raw_outputs)

                # Convert to durations summing to current_cycle_length
                durations = self._proportions_to_durations(
                    proportions,
                    self.current_cycle_length,
                    self.min_phase_time
                )

                junction_schedules[junction_id] = durations

                # Record first phase for vehicle tracking
                action_dict[junction_id] = (0, durations[0])

            # Store schedule for application during cycle
            self.current_schedules = junction_schedules
            self.cycle_start_step = self.current_step

        elif RL_PHASE_ONLY_MODE:
            # Phase-only mode: discrete phase selection
            for i, junction_id in enumerate(self.junction_ids):
                if i < len(actions):
                    phase_idx = int(actions[i])
                    duration = RL_FIXED_PHASE_DURATION

                    try:
                        traci.trafficlight.setPhase(junction_id, phase_idx)
                        traci.trafficlight.setPhaseDuration(
                            junction_id, duration)
                    except Exception:
                        pass

                    action_dict[junction_id] = (phase_idx, duration)
        else:
            # Legacy mode: phase + duration pairs
            action_pairs = actions.reshape(
                self.num_intersections, ACTIONS_PER_INTERSECTION)

            for i, junction_id in enumerate(self.junction_ids):
                phase_idx, duration_idx = action_pairs[i]
                duration = PHASE_DURATION_OPTIONS[int(duration_idx)]

                try:
                    traci.trafficlight.setPhase(junction_id, int(phase_idx))
                    traci.trafficlight.setPhaseDuration(junction_id, duration)
                except Exception:
                    pass

                action_dict[junction_id] = (int(phase_idx), duration)

        # Record decision for vehicle tracking
        self.vehicle_tracker.record_decision(self.current_step, action_dict)

    def _softmax(self, x):
        """Numerically stable softmax.

        Args:
            x: Array of raw values

        Returns:
            Array of probabilities summing to 1.0
        """
        # Handle empty array case (should not happen in normal operation)
        if len(x) == 0:
            raise ValueError(
                f"Cannot apply softmax to empty array. "
                f"Action shape: {len(x)}, Expected: {len(self.junction_ids) * 4}. "
                f"This indicates an action space dimension mismatch."
            )

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
        durations = [min_phase_time + (p * available_time) for p in proportions]

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
        if not self.current_schedules:
            return

        for junction_id, durations in self.current_schedules.items():
            # Get actual number of phases for this junction
            actual_num_phases = self.junction_phase_counts.get(junction_id, 4)

            # Only use the durations for phases that actually exist
            valid_durations = durations[:actual_num_phases]

            # Find which phase should be active at this time
            elapsed = 0
            for phase_idx, duration in enumerate(valid_durations):
                if time_in_cycle < elapsed + duration:
                    # This is the active phase - apply it
                    try:
                        traci.trafficlight.setPhase(junction_id, phase_idx)
                        traci.trafficlight.setPhaseDuration(junction_id, duration)
                    except Exception:
                        pass
                    break
                elapsed += duration

    def _select_next_cycle_length(self):
        """Select cycle length for next decision based on strategy."""
        if len(self.cycle_lengths) == 1:
            # Fixed cycle length
            return

        import random

        if self.cycle_strategy == 'random':
            self.current_cycle_length = random.choice(self.cycle_lengths)
        elif self.cycle_strategy == 'sequential':
            idx = self.decision_count % len(self.cycle_lengths)
            self.current_cycle_length = self.cycle_lengths[idx]
        elif self.cycle_strategy == 'adaptive':
            # Future enhancement: adapt based on traffic conditions
            # For now, use random
            self.current_cycle_length = random.choice(self.cycle_lengths)
        # else: 'fixed' - do nothing, keep current

    def _get_current_avg_waiting_time(self) -> float:
        """Get current average waiting time across all vehicles.

        Returns:
            float: Average waiting time in seconds
        """
        if not self.traci_connected:
            return 0.0

        try:
            active_vehicles = traci.vehicle.getIDList()
            if not active_vehicles:
                return 0.0

            total_waiting_time = 0.0
            for vehicle_id in active_vehicles:
                try:
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    total_waiting_time += waiting_time
                except Exception:
                    continue

            return total_waiting_time / len(active_vehicles)
        except Exception:
            return 0.0

    def _get_network_performance_metrics(self) -> Tuple[float, int]:
        """Get current network performance metrics for progressive bonuses.

        Returns:
            Tuple[float, int]: (average_speed_kmh, bottleneck_count)
        """
        if not self.traci_connected:
            return 0.0, 0

        try:
            # Calculate average network speed
            total_speed = 0.0
            vehicle_count = 0
            active_vehicles = traci.vehicle.getIDList()

            for vehicle_id in active_vehicles:
                try:
                    speed_ms = traci.vehicle.getSpeed(vehicle_id)
                    total_speed += speed_ms * 3.6  # Convert to km/h
                    vehicle_count += 1
                except Exception:
                    continue

            avg_speed = total_speed / max(vehicle_count, 1)

            # Get bottleneck count from traffic analyzer if available
            bottleneck_count = 0
            if hasattr(self, 'traffic_analyzer') and self.traffic_analyzer:
                # Count edges that are detected as bottlenecks
                for edge_id in self.edge_ids:
                    if ':' in edge_id or edge_id not in self.traffic_analyzer.edge_links:
                        continue

                    try:
                        link = self.traffic_analyzer.edge_links[edge_id]
                        current_speed = traci.edge.getLastStepMeanSpeed(
                            edge_id) * 3.6
                        if current_speed < link.q_max_properties.q_max_u:
                            bottleneck_count += 1
                    except Exception:
                        continue

            return avg_speed, bottleneck_count

        except Exception:
            return 0.0, 0

    def _get_total_vehicles_expected(self) -> int:
        """Get the total number of vehicles expected in this simulation.

        Returns:
            int: Total number of vehicles expected
        """
        # Return vehicle count from initialization
        return self.num_vehicles

    def _get_free_port(self):
        """Get a free port for TraCI connection."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _update_action_space_from_sumo(self):
        """Update action space based on actual traffic light phases from SUMO."""
        if not self.traci_connected:
            return

        # Skip action space update for continuous actions (duration-based control)
        # Continuous actions use fixed shape (num_junctions * 4) regardless of actual phases
        from .constants import RL_USE_CONTINUOUS_ACTIONS
        if RL_USE_CONTINUOUS_ACTIONS:
            return

        logger = logging.getLogger(self.__class__.__name__)

        logger.info(f"=== DEBUGGING ACTION SPACE UPDATE ===")
        # logger.info(f"Available junction IDs: {self.junction_ids}")
        # logger.info(f"Number of junctions: {len(self.junction_ids)}")

        # Also check what TraCI sees as traffic lights
        try:
            all_tls = traci.trafficlight.getIDList()
            # logger.info(f"All traffic lights from TraCI: {all_tls}")
        except Exception as e:
            logger.error(f"Failed to get traffic light IDs from TraCI: {e}")
            raise RuntimeError(f"Cannot access TraCI traffic lights: {e}")

        actual_phase_counts = []
        for i, junction_id in enumerate(self.junction_ids):
            try:
                # Get the traffic light logic
                logic = traci.trafficlight.getAllProgramLogics(junction_id)

                if not logic or len(logic) == 0:
                    raise RuntimeError(
                        f"Junction {junction_id}: No traffic light logic found")

                first_program = logic[0]
                if not hasattr(first_program, 'phases'):
                    raise RuntimeError(
                        f"Junction {junction_id}: Logic program has no phases attribute")

                phases = first_program.phases
                num_phases = len(phases)

                if num_phases == 0:
                    raise RuntimeError(
                        f"Junction {junction_id}: Zero phases found")

                actual_phase_counts.append(num_phases)
                # logger.info(f"✓ Junction {junction_id}: Successfully added {num_phases} phases")

            except Exception as e:
                logger.error(
                    f"✗ CRITICAL ERROR processing junction {junction_id}: {e}")
                raise RuntimeError(
                    f"Failed to get phase data for junction {junction_id}: {e}. NO FALLBACKS ALLOWED - SYSTEM MUST HAVE ACCURATE TRAFFIC LIGHT DATA")

        if not actual_phase_counts:
            raise RuntimeError(
                "No valid junctions found - SYSTEM CANNOT PROCEED WITHOUT TRAFFIC LIGHT DATA")

        # Update action space with actual phase counts
        from gymnasium.spaces import MultiDiscrete
        self.action_space = MultiDiscrete(actual_phase_counts)
        # logger.info(f"✓ SUCCESS: Updated action space to: {self.action_space}")
        # logger.info(f"✓ Phase counts per junction: {actual_phase_counts}")
        logger.info(f"=== ACTION SPACE UPDATE COMPLETE ===")

        # Validate that we have variation in phase counts (not all the same)
        unique_counts = set(actual_phase_counts)
        if len(unique_counts) == 1 and list(unique_counts)[0] == 4:
            logger.warning(
                f"⚠️  ALL JUNCTIONS HAVE EXACTLY 4 PHASES - This may indicate a systematic issue")
        else:
            logger.info(
                f"✓ Action space has variation: unique phase counts = {sorted(unique_counts)}")

    def _update_observation_space_from_sumo(self):
        """Update observation space based on actual network dimensions from SUMO."""
        if not self.traci_connected:
            return

        logger = logging.getLogger(self.__class__.__name__)

        try:
            # Calculate actual state vector size based on SUMO network
            actual_edge_count = len([e for e in self.edge_ids if ':' not in e])
            actual_junction_count = len(self.junction_ids)

            from .constants import (
                RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
                RL_DYNAMIC_NETWORK_FEATURES_COUNT, STATE_NORMALIZATION_MIN, STATE_NORMALIZATION_MAX
            )

            actual_state_size = (
                actual_edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                actual_junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                RL_DYNAMIC_NETWORK_FEATURES_COUNT +
                1  # +1 for normalized cycle_length feature
            )

            # Update observation space with actual dimensions
            import gymnasium as gym
            self.observation_space = gym.spaces.Box(
                low=STATE_NORMALIZATION_MIN,
                # Add tolerance for floating point precision
                high=STATE_NORMALIZATION_MAX + 0.01,
                shape=(actual_state_size,),
                dtype='float32'
            )

            logger.info(
                f"Updated observation space to: {actual_state_size} features")
            logger.info(
                f"  Edge features: {actual_edge_count} × {RL_DYNAMIC_EDGE_FEATURES_COUNT} = {actual_edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT}")
            logger.info(
                f"  Junction features: {actual_junction_count} × {RL_DYNAMIC_JUNCTION_FEATURES_COUNT} = {actual_junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT}")
            logger.info(
                f"  Network features: {RL_DYNAMIC_NETWORK_FEATURES_COUNT}")
            logger.info(
                f"  Cycle length feature: 1")

        except Exception as e:
            logger.error(f"Failed to update observation space from SUMO: {e}")
            logger.info("Keeping default observation space")

    def _parse_network_dimensions_from_file(self, net_file: str):
        """Parse edge/junction counts from existing network XML file.

        Args:
            net_file: Path to the network XML file

        Returns:
            Tuple[int, int]: (actual_edge_count, actual_junction_count)
        """
        import xml.etree.ElementTree as ET
        logger = logging.getLogger(self.__class__.__name__)

        tree = ET.parse(net_file)
        root = tree.getroot()

        # Count edges (exclude internal edges with ':')
        all_edges = root.findall('.//edge')
        actual_edges = [e for e in all_edges if ':' not in e.get('id', '')]
        actual_edge_count = len(actual_edges)

        # Count traffic lights (junctions with traffic lights)
        junctions = root.findall(".//junction[@type='traffic_light']")
        actual_junction_count = len(junctions)

        logger.info(f"Parsed network dimensions from: {net_file}")
        logger.info(f"  Edges: {actual_edge_count}")
        logger.info(f"  Junctions: {actual_junction_count}")

        return actual_edge_count, actual_junction_count

    def _get_actual_network_dimensions(self):
        """
        Get actual edge and junction counts from the network.

        First checks if a pre-generated network path was provided. Then checks if
        network files already exist in the workspace. If neither, generates network temporarily.

        This is needed because edge splitting creates many more edges than the basic
        grid formula predicts.

        Returns:
            Tuple[int, int]: (actual_edge_count, actual_junction_count)
        """
        logger = logging.getLogger(self.__class__.__name__)

        # Use print() for reliable output (logging gets lost in subprocess noise)
        print(f"[DIM CHECK] network_path='{self.network_path}'", flush=True)

        # If network_path was provided, it MUST exist (we verified at generation time)
        if self.network_path:
            net_file = os.path.join(self.network_path, "grid.net.xml")
            dir_exists = os.path.exists(self.network_path)
            file_exists = os.path.exists(net_file)
            print(f"[DIM CHECK] dir_exists={dir_exists}, file_exists={file_exists}", flush=True)

            if file_exists:
                print(f"[DIM CHECK] ✅ Using pre-generated network: {net_file}", flush=True)
                try:
                    return self._parse_network_dimensions_from_file(net_file)
                except Exception as e:
                    logger.warning(f"Failed to parse pre-generated network: {e}")
                    print(f"[DIM CHECK] WARNING: Parse failed: {e}", flush=True)
            else:
                # This should not happen - network_path was set but file doesn't exist
                print(f"[DIM CHECK] ERROR: network_path set but file missing!", flush=True)
                if dir_exists:
                    contents = os.listdir(self.network_path)
                    print(f"[DIM CHECK] Dir contents: {contents}", flush=True)

        # SECOND: Check if network files already exist in current workspace
        from src.config import CONFIG
        if CONFIG.network_file and Path(CONFIG.network_file).exists():
            print(f"[DIM CHECK] Using existing workspace network: {CONFIG.network_file}", flush=True)
            logger.info(f"Using existing workspace network: {CONFIG.network_file}")
            try:
                return self._parse_network_dimensions_from_file(CONFIG.network_file)
            except Exception as e:
                logger.warning(f"Failed to parse existing network XML, will regenerate: {e}")

        # Network doesn't exist yet - need to generate temporarily
        print(f"[DIM CHECK] ⚠️ No network found, generating temporarily...", flush=True)
        logger.info("Generating network temporarily for dimension check")
        temp_dir = tempfile.mkdtemp(prefix="rl_network_count_")

        try:
            # Save current workspace
            original_workspace = self.workspace

            # Use temporary workspace
            self.cli_args['workspace'] = temp_dir

            # Update global CONFIG to point to temporary workspace
            from src.config import CONFIG
            original_config_output_dir = CONFIG._output_dir
            CONFIG.update_workspace(temp_dir)

            # Create the workspace subdirectory
            workspace_subdir = Path(temp_dir) / "workspace"
            workspace_subdir.mkdir(parents=True, exist_ok=True)

            # Ensure all seed-related attributes exist
            if 'seed' not in self.cli_args:
                self.cli_args['seed'] = None
            if 'network_seed' not in self.cli_args:
                self.cli_args['network_seed'] = None
            if 'private_traffic_seed' not in self.cli_args:
                self.cli_args['private_traffic_seed'] = None
            if 'public_traffic_seed' not in self.cli_args:
                self.cli_args['public_traffic_seed'] = None

            # FIX: Ensure junctions_to_remove exists with proper default
            if 'junctions_to_remove' not in self.cli_args:
                self.cli_args['junctions_to_remove'] = '0'  # Default: no junctions removed

            # Convert dict to namespace object for pipeline steps
            import argparse
            args_namespace = argparse.Namespace(**self.cli_args)

            # Run pipeline steps individually to generate network
            from src.pipeline.steps.network_generation_step import NetworkGenerationStep
            from src.pipeline.steps.zone_generation_step import ZoneGenerationStep
            from src.network.split_edges_with_lanes import execute_edge_splitting
            from src.sumo_integration.sumo_utils import execute_network_rebuild

            # Step 1: Network Generation
            network_step = NetworkGenerationStep(args_namespace)
            network_step.run()

            # Step 2: Zone Generation
            zone_step = ZoneGenerationStep(args_namespace)
            zone_step.run()

            # Step 3: Edge Splitting with Lane Assignment
            execute_edge_splitting(args_namespace)

            # Step 4: Network Rebuild
            execute_network_rebuild(args_namespace)

            # Parse network XML directly - no SUMO/TraCI needed
            try:
                actual_edge_count, actual_junction_count = self._parse_network_dimensions_from_file(CONFIG.network_file)
            except Exception as e:
                logger.error(f"Failed to parse generated network XML: {e}")
                # Fall back to estimated values
                grid_dimension = int(self.cli_args['grid_dimension'])
                actual_edge_count = 2 * 2 * (2 * grid_dimension * (grid_dimension - 1))
                actual_junction_count = self.num_intersections
                logger.warning(f"Using estimated dimensions: {actual_edge_count} edges, {actual_junction_count} junctions")

            # Restore original workspace
            self.cli_args['workspace'] = original_workspace
            CONFIG._output_dir = original_config_output_dir
            CONFIG._update_paths()

            return actual_edge_count, actual_junction_count

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

    def enable_debug_mode(self):
        """Enable detailed state debugging and logging."""
        self._debug_state = True
        if self.traffic_analyzer is not None:
            self.traffic_analyzer.debug = True

    def disable_debug_mode(self):
        """Disable detailed state debugging and logging."""
        self._debug_state = False
        if self.traffic_analyzer is not None:
            self.traffic_analyzer.debug = False
