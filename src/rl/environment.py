"""
RL Environment for Traffic Signal Control.

This module implements the OpenAI Gymnasium environment that interfaces
with SUMO simulation for reinforcement learning training.
"""

import os
import tempfile
import subprocess
import socket
import traci
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

from .constants import (
    STATE_NORMALIZATION_MIN, STATE_NORMALIZATION_MAX,
    EDGE_FEATURES_COUNT, JUNCTION_FEATURES_COUNT,
    ACTIONS_PER_INTERSECTION, PHASE_DURATION_OPTIONS,
    MIN_PHASE_DURATION, MAX_PHASE_DURATION, MIN_GREEN_TIME, MAX_GREEN_TIME,
    DECISION_INTERVAL_SECONDS, MEASUREMENT_INTERVAL_STEPS,
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
        import shlex
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
        grid_dimension = int(args.grid_dimension)
        instance.num_intersections = grid_dimension * grid_dimension
        instance.num_vehicles = args.num_vehicles
        instance.end_time = args.end_time
        instance.workspace = getattr(args, 'workspace', 'workspace')
        instance.cli_args = vars(args)

        # Calculate state vector size dynamically
        estimated_num_edges = 2 * 2 * \
            (2 * grid_dimension * (grid_dimension - 1))
        from .constants import (
            RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
            RL_DYNAMIC_NETWORK_FEATURES_COUNT
        )
        instance.state_vector_size = (estimated_num_edges * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                                      instance.num_intersections * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                                      RL_DYNAMIC_NETWORK_FEATURES_COUNT)

        # Define observation space
        instance.observation_space = gym.spaces.Box(
            low=STATE_NORMALIZATION_MIN,
            high=STATE_NORMALIZATION_MAX + 0.01,
            shape=(instance.state_vector_size,),
            dtype=np.float32
        )

        # Define action space
        if RL_PHASE_ONLY_MODE:
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

    def __init__(self, env_params_string: str, episode_number: int = 0):
        """Initialize the traffic control environment.

        Args:
            env_params_string: Raw parameter string for environment (e.g., "--network-seed 42 --grid_dimension 5 ...")
            episode_number: Episode number for reward logging (0 = auto-increment)
        """
        super().__init__()
        import shlex

        # Parse parameter string into CLI args
        parser = create_argument_parser()
        args_list = shlex.split(env_params_string)
        self.cli_args = vars(parser.parse_args(args_list))

        # Extract key parameters from CLI args
        grid_dimension = int(self.cli_args['grid_dimension'])
        self.num_intersections = grid_dimension * grid_dimension
        self.num_vehicles = self.cli_args['num_vehicles']
        self.end_time = self.cli_args['end_time']
        self.workspace = self.cli_args.get('workspace', 'workspace')

        # Calculate state vector size dynamically
        # Formula: E×6 + J×2 + 5 (edges×features + junctions×features + network_features)
        estimated_num_edges = 2 * 2 * \
            (2 * grid_dimension * (grid_dimension - 1))
        from .constants import (
            RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
            RL_DYNAMIC_NETWORK_FEATURES_COUNT
        )
        self.state_vector_size = (estimated_num_edges * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                                  self.num_intersections * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                                  RL_DYNAMIC_NETWORK_FEATURES_COUNT)

        # Define observation space: normalized state vector [0, 1]
        # State includes traffic features (edges) + signal features (junctions)
        state_size = self.state_vector_size
        self.observation_space = gym.spaces.Box(
            low=STATE_NORMALIZATION_MIN,
            # Add tolerance for floating point precision
            high=STATE_NORMALIZATION_MAX + 0.01,
            shape=(state_size,),
            dtype=np.float32
        )

        # Define action space based on control mode
        # Note: Final action space will be set after SUMO is started and we can read actual traffic light phases
        if RL_PHASE_ONLY_MODE:
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
        self.reward_log_file = None
        self.reward_log_writer = None
        self.episode_number = episode_number  # Track episode for log filenames

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode.

        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
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

        # Initialize reward analysis CSV logging (only if episode_number was provided)
        if self.episode_number > 0:
            self._initialize_reward_logging()

        # Get initial observation
        observation = self._get_observation()
        info = {'episode_step': self.current_step}

        return observation, info

    def step(self, action):
        """Execute one simulation step with the given action.

        Args:
            action: RL agent's action (phase + duration for all intersections)

        Returns:
            observation: New state after action execution
            reward: Reward signal for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        if not self.traci_connected:
            raise RuntimeError("SUMO simulation not connected")

        # Apply traffic light actions immediately
        self._apply_traffic_light_actions(action)

        # Advance simulation by decision interval
        for _ in range(DECISION_INTERVAL_SECONDS):
            if self.traci_connected:
                # Advance SUMO simulation
                traci.simulationStep()
                self.current_step += 1

                # Update vehicle tracker every measurement interval
                if self.current_step % MEASUREMENT_INTERVAL_STEPS == 0:
                    self.vehicle_tracker.update_vehicles(self.current_step)

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

                import logging
                logger = logging.getLogger(self.__class__.__name__)

            except Exception:
                # TraCI not connected and we're not in training mode
                import logging
                logger = logging.getLogger(self.__class__.__name__)
                logger.error(
                    f"=== NO TRACI CONNECTION AVAILABLE ===\nReturning zero observation")
                return np.zeros(self.state_vector_size, dtype=np.float32)

        # Initialize traffic analyzer on first call
        if self.traffic_analyzer is None:
            from .traffic_analysis import RLTrafficAnalyzer
            debug_mode = hasattr(self, '_debug_state') and self._debug_state
            self.traffic_analyzer = RLTrafficAnalyzer(
                self.edge_ids, debug=debug_mode)

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

        # Debug logging for feature construction (only when debug mode is enabled)
        if hasattr(self, '_debug_state') and self._debug_state and self.current_step % 10 == 0:  # Log every 10 steps
            import logging
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
            import logging
            logger = logging.getLogger(self.__class__.__name__)
            logger.warning(
                f"Step {self.current_step}: All {len(observation)} features are zero - possible data collection issue")

        return observation_array

    def _compute_reward(self):
        """Multi-objective reward function focusing on throughput, waiting time, and flow.

        Returns:
            float: Reward value combining throughput rewards, waiting penalties, and flow metrics
        """
        if not self.vehicle_tracker:
            return DEFAULT_FALLBACK_VALUE

        # ═══════════════════════════════════════════════════════════
        # 1. THROUGHPUT REWARDS (Primary Goal)
        # ═══════════════════════════════════════════════════════════
        # Immediate reward for every vehicle that completes this step
        vehicles_completed_this_step = self.vehicle_tracker.vehicles_completed_this_step
        throughput_reward = vehicles_completed_this_step * REWARD_THROUGHPUT_PER_VEHICLE

        # ═══════════════════════════════════════════════════════════
        # 2. WAITING TIME PENALTIES (Reduce Congestion)
        # ═══════════════════════════════════════════════════════════
        # Penalize increases in individual vehicle waiting times
        waiting_penalty = self.vehicle_tracker.compute_intermediate_rewards()

        # Additional penalty for vehicles with excessive waiting (>5 minutes)
        excessive_waiting_penalty = 0.0
        try:
            active_vehicles = traci.vehicle.getIDList()
            for vehicle_id in active_vehicles:
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                if waiting_time > REWARD_EXCESSIVE_WAITING_THRESHOLD:
                    excessive_waiting_penalty -= REWARD_EXCESSIVE_WAITING_PENALTY
        except:
            pass

        # ═══════════════════════════════════════════════════════════
        # 3. NETWORK FLOW EFFICIENCY (Keep traffic moving)
        # ═══════════════════════════════════════════════════════════
        avg_speed, bottleneck_count = self._get_network_performance_metrics()

        # Reward for maintaining good average speed
        speed_reward = (avg_speed / REWARD_SPEED_NORMALIZATION) * \
            REWARD_SPEED_REWARD_FACTOR

        # Penalty for bottlenecks (edges with congestion)
        bottleneck_penalty = -bottleneck_count * REWARD_BOTTLENECK_PENALTY_PER_EDGE

        # ═══════════════════════════════════════════════════════════
        # 4. INSERTION RATE BONUS (Get vehicles into network)
        # ═══════════════════════════════════════════════════════════
        insertion_bonus = 0.0
        try:
            loaded_count = traci.simulation.getLoadedNumber()
            departed_count = traci.simulation.getDepartedNumber()
            waiting_to_insert = loaded_count - departed_count
            if waiting_to_insert < REWARD_INSERTION_THRESHOLD:
                insertion_bonus = REWARD_INSERTION_BONUS
        except:
            pass

        # ═══════════════════════════════════════════════════════════
        # TOTAL REWARD
        # ═══════════════════════════════════════════════════════════
        total_reward = (
            throughput_reward +           # +10 per completed vehicle (PRIMARY)
            waiting_penalty +              # Negative (from vehicle tracker)
            excessive_waiting_penalty +    # Extra penalty for long waits
            speed_reward +                 # Reward for good network speed
            bottleneck_penalty +           # Penalty for congested edges
            insertion_bonus                # Bonus for clearing insertion queue
        )

        # CSV logging for reward validation (only when episode_number > 0)
        if self.episode_number > 0 and self.current_step % 100 == 0:
            self._log_reward_data(
                step=self.current_step,
                avg_speed=avg_speed,
                bottleneck_count=bottleneck_count,
                active_vehicles=len(traci.vehicle.getIDList()
                                    ) if self.traci_connected else 0,
                waiting_to_insert=waiting_to_insert if 'waiting_to_insert' in locals() else 0,
                vehicles_with_excessive_waiting=sum(1 for vid in traci.vehicle.getIDList()
                                                    if traci.vehicle.getWaitingTime(vid) > REWARD_EXCESSIVE_WAITING_THRESHOLD) if self.traci_connected else 0,
                vehicles_completed_this_step=vehicles_completed_this_step,
                throughput_reward=throughput_reward,
                waiting_penalty=waiting_penalty,
                excessive_waiting_penalty=excessive_waiting_penalty,
                speed_reward=speed_reward,
                bottleneck_penalty=bottleneck_penalty,
                insertion_bonus=insertion_bonus,
                total_reward=total_reward
            )

        # Reset step counter
        self.vehicle_tracker.vehicles_completed_this_step = 0

        return total_reward

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
            import logging
            logger = logging.getLogger(self.__class__.__name__)
            logger.warning(f"Error collecting final statistics: {e}")

        return stats

    def _initialize_reward_logging(self):
        """Initialize CSV file for reward analysis logging."""
        import csv

        # Close previous log file if exists
        if self.reward_log_file:
            try:
                self.reward_log_file.close()
            except:
                pass

        # Create new log file for this episode
        log_filename = f"reward_analysis_episode_{self.episode_number}.csv"
        self.reward_log_file = open(log_filename, 'w', newline='')
        self.reward_log_writer = csv.writer(self.reward_log_file)

        # Write header
        self.reward_log_writer.writerow([
            'step', 'avg_speed', 'bottleneck_count', 'active_vehicles',
            'waiting_to_insert', 'vehicles_with_excessive_waiting', 'vehicles_completed_this_step',
            'throughput_reward', 'waiting_penalty', 'excessive_waiting_penalty',
            'speed_reward', 'bottleneck_penalty', 'insertion_bonus', 'total_reward'
        ])
        self.reward_log_file.flush()

        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"Initialized reward analysis logging: {log_filename}")

    def _log_reward_data(self, **kwargs):
        """Log reward components and state to CSV."""
        if self.reward_log_writer:
            # Write row in same order as header
            self.reward_log_writer.writerow([
                kwargs['step'],
                kwargs['avg_speed'],
                kwargs['bottleneck_count'],
                kwargs['active_vehicles'],
                kwargs['waiting_to_insert'],
                kwargs['vehicles_with_excessive_waiting'],
                kwargs['vehicles_completed_this_step'],
                kwargs['throughput_reward'],
                kwargs['waiting_penalty'],
                kwargs['excessive_waiting_penalty'],
                kwargs['speed_reward'],
                kwargs['bottleneck_penalty'],
                kwargs['insertion_bonus'],
                kwargs['total_reward']
            ])
            self.reward_log_file.flush()

    def close(self):
        """Clean up environment resources."""
        # Close reward log file
        if self.reward_log_file:
            try:
                self.reward_log_file.close()
                import logging
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
                import shutil
                shutil.rmtree(self.workspace_dir)
            except:
                pass

    def _generate_sumo_files(self):
        """Generate SUMO network and route files using existing pipeline."""
        # Create args object from stored CLI args
        if self.cli_args is None:
            raise ValueError(
                "Unable to generate CLI args - no parameters provided")

        # Convert dict to Namespace object
        import argparse
        args = argparse.Namespace(**self.cli_args)

        # Execute pipeline to generate SUMO files (Steps 1-7 only, skip simulation)
        # RL environment manages its own SUMO simulation via TraCI
        pipeline = PipelineFactory.create_pipeline(args)
        if hasattr(pipeline, 'execute_file_generation_only'):
            pipeline.execute_file_generation_only()
        else:
            # Fallback for other pipeline types (e.g., SamplePipeline)
            pipeline.execute()

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
        ]
        traci.start(sumo_cmd, port=self.traci_port)
        self.traci_connected = True

        # Get network topology information
        self.junction_ids = traci.trafficlight.getIDList()
        self.edge_ids = traci.edge.getIDList()

        # Update action space with actual traffic light phases
        self._update_action_space_from_sumo()

        # Update observation space with actual network dimensions
        self._update_observation_space_from_sumo()

    def _apply_traffic_light_actions(self, actions):
        """Apply RL actions immediately to traffic lights."""
        if not self.traci_connected:
            return

        # Record actions for vehicle tracking
        action_dict = {}

        if RL_PHASE_ONLY_MODE:
            # Phase-only mode: actions is a flat array: [phase0, phase1, phase2, ...]
            # Each element is a phase index for the corresponding intersection
            for i, junction_id in enumerate(self.junction_ids):
                if i < len(actions):
                    phase_idx = int(actions[i])
                    duration = RL_FIXED_PHASE_DURATION

                    # Apply the RL decision (phase-only with fixed duration)
                    try:
                        traci.trafficlight.setPhase(junction_id, phase_idx)
                        traci.trafficlight.setPhaseDuration(
                            junction_id, duration)
                    except Exception as e:
                        # Log error but continue with other intersections
                        pass

                    action_dict[junction_id] = (phase_idx, duration)
        else:
            # Legacy mode: actions is a flat array: [phase0, duration0, phase1, duration1, ...]
            action_pairs = actions.reshape(
                self.num_intersections, ACTIONS_PER_INTERSECTION)

            # Apply RL actions to traffic signals
            for i, junction_id in enumerate(self.junction_ids):
                phase_idx, duration_idx = action_pairs[i]
                duration = PHASE_DURATION_OPTIONS[int(duration_idx)]

                # Apply the RL decision
                try:
                    traci.trafficlight.setPhase(junction_id, int(phase_idx))
                    traci.trafficlight.setPhaseDuration(junction_id, duration)
                except Exception as e:
                    # Log error but continue with other intersections
                    pass

                action_dict[junction_id] = (int(phase_idx), duration)

        # Record decision for vehicle tracking
        self.vehicle_tracker.record_decision(self.current_step, action_dict)

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

        import logging
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

        import logging
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
                RL_DYNAMIC_NETWORK_FEATURES_COUNT
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

        except Exception as e:
            logger.error(f"Failed to update observation space from SUMO: {e}")
            logger.info("Keeping default observation space")

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
