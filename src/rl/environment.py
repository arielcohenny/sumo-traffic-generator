"""
RL Environment for Traffic Signal Control.

This module implements the OpenAI Gymnasium environment that interfaces
with SUMO simulation for reinforcement learning training.
"""

import os
import tempfile
import subprocess
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
    DEFAULT_INITIAL_STEP, DEFAULT_INITIAL_TIME, DEFAULT_FALLBACK_VALUE, DEFAULT_OBSERVATION_PADDING
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

    def __init__(self, config):
        """Initialize the traffic control environment.

        Args:
            config: Configuration object or CLI args dict containing network and simulation parameters
        """
        super().__init__()

        # Handle both config objects and CLI args dictionaries
        if isinstance(config, dict):
            # For parallel environments that pass CLI args
            from .config import get_rl_config
            self.config = get_rl_config()  # Get default config
            self.cli_args = config  # Store CLI args for pipeline execution
        else:
            # For single environments that pass config objects
            self.config = config
            self.cli_args = None

        # Define observation space: normalized state vector [0, 1]
        # State includes traffic features (edges) + signal features (junctions)
        state_size = self.config.state_vector_size_estimate
        self.observation_space = gym.spaces.Box(
            low=STATE_NORMALIZATION_MIN,
            high=STATE_NORMALIZATION_MAX,
            shape=(state_size,),
            dtype=np.float32
        )

        # Define action space: discrete actions for phase + duration per intersection
        # Each intersection: [phase_id, duration_index]
        # Phase IDs: 0-(NUM_TRAFFIC_LIGHT_PHASES-1), Duration indices: 0-(NUM_PHASE_DURATION_OPTIONS-1)
        self.action_space = gym.spaces.MultiDiscrete([NUM_TRAFFIC_LIGHT_PHASES, NUM_PHASE_DURATION_OPTIONS] * self.config.num_intersections)

        # Initialize simulation state
        self.current_step = DEFAULT_INITIAL_STEP
        self.episode_start_time = DEFAULT_INITIAL_TIME
        self.workspace_dir = None
        self.vehicle_tracker = None
        self.junction_ids = []
        self.edge_ids = []

        # TraCI connection state
        self.traci_connected = False

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
            # Use workspace from CLI args (parallel environment)
            self.workspace_dir = self.cli_args['workspace']
            os.makedirs(self.workspace_dir, exist_ok=True)
        elif hasattr(self.config, 'workspace'):
            # Use configured workspace (backward compatibility)
            self.workspace_dir = self.config.workspace
            os.makedirs(self.workspace_dir, exist_ok=True)
        else:
            # Create temporary workspace for single environment
            self.workspace_dir = tempfile.mkdtemp(prefix="rl_episode_")

        # Generate SUMO configuration using existing pipeline
        self._generate_sumo_files()

        # Start SUMO simulation with TraCI
        self._start_sumo_simulation()

        # Initialize vehicle tracker
        self.vehicle_tracker = VehicleTracker()

        # Reset episode state
        self.current_step = DEFAULT_INITIAL_STEP
        self.episode_start_time = DEFAULT_INITIAL_TIME

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

        # Apply traffic light actions
        self._apply_traffic_light_actions(action)

        # Record decision for vehicle tracking
        action_dict = {}
        action_pairs = action.reshape(self.config.num_intersections, ACTIONS_PER_INTERSECTION)
        for i, junction_id in enumerate(self.junction_ids):
            phase_idx, duration_idx = action_pairs[i]
            duration = PHASE_DURATION_OPTIONS[int(duration_idx)]
            action_dict[junction_id] = (int(phase_idx), duration)

        self.vehicle_tracker.record_decision(self.current_step, action_dict)

        # Advance simulation by decision interval
        for _ in range(DECISION_INTERVAL_SECONDS):
            if self.traci_connected:
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
        """Collect current network state for RL agent.

        Returns:
            np.array: Normalized state vector containing traffic indicators
        """
        if not self.traci_connected:
            # Return zero observation if not connected
            return np.zeros(self.config.state_vector_size_estimate, dtype=np.float32)

        observation = []

        # Collect traffic features from edges (4 features per edge)
        for edge_id in self.edge_ids:
            # Skip internal edges
            if ':' in edge_id:
                continue

            try:
                # Edge speed feature (normalized by speed limit)
                current_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                max_speed = traci.edge.getMaxSpeed(edge_id)
                speed_ratio = current_speed / max_speed if max_speed > 0 else DEFAULT_FALLBACK_VALUE

                # Edge density feature (vehicles per meter)
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                edge_length = traci.edge.getLength(edge_id)
                density = vehicle_count / edge_length if edge_length > 0 else DEFAULT_FALLBACK_VALUE
                # Normalize density using configurable threshold
                density_normalized = min(density / MAX_DENSITY_VEHICLES_PER_METER, STATE_NORMALIZATION_MAX)

                # Edge flow feature (vehicles per second)
                flow = traci.edge.getLastStepVehicleNumber(edge_id) / STATE_NORMALIZATION_MAX  # per second
                # Normalize flow using configurable threshold
                flow_normalized = min(flow / MAX_FLOW_VEHICLES_PER_SECOND, STATE_NORMALIZATION_MAX)

                # Congestion flag (binary: waiting time > threshold)
                waiting_time = traci.edge.getWaitingTime(edge_id)
                congestion_flag = STATE_NORMALIZATION_MAX if waiting_time > CONGESTION_WAITING_TIME_THRESHOLD else STATE_NORMALIZATION_MIN

                observation.extend([speed_ratio, density_normalized, flow_normalized, congestion_flag])

            except Exception as e:
                # Default values if edge data unavailable
                observation.extend([DEFAULT_FALLBACK_VALUE] * EDGE_FEATURES_COUNT)

        # Collect signal features from junctions (2 features per junction)
        for junction_id in self.junction_ids:
            try:
                # Current phase (normalized by total phases)
                current_phase = traci.trafficlight.getPhase(junction_id)
                total_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
                phase_normalized = current_phase / total_phases if total_phases > 0 else DEFAULT_FALLBACK_VALUE

                # Remaining duration (normalized by max phase duration)
                remaining_duration = traci.trafficlight.getNextSwitch(junction_id) - traci.simulation.getTime()
                duration_normalized = remaining_duration / MAX_PHASE_DURATION

                observation.extend([phase_normalized, duration_normalized])

            except Exception as e:
                # Default values if junction data unavailable
                observation.extend([DEFAULT_FALLBACK_VALUE] * JUNCTION_FEATURES_COUNT)

        # Ensure observation has correct size
        expected_size = self.config.state_vector_size_estimate
        if len(observation) < expected_size:
            # Pad with zeros if too short
            observation.extend([DEFAULT_OBSERVATION_PADDING] * (expected_size - len(observation)))
        elif len(observation) > expected_size:
            # Truncate if too long
            observation = observation[:expected_size]

        return np.array(observation, dtype=np.float32)

    def _compute_reward(self):
        """Compute reward signal based on vehicle performance.

        Returns:
            float: Reward value (penalties + bonuses)
        """
        if not self.vehicle_tracker:
            return DEFAULT_FALLBACK_VALUE

        # Get intermediate rewards (vehicle penalties)
        intermediate_reward = self.vehicle_tracker.compute_intermediate_rewards()

        # Episode reward will be added at episode end
        # For now, just return the intermediate penalty signal
        return intermediate_reward

    def _is_terminated(self):
        """Check if episode should end naturally.

        Returns:
            bool: True if simulation completed successfully
        """
        if not self.traci_connected:
            return True

        # Check if simulation time reached configured end_time
        current_time = traci.simulation.getTime()
        if current_time >= self.config.end_time:
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
        if self.current_step > self.config.end_time * 2:
            return True

        return False

    def close(self):
        """Clean up environment resources."""
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
        # Use stored CLI args if available (parallel environment)
        if self.cli_args is not None:
            cli_args = self.cli_args.copy()
        elif hasattr(self.config, 'get_cli_args'):
            # Convert config to CLI args format (single environment)
            cli_args = self.config.get_cli_args()
        else:
            # Fallback for unexpected config types
            raise ValueError("Unable to generate CLI args from config")

        # Create argument namespace for pipeline
        parser = create_argument_parser()
        args_list = []
        for key, value in cli_args.items():
            args_list.extend([f'--{key}', str(value)])

        # Add workspace argument
        args_list.extend(['--workspace', self.workspace_dir])

        # Parse args explicitly from the constructed list instead of sys.argv
        args = parser.parse_args(args_list)

        # Execute pipeline to generate SUMO files
        pipeline = PipelineFactory.create_pipeline(args)
        pipeline.execute()

    def _start_sumo_simulation(self):
        """Start SUMO simulation with TraCI connection."""
        # Find the generated SUMO config file
        # Pipeline creates files in workspace/ subdirectory
        workspace_subdir = os.path.join(self.workspace_dir, 'workspace')
        search_dirs = [workspace_subdir, self.workspace_dir]

        sumo_config = None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.sumocfg'):
                        sumo_config = os.path.join(search_dir, file)
                        break
                if sumo_config:
                    break

        if not sumo_config:
            raise RuntimeError(f"No SUMO config file found in {self.workspace_dir} or {workspace_subdir}")

        # Start SUMO with TraCI
        sumo_cmd = ['sumo', '-c', sumo_config, '--start', '--quit-on-end']
        traci.start(sumo_cmd)
        self.traci_connected = True

        # Get network topology information
        self.junction_ids = traci.trafficlight.getIDList()
        self.edge_ids = traci.edge.getIDList()

    def _apply_traffic_light_actions(self, actions):
        """Apply RL actions to SUMO traffic lights."""
        # actions is a flat array: [phase0, duration0, phase1, duration1, ...]
        action_pairs = actions.reshape(self.config.num_intersections, ACTIONS_PER_INTERSECTION)

        for i, junction_id in enumerate(self.junction_ids):
            phase_idx, duration_idx = action_pairs[i]

            # Set traffic light phase
            traci.trafficlight.setPhase(junction_id, int(phase_idx))

            # Set phase duration (convert index to actual duration)
            duration = PHASE_DURATION_OPTIONS[int(duration_idx)]
            traci.trafficlight.setPhaseDuration(junction_id, duration)