"""
RL Training Configuration for Traffic Signal Control.

This module defines the fixed training configuration for network-specific
RL training. These values are FIXED and cannot be changed after Phase 2
implementation begins without significant code modifications.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import logging

from .constants import (
    RL_TRAINING_GRID_DIMENSION, RL_TRAINING_BLOCK_SIZE_M, RL_TRAINING_JUNCTIONS_TO_REMOVE,
    RL_TRAINING_NUM_VEHICLES, RL_TRAINING_VEHICLE_TYPES, RL_TRAINING_END_TIME,
    RL_TRAINING_DECISION_INTERVAL, RL_TRAINING_MEASUREMENT_INTERVAL,
    RL_TRAINING_NUM_INTERSECTIONS, RL_TRAINING_STATE_VECTOR_SIZE_ESTIMATE,
    RL_TRAINING_ACTION_VECTOR_SIZE, MIN_STATE_VECTOR_SIZE, MAX_STATE_VECTOR_SIZE,
    MIN_ACTION_VECTOR_SIZE, MAX_ACTION_VECTOR_SIZE, DEFAULT_N_PARALLEL_ENVS,
    MIN_PARALLEL_ENVS, MAX_PARALLEL_ENVS, PARALLEL_WORKSPACE_PREFIX,
    DEFAULT_STEP_LENGTH, DEFAULT_TRAINING_SEED
)


@dataclass(frozen=True)  # Immutable to prevent accidental changes
class RLTrainingConfig:
    """
    Fixed training configuration for network-specific RL training.

    These values are locked for the entire implementation to ensure
    consistent state/action space dimensions and neural network architecture.

    WARNING: Changing these values after Phase 2 implementation begins
    requires significant code modifications throughout the RL system.
    """

    # Network parameters (FIXED for network-specific training)
    grid_dimension: int = RL_TRAINING_GRID_DIMENSION
    block_size_m: int = RL_TRAINING_BLOCK_SIZE_M
    junctions_to_remove: int = RL_TRAINING_JUNCTIONS_TO_REMOVE

    # Traffic parameters (FIXED)
    num_vehicles: int = RL_TRAINING_NUM_VEHICLES
    vehicle_types: str = RL_TRAINING_VEHICLE_TYPES
    end_time: int = RL_TRAINING_END_TIME

    # RL-specific parameters (FIXED)
    decision_interval_seconds: int = RL_TRAINING_DECISION_INTERVAL
    measurement_interval_steps: int = RL_TRAINING_MEASUREMENT_INTERVAL

    # Parallel execution parameters
    n_parallel_envs: int = DEFAULT_N_PARALLEL_ENVS

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()

    @property
    def num_intersections(self) -> int:
        """Number of intersections in the grid network."""
        return self.grid_dimension * self.grid_dimension

    @property
    def estimated_num_edges(self) -> int:
        """Estimated number of edges in 3x3 grid (exact value determined at runtime)."""
        if self.grid_dimension == 3:
            return 12  # Typical for 3x3 grid
        else:
            # Rough estimate for other sizes
            return (self.grid_dimension * (self.grid_dimension + 1) * 2)

    @property
    def state_vector_size_estimate(self) -> int:
        """
        Estimated state vector size: E×4 + J×2
        (edges×traffic_features + junctions×signal_features)
        """
        return self.estimated_num_edges * 4 + self.num_intersections * 2

    @property
    def action_vector_size(self) -> int:
        """
        Action vector size: J×2 (intersections × [phase_selection, duration_selection])
        """
        return self.num_intersections * 2

    @property
    def vehicle_density_per_intersection(self) -> float:
        """Average number of vehicles per intersection."""
        return self.num_vehicles / self.num_intersections

    @property
    def episode_duration_minutes(self) -> float:
        """Episode duration in minutes for human-readable display."""
        return self.end_time / 60.0

    @property
    def decisions_per_episode(self) -> int:
        """Number of RL decisions per episode."""
        return self.end_time // self.decision_interval_seconds

    def get_cli_args(self) -> dict:
        """
        Get configuration as CLI arguments for existing pipeline integration.

        Returns:
            dict: CLI arguments compatible with existing argument parser
        """
        return {
            'grid_dimension': self.grid_dimension,
            'block_size_m': self.block_size_m,
            'junctions_to_remove': self.junctions_to_remove,
            'num_vehicles': self.num_vehicles,
            'vehicle_types': self.vehicle_types,
            'end-time': self.end_time,  # Use hyphen for CLI compatibility
            'seed': DEFAULT_TRAINING_SEED,  # Fixed seed for reproducible training
            'step-length': DEFAULT_STEP_LENGTH,  # Use hyphen for CLI compatibility
            'traffic_control': 'rl'  # Specify RL traffic control
        }

    def get_cli_args_for_env(self, env_index: int, base_workspace: str = "rl_training", max_envs: int = None) -> dict:
        """
        Get CLI arguments for a specific parallel environment with unique workspace.

        Args:
            env_index: Index of the parallel environment (0-based)
            base_workspace: Base workspace directory name
            max_envs: Maximum number of environments (if None, uses self.n_parallel_envs)

        Returns:
            dict: CLI arguments with unique workspace path for this environment
        """
        if max_envs is None:
            max_envs = getattr(self, 'n_parallel_envs', 1)

        if env_index < 0 or env_index >= max_envs:
            raise ValueError(f"Environment index {env_index} outside valid range [0, {max_envs-1}]")

        cli_args = self.get_cli_args()
        # Create unique workspace path: base_workspace/env_000, env_001, etc.
        env_workspace = f"{base_workspace}/env_{env_index:03d}"
        cli_args['workspace'] = env_workspace
        # Add unique seed for each environment to ensure different traffic patterns
        cli_args['seed'] = DEFAULT_TRAINING_SEED + env_index

        return cli_args

    def _validate_configuration(self) -> None:
        """Validate configuration parameters and raise errors for invalid values."""
        logger = logging.getLogger(__name__)

        # Validate network parameters
        if self.grid_dimension < 1 or self.grid_dimension > 10:
            raise ValueError(f"Grid dimension {self.grid_dimension} outside valid range [1, 10]")

        if self.block_size_m < 50 or self.block_size_m > 500:
            raise ValueError(f"Block size {self.block_size_m}m outside valid range [50, 500]")

        if self.junctions_to_remove < 0 or self.junctions_to_remove >= self.num_intersections:
            raise ValueError(f"Junctions to remove {self.junctions_to_remove} invalid for {self.num_intersections} total")

        # Validate traffic parameters
        if self.num_vehicles < 1 or self.num_vehicles > 10000:
            raise ValueError(f"Vehicle count {self.num_vehicles} outside valid range [1, 10000]")

        if self.end_time < 300 or self.end_time > 86400:  # 5 minutes to 24 hours
            raise ValueError(f"End time {self.end_time}s outside valid range [300, 86400]")

        # Validate RL parameters
        if self.decision_interval_seconds < 1 or self.decision_interval_seconds > 60:
            raise ValueError(f"Decision interval {self.decision_interval_seconds}s outside valid range [1, 60]")

        if self.measurement_interval_steps < 1 or self.measurement_interval_steps > 100:
            raise ValueError(f"Measurement interval {self.measurement_interval_steps} outside valid range [1, 100]")

        # Validate parallel execution parameters
        if not (MIN_PARALLEL_ENVS <= self.n_parallel_envs <= MAX_PARALLEL_ENVS):
            raise ValueError(f"Parallel environments {self.n_parallel_envs} outside valid range [{MIN_PARALLEL_ENVS}, {MAX_PARALLEL_ENVS}]")

        # Validate derived dimensions
        if not (MIN_STATE_VECTOR_SIZE <= self.state_vector_size_estimate <= MAX_STATE_VECTOR_SIZE):
            raise ValueError(f"Estimated state vector size {self.state_vector_size_estimate} outside safe range")

        if not (MIN_ACTION_VECTOR_SIZE <= self.action_vector_size <= MAX_ACTION_VECTOR_SIZE):
            raise ValueError(f"Action vector size {self.action_vector_size} outside safe range")

        # Check vehicle density for realism
        if self.vehicle_density_per_intersection > 50:
            logger.warning(f"High vehicle density: {self.vehicle_density_per_intersection:.1f} vehicles/intersection")
        elif self.vehicle_density_per_intersection < 5:
            logger.warning(f"Low vehicle density: {self.vehicle_density_per_intersection:.1f} vehicles/intersection")

        logger.info(f"RL Training Configuration validated successfully:")
        logger.info(f"  Network: {self.grid_dimension}×{self.grid_dimension} grid, {self.num_intersections} intersections")
        logger.info(f"  Traffic: {self.num_vehicles} vehicles, {self.episode_duration_minutes:.0f} min episodes")
        logger.info(f"  RL: {self.decisions_per_episode} decisions/episode, {self.decision_interval_seconds}s intervals")
        logger.info(f"  Dimensions: ~{self.state_vector_size_estimate} state, {self.action_vector_size} actions")
        logger.info(f"  Parallel: {self.n_parallel_envs} environments for efficient training")

    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        return f"""RL Training Configuration (FIXED VALUES):
Network: {self.grid_dimension}×{self.grid_dimension} grid ({self.num_intersections} intersections)
Traffic: {self.num_vehicles} vehicles over {self.episode_duration_minutes:.0f} minutes
Density: {self.vehicle_density_per_intersection:.1f} vehicles per intersection
RL Control: {self.decisions_per_episode} decisions per episode ({self.decision_interval_seconds}s intervals)
Parallel Training: {self.n_parallel_envs} environments for efficient learning
Estimated Dimensions: {self.state_vector_size_estimate} state features, {self.action_vector_size} action choices"""


# Global instance for use throughout RL system
# This ensures all components use identical configuration
RL_CONFIG = RLTrainingConfig()


def get_rl_config() -> RLTrainingConfig:
    """
    Get the global RL training configuration.

    Returns:
        RLTrainingConfig: The fixed training configuration instance
    """
    return RL_CONFIG


def validate_rl_environment_compatibility(actual_state_size: int, actual_action_size: int) -> bool:
    """
    Validate that actual environment dimensions match expected configuration.

    Args:
        actual_state_size: Actual state vector size from implemented environment
        actual_action_size: Actual action vector size from implemented environment

    Returns:
        bool: True if dimensions match expectations, False otherwise

    Raises:
        ValueError: If dimensions are incompatible with neural network training
    """
    config = get_rl_config()
    logger = logging.getLogger(__name__)

    # Check state vector size
    state_diff = abs(actual_state_size - config.state_vector_size_estimate)
    if state_diff > 10:  # Allow some variation in edge count estimation
        logger.warning(f"State vector size {actual_state_size} differs significantly from estimate {config.state_vector_size_estimate}")

    # Action vector size must match exactly
    if actual_action_size != config.action_vector_size:
        raise ValueError(f"Action vector size mismatch: actual={actual_action_size}, expected={config.action_vector_size}")

    # Final validation
    if not (MIN_STATE_VECTOR_SIZE <= actual_state_size <= MAX_STATE_VECTOR_SIZE):
        raise ValueError(f"Actual state vector size {actual_state_size} outside safe training range")

    if not (MIN_ACTION_VECTOR_SIZE <= actual_action_size <= MAX_ACTION_VECTOR_SIZE):
        raise ValueError(f"Actual action vector size {actual_action_size} outside safe training range")

    logger.info(f"Environment dimensions validated: {actual_state_size} state, {actual_action_size} actions")
    return True