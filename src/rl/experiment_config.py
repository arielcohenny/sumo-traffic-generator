"""
Experiment configuration for RL training.

This module provides a config-driven approach to RL experiments.
Each experiment is defined by a YAML file that specifies all tunable
parameters. This replaces hardcoded values in constants.py for
experiment-specific settings.

IMPORTANT: This module must NOT import from any other src/rl/ module
at the top level to prevent circular dependencies. Registry imports
are deferred to validate_experiment().
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Complete configuration for an RL training experiment.

    Default values match the current hardcoded values in constants.py
    so that running without a config file produces identical behavior.
    """

    # Experiment metadata
    name: str = "default"
    description: str = ""
    config_version: int = 1  # Schema version for model compatibility checks

    # PPO hyperparameters
    learning_rate: float = 1e-4
    clip_range: float = 0.2
    batch_size: int = 2048
    n_steps: int = 4096
    n_epochs: int = 5
    gamma: float = 0.995
    gae_lambda: float = 0.98
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01
    log_std_init: float = -1.0

    # LR schedule
    lr_schedule: str = "exponential"  # "constant", "exponential", "linear"
    lr_final: float = 5e-6
    lr_decay_rate: float = 0.99995

    # Entropy schedule
    ent_schedule: bool = True
    ent_initial: float = 0.02
    ent_final: float = 0.001
    ent_decay_steps: int = 500_000

    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"  # "relu", "tanh", "elu", "leaky_relu"

    # Reward function
    reward_function: str = "empirical"  # key into REWARD_REGISTRY
    reward_params: Dict = field(default_factory=dict)

    # Observation features (lists of feature names from registries)
    edge_features: List[str] = field(default_factory=lambda: [
        "speed_ratio",
        "congestion_flag",
        "normalized_density",
        "is_bottleneck",
        "normalized_time_loss",
        "speed_trend",
    ])
    junction_features: List[str] = field(default_factory=lambda: [
        "phase_normalized",
        "duration_normalized",
    ])
    network_features: List[str] = field(default_factory=lambda: [
        "bottleneck_ratio",
        "cost_normalized",
        "vehicles_normalized",
        "avg_speed_normalized",
        "congestion_ratio",
    ])

    # Action space
    action_mode: str = "continuous"  # "continuous", "discrete"
    phases_per_junction: int = 4
    action_low: float = -10.0
    action_high: float = 10.0

    # Training
    total_timesteps: int = 100_000
    checkpoint_freq: int = 10_000
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 70.0

    # Simulation (passed through to SUMO pipeline)
    grid_dimension: int = 5
    num_vehicles: int = 4500
    end_time: int = 7200
    network_seed: int = 42
    routing_strategy: str = "shortest 70 realtime 30"
    vehicle_types: str = "passenger 90 public 10"
    departure_pattern: str = "six_periods"

    # Cycle
    cycle_lengths: List[int] = field(default_factory=lambda: [90])
    cycle_strategy: str = "fixed"

    # Observation space tolerance (currently hardcoded as +0.01 in 3 places)
    obs_space_tolerance: float = 0.01

    # Inheritance (not saved in resolved configs)
    base_config: str = ""

    @property
    def edge_feature_count(self) -> int:
        """Number of edge features (replaces RL_DYNAMIC_EDGE_FEATURES_COUNT)."""
        return len(self.edge_features)

    @property
    def junction_feature_count(self) -> int:
        """Number of junction features (replaces RL_DYNAMIC_JUNCTION_FEATURES_COUNT)."""
        return len(self.junction_features)

    @property
    def network_feature_count(self) -> int:
        """Number of network features (replaces RL_DYNAMIC_NETWORK_FEATURES_COUNT)."""
        return len(self.network_features)


def load_experiment(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file with inheritance support.

    If the YAML contains a `base_config` field, loads the base config first,
    then applies overrides from the current file. Inheritance is single-level
    (no recursive chaining) to keep things simple and debuggable.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        Fully resolved ExperimentConfig.
    """
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}

    base_path = overrides.pop("base_config", None)
    if base_path:
        # Resolve relative paths relative to the current config file
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_path)
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        # Remove base_config from base to prevent chaining
        base.pop("base_config", None)
        base.update(overrides)  # Overrides win
        overrides = base

    config = ExperimentConfig(**overrides)
    validate_experiment(config)
    return config


def validate_experiment(config: ExperimentConfig):
    """Validate experiment config at load time to catch errors early.

    Uses deferred imports from registries to avoid circular dependencies.
    """
    # Feature names must be in their respective registries
    from .traffic_analysis import (
        EDGE_FEATURE_REGISTRY,
        JUNCTION_FEATURE_REGISTRY,
        NETWORK_FEATURE_REGISTRY,
    )

    for feat in config.edge_features:
        if feat not in EDGE_FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown edge feature: '{feat}'. "
                f"Valid: {list(EDGE_FEATURE_REGISTRY.keys())}"
            )
    for feat in config.junction_features:
        if feat not in JUNCTION_FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown junction feature: '{feat}'. "
                f"Valid: {list(JUNCTION_FEATURE_REGISTRY.keys())}"
            )
    for feat in config.network_features:
        if feat not in NETWORK_FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown network feature: '{feat}'. "
                f"Valid: {list(NETWORK_FEATURE_REGISTRY.keys())}"
            )

    # Reward function must be in registry
    from .reward import REWARD_REGISTRY

    if config.reward_function not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: '{config.reward_function}'. "
            f"Valid: {list(REWARD_REGISTRY.keys())}"
        )

    # Activation must be valid
    valid_activations = {"relu", "tanh", "elu", "leaky_relu"}
    if config.activation not in valid_activations:
        raise ValueError(
            f"Unknown activation: '{config.activation}'. "
            f"Valid: {valid_activations}"
        )

    # Action mode must be valid
    valid_action_modes = {"continuous", "discrete"}
    if config.action_mode not in valid_action_modes:
        raise ValueError(
            f"Unknown action_mode: '{config.action_mode}'. "
            f"Valid: {valid_action_modes}"
        )

    # LR schedule must be valid
    valid_lr_schedules = {"constant", "exponential", "linear"}
    if config.lr_schedule not in valid_lr_schedules:
        raise ValueError(
            f"Unknown lr_schedule: '{config.lr_schedule}'. "
            f"Valid: {valid_lr_schedules}"
        )

    # Numerical bounds
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if not (0 < config.clip_range < 1):
        raise ValueError(f"clip_range must be in (0, 1), got {config.clip_range}")
    if config.action_low >= config.action_high:
        raise ValueError(
            f"action_low ({config.action_low}) must be < action_high ({config.action_high})"
        )
    if config.total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {config.total_timesteps}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {config.n_steps}")

    # Feature lists must not be empty
    if not config.edge_features:
        raise ValueError("edge_features must not be empty")
    if not config.junction_features:
        raise ValueError("junction_features must not be empty")
    if not config.network_features:
        raise ValueError("network_features must not be empty")


def save_experiment(config: ExperimentConfig, path: str):
    """Save resolved experiment config to YAML for reproducibility.

    Saves the fully-resolved config (after inheritance), so the saved file
    is a complete standalone record of what was used.
    """
    import dataclasses

    data = dataclasses.asdict(config)
    # Don't save inheritance reference in resolved config
    data.pop("base_config", None)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
