"""
Config loader for single-env RL training from a unified config.yaml.

Reads a single YAML file with sections (network, traffic, algorithm, reward,
execution) and converts it to an ExperimentConfig suitable for single-env
training. This is used by the experiment-folder workflow where each
experiment has its own self-contained config.yaml.
"""

import dataclasses
import glob
import os
import re
from typing import Dict, Optional, Tuple

import yaml

# Add project root so src.rl imports work when run from any directory
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.rl.experiment_config import ExperimentConfig, validate_experiment


# ── Section-to-field mappings ──────────────────────────────────────────

_NETWORK_FIELDS = {
    "grid_dimension",
    "network_seed",
    "junctions_to_remove",
    "block_size_m",
    "lane_count",
    "step_length",
    "land_use_block_size_m",
    "attractiveness",
    "traffic_light_strategy",
}

_TRAFFIC_FIELDS = {
    "num_vehicles",
    "routing_strategy",
    "vehicle_types",
    "passenger_routes",
    "departure_pattern",
    "start_time_hour",
    "end_time",
    "private_traffic_seed",
    "public_traffic_seed",
    # traffic_control is handled separately (not an ExperimentConfig field)
}

_ALGORITHM_FIELDS = {
    "learning_rate",
    "clip_range",
    "batch_size",
    "n_steps",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "max_grad_norm",
    "ent_coef",
    "log_std_init",
    "net_arch",
    "activation",
    "lr_schedule",
    "lr_final",
    "lr_decay_rate",
    "ent_schedule",
    "ent_initial",
    "ent_final",
    "ent_decay_steps",
}

_EXECUTION_FIELDS = {
    "total_timesteps",
    "checkpoint_freq",
    "early_stopping_patience",
    "early_stopping_min_delta",
}

# Scenario-specific traffic keys that go into the scenario dict
_SCENARIO_KEYS = {
    "num_vehicles",
    "routing_strategy",
    "vehicle_types",
    "passenger_routes",
    "departure_pattern",
    "start_time_hour",
    "private_traffic_seed",
    "public_traffic_seed",
}


def load_single_config(
    config_path: str,
) -> Tuple[ExperimentConfig, Dict, str]:
    """Load a unified config.yaml and return an ExperimentConfig plus extras.

    The YAML is expected to have top-level sections:
        network:   -> simulation/network fields
        traffic:   -> default traffic fields + scenario dict
        algorithm: -> PPO hyperparameters
        reward:    -> reward_function + reward_params
        execution: -> timesteps, checkpoints, early stopping

    Args:
        config_path: Path to the unified config.yaml file.

    Returns:
        A 3-tuple of:
        - config: ExperimentConfig with scenarios=[] (single-env mode).
        - scenario_dict: Traffic parameters dict for build_env_params_string.
        - traffic_control: The traffic control method string (default "rl").
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    merged: Dict = {}
    scenario_dict: Dict = {}
    traffic_control = "rl"

    # ── network section ────────────────────────────────────────────────
    network_sec = raw.get("network", {}) or {}
    for key, value in network_sec.items():
        if key in _NETWORK_FIELDS:
            merged[key] = value

    # ── traffic section ────────────────────────────────────────────────
    traffic_sec = raw.get("traffic", {}) or {}
    for key, value in traffic_sec.items():
        if key == "traffic_control":
            traffic_control = str(value)
        elif key in _TRAFFIC_FIELDS:
            merged[key] = value
            # Also populate scenario dict for scenario-level keys
            if key in _SCENARIO_KEYS:
                scenario_dict[key] = value

    # ── algorithm section ──────────────────────────────────────────────
    algorithm_sec = raw.get("algorithm", {}) or {}
    for key, value in algorithm_sec.items():
        if key in _ALGORITHM_FIELDS:
            merged[key] = value

    # ── reward section ─────────────────────────────────────────────────
    reward_sec = raw.get("reward", {}) or {}
    if "function" in reward_sec:
        merged["reward_function"] = reward_sec["function"]
    # Everything else in the reward section becomes reward_params
    reward_params = {k: v for k, v in reward_sec.items() if k != "function"}
    if reward_params:
        merged["reward_params"] = reward_params

    # ── execution section ──────────────────────────────────────────────
    execution_sec = raw.get("execution", {}) or {}
    for key, value in execution_sec.items():
        if key == "eval_freq":
            # eval_freq maps to checkpoint_freq
            merged["checkpoint_freq"] = value
        elif key in _EXECUTION_FIELDS:
            merged[key] = value

    # ── Build ExperimentConfig ─────────────────────────────────────────
    # Filter to only valid ExperimentConfig fields
    valid_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    # Single-env: no multi-scenario
    filtered["scenarios"] = []

    config = ExperimentConfig(**filtered)
    validate_experiment(config)

    return config, scenario_dict, traffic_control


def build_env_params_string(
    config: ExperimentConfig,
    scenario: Dict,
    traffic_control: str,
) -> str:
    """Build a complete --env-params string from config + scenario + traffic_control.

    Same logic as build_env_params_for_scenario in experiment_config.py, but
    uses the provided traffic_control instead of hardcoding "tree_method".

    Args:
        config: ExperimentConfig with network and execution params.
        scenario: Dict of per-scenario traffic params.
        traffic_control: Traffic control method string (e.g. "rl", "tree_method").

    Returns:
        Complete env-params string for one environment.
    """
    # Network params (shared)
    parts = [
        f"--network-seed {config.network_seed}",
        f"--grid_dimension {config.grid_dimension}",
        f"--junctions_to_remove {config.junctions_to_remove}",
        f"--block_size_m {config.block_size_m}",
        f"--lane_count {config.lane_count}",
        f"--step-length {config.step_length}",
        f"--land_use_block_size_m {config.land_use_block_size_m}",
        f"--attractiveness {config.attractiveness}",
        f"--traffic_light_strategy {config.traffic_light_strategy}",
    ]

    # Execution params (shared)
    parts.append(f"--end-time {config.end_time}")

    # Traffic control from caller (not hardcoded)
    parts.append(f"--traffic_control {traffic_control}")

    # Scenario-specific traffic params
    scenario_param_map = {
        "num_vehicles": "--num_vehicles",
        "private_traffic_seed": "--private-traffic-seed",
        "public_traffic_seed": "--public-traffic-seed",
        "routing_strategy": "--routing_strategy",
        "vehicle_types": "--vehicle_types",
        "passenger_routes": "--passenger-routes",
        "departure_pattern": "--departure_pattern",
        "start_time_hour": "--start_time_hour",
    }

    for key, flag in scenario_param_map.items():
        if key in scenario:
            value = scenario[key]
            # Quote string values that contain spaces
            if isinstance(value, str) and " " in value:
                parts.append(f"{flag} '{value}'")
            else:
                parts.append(f"{flag} {value}")

    # Fall back to config-level defaults for params not in scenario
    if "num_vehicles" not in scenario:
        parts.append(f"--num_vehicles {config.num_vehicles}")
    if "routing_strategy" not in scenario:
        parts.append(f"--routing_strategy '{config.routing_strategy}'")
    if "vehicle_types" not in scenario:
        parts.append(f"--vehicle_types '{config.vehicle_types}'")
    if "departure_pattern" not in scenario:
        parts.append(f"--departure_pattern {config.departure_pattern}")
    if "start_time_hour" not in scenario:
        parts.append(f"--start_time_hour {config.start_time_hour}")

    return " ".join(parts)


def find_latest_checkpoint(
    checkpoint_dir: str,
) -> Tuple[Optional[str], int]:
    """Find the latest checkpoint .zip file by parsing step counts from filenames.

    Expects filenames like ``rl_traffic_model_4096_steps.zip``.

    Args:
        checkpoint_dir: Directory to search for checkpoint .zip files.

    Returns:
        A 2-tuple of (path, steps):
        - path: Absolute path to the latest checkpoint, or None if none found.
        - steps: The step count parsed from the filename, or 0 if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    pattern = os.path.join(checkpoint_dir, "*.zip")
    zip_files = glob.glob(pattern)
    if not zip_files:
        return None, 0

    step_re = re.compile(r"_(\d+)_steps\.zip$")
    best_path: Optional[str] = None
    best_steps = 0

    for path in zip_files:
        basename = os.path.basename(path)
        match = step_re.search(basename)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = os.path.abspath(path)

    return best_path, best_steps


def has_checkpoints(experiment_dir: str) -> bool:
    """Check whether an experiment directory contains any checkpoint .zip files.

    Looks in ``<experiment_dir>/checkpoint/`` for .zip files.

    Args:
        experiment_dir: Root directory of the experiment.

    Returns:
        True if at least one .zip file exists in the checkpoint sub-directory.
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        return False
    pattern = os.path.join(checkpoint_dir, "*.zip")
    return len(glob.glob(pattern)) > 0
