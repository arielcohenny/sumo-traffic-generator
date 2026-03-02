# Single-Env Parallel Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create two zero-argument scripts (`train_single.py`, `resume_single.py`) that read a single `config.yaml` from CWD and run/resume single-env RL training with all outputs in the same folder.

**Architecture:** A shared `config_loader.py` module handles YAML parsing, converting the unified config format into an `ExperimentConfig`, building env_params strings, and checkpoint discovery. Two thin scripts import from it and call `train_rl_policy()`.

**Tech Stack:** Python, PyYAML, stable-baselines3 (existing), ExperimentConfig dataclass (existing)

---

### Task 1: Create `rl/server/config_loader.py` - YAML parsing and config conversion

**Files:**
- Create: `rl/server/config_loader.py`

**Step 1: Write the config_loader module**

```python
"""
Config loader for single-env training.

Reads a unified config.yaml (network + traffic + algorithm + reward + execution
in one file) and converts it to an ExperimentConfig for train_rl_policy().
"""

import glob
import os
import re
import sys

import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rl.experiment_config import ExperimentConfig, validate_experiment


def load_single_config(config_path: str) -> ExperimentConfig:
    """Load a unified config.yaml and return an ExperimentConfig.

    The unified format has sections: network, traffic, algorithm, reward, execution.
    These get flattened into ExperimentConfig fields.

    Args:
        config_path: Path to config.yaml

    Returns:
        Fully validated ExperimentConfig
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Flatten sections into a single dict matching ExperimentConfig fields
    merged = {}

    # Network section -> ExperimentConfig simulation fields
    network = raw.get("network", {})
    network_field_map = {
        "grid_dimension": "grid_dimension",
        "junctions_to_remove": "junctions_to_remove",
        "block_size_m": "block_size_m",
        "lane_count": "lane_count",
        "step_length": "step_length",
        "land_use_block_size_m": "land_use_block_size_m",
        "attractiveness": "attractiveness",
        "traffic_light_strategy": "traffic_light_strategy",
        "network_seed": "network_seed",
    }
    for yaml_key, config_key in network_field_map.items():
        if yaml_key in network:
            merged[config_key] = network[yaml_key]

    # Traffic section -> scenario dict + config defaults
    traffic = raw.get("traffic", {})
    traffic_field_map = {
        "num_vehicles": "num_vehicles",
        "routing_strategy": "routing_strategy",
        "vehicle_types": "vehicle_types",
        "passenger_routes": "passenger_routes",
        "departure_pattern": "departure_pattern",
        "start_time_hour": "start_time_hour",
        "end_time": "end_time",
        "traffic_control": None,  # handled separately
    }
    # Build a scenario dict from traffic params (for build_env_params_for_scenario)
    scenario = {}
    for yaml_key, config_key in traffic_field_map.items():
        if yaml_key in traffic:
            if config_key is not None:
                merged[config_key] = traffic[yaml_key]
            scenario[yaml_key] = traffic[yaml_key]

    # Seed fields go into scenario
    for seed_key in ["private_traffic_seed", "public_traffic_seed"]:
        if seed_key in traffic:
            scenario[seed_key] = traffic[seed_key]

    # Store the traffic_control value for env_params building
    traffic_control = traffic.get("traffic_control", "rl")

    # Algorithm section -> PPO hyperparameters
    algorithm = raw.get("algorithm", {})
    algo_field_map = {
        "learning_rate": "learning_rate",
        "clip_range": "clip_range",
        "batch_size": "batch_size",
        "n_steps": "n_steps",
        "n_epochs": "n_epochs",
        "gamma": "gamma",
        "gae_lambda": "gae_lambda",
        "max_grad_norm": "max_grad_norm",
        "ent_coef": "ent_coef",
        "log_std_init": "log_std_init",
        "net_arch": "net_arch",
        "activation": "activation",
        "lr_schedule": "lr_schedule",
        "lr_final": "lr_final",
        "lr_decay_rate": "lr_decay_rate",
        "ent_schedule": "ent_schedule",
        "ent_initial": "ent_initial",
        "ent_final": "ent_final",
        "ent_decay_steps": "ent_decay_steps",
    }
    for yaml_key, config_key in algo_field_map.items():
        if yaml_key in algorithm:
            merged[config_key] = algorithm[yaml_key]

    # Reward section
    reward = raw.get("reward", {})
    if "function" in reward:
        merged["reward_function"] = reward["function"]
    reward_params = {k: v for k, v in reward.items() if k != "function"}
    if reward_params:
        merged["reward_params"] = reward_params

    # Execution section
    execution = raw.get("execution", {})
    exec_field_map = {
        "total_timesteps": "total_timesteps",
        "checkpoint_freq": "checkpoint_freq",
        "eval_freq": "checkpoint_freq",  # eval_freq maps to checkpoint_freq
        "early_stopping_patience": "early_stopping_patience",
        "early_stopping_min_delta": "early_stopping_min_delta",
    }
    for yaml_key, config_key in exec_field_map.items():
        if yaml_key in execution:
            merged[config_key] = execution[yaml_key]

    # No scenarios for single-env training
    merged["scenarios"] = []

    # Filter to valid ExperimentConfig fields only
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    config = ExperimentConfig(**filtered)
    validate_experiment(config)

    return config, scenario, traffic_control


def build_env_params_string(config: ExperimentConfig, scenario: dict, traffic_control: str) -> str:
    """Build the env_params CLI string from config + scenario + traffic_control.

    Similar to build_env_params_for_scenario but uses the traffic_control
    from the config instead of hardcoding tree_method.

    Args:
        config: ExperimentConfig with network params
        scenario: Dict of traffic params
        traffic_control: Traffic control method (e.g., "rl")

    Returns:
        Complete env-params string for the environment.
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

    # Execution params
    parts.append(f"--end-time {config.end_time}")

    # Traffic control from config (not hardcoded)
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
            if isinstance(value, str) and " " in value:
                parts.append(f"{flag} '{value}'")
            else:
                parts.append(f"{flag} {value}")

    # Fall back to config defaults for missing params
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


def find_latest_checkpoint(checkpoint_dir: str) -> tuple:
    """Find the latest checkpoint .zip file by step count.

    Args:
        checkpoint_dir: Path to the checkpoint/ directory

    Returns:
        Tuple of (checkpoint_path, timesteps) or (None, 0) if none found.
    """
    pattern = os.path.join(checkpoint_dir, "rl_traffic_model_*_steps.zip")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    # Extract step counts and find the maximum
    best_path = None
    best_steps = 0

    for cp in checkpoints:
        match = re.search(r'rl_traffic_model_(\d+)_steps\.zip', os.path.basename(cp))
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = cp

    return best_path, best_steps


def has_checkpoints(experiment_dir: str) -> bool:
    """Check if the experiment directory has any checkpoint .zip files."""
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        return False
    pattern = os.path.join(checkpoint_dir, "*.zip")
    return len(glob.glob(pattern)) > 0
```

**Step 2: Commit**

```bash
git add rl/server/config_loader.py
git commit -m "Add config_loader module for single-env training"
```

---

### Task 2: Create `rl/server/train_single.py` - Fresh training script

**Files:**
- Create: `rl/server/train_single.py`

**Step 1: Write the train_single script**

```python
#!/usr/bin/env python3
"""
Single-env RL training script.

Run from inside an experiment folder containing a config.yaml:

    cd rl/experiments/my_experiment/
    python ../../server/train_single.py

Reads config.yaml from the current directory.
Writes all outputs (checkpoints, models, logs) to the current directory.
Errors if checkpoints already exist (use resume_single.py to continue).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMPY_DISABLE_THREADING'] = '1'
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import logging
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rl.server.config_loader import (
    load_single_config,
    build_env_params_string,
    has_checkpoints,
)
from src.rl.training import train_rl_policy
from src.rl.experiment_config import save_experiment


def setup_logging(log_file: str = None):
    """Setup production logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.getLogger('src.rl.controller').setLevel(logging.WARNING)
    logging.getLogger('src.rl.environment').setLevel(logging.WARNING)
    logging.getLogger('TrafficControlEnv').setLevel(logging.WARNING)
    logging.getLogger('src.rl.training').setLevel(logging.INFO)


def main():
    experiment_dir = os.getcwd()
    config_path = os.path.join(experiment_dir, "config.yaml")

    # Validate: config.yaml must exist
    if not os.path.isfile(config_path):
        print(f"ERROR: No config.yaml found in {experiment_dir}")
        print("Create a config.yaml in this directory before running.")
        sys.exit(1)

    # Safety: refuse to overwrite existing checkpoints
    if has_checkpoints(experiment_dir):
        print(f"ERROR: Checkpoints already exist in {experiment_dir}/checkpoint/")
        print("Use resume_single.py to continue training, or remove checkpoints to start fresh.")
        sys.exit(1)

    # Setup logging to experiment directory
    setup_logging(log_file=os.path.join(experiment_dir, "training.log"))
    logger = logging.getLogger(__name__)

    logger.info(f"Starting fresh training in: {experiment_dir}")

    # Load config
    config, scenario, traffic_control = load_single_config(config_path)
    env_params_string = build_env_params_string(config, scenario, traffic_control)

    logger.info(f"env_params: {env_params_string}")
    logger.info(f"total_timesteps: {config.total_timesteps}")
    logger.info(f"checkpoint_freq: {config.checkpoint_freq}")

    # Save resolved config for reproducibility
    save_experiment(config, os.path.join(experiment_dir, "experiment.yaml"))

    # Train
    model = train_rl_policy(
        env_params_string=env_params_string,
        total_timesteps=config.total_timesteps,
        checkpoint_freq=config.checkpoint_freq,
        use_parallel=False,
        n_envs=1,
        resume_from_model=None,
        cycle_lengths=config.cycle_lengths,
        cycle_strategy=config.cycle_strategy,
        experiment_config=config,
        env_params_list=None,
        models_dir=experiment_dir,
        initial_timesteps_override=None,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add rl/server/train_single.py
git commit -m "Add train_single.py for fresh single-env training"
```

---

### Task 3: Create `rl/server/resume_single.py` - Resume training script

**Files:**
- Create: `rl/server/resume_single.py`

**Step 1: Write the resume_single script**

```python
#!/usr/bin/env python3
"""
Resume single-env RL training from latest checkpoint.

Run from inside an experiment folder that has existing checkpoints:

    cd rl/experiments/my_experiment/
    python ../../server/resume_single.py

Reads config.yaml from the current directory.
Finds the latest checkpoint in checkpoint/ and resumes from it.
Errors if no checkpoints found (use train_single.py to start fresh).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMPY_DISABLE_THREADING'] = '1'
os.environ['NPY_DISABLE_LONGDOUBLE_FPFFLAGS'] = '1'

import logging
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rl.server.config_loader import (
    load_single_config,
    build_env_params_string,
    find_latest_checkpoint,
)
from src.rl.training import train_rl_policy
from src.rl.experiment_config import save_experiment


def setup_logging(log_file: str = None):
    """Setup production logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.getLogger('src.rl.controller').setLevel(logging.WARNING)
    logging.getLogger('src.rl.environment').setLevel(logging.WARNING)
    logging.getLogger('TrafficControlEnv').setLevel(logging.WARNING)
    logging.getLogger('src.rl.training').setLevel(logging.INFO)


def main():
    experiment_dir = os.getcwd()
    config_path = os.path.join(experiment_dir, "config.yaml")

    # Validate: config.yaml must exist
    if not os.path.isfile(config_path):
        print(f"ERROR: No config.yaml found in {experiment_dir}")
        print("Create a config.yaml in this directory before running.")
        sys.exit(1)

    # Find latest checkpoint
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    checkpoint_path, initial_timesteps = find_latest_checkpoint(checkpoint_dir)

    if checkpoint_path is None:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}")
        print("Use train_single.py to start fresh training first.")
        sys.exit(1)

    # Setup logging (append to existing log)
    setup_logging(log_file=os.path.join(experiment_dir, "training.log"))
    logger = logging.getLogger(__name__)

    logger.info(f"Resuming training in: {experiment_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Initial timesteps: {initial_timesteps}")

    # Load config
    config, scenario, traffic_control = load_single_config(config_path)
    env_params_string = build_env_params_string(config, scenario, traffic_control)

    logger.info(f"env_params: {env_params_string}")
    logger.info(f"total_timesteps: {config.total_timesteps}")

    # Train (resume)
    model = train_rl_policy(
        env_params_string=env_params_string,
        total_timesteps=config.total_timesteps,
        checkpoint_freq=config.checkpoint_freq,
        use_parallel=False,
        n_envs=1,
        resume_from_model=checkpoint_path,
        cycle_lengths=config.cycle_lengths,
        cycle_strategy=config.cycle_strategy,
        experiment_config=config,
        env_params_list=None,
        models_dir=experiment_dir,
        initial_timesteps_override=initial_timesteps,
    )

    logger.info("Resumed training complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add rl/server/resume_single.py
git commit -m "Add resume_single.py for resuming single-env training"
```

---

### Task 4: Create example config and verify integration

**Files:**
- Create: `rl/experiments/example/config.yaml`

**Step 1: Create example experiment folder with config**

```yaml
# Example config for single-env training.
# Copy this folder, modify settings, and run train_single.py from inside it.

# --- Network ---
network:
  grid_dimension: 6
  junctions_to_remove: 2
  block_size_m: 280
  lane_count: realistic
  step_length: 1.0
  land_use_block_size_m: 25.0
  attractiveness: land_use
  traffic_light_strategy: partial_opposites
  network_seed: 24208

# --- Traffic / Environment ---
traffic:
  num_vehicles: 22000
  routing_strategy: "realtime 100"
  vehicle_types: "passenger 100"
  passenger_routes: "in 0 out 0 inner 100 pass 0"
  departure_pattern: uniform
  private_traffic_seed: 72632
  public_traffic_seed: 27031
  start_time_hour: 8.0
  end_time: 7300
  traffic_control: rl

# --- Algorithm (PPO) ---
algorithm:
  learning_rate: 1.0e-4
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  net_arch: [256, 256]

# --- Reward ---
reward:
  function: empirical
  throughput_bonus: 0.2

# --- Execution ---
execution:
  total_timesteps: 40960
  checkpoint_freq: 4096
  eval_freq: 4096
  early_stopping_patience: 50
```

**Step 2: Verify config loads correctly (dry run)**

Run from project root:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from rl.server.config_loader import load_single_config, build_env_params_string
config, scenario, tc = load_single_config('rl/experiments/example/config.yaml')
print('Config loaded OK')
print(f'  grid_dimension: {config.grid_dimension}')
print(f'  learning_rate: {config.learning_rate}')
print(f'  total_timesteps: {config.total_timesteps}')
print(f'  traffic_control: {tc}')
print(f'  scenario: {scenario}')
env_params = build_env_params_string(config, scenario, tc)
print(f'  env_params: {env_params}')
"
```

Expected: Config loads without errors, env_params string matches expected format.

**Step 3: Commit**

```bash
git add rl/experiments/example/config.yaml
git commit -m "Add example config for single-env training"
```

---

### Task 5: Integration test - run a short training locally

**Step 1: Run a quick training from the example folder**

```bash
cd rl/experiments/example/
python ../../server/train_single.py
```

Expected: Training starts, creates checkpoint/, best_model/, eval_logs/, workspace/ in the experiment folder. Runs for 40960 timesteps (short test).

**Step 2: Test resume**

```bash
cd rl/experiments/example/
python ../../server/resume_single.py
```

Expected: Finds latest checkpoint, prints the step count, resumes training.

**Step 3: Test safety checks**

```bash
# Should error: checkpoints already exist
cd rl/experiments/example/
python ../../server/train_single.py
# Expected: ERROR: Checkpoints already exist...

# Should error: no config.yaml
cd /tmp
python /path/to/rl/server/train_single.py
# Expected: ERROR: No config.yaml found...
```

**Step 4: Commit any fixes from testing**

```bash
git add -u
git commit -m "Fix integration issues from single-env training test"
```
