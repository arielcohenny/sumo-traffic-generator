# Refactoring Plan: Multi-Scenario RL Training Infrastructure

## Goal

Restructure the RL training system so that:
1. Configuration is cleanly separated into 5 independent concerns (network, scenarios, algorithm, reward, execution)
2. Multiple traffic scenarios run in parallel, each as a separate SUMO environment, training one shared PPO model
3. Local (laptop) and server (HPC) training have dedicated scripts suited to their needs
4. Batch training with `--resume-from` chains multiple scenario batches into one model
5. All RL operational files (configs, scripts, models) live under a top-level `rl/` directory, separate from the engine (`src/rl/`)

## Current State

(Based on `feature/refactor-rl-config` branch after first refactoring — commit `8c403be`)

**Training engine (`src/rl/`):**
- `environment.py` — TrafficControlEnv (Gymnasium env wrapping SUMO)
- `training.py` — PPO training pipeline, `create_vectorized_env()`, `train_rl_policy()`
- `experiment_config.py` — ExperimentConfig dataclass, `load_experiment()`, `validate_experiment()`, `save_experiment()` (created in first refactoring)
- `reward.py` — RewardCalculator, REWARD_REGISTRY with BaseReward/EmpiricalReward/ThroughputReward/MultiObjectiveReward
- `traffic_analysis.py` — RLTrafficAnalyzer with feature registries (EDGE/JUNCTION/NETWORK_FEATURE_REGISTRY)
- `utils.py` — shared softmax() and proportions_to_durations() (created in first refactoring)
- `constants.py` — ~150 hardcoded defaults (PPO hyperparams, feature toggles, training params). These serve as fallback values when no experiment config is provided.
- `controller.py` — RLController for inference (used by traffic controller factory at simulation time)
- `vehicle_tracker.py` — vehicle tracking utilities
- `demonstration_collector.py` — imitation learning data collection

**Broken files (dead imports of non-existent `src.rl.config` / `rl.config`):**
- `train_model.py` — broken, has deprecation notice pointing to `scripts/train_rl_production.py`
- `validate_config.py` — broken, has deprecation notice pointing to `experiment_config.validate_experiment()`

**Fixed in first refactoring (no longer broken):**
- `training_utils.py` — broken imports replaced with deprecation warnings and basic import checks

**Operational files:**
- `scripts/train_rl_production.py` — **only working training entry point**, accepts `--experiment` YAML or raw `--env-params`
- `hpc/train_rl.slurm` — SLURM wrapper for server submission
- `experiments/baseline.yaml` — monolithic config (all params in one flat YAML, created in first refactoring)
- `experiments/throughput_focused.yaml` — inherits from baseline, overrides reward (created in first refactoring)
- `models/` — training output (gitignored)

**Key limitation:** `create_vectorized_env` passes the same `env_params_string` to all parallel environments. All envs see identical traffic scenarios (only differing by random seed).

**Constants vs YAML relationship:** When `experiment_config` is provided, its values are used. When `None`, code falls back to `constants.py` values. This dual-path was established in the first refactoring and is preserved in this plan.

## Target State

### Directory Structure

```
rl/                                  # Top-level RL workspace (operational, not library code)
├── configs/                         # Modular configuration files
│   ├── network/
│   │   └── grid6_realistic.yaml     # Network topology definition
│   ├── scenarios/
│   │   └── heavy_load.yaml          # List of traffic scenario parameter sets
│   ├── algorithm/
│   │   └── ppo_default.yaml         # PPO hyperparameters + architecture
│   ├── reward/
│   │   └── empirical.yaml           # Reward function name + params
│   └── execution/
│       └── long_run.yaml            # Timesteps, checkpoints, batch size, parallel envs
│
├── local/                           # Laptop training scripts
│   └── train.py                     # Single-env, quick iteration, debug-friendly
│
├── server/                          # HPC training scripts
│   ├── train.py                     # Multi-scenario parallel, batch orchestration
│   └── train_rl.slurm              # SLURM job script
│
└── models/                          # Training output (gitignored)
    └── heavy_load_20260201_150000/
        ├── final_model.zip
        ├── resolved_config.yaml     # Full snapshot of all 5 configs merged
        ├── checkpoints/
        └── network/

src/rl/                              # Engine (unchanged, library code)
    ├── environment.py
    ├── training.py
    ├── experiment_config.py
    ├── reward.py
    ├── traffic_analysis.py
    └── ...
```

### Config File Formats

#### `rl/configs/network/grid6_realistic.yaml`
```yaml
network_seed: 24208
grid_dimension: 6
junctions_to_remove: 2
block_size_m: 280
lane_count: realistic
step_length: 1.0
land_use_block_size_m: 25.0
attractiveness: land_use
traffic_light_strategy: partial_opposites
```

#### `rl/configs/scenarios/heavy_load.yaml`
```yaml
# Each entry becomes one parallel SUMO environment.
# All share the same network; only traffic params differ.
scenarios:
  - name: "heavy_realtime_s1"
    num_vehicles: 22000
    private_traffic_seed: 72632
    public_traffic_seed: 27031
    routing_strategy: "realtime 100"
    vehicle_types: "passenger 100"
    passenger_routes: "in 0 out 0 inner 100 pass 0"
    departure_pattern: uniform
    start_time_hour: 8.0

  - name: "heavy_mixed_s1"
    num_vehicles: 25000
    private_traffic_seed: 41893
    public_traffic_seed: 55102
    routing_strategy: "shortest 70 realtime 30"
    vehicle_types: "passenger 90 public 10"
    departure_pattern: uniform
    start_time_hour: 7.0

  # ... more scenarios
```

#### `rl/configs/algorithm/ppo_default.yaml`
```yaml
learning_rate: 1.0e-4
clip_range: 0.2
batch_size: 2048
n_steps: 4096
n_epochs: 5
gamma: 0.995
gae_lambda: 0.98
max_grad_norm: 0.5
ent_coef: 0.01
log_std_init: -1.0
lr_schedule: "exponential"
lr_final: 5.0e-6
lr_decay_rate: 0.99995
ent_schedule: true
ent_initial: 0.02
ent_final: 0.001
ent_decay_steps: 500000
net_arch: [256, 256]
activation: "relu"
```

#### `rl/configs/reward/empirical.yaml`
```yaml
reward_function: "empirical"
reward_params: {}
```

#### `rl/configs/execution/long_run.yaml`
```yaml
total_timesteps: 500000
checkpoint_freq: 50000
early_stopping_patience: 10
early_stopping_min_delta: 70.0
end_time: 7200
parallel_envs: 8
single_env: false
```

### How Configs Compose

The training scripts accept all 5 config files and merge them into one `ExperimentConfig`:

```bash
# Server: multi-scenario parallel training
python rl/server/train.py \
  --network rl/configs/network/grid6_realistic.yaml \
  --scenarios rl/configs/scenarios/heavy_load.yaml \
  --algorithm rl/configs/algorithm/ppo_default.yaml \
  --reward rl/configs/reward/empirical.yaml \
  --execution rl/configs/execution/long_run.yaml

# Resume with new scenarios (same model continues learning)
python rl/server/train.py \
  --network rl/configs/network/grid6_realistic.yaml \
  --scenarios rl/configs/scenarios/heavy_load_batch2.yaml \
  --algorithm rl/configs/algorithm/ppo_default.yaml \
  --reward rl/configs/reward/empirical.yaml \
  --execution rl/configs/execution/long_run.yaml \
  --resume-from rl/models/heavy_load_20260201/final_model.zip

# Local: single-env quick test
python rl/local/train.py \
  --network rl/configs/network/grid6_realistic.yaml \
  --scenarios rl/configs/scenarios/heavy_load.yaml \
  --algorithm rl/configs/algorithm/ppo_default.yaml \
  --reward rl/configs/reward/empirical.yaml \
  --execution rl/configs/execution/quick_test.yaml
```

Any individual param can still be overridden on CLI:
```bash
python rl/server/train.py \
  --network ... --scenarios ... --algorithm ... --reward ... --execution ... \
  --timesteps 200000 \
  --parallel-envs 4
```

Priority: **CLI > execution.yaml > algorithm.yaml > defaults**

### What Each Training Script Does

#### `rl/local/train.py` (laptop)
- Designed for quick iteration and debugging
- Defaults to single env (`DummyVecEnv`)
- Picks first scenario from scenarios file (or all if `--all-scenarios`)
- Shorter default timesteps
- Verbose logging, debug-friendly output
- Uses `rl/models/` for output

#### `rl/server/train.py` (HPC)
- Designed for production parallel training
- Creates one parallel env per scenario (`SubprocVecEnv`)
- If scenarios > available parallel slots, logs a warning (user should split into batches)
- Saves resolved config snapshot for reproducibility
- Integrates with SLURM via `rl/server/train_rl.slurm`
- Uses `rl/models/` for output

---

## Implementation Steps

### Step 1: Create directory structure

Create the `rl/` directory tree:
```
rl/configs/network/
rl/configs/scenarios/
rl/configs/algorithm/
rl/configs/reward/
rl/configs/execution/
rl/local/
rl/server/
rl/models/       (gitignored)
```

Add `rl/models/` to `.gitignore`.

**Verification:** Directories exist, gitignore updated.

### Step 2: Refactor ExperimentConfig to support modular loading

Update `src/rl/experiment_config.py`:

- Add `load_modular_config()` function that accepts 5 separate YAML paths and merges them into one `ExperimentConfig`
- Add `scenarios` field to `ExperimentConfig`: `scenarios: List[Dict] = field(default_factory=list)` — holds the list of per-env traffic parameter dicts
- Add network-specific fields that are currently missing from ExperimentConfig: `junctions_to_remove`, `block_size_m`, `lane_count`, `step_length`, `land_use_block_size_m`, `attractiveness`, `traffic_light_strategy`, `passenger_routes`, `start_time_hour`
- Keep existing `load_experiment()` working for backward compatibility (monolithic YAML still works)
- Add `build_env_params_for_scenario(config, scenario_dict)` function that builds a complete `--env-params` string from network config + one scenario dict. The string must include all params that `create_argument_parser()` expects:
  - From network config: `--network-seed`, `--grid_dimension`, `--junctions_to_remove`, `--block_size_m`, `--lane_count`, `--step-length`, `--land_use_block_size_m`, `--attractiveness`, `--traffic_light_strategy`
  - From scenario dict: `--num_vehicles`, `--private-traffic-seed`, `--public-traffic-seed`, `--routing_strategy`, `--vehicle_types`, `--passenger-routes`, `--departure_pattern`, `--start_time_hour`
  - From execution config: `--end-time` (shared simulation duration across all scenarios — belongs in execution, not per-scenario, since all envs in a batch must run for the same duration for PPO's rollout buffer to work correctly)
  - Hardcoded by builder: `--traffic_control tree_method` (RL training always uses tree_method since the RL agent IS the controller; this is not configurable per-scenario or per-network)
- Add validation: all scenarios must omit network params (those come from network config only)

**Verification:** Unit test — load 5 separate YAML files, get a valid ExperimentConfig with scenarios list populated.

### Step 3: Update training.py for per-scenario env creation

Modify `create_vectorized_env()`:

- Accept `env_params_list: List[str]` (one env-params string per env) instead of single `env_params_string`
- Each `make_env()` call gets its own params string: `env_fns = [make_env(env_params_list[i], i, ...) for i in range(n_envs)]`
- Keep backward compatibility: if a single string is passed (not a list), replicate it N times (existing behavior)
- `n_envs` is derived from `len(env_params_list)` when using scenarios

Modify `train_rl_policy()`:
- Accept `env_params_list: List[str] = None` as alternative to `env_params_string`
- If `env_params_list` is provided, pass it to `create_vectorized_env`
- Network generation uses network params only (shared across all scenarios)
- `_generate_network_once()` is called once with network-only params; the resulting `network_path` is shared by all scenarios (this already works correctly — each env gets `--use-network-from` appended)

Also pass additional existing params through: `cycle_lengths`, `cycle_strategy`, `network_path`, `network_dimensions` must all continue to flow to `make_env()`.

Note: `network_path` is passed as a separate kwarg to `make_env()` → `TrafficControlEnv()`, NOT appended to the env-params string. So the list-based approach works cleanly — each env gets its own params string, but the shared network path is a separate argument.

**Fix pre-existing bug at line ~898:** The eval env creation code references `config.get_cli_args_for_env()` from the deleted `src.rl.config` module. This will crash when `use_parallel and n_envs > SINGLE_ENV_THRESHOLD`. Fix by creating the eval env using `make_env()` with the first scenario's params instead.

**Verification:** Unit test — create vectorized env with 2 different env-params strings, verify each env gets its own params.

### Step 4: Create seed config files

Create initial config files based on the current `baseline.yaml`:

- `rl/configs/network/grid6_realistic.yaml` — the user's specified network params
- `rl/configs/scenarios/heavy_load.yaml` — 2-3 example scenarios
- `rl/configs/algorithm/ppo_default.yaml` — current PPO defaults from baseline.yaml
- `rl/configs/reward/empirical.yaml` — current reward config
- `rl/configs/execution/long_run.yaml` — production execution params
- `rl/configs/execution/quick_test.yaml` — local testing params (small timesteps, single env)

**Verification:** All files parse correctly with `load_modular_config()`.

### Step 5: Create `rl/local/train.py`

Local training script:
- Argument parser with `--network`, `--scenarios`, `--algorithm`, `--reward`, `--execution` (all 5 config paths)
- CLI overrides for common params (`--timesteps`, `--single-env`)
- Defaults to single env, first scenario only
- `--all-scenarios` flag to use all scenarios even locally
- Calls `load_modular_config()` to merge configs
- Builds env-params list from scenarios
- Calls `train_rl_policy()` from `src.rl.training`
- Output goes to `rl/models/`
- Verbose logging suitable for terminal use

**Verification:** Run locally with `--timesteps 100` smoke test — config loads, env creates, training starts and completes.

### Step 6: Create `rl/server/train.py`

Server training script:
- Same 5 config arguments as local
- Defaults to parallel envs (one per scenario)
- `--resume-from` for batch chaining
- `--model-name` for organizing output
- Saves resolved config snapshot in model directory
- Validates scenario count vs parallel env count
- Calls `train_rl_policy()` from `src.rl.training`
- Output goes to `rl/models/`
- Production-grade logging (less verbose, file logging)

**Verification:** Run locally with `--timesteps 100` and 2 scenarios — both envs get different params.

### Step 7: Create `rl/server/train_rl.slurm`

SLURM job script:
- Accepts config paths as `--export` variables
- Activates virtual environment
- Calls `rl/server/train.py` with config paths
- Support for `RESUME_FROM` variable for batch chaining
- Dynamic job naming from config

**Verification:** Script parses without errors (`bash -n`).

### Step 8: Migrate existing files and clean up dead code

- Keep `experiments/baseline.yaml` and `experiments/throughput_focused.yaml` in place as legacy monolithic configs — `load_experiment()` continues to work for backward compatibility. Do NOT move or copy them; new modular configs in `rl/configs/` are the path forward, and old monolithic format remains supported but is not duplicated.
- Move `hpc/train_rl.slurm` to `rl/server/train_rl.slurm`. Remove `hpc/` directory.
- Keep `scripts/train_rl_production.py` working (mark as legacy, points to new scripts) — this is currently the **only working training entry point**
- Delete broken files that import non-existent `src.rl.config` / `rl.config`:
  - `src/rl/train_model.py` — already has deprecation notice, superseded by `scripts/train_rl_production.py`
  - `src/rl/validate_config.py` — already has deprecation notice, superseded by `experiment_config.validate_experiment()`
- Add deprecation notices where appropriate

**Verification:** Old monolithic `load_experiment('experiments/baseline.yaml')` still works. New modular loading also works. Deleted files are not imported anywhere.

### Step 9: Run full test suite

- Run `pytest tests/` — all existing tests must pass (verify actual count at execution time)
- Run local smoke test: `python rl/local/train.py --network ... --scenarios ... --algorithm ... --reward ... --execution ... --timesteps 100`
- Run multi-scenario smoke test: `python rl/server/train.py` with 2 scenarios, verify each env gets different traffic params (check SUMO logs show different vehicle counts / routing strategies)
- Verify `--resume-from` works: run server training, then resume with a different scenarios file

**Verification:** All existing tests pass. All smoke tests complete without error. Resume produces a model that inherits from the first run's checkpoint.

---

## What Does NOT Change

- `src/rl/environment.py` — no changes needed
- `src/rl/training.py` — minimal change (accept list of env-params in addition to single string)
- `src/rl/reward.py` — no changes
- `src/rl/traffic_analysis.py` — no changes
- `src/rl/constants.py` — no changes (remains as fallback defaults when no YAML config is provided)
- `src/rl/controller.py` — RLController for inference, untouched
- `src/rl/vehicle_tracker.py` — vehicle tracking utilities, untouched
- `src/rl/demonstration_collector.py` — imitation learning collector, untouched
- `src/rl/utils.py` — shared utility functions, untouched
- `src/rl/training_utils.py` — model comparison and validation utilities, untouched (broken imports already fixed in first refactoring)
- All existing tests — no changes
- The simulation pipeline (`src/cli.py`, `src/network/`, `src/traffic/`, `src/orchestration/`, `src/sumo_integration/`) — completely decoupled from RL, no changes

## Risk Assessment

- **Observation/action space mismatch across scenarios:** All scenarios share the same network, so edge count and junction count are identical. The obs/action dimensions will match. This is enforced by having a single `network:` config.
- **Backward compatibility:** Monolithic YAML loading (`load_experiment()`) continues to work. New modular loading is additive.
- **SUMO process isolation:** Already handled by `SubprocVecEnv` — each env runs in its own subprocess with its own workspace directory.
