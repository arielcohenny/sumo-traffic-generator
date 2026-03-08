# Single-Env Parallel Training Design

## Problem

Currently, RL training on TAU HPC uses multi-scenario parallel environments (e.g., 4 envs sharing one policy via `rl/server/train.py`). We want to run multiple independent single-env training jobs in parallel, each with its own settings and self-contained folder.

## Design

### Two Scripts, Zero Arguments, CWD-Based

**`rl/server/train_single.py`** - Always starts fresh training:
```bash
cd rl/experiments/exp_001/
python ../../server/train_single.py
```

**`rl/server/resume_single.py`** - Always resumes from latest checkpoint:
```bash
cd rl/experiments/exp_001/
python ../../server/resume_single.py
```

### Single Config YAML

Each experiment folder contains one `config.yaml` with all settings:

```yaml
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
  total_timesteps: 500000
  checkpoint_freq: 4096
  eval_freq: 4096
  early_stopping_patience: 50
```

### Self-Contained Folder Structure

After training, each experiment folder contains everything:

```
rl/experiments/exp_001/
├── config.yaml              # Config (user-created)
├── checkpoint/              # All checkpoints
│   ├── rl_traffic_model_4096_steps.zip
│   └── ...
├── best_model/              # Best model from eval
│   └── best_model.zip
├── eval_logs/               # evaluations.npz
├── tensorboard/             # TB logs
├── workspace/               # SUMO simulation files
└── training.log             # Training log
```

### Script Behavior

Both scripts share core logic via `rl/server/config_loader.py`:

1. Read `config.yaml` from `os.getcwd()`
2. Build `env_params` string from network + traffic sections
3. Call `train_rl_policy()` with `n_envs=1`, `models_dir=os.getcwd()`

**Safety checks:**
- `train_single.py`: errors if `checkpoint/` already has `.zip` files
- `resume_single.py`: errors if `checkpoint/` has no `.zip` files
- Both: error if `config.yaml` not found in CWD

**Resume logic:**
- Finds latest checkpoint by step count in filename
- Infers `initial_timesteps` from checkpoint filename (e.g., `_4096_steps.zip` -> 4096)

### Shared Module: `rl/server/config_loader.py`

Handles:
- YAML parsing and validation
- Building env_params string from config sections
- Finding latest checkpoint in a directory
- Extracting timestep count from checkpoint filename

### PBS Submission Example

```bash
# Submit 3 independent training jobs
echo 'cd ~/sumo-traffic-generator/rl/experiments/exp_high_lr && module load Python-3.10.2 && source ../../../.venv/bin/activate && python ../../server/train_single.py' | qsub -q parallel -l walltime=48:00:00,mem=32gb,vmem=40gb,ncpus=8 -N exp_high_lr

echo 'cd ~/sumo-traffic-generator/rl/experiments/exp_low_lr && module load Python-3.10.2 && source ../../../.venv/bin/activate && python ../../server/train_single.py' | qsub -q parallel -l walltime=48:00:00,mem=32gb,vmem=40gb,ncpus=8 -N exp_low_lr

echo 'cd ~/sumo-traffic-generator/rl/experiments/exp_more_vehicles && module load Python-3.10.2 && source ../../../.venv/bin/activate && python ../../server/train_single.py' | qsub -q parallel -l walltime=48:00:00,mem=32gb,vmem=40gb,ncpus=8 -N exp_more_vehicles

# Resume
echo 'cd ~/sumo-traffic-generator/rl/experiments/exp_low_lr && module load Python-3.10.2 && source ../../../.venv/bin/activate && python ../../server/resume_single.py' | qsub -q parallel -l walltime=48:00:00,mem=32gb,vmem=40gb,ncpus=8 -N exp_low_lr_resume
```
