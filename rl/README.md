# RL Training

Reinforcement learning pipeline for learning traffic signal control policies using the SUMO simulator as a Gymnasium environment (PPO via Stable-Baselines3).

## Directory Structure

```
rl/
├── local/          # Local development training
├── server/         # TAU HPC server training and SLURM scripts
├── scripts/        # Analysis, demonstration collection, and utility scripts
├── configs/        # Configuration files (demonstration collection configs)
├── experiments/    # Self-contained experiment folders
└── models/         # Training outputs (gitignored)
```

## Configuration

Training is configured via 5 modular YAML files:

| Config | Purpose | Example |
|--------|---------|---------|
| `network` | Grid topology, lanes, zones | `configs/network/grid6_realistic.yaml` |
| `scenarios` | Traffic patterns, vehicle counts, seeds | `configs/scenarios/exp_090955.yaml` |
| `algorithm` | PPO hyperparameters, network architecture | `configs/algorithm/ppo_default.yaml` |
| `reward` | Reward function and weights | `configs/reward/empirical.yaml` |
| `execution` | Timesteps, checkpoint frequency, early stopping | `configs/execution/long_run.yaml` |

## Training Modes

### Local (development)

```bash
python rl/local/train.py \
  --network rl/configs/network/grid6_realistic.yaml \
  --scenarios rl/configs/scenarios/heavy_load.yaml \
  --algorithm rl/configs/algorithm/ppo_default.yaml \
  --reward rl/configs/reward/empirical.yaml \
  --execution rl/configs/execution/quick_test.yaml
```

### Server (TAU HPC)

```bash
python rl/server/train.py \
  --network rl/configs/network/grid6_realistic.yaml \
  --scenarios rl/configs/scenarios/exp_090955.yaml \
  --algorithm rl/configs/algorithm/ppo_default.yaml \
  --reward rl/configs/reward/empirical.yaml \
  --execution rl/configs/execution/long_run.yaml \
  --models-dir rl/models
```

Supports multi-scenario parallel training (multiple environments with different traffic patterns), resume from checkpoint (`--resume-from`), and pre-trained initialization (`--pretrain-from`).

### Self-contained experiment

```bash
# Copy template and edit config
cp -r rl/experiments/example rl/experiments/my_exp
nano rl/experiments/my_exp/config.yaml

# Train (all output stays in experiment folder)
cd rl/experiments/my_exp
python ../../server/train_single.py

# Resume anytime
python ../../server/resume_single.py
```

Experiment folders use a single unified `config.yaml` instead of 5 separate files.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_tree_method_demonstrations.py` | Collect expert trajectories from Tree Method |
| `scripts/collect_demonstrations_multiprocess.py` | Parallel demonstration collection |
| `scripts/pretrain_from_demonstrations.py` | Behavioral cloning from collected demonstrations |
| `scripts/run_reward_validation_training.py` | Validate reward function with short training runs |
| `scripts/analyze_reward_validation.py` | Analyze reward component correlations |
| `server/compare_checkpoints.py` | Evaluate all checkpoints on same scenario |

## Imitation Learning Workflow

1. Collect expert demonstrations from Tree Method algorithm
2. Pre-train RL policy via behavioral cloning
3. Fine-tune with PPO for further improvement

```bash
# Collect
python rl/scripts/collect_tree_method_demonstrations.py --scenarios 500 --base-seed 42

# Pre-train
python rl/scripts/pretrain_from_demonstrations.py --input <demo_file>.npz

# Fine-tune
python rl/server/train.py ... --pretrain-from <pretrained_model>.zip
```
