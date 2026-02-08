# Executing on TAU HPC Power Cluster

## Project Organization

### Folder Structure

```
rl/
├── configs/                        # Reusable config building blocks (COMMITTED)
│   ├── algorithm/
│   │   └── ppo_default.yaml        # PPO hyperparameters
│   ├── reward/
│   │   └── empirical.yaml          # Reward function config
│   ├── network/
│   │   └── grid6_realistic.yaml    # Network topology
│   ├── scenarios/
│   │   └── exp_090955.yaml         # Traffic scenarios (4 parallel envs)
│   └── execution/
│       └── long_run.yaml           # Timesteps, checkpointing
│
├── experiments/                    # Experiment folders (COMMITTED)
│   └── exp_20260105_090955/
│       ├── config.yaml             # Full resolved config
│       ├── resume.zip              # Best checkpoint (~20MB)
│       └── notes.md                # Results, observations
│
└── models/                         # Training outputs (GITIGNORED)
    └── exp_20260105_090955/
        ├── checkpoint/             # All checkpoints
        ├── eval_logs/              # evaluations.npz
        └── tensorboard/
```

### What Gets Committed vs Gitignored

| Location | Committed? | Contents |
|----------|------------|----------|
| `rl/configs/` | Yes | Reusable config building blocks |
| `rl/experiments/*/config.yaml` | Yes | Experiment configuration |
| `rl/experiments/*/resume.zip` | Yes | Best checkpoint to resume from (~20MB) |
| `rl/experiments/*/notes.md` | Yes | Results and observations |
| `rl/models/` | **No** | All training outputs, full checkpoint history |

### Critical Training Parameters

These parameters were empirically validated. Changing them can collapse training:

| Parameter | Value | Location |
|-----------|-------|----------|
| `learning_rate` | `1e-4` | `rl/configs/algorithm/ppo_default.yaml` |
| `throughput_bonus` | `0.2` | `rl/configs/reward/empirical.yaml` via `reward_params` |

**Example reward config (`rl/configs/reward/empirical.yaml`):**
```yaml
reward_function: "empirical"
reward_params:
  throughput_bonus: 0.2
```

**Warning**: Default `throughput_bonus` in code is `0.1`. Training collapses without explicitly setting `0.2`.

---

## Server Details

- **Host**: `power.tau.ac.il`
- **Account**: `efratbl`
- **Job scheduler**: PBS (qsub/qstat/qdel)
- **Queue**: `parallel` (max 80GB mem, 2400h walltime)
- **VPN required**: Yes (for off-campus access)

## First-Time Setup

### 1. Connect to the server

```bash
ssh efratbl@power.tau.ac.il
```

### 2. Clone the repository

```bash
git clone https://github.com/arielcohenny/sumo-traffic-generator.git
cd sumo-traffic-generator
```

### 3. Load Python 3.10 module

```bash
module load Python-3.10.2
```

The module makes `python3.10` available at `/tmp/efratbl_venv/bin/python3.10`. Note: `python3` still points to the system Python 3.4 — always use `python3.10` explicitly.

### 4. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 5. Install PyTorch (CPU-only)

```bash
pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### 6. Install remaining dependencies

```bash
pip install --no-cache-dir \
  numpy==1.24.4 \
  shapely==2.0.1 \
  pandas==2.0.3 \
  pyarrow==12.0.1 \
  geopandas==0.13.2 \
  networkx==3.1 \
  eclipse-sumo \
  sumolib \
  traci \
  xmltodict \
  alive-progress \
  pytest \
  matplotlib \
  seaborn \
  scipy==1.11.4 \
  gymnasium==0.29.1 \
  stable-baselines3==2.1.0 \
  tqdm \
  rich \
  pyyaml \
  streamlit \
  plotly
```

**Important**: Do NOT use `pip install -r requirements.txt` on the server. The server has old cmake/gcc and cannot compile packages from source. The pinned versions above have pre-built wheels.

**Important**: Always use `--no-cache-dir` to avoid filling the home directory disk quota.

### 7. Create output directories

```bash
mkdir -p logs models rl/models
```

### 8. Verify installation

```bash
python3 -c "import sumolib; print('sumolib OK')"
python3 -c "import stable_baselines3; print('SB3 OK')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import gymnasium; print('gymnasium OK')"
```

### 9. Check SUMO availability

```bash
module avail sumo 2>&1
which sumo || echo "SUMO not available"
```

If SUMO is not available as a module, it may need to be installed or requested from the admin.

## Updating Code on the Server

All code changes go through git. Never edit files directly on the server.

```bash
# On your local machine: commit and push
git add <files>
git commit -m "description"
git push origin main

# On the server: pull latest
cd ~/sumo-traffic-generator
git pull
```

## Running RL Training

Jobs are submitted via **PBS** using `qsub`. The training script is `rl/server/train.py`.

### Submit a training job

```bash
echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/empirical.yaml --execution rl/configs/execution/long_run.yaml --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp090955 -o rl/models/pbs_output.log -e rl/models/pbs_error.log
```

### Resume from checkpoint

```bash
echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/empirical.yaml --execution rl/configs/execution/long_run.yaml --resume-from rl/experiments/exp_20260105_090955/resume.zip --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp090955 -o rl/models/pbs_output.log -e rl/models/pbs_error.log
```

### Running multiple experiments

Submit multiple jobs with different configs — PBS runs them in parallel if resources are available:

```bash
# Experiment 1
echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/empirical.yaml --execution rl/configs/execution/long_run.yaml --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp1 -o rl/models/pbs_exp1_output.log -e rl/models/pbs_exp1_error.log

# Experiment 2 (different reward config)
echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/throughput.yaml --execution rl/configs/execution/long_run.yaml --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp2 -o rl/models/pbs_exp2_output.log -e rl/models/pbs_exp2_error.log
```

## Monitoring Jobs

```bash
# Check job status
qstat -u $USER

# Watch job output in real-time
tail -f rl/models/pbs_output.log

# Check job errors
cat rl/models/pbs_error.log

# Cancel a job
qdel <job_id>

# Cancel all your jobs
qdel $(qstat -u $USER | awk 'NR>5{print $1}')
```

## PBS Queue Reference

| Queue | Max Memory | Max Walltime | Max CPUs | Use Case |
|-------|-----------|-------------|----------|----------|
| `parallel` | 80GB | 2400h | multiple | RL training (recommended) |
| `inf` | 1GB | 2400h | 1 | Single-CPU tasks only |
| `short` | 1.8GB | 60h | 1 | Quick single-CPU tasks |

Check queue limits:
```bash
qstat -Qf <queue_name> | grep -i "resources_max"
```

## Config Files

Experiments are configured through YAML files in `rl/configs/`:

```
rl/configs/
├── network/          # Grid size, lanes, block size
├── scenarios/        # Vehicle count, seeds, parallel envs (one entry per env)
├── algorithm/        # PPO hyperparameters
├── reward/           # Reward function selection and weights
└── execution/        # Timesteps, parallel envs, checkpointing
```

To create a new experiment, copy and modify an existing config file, commit it, push, and pull on the server.

## Troubleshooting

### Disk quota exceeded

```bash
# Check what's using space
du -sh ~/* 2>/dev/null | sort -rh | head -20
du -sh ~/.cache ~/.conda ~/.local 2>/dev/null

# Clean pip cache
rm -rf ~/.cache/pip
pip cache purge 2>/dev/null
```

### Wrong Python version in venv

If `python3 --version` inside the venv shows 3.4, the venv was created with the wrong Python. Delete and recreate:

```bash
deactivate
rm -rf .venv
module load Python-3.10.2
python3.10 -m venv .venv
source .venv/bin/activate
```

### Package build failures

Never use `pip install -r requirements.txt` on the server. The server's cmake (2.8) and gcc are too old to compile packages from source. Use the pinned versions in the setup section above, which have pre-built wheels.

---

## Experiment Workflows

### Creating a New Experiment

1. **Create experiment folder locally:**
   ```bash
   mkdir -p rl/experiments/exp_YYYYMMDD_description
   ```

2. **Create config.yaml** (references building blocks):
   ```yaml
   # rl/experiments/exp_20260206_grid6_heavy/config.yaml
   name: "grid6_heavy_traffic"
   description: "6x6 grid with 22k vehicles, empirical reward"

   # Reference config files
   network: "../configs/network/grid6_realistic.yaml"
   scenarios: "../configs/scenarios/exp_090955.yaml"
   algorithm: "../configs/algorithm/ppo_default.yaml"
   reward: "../configs/reward/empirical.yaml"
   execution: "../configs/execution/long_run.yaml"
   ```

3. **Add resume checkpoint** (if continuing from existing training):
   ```bash
   cp <path-to-checkpoint>.zip rl/experiments/exp_YYYYMMDD_description/resume.zip
   ```

4. **Create notes.md:**
   ```markdown
   # Experiment: grid6_heavy_traffic

   ## Goal
   Train RL agent on 6x6 grid with heavy traffic.

   ## Parameters
   - Vehicles: 22,000
   - Grid: 6x6
   - Reward: empirical with throughput_bonus=0.2

   ## Results
   - Started: 2026-02-06
   - Best reward: (pending)
   ```

5. **Commit and push:**
   ```bash
   git add rl/experiments/exp_YYYYMMDD_description/
   git commit -m "Add experiment: grid6_heavy_traffic"
   git push
   ```

6. **On TAU, pull and run:**
   ```bash
   git pull
   echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/empirical.yaml --execution rl/configs/execution/long_run.yaml --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp_name -o rl/models/pbs_output.log -e rl/models/pbs_error.log
   ```

### Continuing an Existing Experiment

1. **On TAU**: After training produces a better checkpoint, identify the best one by checking results.

2. **Update resume.zip on TAU:**
   ```bash
   cp rl/models/exp_YYYYMMDD_description/checkpoint/rl_traffic_model_XXXXX_steps.zip \
      rl/experiments/exp_YYYYMMDD_description/resume.zip
   ```

3. **Commit and push from TAU:**
   ```bash
   git add rl/experiments/exp_YYYYMMDD_description/resume.zip
   git commit -m "Update checkpoint: XXXXX steps, reward=YYY"
   git push
   ```

4. **Pull locally** to get the updated checkpoint:
   ```bash
   git pull
   ```

### Checking Training Results

Training results are stored in `eval_logs/evaluations.npz`. Check them with:

```bash
python -c "
import numpy as np
data = np.load('rl/models/<experiment>/eval_logs/evaluations.npz')
timesteps = data['timesteps']
results = data['results']

print('Last 15 evaluations:')
for i in range(-15, 0):
    if abs(i) <= len(timesteps):
        print(f'  Step {timesteps[i]:>7}: mean={results[i].mean():.1f}, std={results[i].std():.1f}')
"
```

**What to look for:**

| Mean Reward | Status |
|-------------|--------|
| ~110+ | Good - training is working |
| ~85 | Bad - training collapsed (check parameters) |
| Gradually increasing | Excellent - model is learning |

### Syncing Between Local and TAU

**Local → TAU** (code changes, new experiments):
```bash
# Local
git add <files>
git commit -m "description"
git push

# TAU
git pull
```

**TAU → Local** (updated checkpoints):
```bash
# TAU (after training)
git add rl/experiments/*/resume.zip
git commit -m "Update checkpoint: XXX steps"
git push

# Local
git pull
```

**Large files** (full checkpoint history, not in git):
```bash
# Copy from TAU to local
scp -r efratbl@power.tau.ac.il:~/sumo-traffic-generator/rl/models/exp_XXX ./rl/models/

# Copy from local to TAU
scp -r ./rl/models/exp_XXX efratbl@power.tau.ac.il:~/sumo-traffic-generator/rl/models/
```

---

## Quick Reference

### Connect and Pull
```bash
ssh efratbl@power.tau.ac.il && cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && git pull
```

### Submit exp_090955 Training (4 parallel envs, resume from checkpoint)
```bash
echo 'cd ~/sumo-traffic-generator && module load Python-3.10.2 && source .venv/bin/activate && python rl/server/train.py --network rl/configs/network/grid6_realistic.yaml --scenarios rl/configs/scenarios/exp_090955.yaml --algorithm rl/configs/algorithm/ppo_default.yaml --reward rl/configs/reward/empirical.yaml --execution rl/configs/execution/long_run.yaml --resume-from rl/experiments/exp_20260105_090955/resume.zip --models-dir rl/models' | qsub -q parallel -l walltime=48:00:00,mem=32gb,ncpus=8 -N exp090955 -o rl/models/pbs_output.log -e rl/models/pbs_error.log
```

### Check Results
```bash
python -c "import numpy as np; d=np.load('rl/models/rl_<timestamp>/eval_logs/evaluations.npz'); [print(f'Step {s}: mean={r.mean():.1f}') for s,r in zip(d['timesteps'][-10:],d['results'][-10:])]"
```

### Monitor Job
```bash
qstat -u $USER

tail -f rl/models/pbs_output.log

qdel <job_id>
```
