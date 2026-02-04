# Executing on TAU HPC Power Cluster

## Server Details

- **Host**: `power.tau.ac.il`
- **Account**: `efratbl`
- **SLURM account**: `public-efratbl_v2`
- **Partitions**: `power-general-public-pool`, `power-general-shared-pool`
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
git push origin develop

# On the server: pull latest
cd ~/sumo-traffic-generator
git pull
```

## Running RL Training

### Submit a training job

Training jobs are submitted via SLURM using modular config files:

```bash
cd ~/sumo-traffic-generator
source .venv/bin/activate

sbatch --account=public-efratbl_v2 \
  --partition=power-general-public-pool \
  --export=ALL,\
NETWORK=rl/configs/network/grid6_realistic.yaml,\
SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
REWARD=rl/configs/reward/empirical.yaml,\
EXECUTION=rl/configs/execution/long_run.yaml,\
JOB_NAME=my_experiment \
  rl/server/train_rl.slurm
```

### Quick test run

```bash
sbatch --account=public-efratbl_v2 \
  --partition=power-general-public-pool \
  --export=ALL,\
NETWORK=rl/configs/network/grid6_realistic.yaml,\
SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
REWARD=rl/configs/reward/empirical.yaml,\
EXECUTION=rl/configs/execution/quick_test.yaml,\
JOB_NAME=quick_test \
  rl/server/train_rl.slurm
```

### Resume from checkpoint

```bash
sbatch --account=public-efratbl_v2 \
  --partition=power-general-public-pool \
  --export=ALL,\
NETWORK=rl/configs/network/grid6_realistic.yaml,\
SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
REWARD=rl/configs/reward/empirical.yaml,\
EXECUTION=rl/configs/execution/long_run.yaml,\
RESUME_FROM=rl/models/my_model/final_model.zip,\
JOB_NAME=resumed \
  rl/server/train_rl.slurm
```

### Running multiple experiments in parallel

Submit multiple jobs with different configs:

```bash
# Experiment 1: empirical reward
sbatch --account=public-efratbl_v2 \
  --partition=power-general-public-pool \
  --export=ALL,\
NETWORK=rl/configs/network/grid6_realistic.yaml,\
SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
REWARD=rl/configs/reward/empirical.yaml,\
EXECUTION=rl/configs/execution/long_run.yaml,\
JOB_NAME=empirical \
  rl/server/train_rl.slurm

# Experiment 2: throughput reward (create the config first)
sbatch --account=public-efratbl_v2 \
  --partition=power-general-public-pool \
  --export=ALL,\
NETWORK=rl/configs/network/grid6_realistic.yaml,\
SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
REWARD=rl/configs/reward/throughput.yaml,\
EXECUTION=rl/configs/execution/long_run.yaml,\
JOB_NAME=throughput \
  rl/server/train_rl.slurm
```

Each `sbatch` submits an independent job. SLURM runs them in parallel if resources are available.

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f rl/models/slurm_<job_id>.log

# Check job errors
cat rl/models/slurm_<job_id>.err

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Config Files

Experiments are configured through YAML files in `rl/configs/`:

```
rl/configs/
├── network/          # Grid size, lanes, block size
├── scenarios/        # Vehicle count, simulation time, routing
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
