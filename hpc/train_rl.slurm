#!/bin/bash
# =============================================================================
# LEGACY: SLURM Job Script for RL Training on TAU HPC Power Cluster
# Superseded by: rl/server/train_rl.slurm (multi-scenario parallel training)
# =============================================================================
#
# Usage:
#   sbatch --export=EXPERIMENT=experiments/baseline.yaml train_rl.slurm
#   sbatch --export=EXPERIMENT=experiments/throughput_focused.yaml,EXPERIMENT_NAME=throughput train_rl.slurm
#   sbatch --export=TIMESTEPS=500000 train_rl.slurm  # Legacy: uses --env-params
#
# Monitor:
#   squeue -u $USER                                 # Check job status
#   tail -f logs/train_rl_<job_id>.log             # Watch output
#   scancel <job_id>                               # Cancel job
# =============================================================================

#SBATCH --job-name=rl_${EXPERIMENT_NAME:-traffic}
#SBATCH --account=public-efratbl_v2
#SBATCH --partition=power-general-public-pool
#SBATCH --qos=public

# CPU Configuration (RL training is CPU-bound due to SUMO simulation)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Time limit (format: D-HH:MM:SS)
#SBATCH --time=2-00:00:00

# Output files
#SBATCH --output=logs/train_rl_%j.log
#SBATCH --error=logs/train_rl_%j.err

# Email notifications (optional - uncomment and set your email)
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@tau.ac.il

# =============================================================================
# Configuration (can be overridden via --export)
# =============================================================================
# Experiment config file (preferred - single source of truth)
EXPERIMENT="${EXPERIMENT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-traffic}"

# Training overrides (apply to both experiment and legacy modes)
TIMESTEPS="${TIMESTEPS:-}"
PARALLEL_ENVS="${PARALLEL_ENVS:-4}"
MODEL_NAME="${MODEL_NAME:-rl_traffic_hpc}"

# Legacy mode variables (used only when EXPERIMENT is not set)
GRID_DIM="${GRID_DIM:-5}"
NUM_VEHICLES="${NUM_VEHICLES:-4500}"
NETWORK_SEED="${NETWORK_SEED:-42}"
END_TIME="${END_TIME:-7200}"

# =============================================================================
# Environment Setup
# =============================================================================
echo "==========================================="
echo "RL Training Job Starting"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"
echo "==========================================="

# Set project directory
PROJECT_DIR="${PROJECT_DIR:-$HOME/sumo-traffic-generator}"
cd "$PROJECT_DIR"

# Create logs and models directories (must exist for SLURM output files)
mkdir -p logs models

# Load modules
echo "Loading modules..."
module purge
module load python 2>/dev/null || echo "WARNING: Could not load python module"
module load sumo 2>/dev/null || echo "WARNING: Could not load sumo module"

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_DIR/.venv/bin/activate"

# Set SUMO environment
if [ -z "$SUMO_HOME" ] && command -v sumo &> /dev/null; then
    export SUMO_HOME="$(dirname $(dirname $(which sumo)))"
fi

# Set PyTorch threading for multiprocessing compatibility
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMPY_DISABLE_THREADING=1
export NPY_DISABLE_LONGDOUBLE_FPFFLAGS=1

# Enable real-time log monitoring
export PYTHONUNBUFFERED=1

# =============================================================================
# Verify Environment
# =============================================================================
echo ""
echo "Environment verification..."
echo "Python: $(python --version)"
echo "SUMO_HOME: ${SUMO_HOME:-not set}"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import stable_baselines3; print('stable-baselines3: OK')"
python -c "import sumolib; print('sumolib: OK')"

# =============================================================================
# Build Training Command
# =============================================================================
echo ""
echo "Training Configuration:"
echo "  Parallel Envs: $PARALLEL_ENVS"

# Build base command
TRAIN_CMD="python scripts/train_rl_production.py \
    --parallel-envs $PARALLEL_ENVS \
    --model-name $MODEL_NAME \
    --models-dir models \
    --tensorboard \
    --checkpoint-freq 10000 \
    --save-every 50000 \
    --log-level INFO"

if [ -n "$EXPERIMENT" ]; then
    # Experiment-driven mode: YAML is the single source of truth
    echo "  Mode: experiment-driven"
    echo "  Experiment: $EXPERIMENT"
    TRAIN_CMD="$TRAIN_CMD --experiment $EXPERIMENT"
else
    # Legacy mode: build --env-params from shell variables
    echo "  Mode: legacy (env-params)"
    echo "  Grid Dimension: $GRID_DIM"
    echo "  Vehicles: $NUM_VEHICLES"
    echo "  Network Seed: $NETWORK_SEED"
    echo "  End Time: $END_TIME seconds"

    ENV_PARAMS="--network-seed $NETWORK_SEED \
--grid_dimension $GRID_DIM \
--num_vehicles $NUM_VEHICLES \
--end-time $END_TIME \
--traffic_control rl \
--lane_count realistic \
--routing_strategy \"shortest 70 realtime 30\" \
--vehicle_types \"passenger 90 public 10\" \
--departure_pattern six_periods"

    TRAIN_CMD="$TRAIN_CMD --env-params \"$ENV_PARAMS\""
fi

# Add timesteps override if provided
if [ -n "$TIMESTEPS" ]; then
    echo "  Timesteps override: $TIMESTEPS"
    TRAIN_CMD="$TRAIN_CMD --timesteps $TIMESTEPS"
fi

echo ""

# =============================================================================
# Run Training
# =============================================================================
echo "Starting RL training..."
echo "==========================================="

eval $TRAIN_CMD

TRAIN_EXIT_CODE=$?

# =============================================================================
# Cleanup and Summary
# =============================================================================
echo ""
echo "==========================================="
echo "Training Complete"
echo "==========================================="
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Models saved in: $PROJECT_DIR/models/"
    ls -la "$PROJECT_DIR/models/" | tail -10
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs/train_rl_${SLURM_JOB_ID}.err for errors"
fi

exit $TRAIN_EXIT_CODE
