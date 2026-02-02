#!/bin/bash
#SBATCH --job-name=rl_${JOB_NAME:-traffic}
#SBATCH --output=rl/models/slurm_%j.log
#SBATCH --error=rl/models/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================================
# RL Training SLURM Script (Multi-Scenario Parallel Training)
# ============================================================================
#
# Usage:
#   # Basic training with modular configs:
#   sbatch --export=ALL,\
#     NETWORK=rl/configs/network/grid6_realistic.yaml,\
#     SCENARIOS=rl/configs/scenarios/heavy_load.yaml,\
#     ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
#     REWARD=rl/configs/reward/empirical.yaml,\
#     EXECUTION=rl/configs/execution/long_run.yaml,\
#     JOB_NAME=heavy_load \
#     rl/server/train_rl.slurm
#
#   # Resume from checkpoint (batch chaining):
#   sbatch --export=ALL,\
#     NETWORK=rl/configs/network/grid6_realistic.yaml,\
#     SCENARIOS=rl/configs/scenarios/heavy_load_batch2.yaml,\
#     ALGORITHM=rl/configs/algorithm/ppo_default.yaml,\
#     REWARD=rl/configs/reward/empirical.yaml,\
#     EXECUTION=rl/configs/execution/long_run.yaml,\
#     RESUME_FROM=rl/models/rl_20260201_150000/final_model.zip,\
#     JOB_NAME=heavy_load_b2 \
#     rl/server/train_rl.slurm
#
# Required environment variables:
#   NETWORK    - Path to network config YAML
#   SCENARIOS  - Path to scenarios config YAML
#   ALGORITHM  - Path to algorithm config YAML
#   REWARD     - Path to reward config YAML
#   EXECUTION  - Path to execution config YAML
#
# Optional environment variables:
#   RESUME_FROM  - Path to model checkpoint to resume from
#   TIMESTEPS    - Override total timesteps
#   JOB_NAME     - Custom job name suffix
# ============================================================================

echo "=== RL Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Network: ${NETWORK}"
echo "Scenarios: ${SCENARIOS}"
echo "Algorithm: ${ALGORITHM}"
echo "Reward: ${REWARD}"
echo "Execution: ${EXECUTION}"
echo "Resume from: ${RESUME_FROM:-none}"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Ensure output directory exists
mkdir -p rl/models

# Build command
CMD="python rl/server/train.py \
    --network ${NETWORK} \
    --scenarios ${SCENARIOS} \
    --algorithm ${ALGORITHM} \
    --reward ${REWARD} \
    --execution ${EXECUTION} \
    --models-dir rl/models"

# Add optional arguments
if [ -n "${RESUME_FROM}" ]; then
    CMD="${CMD} --resume-from ${RESUME_FROM}"
fi

if [ -n "${TIMESTEPS}" ]; then
    CMD="${CMD} --timesteps ${TIMESTEPS}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

EXIT_CODE=$?
echo ""
echo "=== Job Complete ==="
echo "Exit code: ${EXIT_CODE}"
echo "Date: $(date)"

exit ${EXIT_CODE}
