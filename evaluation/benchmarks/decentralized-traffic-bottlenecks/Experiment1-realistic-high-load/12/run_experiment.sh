#!/bin/bash

# Individual Run Experiment for Decentralized Traffic Bottleneck
# Runs Tree Method vs Actuated vs Fixed on specific dataset

set -e

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/../../../../.." && pwd)"

# Extract experiment and run info from path
EXPERIMENT=$(basename "$(dirname "$RUN_DIR")")
RUN_NUM=$(basename "$RUN_DIR")

# Build dataset path
DATASET_PATH="$PROJECT_ROOT/evaluation/datasets/decentralized_traffic_bottleneck/$EXPERIMENT/$RUN_NUM"
RESULTS_DIR="$RUN_DIR/results"

echo "Running Decentralized Traffic Bottleneck Benchmark"
echo "Experiment: $EXPERIMENT"
echo "Run: $RUN_NUM"
echo "Dataset: $DATASET_PATH"
echo "Results: $RESULTS_DIR"

# Verify dataset exists
if [ ! -f "$DATASET_PATH/simulation.sumocfg.xml" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please verify the dataset directory exists."
    exit 1
fi

# Change to project root and activate virtual environment
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYTHON_CMD="python"
else
    echo "Warning: No virtual environment found, using system Python"
    PYTHON_CMD="python3"
fi

# Get simulation duration from dataset
END_TIME=$(grep -o 'end value="[0-9]*"' "$DATASET_PATH/simulation.sumocfg.xml" | grep -o '[0-9]*')
echo "Simulation duration: $END_TIME seconds"

# Set seed for reproducible results (base + run number)
SEED=234467
echo "Using seed: $SEED"

# Run Tree Method
echo "Running Tree Method..."
env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
    --tree_method_sample "$DATASET_PATH" \
    --traffic_control tree_method \
    --end-time $END_TIME \
    --seed $SEED \
    > "$RESULTS_DIR/tree_method/simulation.log" 2>&1

echo "Tree Method completed."

# Run SUMO Actuated
echo "Running SUMO Actuated..."
env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
    --tree_method_sample "$DATASET_PATH" \
    --traffic_control actuated \
    --end-time $END_TIME \
    --seed $SEED \
    > "$RESULTS_DIR/actuated/simulation.log" 2>&1

echo "Actuated completed."

# Run Fixed timing
echo "Running Fixed timing..."
env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
    --tree_method_sample "$DATASET_PATH" \
    --traffic_control fixed \
    --end-time $END_TIME \
    --seed $SEED \
    > "$RESULTS_DIR/fixed/simulation.log" 2>&1

echo "Fixed completed."

echo "Experiment completed successfully!"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To analyze results: python analyze_results.py"
