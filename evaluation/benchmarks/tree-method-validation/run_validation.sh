#!/bin/bash

# Tree Method Validation Using Original Datasets
# Tests Tree Method vs Actuated vs Fixed using the original decentralized_traffic_bottleneck datasets
# Exactly as specified in the original Tree Method research

set -e

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
RESULTS_DIR="$EXPERIMENT_DIR/results"
DATASETS_DIR="$PROJECT_ROOT/evaluation/datasets/decentralized_traffic_bottleneck"

echo "Starting Tree Method Validation using Original Datasets"
echo "Project root: $PROJECT_ROOT"
echo "Results dir: $RESULTS_DIR"
echo "Datasets dir: $DATASETS_DIR"

# Create results directory structure
mkdir -p "$RESULTS_DIR/tree_method"
mkdir -p "$RESULTS_DIR/actuated"
mkdir -p "$RESULTS_DIR/fixed"

# Use Experiment1-realistic-high-load (25,470 vehicles, 7,300 seconds)
DATASET_NAME="Experiment1-realistic-high-load"
DATASET_PATH="$DATASETS_DIR/$DATASET_NAME"
NUM_RUNS=20

# Change to project root and activate virtual environment
cd "$PROJECT_ROOT"

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
else
    echo "Warning: No virtual environment found, using system Python"
    PYTHON_CMD="python3"
fi

echo "Running Tree Method validation on $DATASET_NAME dataset..."
echo "Dataset contains 25,470 vehicles, 7,300 second duration"

# Run Tree Method experiments using --tree_method_sample
echo "Running Tree Method experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
        --tree_method_sample "$DATASET_PATH/$i" \
        --traffic_control tree_method \
        --end-time 7300 \
        > "$RESULTS_DIR/tree_method/run_${i}.log" 2>&1
done

# Run SUMO Actuated experiments  
echo "Running SUMO Actuated experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
        --tree_method_sample "$DATASET_PATH/$i" \
        --traffic_control actuated \
        --end-time 7300 \
        > "$RESULTS_DIR/actuated/run_${i}.log" 2>&1
done

# Run Fixed timing experiments
echo "Running Fixed timing experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT" $PYTHON_CMD -m src.cli \
        --tree_method_sample "$DATASET_PATH/$i" \
        --traffic_control fixed \
        --end-time 7300 \
        > "$RESULTS_DIR/fixed/run_${i}.log" 2>&1
done

echo "Tree Method validation completed!"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Run analysis script: python analyze_validation.py"
echo "2. View Tree Method vs Actuated vs Fixed comparison"