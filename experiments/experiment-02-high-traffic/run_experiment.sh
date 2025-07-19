#!/bin/bash

# Experiment 02: High Traffic Load Comparison
# Based on Tree Method experimental design from Traffic Control paper
# Compares Tree Method vs SUMO Actuated vs Fixed vs Random traffic control

set -e

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
RESULTS_DIR="$EXPERIMENT_DIR/results"

echo "Starting Experiment 02: High Traffic Load"
echo "Project root: $PROJECT_ROOT"
echo "Results dir: $RESULTS_DIR"

# Create results directory structure
mkdir -p "$RESULTS_DIR/tree_method"
mkdir -p "$RESULTS_DIR/actuated"
mkdir -p "$RESULTS_DIR/fixed"
mkdir -p "$RESULTS_DIR/random"

# Experiment parameters (high traffic load)
GRID_DIMENSION=5
BLOCK_SIZE=200
NUM_VEHICLES=1200  # High traffic load
END_TIME=7200  # 2 hours (7200 seconds)
STEP_LENGTH=1.0
NUM_RUNS=20

# Change to project root and activate virtual environment
cd "$PROJECT_ROOT"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found, using system Python"
fi

echo "Running $NUM_RUNS experiments for each traffic control method..."

# Run Tree Method experiments
echo "Running Tree Method experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 python3 -m src.cli \
        --grid_dimension $GRID_DIMENSION \
        --block_size_m $BLOCK_SIZE \
        --num_vehicles $NUM_VEHICLES \
        --end-time $END_TIME \
        --step-length $STEP_LENGTH \
        --traffic_control tree_method \
        --seed $i \
        --departure_pattern uniform \
        > "$RESULTS_DIR/tree_method/run_${i}.log" 2>&1
done

# Run SUMO Actuated experiments
echo "Running SUMO Actuated experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 python3 -m src.cli \
        --grid_dimension $GRID_DIMENSION \
        --block_size_m $BLOCK_SIZE \
        --num_vehicles $NUM_VEHICLES \
        --end-time $END_TIME \
        --step-length $STEP_LENGTH \
        --traffic_control actuated \
        --seed $i \
        --departure_pattern uniform \
        > "$RESULTS_DIR/actuated/run_${i}.log" 2>&1
done

# Run Fixed timing experiments
echo "Running Fixed timing experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 python3 -m src.cli \
        --grid_dimension $GRID_DIMENSION \
        --block_size_m $BLOCK_SIZE \
        --num_vehicles $NUM_VEHICLES \
        --end-time $END_TIME \
        --step-length $STEP_LENGTH \
        --traffic_control fixed \
        --seed $i \
        --departure_pattern uniform \
        > "$RESULTS_DIR/fixed/run_${i}.log" 2>&1
done

# Run Random experiments (using random routing strategy as proxy)
echo "Running Random experiments..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS"
    env PYTHONUNBUFFERED=1 python3 -m src.cli \
        --grid_dimension $GRID_DIMENSION \
        --block_size_m $BLOCK_SIZE \
        --num_vehicles $NUM_VEHICLES \
        --end-time $END_TIME \
        --step-length $STEP_LENGTH \
        --traffic_control fixed \
        --routing_strategy "shortest 25 realtime 25 fastest 25 attractiveness 25" \
        --seed $i \
        --departure_pattern uniform \
        > "$RESULTS_DIR/random/run_${i}.log" 2>&1
done

echo "Experiment 02 completed!"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Run analysis script: python analyze_results.py"
echo "2. View comparative metrics and plots"