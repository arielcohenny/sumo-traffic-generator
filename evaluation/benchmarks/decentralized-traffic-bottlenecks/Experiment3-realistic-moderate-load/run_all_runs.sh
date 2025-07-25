#!/bin/bash

# Run All Runs for Experiment3-realistic-moderate-load
# Executes all 20 individual runs in sequence

set -e

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")

echo "Running all runs for $EXPERIMENT_NAME"
echo "This will execute 20 individual experiments (Tree Method + Actuated + Fixed each)"
echo "WARNING: This will take a very long time!"
echo ""

read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run all individual experiments
for run in {1..20}; do
    echo "==============================================="
    echo "Starting Run $run/20 for $EXPERIMENT_NAME"
    echo "==============================================="
    
    cd "$EXPERIMENT_DIR/$run"
    ./run_experiment.sh
    
    echo "Completed Run $run/20"
    echo ""
done

echo "All runs completed for $EXPERIMENT_NAME!"
echo "Run analysis with: python analyze_experiment.py"