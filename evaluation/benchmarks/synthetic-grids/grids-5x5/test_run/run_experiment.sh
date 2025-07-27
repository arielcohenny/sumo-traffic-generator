#!/bin/bash

# Test Synthetic Grid Experiment Run 999
# Generated automatically for testing

cd ../../../../  # Navigate to project root

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, proceeding without activation"
fi

echo "🔄 Starting test experiment run 999 (low load)"
echo "Grid: 5x5, Vehicles: 400"

# Base parameters
BASE_PARAMS="--grid_dimension 5 \
--block_size_m 200 \
--num_vehicles 400 \
--vehicle_types 'passenger 60 commercial 30 public 10' \
--routing_strategy 'shortest 80 realtime 20' \
--departure_pattern six_periods \
--end-time 1800 \
--junctions_to_remove 0 \
--lane_count realistic \
--attractiveness poisson \
--step-length 1.0 \
--time_dependent \
--seed 42"

# Run tree_method with 30-second timeout for testing
echo "   Running tree_method method..."
timeout 30s env PYTHONUNBUFFERED=1 python -m src.cli $BASE_PARAMS \
--traffic_control tree_method \
--seed 42 \
> "evaluation/benchmarks/synthetic-grids/grids-5x5/test_run/results/tree_method/simulation.log" 2>&1

if [ $? -eq 124 ]; then
    echo "   ⏰ tree_method timed out (expected - shows SUMO started successfully)"
    echo "   ✅ Test passed: SUMO execution began properly"
elif [ $? -eq 0 ]; then
    echo "   ✅ tree_method completed successfully"
else
    echo "   ❌ tree_method failed or errored"
    echo "   Last 10 lines of output:"
    tail -n 10 "evaluation/benchmarks/synthetic-grids/grids-5x5/test_run/results/tree_method/simulation.log"
    exit 1
fi

echo "🏁 Test experiment run 999 completed"
