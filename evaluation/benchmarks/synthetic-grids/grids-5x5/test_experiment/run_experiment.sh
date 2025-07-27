#!/bin/bash

# Test Synthetic Grid Experiment
# Generated using the corrected script format

cd ../../../../../  # Navigate to project root

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

echo "🔄 Starting test experiment (light load)"
echo "Grid: 5x5, Vehicles: 400"

# Run tree_method with corrected argument format
echo "   Running tree_method method..."
SEED=42

# Run with proper argument quoting and timeout
python3 -c "
import subprocess
import sys
import time

start_time = time.time()

try:
    result = subprocess.run([
        'env', 'PYTHONUNBUFFERED=1', 'python', '-m', 'src.cli',
        '--grid_dimension', '5',
        '--block_size_m', '200',
        '--num_vehicles', '400',
        '--vehicle_types', 'passenger 60 commercial 30 public 10',
        '--routing_strategy', 'shortest 80 realtime 20',
        '--departure_pattern', 'six_periods',
        '--end-time', '7300',
        '--junctions_to_remove', '0',
        '--lane_count', 'realistic',
        '--attractiveness', 'poisson',
        '--step-length', '1.0',
        '--time_dependent',
        '--traffic_control', 'tree_method',
        '--seed', '42'
    ],
    timeout=120,  # 2 minute timeout for testing
    text=True,
    cwd='.',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
    )
    
    execution_time = time.time() - start_time
    
    # Write output to log file
    with open('evaluation/benchmarks/synthetic-grids/grids-5x5/test_experiment/results/tree_method/simulation.log', 'w') as f:
        f.write(result.stdout)
    
    if result.returncode == 0:
        print(f'   ✅ tree_method completed successfully in {execution_time:.1f} seconds')
        sys.exit(0)
    else:
        print(f'   ❌ tree_method failed with return code {result.returncode} after {execution_time:.1f} seconds')
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f'   ⏰ tree_method timed out after {execution_time:.1f} seconds')
    sys.exit(124)
except Exception as e:
    execution_time = time.time() - start_time
    print(f'   ❌ tree_method failed with error after {execution_time:.1f} seconds: {e}')
    sys.exit(1)
"

# Check the exit code from Python script
if [ $? -eq 0 ]; then
    echo "   ✅ tree_method completed successfully"
elif [ $? -eq 124 ]; then
    echo "   ⏰ tree_method timed out"
else
    echo "   ❌ tree_method failed - check log for details"
    if [ -f "evaluation/benchmarks/synthetic-grids/grids-5x5/test_experiment/results/tree_method/simulation.log" ]; then
        echo "   Last 3 lines of log:"
        tail -n 3 "evaluation/benchmarks/synthetic-grids/grids-5x5/test_experiment/results/tree_method/simulation.log"
    fi
fi

echo "🏁 Test experiment completed"