#!/bin/bash

# Synthetic Grid Experiment Run 154
# Generated automatically from experiment_config.json

cd ../../../../../  # Navigate to project root

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

echo "🔄 Starting experiment run 154 (moderate load)"
echo "Grid: 5x5, Vehicles: 800"

# Run each traffic control method

echo "   Running tree_method method..."
SEED=$((42 + 0))

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
        '--num_vehicles', '800',
        '--vehicle_types', 'passenger 60 commercial 30 public 10',
        '--routing_strategy', 'shortest 30 realtime 50 fastest 20',
        '--departure_pattern', 'uniform',
        '--end-time', '86400',
        '--junctions_to_remove', '0',
        '--lane_count', 'realistic',
        '--attractiveness', 'poisson',
        '--step-length', '1.0',
        '--time_dependent',
        '--traffic_control', 'tree_method',
        '--seed', str($SEED)
    ],
    timeout=3600,
    text=True,
    cwd='.',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
    )
    
    execution_time = time.time() - start_time
    
    # Write output to log file
    with open('evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/tree_method/simulation.log', 'w') as f:
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
    if [ -f "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/tree_method/simulation.log" ]; then
        echo "   Last 3 lines of log:"
        tail -n 3 "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/tree_method/simulation.log"
    fi
fi

echo "   Running actuated method..."
SEED=$((42 + 20))

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
        '--num_vehicles', '800',
        '--vehicle_types', 'passenger 60 commercial 30 public 10',
        '--routing_strategy', 'shortest 30 realtime 50 fastest 20',
        '--departure_pattern', 'uniform',
        '--end-time', '86400',
        '--junctions_to_remove', '0',
        '--lane_count', 'realistic',
        '--attractiveness', 'poisson',
        '--step-length', '1.0',
        '--time_dependent',
        '--traffic_control', 'actuated',
        '--seed', str($SEED)
    ],
    timeout=3600,
    text=True,
    cwd='.',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
    )
    
    execution_time = time.time() - start_time
    
    # Write output to log file
    with open('evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/actuated/simulation.log', 'w') as f:
        f.write(result.stdout)
    
    if result.returncode == 0:
        print(f'   ✅ actuated completed successfully in {execution_time:.1f} seconds')
        sys.exit(0)
    else:
        print(f'   ❌ actuated failed with return code {result.returncode} after {execution_time:.1f} seconds')
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f'   ⏰ actuated timed out after {execution_time:.1f} seconds')
    sys.exit(124)
except Exception as e:
    execution_time = time.time() - start_time
    print(f'   ❌ actuated failed with error after {execution_time:.1f} seconds: {e}')
    sys.exit(1)
"

# Check the exit code from Python script
if [ $? -eq 0 ]; then
    echo "   ✅ actuated completed successfully"
elif [ $? -eq 124 ]; then
    echo "   ⏰ actuated timed out"
else
    echo "   ❌ actuated failed - check log for details"
    if [ -f "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/actuated/simulation.log" ]; then
        echo "   Last 3 lines of log:"
        tail -n 3 "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/actuated/simulation.log"
    fi
fi

echo "   Running fixed method..."
SEED=$((42 + 40))

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
        '--num_vehicles', '800',
        '--vehicle_types', 'passenger 60 commercial 30 public 10',
        '--routing_strategy', 'shortest 30 realtime 50 fastest 20',
        '--departure_pattern', 'uniform',
        '--end-time', '86400',
        '--junctions_to_remove', '0',
        '--lane_count', 'realistic',
        '--attractiveness', 'poisson',
        '--step-length', '1.0',
        '--time_dependent',
        '--traffic_control', 'fixed',
        '--seed', str($SEED)
    ],
    timeout=3600,
    text=True,
    cwd='.',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
    )
    
    execution_time = time.time() - start_time
    
    # Write output to log file
    with open('evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/fixed/simulation.log', 'w') as f:
        f.write(result.stdout)
    
    if result.returncode == 0:
        print(f'   ✅ fixed completed successfully in {execution_time:.1f} seconds')
        sys.exit(0)
    else:
        print(f'   ❌ fixed failed with return code {result.returncode} after {execution_time:.1f} seconds')
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f'   ⏰ fixed timed out after {execution_time:.1f} seconds')
    sys.exit(124)
except Exception as e:
    execution_time = time.time() - start_time
    print(f'   ❌ fixed failed with error after {execution_time:.1f} seconds: {e}')
    sys.exit(1)
"

# Check the exit code from Python script
if [ $? -eq 0 ]; then
    echo "   ✅ fixed completed successfully"
elif [ $? -eq 124 ]; then
    echo "   ⏰ fixed timed out"
else
    echo "   ❌ fixed failed - check log for details"
    if [ -f "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/fixed/simulation.log" ]; then
        echo "   Last 3 lines of log:"
        tail -n 3 "evaluation/benchmarks/synthetic-grids/grids-5x5/154/results/fixed/simulation.log"
    fi
fi

echo "🏁 Experiment run 154 completed"
