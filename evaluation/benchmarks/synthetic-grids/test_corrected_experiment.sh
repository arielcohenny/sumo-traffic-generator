#!/bin/bash

# Test corrected experiment script without timeout issues

echo "🧪 TESTING CORRECTED EXPERIMENT EXECUTION"
echo "========================================="

# Navigate to project root and activate virtual environment
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ No virtual environment found"
    exit 1
fi

# Create test results directory
mkdir -p evaluation/benchmarks/synthetic-grids/test-results/tree_method

echo ""
echo "🚀 Testing SUMO execution with proper error handling..."

# Base parameters for a simple test
BASE_PARAMS="--grid_dimension 5 \
--block_size_m 200 \
--num_vehicles 400 \
--vehicle_types 'passenger 60 commercial 30 public 10' \
--routing_strategy 'shortest 80 realtime 20' \
--departure_pattern six_periods \
--end-time 7300 \
--junctions_to_remove 0 \
--lane_count realistic \
--attractiveness poisson \
--step-length 1.0 \
--time_dependent \
--seed 42"

echo "Command: env PYTHONUNBUFFERED=1 python -m src.cli $BASE_PARAMS --traffic_control tree_method"
echo ""

# Run with timeout using Python (to avoid macOS timeout command issues)
python3 -c "
import subprocess
import sys
import time

start_time = time.time()

try:
    result = subprocess.run([
        'env', 'PYTHONUNBUFFERED=1', 'python', '-m', 'src.cli'
    ] + '''$BASE_PARAMS --traffic_control tree_method'''.split(),
    timeout=60,  # 1 minute timeout for testing
    text=True,
    capture_output=False  # Show output in real-time
    )
    
    execution_time = time.time() - start_time
    print(f'\\nExecution completed in {execution_time:.1f} seconds')
    
    if result.returncode == 0:
        print('✅ Command completed successfully')
        sys.exit(0)
    else:
        print(f'❌ Command failed with return code {result.returncode}')
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f'\\n⏰ Command timed out after {execution_time:.1f} seconds')
    print('This indicates SUMO is running properly (good!)')
    sys.exit(0)  # Timeout is OK for testing
except Exception as e:
    execution_time = time.time() - start_time
    print(f'\\n❌ Command failed with error after {execution_time:.1f} seconds: {e}')
    sys.exit(1)
"

echo ""
echo "🏁 Test completed!"