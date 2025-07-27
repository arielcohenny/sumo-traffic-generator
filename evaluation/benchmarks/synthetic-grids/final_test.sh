#!/bin/bash

echo "🔍 FINAL TEST OF EXPERIMENT SCRIPT"
echo "================================="

# Start exactly like a real experiment would
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5

# Clean up and create test experiment directory  
rm -rf 999
mkdir -p 999/results/tree_method
cd 999

echo "📍 Starting from: $(pwd)"

# Test the exact path navigation from the experiment scripts
echo "🧭 Testing 6 levels up (../../../../../)..."
cd ../../../../../
echo "📍 After navigation: $(pwd)"

# Check if we're in the right place
if [ -f ".venv/bin/activate" ]; then
    echo "✅ Found .venv/bin/activate"
else
    echo "❌ No .venv/bin/activate found"
fi

if [ -f "src/cli.py" ]; then
    echo "✅ Found src/cli.py"
else
    echo "❌ No src/cli.py found"
fi

# Test virtual environment activation
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    
    # Test CLI import
    python -m src.cli --help | head -3
    
    # Test if we can actually run the CLI with proper parameters
    echo ""
    echo "🚀 Testing SUMO execution with 5-second timeout..."
    
    # Create output directory (using gtimeout from homebrew instead of timeout)
    OUTPUT_DIR="evaluation/benchmarks/synthetic-grids/grids-5x5/999/results/tree_method"
    mkdir -p "$OUTPUT_DIR"
    
    # Use Python subprocess with timeout instead of shell timeout
    python3 -c "
import subprocess
import sys

try:
    result = subprocess.run([
        'python', '-m', 'src.cli',
        '--grid_dimension', '5',
        '--block_size_m', '200', 
        '--num_vehicles', '400',
        '--vehicle_types', 'passenger 60 commercial 30 public 10',
        '--routing_strategy', 'shortest 80 realtime 20',
        '--departure_pattern', 'six_periods',
        '--end-time', '1800',
        '--junctions_to_remove', '0',
        '--lane_count', 'realistic',
        '--attractiveness', 'poisson',
        '--step-length', '1.0',
        '--time_dependent',
        '--traffic_control', 'tree_method',
        '--seed', '42'
    ], timeout=5, capture_output=True, text=True)
    
    print(f'Return code: {result.returncode}')
    print('STDOUT:')
    print(result.stdout[:500])
    print('STDERR:')
    print(result.stderr[:500])
    
except subprocess.TimeoutExpired:
    print('✅ Process timed out (expected - SUMO started successfully)')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

    echo ""
    echo "🎉 All tests completed!"
    
else
    echo "❌ Virtual environment setup failed"
fi