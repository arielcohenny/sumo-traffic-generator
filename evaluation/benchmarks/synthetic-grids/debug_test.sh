#!/bin/bash

echo "🔍 DEBUGGING EXPERIMENT SCRIPT EXECUTION"
echo "========================================"

# Create test directory exactly like the real experiments
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5

# Clean up any existing test
rm -rf 999
mkdir -p 999/results/tree_method

cd 999

echo "📍 Current directory: $(pwd)"

# Test path navigation
echo "🧭 Testing path navigation..."
cd ../../../../
echo "📍 After cd ../../../../: $(pwd)"

# Check for required files
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

echo ""
echo "🔄 Testing virtual environment activation..."
source .venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "🧪 Testing basic CLI execution..."
python -m src.cli --help | head -3

echo ""
echo "🚀 Testing actual command structure..."
echo "This is the command that would be executed:"
echo "env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 200 --num_vehicles 400 --vehicle_types 'passenger 60 commercial 30 public 10' --routing_strategy 'shortest 80 realtime 20' --departure_pattern six_periods --end-time 1800 --junctions_to_remove 0 --lane_count realistic --attractiveness poisson --step-length 1.0 --time_dependent --traffic_control tree_method --seed 42"

echo ""
echo "📂 Output directory would be:"
echo "evaluation/benchmarks/synthetic-grids/grids-5x5/999/results/tree_method/simulation.log"

# Check if output directory path exists
OUTPUT_DIR="evaluation/benchmarks/synthetic-grids/grids-5x5/999/results/tree_method"
if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ Output directory exists: $OUTPUT_DIR"
else
    echo "❌ Output directory missing: $OUTPUT_DIR"
    echo "   Creating it now..."
    mkdir -p "$OUTPUT_DIR"
fi

echo ""
echo "🎯 Testing short SUMO execution (10 seconds)..."
timeout 10s env PYTHONUNBUFFERED=1 python -m src.cli \
--grid_dimension 5 \
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
--traffic_control tree_method \
--seed 42 > "$OUTPUT_DIR/simulation.log" 2>&1 &

# Wait for the command and capture its exit code
wait $!
EXIT_CODE=$?

echo "Command exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 124 ]; then
    echo "✅ Command timed out (expected - SUMO started successfully)"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Command completed successfully"
else
    echo "❌ Command failed with exit code $EXIT_CODE"
fi

# Check what was written to the log
if [ -f "$OUTPUT_DIR/simulation.log" ]; then
    LOG_SIZE=$(wc -c < "$OUTPUT_DIR/simulation.log")
    echo "✅ Log file created, size: $LOG_SIZE bytes"
    
    if [ $LOG_SIZE -gt 0 ]; then
        echo "📄 First 10 lines of log:"
        head -10 "$OUTPUT_DIR/simulation.log"
        echo ""
        echo "📄 Last 10 lines of log:"
        tail -10 "$OUTPUT_DIR/simulation.log"
    else
        echo "⚠️  Log file is empty"
    fi
else
    echo "❌ No log file created"
fi

echo ""
echo "🏁 Debug test completed!"