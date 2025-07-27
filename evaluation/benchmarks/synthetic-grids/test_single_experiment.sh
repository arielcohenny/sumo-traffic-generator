#!/bin/bash

# Manual test script for debugging single experiment
# This helps verify that the path navigation and commands work correctly

echo "🧪 TESTING SINGLE EXPERIMENT EXECUTION"
echo "======================================="

# Create test directory
TEST_DIR="/tmp/synthetic_grids_test"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "📁 Test directory: $TEST_DIR"

# Calculate path to project root
PROJECT_ROOT="/Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator"
echo "🎯 Project root: $PROJECT_ROOT"

# Test navigation
echo "🧭 Testing path navigation..."
cd "$PROJECT_ROOT"
if [ $? -eq 0 ]; then
    echo "✅ Successfully navigated to project root"
    pwd
else
    echo "❌ Failed to navigate to project root"
    exit 1
fi

# Test basic CLI command
echo "🚀 Testing basic CLI command..."
python3 -c "import sys; sys.path.append('.'); from src.cli import main; print('✅ CLI import successful')"

if [ $? -eq 0 ]; then
    echo "✅ CLI import test passed"
else
    echo "❌ CLI import test failed"
    exit 1
fi

# Test actual SUMO command (dry run)
echo "🔄 Testing SUMO command structure..."
echo "Command would be:"
echo "env PYTHONUNBUFFERED=1 python -m src.cli \\"
echo "  --grid_dimension 5 \\"
echo "  --block_size_m 200 \\"
echo "  --num_vehicles 400 \\"
echo "  --vehicle_types 'passenger 60 commercial 30 public 10' \\"
echo "  --routing_strategy 'shortest 80 realtime 20' \\"
echo "  --departure_pattern six_periods \\"
echo "  --end-time 7300 \\"
echo "  --junctions_to_remove 0 \\"
echo "  --lane_count realistic \\"
echo "  --attractiveness poisson \\"
echo "  --step-length 1.0 \\"
echo "  --time_dependent \\"
echo "  --traffic_control tree_method \\"
echo "  --seed 42"

echo ""
echo "🧪 Test completed successfully!"
echo "The path navigation and command structure appear to be correct."

# Clean up
rm -rf "$TEST_DIR"