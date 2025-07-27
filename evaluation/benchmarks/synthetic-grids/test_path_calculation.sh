#!/bin/bash

# Test path calculation from the actual grids-5x5 directory
echo "🧪 TESTING PATH CALCULATION FROM GRIDS-5X5"
echo "=========================================="

# Start from grids-5x5 directory
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5
echo "📍 Starting directory: $(pwd)"

# Test path navigation with different levels
echo ""
echo "🔄 Testing 4 levels up (../../../../):"
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5
cd ../../../../
echo "   Result: $(pwd)"
if [ -f "src/cli.py" ]; then
    echo "   ✅ Found src/cli.py - correct path!"
else
    echo "   ❌ No src/cli.py found"
fi

echo ""
echo "🔄 Testing 5 levels up (../../../../../):"
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5
cd ../../../../../
echo "   Result: $(pwd)"
if [ -f "src/cli.py" ]; then
    echo "   ✅ Found src/cli.py"
else
    echo "   ❌ No src/cli.py found"
fi

echo ""
echo "🔄 Testing virtual environment activation:"
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5
cd ../../../../

if [ -f ".venv/bin/activate" ]; then
    echo "   ✅ Virtual environment found"
    source .venv/bin/activate
    echo "   ✅ Virtual environment activated"
    
    echo ""
    echo "🔄 Testing CLI import:"
    python3 -c "import sys; sys.path.append('.'); from src.cli import main; print('✅ CLI import successful')"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ CLI import test passed"
    else
        echo "   ❌ CLI import test failed"
        exit 1
    fi
else
    echo "   ❌ No virtual environment found"
fi

echo ""
echo "🎉 Path calculation test completed!"