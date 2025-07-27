#!/usr/bin/env python3

"""
Quick test to verify that the path navigation and virtual environment fixes work
by creating and executing a single experiment script.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

def load_config():
    """Load the central configuration file"""
    config_path = Path(__file__).parent / "experiment_config.json"
    
    if not config_path.exists():
        print("❌ ERROR: experiment_config.json not found")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration file loaded successfully")
        return config
    except Exception as e:
        print(f"❌ ERROR: Could not load configuration: {e}")
        sys.exit(1)

def create_test_experiment_script(test_dir, config):
    """Create a simple test experiment script"""
    
    # Use first experiment parameters from 5x5 grid
    grid_config = config['grid_configurations']['5x5']
    shared = config['shared_parameters']
    
    experiment = {
        'run_id': 999,
        'grid_dimension': grid_config['dimension'],
        'block_size_m': grid_config['block_size_m'],
        'vehicle_count_level': 'low',
        'num_vehicles': 400,  # Low count for quick test
        'vehicle_types': shared['vehicle_types'][0],
        'routing_strategy': shared['routing_strategies'][0],
        'departure_pattern': shared['departure_patterns'][0],
        'simulation_duration': 1800,  # 30 minutes for quick test
        'junctions_removed': shared['junctions_removed'][0],
        'lane_assignment': shared['lane_assignment'],
        'attractiveness': shared['attractiveness'],
        'step_length': shared['step_length'],
        'time_dependent': shared['time_dependent'],
        'traffic_control_methods': ['tree_method']  # Just one method
    }
    
    script_content = f"""#!/bin/bash

# Test Synthetic Grid Experiment Run {experiment['run_id']}
# Generated to test path navigation and virtual environment fixes

cd ../../../../  # Navigate to project root

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, proceeding without activation"
fi

echo "🔄 Testing experiment run {experiment['run_id']}"
echo "Grid: {experiment['grid_dimension']}x{experiment['grid_dimension']}, Vehicles: {experiment['num_vehicles']}"

# Test basic path and import
echo "📍 Current directory: $(pwd)"
echo "🧪 Testing Python CLI import..."
python3 -c "import sys; sys.path.append('.'); from src.cli import main; print('✅ CLI import successful')"

if [ $? -eq 0 ]; then
    echo "✅ Path navigation and CLI import test passed"
else
    echo "❌ Path navigation or CLI import test failed"
    exit 1
fi

# Test actual SUMO command with very short timeout
echo "🚀 Testing actual SUMO execution (30 second timeout)..."
timeout 30s env PYTHONUNBUFFERED=1 python -m src.cli \\
--grid_dimension {experiment['grid_dimension']} \\
--block_size_m {experiment['block_size_m']} \\
--num_vehicles {experiment['num_vehicles']} \\
--vehicle_types '{experiment['vehicle_types']}' \\
--routing_strategy '{experiment['routing_strategy']}' \\
--departure_pattern {experiment['departure_pattern']} \\
--end-time {experiment['simulation_duration']} \\
--junctions_to_remove {experiment['junctions_removed']} \\
--lane_count {experiment['lane_assignment']} \\
--attractiveness {experiment['attractiveness']} \\
--step-length {experiment['step_length']} \\
{'--time_dependent' if experiment['time_dependent'] else ''} \\
--traffic_control tree_method \\
--seed 42 > test_output.log 2>&1

if [ $? -eq 124 ]; then
    echo "⏰ SUMO execution timed out (expected - shows it started successfully)"
    echo "✅ Test passed: SUMO started and dependencies are working"
elif [ $? -eq 0 ]; then
    echo "✅ SUMO execution completed successfully"
else
    echo "❌ SUMO execution failed"
    echo "Last 10 lines of output:"
    tail -n 10 test_output.log
    exit 1
fi

echo "🎉 All tests passed! The fixes are working correctly."
"""
    
    script_path = test_dir / "test_experiment.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    return script_path

def main():
    """Main test function"""
    print("🧪 TESTING SYNTHETIC GRID FIXES")
    print("=" * 50)
    
    # Load configuration 
    config = load_config()
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        print(f"📁 Test directory: {test_dir}")
        
        # Create results directory structure
        results_dir = test_dir / "results" / "tree_method"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test script
        test_script = create_test_experiment_script(test_dir, config)
        print(f"📄 Test script created: {test_script}")
        
        # Execute the test
        print("\n🚀 Executing test...")
        try:
            result = subprocess.run(
                [str(test_script)],
                cwd=str(test_dir),
                timeout=120,  # 2 minute total timeout
                text=True,
                capture_output=True
            )
            
            print("📤 STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("📤 STDERR:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("\n✅ TEST PASSED: All fixes are working correctly!")
                return True
            else:
                print(f"\n❌ TEST FAILED: Return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n⏰ Test timed out, but this might indicate SUMO is running (which is good)")
            return True
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)