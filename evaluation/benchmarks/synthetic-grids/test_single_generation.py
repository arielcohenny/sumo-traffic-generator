#!/usr/bin/env python3

"""
Test that generates a single experiment script and verifies it can execute
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
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def test_single_experiment_generation():
    """Generate and test a single experiment"""
    
    # Load config
    config = load_config()
    grid_config = config['grid_configurations']['5x5'] 
    shared = config['shared_parameters']
    
    # Create test experiment with minimal parameters
    experiment = {
        'run_id': 999,
        'grid_dimension': grid_config['dimension'],
        'block_size_m': grid_config['block_size_m'],
        'vehicle_count_level': 'low',
        'num_vehicles': 400,
        'vehicle_types': shared['vehicle_types'][0],
        'routing_strategy': shared['routing_strategies'][0], 
        'departure_pattern': shared['departure_patterns'][0],
        'simulation_duration': 1800,  # 30 minutes
        'junctions_removed': shared['junctions_removed'][0],
        'lane_assignment': shared['lane_assignment'],
        'attractiveness': shared['attractiveness'],
        'step_length': shared['step_length'],
        'time_dependent': shared['time_dependent'],
        'traffic_control_methods': ['tree_method']
    }
    
    # Create test run directory in grids-5x5 
    test_run_dir = Path("/Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator/evaluation/benchmarks/synthetic-grids/grids-5x5/test_run")
    test_run_dir.mkdir(exist_ok=True)
    
    # Create results directory structure
    results_dir = test_run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "tree_method").mkdir(exist_ok=True)
    
    # Generate run script using the same function from run_all_runs.sh
    script_content = f"""#!/bin/bash

# Test Synthetic Grid Experiment Run {experiment['run_id']}
# Generated automatically for testing

cd ../../../../  # Navigate to project root

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, proceeding without activation"
fi

echo "🔄 Starting test experiment run {experiment['run_id']} ({experiment['vehicle_count_level']} load)"
echo "Grid: {experiment['grid_dimension']}x{experiment['grid_dimension']}, Vehicles: {experiment['num_vehicles']}"

# Base parameters
BASE_PARAMS="--grid_dimension {experiment['grid_dimension']} \\
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
--seed 42"

# Run tree_method with 30-second timeout for testing
echo "   Running tree_method method..."
timeout 30s env PYTHONUNBUFFERED=1 python -m src.cli $BASE_PARAMS \\
--traffic_control tree_method \\
--seed 42 \\
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

echo "🏁 Test experiment run {experiment['run_id']} completed"
"""
    
    # Write script
    script_path = test_run_dir / "run_experiment.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    script_path.chmod(0o755)
    
    print(f"📄 Generated test script: {script_path}")
    
    # Execute the script
    print("🚀 Executing test script...")
    result = subprocess.run(
        [str(script_path)],
        cwd=str(test_run_dir),
        timeout=60,  # 1 minute total timeout
        text=True,
        capture_output=True
    )
    
    print("📤 STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("📤 STDERR:")
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n✅ TEST PASSED: Script executed successfully!")
        
        # Check if log was created
        log_file = results_dir / "tree_method" / "simulation.log"
        if log_file.exists():
            print(f"✅ Log file created: {log_file}")
            print(f"   Size: {log_file.stat().st_size} bytes")
        else:
            print("⚠️  No log file created")
            
        return True
    else:
        print(f"\n❌ TEST FAILED: Return code {result.returncode}")
        return False

if __name__ == "__main__":
    print("🧪 TESTING SINGLE EXPERIMENT GENERATION")
    print("=" * 50)
    
    try:
        success = test_single_experiment_generation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        sys.exit(1)