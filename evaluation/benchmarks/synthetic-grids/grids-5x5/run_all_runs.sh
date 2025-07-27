#!/usr/bin/env python3

"""
Grid-Specific Experiment Runner for 5x5 Synthetic Grid
Generates and executes all parameter combinations for this grid size
"""

import os
import sys
import json
import itertools
import subprocess
import time
from pathlib import Path
from datetime import datetime

def load_config():
    """Load the central configuration file"""
    config_path = Path(__file__).parent.parent / "experiment_config.json"
    
    if not config_path.exists():
        print("❌ ERROR: experiment_config.json not found")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration file loaded successfully")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in experiment_config.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Could not load configuration: {e}")
        sys.exit(1)

def validate_config(config):
    """Validate configuration for this grid size"""
    grid_size = "5x5"
    
    if grid_size not in config['grid_configurations']:
        print(f"❌ ERROR: Configuration for {grid_size} not found")
        sys.exit(1)
    
    required_params = ['vehicle_types', 'routing_strategies', 'departure_patterns', 
                      'simulation_durations', 'junctions_removed', 'traffic_control_methods']
    
    for param in required_params:
        if param not in config['shared_parameters']:
            print(f"❌ ERROR: Required parameter '{param}' not found in configuration")
            sys.exit(1)
    
    return True

def generate_experiment_combinations(config):
    """Generate all parameter combinations for this grid"""
    grid_size = "5x5"
    grid_config = config['grid_configurations'][grid_size]
    shared = config['shared_parameters']
    
    # Generate all combinations
    combinations = list(itertools.product(
        grid_config['vehicle_counts'].items(),  # (name, count) pairs
        shared['vehicle_types'],
        shared['routing_strategies'], 
        shared['departure_patterns'],
        shared['simulation_durations'],
        shared['junctions_removed']
    ))
    
    experiments = []
    for i, (vehicle_count_pair, vehicle_types, routing_strategy, departure_pattern, 
            simulation_duration, junctions_removed) in enumerate(combinations, 1):
        
        vehicle_count_name, vehicle_count = vehicle_count_pair
        
        experiment = {
            'run_id': i,
            'grid_dimension': grid_config['dimension'],
            'block_size_m': grid_config['block_size_m'],
            'vehicle_count_level': vehicle_count_name,
            'num_vehicles': vehicle_count,
            'vehicle_types': vehicle_types,
            'routing_strategy': routing_strategy,
            'departure_pattern': departure_pattern,
            'simulation_duration': simulation_duration,
            'junctions_removed': junctions_removed,
            'lane_assignment': shared['lane_assignment'],
            'attractiveness': shared['attractiveness'],
            'step_length': shared['step_length'],
            'time_dependent': shared['time_dependent'],
            'traffic_control_methods': shared['traffic_control_methods']
        }
        experiments.append(experiment)
    
    return experiments

def display_experiment_summary(experiments, config):
    """Display experiment summary and get user confirmation"""
    grid_size = "5x5"
    
    print(f"""
🔬 SYNTHETIC GRID EXPERIMENT RUNNER - {grid_size.upper()}
{'=' * 80}

📊 EXPERIMENT MATRIX:
   Grid Dimension: {experiments[0]['grid_dimension']}x{experiments[0]['grid_dimension']}
   Block Size: {experiments[0]['block_size_m']}m
   Vehicle Count Levels: {len(set(exp['vehicle_count_level'] for exp in experiments))}
   Vehicle Types: {len(config['shared_parameters']['vehicle_types'])}
   Routing Strategies: {len(config['shared_parameters']['routing_strategies'])}
   Departure Patterns: {len(config['shared_parameters']['departure_patterns'])}  
   Simulation Durations: {len(config['shared_parameters']['simulation_durations'])}
   Junction Removal Levels: {len(config['shared_parameters']['junctions_removed'])}

📈 TOTALS:
   Unique Experiments: {len(experiments)}
   Traffic Control Methods per Experiment: {len(config['shared_parameters']['traffic_control_methods'])}
   Total Simulations: {len(experiments) * len(config['shared_parameters']['traffic_control_methods'])}
   Runs per Experiment: {config['execution_settings']['runs_per_experiment']}
   Total Runs: {len(experiments) * len(config['shared_parameters']['traffic_control_methods']) * config['execution_settings']['runs_per_experiment']}

⏱️  EXECUTION ESTIMATE:
   Timeout per run: {config['execution_settings']['timeout_minutes']} minutes
   Conservative total time: {len(experiments) * len(config['shared_parameters']['traffic_control_methods']) * config['execution_settings']['runs_per_experiment'] * config['execution_settings']['timeout_minutes'] * 0.8 / 60:.1f} hours

💾 STORAGE ESTIMATE:
   Disk space needed: ~{len(experiments) * len(config['shared_parameters']['traffic_control_methods']) * config['execution_settings']['runs_per_experiment'] * 0.5:.1f} MB
""")

def confirm_execution():
    """Get user confirmation"""
    try:
        response = input("\n🚀 Continue with experiment execution? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment cancelled by user")
        return False

def create_run_directory(run_id, experiment, config):
    """Create directory structure for a specific run"""
    run_dir = Path(__file__).parent / str(run_id)
    run_dir.mkdir(exist_ok=True)
    
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create method-specific result directories
    for method in experiment['traffic_control_methods']:
        method_dir = results_dir / method
        method_dir.mkdir(exist_ok=True)
    
    return run_dir

def create_run_experiment_script(run_dir, experiment, config):
    """Create the run_experiment.sh script for a specific parameter combination"""
    script_content = f"""#!/bin/bash

# Synthetic Grid Experiment Run {experiment['run_id']}
# Generated automatically from experiment_config.json

cd ../../../../../  # Navigate to project root

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

echo "🔄 Starting experiment run {experiment['run_id']} ({experiment['vehicle_count_level']} load)"
echo "Grid: {experiment['grid_dimension']}x{experiment['grid_dimension']}, Vehicles: {experiment['num_vehicles']}"

# Run each traffic control method
"""

    for i, method in enumerate(experiment['traffic_control_methods']):
        seed_offset = i * config['execution_settings']['runs_per_experiment']
        time_dependent_flag = '--time_dependent' if experiment['time_dependent'] else ''
        script_content += f"""
echo "   Running {method} method..."
SEED=$((42 + {seed_offset}))

# Run with proper argument quoting and timeout
python3 -c "
import subprocess
import sys
import time

start_time = time.time()

try:
    result = subprocess.run([
        'env', 'PYTHONUNBUFFERED=1', 'python', '-m', 'src.cli',
        '--grid_dimension', '{experiment['grid_dimension']}',
        '--block_size_m', '{experiment['block_size_m']}',
        '--num_vehicles', '{experiment['num_vehicles']}',
        '--vehicle_types', '{experiment['vehicle_types']}',
        '--routing_strategy', '{experiment['routing_strategy']}',
        '--departure_pattern', '{experiment['departure_pattern']}',
        '--end-time', '{experiment['simulation_duration']}',
        '--junctions_to_remove', '{experiment['junctions_removed']}',
        '--lane_count', '{experiment['lane_assignment']}',
        '--attractiveness', '{experiment['attractiveness']}',
        '--step-length', '{experiment['step_length']}',
        {("'--time_dependent'," if experiment['time_dependent'] else "")}
        '--traffic_control', '{method}',
        '--seed', str($SEED)
    ],
    timeout={config['execution_settings']['timeout_minutes'] * 60},
    text=True,
    cwd='.',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
    )
    
    execution_time = time.time() - start_time
    
    # Write output to log file
    with open('evaluation/benchmarks/synthetic-grids/grids-5x5/{experiment['run_id']}/results/{method}/simulation.log', 'w') as f:
        f.write(result.stdout)
    
    if result.returncode == 0:
        print(f'   ✅ {method} completed successfully in {{execution_time:.1f}} seconds')
        sys.exit(0)
    else:
        print(f'   ❌ {method} failed with return code {{result.returncode}} after {{execution_time:.1f}} seconds')
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f'   ⏰ {method} timed out after {{execution_time:.1f}} seconds')
    sys.exit(124)
except Exception as e:
    execution_time = time.time() - start_time
    print(f'   ❌ {method} failed with error after {{execution_time:.1f}} seconds: {{e}}')
    sys.exit(1)
"

# Check the exit code from Python script
if [ $? -eq 0 ]; then
    echo "   ✅ {method} completed successfully"
elif [ $? -eq 124 ]; then
    echo "   ⏰ {method} timed out"
else
    echo "   ❌ {method} failed - check log for details"
    if [ -f "evaluation/benchmarks/synthetic-grids/grids-5x5/{experiment['run_id']}/results/{method}/simulation.log" ]; then
        echo "   Last 3 lines of log:"
        tail -n 3 "evaluation/benchmarks/synthetic-grids/grids-5x5/{experiment['run_id']}/results/{method}/simulation.log"
    fi
fi
"""

    script_content += f"""
echo "🏁 Experiment run {experiment['run_id']} completed"
"""

    script_path = run_dir / "run_experiment.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    return script_path

def create_analyze_results_script(run_dir, experiment):
    """Create the analyze_results.py script for a specific run"""
    # Read the existing analyze_results.py from decentralized-traffic-bottlenecks as template
    template_path = Path(__file__).parent.parent.parent / "decentralized-traffic-bottlenecks" / "Experiment1-realistic-high-load" / "1" / "analyze_results.py"
    
    if template_path.exists():
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Adapt the template for synthetic grids
        adapted_content = template_content.replace(
            'decentralized-traffic-bottlenecks', 'synthetic-grids'
        ).replace(
            'Experiment1-realistic-high-load', f'grids-5x5'
        )
        
        script_path = run_dir / "analyze_results.py"
        with open(script_path, 'w') as f:
            f.write(adapted_content)
        
        script_path.chmod(0o755)
        return script_path
    else:
        print(f"   Warning: Could not find template analyze_results.py")
        return None

def execute_experiment(run_id, experiment, config):
    """Execute a single experiment"""
    print(f"\\n🔄 Executing experiment {run_id}/{len(experiments)} ({experiment['vehicle_count_level']} load)")
    
    # Create directory structure
    run_dir = create_run_directory(run_id, experiment, config)
    
    # Create execution script
    run_script = create_run_experiment_script(run_dir, experiment, config)
    
    # Create analysis script
    analyze_script = create_analyze_results_script(run_dir, experiment)
    
    # Execute the experiment
    start_time = time.time()
    try:
        # Run with output visible for debugging
        result = subprocess.run(
            [str(run_script)],
            cwd=str(run_dir),
            timeout=config['execution_settings']['timeout_minutes'] * 60 * len(experiment['traffic_control_methods']) * 1.2,
            text=True
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Experiment {run_id} completed in {execution_time/60:.1f} minutes")
            return True
        else:
            print(f"   ❌ Experiment {run_id} failed after {execution_time/60:.1f} minutes")
            print(f"   Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"   ⏰ Experiment {run_id} timed out after {execution_time/60:.1f} minutes")
        return False
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"   ❌ Experiment {run_id} failed with error: {e}")
        return False

def main():
    """Main execution function"""
    print("SYNTHETIC GRID EXPERIMENT RUNNER - 5x5 GRID")
    print("=" * 80)
    
    # Load and validate configuration
    config = load_config()
    validate_config(config)
    
    # Generate experiment combinations
    print("\\n📊 Generating experiment matrix...")
    experiments = generate_experiment_combinations(config)
    
    # Display summary and get confirmation
    display_experiment_summary(experiments, config)
    
    if not confirm_execution():
        print("Experiment execution cancelled.")
        sys.exit(0)
    
    # Execute experiments
    start_time = time.time()
    print(f"\\n🚀 Starting {len(experiments)} experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_experiments = 0
    for experiment in experiments:
        success = execute_experiment(experiment['run_id'], experiment, config)
        if success:
            successful_experiments += 1
    
    # Summary
    total_time = time.time() - start_time
    print(f"\\n{'=' * 80}")
    print("🏁 5x5 GRID EXPERIMENTS COMPLETE")
    print(f"{'=' * 80}")
    
    print(f"✅ Successful experiments: {successful_experiments}/{len(experiments)}")
    print(f"⏱️  Total execution time: {total_time/3600:.1f} hours")
    print(f"⏱️  Average time per experiment: {total_time/len(experiments)/60:.1f} minutes")
    
    if successful_experiments == len(experiments):
        print("\\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("   Next step: Run python analyze_experiment.py")
        sys.exit(0)
    else:
        print(f"\\n⚠️  {len(experiments) - successful_experiments} experiment(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    # Make experiments available globally for execute_experiment function
    config = load_config()
    validate_config(config)
    experiments = generate_experiment_combinations(config)
    main()