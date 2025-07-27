#!/usr/bin/env python3

"""
Master Experiment Runner for Synthetic Grid Benchmark Suite
Executes all grid sizes (5x5, 7x7, 9x9) sequentially with comprehensive validation
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

def load_config():
    """Load and validate the central configuration file"""
    config_path = Path(__file__).parent / "experiment_config.json"
    
    if not config_path.exists():
        print("❌ ERROR: experiment_config.json not found")
        print(f"   Expected location: {config_path}")
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

def calculate_total_experiments(config):
    """Calculate total number of experiments across all grids"""
    shared = config['shared_parameters']
    
    experiments_per_grid = (
        len(list(config['grid_configurations'].values())[0]['vehicle_counts']) *
        len(shared['vehicle_types']) *
        len(shared['routing_strategies']) *
        len(shared['departure_patterns']) *
        len(shared['simulation_durations']) *
        len(shared['junctions_removed'])
    )
    
    total_grids = len(config['grid_configurations'])
    total_methods = len(shared['traffic_control_methods'])
    runs_per_experiment = config['execution_settings']['runs_per_experiment']
    
    return {
        'experiments_per_grid': experiments_per_grid,
        'total_grids': total_grids,
        'total_methods': total_methods,
        'runs_per_experiment': runs_per_experiment,
        'total_experiments': experiments_per_grid * total_grids,
        'total_simulations': experiments_per_grid * total_grids * total_methods,
        'total_runs': experiments_per_grid * total_grids * total_methods * runs_per_experiment
    }

def estimate_execution_time(totals, timeout_minutes):
    """Estimate total execution time"""
    # Conservative estimate: assume each run takes 80% of timeout
    avg_run_time_minutes = timeout_minutes * 0.8
    total_time_minutes = totals['total_runs'] * avg_run_time_minutes
    
    return {
        'minutes': total_time_minutes,
        'hours': total_time_minutes / 60,
        'days': total_time_minutes / (60 * 24)
    }

def display_experiment_summary(config, totals, time_estimate):
    """Display comprehensive experiment summary"""
    print(f"""
🔬 SYNTHETIC GRID BENCHMARK SUITE
{'=' * 80}

📊 EXPERIMENT SCOPE:
   Grid Sizes: {', '.join(config['grid_configurations'].keys())}
   Experiments per Grid: {totals['experiments_per_grid']:,}
   Traffic Control Methods: {', '.join(config['shared_parameters']['traffic_control_methods'])}
   Runs per Experiment: {totals['runs_per_experiment']}

📈 TOTALS:
   Total Unique Experiments: {totals['total_experiments']:,}
   Total Simulations (×{totals['total_methods']} methods): {totals['total_simulations']:,}
   Total Runs (×{totals['runs_per_experiment']} iterations): {totals['total_runs']:,}

⏱️  ESTIMATED EXECUTION TIME:
   Conservative Estimate: {time_estimate['hours']:.1f} hours ({time_estimate['days']:.1f} days)
   Per Grid: {time_estimate['hours']/totals['total_grids']:.1f} hours
   
💾 STORAGE ESTIMATE:
   Approximate disk space needed: {totals['total_runs'] * 0.5:.1f} MB
   (Based on ~0.5MB per simulation log)

🔧 CONFIGURATION:
   Timeout per run: {config['execution_settings']['timeout_minutes']} minutes
   GUI mode: {config['execution_settings']['gui']}
   Parallel execution: {config['execution_settings']['parallel_execution']}
""")

def confirm_execution():
    """Get user confirmation before starting expensive computation"""
    try:
        response = input("\n🚀 Continue with experiment execution? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment cancelled by user")
        return False

def run_grid_experiments(grid_name, config):
    """Execute experiments for a specific grid size"""
    print(f"\n{'=' * 60}")
    print(f"🔄 STARTING GRID: {grid_name}")
    print(f"{'=' * 60}")
    
    grid_dir = Path(__file__).parent / f"grids-{grid_name}"
    if not grid_dir.exists():
        print(f"❌ ERROR: Grid directory not found: {grid_dir}")
        return False
    
    run_script = grid_dir / "run_all_runs.sh" 
    if not run_script.exists():
        print(f"❌ ERROR: Run script not found: {run_script}")
        return False
    
    # Execute the grid-specific run script
    start_time = time.time()
    try:
        print(f"   Executing: {run_script}")
        result = subprocess.run(
            [str(run_script)],
            cwd=str(grid_dir),
            timeout=config['execution_settings']['timeout_minutes'] * 60 * 10,  # 10x timeout for full grid
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {grid_name} grid completed successfully in {execution_time/3600:.1f} hours")
            return True
        else:
            print(f"❌ {grid_name} grid failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"⏰ {grid_name} grid timed out after {execution_time/3600:.1f} hours")
        return False
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {grid_name} grid failed with error: {e}")
        return False

def main():
    """Main execution function"""
    print("SYNTHETIC GRID BENCHMARK SUITE - MASTER RUNNER")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Calculate experiment totals
    totals = calculate_total_experiments(config)
    time_estimate = estimate_execution_time(totals, config['execution_settings']['timeout_minutes'])
    
    # Display summary
    display_experiment_summary(config, totals, time_estimate)
    
    # Get user confirmation
    if not confirm_execution():
        print("Experiment execution cancelled.")
        sys.exit(0)
    
    # Record start time
    master_start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    
    print(f"\n🚀 STARTING ALL EXPERIMENTS AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute each grid size
    grid_results = {}
    for grid_name in config['grid_configurations'].keys():
        grid_results[grid_name] = run_grid_experiments(grid_name, config)
    
    # Calculate total execution time
    total_execution_time = time.time() - master_start_time
    end_timestamp = datetime.now().isoformat()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("🏁 MASTER EXECUTION COMPLETE")
    print(f"{'=' * 80}")
    
    successful_grids = sum(1 for success in grid_results.values() if success)
    total_grids = len(grid_results)
    
    print(f"✅ Successful grids: {successful_grids}/{total_grids}")
    print(f"⏱️  Total execution time: {total_execution_time/3600:.1f} hours")
    
    for grid_name, success in grid_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {grid_name}: {status}")
    
    # Save execution log
    log_data = {
        'start_time': start_timestamp,
        'end_time': end_timestamp,
        'execution_time_hours': total_execution_time / 3600,
        'grid_results': grid_results,
        'successful_grids': successful_grids,
        'total_grids': total_grids,
        'experiment_totals': totals,
        'config_version': config['experiment_metadata']['version']
    }
    
    log_file = Path(__file__).parent / "master_execution.log"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n📄 Execution log saved to: {log_file}")
    
    if successful_grids == total_grids:
        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("   Next step: Run python analyze_all_experiments.py")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_grids - successful_grids} grid(s) failed. Check individual logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()