#!/usr/bin/env python3

"""
Experiment-Level Analysis for Synthetic Grid 9x9
Aggregates and analyzes results from all experiments in this grid size
Provides comprehensive statistical analysis and performance comparisons
"""

import os
import json
import glob
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import statistics

def load_config():
    """Load the central configuration file"""
    config_path = Path(__file__).parent.parent / "experiment_config.json"
    
    if not config_path.exists():
        print("❌ ERROR: experiment_config.json not found")
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Could not load configuration: {e}")
        return None

def regenerate_run_analyses(experiment_dir):
    """Automatically run individual analyze_results.py scripts to ensure all JSON files are up-to-date"""
    print("Regenerating analysis results for all completed runs...")
    
    regenerated_count = 0
    skipped_count = 0
    
    # Find all run directories (numbered directories)
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs.sort(key=lambda x: int(x.name))
    
    for run_dir in run_dirs:
        results_dir = run_dir / "results"
        analyze_script = run_dir / "analyze_results.py"
        
        # Check if this run has simulation results and an analysis script
        if not analyze_script.exists():
            continue
            
        # Check if any simulation logs exist (tree_method, actuated, or fixed)
        has_results = False
        for method in ['tree_method', 'actuated', 'fixed']:
            log_file = results_dir / method / "simulation.log"
            if log_file.exists() and log_file.stat().st_size > 0:
                has_results = True
                break
        
        if not has_results:
            skipped_count += 1
            continue
            
        # Run the individual analysis script
        try:
            print(f"  Analyzing run {run_dir.name}...")
            # Change to run directory and execute the script
            result = subprocess.run(
                [sys.executable, str(analyze_script)], 
                cwd=str(run_dir),
                capture_output=True, 
                text=True,
                timeout=60  # 60 second timeout per analysis
            )
            
            if result.returncode == 0:
                regenerated_count += 1
            else:
                print(f"    Warning: Analysis failed for run {run_dir.name}: {result.stderr.strip()}")
                skipped_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"    Warning: Analysis timed out for run {run_dir.name}")
            skipped_count += 1
        except Exception as e:
            print(f"    Warning: Could not analyze run {run_dir.name}: {e}")
            skipped_count += 1
    
    print(f"Analysis regeneration complete: {regenerated_count} runs analyzed, {skipped_count} runs skipped")
    print()

def load_run_results(experiment_dir):
    """Load results from all runs using the analysis_results.json files"""
    all_results = {
        'tree_method': [],
        'actuated': [],
        'fixed': []
    }
    
    # Find all run directories
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs.sort(key=lambda x: int(x.name))
    
    for run_dir in run_dirs:
        results_file = run_dir / "results" / "analysis_results.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract our implementation results
                our_results = data.get('our_results', {})
                for method in ['tree', 'actuated', 'fixed']:
                    method_key = 'tree_method' if method == 'tree' else method
                    if method in our_results and our_results[method]:
                        result = our_results[method].copy()
                        result['run'] = int(run_dir.name)
                        all_results[method_key].append(result)
                        
            except Exception as e:
                print(f"Warning: Could not load results for run {run_dir.name}: {e}")
        else:
            print(f"Warning: Missing analysis results for run {run_dir.name}")
    
    return all_results

def calculate_statistics(data_list, metric):
    """Calculate statistics for a given metric across all runs"""
    values = [item[metric] for item in data_list if item.get(metric) is not None]
    if not values:
        return None
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values),
        'values': values
    }

def analyze_experiment():
    """Main experiment-level analysis function"""
    experiment_dir = Path(__file__).parent
    experiment_name = experiment_dir.name
    
    print(f"SYNTHETIC GRID EXPERIMENT ANALYSIS")
    print(f"Grid Size: {experiment_name}")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)
    
    grid_config = config['grid_configurations']['9x9']
    
    # Automatically regenerate analysis results from simulation logs
    regenerate_run_analyses(experiment_dir)
    
    # Load results from all runs
    all_results = load_run_results(experiment_dir)
    
    # Check if we have any results
    total_results = sum(len(results) for results in all_results.values())
    if total_results == 0:
        print("❌ No results found! Please run experiments first.")
        print("   Use: ./run_all_runs.sh")
        sys.exit(1)
    
    # Generate summary statistics
    print(f"\nSUMMARY STATISTICS (9x9 Grid - {grid_config['dimension']}x{grid_config['dimension']}):")
    print("=" * 70)
    
    summary_stats = {}
    
    for method in ['tree_method', 'actuated', 'fixed']:
        results = all_results[method]
        method_name = method.replace('_', ' ').title()
        
        print(f"\n{method_name}:")
        print(f"  Runs completed: {len(results)}")
        
        if results:
            # Calculate statistics for each metric
            for metric in ['vehicles_entered', 'vehicles_arrived', 'avg_duration', 'completion_rate']:
                stats = calculate_statistics(results, metric)
                if stats:
                    if metric == 'completion_rate':
                        print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.3%} ± {stats['std']:.3%}")
                        print(f"    Range: {stats['min']:.3%} - {stats['max']:.3%}")
                    elif metric == 'avg_duration':
                        print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.1f} ± {stats['std']:.1f} steps")
                        print(f"    Range: {stats['min']:.1f} - {stats['max']:.1f} steps")
                    else:
                        print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:,.0f} ± {stats['std']:,.0f}")
                        print(f"    Range: {stats['min']:,.0f} - {stats['max']:,.0f}")
                    
                    # Store for comparisons
                    if method not in summary_stats:
                        summary_stats[method] = {}
                    summary_stats[method][metric] = stats

    # Method-to-method comparisons
    print(f"\nMETHOD PERFORMANCE COMPARISON (9x9 Grid):")
    print("=" * 50)
    
    comparisons = [
        ('tree_method', 'actuated'),
        ('tree_method', 'fixed'),
        ('actuated', 'fixed')
    ]
    
    for method1, method2 in comparisons:
        if method1 in summary_stats and method2 in summary_stats:
            method1_name = method1.replace('_', ' ').title()
            method2_name = method2.replace('_', ' ').title()
            print(f"\n{method1_name} vs {method2_name}:")
            
            # Vehicles arrived comparison
            if 'vehicles_arrived' in summary_stats[method1] and 'vehicles_arrived' in summary_stats[method2]:
                mean1 = summary_stats[method1]['vehicles_arrived']['mean']
                mean2 = summary_stats[method2]['vehicles_arrived']['mean']
                improvement = ((mean1 - mean2) / mean2) * 100
                print(f"  Vehicles Arrived: {mean1:,.0f} vs {mean2:,.0f} ({improvement:+.1f}%)")
            
            # Duration comparison
            if 'avg_duration' in summary_stats[method1] and 'avg_duration' in summary_stats[method2]:
                mean1 = summary_stats[method1]['avg_duration']['mean']
                mean2 = summary_stats[method2]['avg_duration']['mean']
                improvement = ((mean2 - mean1) / mean2) * 100  # Lower is better for duration
                print(f"  Avg Duration: {mean1:.1f} vs {mean2:.1f} steps ({improvement:+.1f}% improvement)")
            
            # Completion rate comparison
            if 'completion_rate' in summary_stats[method1] and 'completion_rate' in summary_stats[method2]:
                mean1 = summary_stats[method1]['completion_rate']['mean']
                mean2 = summary_stats[method2]['completion_rate']['mean']
                improvement = ((mean1 - mean2) / mean2) * 100
                print(f"  Completion Rate: {mean1:.3%} vs {mean2:.3%} ({improvement:+.1f}%)")

    # Tree Method validation summary for this grid size
    if 'tree_method' in summary_stats:
        print(f"\nTREE METHOD VALIDATION (9x9 Grid):")
        print("=" * 40)
        
        tree_stats = summary_stats['tree_method']
        
        if 'actuated' in summary_stats and 'fixed' in summary_stats:
            actuated_stats = summary_stats['actuated']
            fixed_stats = summary_stats['fixed']
            
            # Calculate improvements
            if 'vehicles_arrived' in tree_stats and 'vehicles_arrived' in actuated_stats:
                tree_vs_actuated = ((tree_stats['vehicles_arrived']['mean'] - actuated_stats['vehicles_arrived']['mean']) / actuated_stats['vehicles_arrived']['mean']) * 100
                print(f"Tree Method vs SUMO Actuated:")
                print(f"  Vehicles Arrived Improvement: {tree_vs_actuated:+.1f}%")
            
            if 'vehicles_arrived' in tree_stats and 'vehicles_arrived' in fixed_stats:
                tree_vs_fixed = ((tree_stats['vehicles_arrived']['mean'] - fixed_stats['vehicles_arrived']['mean']) / fixed_stats['vehicles_arrived']['mean']) * 100
                print(f"\nTree Method vs Fixed Timing:")
                print(f"  Vehicles Arrived Improvement: {tree_vs_fixed:+.1f}%")
            
            if 'completion_rate' in tree_stats:
                print(f"\nOverall Performance (9x9 Grid):")
                completion_rate = tree_stats['completion_rate']['mean']
                avg_duration = tree_stats['avg_duration']['mean'] if 'avg_duration' in tree_stats else 0
                vehicles_arrived = tree_stats['vehicles_arrived']['mean'] if 'vehicles_arrived' in tree_stats else 0
                print(f"  Completion Rate: {completion_rate:.3%}")
                print(f"  Average Duration: {avg_duration:.1f} steps")
                print(f"  Vehicles Arrived: {vehicles_arrived:,.0f}")

    # Save comprehensive results
    experiment_results = {
        'experiment': experiment_name,
        'grid_configuration': grid_config,
        'our_implementation_summary': summary_stats,
        'analysis_date': datetime.now().isoformat(),
        'total_runs_analyzed': sum(len(results) for results in all_results.values()),
        'runs_per_method': {method: len(results) for method, results in all_results.items()},
        'configuration_used': {
            'vehicle_count_levels': list(grid_config['vehicle_counts'].keys()),
            'vehicle_types': config['shared_parameters']['vehicle_types'],
            'routing_strategies': config['shared_parameters']['routing_strategies'],
            'departure_patterns': config['shared_parameters']['departure_patterns'],
            'simulation_durations': config['shared_parameters']['simulation_durations'],
            'junctions_removed': config['shared_parameters']['junctions_removed']
        }
    }
    
    results_file = experiment_dir / f"{experiment_name}_experiment_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    print(f"\n📄 Experiment analysis saved to: {results_file}")
    print(f"\n🎉 9x9 GRID ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"✅ Methods analyzed: {len([m for m in summary_stats.keys() if summary_stats[m]])}/3")
    print(f"📊 Total results processed: {total_results}")
    print(f"💾 Results saved to: {results_file.name}")

if __name__ == "__main__":
    analyze_experiment()