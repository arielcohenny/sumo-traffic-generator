#!/usr/bin/env python3

"""
Experiment-Level Analysis for Decentralized Traffic Bottleneck
Aggregates and analyzes results from all 20 runs of an experiment
Provides comprehensive statistical analysis and validation against original results
"""

import os
import json
import glob
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import statistics

def regenerate_run_analyses(experiment_dir):
    """Automatically run individual analyze_results.py scripts to ensure all JSON files are up-to-date"""
    print("Regenerating analysis results for all completed runs...")
    
    regenerated_count = 0
    skipped_count = 0
    
    for run in range(1, 21):
        run_dir = experiment_dir / str(run)
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
            print(f"  Analyzing run {run}...")
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
                print(f"    Warning: Analysis failed for run {run}: {result.stderr.strip()}")
                skipped_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"    Warning: Analysis timed out for run {run}")
            skipped_count += 1
        except Exception as e:
            print(f"    Warning: Could not analyze run {run}: {e}")
            skipped_count += 1
    
    print(f"Analysis regeneration complete: {regenerated_count} runs analyzed, {skipped_count} runs skipped")
    print()

def load_run_results(experiment_dir):
    """Load results from all runs using the new analysis_results.json files"""
    all_results = {
        'tree': [],
        'actuated': [],
        'fixed': []
    }
    
    original_results = {
        'tree': [],
        'actuated': [],
        'fixed (uniform)': []
    }
    
    for run in range(1, 21):
        run_dir = experiment_dir / str(run) / "results"
        results_file = run_dir / "analysis_results.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract our implementation results
                our_results = data.get('our_results', {})
                for method in ['tree', 'actuated', 'fixed']:
                    if method in our_results:
                        result = our_results[method].copy()
                        result['run'] = run
                        all_results[method].append(result)
                
                # Extract original research results (only need to do this once per run)
                if run == 1 or not original_results['tree']:  # First run or if we don't have original data yet
                    orig_data = data.get('decentralized_traffic_bottlenecks_original_results', {})
                    for orig_key, our_key in [('tree', 'tree'), ('actuated', 'actuated'), ('fixed (uniform)', 'fixed (uniform)')]:
                        if orig_key in orig_data:
                            original_results[orig_key].append(orig_data[orig_key])
                            
            except Exception as e:
                print(f"Warning: Could not load results for run {run}: {e}")
        else:
            print(f"Warning: Missing analysis results for run {run}")
    
    return all_results, original_results

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
        'count': len(values)
    }

def analyze_experiment():
    """Main experiment-level analysis function"""
    experiment_dir = Path(__file__).parent
    experiment_name = experiment_dir.name
    
    print(f"EXPERIMENT-LEVEL ANALYSIS")
    print(f"Experiment: {experiment_name}")
    print("=" * 80)
    print()
    
    # Automatically regenerate analysis results from simulation logs
    regenerate_run_analyses(experiment_dir)
    
    # Load results from all runs
    all_results, original_results = load_run_results(experiment_dir)
    
    # Generate summary statistics
    print(f"\nSUMMARY STATISTICS (Our Implementation):")
    print("=" * 50)
    
    summary_stats = {}
    
    for method in ['tree', 'actuated', 'fixed']:
        results = all_results[method]
        method_name = method.replace('_', ' ').title()
        
        print(f"\n{method_name}:")
        print(f"  Runs completed: {len(results)}/20")
        
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
    
    # Original research results comparison
    print(f"\nORIGINAL RESEARCH RESULTS COMPARISON:")
    print("=" * 50)
    
    # Get the dataset-level original results
    dataset_dir = Path(str(experiment_dir).replace(
        'evaluation/benchmarks/decentralized-traffic-bottlenecks', 
        'evaluation/datasets/decentralized_traffic_bottleneck'
    ))
    
    original_dataset_results = load_original_dataset_results(dataset_dir)
    
    comparison_mapping = {
        'tree': 'tree',
        'actuated': 'actuated', 
        'fixed': 'fixed (uniform)'
    }
    
    for our_method, orig_method in comparison_mapping.items():
        if our_method in summary_stats and orig_method in original_dataset_results:
            our_stats = summary_stats[our_method]
            orig_stats = original_dataset_results[orig_method]
            
            method_name = our_method.replace('_', ' ').title()
            print(f"\n{method_name} Comparison:")
            
            # Compare vehicles arrived
            if 'vehicles_arrived' in our_stats and 'vehicles_arrived' in orig_stats:
                our_mean = our_stats['vehicles_arrived']['mean']
                orig_mean = orig_stats['vehicles_arrived']['mean']
                diff_pct = ((our_mean - orig_mean) / orig_mean) * 100
                print(f"  Vehicles Arrived: {our_mean:,.0f} vs {orig_mean:,.0f} ({diff_pct:+.1f}%)")
                
            # Compare average duration
            if 'avg_duration' in our_stats and 'avg_duration' in orig_stats:
                our_mean = our_stats['avg_duration']['mean']
                orig_mean = orig_stats['avg_duration']['mean']
                diff_pct = ((our_mean - orig_mean) / orig_mean) * 100  
                print(f"  Avg Duration: {our_mean:.1f} vs {orig_mean:.1f} steps ({diff_pct:+.1f}%)")

    # Method-to-method comparisons
    print(f"\nMETHOD PERFORMANCE COMPARISON:")
    print("=" * 40)
    
    comparisons = [
        ('tree', 'actuated'),
        ('tree', 'fixed'),
        ('actuated', 'fixed')
    ]
    
    for method1, method2 in comparisons:
        if method1 in summary_stats and method2 in summary_stats:
            print(f"\n{method1.title()} vs {method2.title()}:")
            
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
    
    # Save comprehensive results
    experiment_results = {
        'experiment': experiment_name,
        'our_implementation_summary': summary_stats,
        'original_research_summary': original_dataset_results,
        'analysis_date': datetime.now().isoformat(),
        'total_runs_analyzed': sum(len(results) for results in all_results.values()),
        'runs_per_method': {method: len(results) for method, results in all_results.items()}
    }
    
    results_file = experiment_dir / f"{experiment_name}_experiment_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    print(f"\nExperiment analysis saved to: {results_file}")
    print("\nExperiment-level analysis complete!")

def load_original_dataset_results(dataset_dir):
    """Load original research results from the dataset CSV files"""
    original_results = {
        'tree': {'vehicles_arrived': {'values': []}, 'avg_duration': {'values': []}},
        'actuated': {'vehicles_arrived': {'values': []}, 'avg_duration': {'values': []}},
        'fixed (uniform)': {'vehicles_arrived': {'values': []}, 'avg_duration': {'values': []}}
    }
    
    try:
        # Load vehicles arrived data
        arrived_file = dataset_dir / "results_vehicles_num_arrived.txt"
        if arrived_file.exists():
            with open(arrived_file, 'r') as f:
                lines = f.readlines()
                for i in range(1, min(21, len(lines))):  # Skip header, process runs 1-20
                    parts = lines[i].strip().split(',')
                    if len(parts) >= 4:
                        # Format: Run,Tree,Uniform,Actuated
                        original_results['tree']['vehicles_arrived']['values'].append(int(parts[1]))
                        original_results['fixed (uniform)']['vehicles_arrived']['values'].append(int(parts[2]))
                        original_results['actuated']['vehicles_arrived']['values'].append(int(parts[3]))
        
        # Load average duration data
        duration_file = dataset_dir / "results_vehicles_avg_duration.txt"
        if duration_file.exists():
            with open(duration_file, 'r') as f:
                lines = f.readlines()
                for i in range(1, min(21, len(lines))):  # Skip header, process runs 1-20
                    parts = lines[i].strip().split(',')
                    if len(parts) >= 4:
                        # Format: Run,CurrentTree,Uniform,Actuated
                        original_results['tree']['avg_duration']['values'].append(int(parts[1]))
                        original_results['fixed (uniform)']['avg_duration']['values'].append(int(parts[2]))
                        original_results['actuated']['avg_duration']['values'].append(int(parts[3]))
        
        # Calculate statistics for original results
        for method in original_results:
            for metric in ['vehicles_arrived', 'avg_duration']:
                values = original_results[method][metric]['values']
                if values:
                    original_results[method][metric] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'values': values
                    }
                    
    except Exception as e:
        print(f"Warning: Could not load original dataset results: {e}")
    
    return original_results

if __name__ == "__main__":
    analyze_experiment()