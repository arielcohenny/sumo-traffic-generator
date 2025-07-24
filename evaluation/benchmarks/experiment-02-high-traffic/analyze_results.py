#!/usr/bin/env python3

"""
Analysis script for Experiment 02: High Traffic Load
Extracts metrics from SUMO simulation logs and generates comparative analysis
Based on Tree Method experimental methodology from Traffic Control paper
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_metrics_from_log(log_file):
    """Extract key metrics from SUMO simulation log"""
    metrics = {
        'avg_travel_time': None,
        'throughput': None,
        'total_vehicles': None,
        'simulation_time': None,
        'completed_trips': None
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract average travel time (if logged)
        travel_time_match = re.search(r'Average travel time:\s*(\d+\.?\d*)', content)
        if travel_time_match:
            metrics['avg_travel_time'] = float(travel_time_match.group(1))
            
        # Extract throughput/completed trips
        throughput_match = re.search(r'Vehicles reached destination:\s*(\d+)', content)
        if throughput_match:
            metrics['throughput'] = int(throughput_match.group(1))
            
        # Extract total vehicles
        total_vehicles_match = re.search(r'Total vehicles:\s*(\d+)', content)
        if total_vehicles_match:
            metrics['total_vehicles'] = int(total_vehicles_match.group(1))
            
        # Extract simulation time
        sim_time_match = re.search(r'Simulation time:\s*(\d+\.?\d*)', content)
        if sim_time_match:
            metrics['simulation_time'] = float(sim_time_match.group(1))
            
        # Extract completion rate
        completion_rate_match = re.search(r'Completion rate:\s*(\d+\.?\d*)', content)
        if completion_rate_match:
            metrics['completion_rate'] = float(completion_rate_match.group(1))
        elif metrics['throughput'] and metrics['total_vehicles']:
            metrics['completion_rate'] = metrics['throughput'] / metrics['total_vehicles']
            
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        
    return metrics

def analyze_experiment_results(results_dir):
    """Analyze results from all traffic control methods"""
    methods = ['tree_method', 'actuated', 'fixed']
    all_results = {}
    
    for method in methods:
        method_dir = os.path.join(results_dir, method)
        if not os.path.exists(method_dir):
            print(f"Warning: {method_dir} not found")
            continue
            
        log_files = glob.glob(os.path.join(method_dir, "run_*.log"))
        method_results = []
        
        for log_file in sorted(log_files):
            metrics = extract_metrics_from_log(log_file)
            run_number = int(re.search(r'run_(\d+)\.log', log_file).group(1))
            metrics['run'] = run_number
            metrics['method'] = method
            method_results.append(metrics)
            
        all_results[method] = method_results
        print(f"Processed {len(method_results)} runs for {method}")
    
    return all_results

def generate_summary_statistics(all_results):
    """Generate summary statistics for all methods"""
    summary = {}
    
    for method, results in all_results.items():
        # Extract valid metrics (non-None values)
        travel_times = [r['avg_travel_time'] for r in results if r.get('avg_travel_time') is not None]
        throughputs = [r['throughput'] for r in results if r.get('throughput') is not None]
        completion_rates = [r.get('completion_rate', 0) for r in results if r.get('completion_rate') is not None]
        
        summary[method] = {
            'travel_time_mean': np.mean(travel_times) if travel_times else None,
            'travel_time_std': np.std(travel_times) if travel_times else None,
            'throughput_mean': np.mean(throughputs) if throughputs else None,
            'throughput_std': np.std(throughputs) if throughputs else None,
            'completion_rate_mean': np.mean(completion_rates) if completion_rates else None,
            'completion_rate_std': np.std(completion_rates) if completion_rates else None,
            'num_runs': len(results)
        }
    
    return summary

def create_comparison_plots(all_results, output_dir):
    """Create comparison plots similar to Tree Method paper"""
    # Prepare data for plotting
    plot_data = []
    for method, results in all_results.items():
        for result in results:
            if result['avg_travel_time'] is not None:
                plot_data.append({
                    'method': method,
                    'travel_time': result['avg_travel_time'],
                    'throughput': result['throughput'],
                    'run': result['run']
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average Travel Time Comparison
    sns.boxplot(data=df, x='method', y='travel_time', ax=ax1)
    ax1.set_title('Average Travel Time Comparison\n(High Traffic Load)')
    ax1.set_xlabel('Traffic Control Method')
    ax1.set_ylabel('Average Travel Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Throughput Comparison
    sns.boxplot(data=df, x='method', y='throughput', ax=ax2)
    ax2.set_title('Throughput Comparison\n(High Traffic Load)')
    ax2.set_xlabel('Traffic Control Method')
    ax2.set_ylabel('Throughput (vehicles)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_traffic_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory

def calculate_improvements(all_results):
    """Calculate percentage improvements of Tree Method vs baselines"""
    if 'tree_method' not in all_results:
        return {}
    
    tree_results = all_results['tree_method']
    improvements = {}
    
    # Calculate Tree Method averages
    tree_travel_times = [r['avg_travel_time'] for r in tree_results if r['avg_travel_time'] is not None]
    tree_throughputs = [r['throughput'] for r in tree_results if r['throughput'] is not None]
    
    tree_avg_travel = np.mean(tree_travel_times) if tree_travel_times else None
    tree_avg_throughput = np.mean(tree_throughputs) if tree_throughputs else None
    
    for method, results in all_results.items():
        if method == 'tree_method':
            continue
            
        method_travel_times = [r['avg_travel_time'] for r in results if r['avg_travel_time'] is not None]
        method_throughputs = [r['throughput'] for r in results if r['throughput'] is not None]
        
        method_avg_travel = np.mean(method_travel_times) if method_travel_times else None
        method_avg_throughput = np.mean(method_throughputs) if method_throughputs else None
        
        improvements[method] = {}
        
        # Travel time improvement (reduction is improvement)
        if tree_avg_travel and method_avg_travel:
            travel_improvement = (method_avg_travel - tree_avg_travel) / method_avg_travel * 100
            improvements[method]['travel_time_improvement'] = travel_improvement
            
        # Throughput improvement (increase is improvement)
        if tree_avg_throughput and method_avg_throughput:
            throughput_improvement = (tree_avg_throughput - method_avg_throughput) / method_avg_throughput * 100
            improvements[method]['throughput_improvement'] = throughput_improvement
    
    return improvements

def main():
    """Main analysis function"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run the experiment first: ./run_experiment.sh")
        return
    
    print("Analyzing Experiment 02: High Traffic Load Results")
    print("=" * 60)
    
    # Analyze results
    all_results = analyze_experiment_results(results_dir)
    
    # Generate summary statistics
    summary = generate_summary_statistics(all_results)
    
    # Calculate improvements
    improvements = calculate_improvements(all_results)
    
    # Print summary
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    for method, stats in summary.items():
        print(f"\n{method.upper()}:")
        print(f"  Runs: {stats['num_runs']}")
        if stats['travel_time_mean']:
            print(f"  Avg Travel Time: {stats['travel_time_mean']:.2f} ± {stats['travel_time_std']:.2f}s")
        if stats['throughput_mean']:
            print(f"  Throughput: {stats['throughput_mean']:.1f} ± {stats['throughput_std']:.1f} vehicles")
        if stats['completion_rate_mean']:
            print(f"  Completion Rate: {stats['completion_rate_mean']:.2%} ± {stats['completion_rate_std']:.2%}")
    
    # Print improvements
    print("\nTREE METHOD IMPROVEMENTS:")
    print("-" * 40)
    for method, impr in improvements.items():
        print(f"\nvs {method.upper()}:")
        if 'travel_time_improvement' in impr:
            print(f"  Travel Time: {impr['travel_time_improvement']:.1f}% reduction")
        if 'throughput_improvement' in impr:
            print(f"  Throughput: {impr['throughput_improvement']:.1f}% increase")
    
    # Save summary to JSON
    summary_file = results_dir / "summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': summary,
            'improvements': improvements
        }, f, indent=2)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, results_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()