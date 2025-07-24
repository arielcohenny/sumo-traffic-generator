#!/usr/bin/env python3

"""
Tree Method Validation Analysis
Analyzes results from Tree Method validation using original datasets
Compares Tree Method vs SUMO Actuated vs Fixed timing using 25,470 vehicles over 7,300 seconds
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
from scipy import stats

def extract_metrics_from_log(log_file):
    """Extract key metrics from Tree Method validation log"""
    metrics = {
        'avg_travel_time': None,
        'throughput': None,
        'total_vehicles': None,
        'simulation_time': None,
        'completed_trips': None,
        'completion_rate': None
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract total simulation steps (as proxy for simulation duration)
        sim_steps_match = re.search(r'Total simulation steps:\s*(\d+)', content)
        if sim_steps_match:
            metrics['simulation_time'] = int(sim_steps_match.group(1))
            metrics['avg_travel_time'] = metrics['simulation_time']  # Proxy for travel time
            
        # Extract completed vehicles
        completed_match = re.search(r'Completed vehicles:\s*(\d+)', content)
        if completed_match:
            metrics['throughput'] = int(completed_match.group(1))
            metrics['completed_trips'] = int(completed_match.group(1))
            
        # Extract total vehicles from SUMO output (more accurate for high-volume simulations)
        sumo_total_match = re.search(r'vehicles TOT (\d+) ACT \d+ BUF \d+\)\s*$', content, re.MULTILINE)
        if sumo_total_match:
            metrics['total_vehicles'] = int(sumo_total_match.group(1))
        else:
            # Fallback to pipeline total
            total_vehicles_match = re.search(r'Total vehicles:\s*(\d+)', content)
            if total_vehicles_match:
                metrics['total_vehicles'] = int(total_vehicles_match.group(1))
            
        # Extract completion rate
        completion_rate_match = re.search(r'Completion rate:\s*(\d+\.?\d*)%?', content)
        if completion_rate_match:
            rate = float(completion_rate_match.group(1))
            # Convert to decimal if it's a percentage
            metrics['completion_rate'] = rate / 100.0 if rate > 1.0 else rate
        elif metrics['throughput'] and metrics['total_vehicles'] and metrics['total_vehicles'] > 0:
            metrics['completion_rate'] = metrics['throughput'] / metrics['total_vehicles']
            
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        
    return metrics

def analyze_validation_results(results_dir):
    """Analyze validation results from all traffic control methods"""
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

def generate_validation_statistics(all_results):
    """Generate comprehensive validation statistics for Tree Method comparison"""
    summary = {}
    
    for method, results in all_results.items():
        # Extract valid metrics (non-None values)
        travel_times = [r['avg_travel_time'] for r in results if r['avg_travel_time'] is not None]
        throughputs = [r['throughput'] for r in results if r['throughput'] is not None]
        completion_rates = [r['completion_rate'] for r in results if r['completion_rate'] is not None]
        total_vehicles = [r['total_vehicles'] for r in results if r['total_vehicles'] is not None]
        
        # Calculate comprehensive statistics for Tree Method validation
        summary[method] = {
            # Basic statistics
            'travel_time_mean': np.mean(travel_times) if travel_times else None,
            'travel_time_std': np.std(travel_times) if travel_times else None,
            'travel_time_median': np.median(travel_times) if travel_times else None,
            'travel_time_min': np.min(travel_times) if travel_times else None,
            'travel_time_max': np.max(travel_times) if travel_times else None,
            
            'throughput_mean': np.mean(throughputs) if throughputs else None,
            'throughput_std': np.std(throughputs) if throughputs else None,
            'throughput_median': np.median(throughputs) if throughputs else None,
            'throughput_min': np.min(throughputs) if throughputs else None,
            'throughput_max': np.max(throughputs) if throughputs else None,
            
            'completion_rate_mean': np.mean(completion_rates) if completion_rates else None,
            'completion_rate_std': np.std(completion_rates) if completion_rates else None,
            
            'total_vehicles_mean': np.mean(total_vehicles) if total_vehicles else None,
            'total_vehicles_std': np.std(total_vehicles) if total_vehicles else None,
            
            # Statistical validation metrics
            'num_runs': len(results),
            'sample_size': len(travel_times),
            
            # Confidence intervals (95%)
            'travel_time_ci_lower': np.mean(travel_times) - 1.96 * np.std(travel_times) / np.sqrt(len(travel_times)) if travel_times else None,
            'travel_time_ci_upper': np.mean(travel_times) + 1.96 * np.std(travel_times) / np.sqrt(len(travel_times)) if travel_times else None,
            
            'throughput_ci_lower': np.mean(throughputs) - 1.96 * np.std(throughputs) / np.sqrt(len(throughputs)) if throughputs else None,
            'throughput_ci_upper': np.mean(throughputs) + 1.96 * np.std(throughputs) / np.sqrt(len(throughputs)) if throughputs else None,
            
            # Tree Method efficiency metrics
            'efficiency_ratio': np.mean(throughputs) / np.mean(travel_times) if travel_times and throughputs else None,
        }
    
    return summary

def perform_validation_tests(all_results):
    """Perform statistical significance tests for Tree Method validation"""
    methods = list(all_results.keys())
    results = {}
    
    # Extract data for all methods
    method_data = {}
    for method in methods:
        travel_times = [r['avg_travel_time'] for r in all_results[method] if r['avg_travel_time'] is not None]
        throughputs = [r['throughput'] for r in all_results[method] if r['throughput'] is not None]
        completion_rates = [r['completion_rate'] for r in all_results[method] if r['completion_rate'] is not None]
        method_data[method] = {
            'travel_times': travel_times,
            'throughputs': throughputs,
            'completion_rates': completion_rates
        }
    
    # Perform pairwise comparisons
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:  # Avoid duplicate comparisons
                comparison_key = f"{method1}_vs_{method2}"
                
                # Travel time comparison
                travel_t_stat, travel_p_value = stats.ttest_ind(
                    method_data[method1]['travel_times'], 
                    method_data[method2]['travel_times']
                )
                
                # Throughput comparison  
                throughput_t_stat, throughput_p_value = stats.ttest_ind(
                    method_data[method1]['throughputs'], 
                    method_data[method2]['throughputs']
                )
                
                # Completion rate comparison
                completion_t_stat, completion_p_value = stats.ttest_ind(
                    method_data[method1]['completion_rates'], 
                    method_data[method2]['completion_rates']
                )
                
                # Mann-Whitney U tests (non-parametric)
                travel_u_stat, travel_u_p = stats.mannwhitneyu(
                    method_data[method1]['travel_times'], 
                    method_data[method2]['travel_times'],
                    alternative='two-sided'
                )
                
                throughput_u_stat, throughput_u_p = stats.mannwhitneyu(
                    method_data[method1]['throughputs'], 
                    method_data[method2]['throughputs'],
                    alternative='two-sided'
                )
                
                results[comparison_key] = {
                    'travel_time_t_test': {
                        't_statistic': travel_t_stat,
                        'p_value': travel_p_value,
                        'significant': travel_p_value < 0.05
                    },
                    'throughput_t_test': {
                        't_statistic': throughput_t_stat,
                        'p_value': throughput_p_value,
                        'significant': throughput_p_value < 0.05
                    },
                    'completion_rate_t_test': {
                        't_statistic': completion_t_stat,
                        'p_value': completion_p_value,
                        'significant': completion_p_value < 0.05
                    },
                    'travel_time_mannwhitney': {
                        'u_statistic': travel_u_stat,
                        'p_value': travel_u_p,
                        'significant': travel_u_p < 0.05
                    },
                    'throughput_mannwhitney': {
                        'u_statistic': throughput_u_stat,
                        'p_value': throughput_u_p,
                        'significant': throughput_u_p < 0.05
                    }
                }
    
    return results

def create_validation_plots(all_results, output_dir):
    """Create Tree Method validation plots"""
    # Prepare data for plotting
    plot_data = []
    for method, results in all_results.items():
        for result in results:
            if result['avg_travel_time'] is not None and result['throughput'] is not None:
                plot_data.append({
                    'method': method.replace('_', ' ').title(),
                    'travel_time': result['avg_travel_time'],
                    'throughput': result['throughput'],
                    'completion_rate': result['completion_rate'] or 0.0,
                    'run': result['run']
                })
    
    if not plot_data:
        print("Warning: No data available for plotting")
        return
        
    df = pd.DataFrame(plot_data)
    
    # Create comprehensive validation plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Simulation Duration Comparison
    sns.boxplot(data=df, x='method', y='travel_time', ax=ax1)
    ax1.set_title('Tree Method Validation: Simulation Duration\n(25,470 vehicles, 7,300 seconds)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Traffic Control Method')
    ax1.set_ylabel('Simulation Duration (steps)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Throughput Comparison
    sns.boxplot(data=df, x='method', y='throughput', ax=ax2)
    ax2.set_title('Tree Method Validation: Throughput\n(Completed Vehicles)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Traffic Control Method')
    ax2.set_ylabel('Completed Vehicles')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Completion Rate Comparison
    sns.boxplot(data=df, x='method', y='completion_rate', ax=ax3)
    ax3.set_title('Tree Method Validation: Completion Rate\n(Success Rate)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Traffic Control Method')
    ax3.set_ylabel('Completion Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Efficiency Scatter Plot
    sns.scatterplot(data=df, x='travel_time', y='throughput', hue='method', ax=ax4, s=60, alpha=0.7)
    ax4.set_title('Tree Method Validation: Efficiency Analysis\n(Throughput vs Duration)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Simulation Duration (steps)')
    ax4.set_ylabel('Completed Vehicles')
    ax4.legend(title='Method')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_method_validation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Tree Method validation plots saved to: {os.path.join(output_dir, 'tree_method_validation.png')}")

def main():
    """Main Tree Method validation analysis function"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run the validation first: ./run_validation.sh")
        return
    
    print("TREE METHOD VALIDATION ANALYSIS")
    print("Using Original Decentralized Traffic Bottleneck Datasets")
    print("25,470 vehicles, 7,300 seconds duration")
    print("=" * 80)
    
    # Analyze validation results
    all_results = analyze_validation_results(results_dir)
    
    # Generate validation statistics
    summary = generate_validation_statistics(all_results)
    
    # Perform statistical significance tests
    print("\nPerforming Tree Method validation statistical tests...")
    statistical_tests = perform_validation_tests(all_results)
    
    # Print comprehensive validation summary
    print("\nTREE METHOD VALIDATION RESULTS:")
    print("=" * 80)
    for method, stats_data in summary.items():
        method_name = method.upper().replace('_', ' ')
        print(f"\n{method_name}:")
        print(f"  Sample Size: {stats_data['num_runs']} runs")
        
        if stats_data['total_vehicles_mean']:
            print(f"  Average Total Vehicles: {stats_data['total_vehicles_mean']:.0f}")
        
        if stats_data['travel_time_mean']:
            print(f"  Simulation Duration:")
            print(f"    Mean: {stats_data['travel_time_mean']:.2f} ± {stats_data['travel_time_std']:.2f} steps")
            print(f"    95% CI: [{stats_data['travel_time_ci_lower']:.2f}, {stats_data['travel_time_ci_upper']:.2f}]")
            
        if stats_data['throughput_mean']:
            print(f"  Throughput (Completed Vehicles):")
            print(f"    Mean: {stats_data['throughput_mean']:.0f} ± {stats_data['throughput_std']:.0f} vehicles")
            print(f"    95% CI: [{stats_data['throughput_ci_lower']:.0f}, {stats_data['throughput_ci_upper']:.0f}]")
            
        if stats_data['completion_rate_mean']:
            print(f"  Completion Rate: {stats_data['completion_rate_mean']:.3%} ± {stats_data['completion_rate_std']:.3%}")
            
        if stats_data['efficiency_ratio']:
            print(f"  Efficiency Ratio: {stats_data['efficiency_ratio']:.6f} vehicles/step")
    
    # Print statistical significance results
    print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
    print("=" * 60)
    for comparison, test_results in statistical_tests.items():
        method1, method2 = comparison.split('_vs_')
        print(f"\n{method1.upper()} vs {method2.upper()}:")
        
        # Throughput comparison (most important for Tree Method validation)
        throughput_t = test_results['throughput_t_test']
        throughput_u = test_results['throughput_mannwhitney']
        print(f"  Throughput (Completed Vehicles):")
        print(f"    T-test: t={throughput_t['t_statistic']:.3f}, p={throughput_t['p_value']:.4f} {'***' if throughput_t['significant'] else 'ns'}")
        print(f"    Mann-Whitney U: U={throughput_u['u_statistic']:.0f}, p={throughput_u['p_value']:.4f} {'***' if throughput_u['significant'] else 'ns'}")
        
        # Completion rate comparison
        completion_t = test_results['completion_rate_t_test']
        print(f"  Completion Rate:")
        print(f"    T-test: t={completion_t['t_statistic']:.3f}, p={completion_t['p_value']:.4f} {'***' if completion_t['significant'] else 'ns'}")
    
    print(f"\nLegend: *** = significant (p < 0.05), ns = not significant")
    
    # Save comprehensive validation results
    comprehensive_results = {
        'validation_summary': summary,
        'statistical_tests': statistical_tests,
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'dataset': 'Experiment1-realistic-high-load',
            'total_vehicles': 25470,
            'simulation_duration': 7300,
            'total_runs': sum(len(results) for results in all_results.values()),
            'methods_compared': list(all_results.keys()),
            'significance_level': 0.05
        }
    }
    
    validation_file = results_dir / "tree_method_validation.json"
    with open(validation_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"\nTree Method validation results saved to: {validation_file}")
    
    # Create validation plots
    print("\nGenerating Tree Method validation plots...")
    create_validation_plots(all_results, results_dir)
    
    print("\nTree Method validation analysis complete!")

if __name__ == "__main__":
    main()