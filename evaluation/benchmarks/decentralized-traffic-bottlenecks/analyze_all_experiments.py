#!/usr/bin/env python3

"""
Master Analysis for All Decentralized Traffic Bottleneck Experiments
Aggregates and compares results across all 4 experiments (80 total runs)
Provides comprehensive validation against original Tree Method research
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
import statistics

def load_all_experiment_results():
    """Load results from all experiments using analysis_results.json files"""
    benchmark_dir = Path(__file__).parent
    experiments = ['Experiment1-realistic-high-load', 'Experiment2-rand-high-load', 
                   'Experiment3-realistic-moderate-load', 'Experiment4-and-moderate-load']
    
    all_results = {
        'tree': [],
        'actuated': [],
        'fixed': []
    }
    
    all_original_results = {
        'tree': [],
        'actuated': [],
        'fixed (uniform)': []
    }
    
    experiment_summaries = {}
    
    print("Loading results from all experiments...")
    
    for experiment in experiments:
        print(f"  Processing {experiment}...")
        exp_dir = benchmark_dir / experiment
        
        if not exp_dir.exists():
            print(f"    Warning: {experiment} directory not found")
            continue
        
        exp_results = {
            'tree': [],
            'actuated': [],
            'fixed': []
        }
        
        # Load individual run results
        runs_found = 0
        for run in range(1, 21):
            results_file = exp_dir / str(run) / "results" / "analysis_results.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract our implementation results
                    our_results = data.get('our_results', {})
                    for method in ['tree', 'actuated', 'fixed']:
                        if method in our_results and our_results[method]:
                            result = our_results[method].copy()
                            result['experiment'] = experiment
                            result['run'] = run
                            all_results[method].append(result)
                            exp_results[method].append(result)
                    
                    runs_found += 1
                    
                except Exception as e:
                    print(f"    Warning: Could not load {experiment} run {run}: {e}")
        
        print(f"    Found {runs_found}/20 completed runs")
        experiment_summaries[experiment] = {
            'runs_completed': runs_found,
            'results': exp_results
        }
    
    # Load original research results from dataset CSV files
    print("  Loading original research results...")
    for experiment in experiments:
        dataset_dir = benchmark_dir.parent.parent.parent / 'datasets' / 'decentralized_traffic_bottleneck' / experiment
        
        if dataset_dir.exists():
            orig_results = load_original_dataset_results(dataset_dir)
            for method, data in orig_results.items():
                if 'values' in data.get('vehicles_arrived', {}):
                    for value in data['vehicles_arrived']['values']:
                        all_original_results[method].append({
                            'experiment': experiment,
                            'vehicles_arrived': value
                        })
    
    return all_results, all_original_results, experiment_summaries

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
                        
    except Exception as e:
        print(f"    Warning: Could not load original dataset results from {dataset_dir}: {e}")
    
    return original_results

def calculate_method_statistics(method_data, metric):
    """Calculate statistics for a method across all runs"""
    values = [run[metric] for run in method_data if run.get(metric) is not None]
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

def analyze_master_results():
    """Main master analysis function"""
    print("DECENTRALIZED TRAFFIC BOTTLENECK MASTER ANALYSIS")
    print("All Experiments Comprehensive Validation")
    print("=" * 80)
    
    # Load all results
    all_results, all_original_results, experiment_summaries = load_all_experiment_results()
    
    # Check if we have any results
    total_runs = sum(len(results) for results in all_results.values())
    if total_runs == 0:
        print("❌ No results found! Please run experiments first.")
        print("   Use: ./run_all_experiments.sh")
        return
    
    total_completed_runs = total_runs // 3  # Divide by 3 methods
    print(f"📊 Total results collected: {total_runs} (from {total_completed_runs} completed runs)")
    print(f"📈 Target: 240 results (from 80 runs)")
    print(f"🎯 Completion: {total_runs}/240 ({total_runs/240*100:.1f}%)")
    print("")
    
    # Per-experiment summary
    print("PER-EXPERIMENT SUMMARY:")
    print("=" * 40)
    for experiment, summary in experiment_summaries.items():
        runs_completed = summary['runs_completed']
        print(f"{experiment}: {runs_completed}/20 runs completed ({runs_completed/20*100:.0f}%)")
    print("")
    
    # Overall method statistics
    print("OVERALL METHOD STATISTICS (All Experiments Combined):")
    print("=" * 60)
    
    master_stats = {}
    
    for method in ['tree', 'actuated', 'fixed']:
        method_data = all_results[method]
        method_name = method.replace('_', ' ').title()
        
        print(f"\n{method_name}:")
        print(f"  Total runs: {len(method_data)}")
        
        if method_data:
            # Calculate statistics for each metric
            for metric in ['vehicles_entered', 'vehicles_arrived', 'avg_duration', 'completion_rate']:
                stats = calculate_method_statistics(method_data, metric)
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
                    if method not in master_stats:
                        master_stats[method] = {}
                    master_stats[method][metric] = stats

    # Load original research statistics
    print("\nORIGINAL RESEARCH COMPARISON (All Experiments):")
    print("=" * 55)
    
    original_stats = {}
    for method, data_list in all_original_results.items():
        if data_list:
            values = [item['vehicles_arrived'] for item in data_list]
            original_stats[method] = {
                'vehicles_arrived': {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'count': len(values),
                    'values': values
                }
            }
    
    # Compare our results with original research
    comparison_mapping = {
        'tree': 'tree',
        'actuated': 'actuated',
        'fixed': 'fixed (uniform)'
    }
    
    for our_method, orig_method in comparison_mapping.items():
        if our_method in master_stats and orig_method in original_stats:
            our_stats = master_stats[our_method]
            orig_stats = original_stats[orig_method]
            
            method_name = our_method.replace('_', ' ').title()
            print(f"\n{method_name} vs Original Research:")
            
            # Compare vehicles arrived
            if 'vehicles_arrived' in our_stats and 'vehicles_arrived' in orig_stats:
                our_mean = our_stats['vehicles_arrived']['mean']
                orig_mean = orig_stats['vehicles_arrived']['mean']
                diff_pct = ((our_mean - orig_mean) / orig_mean) * 100
                print(f"  Vehicles Arrived: {our_mean:,.0f} vs {orig_mean:,.0f} ({diff_pct:+.1f}%)")
    
    # Method-to-method performance comparison
    print(f"\nMETHOD PERFORMANCE COMPARISON:")
    print("=" * 40)
    
    comparisons = [
        ('tree', 'actuated'),
        ('tree', 'fixed'),
        ('actuated', 'fixed')
    ]
    
    for method1, method2 in comparisons:
        if method1 in master_stats and method2 in master_stats:
            print(f"\n{method1.title()} vs {method2.title()}:")
            
            # Vehicles arrived comparison
            if 'vehicles_arrived' in master_stats[method1] and 'vehicles_arrived' in master_stats[method2]:
                mean1 = master_stats[method1]['vehicles_arrived']['mean']
                mean2 = master_stats[method2]['vehicles_arrived']['mean']
                improvement = ((mean1 - mean2) / mean2) * 100
                print(f"  Vehicles Arrived: {mean1:,.0f} vs {mean2:,.0f} ({improvement:+.1f}%)")
            
            # Duration comparison (lower is better)
            if 'avg_duration' in master_stats[method1] and 'avg_duration' in master_stats[method2]:
                mean1 = master_stats[method1]['avg_duration']['mean']
                mean2 = master_stats[method2]['avg_duration']['mean']
                improvement = ((mean2 - mean1) / mean2) * 100  # Lower is better
                print(f"  Avg Duration: {mean1:.1f} vs {mean2:.1f} steps ({improvement:+.1f}% improvement)")
            
            # Completion rate comparison
            if 'completion_rate' in master_stats[method1] and 'completion_rate' in master_stats[method2]:
                mean1 = master_stats[method1]['completion_rate']['mean']
                mean2 = master_stats[method2]['completion_rate']['mean']
                improvement = ((mean1 - mean2) / mean2) * 100
                print(f"  Completion Rate: {mean1:.3%} vs {mean2:.3%} ({improvement:+.1f}%)")

    # Tree Method validation summary
    if 'tree' in master_stats:
        print(f"\nTREE METHOD VALIDATION SUMMARY:")
        print("=" * 45)
        
        tree_stats = master_stats['tree']
        
        if 'actuated' in master_stats and 'fixed' in master_stats:
            actuated_stats = master_stats['actuated']
            fixed_stats = master_stats['fixed']
            
            # Calculate improvements
            if 'vehicles_arrived' in tree_stats and 'vehicles_arrived' in actuated_stats:
                tree_vs_actuated = ((tree_stats['vehicles_arrived']['mean'] - actuated_stats['vehicles_arrived']['mean']) / actuated_stats['vehicles_arrived']['mean']) * 100
                print(f"Tree Method vs SUMO Actuated:")
                print(f"  Vehicles Arrived Improvement: {tree_vs_actuated:+.1f}%")
            
            if 'vehicles_arrived' in tree_stats and 'vehicles_arrived' in fixed_stats:
                tree_vs_fixed = ((tree_stats['vehicles_arrived']['mean'] - fixed_stats['vehicles_arrived']['mean']) / fixed_stats['vehicles_arrived']['mean']) * 100
                print(f"\nTree Method vs Fixed Timing:")
                print(f"  Vehicles Arrived Improvement: {tree_vs_fixed:+.1f}%")
            
            if 'completion_rate' in tree_stats and 'completion_rate' in actuated_stats:
                completion_improvement = (tree_stats['completion_rate']['mean'] - actuated_stats['completion_rate']['mean']) * 100
                print(f"\nCompletion Rate Advantage:")
                print(f"  vs Actuated: {completion_improvement:+.2f} percentage points")
            
            if 'completion_rate' in tree_stats and 'completion_rate' in fixed_stats:
                completion_improvement = (tree_stats['completion_rate']['mean'] - fixed_stats['completion_rate']['mean']) * 100
                print(f"  vs Fixed: {completion_improvement:+.2f} percentage points")

    # Save comprehensive master results
    master_results = {
        'analysis_date': datetime.now().isoformat(),
        'total_runs_analyzed': total_completed_runs,
        'total_results_collected': total_runs,
        'completion_percentage': round(total_runs/240*100, 1),
        'experiment_summaries': {
            exp: {'runs_completed': summary['runs_completed']}
            for exp, summary in experiment_summaries.items()
        },
        'our_implementation_master_stats': {
            method: {
                metric: {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'count': stats['count']
                }
                for metric, stats in method_stats.items()
            }
            for method, method_stats in master_stats.items()
        },
        'original_research_comparison': {
            method: {
                'vehicles_arrived': {
                    'mean': float(stats['vehicles_arrived']['mean']),
                    'std': float(stats['vehicles_arrived']['std']),
                    'count': stats['vehicles_arrived']['count']
                }
            }
            for method, stats in original_stats.items()
        }
    }
    
    # Add Tree Method validation metrics if available
    if 'tree' in master_stats and 'actuated' in master_stats and 'fixed' in master_stats:
        tree_data = master_stats['tree']
        actuated_data = master_stats['actuated']
        fixed_data = master_stats['fixed']
        
        validation_metrics = {}
        
        if 'vehicles_arrived' in tree_data and 'vehicles_arrived' in actuated_data:
            validation_metrics['vs_actuated_vehicles_improvement_pct'] = float(
                ((tree_data['vehicles_arrived']['mean'] - actuated_data['vehicles_arrived']['mean']) / actuated_data['vehicles_arrived']['mean']) * 100
            )
        
        if 'vehicles_arrived' in tree_data and 'vehicles_arrived' in fixed_data:
            validation_metrics['vs_fixed_vehicles_improvement_pct'] = float(
                ((tree_data['vehicles_arrived']['mean'] - fixed_data['vehicles_arrived']['mean']) / fixed_data['vehicles_arrived']['mean']) * 100
            )
        
        if 'completion_rate' in tree_data and 'completion_rate' in actuated_data:
            validation_metrics['vs_actuated_completion_improvement_pts'] = float(
                (tree_data['completion_rate']['mean'] - actuated_data['completion_rate']['mean']) * 100
            )
        
        if 'completion_rate' in tree_data and 'completion_rate' in fixed_data:
            validation_metrics['vs_fixed_completion_improvement_pts'] = float(
                (tree_data['completion_rate']['mean'] - fixed_data['completion_rate']['mean']) * 100
            )
        
        master_results['tree_method_validation'] = validation_metrics

    results_file = Path(__file__).parent / "master_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    
    print(f"\n📄 Master analysis results saved to: {results_file}")
    
    # Summary and next steps
    print(f"\n🎯 MASTER ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"Analyzed {total_completed_runs}/80 completed runs ({total_completed_runs/80*100:.1f}%)")
    print(f"Total simulation results: {total_runs}/240")
    print("")
    print("📋 Next steps:")
    if total_completed_runs < 80:
        print("  1. Complete remaining experiments with: ./run_all_experiments.sh")
        print("  2. Re-run this analysis after more experiments complete")
    else:
        print("  1. Review detailed results in master_analysis_results.json")
        print("  2. Check individual experiment analyses in each experiment folder")
        print("  3. Use results for research validation and publication")
    print("")
    print("✅ Master analysis complete!")

if __name__ == "__main__":
    analyze_master_results()