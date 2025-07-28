#!/usr/bin/env python3

"""
Experiment-Level Analysis for Synthetic Grid 5x5
Aggregates and analyzes results from all experiments in this grid size
Provides comprehensive statistical analysis and performance comparisons
"""

import os
import json
import glob
import subprocess
import sys
import re
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


def extract_metrics_from_log(log_file):
    """Extract traffic metrics from simulation log"""
    metrics = {
        'vehicles_entered': None,
        'vehicles_arrived': None,
        'avg_duration': None,
        'completion_rate': None,
        'total_steps': None
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Extract total simulation steps
        sim_steps_match = re.search(
            r'Total simulation steps:\s*(\d+)', content)
        if sim_steps_match:
            metrics['total_steps'] = int(sim_steps_match.group(1))

        # Extract vehicles entered - multiple patterns for different log formats
        # Pattern 1: SUMO "Inserted:" statistics (decentralized-traffic-bottlenecks format)
        inserted_match = re.search(r'Inserted:\s*(\d+)', content)
        if inserted_match:
            metrics['vehicles_entered'] = int(inserted_match.group(1))
        else:
            # Pattern 2: Extract from final SUMO step line (synthetic-grids format)
            # Find the last "vehicles TOT X" line
            vehicles_tot_matches = re.findall(r'vehicles TOT (\d+)', content)
            if vehicles_tot_matches:
                metrics['vehicles_entered'] = int(vehicles_tot_matches[-1])

        # Extract vehicles that completed - multiple patterns for different log formats
        # Pattern 1: Statistics section "Statistics (avg of X):" (decentralized-traffic-bottlenecks)
        stats_match = re.search(r'Statistics \(avg of (\d+)\):', content)
        if stats_match:
            metrics['vehicles_arrived'] = int(stats_match.group(1))
        else:
            # Pattern 2: Tree Method controller statistics (synthetic-grids format)
            tree_completed_match = re.search(
                r'Tree Method - Vehicles completed:\s*(\d+)', content)
            if tree_completed_match:
                metrics['vehicles_arrived'] = int(
                    tree_completed_match.group(1))
            else:
                # Pattern 3: Check if simulation ended with 0 active vehicles (all completed)
                # Find all matches and take the last one (final state)
                vehicles_tot_act_matches = re.findall(
                    r'vehicles TOT (\d+) ACT 0', content)
                if vehicles_tot_act_matches:
                    metrics['vehicles_arrived'] = int(
                        vehicles_tot_act_matches[-1])

        # Extract average duration from Graph object statistics (priority) or SUMO statistics (fallback)
        # Try Graph object duration first (Tree Method, Fixed, Actuated)
        graph_duration_patterns = [
            r'Tree Method - Average duration:\s*([\d.]+)\s*steps',
            r'Fixed - Average duration:\s*([\d.]+)\s*steps',
            r'Actuated - Average duration:\s*([\d.]+)\s*steps'
        ]

        graph_duration_found = False
        for pattern in graph_duration_patterns:
            graph_duration_match = re.search(pattern, content)
            if graph_duration_match:
                metrics['avg_duration'] = float(graph_duration_match.group(1))
                graph_duration_found = True
                break

        # Fallback to SUMO statistics if Graph object duration not found
        if not graph_duration_found:
            duration_match = re.search(r'Duration:\s*([\d.]+)', content)
            if duration_match:
                metrics['avg_duration'] = float(duration_match.group(1))

        # Calculate completion rate
        if metrics['vehicles_arrived'] is not None and metrics['vehicles_entered'] is not None and metrics['vehicles_entered'] > 0:
            metrics['completion_rate'] = metrics['vehicles_arrived'] / \
                metrics['vehicles_entered']

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics


def load_run_results(experiment_dir):
    """Load results from all runs by directly parsing simulation logs"""
    all_results = {
        'tree_method': [],
        'actuated': [],
        'fixed': []
    }

    print("Loading results directly from simulation logs...")

    # Find all run directories (numbered directories)
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()
                and d.name.isdigit()]
    run_dirs.sort(key=lambda x: int(x.name))

    processed_count = 0

    for run_dir in run_dirs:
        results_dir = run_dir / "results"

        if not results_dir.exists():
            continue

        run_processed = False

        # Extract results for each method by directly parsing simulation logs
        method_mapping = {
            'tree_method': 'tree_method',
            'actuated': 'actuated',
            'fixed': 'fixed'
        }

        for method_key, method_dir in method_mapping.items():
            log_file = results_dir / method_dir / "simulation.log"

            if log_file.exists() and log_file.stat().st_size > 0:
                try:
                    metrics = extract_metrics_from_log(log_file)
                    if any(metrics[key] is not None for key in ['vehicles_entered', 'vehicles_arrived', 'avg_duration']):
                        metrics['run'] = int(run_dir.name)
                        all_results[method_key].append(metrics)
                        run_processed = True

                except Exception as e:
                    print(f"Warning: Could not parse {log_file}: {e}")

        if run_processed:
            processed_count += 1

    print(f"Processed {processed_count} runs with simulation results")
    return all_results


def calculate_statistics(data_list, metric):
    """Calculate statistics for a given metric across all runs"""
    values = [item[metric]
              for item in data_list if item.get(metric) is not None]
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
    
    # Capture all screen output
    analysis_output = []
    
    def capture_print(*args, **kwargs):
        """Capture print output while still displaying it"""
        output = ' '.join(str(arg) for arg in args)
        # Clean up the output: remove leading newlines and emojis
        clean_output = output.lstrip('\n')
        # Replace emojis with plain text
        clean_output = clean_output.replace('📄', 'FILE:')
        clean_output = clean_output.replace('🎉', 'COMPLETE:')
        clean_output = clean_output.replace('✅', 'SUCCESS:')
        clean_output = clean_output.replace('📊', 'STATS:')
        clean_output = clean_output.replace('💾', 'SAVED:')
        
        if clean_output.strip():  # Only add non-empty lines
            analysis_output.append(clean_output)
        print(*args, **kwargs)

    capture_print(f"SYNTHETIC GRID EXPERIMENT ANALYSIS")
    capture_print(f"Grid Size: {experiment_name}")
    capture_print("=" * 80)
    capture_print()

    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)

    grid_config = config['grid_configurations']['5x5']

    # Load results from all runs by directly parsing simulation logs
    all_results = load_run_results(experiment_dir)

    # Check if we have any results
    total_results = sum(len(results) for results in all_results.values())
    if total_results == 0:
        print("❌ No results found! Please run experiments first.")
        print("   Use: ./run_all_runs.sh")
        sys.exit(1)

    # Generate summary statistics
    capture_print(
        f"\nSUMMARY STATISTICS (5x5 Grid - {grid_config['dimension']}x{grid_config['dimension']}):")
    capture_print("=" * 70)

    summary_stats = {}

    for method in ['tree_method', 'actuated', 'fixed']:
        results = all_results[method]
        method_name = method.replace('_', ' ').title()

        capture_print(f"\n{method_name}:")
        capture_print(f"  Runs completed: {len(results)}")

        if results:
            # Calculate totals for each metric
            for metric in ['vehicles_entered', 'vehicles_arrived', 'avg_duration', 'completion_rate']:
                stats = calculate_statistics(results, metric)
                if stats:
                    if metric in ['vehicles_entered', 'vehicles_arrived']:
                        # Show total sum for vehicle counts
                        total = sum(stats['values'])
                        capture_print(
                            f"  {metric.replace('_', ' ').title()}: {total:,.0f}")
                    elif metric == 'avg_duration':
                        # Show average duration across all runs
                        capture_print(
                            f"  {metric.replace('_', ' ').title()}: {stats['mean']:.1f} steps")
                    elif metric == 'completion_rate':
                        # Calculate overall completion rate from totals
                        vehicles_entered_stats = calculate_statistics(
                            results, 'vehicles_entered')
                        vehicles_arrived_stats = calculate_statistics(
                            results, 'vehicles_arrived')
                        if vehicles_entered_stats and vehicles_arrived_stats:
                            total_entered = sum(
                                vehicles_entered_stats['values'])
                            total_arrived = sum(
                                vehicles_arrived_stats['values'])
                            overall_rate = total_arrived / total_entered if total_entered > 0 else 0
                            capture_print(
                                f"  {metric.replace('_', ' ').title()}: {overall_rate:.3%}")

                    # Store for comparisons
                    if method not in summary_stats:
                        summary_stats[method] = {}
                    summary_stats[method][metric] = stats

    # Method-to-method comparisons
    capture_print(f"\nMETHOD PERFORMANCE COMPARISON (5x5 Grid):")
    capture_print("=" * 50)

    comparisons = [
        ('tree_method', 'actuated'),
        ('tree_method', 'fixed'),
        ('actuated', 'fixed')
    ]

    for method1, method2 in comparisons:
        if method1 in summary_stats and method2 in summary_stats:
            method1_name = method1.replace('_', ' ').title()
            method2_name = method2.replace('_', ' ').title()
            capture_print(f"\n{method1_name} vs {method2_name}:")

            # Vehicles arrived comparison - use totals
            if 'vehicles_arrived' in summary_stats[method1] and 'vehicles_arrived' in summary_stats[method2]:
                total1 = sum(summary_stats[method1]
                             ['vehicles_arrived']['values'])
                total2 = sum(summary_stats[method2]
                             ['vehicles_arrived']['values'])
                improvement = ((total1 - total2) / total2) * 100
                capture_print(
                    f"  Vehicles Arrived: {total1:,.0f} vs {total2:,.0f} ({improvement:+.1f}%)")

            # Duration comparison - use averages
            if 'avg_duration' in summary_stats[method1] and 'avg_duration' in summary_stats[method2]:
                mean1 = summary_stats[method1]['avg_duration']['mean']
                mean2 = summary_stats[method2]['avg_duration']['mean']
                improvement = ((mean2 - mean1) / mean2) * \
                    100  # Lower is better for duration
                capture_print(
                    f"  Avg Duration: {mean1:.1f} vs {mean2:.1f} steps ({improvement:+.1f}% improvement)")

            # Completion rate comparison - use overall rates
            if 'vehicles_entered' in summary_stats[method1] and 'vehicles_arrived' in summary_stats[method1] and 'vehicles_entered' in summary_stats[method2] and 'vehicles_arrived' in summary_stats[method2]:
                entered1 = sum(summary_stats[method1]
                               ['vehicles_entered']['values'])
                arrived1 = sum(summary_stats[method1]
                               ['vehicles_arrived']['values'])
                rate1 = arrived1 / entered1 if entered1 > 0 else 0

                entered2 = sum(summary_stats[method2]
                               ['vehicles_entered']['values'])
                arrived2 = sum(summary_stats[method2]
                               ['vehicles_arrived']['values'])
                rate2 = arrived2 / entered2 if entered2 > 0 else 0

                improvement = ((rate1 - rate2) / rate2) * \
                    100 if rate2 > 0 else 0
                capture_print(
                    f"  Completion Rate: {rate1:.3%} vs {rate2:.3%} ({improvement:+.1f}%)")

    # Save comprehensive results
    experiment_results = {
        'experiment': experiment_name,
        'grid_configuration': grid_config,
        'configuration_used': {
            'vehicle_count_levels': list(grid_config['vehicle_counts'].keys()),
            'vehicle_types': config['shared_parameters']['vehicle_types'],
            'routing_strategies': config['shared_parameters']['routing_strategies'],
            'departure_patterns': config['shared_parameters']['departure_patterns'],
            'simulation_durations': config['shared_parameters']['simulation_durations'],
            'junctions_removed': config['shared_parameters']['junctions_removed']
        },
        'analysis_output': analysis_output,
        'analysis_date': datetime.now().isoformat(),
        'total_runs_analyzed': sum(len(results) for results in all_results.values()),
        'runs_per_method': {method: len(results) for method, results in all_results.items()}
    }

    results_file = experiment_dir / \
        f"{experiment_name}_experiment_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)

    capture_print(f"\n📄 Experiment analysis saved to: {results_file}")
    capture_print(f"\n🎉 5x5 GRID ANALYSIS COMPLETE!")
    capture_print("=" * 40)
    capture_print(
        f"✅ Methods analyzed: {len([m for m in summary_stats.keys() if summary_stats[m]])}/3")
    capture_print(f"📊 Total results processed: {total_results}")
    capture_print(f"💾 Results saved to: {results_file.name}")
    
    # Update the analysis output in the results after final messages
    experiment_results['analysis_output'] = analysis_output
    
    # Save the updated results with complete output
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)


if __name__ == "__main__":
    analyze_experiment()
