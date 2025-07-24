#!/usr/bin/env python3

"""
Decentralized Traffic Bottleneck Run Analysis
Analyzes results from individual run comparing Tree Method vs Actuated vs Fixed
Compares against original decentralized-traffic-bottlenecks research results
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

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
        sim_steps_match = re.search(r'Total simulation steps:\s*(\d+)', content)
        if sim_steps_match:
            metrics['total_steps'] = int(sim_steps_match.group(1))
            
        # Extract vehicles entered from SUMO performance statistics
        inserted_match = re.search(r'Inserted:\s*(\d+)', content)
        if inserted_match:
            metrics['vehicles_entered'] = int(inserted_match.group(1))
            
        # Extract vehicles that completed from Statistics section
        # Pattern: "Statistics (avg of X):" where X is completed vehicles
        stats_match = re.search(r'Statistics \(avg of (\d+)\):', content)
        if stats_match:
            metrics['vehicles_arrived'] = int(stats_match.group(1))
            
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
            metrics['completion_rate'] = metrics['vehicles_arrived'] / metrics['vehicles_entered']
            
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        
    return metrics

def load_original_research_results(dataset_dir, run_number):
    """Load original decentralized-traffic-bottlenecks research results for specific run"""
    original_results = {
        'tree': {'vehicles_arrived': None, 'avg_duration': None},
        'actuated': {'vehicles_arrived': None, 'avg_duration': None},
        'fixed (uniform)': {'vehicles_arrived': None, 'avg_duration': None}
    }
    
    try:
        # Load vehicles arrived data
        arrived_file = dataset_dir / "results_vehicles_num_arrived.txt"
        if arrived_file.exists():
            with open(arrived_file, 'r') as f:
                lines = f.readlines()
                # Header is line 0, data starts at line 1, so run N is at line N
                if len(lines) > run_number:
                    data_line = lines[run_number].strip()
                    if data_line:
                        # Format: Run,Tree,Uniform,Actuated
                        parts = data_line.split(',')
                        if len(parts) >= 4:
                            original_results['tree']['vehicles_arrived'] = int(parts[1])
                            original_results['fixed (uniform)']['vehicles_arrived'] = int(parts[2])
                            original_results['actuated']['vehicles_arrived'] = int(parts[3])
        
        # Load average duration data
        duration_file = dataset_dir / "results_vehicles_avg_duration.txt"
        if duration_file.exists():
            with open(duration_file, 'r') as f:
                lines = f.readlines()
                # Header is line 0, data starts at line 1, so run N is at line N
                if len(lines) > run_number:
                    data_line = lines[run_number].strip()
                    if data_line:
                        # Format: Run,CurrentTree,Uniform,Actuated
                        parts = data_line.split(',')
                        if len(parts) >= 4:
                            # Note: CurrentTree is in column 1, Uniform in column 2, Actuated in column 3
                            original_results['tree']['avg_duration'] = int(parts[1])
                            original_results['fixed (uniform)']['avg_duration'] = int(parts[2])
                            original_results['actuated']['avg_duration'] = int(parts[3])
                            
    except Exception as e:
        print(f"Warning: Could not load original research results: {e}")
        
    return original_results

def analyze_run_results():
    """Main analysis function for individual run"""
    run_dir = Path(__file__).parent
    results_dir = run_dir / "results"
    
    # Extract experiment and run info
    experiment = run_dir.parent.name
    run_num = run_dir.name
    run_number = int(run_num)
    
    # Path to dataset directory with original results
    # Navigate from run_dir up to project root, then to dataset
    project_root = run_dir.parent.parent.parent.parent.parent  # Go up 5 levels from run dir  
    dataset_dir = project_root / 'evaluation' / 'datasets' / 'decentralized_traffic_bottleneck' / experiment
    
    print(f"DECENTRALIZED TRAFFIC BOTTLENECK ANALYSIS")
    print(f"Experiment: {experiment}")
    print(f"Run: {run_num}")
    print("=" * 60)
    
    # Extract our implementation results
    methods = ['tree', 'actuated', 'fixed']
    method_mapping = {
        'tree': 'tree_method',
        'actuated': 'actuated', 
        'fixed': 'fixed'
    }
    
    our_results = {}
    
    for method in methods:
        log_file = results_dir / method_mapping[method] / "simulation.log"
        if log_file.exists():
            our_results[method] = extract_metrics_from_log(log_file)
        else:
            print(f"Warning: {log_file} not found")
            our_results[method] = {}
    
    # Load original research results for this specific run
    original_results = load_original_research_results(dataset_dir, run_number)
    
    # Print method comparison
    print("\nOUR IMPLEMENTATION RESULTS:")
    print("-" * 40)
    for method, metrics in our_results.items():
        method_name = method.replace('_', ' ').title()
        print(f"\n{method_name}:")
        if metrics.get('vehicles_entered'):
            print(f"  Vehicles Entered: {metrics['vehicles_entered']:,}")
        if metrics.get('vehicles_arrived'):
            print(f"  Vehicles Arrived: {metrics['vehicles_arrived']:,}")
        if metrics.get('avg_duration'):
            print(f"  Avg Duration: {metrics['avg_duration']:.2f} steps")
        if metrics.get('completion_rate'):
            print(f"  Completion Rate: {metrics['completion_rate']:.2%}")
        if metrics.get('total_steps'):
            print(f"  Total Steps: {metrics['total_steps']:,}")
    
    # Print original research results
    print(f"\nORIGINAL RESEARCH RESULTS (Run {run_num}):")
    print("-" * 40)
    for method, metrics in original_results.items():
        print(f"\n{method.title()}:")
        if metrics.get('vehicles_arrived'):
            print(f"  Vehicles Arrived: {metrics['vehicles_arrived']:,}")
        if metrics.get('avg_duration'):
            print(f"  Avg Duration: {metrics['avg_duration']:.0f} steps")
    
    # Print comparison analysis
    print(f"\nCOMPARISON ANALYSIS:")
    print("-" * 40)
    
    comparison_mapping = {
        'tree': 'tree',
        'actuated': 'actuated',
        'fixed': 'fixed (uniform)'
    }
    
    for our_method, orig_method in comparison_mapping.items():
        if our_method in our_results and orig_method in original_results:
            our_metrics = our_results[our_method]
            orig_metrics = original_results[orig_method]
            method_name = our_method.replace('_', ' ').title()
            
            print(f"\n{method_name} Comparison:")
            
            # Compare vehicles arrived
            if our_metrics.get('vehicles_arrived') and orig_metrics.get('vehicles_arrived'):
                our_arrived = our_metrics['vehicles_arrived']
                orig_arrived = orig_metrics['vehicles_arrived']
                diff_pct = ((our_arrived - orig_arrived) / orig_arrived) * 100
                print(f"  Vehicles Arrived: {our_arrived:,} vs {orig_arrived:,} ({diff_pct:+.1f}%)")
            
            # Compare average duration
            if our_metrics.get('avg_duration') and orig_metrics.get('avg_duration'):
                our_duration = our_metrics['avg_duration']
                orig_duration = orig_metrics['avg_duration']
                diff_pct = ((our_duration - orig_duration) / orig_duration) * 100
                print(f"  Avg Duration: {our_duration:.1f} vs {orig_duration:.0f} steps ({diff_pct:+.1f}%)")
    
    # Save results in format matching your modifications
    analysis_results = {
        'experiment': experiment,
        'run': run_num,
        'our_results': {
            'tree': our_results.get('tree', {}),
            'actuated': our_results.get('actuated', {}),
            'fixed': our_results.get('fixed', {})
        },
        'decentralized_traffic_bottlenecks_original_results': original_results,
        'analysis_date': datetime.now().isoformat()
    }
    
    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to: {results_file}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_run_results()