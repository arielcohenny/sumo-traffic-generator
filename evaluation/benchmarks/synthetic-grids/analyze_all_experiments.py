#!/usr/bin/env python3

"""
Master Analysis Script for Synthetic Grid Benchmark Suite
Aggregates and analyzes results across all grid sizes with comprehensive statistical analysis
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import statistics

def load_config():
    """Load the central configuration file"""
    config_path = Path(__file__).parent / "experiment_config.json"
    
    if not config_path.exists():
        print("❌ ERROR: experiment_config.json not found")
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Could not load configuration: {e}")
        return None

def run_grid_analysis(grid_name):
    """Execute analysis for a specific grid size"""
    grid_dir = Path(__file__).parent / f"grids-{grid_name}"
    analyze_script = grid_dir / "analyze_experiment.py"
    
    if not analyze_script.exists():
        print(f"   ⚠️  Warning: Analysis script not found for {grid_name}")
        return False
    
    try:
        print(f"   Analyzing {grid_name} grid...")
        result = subprocess.run(
            [sys.executable, str(analyze_script)],
            cwd=str(grid_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for analysis
        )
        
        if result.returncode == 0:
            print(f"   ✅ {grid_name} analysis completed")
            return True
        else:
            print(f"   ❌ {grid_name} analysis failed: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {grid_name} analysis timed out")
        return False
    except Exception as e:
        print(f"   ❌ {grid_name} analysis error: {e}")
        return False

def load_grid_results(config):
    """Load analysis results from all grid sizes"""
    all_results = {
        'tree_method': [],
        'actuated': [],
        'fixed': []
    }
    
    grid_summaries = {}
    
    for grid_name in config['grid_configurations'].keys():
        grid_dir = Path(__file__).parent / f"grids-{grid_name}"
        results_file = grid_dir / f"grids-{grid_name}_experiment_analysis.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract grid-level summary
                grid_summaries[grid_name] = data.get('our_implementation_summary', {})
                
                # Add individual results with grid identification
                for method in ['tree', 'actuated', 'fixed']:
                    method_key = 'tree_method' if method == 'tree' else method
                    if method in data.get('our_implementation_summary', {}):
                        method_data = data['our_implementation_summary'][method].copy()
                        method_data['grid_size'] = grid_name
                        all_results[method_key].append(method_data)
                        
            except Exception as e:
                print(f"   Warning: Could not load results for {grid_name}: {e}")
        else:
            print(f"   Warning: No results file found for {grid_name}")
    
    return all_results, grid_summaries

def calculate_method_statistics(method_results, metric):
    """Calculate statistics for a method across all grids"""
    values = []
    for grid_data in method_results:
        if metric in grid_data and grid_data[metric] is not None:
            if isinstance(grid_data[metric], dict) and 'mean' in grid_data[metric]:
                values.append(grid_data[metric]['mean'])
            else:
                values.append(grid_data[metric])
    
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

def analyze_scalability(all_results, config):
    """Analyze how performance scales with grid size"""
    scalability_analysis = {}
    
    # Group results by grid size for each method
    for method in ['tree_method', 'actuated', 'fixed']:
        grid_performance = {}
        
        for result in all_results[method]:
            grid_size = result['grid_size']
            if grid_size not in grid_performance:
                grid_performance[grid_size] = []
            grid_performance[grid_size].append(result)
        
        # Calculate average performance per grid size
        grid_averages = {}
        for grid_size, results in grid_performance.items():
            grid_config = config['grid_configurations'][grid_size]
            
            avg_metrics = {}
            for metric in ['vehicles_entered', 'vehicles_arrived', 'avg_duration', 'completion_rate']:
                values = []
                for result in results:
                    if metric in result and result[metric] is not None:
                        if isinstance(result[metric], dict) and 'mean' in result[metric]:
                            values.append(result[metric]['mean'])
                        else:
                            values.append(result[metric])
                
                if values:
                    avg_metrics[metric] = {
                        'mean': statistics.mean(values),
                        'grid_dimension': grid_config['dimension'],
                        'grid_area': grid_config['dimension'] ** 2
                    }
            
            grid_averages[grid_size] = avg_metrics
        
        scalability_analysis[method] = grid_averages
    
    return scalability_analysis

def generate_performance_comparison(all_results):
    """Generate comprehensive performance comparison between methods"""
    comparison = {}
    
    methods = ['tree_method', 'actuated', 'fixed']
    metrics = ['vehicles_entered', 'vehicles_arrived', 'avg_duration', 'completion_rate']
    
    # Calculate overall statistics for each method
    method_stats = {}
    for method in methods:
        method_stats[method] = {}
        for metric in metrics:
            stats = calculate_method_statistics(all_results[method], metric)
            if stats:
                method_stats[method][metric] = stats
    
    # Generate pairwise comparisons
    comparisons = [
        ('tree_method', 'actuated'),
        ('tree_method', 'fixed'),
        ('actuated', 'fixed')
    ]
    
    for method1, method2 in comparisons:
        comparison_key = f"{method1}_vs_{method2}"
        comparison[comparison_key] = {}
        
        for metric in metrics:
            if (method1 in method_stats and metric in method_stats[method1] and
                method2 in method_stats and metric in method_stats[method2]):
                
                mean1 = method_stats[method1][metric]['mean']
                mean2 = method_stats[method2][metric]['mean']
                
                if metric == 'avg_duration':
                    # Lower is better for duration
                    improvement = ((mean2 - mean1) / mean2) * 100
                else:
                    # Higher is better for other metrics
                    improvement = ((mean1 - mean2) / mean2) * 100
                
                comparison[comparison_key][metric] = {
                    f'{method1}_mean': mean1,
                    f'{method2}_mean': mean2,
                    'improvement_pct': improvement
                }
    
    return comparison, method_stats

def display_analysis_summary(all_results, grid_summaries, scalability, comparison, method_stats, config):
    """Display comprehensive analysis summary"""
    print(f"\n{'=' * 80}")
    print("🔬 SYNTHETIC GRID BENCHMARK ANALYSIS")
    print(f"{'=' * 80}")
    
    # Overall summary
    total_experiments = sum(len(results) for results in all_results.values())
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Grid sizes analyzed: {len(grid_summaries)}")
    print(f"   Total method results: {total_experiments}")
    print(f"   Results per method: Tree={len(all_results['tree_method'])}, Actuated={len(all_results['actuated'])}, Fixed={len(all_results['fixed'])}")
    
    # Per-grid summary
    print(f"\n📈 PER-GRID RESULTS:")
    for grid_name, summary in grid_summaries.items():
        grid_config = config['grid_configurations'][grid_name]
        print(f"   {grid_name} (dimension {grid_config['dimension']}):")
        for method in ['tree', 'actuated', 'fixed']:
            if method in summary:
                if 'vehicles_arrived' in summary[method]:
                    arrived = summary[method]['vehicles_arrived']['mean']
                    completion = summary[method]['completion_rate']['mean'] if 'completion_rate' in summary[method] else 0
                    print(f"      {method.title()}: {arrived:,.0f} vehicles ({completion:.1%} completion)")
    
    # Method performance comparison
    print(f"\n🏆 OVERALL METHOD PERFORMANCE:")
    for method in ['tree_method', 'actuated', 'fixed']:
        if method in method_stats:
            method_name = method.replace('_', ' ').title()
            print(f"\n   {method_name}:")
            for metric in ['vehicles_arrived', 'completion_rate', 'avg_duration']:
                if metric in method_stats[method]:
                    stats = method_stats[method][metric]
                    if metric == 'completion_rate':
                        print(f"      {metric.replace('_', ' ').title()}: {stats['mean']:.3%} ± {stats['std']:.3%}")
                    elif metric == 'avg_duration':
                        print(f"      {metric.replace('_', ' ').title()}: {stats['mean']:.1f} ± {stats['std']:.1f} steps")
                    else:
                        print(f"      {metric.replace('_', ' ').title()}: {stats['mean']:,.0f} ± {stats['std']:,.0f}")
    
    # Tree Method validation summary
    if 'tree_method_vs_actuated' in comparison:
        print(f"\n🌳 TREE METHOD VALIDATION:")
        
        tree_vs_actuated = comparison['tree_method_vs_actuated']
        tree_vs_fixed = comparison['tree_method_vs_fixed']
        
        if 'vehicles_arrived' in tree_vs_actuated:
            print(f"   vs SUMO Actuated:")
            print(f"      Vehicles Arrived: {tree_vs_actuated['vehicles_arrived']['improvement_pct']:+.1f}% improvement")
            
        if 'vehicles_arrived' in tree_vs_fixed:
            print(f"   vs Fixed Timing:")
            print(f"      Vehicles Arrived: {tree_vs_fixed['vehicles_arrived']['improvement_pct']:+.1f}% improvement")
            
        if 'completion_rate' in tree_vs_actuated and 'completion_rate' in tree_vs_fixed:
            print(f"   Completion Rate Advantage:")
            actuated_comp = tree_vs_actuated['completion_rate']['improvement_pct']
            fixed_comp = tree_vs_fixed['completion_rate']['improvement_pct']
            print(f"      vs Actuated: {actuated_comp:+.1f} percentage points")
            if 'completion_rate' in tree_vs_fixed:
                print(f"      vs Fixed: {fixed_comp:+.1f} percentage points")

def save_master_analysis(all_results, grid_summaries, scalability, comparison, method_stats, config):
    """Save comprehensive master analysis results"""
    master_results = {
        'analysis_metadata': {
            'analysis_date': datetime.now().isoformat(),
            'framework_version': config['experiment_metadata']['version'],
            'analyzer_version': '1.0'
        },
        'experiment_summary': {
            'total_grids_analyzed': len(grid_summaries),
            'total_method_results': sum(len(results) for results in all_results.values()),
            'grids_analyzed': list(grid_summaries.keys())
        },
        'per_grid_summaries': grid_summaries,
        'overall_method_statistics': method_stats,
        'scalability_analysis': scalability,
        'performance_comparisons': comparison,
        'tree_method_validation': {
            'vs_actuated_improvement': comparison.get('tree_method_vs_actuated', {}),
            'vs_fixed_improvement': comparison.get('tree_method_vs_fixed', {})
        }
    }
    
    results_file = Path(__file__).parent / "master_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(master_results, f, indent=2, default=str)
    
    return results_file

def main():
    """Main analysis function"""
    print("SYNTHETIC GRID BENCHMARK SUITE - MASTER ANALYSIS")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Run individual grid analyses first
    print("\n🔄 Running individual grid analyses...")
    
    analysis_results = {}
    for grid_name in config['grid_configurations'].keys():
        analysis_results[grid_name] = run_grid_analysis(grid_name)
    
    successful_analyses = sum(1 for success in analysis_results.values() if success)
    total_grids = len(analysis_results)
    
    print(f"\n✅ Grid analyses completed: {successful_analyses}/{total_grids}")
    
    if successful_analyses == 0:
        print("❌ No grid analyses completed successfully. Cannot proceed with master analysis.")
        sys.exit(1)
    
    # Load all results
    print("\n📊 Loading and aggregating results...")
    all_results, grid_summaries = load_grid_results(config)
    
    if not any(all_results.values()):
        print("❌ No experiment results found. Please run experiments first.")
        sys.exit(1)
    
    # Perform comprehensive analysis
    print("\n📈 Performing scalability analysis...")
    scalability = analyze_scalability(all_results, config)
    
    print("🏆 Generating performance comparisons...")
    comparison, method_stats = generate_performance_comparison(all_results)
    
    # Display results
    display_analysis_summary(all_results, grid_summaries, scalability, comparison, method_stats, config)
    
    # Save results
    print(f"\n💾 Saving master analysis results...")
    results_file = save_master_analysis(all_results, grid_summaries, scalability, comparison, method_stats, config)
    
    print(f"\n📄 Master analysis results saved to: {results_file}")
    
    # Final summary
    print(f"\n🎉 MASTER ANALYSIS COMPLETE!")
    print("=" * 40)
    print(f"✅ Grids analyzed: {successful_analyses}/{total_grids}")
    print(f"📊 Total method results: {sum(len(results) for results in all_results.values())}")
    print(f"💾 Results saved to: {results_file.name}")
    
    if successful_analyses < total_grids:
        failed_grids = [grid for grid, success in analysis_results.items() if not success]
        print(f"\n⚠️  Failed grid analyses: {', '.join(failed_grids)}")
        print("   Check individual grid logs for details.")

if __name__ == "__main__":
    main()