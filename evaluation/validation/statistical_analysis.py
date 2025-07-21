#!/usr/bin/env python3
"""
Statistical analysis and comparison script for validation results.

This script performs comprehensive statistical analysis comparing our Tree Method
implementation results with the original research results.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class ValidationAnalyzer:
    """Analyzer for validation results and statistical comparisons."""
    
    def __init__(self, validation_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            validation_dir: Path to validation directory
        """
        self.validation_dir = validation_dir
        self.results_dir = validation_dir / "results"
        self.baselines_dir = validation_dir / "baselines"
        self.logger = logging.getLogger(__name__)
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
    
    def load_validation_results(self, pattern: str = "*.json") -> List[Dict]:
        """
        Load all validation result files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of validation result dictionaries
        """
        results = []
        
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = result_file.name
                    results.append(data)
            except Exception as e:
                self.logger.warning(f"Error loading {result_file}: {e}")
        
        return results
    
    def extract_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        Extract metrics into a pandas DataFrame for analysis.
        
        Args:
            results: List of validation results
            
        Returns:
            DataFrame with extracted metrics
        """
        rows = []
        
        for result in results:
            try:
                row = {
                    'source_file': result.get('source_file', 'unknown'),
                    'traffic_control': result.get('our_results', {}).get('traffic_control', 'unknown')
                }
                
                # Our results
                our_results = result.get('our_results', {})
                row.update({
                    'our_avg_travel_time': our_results.get('avg_travel_time'),
                    'our_completed_vehicles': our_results.get('completed_vehicles'),
                    'our_total_vehicles': our_results.get('total_vehicles'),
                    'our_completion_rate': our_results.get('completion_rate'),
                    'execution_time': our_results.get('execution_time_seconds')
                })
                
                # Original results
                original_results = result.get('original_results', {})
                row.update({
                    'orig_avg_travel_time': original_results.get('original_avg_travel_time'),
                    'orig_completed_vehicles': original_results.get('original_completed_vehicles'),
                    'orig_total_vehicles': original_results.get('original_total_vehicles'),
                    'orig_travel_time_count': original_results.get('original_travel_time_count')
                })
                
                # Differences
                differences = result.get('differences', {})
                row.update({
                    'travel_time_diff_percent': differences.get('travel_time_diff_percent'),
                    'completed_vehicles_diff': differences.get('completed_vehicles_diff')
                })
                
                rows.append(row)
                
            except Exception as e:
                self.logger.warning(f"Error extracting metrics from result: {e}")
        
        return pd.DataFrame(rows)
    
    def statistical_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate statistical summary of validation results.
        
        Args:
            df: DataFrame with validation metrics
            
        Returns:
            Dictionary with statistical summary
        """
        summary = {}
        
        # Group by traffic control method
        for method in df['traffic_control'].unique():
            if pd.isna(method):
                continue
                
            method_data = df[df['traffic_control'] == method]
            method_summary = {}
            
            # Travel time analysis
            travel_time_diffs = method_data['travel_time_diff_percent'].dropna()
            if len(travel_time_diffs) > 0:
                method_summary['travel_time_differences'] = {
                    'count': len(travel_time_diffs),
                    'mean': float(travel_time_diffs.mean()),
                    'std': float(travel_time_diffs.std()),
                    'min': float(travel_time_diffs.min()),
                    'max': float(travel_time_diffs.max()),
                    'median': float(travel_time_diffs.median()),
                    'within_5_percent': int((abs(travel_time_diffs) <= 5).sum()),
                    'within_10_percent': int((abs(travel_time_diffs) <= 10).sum())
                }
            
            # Completion rate analysis
            completion_diffs = method_data['completed_vehicles_diff'].dropna()
            if len(completion_diffs) > 0:
                method_summary['completion_differences'] = {
                    'count': len(completion_diffs),
                    'mean': float(completion_diffs.mean()),
                    'std': float(completion_diffs.std()),
                    'min': float(completion_diffs.min()),
                    'max': float(completion_diffs.max()),
                    'median': float(completion_diffs.median()),
                    'within_25_vehicles': int((abs(completion_diffs) <= 25).sum()),
                    'within_50_vehicles': int((abs(completion_diffs) <= 50).sum())
                }
            
            # Execution time analysis
            exec_times = method_data['execution_time'].dropna()
            if len(exec_times) > 0:
                method_summary['execution_times'] = {
                    'count': len(exec_times),
                    'mean': float(exec_times.mean()),
                    'std': float(exec_times.std()),
                    'min': float(exec_times.min()),
                    'max': float(exec_times.max()),
                    'median': float(exec_times.median())
                }
            
            summary[method] = method_summary
        
        return summary
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between our results and original results.
        
        Args:
            df: DataFrame with validation metrics
            
        Returns:
            Dictionary with correlation analysis
        """
        correlations = {}
        
        # Travel time correlation
        our_times = df['our_avg_travel_time'].dropna()
        orig_times = df['orig_avg_travel_time'].dropna()
        
        # Align data for correlation
        common_indices = df.dropna(subset=['our_avg_travel_time', 'orig_avg_travel_time']).index
        if len(common_indices) > 1:
            our_aligned = df.loc[common_indices, 'our_avg_travel_time']
            orig_aligned = df.loc[common_indices, 'orig_avg_travel_time']
            
            correlation, p_value = stats.pearsonr(our_aligned, orig_aligned)
            correlations['travel_time'] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'sample_size': len(common_indices),
                'significant': p_value < 0.05
            }
        
        # Completion rate correlation
        common_indices = df.dropna(subset=['our_completed_vehicles', 'orig_completed_vehicles']).index
        if len(common_indices) > 1:
            our_aligned = df.loc[common_indices, 'our_completed_vehicles']
            orig_aligned = df.loc[common_indices, 'orig_completed_vehicles']
            
            correlation, p_value = stats.pearsonr(our_aligned, orig_aligned)
            correlations['completion_rate'] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'sample_size': len(common_indices),
                'significant': p_value < 0.05
            }
        
        return correlations
    
    def method_comparison_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Compare performance across different traffic control methods.
        
        Args:
            df: DataFrame with validation metrics
            
        Returns:
            Dictionary with method comparison analysis
        """
        comparison = {}
        
        methods = df['traffic_control'].unique()
        methods = [m for m in methods if not pd.isna(m)]
        
        if len(methods) > 1:
            # Travel time comparison
            for metric in ['our_avg_travel_time', 'travel_time_diff_percent']:
                if metric in df.columns:
                    method_groups = []
                    method_names = []
                    
                    for method in methods:
                        values = df[df['traffic_control'] == method][metric].dropna()
                        if len(values) > 0:
                            method_groups.append(values)
                            method_names.append(method)
                    
                    if len(method_groups) > 1:
                        # Perform ANOVA
                        f_stat, p_value = stats.f_oneway(*method_groups)
                        comparison[metric] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'methods_compared': method_names,
                            'group_means': [float(group.mean()) for group in method_groups]
                        }
        
        return comparison
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Generate visualization plots for validation results.
        
        Args:
            df: DataFrame with validation metrics
            output_dir: Directory to save plots
        """
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Travel time differences by method
        if 'travel_time_diff_percent' in df.columns:
            plt.figure(figsize=(10, 6))
            df_clean = df.dropna(subset=['travel_time_diff_percent', 'traffic_control'])
            if len(df_clean) > 0:
                sns.boxplot(data=df_clean, x='traffic_control', y='travel_time_diff_percent')
                plt.title('Travel Time Difference by Traffic Control Method')
                plt.ylabel('Difference from Original (%)')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'travel_time_differences.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Correlation plot for travel times
        if 'our_avg_travel_time' in df.columns and 'orig_avg_travel_time' in df.columns:
            plt.figure(figsize=(8, 8))
            df_clean = df.dropna(subset=['our_avg_travel_time', 'orig_avg_travel_time'])
            if len(df_clean) > 0:
                plt.scatter(df_clean['orig_avg_travel_time'], df_clean['our_avg_travel_time'], 
                           alpha=0.6, s=50)
                
                # Add diagonal line
                min_val = min(df_clean['orig_avg_travel_time'].min(), df_clean['our_avg_travel_time'].min())
                max_val = max(df_clean['orig_avg_travel_time'].max(), df_clean['our_avg_travel_time'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Agreement')
                
                plt.xlabel('Original Average Travel Time')
                plt.ylabel('Our Average Travel Time')
                plt.title('Travel Time Correlation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'travel_time_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Execution time distribution
        if 'execution_time' in df.columns:
            plt.figure(figsize=(10, 6))
            df_clean = df.dropna(subset=['execution_time'])
            if len(df_clean) > 0:
                plt.hist(df_clean['execution_time'], bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Execution Time (seconds)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Simulation Execution Times')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive validation report.
        
        Args:
            df: DataFrame with validation metrics
            
        Returns:
            Complete validation report
        """
        report = {
            'metadata': {
                'total_validations': len(df),
                'methods_tested': list(df['traffic_control'].unique()),
                'successful_validations': len(df.dropna(subset=['our_avg_travel_time'])),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'statistical_summary': self.statistical_summary(df),
            'correlation_analysis': self.correlation_analysis(df),
            'method_comparison': self.method_comparison_analysis(df)
        }
        
        # Generate visualizations
        viz_dir = self.results_dir / "visualizations"
        self.generate_visualizations(df, viz_dir)
        report['visualizations_path'] = str(viz_dir)
        
        return report

def main():
    """Main analysis function."""
    logger = setup_logging()
    
    # Get validation directory
    project_root = Path(__file__).parent.parent.parent
    validation_dir = project_root / "evaluation" / "validation"
    
    if not validation_dir.exists():
        logger.error(f"Validation directory not found: {validation_dir}")
        return 1
    
    # Initialize analyzer
    analyzer = ValidationAnalyzer(validation_dir)
    
    # Load validation results
    logger.info("Loading validation results...")
    results = analyzer.load_validation_results()
    
    if not results:
        logger.warning("No validation results found")
        return 1
    
    logger.info(f"Loaded {len(results)} validation results")
    
    # Extract metrics
    df = analyzer.extract_metrics(results)
    logger.info(f"Extracted metrics for {len(df)} validations")
    
    # Generate comprehensive report
    logger.info("Generating validation report...")
    report = analyzer.generate_report(df)
    
    # Save report
    report_file = validation_dir / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report saved to: {report_file}")
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    metadata = report['metadata']
    print(f"Total validations: {metadata['total_validations']}")
    print(f"Successful validations: {metadata['successful_validations']}")
    print(f"Methods tested: {', '.join(str(m) for m in metadata['methods_tested'] if m)}")
    
    # Print key statistics
    for method, stats in report['statistical_summary'].items():
        if 'travel_time_differences' in stats:
            tt_stats = stats['travel_time_differences']
            print(f"\n{method.upper()} Travel Time Differences:")
            print(f"  Mean difference: {tt_stats['mean']:.2f}%")
            print(f"  Within 5%: {tt_stats['within_5_percent']}/{tt_stats['count']}")
            print(f"  Within 10%: {tt_stats['within_10_percent']}/{tt_stats['count']}")
    
    return 0

if __name__ == "__main__":
    exit(main())