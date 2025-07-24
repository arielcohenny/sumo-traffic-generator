#!/usr/bin/env python3
"""
Complete validation pipeline runner.

This script runs the complete validation process:
1. Downloads original results (if needed)
2. Runs validation on multiple test cases
3. Generates statistical analysis and report
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_command(cmd: List[str], cwd: Path = None) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list
        cwd: Working directory
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=7500  # 2.1 hour timeout to accommodate 7300s simulation + overhead
        )
        
        if result.returncode == 0:
            logger.info("Command completed successfully")
            return True
        else:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Command timed out")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(description="Run complete validation pipeline")
    parser.add_argument("--download-results", action="store_true",
                       help="Download original results before validation")
    parser.add_argument("--test-cases", nargs="+", 
                       default=["evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1"],
                       help="Test cases to validate (paths)")
    parser.add_argument("--traffic-control", nargs="+",
                       default=["tree_method"],
                       choices=["tree_method", "actuated", "fixed"],
                       help="Traffic control methods to test")
    parser.add_argument("--end-time", type=int, default=7300,
                       help="Simulation duration in seconds (default: 7300 to match original experiments)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip statistical analysis")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    validation_dir = project_root / "evaluation" / "validation"
    results_dir = validation_dir / "results"
    
    # Ensure directories exist
    validation_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    logger.info("Starting complete validation pipeline...")
    
    # Step 1: Download original results if requested
    if args.download_results:
        logger.info("Downloading original research results...")
        download_script = validation_dir / "download_all_original_results.py"
        if not run_command([sys.executable, str(download_script)], project_root):
            logger.error("Failed to download original results")
            return 1
    
    # Step 2: Run validations
    logger.info(f"Running validations for {len(args.test_cases)} test cases...")
    
    validation_script = validation_dir / "run_single_validation.py"
    validation_count = 0
    
    for test_case in args.test_cases:
        for traffic_control in args.traffic_control:
            logger.info(f"Validating {test_case} with {traffic_control}...")
            
            # Generate output filename
            case_name = Path(test_case).name
            experiment_name = Path(test_case).parent.name
            output_file = results_dir / f"{experiment_name}_{case_name}_{traffic_control}.json"
            
            # Run validation
            cmd = [
                sys.executable, str(validation_script),
                test_case,
                "--traffic_control", traffic_control,
                "--end_time", str(args.end_time),
                "--output", str(output_file)
            ]
            
            if run_command(cmd, project_root):
                validation_count += 1
                logger.info(f"Validation completed: {output_file}")
            else:
                logger.error(f"Validation failed for {test_case} with {traffic_control}")
    
    logger.info(f"Completed {validation_count} validations")
    
    # Step 3: Statistical analysis
    if not args.skip_analysis and validation_count > 0:
        logger.info("Running statistical analysis...")
        analysis_script = validation_dir / "statistical_analysis.py"
        if not run_command([sys.executable, str(analysis_script)], project_root):
            logger.error("Statistical analysis failed")
            return 1
    
    logger.info("Validation pipeline completed successfully!")
    
    # Print summary
    print(f"\n=== VALIDATION PIPELINE SUMMARY ===")
    print(f"Test cases validated: {len(args.test_cases)}")
    print(f"Traffic control methods: {', '.join(args.traffic_control)}")
    print(f"Successful validations: {validation_count}")
    print(f"Results directory: {results_dir}")
    print(f"Analysis report: {validation_dir / 'validation_report.json'}")
    
    return 0

if __name__ == "__main__":
    exit(main())