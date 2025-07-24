#!/usr/bin/env python3
"""
Systematic validation script for all available test cases.

This script runs validation across all available test cases in the dataset,
comparing our implementation with original research results.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_available_test_cases(base_dir: Path) -> List[Path]:
    """
    Find all available test cases in the dataset.
    
    Args:
        base_dir: Base dataset directory
        
    Returns:
        List of test case paths
    """
    test_cases = []
    
    # Look for experiment directories
    for experiment_dir in base_dir.glob("Experiment*"):
        if experiment_dir.is_dir():
            # Look for numbered test case directories
            for case_dir in experiment_dir.glob("[0-9]*"):
                if case_dir.is_dir():
                    # Check if required files exist
                    required_files = [
                        "network.net.xml",
                        "vehicles.trips.xml", 
                        "simulation.sumocfg.xml"
                    ]
                    
                    if all((case_dir / f).exists() for f in required_files):
                        test_cases.append(case_dir)
    
    return sorted(test_cases)

def check_original_results(test_case_path: Path) -> bool:
    """
    Check if original results exist for a test case.
    
    Args:
        test_case_path: Path to test case directory
        
    Returns:
        True if original results exist
    """
    original_dir = test_case_path / "original_results"
    
    # Check for key original result files
    key_files = [
        "tree_method_vehicle_stats.txt",
        "tree_method_driving_times.txt"
    ]
    
    return all((original_dir / f).exists() for f in key_files)

def run_validation_case(test_case_path: Path, traffic_control: str, 
                       end_time: int, results_dir: Path) -> Dict:
    """
    Run validation for a single test case.
    
    Args:
        test_case_path: Path to test case
        traffic_control: Traffic control method
        end_time: Simulation duration
        results_dir: Directory to save results
        
    Returns:
        Validation result summary
    """
    logger = logging.getLogger(__name__)
    
    # Generate output filename
    experiment_name = test_case_path.parent.name
    case_name = test_case_path.name
    output_file = results_dir / f"{experiment_name}_{case_name}_{traffic_control}.json"
    
    # Build validation command
    validation_script = Path(__file__).parent / "run_single_validation.py"
    # Use virtual environment python
    venv_python = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    
    cmd = [
        python_cmd, str(validation_script),
        str(test_case_path),
        "--traffic_control", traffic_control,
        "--end_time", str(end_time),
        "--output", str(output_file)
    ]
    
    logger.info(f"Validating {experiment_name}/{case_name} with {traffic_control}")
    
    try:
        # Set up environment - inherit parent environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Run validation with timeout
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=7500,  # 2.1 hour timeout for 7300s simulation + overhead
            env=env
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Success: {experiment_name}/{case_name}")
            return {
                "status": "success",
                "experiment": experiment_name,
                "case": case_name,
                "traffic_control": traffic_control,
                "output_file": str(output_file)
            }
        else:
            logger.warning(f"✗ Failed: {experiment_name}/{case_name} - {result.stderr[:200]}")
            return {
                "status": "failed",
                "experiment": experiment_name, 
                "case": case_name,
                "traffic_control": traffic_control,
                "error": result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        logger.warning(f"⏱ Timeout: {experiment_name}/{case_name}")
        return {
            "status": "timeout",
            "experiment": experiment_name,
            "case": case_name,
            "traffic_control": traffic_control
        }
    except Exception as e:
        logger.error(f"💥 Error: {experiment_name}/{case_name} - {e}")
        return {
            "status": "error",
            "experiment": experiment_name,
            "case": case_name, 
            "traffic_control": traffic_control,
            "error": str(e)
        }

def generate_summary_report(results: List[Dict], output_path: Path) -> None:
    """
    Generate summary report of validation results.
    
    Args:
        results: List of validation results
        output_path: Path to save summary report
    """
    summary = {
        "total_validations": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "timeouts": len([r for r in results if r["status"] == "timeout"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "by_experiment": {},
        "by_traffic_control": {},
        "details": results
    }
    
    # Group by experiment
    for result in results:
        exp = result["experiment"]
        if exp not in summary["by_experiment"]:
            summary["by_experiment"][exp] = {"total": 0, "success": 0, "failed": 0}
        
        summary["by_experiment"][exp]["total"] += 1
        if result["status"] == "success":
            summary["by_experiment"][exp]["success"] += 1
        else:
            summary["by_experiment"][exp]["failed"] += 1
    
    # Group by traffic control
    for result in results:
        tc = result["traffic_control"] 
        if tc not in summary["by_traffic_control"]:
            summary["by_traffic_control"][tc] = {"total": 0, "success": 0, "failed": 0}
            
        summary["by_traffic_control"][tc]["total"] += 1
        if result["status"] == "success":
            summary["by_traffic_control"][tc]["success"] += 1
        else:
            summary["by_traffic_control"][tc]["failed"] += 1
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    """Main systematic validation function."""
    parser = argparse.ArgumentParser(description="Run systematic validation across all test cases")
    parser.add_argument("--experiments", nargs="+", 
                       default=["Experiment1-realistic-high-load"],
                       help="Experiments to validate")
    parser.add_argument("--traffic-control", nargs="+",
                       default=["tree_method"],
                       choices=["tree_method", "actuated", "fixed"],
                       help="Traffic control methods to test")
    parser.add_argument("--end-time", type=int, default=7300,
                       help="Simulation duration in seconds (default: 7300 to match original experiments)")
    parser.add_argument("--max-cases", type=int, default=10,
                       help="Maximum test cases per experiment (default: 10)")
    parser.add_argument("--require-original-results", action="store_true",
                       help="Only run validation for cases with original results")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Get project paths
    project_root = Path(__file__).parent.parent.parent
    datasets_dir = project_root / "evaluation" / "datasets" / "decentralized_traffic_bottleneck"
    validation_dir = project_root / "evaluation" / "validation"
    results_dir = validation_dir / "results"
    
    # Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting systematic validation...")
    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Traffic control methods: {args.traffic_control}")
    logger.info(f"Max cases per experiment: {args.max_cases}")
    
    # Find available test cases
    all_test_cases = find_available_test_cases(datasets_dir)
    logger.info(f"Found {len(all_test_cases)} total test cases")
    
    # Filter by requested experiments
    filtered_cases = []
    for test_case in all_test_cases:
        experiment_name = test_case.parent.name
        if experiment_name in args.experiments:
            # Check original results requirement
            if args.require_original_results and not check_original_results(test_case):
                logger.debug(f"Skipping {test_case} - no original results")
                continue
            filtered_cases.append(test_case)
    
    logger.info(f"Selected {len(filtered_cases)} test cases for validation")
    
    # Limit cases per experiment
    cases_by_experiment = {}
    for case in filtered_cases:
        exp = case.parent.name
        if exp not in cases_by_experiment:
            cases_by_experiment[exp] = []
        if len(cases_by_experiment[exp]) < args.max_cases:
            cases_by_experiment[exp].append(case)
    
    final_cases = []
    for exp_cases in cases_by_experiment.values():
        final_cases.extend(exp_cases)
    
    logger.info(f"Running validation on {len(final_cases)} test cases")
    
    # Run validations
    all_results = []
    start_time = time.time()
    
    for i, test_case in enumerate(final_cases):
        for traffic_control in args.traffic_control:
            logger.info(f"Progress: {i+1}/{len(final_cases)} cases, method: {traffic_control}")
            
            result = run_validation_case(
                test_case, traffic_control, args.end_time, results_dir
            )
            all_results.append(result)
            
            # Small delay between validations
            time.sleep(1)
    
    total_time = time.time() - start_time
    
    # Generate summary report
    summary_file = validation_dir / "systematic_validation_summary.json"
    generate_summary_report(all_results, summary_file)
    
    logger.info(f"Validation completed in {total_time:.1f} seconds")
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary
    successful = len([r for r in all_results if r["status"] == "success"])
    total = len(all_results)
    
    print(f"\n=== SYSTEMATIC VALIDATION SUMMARY ===")
    print(f"Total validations: {total}")
    print(f"Successful: {successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per validation: {total_time/total:.1f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())