#!/usr/bin/env python3
"""
Single test case validation script for Tree Method implementation.

This script runs a single test case through our system and compares the results
with the original research data to establish baseline accuracy.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_simulation(test_case_path: str, traffic_control: str, end_time: int = 7300) -> Dict:
    """
    Run a single simulation and capture results.
    
    Args:
        test_case_path: Path to test case directory
        traffic_control: Traffic control method (tree_method, actuated, fixed)
        end_time: Simulation duration in seconds (default 7300 = ~2 hours like original)
    
    Returns:
        Dictionary with simulation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Build command - use virtual environment python
    venv_python = Path(__file__).parent.parent.parent / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    
    cmd = [
        python_cmd, "-m", "src.cli",
        "--tree_method_sample", test_case_path,
        "--traffic_control", traffic_control,
        "--end-time", str(end_time)
    ]
    
    logger.info(f"Running simulation: {' '.join(cmd)}")
    
    # Run simulation and capture output
    start_time = time.time()
    
    try:
        # Set up environment - inherit parent environment and add required variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
            cwd=Path(__file__).parent.parent.parent  # Run from project root
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"Simulation failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return {"error": "simulation_failed", "stderr": result.stderr}
        
        # Extract metrics from output
        metrics = parse_simulation_output(result.stdout)
        metrics["execution_time_seconds"] = execution_time
        metrics["traffic_control"] = traffic_control
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"Simulation timed out after 300 seconds")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        return {"error": str(e)}

def parse_simulation_output(output: str) -> Dict:
    """
    Parse simulation output to extract key metrics.
    
    Args:
        output: Raw simulation output
        
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}
    
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        
        # Parse SUMO statistics (from SUMO output)
        if "Statistics (avg of" in line and "):" in line:
            # Extract vehicle count from "Statistics (avg of 342):"
            try:
                vehicles = int(line.split("Statistics (avg of ")[1].split(")")[0])
                metrics["completed_vehicles"] = vehicles
            except (ValueError, IndexError):
                pass
        elif "RouteLength:" in line:
            try:
                metrics["avg_route_length"] = float(line.split("RouteLength:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Duration:" in line and "RouteLength:" not in line:
            try:
                metrics["avg_travel_time"] = float(line.split("Duration:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Speed:" in line:
            try:
                metrics["avg_speed"] = float(line.split("Speed:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "WaitingTime:" in line:
            try:
                metrics["avg_waiting_time"] = float(line.split("WaitingTime:")[1].strip())
            except (ValueError, IndexError):
                pass
        
        # Parse our pipeline output metrics
        elif "total_simulation_steps:" in line:
            try:
                metrics["total_steps"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "total_vehicles:" in line:
            try:
                metrics["total_vehicles"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "completion_rate:" in line:
            try:
                rate_str = line.split(":")[1].strip().replace("%", "")
                metrics["completion_rate"] = float(rate_str)
            except (ValueError, IndexError):
                pass
        
        # Parse vehicle counts from SUMO output
        elif "Inserted:" in line and "Loaded:" in line:
            try:
                # "Inserted: 885 (Loaded: 1239)"
                inserted = int(line.split("Inserted:")[1].split("(")[0].strip())
                loaded = int(line.split("Loaded:")[1].split(")")[0].strip())
                metrics["vehicles_inserted"] = inserted
                metrics["vehicles_loaded"] = loaded
            except (ValueError, IndexError):
                pass
    
    return metrics

def load_original_results(test_case_path: str, algorithm: str) -> Dict:
    """
    Load original research results for comparison.
    
    Args:
        test_case_path: Path to test case directory
        algorithm: Algorithm name (tree_method, actuated)
        
    Returns:
        Dictionary with original metrics
    """
    original_dir = Path(test_case_path) / "original_results"
    
    if algorithm == "tree_method":
        stats_file = original_dir / "tree_method_vehicle_stats.txt"
        times_file = original_dir / "tree_method_driving_times.txt"
    elif algorithm == "actuated":
        stats_file = original_dir / "actuated_vehicle_stats.txt"
        times_file = original_dir / "actuated_driving_times.txt"  # May not exist
    else:
        return {"error": f"Unknown algorithm: {algorithm}"}
    
    results = {}
    
    # Parse vehicle stats file
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Parse format: step, vehicles_started, vehicles_ended, cumulative
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 4:
                    results["original_total_vehicles"] = int(last_line[3].strip())
                    results["original_completed_vehicles"] = int(last_line[2].strip())
    
    # Parse driving times file
    if times_file.exists():
        with open(times_file, 'r') as f:
            times = []
            for line in f:
                time_str = line.strip().rstrip(',')
                if time_str:
                    try:
                        times.append(float(time_str))
                    except ValueError:
                        continue
            
            if times:
                results["original_avg_travel_time"] = sum(times) / len(times)
                results["original_travel_time_count"] = len(times)
    
    return results

def compare_results(our_results: Dict, original_results: Dict) -> Dict:
    """
    Compare our results with original research results.
    
    Args:
        our_results: Results from our simulation
        original_results: Original research results
        
    Returns:
        Comparison analysis
    """
    comparison = {
        "our_results": our_results,
        "original_results": original_results,
        "differences": {},
        "tolerances": {}
    }
    
    # Compare travel times
    if "avg_travel_time" in our_results and "original_avg_travel_time" in original_results:
        our_time = our_results["avg_travel_time"]
        orig_time = original_results["original_avg_travel_time"]
        diff_percent = ((our_time - orig_time) / orig_time) * 100
        
        comparison["differences"]["travel_time_diff_percent"] = diff_percent
        comparison["tolerances"]["travel_time_acceptable"] = abs(diff_percent) < 10  # 10% tolerance
    
    # Compare completion rates
    if "completed_vehicles" in our_results and "original_completed_vehicles" in original_results:
        our_completed = our_results["completed_vehicles"]
        orig_completed = original_results["original_completed_vehicles"]
        
        comparison["differences"]["completed_vehicles_diff"] = our_completed - orig_completed
        comparison["tolerances"]["completion_acceptable"] = abs(our_completed - orig_completed) < 50  # 50 vehicle tolerance
    
    return comparison

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate single test case against original results")
    parser.add_argument("test_case", help="Path to test case directory")
    parser.add_argument("--traffic_control", default="tree_method", 
                       choices=["tree_method", "actuated", "fixed"],
                       help="Traffic control method to test")
    parser.add_argument("--end_time", type=int, default=7300,
                       help="Simulation duration in seconds (default: 7300)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Starting validation for: {args.test_case}")
    logger.info(f"Traffic control: {args.traffic_control}")
    
    # Run our simulation
    logger.info("Running our simulation...")
    our_results = run_simulation(args.test_case, args.traffic_control, args.end_time)
    
    if "error" in our_results:
        logger.error(f"Simulation failed: {our_results['error']}")
        return 1
    
    # Load original results
    logger.info("Loading original results...")
    original_results = load_original_results(args.test_case, args.traffic_control)
    
    # Compare results
    logger.info("Comparing results...")
    comparison = compare_results(our_results, original_results)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    else:
        print(json.dumps(comparison, indent=2))
    
    # Summary
    logger.info("=== VALIDATION SUMMARY ===")
    if "travel_time_diff_percent" in comparison["differences"]:
        diff = comparison["differences"]["travel_time_diff_percent"]
        acceptable = comparison["tolerances"]["travel_time_acceptable"]
        logger.info(f"Travel time difference: {diff:.2f}% ({'ACCEPTABLE' if acceptable else 'SIGNIFICANT'})")
    
    if "completed_vehicles_diff" in comparison["differences"]:
        diff = comparison["differences"]["completed_vehicles_diff"]
        acceptable = comparison["tolerances"]["completion_acceptable"]
        logger.info(f"Completed vehicles difference: {diff} ({'ACCEPTABLE' if acceptable else 'SIGNIFICANT'})")
    
    return 0

if __name__ == "__main__":
    exit(main())