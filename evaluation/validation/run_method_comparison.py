#!/usr/bin/env python3
"""
Method Comparison Validation Script

This script runs a comprehensive comparison of all traffic control methods
(Tree Method, Actuated, Fixed) against the same test cases and compares
the relative performance improvements to the original research claims.

Expected improvements from original research:
- Tree Method vs Fixed: 20-45% improvement
- Tree Method vs Actuated: 10-25% improvement
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_single_validation(test_case_path: str, traffic_control: str, end_time: int = 7300) -> Dict:
    """
    Run a single validation and return results.
    
    Args:
        test_case_path: Path to test case directory
        traffic_control: Traffic control method
        end_time: Simulation duration
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    
    # Use virtual environment python
    venv_python = Path(__file__).parent.parent.parent / "venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    
    # Build validation command
    validation_script = Path(__file__).parent / "run_single_validation.py"
    
    cmd = [
        python_cmd, str(validation_script),
        test_case_path,
        "--traffic_control", traffic_control,
        "--end_time", str(end_time)
    ]
    
    logger.info(f"Running {traffic_control}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=7500
        )
        
        if result.returncode == 0:
            # Parse JSON output from stdout - JSON is at the beginning
            try:
                output_text = result.stdout.strip()
                # Find the JSON part (starts with { and ends with })
                json_start = output_text.find('{')
                if json_start != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    json_end = json_start
                    for i, char in enumerate(output_text[json_start:], json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    json_text = output_text[json_start:json_end]
                    return json.loads(json_text)
                else:
                    logger.error(f"No JSON output found for {traffic_control}")
                    logger.debug(f"Full output: {result.stdout[:500]}")
                    return {"error": "no_json_output"}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for {traffic_control}: {e}")
                return {"error": "json_parse_error"}
        else:
            logger.error(f"Validation failed for {traffic_control}: {result.stderr}")
            return {"error": "validation_failed", "stderr": result.stderr}
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout for {traffic_control}")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Error running {traffic_control}: {e}")
        return {"error": str(e)}

def calculate_improvement(baseline_time: float, improved_time: float) -> float:
    """Calculate percentage improvement."""
    if baseline_time == 0:
        return 0.0
    return ((baseline_time - improved_time) / baseline_time) * 100

def analyze_method_comparison(results: Dict) -> Dict:
    """
    Analyze method comparison results and calculate improvements.
    
    Args:
        results: Dictionary with results for each method
        
    Returns:
        Analysis dictionary with improvement percentages
    """
    analysis = {
        "raw_results": results,
        "qa_validation": {},
        "improvements": {},
        "comparison_to_original": {},
        "summary": {}
    }
    
    # Extract travel times with QA logging
    travel_times = {}
    for method, result in results.items():
        if "our_results" in result and "avg_travel_time" in result["our_results"]:
            travel_times[method] = result["our_results"]["avg_travel_time"]
            
            # QA: Log detailed metrics for each method
            our_results = result["our_results"]
            print(f"\nQA: {method.upper()} DETAILED METRICS:")
            print(f"  Vehicles inserted: {our_results.get('vehicles_inserted', 'N/A')}")
            print(f"  Vehicles completed: {our_results.get('completed_vehicles', 'N/A')}")
            print(f"  Average travel time: {our_results.get('avg_travel_time', 'N/A')}s")
            print(f"  Average speed: {our_results.get('avg_speed', 'N/A')}")
            print(f"  Average waiting time: {our_results.get('avg_waiting_time', 'N/A')}s")
        else:
            travel_times[method] = None
    
    # QA: Check for identical results (potential bug)
    valid_times = [t for t in travel_times.values() if t is not None]
    unique_times = set(valid_times)
    analysis["qa_validation"]["unique_travel_times"] = len(unique_times)
    analysis["qa_validation"]["identical_results_detected"] = len(unique_times) < len(valid_times) and len(valid_times) > 1
    
    if len(unique_times) < len(valid_times) and len(valid_times) > 1:
        print(f"\n⚠️  QA WARNING: Only {len(unique_times)} unique travel times out of {len(valid_times)} methods!")
        print("This suggests some methods are producing identical results.")
        for method, time in travel_times.items():
            if time is not None:
                print(f"  {method}: {time}s")
    
    # Calculate improvements
    if travel_times.get("tree_method") and travel_times.get("actuated"):
        tree_vs_actuated = calculate_improvement(
            travel_times["actuated"], 
            travel_times["tree_method"]
        )
        analysis["improvements"]["tree_vs_actuated"] = tree_vs_actuated
    
    if travel_times.get("tree_method") and travel_times.get("fixed"):
        tree_vs_fixed = calculate_improvement(
            travel_times["fixed"],
            travel_times["tree_method"]
        )
        analysis["improvements"]["tree_vs_fixed"] = tree_vs_fixed
    
    # Compare to original research claims
    original_claims = {
        "tree_vs_actuated": {"min": 10, "max": 25},
        "tree_vs_fixed": {"min": 20, "max": 45}
    }
    
    for comparison, improvement in analysis["improvements"].items():
        if comparison in original_claims:
            claim = original_claims[comparison]
            within_range = claim["min"] <= improvement <= claim["max"]
            analysis["comparison_to_original"][comparison] = {
                "our_improvement": improvement,
                "original_claim_min": claim["min"],
                "original_claim_max": claim["max"],
                "within_expected_range": within_range,
                "status": "MATCHES" if within_range else "DIFFERS"
            }
    
    # Summary
    analysis["summary"] = {
        "travel_times": travel_times,
        "total_methods_tested": len([t for t in travel_times.values() if t is not None]),
        "all_methods_successful": all(t is not None for t in travel_times.values()),
        "tree_method_shows_improvement": all(
            imp > 0 for imp in analysis["improvements"].values()
        ) if analysis["improvements"] else False
    }
    
    return analysis

def main():
    """Main method comparison function."""
    parser = argparse.ArgumentParser(description="Run comprehensive method comparison validation")
    parser.add_argument("--test-cases", nargs="+",
                       default=["evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1"],
                       help="Test cases to validate")
    parser.add_argument("--methods", nargs="+",
                       default=["tree_method", "actuated", "fixed"],
                       choices=["tree_method", "actuated", "fixed"],
                       help="Traffic control methods to compare")
    parser.add_argument("--end-time", type=int, default=7300,
                       help="Simulation duration in seconds")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent / "method_comparison_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting method comparison validation...")
    logger.info(f"Test cases: {len(args.test_cases)}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Simulation duration: {args.end_time}s")
    
    all_results = {}
    
    for test_case in args.test_cases:
        logger.info(f"\n=== Testing {test_case} ===")
        
        case_results = {}
        case_name = Path(test_case).name
        experiment_name = Path(test_case).parent.name
        
        # Run all methods for this test case
        for method in args.methods:
            logger.info(f"Running method: {method}")
            start_time = time.time()
            
            result = run_single_validation(test_case, method, args.end_time)
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            case_results[method] = result
            
            if "error" not in result:
                travel_time = result.get("our_results", {}).get("avg_travel_time", "N/A")
                logger.info(f"✓ {method}: {travel_time}s average travel time")
            else:
                logger.error(f"✗ {method}: {result['error']}")
        
        # Analyze results for this test case
        analysis = analyze_method_comparison(case_results)
        
        # Save individual test case results
        case_output_file = args.output_dir / f"{experiment_name}_{case_name}_comparison.json"
        with open(case_output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        all_results[f"{experiment_name}_{case_name}"] = analysis
        
        # Print summary for this test case
        logger.info(f"\n=== Results for {test_case} ===")
        if analysis["summary"]["all_methods_successful"]:
            for method, time_val in analysis["summary"]["travel_times"].items():
                logger.info(f"{method}: {time_val:.2f}s")
            
            for comparison, data in analysis["comparison_to_original"].items():
                improvement = data["our_improvement"]
                status = data["status"]
                expected_range = f"{data['original_claim_min']}-{data['original_claim_max']}%"
                logger.info(f"{comparison}: {improvement:.1f}% improvement ({status}, expected: {expected_range})")
        else:
            logger.warning("Not all methods completed successfully")
    
    # Save overall results
    overall_output_file = args.output_dir / "overall_comparison.json"
    with open(overall_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    successful_cases = 0
    matching_performance = 0
    
    for case_name, analysis in all_results.items():
        print(f"\nTest Case: {case_name}")
        if analysis["summary"]["all_methods_successful"]:
            successful_cases += 1
            print("✓ All methods completed successfully")
            
            for comparison, data in analysis["comparison_to_original"].items():
                improvement = data["our_improvement"]
                status = data["status"]
                expected = f"{data['original_claim_min']}-{data['original_claim_max']}%"
                print(f"  {comparison}: {improvement:.1f}% ({status}, expected: {expected})")
                
                if status == "MATCHES":
                    matching_performance += 1
        else:
            print("✗ Some methods failed")
    
    print(f"\nOVERALL VALIDATION:")
    print(f"Successful test cases: {successful_cases}/{len(all_results)}")
    print(f"Performance matching original claims: {matching_performance}/{len(all_results) * 2}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())