#!/usr/bin/env python3
"""
Download all original research results for comparison.

This script systematically downloads the original algorithm results
from all 80 test cases for comprehensive validation.
"""

import logging
import requests
import time
from pathlib import Path
from typing import List

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_file(url: str, output_path: Path, retries: int = 3) -> bool:
    """
    Download a file with retries.
    
    Args:
        url: URL to download
        output_path: Local path to save file
        retries: Number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(response.text)
                return True
            elif response.status_code == 404:
                logger.warning(f"File not found (404): {url}")
                return False
            else:
                logger.warning(f"HTTP {response.status_code} for {url} (attempt {attempt + 1})")
                
        except Exception as e:
            logger.warning(f"Error downloading {url} (attempt {attempt + 1}): {e}")
            
        if attempt < retries - 1:
            time.sleep(1)  # Wait before retry
    
    logger.error(f"Failed to download after {retries} attempts: {url}")
    return False

def download_experiment_results(experiment: str, base_dir: Path) -> None:
    """
    Download results for all test cases in an experiment.
    
    Args:
        experiment: Experiment name (e.g., 'Experiment1-realistic-high-load')
        base_dir: Base directory for datasets
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading results for {experiment}...")
    
    base_url = "https://raw.githubusercontent.com/nimrodSerokTAU/decentralized-traffic-bottlenecks/main/data"
    
    # Files to download for each algorithm
    algorithm_files = {
        "tree_method": {
            "CurrentTreeDvd/driving_time_distribution.txt": "tree_method_driving_times.txt",
            "CurrentTreeDvd/vehicles_stats.txt": "tree_method_vehicle_stats.txt",
            "CurrentTreeDvd/tree_cost_distribution.txt": "tree_method_costs.txt"
        },
        "actuated": {
            "SUMOActuated/driving_time_distribution.txt": "actuated_driving_times.txt",
            "SUMOActuated/vehicles_stats.txt": "actuated_vehicle_stats.txt",
            "SUMOActuated/tree_cost_distribution.txt": "actuated_costs.txt"
        },
        "random": {
            "Random/driving_time_distribution.txt": "random_driving_times.txt",
            "Random/vehicles_stats.txt": "random_vehicle_stats.txt"
        },
        "uniform": {
            "Uniform/driving_time_distribution.txt": "uniform_driving_times.txt", 
            "Uniform/vehicles_stats.txt": "uniform_vehicle_stats.txt"
        }
    }
    
    successful_downloads = 0
    total_files = 0
    
    # Download results for test cases 1-20
    for case_num in range(1, 21):
        case_dir = base_dir / experiment / str(case_num) / "original_results"
        
        for algorithm, files in algorithm_files.items():
            for source_file, target_file in files.items():
                url = f"{base_url}/{experiment}/{case_num}/{source_file}"
                output_path = case_dir / target_file
                
                total_files += 1
                if download_file(url, output_path):
                    successful_downloads += 1
                
                # Small delay to be respectful to GitHub
                time.sleep(0.1)
    
    success_rate = (successful_downloads / total_files) * 100
    logger.info(f"Downloaded {successful_downloads}/{total_files} files ({success_rate:.1f}%) for {experiment}")

def download_aggregate_results(base_dir: Path) -> None:
    """
    Download aggregate results files.
    
    Args:
        base_dir: Base directory for datasets
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading aggregate results...")
    
    base_url = "https://raw.githubusercontent.com/nimrodSerokTAU/decentralized-traffic-bottlenecks/main/data"
    
    # Main results file
    results_url = f"{base_url}/results.xlsx"
    results_path = base_dir.parent / "validation" / "baselines" / "original_results.xlsx"
    
    if download_file(results_url, results_path):
        logger.info("Downloaded main results.xlsx")
    
    # Download original summary files for each experiment
    experiments = [
        "Experiment1-realistic-high-load",
        "Experiment2-rand-high-load", 
        "Experiment3-realistic-moderate-load",
        "Experiment4-and-moderate-load"
    ]
    
    for experiment in experiments:
        summary_files = [
            "res_ended_vehicles_count.txt",
            "res_time per v.txt"
        ]
        
        for summary_file in summary_files:
            url = f"{base_url}/{experiment}/{summary_file}"
            output_path = base_dir.parent / "validation" / "baselines" / f"{experiment}_{summary_file}"
            download_file(url, output_path)

def main():
    """Main download function."""
    logger = setup_logging()
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    datasets_dir = project_root / "evaluation" / "datasets" / "decentralized_traffic_bottleneck"
    
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return 1
    
    logger.info("Starting download of original research results...")
    
    # Create validation directories
    validation_dir = project_root / "evaluation" / "validation"
    (validation_dir / "baselines").mkdir(parents=True, exist_ok=True)
    
    # Download aggregate results
    download_aggregate_results(datasets_dir)
    
    # Download individual experiment results
    experiments = [
        "Experiment1-realistic-high-load",
        "Experiment2-rand-high-load",
        "Experiment3-realistic-moderate-load", 
        "Experiment4-and-moderate-load"
    ]
    
    for experiment in experiments:
        if (datasets_dir / experiment).exists():
            download_experiment_results(experiment, datasets_dir)
        else:
            logger.warning(f"Experiment directory not found: {experiment}")
    
    logger.info("Download completed!")
    return 0

if __name__ == "__main__":
    exit(main())