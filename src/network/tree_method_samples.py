"""
Tree Method sample network management.

This module handles the setup and preparation of pre-built Tree Method sample
networks for bypass mode testing. It manages file copying, path updates, and
configuration adaptation to integrate sample networks with our pipeline.
"""

import logging
import shutil
from pathlib import Path

from src.config import CONFIG
from src.sumo_integration.sumo_utils import update_sumo_config_paths, override_end_time_from_config


def setup_tree_method_samples(args, sample_folder: str) -> None:
    """Copy and adapt Tree Method sample files for our pipeline.
    
    Args:
        args: Parsed command line arguments from argparse
        sample_folder: Path to the Tree Method sample folder
        
    Raises:
        ValueError: If sample folder or required files are missing
        FileNotFoundError: If required sample files cannot be found
        PermissionError: If files cannot be copied due to permissions
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate sample folder exists
        sample_path = Path(sample_folder)
        if not sample_path.exists():
            raise ValueError(f"Sample folder not found: {sample_folder}")
        
        # Required files in sample folder
        required_files = {
            'network.net.xml': CONFIG.network_file,           # -> workspace/grid.net.xml
            'vehicles.trips.xml': CONFIG.routes_file,         # -> workspace/vehicles.rou.xml  
            'simulation.sumocfg.xml': CONFIG.config_file      # -> workspace/grid.sumocfg
        }
        
        # Copy and rename files to our convention
        for source_name, target_path in required_files.items():
            source_file = sample_path / source_name
            if not source_file.exists():
                raise ValueError(f"Required file missing: {source_file}")
            
            shutil.copy2(source_file, target_path)
            logger.info(f"Copied {source_name} -> {target_path}")
        
        # Update SUMO config file to use our file naming convention
        update_sumo_config_paths()
        
        # Extract and override end time from SUMO config
        override_end_time_from_config(args)
        
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.error(f"Error setting up Tree Method sample: {e}")
        raise