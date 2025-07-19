"""
Tree Method sample pipeline for bypass mode.

This pipeline skips steps 1-8 and goes directly to simulation using
pre-built Tree Method sample networks.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .base_pipeline import BasePipeline
from src.orchestration.simulator import TrafficSimulator
from src.orchestration.traffic_controller import TrafficControllerFactory
from src.validate.errors import ValidationError
from src.config import CONFIG


class SamplePipeline(BasePipeline):
    """Pipeline for Tree Method sample testing (bypass mode)."""
    
    def __init__(self, args):
        super().__init__(args)
        self.sample_folder = args.tree_method_sample
    
    def execute(self) -> None:
        """Execute sample pipeline (bypass Steps 1-8, go to Step 9)."""
        self._validate_output_directory()
        
        self.logger.info(f"Tree Method Sample Mode: Using pre-built network from {self.sample_folder}")
        self.logger.info("Skipping Steps 1-8, going directly to Step 9 (Dynamic Simulation)")
        
        # Setup sample files
        self._setup_sample_files()
        
        # Step 9: Dynamic Simulation
        self._log_step(9, "Dynamic Simulation (Tree Method Sample)")
        self._execute_simulation()
    
    def _setup_sample_files(self) -> None:
        """Copy and adapt Tree Method sample files for our pipeline."""
        try:
            # Validate sample folder exists
            sample_path = Path(self.sample_folder)
            if not sample_path.exists():
                raise ValueError(f"Sample folder not found: {self.sample_folder}")
            
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
                self.logger.info(f"Copied {source_name} -> {target_path}")
            
            # Update SUMO config file to use our file naming convention
            self._update_sumo_config_paths()
            
        except (FileNotFoundError, PermissionError, ValueError) as e:
            self.logger.error(f"Error setting up Tree Method sample: {e}")
            raise
    
    def _update_sumo_config_paths(self) -> None:
        """Update SUMO config file to reference our file naming convention."""
        tree = ET.parse(CONFIG.config_file)
        root = tree.getroot()
        
        # Update file paths to match our naming
        for input_elem in root.findall('.//input'):
            net_file = input_elem.find('net-file')
            if net_file is not None:
                net_file.set('value', 'grid.net.xml')
                
            route_files = input_elem.find('route-files')
            if route_files is not None:
                route_files.set('value', 'vehicles.rou.xml')
        
        # Save updated config
        tree.write(CONFIG.config_file, encoding='utf-8', xml_declaration=True)
        self.logger.info("Updated SUMO config file paths")
    
    def _execute_simulation(self) -> None:
        """Execute dynamic simulation using pre-built sample network."""
        # Validate traffic control compatibility
        if self.args.traffic_control and self.args.traffic_control != 'tree_method':
            self.logger.warning("Tree Method samples optimized for tree_method control")
            self.logger.warning(f"Proceeding with: {self.args.traffic_control}")
        
        # Create traffic controller
        traffic_controller = TrafficControllerFactory.create(self.args.traffic_control, self.args)
        
        # Create and run simulator
        simulator = TrafficSimulator(self.args, traffic_controller)
        metrics = simulator.run()
        
        # Log final metrics
        self.logger.info("=== SAMPLE SIMULATION COMPLETED ===")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")