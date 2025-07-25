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
            
            # Extract and override end time from SUMO config
            self._override_end_time_from_config()
            
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
    
    def _override_end_time_from_config(self) -> None:
        """Extract end time from SUMO config and override CLI argument."""
        try:
            tree = ET.parse(CONFIG.config_file)
            root = tree.getroot()
            
            # Find the end time in the config
            for time_elem in root.findall('.//time'):
                end_elem = time_elem.find('end')
                if end_elem is not None:
                    config_end_time = int(end_elem.get('value'))
                    
                    self.logger.info(f"Found SUMO config end time: {config_end_time} seconds")
                    self.logger.info(f"CLI end time was: {self.args.end_time} seconds")
                    
                    # Override the CLI argument with the config value
                    self.args.end_time = config_end_time
                    
                    self.logger.info(f"Overriding end time to match SUMO config: {config_end_time} seconds")
                    return
            
            # If no end time found in config, warn but continue with CLI value
            self.logger.warning(f"No end time found in SUMO config, using CLI value: {self.args.end_time}")
            
        except (ET.ParseError, ValueError, AttributeError) as e:
            self.logger.warning(f"Error parsing end time from SUMO config: {e}")
            self.logger.warning(f"Continuing with CLI end time: {self.args.end_time}")
    
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
        
        # Final metrics are provided by SUMO's automatic statistics output
        self.logger.info("=== SAMPLE SIMULATION COMPLETED ===")