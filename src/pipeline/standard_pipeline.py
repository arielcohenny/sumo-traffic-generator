"""
Standard pipeline for SUMO traffic generation and simulation.

This pipeline executes all 9 steps of the traffic generation process,
from network generation through dynamic simulation.
"""

from .base_pipeline import BasePipeline
from .steps.network_generation_step import NetworkGenerationStep
from .steps.zone_generation_step import ZoneGenerationStep
from src.network.split_edges_with_lanes import execute_edge_splitting
from src.network.custom_lanes import execute_custom_lanes
from src.network.edge_attrs import execute_attractiveness_assignment
from src.sumo_integration.sumo_utils import execute_network_rebuild, execute_config_generation
from src.traffic.builder import execute_route_generation
from src.orchestration.simulator import execute_standard_simulation


class StandardPipeline(BasePipeline):
    """Standard 9-step pipeline for traffic generation and simulation."""
    
    def execute(self) -> None:
        """Execute the complete 9-step pipeline."""
        self._validate_output_directory()
        
        # Step 1: Network Generation
        self._log_step(1, "Network Generation")
        network_step = NetworkGenerationStep(self.args)
        network_step.run()
        
        # Step 2: Zone Generation
        self._log_step(2, "Zone Generation")
        zone_step = ZoneGenerationStep(self.args)
        zone_step.run()
        
        # Step 3: Edge Splitting with Lane Assignment
        self._log_step(3, "Integrated Edge Splitting with Lane Assignment")
        execute_edge_splitting(self.args)
        
        # Step 3.5: Apply Custom Lane Configurations (if provided)
        execute_custom_lanes(self.args)
        
        # Step 4: Network Rebuild
        self._log_step(4, "Network Rebuild")
        execute_network_rebuild(self.args)
        
        # Step 5: Edge Attractiveness Assignment
        self._log_step(5, "Edge Attractiveness Assignment")
        execute_attractiveness_assignment(self.args)
        
        # Step 6: Vehicle Route Generation
        self._log_step(6, "Vehicle Route Generation")
        execute_route_generation(self.args)
        
        # Step 7: SUMO Configuration Generation
        self._log_step(7, "SUMO Configuration Generation")
        execute_config_generation(self.args)
        
        # Step 8: Dynamic Simulation
        self._log_step(8, "Dynamic Simulation")
        execute_standard_simulation(self.args)
    
