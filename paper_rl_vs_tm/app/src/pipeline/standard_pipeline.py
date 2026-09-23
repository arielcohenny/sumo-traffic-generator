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
from src.network.generate_grid import convert_to_incoming_strategy, convert_to_partial_opposites_strategy, convert_to_green_only_phases, convert_traffic_lights_for_control_mode


class StandardPipeline(BasePipeline):
    """Standard 9-step pipeline for traffic generation and simulation."""

    def execute(self) -> None:
        """Execute the complete 9-step pipeline."""
        self._validate_output_directory()

        # Steps 1-7: File generation
        self._execute_file_generation()

        # Step 8: Dynamic Simulation
        self._log_step(8, "Dynamic Simulation")
        execute_standard_simulation(self.args)

    def execute_file_generation_only(self) -> None:
        """Execute only file generation steps (1-7), skip simulation.

        This is used by RL training which manages its own SUMO simulation.
        """
        self._validate_output_directory()

        # Steps 1-7: File generation
        self._execute_file_generation()

    def _execute_file_generation(self) -> None:
        """Execute Steps 1-7 (file generation only, no simulation)."""
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

        # Step 4.5: Apply traffic light strategy and configure for control mode
        # Must be after network rebuild so multi-head edges exist (for partial_opposites)
        if self.args.traffic_light_strategy == "incoming":
            # Apply incoming strategy (each edge gets own phase)
            convert_to_incoming_strategy()
            # Recalculate equal durations for all phases (ensures 90s cycle)
            convert_to_green_only_phases()
            # Configure traffic light type and parameters based on control mode
            convert_traffic_lights_for_control_mode(self.args.traffic_control)
            # Re-embed updated traffic lights into grid.net.xml
            execute_network_rebuild(self.args)
        elif self.args.traffic_light_strategy == "partial_opposites":
            # Apply partial_opposites strategy (straight+right separate from left+uturn)
            convert_to_partial_opposites_strategy()
            # Recalculate equal durations for all phases (ensures 90s cycle)
            convert_to_green_only_phases()
            # Configure traffic light type and parameters based on control mode
            convert_traffic_lights_for_control_mode(self.args.traffic_control)
            # Re-embed updated traffic lights into grid.net.xml
            execute_network_rebuild(self.args)
        else:  # opposites strategy (default)
            # Opposites strategy already applied by netgenerate
            # Just need to ensure equal durations and configure for control mode
            convert_to_green_only_phases()
            convert_traffic_lights_for_control_mode(self.args.traffic_control)
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
