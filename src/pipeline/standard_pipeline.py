"""
Standard pipeline for SUMO traffic generation and simulation.

This pipeline executes all 9 steps of the traffic generation process,
from network generation through dynamic simulation.
"""

from pathlib import Path

from .base_pipeline import BasePipeline
from .steps.network_generation_step import NetworkGenerationStep
from .steps.zone_generation_step import ZoneGenerationStep
from src.network.split_edges_with_lanes import split_edges_with_flow_based_lanes
from src.network.edge_attrs import assign_edge_attractiveness
from src.network.intelligent_zones import convert_zones_to_projected_coordinates
from src.sumo_integration.sumo_utils import generate_sumo_conf_file, rebuild_network
from src.traffic.builder import generate_vehicle_routes
from src.orchestration.simulator import TrafficSimulator
from src.orchestration.traffic_controller import TrafficControllerFactory
from src.validate.validate_network import verify_rebuild_network, verify_assign_edge_attractiveness, verify_generate_sumo_conf_file
from src.validate.validate_traffic import verify_generate_vehicle_routes
from src.validate.validate_intelligent_zones import verify_convert_zones_to_projected_coordinates
from src.validate.validate_split_edges_with_lanes import verify_split_edges_with_flow_based_lanes
from src.validate.errors import ValidationError
from src.config import CONFIG


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
        self._execute_edge_splitting()
        
        # Step 3.5: Apply Custom Lane Configurations (if provided)
        self._execute_custom_lanes()
        
        # Step 4: Network Rebuild
        self._log_step(4, "Network Rebuild")
        self._execute_network_rebuild()
        
        # Step 5: Zone Coordinate Conversion (OSM Mode Only)
        if self.args.osm_file:
            self._log_step(5, "Zone Coordinate Conversion")
            self._execute_zone_conversion()
        
        # Step 6: Edge Attractiveness Assignment
        self._log_step(6, "Edge Attractiveness Assignment")
        self._execute_attractiveness_assignment()
        
        # Step 7: Vehicle Route Generation
        self._log_step(7, "Vehicle Route Generation")
        self._execute_route_generation()
        
        # Step 8: SUMO Configuration Generation
        self._log_step(8, "SUMO Configuration Generation")
        self._execute_config_generation()
        
        # Step 9: Dynamic Simulation
        self._log_step(9, "Dynamic Simulation")
        self._execute_simulation()
    
    def _execute_edge_splitting(self) -> None:
        """Execute edge splitting with lane assignment."""
        if self.args.lane_count != "0" and not (self.args.lane_count.isdigit() and self.args.lane_count == "0"):
            split_edges_with_flow_based_lanes(
                seed=self._get_seed(),
                min_lanes=CONFIG.MIN_LANES,
                max_lanes=CONFIG.MAX_LANES,
                algorithm=self.args.lane_count,
                block_size_m=self.args.block_size_m
            )
            self.logger.info("Successfully completed integrated edge splitting with lane assignment")
            
            # Validate the split edges
            try:
                verify_split_edges_with_flow_based_lanes(
                    connections_file=str(CONFIG.network_con_file),
                    edges_file=str(CONFIG.network_edg_file),
                    nodes_file=str(CONFIG.network_nod_file)
                )
                self.logger.info("Split edges validation passed successfully")
            except (ValidationError, ValueError) as ve:
                self.logger.error(f"Split edges validation failed: {ve}")
                raise
        else:
            self.logger.info("Skipping lane assignment (lane_count is 0)")
    
    def _execute_custom_lanes(self) -> None:
        """Execute custom lane configuration application."""
        from src.network.custom_lanes import create_custom_lane_config_from_args, apply_custom_lane_configs
        
        # Create custom lane configuration from arguments
        custom_lane_config = create_custom_lane_config_from_args(self.args)
        
        if custom_lane_config and custom_lane_config.edge_configs:
            self._log_step(3.5, "Applying Custom Lane Configurations")
            apply_custom_lane_configs(custom_lane_config)
            self.logger.info(f"Successfully applied custom lane configurations for {len(custom_lane_config.edge_configs)} edges")
        else:
            self.logger.info("No custom lane configurations to apply")
    
    def _execute_network_rebuild(self) -> None:
        """Execute network rebuild."""
        rebuild_network()
        try:
            verify_rebuild_network()
        except ValidationError as ve:
            self.logger.error(f"Failed to rebuild the network: {ve}")
            raise
        self.logger.info("Rebuilt the network successfully")
    
    def _execute_zone_conversion(self) -> None:
        """Execute zone coordinate conversion for OSM networks."""
        if Path(CONFIG.zones_file).exists():
            self.logger.info("Converting OSM zone coordinates from geographic to projected...")
            try:
                convert_zones_to_projected_coordinates(CONFIG.zones_file, CONFIG.network_file)
                self.logger.info("Successfully converted zone coordinates to projected system")
            except Exception as e:
                self.logger.warning(f"Failed to convert zone coordinates: {e}")
                self.logger.warning("Zones will remain in geographic coordinates")
            
            self.logger.info("Validating zone coverage against network bounds...")
            try:
                verify_convert_zones_to_projected_coordinates(CONFIG.zones_file, CONFIG.network_file)
                self.logger.info("Zone coverage validation passed")
            except (ValidationError, Exception) as e:
                self.logger.error(f"Zone coverage validation failed: {e}")
                raise
    
    def _execute_attractiveness_assignment(self) -> None:
        """Execute edge attractiveness assignment."""
        assign_edge_attractiveness(
            self._get_seed(), 
            self.args.attractiveness, 
            self.args.time_dependent, 
            self.args.start_time_hour
        )
        try:
            verify_assign_edge_attractiveness(
                self._get_seed(), 
                self.args.attractiveness, 
                self.args.time_dependent
            )
        except ValidationError as ve:
            self.logger.error(f"Failed to assign edge attractiveness: {ve}")
            raise
        self.logger.info("Assigned edge attractiveness successfully")
    
    def _execute_route_generation(self) -> None:
        """Execute vehicle route generation."""
        generate_vehicle_routes(
            net_file=CONFIG.network_file,
            output_file=CONFIG.routes_file,
            num_vehicles=self.args.num_vehicles,
            seed=self._get_seed(),
            routing_strategy=self.args.routing_strategy,
            vehicle_types=self.args.vehicle_types,
            end_time=self.args.end_time,
            departure_pattern=self.args.departure_pattern
        )
        try:
            verify_generate_vehicle_routes(
                net_file=CONFIG.network_file,
                output_file=CONFIG.routes_file,
                num_vehicles=self.args.num_vehicles,
                seed=self._get_seed(),
            )
        except ValidationError as ve:
            self.logger.error(f"Failed to generate vehicle routes: {ve}")
            raise
        self.logger.info("Generated vehicle routes successfully")
    
    def _execute_config_generation(self) -> None:
        """Execute SUMO configuration generation."""
        sumo_cfg_path = generate_sumo_conf_file(
            CONFIG.config_file,
            CONFIG.network_file,
            route_file=CONFIG.routes_file,
            zones_file=CONFIG.zones_file,
        )
        try:
            verify_generate_sumo_conf_file()
        except ValidationError as ve:
            self.logger.error(f"SUMO configuration validation failed: {ve}")
            raise
        self.logger.info("Generated SUMO configuration file successfully")
    
    def _execute_simulation(self) -> None:
        """Execute dynamic simulation."""
        # Create traffic controller
        traffic_controller = TrafficControllerFactory.create(self.args.traffic_control, self.args)
        
        # Create and run simulator
        simulator = TrafficSimulator(self.args, traffic_controller)
        metrics = simulator.run()
        
        # Log final metrics
        self.logger.info("=== SIMULATION COMPLETED ===")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")
    
    def _get_seed(self) -> int:
        """Get the cached random seed."""
        if hasattr(self.args, '_seed'):
            return self.args._seed
        
        import random
        seed = self.args.seed if self.args.seed is not None else random.randint(0, 2**32 - 1)
        self.args._seed = seed
        return seed