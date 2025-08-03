"""
Traffic controller interfaces and implementations.

This module provides the abstract interface for traffic controllers and
concrete implementations for different traffic control methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

import traci
from src.config import CONFIG


class TrafficController(ABC):
    """Abstract base class for traffic controllers."""
    
    def __init__(self, args: Any):
        """Initialize traffic controller.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize controller-specific objects and data structures."""
        pass
    
    @abstractmethod
    def update(self, step: int) -> None:
        """Update traffic control at given simulation step.
        
        Args:
            step: Current simulation step
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up controller resources."""
        pass


class TreeMethodController(TrafficController):
    """Traffic controller implementing Tree Method algorithm."""
    
    def __init__(self, args: Any):
        super().__init__(args)
        self.tree_data = None
        self.run_config = None
        self.network_data = None
        self.graph = None
        self.seconds_in_cycle = None
        self.iteration_trees = []
    
    def initialize(self) -> None:
        """Initialize Tree Method objects and data structures."""
        from src.traffic_control.decentralized_traffic_bottlenecks.integration import load_tree
        from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
        from src.traffic_control.decentralized_traffic_bottlenecks.classes.network import Network
        from src.traffic_control.decentralized_traffic_bottlenecks.classes.net_data_builder import build_network_json
        from src.validate.validate_simulation import verify_tree_method_integration_setup
        from pathlib import Path
        
        self.logger.info("=== TREE METHOD CONTROLLER INITIALIZATION ===")
        
        # Build network JSON for Tree Method
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        build_network_json(CONFIG.network_file, json_file)
        
        # Load Tree Method objects with original configuration parameters
        from src.traffic_control.decentralized_traffic_bottlenecks.enums import CostType, AlgoType
        self.tree_data, self.run_config = load_tree(
            net_file=CONFIG.network_file,
            cost_type=CostType.TREE_CURRENT_DIVIDED,  # Match original: TREE_CURRENT_DIVIDED
            algo_type=AlgoType.BABY_STEPS,           # Match original: BABY_STEPS
            sumo_cfg=CONFIG.config_file
        )
        
        self.network_data = Network(json_file)
        
        self.graph = Graph(self.args.end_time)
        
        self.graph.build(self.network_data.edges_list, self.network_data.junctions_dict)
        
        self.seconds_in_cycle = self.network_data.calc_cycle_time()
        
        # Verify Tree Method integration setup
        verify_tree_method_integration_setup(
            self.tree_data, self.run_config, self.network_data, self.graph, self.seconds_in_cycle)
    
    def update(self, step: int) -> None:
        """Update Tree Method traffic control at given step."""
        from src.traffic_control.decentralized_traffic_bottlenecks.utils import is_calculation_time, calc_iteration_from_step
        from src.traffic_control.decentralized_traffic_bottlenecks.enums import AlgoType, CostType
        from src.validate.validate_simulation import verify_algorithm_runtime_behavior
        
        
        # Tree Method: Calculation time check
        is_calc_time = is_calculation_time(step, self.seconds_in_cycle)
        
        if is_calc_time:
            iteration = calc_iteration_from_step(step, self.seconds_in_cycle)
            
            if iteration > 0:  # Skip first iteration
                try:
                    # Perform Tree Method calculations
                    ended_iteration = iteration - 1
                    
                    this_iter_trees_costs = self.graph.calculate_iteration(
                        ended_iteration, 
                        self.iteration_trees, 
                        step, 
                        self.seconds_in_cycle,
                        self.run_config.cost_type if self.run_config else CostType.TREE_CURRENT, 
                        self.run_config.algo_type if self.run_config else AlgoType.BABY_STEPS
                    )
                    
                    self.graph.calc_nodes_statistics(ended_iteration, self.seconds_in_cycle)

                except ZeroDivisionError as zde:
                    import traceback
                    self.logger.error(f"Tree Method division by zero at step {step}: {zde}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                except Exception as e:
                    self.logger.error(f"Tree Method calculation failed at step {step}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Tree Method traffic light updates
        try:
            self.graph.update_traffic_lights(step, self.seconds_in_cycle, 
                                           self.run_config.algo_type if self.run_config else AlgoType.BABY_STEPS)
        except Exception as e:
            self.logger.warning(f"Tree Method traffic light update failed at step {step}: {e}")
        
        # Post-step data collection
        try:
            # Skip fill_link_in_step if there are None edges - data quality issue in Tree Method samples
            if hasattr(self.graph, 'all_links'):
                valid_links = [link for link in self.graph.all_links if link.edge_name != "None" and link.edge_name is not None]
                if len(valid_links) == len(self.graph.all_links):
                    self.graph.fill_link_in_step()
            
            self.graph.add_vehicles_to_step()
            self.graph.close_prev_vehicle_step(step)
            current_iteration = calc_iteration_from_step(step, self.seconds_in_cycle)
            self.graph.fill_head_iteration()
        except Exception as e:
            # Only print every 100 steps to avoid spam
            if step % 100 == 0:
                self.logger.warning(f"Tree Method post-processing failed at step {step}: {e}")
        
        # Runtime validation (every 30 steps by default)
        if step % CONFIG.SIMULATION_VERIFICATION_FREQUENCY == 0:
            try:
                # Save phase data - get_traffic_lights_phases doesn't return data, just saves it
                if self.graph:
                    self.graph.get_traffic_lights_phases(step)
                
                # Get actual traffic light states from TraCI
                phase_map = {}
                try:
                    tls_ids = traci.trafficlight.getIDList()
                    for tls_id in tls_ids:
                        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                        phase_map[tls_id] = current_state
                except traci.TraCIException:
                    phase_map = {}  # Fallback to empty if TraCI fails
                    
                verify_algorithm_runtime_behavior(
                    step, phase_map, self.graph, CONFIG.SIMULATION_VERIFICATION_FREQUENCY
                )
            except Exception as ve:
                self.logger.warning(f"Algorithm runtime validation failed at step {step}: {ve}")
    
    def cleanup(self) -> None:
        """Clean up Tree Method resources and report Tree Method statistics."""
        try:
            self.logger.info("=== TREE METHOD CLEANUP STARTED ===")
            if hasattr(self, 'graph') and self.graph:
                # self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")
                
                # Report Tree Method's own duration statistics
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    tree_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info("=== TREE METHOD STATISTICS ===")
                    self.logger.info(f"Tree Method - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(f"Tree Method - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(f"Tree Method - Average duration: {tree_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(f"Tree Method - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info("=== TREE METHOD STATISTICS ===")
                    self.logger.info("Tree Method - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in Tree Method cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Tree Method objects are cleaned up by garbage collection
        pass


class ActuatedController(TrafficController):
    """Traffic controller using SUMO's built-in actuated control."""
    
    def __init__(self, args):
        super().__init__(args)
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize actuated controller - use native SUMO behavior."""
        self.logger.info("=== ACTUATED CONTROLLER INITIALIZATION ===")
        
        # Initialize Graph object for vehicle tracking (same as Tree Method)
        from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
        self.graph = Graph(self.args.end_time)
        
        try:
            import traci
            traffic_lights = traci.trafficlight.getIDList()
            
        except Exception as e:
            self.logger.error(f"ACTUATED INITIALIZATION ERROR: {e}")
    
    def update(self, step: int) -> None:
        """Update actuated control - let SUMO handle everything."""
        # Vehicle tracking (same as Tree Method)
        try:
            if hasattr(self, 'graph') and self.graph:
                self.graph.add_vehicles_to_step()
                self.graph.close_prev_vehicle_step(step)
        except Exception as e:
            if step % 100 == 0:  # Only log every 100 steps to avoid spam
                self.logger.warning(f"Actuated vehicle tracking failed at step {step}: {e}")
    
    def cleanup(self) -> None:
        """Clean up actuated controller resources and report Actuated statistics."""
        try:
            self.logger.info("=== ACTUATED CLEANUP STARTED ===")
            if hasattr(self, 'graph') and self.graph:
                self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")
                
                # Report Actuated method's duration statistics using same calculation as Tree Method
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    actuated_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info("=== ACTUATED STATISTICS ===")
                    self.logger.info(f"Actuated - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(f"Actuated - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(f"Actuated - Average duration: {actuated_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(f"Actuated - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info("=== ACTUATED STATISTICS ===")
                    self.logger.info("Actuated - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in Actuated cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.logger.info("QA: ACTUATED cleanup completed")


class FixedController(TrafficController):
    """Traffic controller using fixed-time control."""
    
    def __init__(self, args):
        super().__init__(args)
        self.traffic_lights = {}
        self.last_logged_states = {}
        self.graph = None
        
    def initialize(self) -> None:
        """Initialize fixed controller with deterministic phase cycling."""
        self.logger.info("=== FIXED CONTROLLER INITIALIZATION ===")
        
        # Initialize Graph object for vehicle tracking (same as Tree Method)
        from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
        self.graph = Graph(self.args.end_time)
        self.logger.info("QA: FIXED - Initialized vehicle tracking system")
        
        try:
            import traci
            traffic_lights = traci.trafficlight.getIDList()
            
            for tl_id in traffic_lights:
                # Get original phase information without modifying
                complete_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                phases = complete_def.phases
                durations = [int(phase.duration) for phase in phases]
                total_cycle = sum(durations)
                
                self.traffic_lights[tl_id] = {
                    'phase_count': len(phases),
                    'durations': durations,
                    'total_cycle': total_cycle,
                    'current_target_phase': 0
                }

            self.logger.info("QA: FIXED - Initialization complete, will use setPhase + setPhaseDuration")
                
        except Exception as e:
            self.logger.error(f"FIXED INITIALIZATION ERROR: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def update(self, step: int) -> None:
        """Update fixed control with deterministic phase cycling."""
        # Vehicle tracking (same as Tree Method)
        try:
            if hasattr(self, 'graph') and self.graph:
                self.graph.add_vehicles_to_step()
                self.graph.close_prev_vehicle_step(step)
        except Exception as e:
            if step % 100 == 0:  # Only log every 100 steps to avoid spam
                self.logger.warning(f"Fixed vehicle tracking failed at step {step}: {e}")
        
        # Fixed timing control logic
        try:
            import traci
            
            for tl_id, info in self.traffic_lights.items():
                # Calculate which phase should be active
                cycle_position = step % info['total_cycle']
                
                # Find correct phase based on cycle position
                cumulative_time = 0
                target_phase = 0
                for phase_idx, duration in enumerate(info['durations']):
                    if cycle_position < cumulative_time + duration:
                        target_phase = phase_idx
                        break
                    cumulative_time += duration
                
                # Get current phase and update if needed
                current_phase = traci.trafficlight.getPhase(tl_id)
                
                if current_phase != target_phase or step % 10 == 0:  # Update every 10 steps or on change
                    # Use same TraCI calls as Tree Method
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, info['durations'][target_phase])
                
        except Exception as e:
            if step % 100 == 0:  # Only log errors every 100 steps to avoid spam
                self.logger.error(f"FIXED UPDATE ERROR at step {step}: {e}")
    
    def cleanup(self) -> None:
        """Clean up fixed controller resources and report Fixed statistics."""
        try:
            self.logger.info("=== FIXED CLEANUP STARTED ===")
            if hasattr(self, 'graph') and self.graph:
                self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")
                
                # Report Fixed method's duration statistics using same calculation as Tree Method
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    fixed_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info("=== FIXED STATISTICS ===")
                    self.logger.info(f"Fixed - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(f"Fixed - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(f"Fixed - Average duration: {fixed_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(f"Fixed - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info("=== FIXED STATISTICS ===")
                    self.logger.info("Fixed - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in Fixed cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.logger.info("QA: FIXED cleanup completed")


class TrafficControllerFactory:
    """Factory for creating traffic controllers."""
    
    @staticmethod
    def create(traffic_control: str, args: Any) -> TrafficController:
        """Create traffic controller based on type.
        
        Args:
            traffic_control: Type of traffic control ('tree_method', 'actuated', 'fixed')
            args: Command line arguments
            
        Returns:
            TrafficController: Appropriate controller instance
            
        Raises:
            ValueError: If traffic_control type is not supported
        """
        if traffic_control == 'tree_method':
            return TreeMethodController(args)
        elif traffic_control == 'actuated':
            return ActuatedController(args)
        elif traffic_control == 'fixed':
            return FixedController(args)
        else:
            raise ValueError(f"Unsupported traffic control type: {traffic_control}")