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
        
        self.logger.info("Initializing Tree Method objects...")
        
        # Build network JSON for Tree Method
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        build_network_json(CONFIG.network_file, json_file)
        self.logger.info(f"Built network JSON file: {json_file}")
        
        # Load Tree Method objects
        self.tree_data, self.run_config = load_tree(
            net_file=CONFIG.network_file,
            sumo_cfg=CONFIG.config_file
        )
        self.logger.info("Loaded network tree and run configuration successfully")
        
        self.network_data = Network(json_file)
        self.logger.info("Loaded network data from JSON")
        
        self.graph = Graph(self.args.end_time)
        self.logger.info("Initialized Tree Method Graph object")
        
        self.graph.build(self.network_data.edges_list, self.network_data.junctions_dict)
        self.logger.info("Built Tree Method Graph from network data")
        
        self.seconds_in_cycle = self.network_data.calc_cycle_time()
        self.logger.info("Built network graph and calculated cycle time")
        
        # Verify Tree Method integration setup
        verify_tree_method_integration_setup(
            self.tree_data, self.run_config, self.network_data, self.graph, self.seconds_in_cycle)
        self.logger.info("Tree Method integration setup verified successfully")
    
    def update(self, step: int) -> None:
        """Update Tree Method traffic control at given step."""
        from src.traffic_control.decentralized_traffic_bottlenecks.utils import is_calculation_time, calc_iteration_from_step
        from src.traffic_control.decentralized_traffic_bottlenecks.enums import AlgoType, CostType
        from src.validate.validate_simulation import verify_algorithm_runtime_behavior
        
        # Tree Method: Calculation time check
        if is_calculation_time(step, self.seconds_in_cycle):
            iteration = calc_iteration_from_step(step, self.seconds_in_cycle)
            if iteration > 0:  # Skip first iteration
                self.logger.debug(f"Tree Method calculation at step {step}, iteration {iteration}")
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
                    self.logger.warning(f"Tree Method division by zero at step {step}: {zde}")
                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                except Exception as e:
                    self.logger.warning(f"Tree Method calculation failed at step {step}: {e}")
        
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
                phase_map = self.graph.get_traffic_lights_phases(step) if self.graph else {}
                verify_algorithm_runtime_behavior(
                    step, phase_map, self.graph, CONFIG.SIMULATION_VERIFICATION_FREQUENCY
                )
            except Exception as ve:
                self.logger.warning(f"Algorithm runtime validation failed at step {step}: {ve}")
    
    def cleanup(self) -> None:
        """Clean up Tree Method resources."""
        # Tree Method objects are cleaned up by garbage collection
        pass


class ActuatedController(TrafficController):
    """Traffic controller using SUMO's built-in actuated control."""
    
    def initialize(self) -> None:
        """Initialize actuated controller (no setup needed)."""
        self.logger.info("Using SUMO Actuated traffic control - no additional setup needed")
    
    def update(self, step: int) -> None:
        """Update actuated control (SUMO handles automatically)."""
        # SUMO Actuated control - let SUMO handle traffic lights automatically
        pass
    
    def cleanup(self) -> None:
        """Clean up actuated controller resources."""
        pass


class FixedController(TrafficController):
    """Traffic controller using fixed-time control."""
    
    def initialize(self) -> None:
        """Initialize fixed controller (no setup needed)."""
        self.logger.info("Using Fixed-time traffic control - no additional setup needed")
    
    def update(self, step: int) -> None:
        """Update fixed control (SUMO handles automatically)."""
        # Fixed-time control - let SUMO use the static timings from grid.tll.xml
        pass
    
    def cleanup(self) -> None:
        """Clean up fixed controller resources."""
        pass


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