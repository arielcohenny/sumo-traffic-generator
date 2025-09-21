"""
Traffic controller interfaces and implementations.

This module provides the abstract interface for traffic controllers and
concrete implementations for different traffic control methods.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging
import traci


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


class ActuatedController(TrafficController):
    """Traffic controller using SUMO's built-in actuated control."""

    def __init__(self, args):
        super().__init__(args)
        self.graph = None

    def initialize(self) -> None:
        """Initialize actuated controller - use native SUMO behavior."""
        self.logger.info("=== ACTUATED CONTROLLER INITIALIZATION ===")

        # Initialize Graph object for vehicle tracking (same as Tree Method)
        from src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.graph import Graph
        self.graph = Graph(self.args.end_time)

    def update(self, step: int) -> None:
        """Update actuated control - let SUMO handle everything."""
        # Vehicle tracking (same as Tree Method)
        try:
            if hasattr(self, 'graph') and self.graph:
                self.graph.add_vehicles_to_step()
                self.graph.close_prev_vehicle_step(step)
        except Exception as e:
            self.logger.warning(
                f"Actuated vehicle tracking failed at step {step}: {e}")

    def cleanup(self) -> None:
        """Clean up actuated controller resources and report Actuated statistics."""
        try:
            self.logger.info("=== ACTUATED CLEANUP STARTED ===")
            if hasattr(self, 'graph') and self.graph:
                self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(
                    f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(
                    f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")

                # Report Actuated method's duration statistics using same calculation as Tree Method
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    actuated_avg_duration = self.graph.vehicle_total_time / \
                        self.graph.ended_vehicles_count
                    self.logger.info("=== ACTUATED STATISTICS ===")
                    self.logger.info(
                        f"Actuated - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(
                        f"Actuated - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(
                        f"Actuated - Average duration: {actuated_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(
                            f"Actuated - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info("=== ACTUATED STATISTICS ===")
                    self.logger.info(
                        "Actuated - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")

        except Exception as e:
            self.logger.error(f"Error in Actuated cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")


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
        from src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.graph import Graph
        self.graph = Graph(self.args.end_time)

        try:
            traffic_lights = traci.trafficlight.getIDList()

            for tl_id in traffic_lights:
                # Get original phase information without modifying
                complete_def = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[
                    0]
                phases = complete_def.phases
                durations = [int(phase.duration) for phase in phases]
                total_cycle = sum(durations)

                self.traffic_lights[tl_id] = {
                    'phase_count': len(phases),
                    'durations': durations,
                    'total_cycle': total_cycle,
                    'current_target_phase': 0
                }

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
            self.logger.warning(
                f"Fixed vehicle tracking failed at step {step}: {e}")

        # Fixed timing control logic
        try:
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
                    traci.trafficlight.setPhaseDuration(
                        tl_id, info['durations'][target_phase])

        except Exception as e:
            self.logger.error(f"FIXED UPDATE ERROR at step {step}: {e}")

    def cleanup(self) -> None:
        """Clean up fixed controller resources and report Fixed statistics."""
        try:
            self.logger.info("=== FIXED CLEANUP STARTED ===")
            if hasattr(self, 'graph') and self.graph:
                self.logger.info(f"Graph object exists: {type(self.graph)}")
                self.logger.info(
                    f"Ended vehicles count: {getattr(self.graph, 'ended_vehicles_count', 'N/A')}")
                self.logger.info(
                    f"Vehicle total time: {getattr(self.graph, 'vehicle_total_time', 'N/A')}")

                # Report Fixed method's duration statistics using same calculation as Tree Method
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    fixed_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info("=== FIXED STATISTICS ===")
                    self.logger.info(
                        f"Fixed - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(
                        f"Fixed - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(
                        f"Fixed - Average duration: {fixed_avg_duration:.2f} steps")
                    if hasattr(self.graph, 'driving_Time_seconds'):
                        self.logger.info(
                            f"Fixed - Individual durations collected: {len(self.graph.driving_Time_seconds)}")
                else:
                    self.logger.info("=== FIXED STATISTICS ===")
                    self.logger.info(
                        "Fixed - No completed vehicles found or graph not properly initialized")
            else:
                self.logger.info("Graph object not found or not initialized")

        except Exception as e:
            self.logger.error(f"Error in Fixed cleanup: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")


class TrafficControllerFactory:
    """Factory for creating traffic controllers."""

    @staticmethod
    def create(traffic_control: str, args: Any) -> TrafficController:
        """Create traffic controller based on type.

        Args:
            traffic_control: Type of traffic control ('tree_method', 'atlcs', 'actuated', 'fixed', 'rl')
            args: Command line arguments

        Returns:
            TrafficController: Appropriate controller instance

        Raises:
            ValueError: If traffic_control type is not supported
        """
        if traffic_control == 'tree_method':
            from src.traffic_control.decentralized_traffic_bottlenecks.tree_method.controller import TreeMethodController
            return TreeMethodController(args)
        elif traffic_control == 'atlcs':
            from src.traffic_control.decentralized_traffic_bottlenecks.atlcs.controller import ATLCSController
            return ATLCSController(args)
        elif traffic_control == 'actuated':
            return ActuatedController(args)
        elif traffic_control == 'fixed':
            return FixedController(args)
        elif traffic_control == 'rl':
            from src.rl.controller import RLController
            return RLController(args)
        else:
            raise ValueError(
                f"Unsupported traffic control type: {traffic_control}")
