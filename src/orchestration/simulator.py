"""
Traffic simulation execution.

This module provides the main simulation runner that handles TraCI communication,
vehicle routing, and coordination with traffic controllers.
"""

from typing import Any, Dict
import logging

import traci
import sumolib

from src.config import CONFIG
from .traffic_controller import TrafficController
from src.utils.statistics import parse_sumo_statistics_file, format_cli_statistics_output


class TrafficSimulator:
    """Main traffic simulation runner."""

    def __init__(self, args: Any, traffic_controller: TrafficController):
        """Initialize simulator.

        Args:
            args: Command line arguments
            traffic_controller: Traffic controller instance
        """
        self.args = args
        self.traffic_controller = traffic_controller
        self.logger = logging.getLogger(self.__class__.__name__)

        # Routing strategy tracking
        self.routing_strategies = self._parse_routing_strategy()
        self.last_realtime_reroute = 0
        self.last_fastest_reroute = 0

        # Simulation metrics
        self.total_vehicles = 0
        self.completed_vehicles = 0

    def run(self) -> Dict[str, Any]:
        """Run the complete simulation.

        Returns:
            Dict containing simulation metrics
        """
        try:
            self._initialize_simulation()
            final_step = self._run_simulation_loop()
        finally:
            self._cleanup_simulation()

        # Calculate final metrics after cleanup (statistics file is written after traci.close())
        metrics = self._calculate_final_metrics(final_step)
        return metrics

    def _initialize_simulation(self) -> None:
        """Initialize simulation components."""
        # self.logger.info("Initializing traffic simulation...")

        # Choose SUMO binary based on GUI flag
        if self.args.gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        # Start TraCI FIRST
        # self.logger.info("Starting SUMO simulation with TraCI...")
        traci.start(
            [sumo_binary, '-c', str(CONFIG.config_file), '--no-step-log'])

        # Initialize traffic controller AFTER TraCI is connected
        self.traffic_controller.initialize()

    def _run_simulation_loop(self) -> Dict[str, Any]:
        """Run the main simulation loop.

        Returns:
            Dict containing simulation metrics
        """
        step = 0

        # Main simulation loop
        while traci.simulation.getMinExpectedNumber() > 0 and step < self.args.end_time:

            # Traffic control updates BEFORE simulation step
            self.traffic_controller.update(step)

            # Dynamic rerouting logic
            self._handle_dynamic_rerouting(step)

            # Core simulation step
            traci.simulationStep()

            step += 1

        # Return final step for metrics calculation after cleanup
        return step

    def _handle_dynamic_rerouting(self, step: int) -> None:
        """Handle dynamic vehicle rerouting based on strategies."""
        if not self.routing_strategies:
            return

        if "realtime" in self.routing_strategies and self._should_reroute(step, "realtime", self.last_realtime_reroute, 30):
            self._reroute_vehicles_by_strategy("realtime")
            self.last_realtime_reroute = step

        if "fastest" in self.routing_strategies and self._should_reroute(step, "fastest", self.last_fastest_reroute, 45):
            self._reroute_vehicles_by_strategy("fastest")
            self.last_fastest_reroute = step

    def _should_reroute(self, step: int, strategy: str, last_reroute: int, interval: int) -> bool:
        """Check if vehicles should be rerouted based on strategy and interval."""
        return step - last_reroute >= interval

    def _reroute_vehicles_by_strategy(self, strategy_type: str) -> None:
        """Reroute vehicles based on strategy type."""
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            try:
                if strategy_type == "realtime":
                    traci.vehicle.rerouteEffort(vehicle_id)
                elif strategy_type == "fastest":
                    traci.vehicle.rerouteTraveltime(vehicle_id)
            except traci.exceptions.TraCIException:
                # Vehicle might have left the simulation
                continue

    def _calculate_final_metrics(self, final_step: int) -> Dict[str, Any]:
        """Calculate final simulation metrics using SUMO statistics file.

        Args:
            final_step: Final simulation step

        Returns:
            Dict containing simulation metrics
        """
        # Parse SUMO statistics file for comprehensive metrics
        # Note: This is called AFTER traci.close() so the statistics file is complete
        workspace_path = getattr(self.args, 'workspace', 'workspace')
        stats_file = f'{workspace_path}/workspace/sumo_statistics.xml'
        stats = parse_sumo_statistics_file(stats_file)

        # Build metrics dictionary
        if stats:
            metrics = {
                'total_simulation_steps': final_step,
                'vehicles_loaded': stats['loaded'],
                'vehicles_inserted': stats['inserted'],
                'vehicles_still_running': stats['running'],
                'vehicles_waiting': stats['waiting'],
                'vehicles_arrived': stats['arrived'],
                'completion_rate': stats['completion_rate'],
                'traffic_control_method': self.args.traffic_control
            }
        else:
            # Minimal metrics if statistics file is not available
            metrics = {
                'total_simulation_steps': final_step,
                'traffic_control_method': self.args.traffic_control
            }

        # Log formatted statistics using shared formatter
        log_messages = format_cli_statistics_output(
            stats,
            self.args.traffic_control,
            final_step
        )

        for message in log_messages:
            self.logger.info(message)

        return metrics

    def _cleanup_simulation(self) -> None:
        """Clean up simulation resources."""
        try:
            self.traffic_controller.cleanup()
            traci.close()
            self.logger.info("Simulation completed successfully!")
        except Exception as e:
            self.logger.error(f"Error during simulation cleanup: {e}")

    def _parse_routing_strategy(self) -> Dict[str, float]:
        """Parse routing strategy string to extract individual strategies and percentages."""
        if not hasattr(self.args, 'routing_strategy') or not self.args.routing_strategy:
            return {}

        strategies = {}
        parts = self.args.routing_strategy.split()
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                strategy = parts[i]
                percentage = float(parts[i + 1])
                strategies[strategy] = percentage
        return strategies


def execute_standard_simulation(args) -> None:
    """Execute standard dynamic simulation."""
    import logging
    from .traffic_controller import TrafficControllerFactory

    logger = logging.getLogger(__name__)

    # Create traffic controller
    traffic_controller = TrafficControllerFactory.create(
        args.traffic_control, args)

    # Create and run simulator
    simulator = TrafficSimulator(args, traffic_controller)
    metrics = simulator.run()

    # Final metrics are logged by the simulator's unified statistics system


def execute_sample_simulation(args) -> None:
    """Execute sample dynamic simulation using pre-built sample network."""
    import logging
    from .traffic_controller import TrafficControllerFactory

    logger = logging.getLogger(__name__)

    # Validate traffic control compatibility
    if args.traffic_control and args.traffic_control != 'tree_method':
        logger.warning("Tree Method samples optimized for tree_method control")
        logger.warning(f"Proceeding with: {args.traffic_control}")

    # Create traffic controller
    traffic_controller = TrafficControllerFactory.create(
        args.traffic_control, args)

    # Create and run simulator
    simulator = TrafficSimulator(args, traffic_controller)
    metrics = simulator.run()

    # Final metrics are provided by SUMO's automatic statistics output
    logger.info("=== SAMPLE SIMULATION COMPLETED ===")
