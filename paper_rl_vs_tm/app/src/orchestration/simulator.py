"""
Traffic simulation execution.

This module provides the main simulation runner that handles TraCI communication,
vehicle routing, and coordination with traffic controllers.
"""

from typing import Any, Dict, Optional
import logging
from pathlib import Path

import traci
import sumolib

from src.config import CONFIG
from .traffic_controller import TrafficController, TrafficControllerFactory
from .batch_simulator import BatchSimulator
from src.utils.statistics import parse_sumo_statistics_file, format_cli_statistics_output
from src.utils.metric_logger import MetricLogger


class TrafficSimulator:
    """Main traffic simulation runner."""

    def __init__(self, args: Any, traffic_controller: TrafficController, metric_logger: Optional[MetricLogger] = None):
        """Initialize simulator.

        Args:
            args: Command line arguments
            traffic_controller: Traffic controller instance
            metric_logger: Optional metric logger for collecting simulation metrics
        """
        self.args = args
        self.traffic_controller = traffic_controller
        self.metric_logger = metric_logger
        self.logger = logging.getLogger(self.__class__.__name__)

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
            [sumo_binary, '-c', str(CONFIG.config_file), '--no-step-log', '--no-warnings'])

        # Initialize traffic controller AFTER TraCI is connected
        self.traffic_controller.initialize()

        # Initialize metric logger if provided
        if self.metric_logger:
            self.metric_logger.initialize_from_traci()
            self.logger.info(f"MetricLogger initialized: logging to {self.metric_logger.output_path}")

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

            # Collect metrics at regular intervals (if metric logger is enabled)
            if self.metric_logger and self.metric_logger.should_log(step):
                self.metric_logger.log_metrics(step)

            # Dynamic rerouting logic handled by SUMO natively via rerouting devices
            # (no Python code needed - SUMO automatically reroutes vehicles with rerouting devices)

            # Core simulation step
            traci.simulationStep()

            step += 1

        # Log final metrics if metric logger is enabled
        if self.metric_logger:
            self.metric_logger.log_metrics(step)

        # Return final step for metrics calculation after cleanup
        return step

    # REMOVED: _handle_dynamic_rerouting and related methods
    # Dynamic rerouting is now handled natively by SUMO via rerouting devices configured in vehicles.rou.xml
    # No Python code needed - SUMO automatically reroutes vehicles every 30 seconds

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


def execute_standard_simulation(args) -> None:
    """Execute standard dynamic simulation.

    Routes to appropriate simulator based on traffic control method:
    - tree_method/atlcs/rl: TraCISimulator (needs Python control)
    - fixed/actuated: BatchSimulator (SUMO native, no TraCI needed)
    """
    logger = logging.getLogger(__name__)

    # Create metric logger if requested
    metric_logger = None
    if hasattr(args, 'metric_log_path') and args.metric_log_path:
        metric_logger = MetricLogger(
            output_path=Path(args.metric_log_path),
            interval_seconds=90
        )
        logger.info(f"Metric logging enabled: {args.metric_log_path}")

    # Determine if TraCI is needed based on traffic control method
    needs_traci = args.traffic_control in ['tree_method', 'atlcs', 'rl']

    if needs_traci:
        # Dynamic control methods - need TraCI for runtime manipulation
        logger.info(
            f"Using TraCI simulator for {args.traffic_control} (dynamic control)")

        # Create traffic controller
        traffic_controller = TrafficControllerFactory.create(
            args.traffic_control, args)

        # Create and run TraCI simulator
        simulator = TrafficSimulator(args, traffic_controller, metric_logger)
        metrics = simulator.run()
    else:
        # Native SUMO methods - no TraCI needed (unless metric logging is enabled)
        # logger.info(f"Using batch simulator for {args.traffic_control} (native SUMO control)")
        # logger.info("SUMO will handle traffic lights automatically from XML configuration")

        # Create and run batch simulator (no traffic controller needed)
        simulator = BatchSimulator(args, metric_logger)
        metrics = simulator.run()

    # Final metrics are logged by the simulator's unified statistics system


def execute_sample_simulation(args) -> None:
    """Execute sample dynamic simulation using pre-built sample network."""
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
