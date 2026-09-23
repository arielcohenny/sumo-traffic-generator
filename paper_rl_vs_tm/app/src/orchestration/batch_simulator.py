"""
Batch SUMO simulator without TraCI for fixed and actuated traffic control.

This simulator runs SUMO in batch mode (no Python/TraCI control) for traffic
control methods that don't need runtime intervention (fixed and actuated).
SUMO handles everything based on the XML configuration files.

Note: When metric logging is enabled, this simulator uses TraCI to collect
metrics but does not perform any traffic control.
"""

import subprocess
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import sumolib
import traci

from src.config import CONFIG
from src.utils.statistics import parse_sumo_statistics_file, format_cli_statistics_output
from src.utils.metric_logger import MetricLogger


class BatchSimulator:
    """SUMO simulator without TraCI for native traffic control methods.

    Used for fixed and actuated traffic control where SUMO handles everything
    natively based on XML configuration. No Python/TraCI control needed.
    """

    def __init__(self, args: Any, metric_logger: Optional[MetricLogger] = None):
        """Initialize batch simulator.

        Args:
            args: Command line arguments containing simulation parameters
            metric_logger: Optional metric logger for collecting simulation metrics
        """
        self.args = args
        self.metric_logger = metric_logger
        self.logger = logging.getLogger(__name__)

    def run(self) -> Dict[str, Any]:
        """Run SUMO simulation in batch mode.

        Returns:
            Dictionary containing simulation metrics
        """
        self.logger.info(f"Starting SUMO batch simulation (traffic_control={self.args.traffic_control})")

        try:
            # If metric logging is enabled, use TraCI mode (for metric collection only)
            if self.metric_logger:
                self.logger.info("Running SUMO with TraCI for metric collection (no traffic control)")
                self._run_sumo_with_metrics()
            else:
                self.logger.info("Running SUMO natively - no TraCI/Python control")
                self._run_sumo_batch()

            # Parse and return statistics
            metrics = self._calculate_final_metrics()

            self.logger.info("Batch simulation completed successfully!")
            return metrics

        except Exception as e:
            self.logger.error(f"Batch simulation failed: {e}")
            raise

    def _run_sumo_with_metrics(self) -> None:
        """Run SUMO with TraCI for metric collection (no traffic control)."""
        # Choose SUMO binary based on GUI flag
        if self.args.gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        try:
            # Start TraCI
            traci.start([
                sumo_binary,
                '-c', str(CONFIG.config_file),
                '--no-step-log',
                '--no-warnings'
            ])

            # Initialize metric logger
            if self.metric_logger:
                self.metric_logger.initialize_from_traci()
                self.logger.info(f"MetricLogger initialized: logging to {self.metric_logger.output_path}")

            # Run simulation loop with metric collection (NO traffic control)
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step < self.args.end_time:
                # Collect metrics at regular intervals
                if self.metric_logger and self.metric_logger.should_log(step):
                    self.metric_logger.log_metrics(step)

                # Just run the simulation step - no traffic control
                traci.simulationStep()
                step += 1

            # Log final metrics
            if self.metric_logger:
                self.metric_logger.log_metrics(step)

        finally:
            # Close TraCI
            traci.close()

    def _run_sumo_batch(self) -> None:
        """Run SUMO in batch mode without TraCI."""
        # Choose SUMO binary based on GUI flag
        if self.args.gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
            self.logger.info("Running SUMO-GUI in visual mode")
        else:
            sumo_binary = sumolib.checkBinary('sumo')
            self.logger.info("Running SUMO in batch mode")

        # Build SUMO command
        cmd = [
            sumo_binary,
            '-c', str(CONFIG.config_file),
            '--no-step-log',
            '--no-warnings',
            '--duration-log.statistics',  # Enable statistics output
            '--end', str(self.args.end_time)  # Explicit end time
        ]

        # Log command for debugging
        self.logger.debug(f"SUMO command: {' '.join(cmd)}")

        # Run SUMO (blocks until simulation completes)
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Log any SUMO output
            if result.stdout:
                self.logger.debug(f"SUMO stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"SUMO stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMO process failed with return code {e.returncode}")
            self.logger.error(f"SUMO stdout: {e.stdout}")
            self.logger.error(f"SUMO stderr: {e.stderr}")
            raise RuntimeError(f"SUMO batch execution failed: {e}")

    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final simulation metrics from SUMO output files.

        Returns:
            Dict containing simulation metrics
        """
        # Parse SUMO statistics file
        workspace_path = getattr(self.args, 'workspace', 'workspace')
        stats_file = f'{workspace_path}/workspace/sumo_statistics.xml'

        # Check if statistics file exists
        if not Path(stats_file).exists():
            self.logger.warning(f"Statistics file not found: {stats_file}")
            return {
                'total_simulation_steps': self.args.end_time,
                'traffic_control_method': self.args.traffic_control
            }

        stats = parse_sumo_statistics_file(stats_file)

        # Build metrics dictionary
        if stats:
            metrics = {
                'total_simulation_steps': self.args.end_time,
                'vehicles_loaded': stats['loaded'],
                'vehicles_inserted': stats['inserted'],
                'vehicles_still_running': stats['running'],
                'vehicles_waiting': stats['waiting'],
                'vehicles_arrived': stats['arrived'],
                'completion_rate': stats['completion_rate'],
                'traffic_control_method': self.args.traffic_control
            }
        else:
            # Minimal metrics if statistics file parsing failed
            metrics = {
                'total_simulation_steps': self.args.end_time,
                'traffic_control_method': self.args.traffic_control
            }

        # Log formatted statistics
        log_messages = format_cli_statistics_output(
            stats,
            self.args.traffic_control,
            self.args.end_time
        )

        for message in log_messages:
            self.logger.info(message)

        return metrics
