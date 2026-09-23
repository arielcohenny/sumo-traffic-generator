"""
Extract metrics from SUMO simulation output files.

Parses tripinfo.xml, summary.xml, and sumo_statistics.xml to extract
detailed metrics for comparison analysis.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

from src.orchestration.run_spec import RunSpec, RunMetrics
from src.utils.statistics import parse_sumo_statistics_file


class MetricsExtractor:
    """Extract and aggregate metrics from SUMO output files.

    Parses multiple SUMO output files to build comprehensive metrics:
    - tripinfo.xml: Per-vehicle trip data (travel time, waiting time per vehicle)
    - summary.xml: Per-timestep simulation summary (queue data)
    - sumo_statistics.xml: Aggregate statistics
    """

    def __init__(self):
        """Initialize metrics extractor."""
        self.logger = logging.getLogger(__name__)

    def extract_from_run(
        self,
        run_path: Path,
        run_spec: RunSpec
    ) -> RunMetrics:
        """Extract all metrics from a completed simulation run.

        Args:
            run_path: Path to the run's workspace directory
            run_spec: Specification of the run (for metadata)

        Returns:
            RunMetrics instance with all extracted metrics
        """
        run_path = Path(run_path)

        # Initialize metrics with spec info
        metrics = RunMetrics(
            name=run_spec.name,
            traffic_control=run_spec.traffic_control,
            private_traffic_seed=run_spec.private_traffic_seed,
            public_traffic_seed=run_spec.public_traffic_seed,
        )

        # Parse tripinfo.xml for per-vehicle data
        tripinfo_path = run_path / "tripinfo.xml"
        if tripinfo_path.exists():
            trip_data = self._parse_tripinfo(tripinfo_path)
            if trip_data:
                metrics.travel_times = trip_data["travel_times"]
                metrics.waiting_times = trip_data["waiting_times"]
                metrics.vehicles_arrived = trip_data["vehicles_arrived"]

                # Calculate statistics from raw data
                if metrics.travel_times:
                    metrics.avg_travel_time = statistics.mean(metrics.travel_times)
                    metrics.std_travel_time = statistics.stdev(metrics.travel_times) if len(metrics.travel_times) > 1 else 0.0
                    metrics.min_travel_time = min(metrics.travel_times)
                    metrics.max_travel_time = max(metrics.travel_times)

                if metrics.waiting_times:
                    metrics.avg_waiting_time = statistics.mean(metrics.waiting_times)
                    metrics.std_waiting_time = statistics.stdev(metrics.waiting_times) if len(metrics.waiting_times) > 1 else 0.0

        # Parse sumo_statistics.xml for aggregate data
        stats_path = run_path / "sumo_statistics.xml"
        if stats_path.exists():
            stats = parse_sumo_statistics_file(str(stats_path))
            if stats:
                metrics.vehicles_departed = stats.get("inserted", 0)
                metrics.throughput = stats.get("throughput", 0.0)
                metrics.simulation_time = stats.get("sim_hours", 0.0) * 3600  # Convert to seconds

                # Use stats file values if tripinfo wasn't available
                if metrics.vehicles_arrived == 0:
                    metrics.vehicles_arrived = stats.get("arrived", 0)
                if metrics.avg_travel_time == 0:
                    metrics.avg_travel_time = stats.get("avg_duration", 0.0)
                if metrics.avg_waiting_time == 0:
                    metrics.avg_waiting_time = stats.get("avg_waiting_time", 0.0)

        # Calculate completion rate
        if metrics.vehicles_departed > 0:
            metrics.completion_rate = metrics.vehicles_arrived / metrics.vehicles_departed
        else:
            metrics.completion_rate = 0.0

        # Parse summary.xml for queue data
        summary_path = run_path / "summary.xml"
        if summary_path.exists():
            queue_data = self._parse_summary(summary_path)
            if queue_data:
                metrics.avg_queue_length = queue_data["avg_halting"]
                metrics.max_queue_length = queue_data["max_halting"]

        return metrics

    def _parse_tripinfo(self, tripinfo_path: Path) -> Optional[Dict]:
        """Parse tripinfo.xml for per-vehicle trip data.

        Args:
            tripinfo_path: Path to tripinfo.xml file

        Returns:
            Dictionary with travel_times, waiting_times, vehicles_arrived
        """
        try:
            tree = ET.parse(tripinfo_path)
            root = tree.getroot()

            travel_times = []
            waiting_times = []
            vehicles_arrived = 0

            for tripinfo in root.findall("tripinfo"):
                # duration = depart time to arrival time
                duration = float(tripinfo.get("duration", 0))
                waiting_time = float(tripinfo.get("waitingTime", 0))

                travel_times.append(duration)
                waiting_times.append(waiting_time)
                vehicles_arrived += 1

            return {
                "travel_times": travel_times,
                "waiting_times": waiting_times,
                "vehicles_arrived": vehicles_arrived,
            }

        except ET.ParseError as e:
            self.logger.warning(f"Failed to parse tripinfo.xml: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error reading tripinfo.xml: {e}")
            return None

    def _parse_summary(self, summary_path: Path) -> Optional[Dict]:
        """Parse summary.xml for timestep-level data.

        Extracts queue length information (halting vehicles) from summary.

        Args:
            summary_path: Path to summary.xml file

        Returns:
            Dictionary with avg_halting, max_halting
        """
        try:
            tree = ET.parse(summary_path)
            root = tree.getroot()

            halting_counts = []

            for step in root.findall("step"):
                # 'halting' attribute indicates vehicles with speed < 0.1 m/s
                halting = int(step.get("halting", 0))
                halting_counts.append(halting)

            if halting_counts:
                return {
                    "avg_halting": statistics.mean(halting_counts),
                    "max_halting": max(halting_counts),
                }

            return None

        except ET.ParseError as e:
            self.logger.warning(f"Failed to parse summary.xml: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error reading summary.xml: {e}")
            return None

    def extract_from_multiple_runs(
        self,
        runs: List[Tuple[Path, RunSpec]]
    ) -> List[RunMetrics]:
        """Extract metrics from multiple runs.

        Args:
            runs: List of (run_path, run_spec) tuples

        Returns:
            List of RunMetrics for each run
        """
        results = []
        for run_path, run_spec in runs:
            try:
                metrics = self.extract_from_run(run_path, run_spec)
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to extract metrics for {run_spec.name}: {e}")
                # Add a metrics object with error flag
                metrics = RunMetrics(
                    name=run_spec.name,
                    traffic_control=run_spec.traffic_control,
                    private_traffic_seed=run_spec.private_traffic_seed,
                    public_traffic_seed=run_spec.public_traffic_seed,
                )
                results.append(metrics)

        return results
