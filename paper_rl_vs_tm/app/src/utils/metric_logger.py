"""
MetricLogger for collecting comprehensive traffic metrics during simulation runs.

This module provides the MetricLogger class for collecting traffic metrics every 90 seconds
to support empirical reward function design through comparative analysis of Tree Method vs Fixed timing.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import traci

from src.traffic_control.decentralized_traffic_bottlenecks.shared.config import (
    MAX_DENSITY, MIN_VELOCITY, M, L
)

logger = logging.getLogger(__name__)


class MetricLogger:
    """
    Collects comprehensive traffic metrics at regular intervals during simulation.

    Metrics collected include:
    - System-level: total vehicles, waiting time, average speed, throughput
    - Edge-level: occupancy, density, bottleneck count, flow
    - Traffic light: phase distribution, average duration
    - Performance: arrivals, trip duration, waiting time per vehicle
    """

    def __init__(
        self,
        output_path: Path,
        interval_seconds: int = 90,
        edge_ids: Optional[List[str]] = None,
        traffic_light_ids: Optional[List[str]] = None
    ):
        """
        Initialize MetricLogger.

        Args:
            output_path: Path to CSV file for logging metrics
            interval_seconds: Interval between metric collections (default: 90s)
            edge_ids: List of edge IDs to monitor (if None, will auto-detect)
            traffic_light_ids: List of traffic light IDs (if None, will auto-detect)
        """
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.edge_ids = edge_ids or []
        self.traffic_light_ids = traffic_light_ids or []
        self.last_log_time = 0
        self.edge_properties = {}

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with column headers."""
        headers = [
            'timestamp',
            'total_vehicles',
            'vehicles_waiting',
            'total_waiting_time',
            'avg_speed_kmh',
            'throughput',
            'avg_occupancy',
            'avg_density',
            'num_bottleneck_edges',
            'total_flow',
            'avg_edge_travel_time',
            'vehicles_arrived',
            'avg_trip_duration',
            'avg_waiting_per_vehicle',
            'phase_green_ratio',
            'avg_queue_length',
            'total_vehicle_minutes_lost'
        ]

        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def initialize_from_traci(self):
        """
        Auto-detect edge IDs and traffic light IDs from TraCI.
        Cache edge properties for metric calculations.
        """
        if not self.edge_ids:
            self.edge_ids = traci.edge.getIDList()
            # Filter out internal edges (starting with ':')
            self.edge_ids = [e for e in self.edge_ids if not e.startswith(':')]
            logger.info(f"MetricLogger: Auto-detected {len(self.edge_ids)} edges")

        if not self.traffic_light_ids:
            self.traffic_light_ids = traci.trafficlight.getIDList()
            logger.info(f"MetricLogger: Auto-detected {len(self.traffic_light_ids)} traffic lights")

        # Cache edge properties for calculations
        for edge_id in self.edge_ids:
            lane_id = f"{edge_id}_0"
            try:
                self.edge_properties[edge_id] = {
                    'lanes': traci.edge.getLaneNumber(edge_id),
                    'length': traci.lane.getLength(lane_id),
                    'free_flow_speed': traci.lane.getMaxSpeed(lane_id) * 3.6  # m/s to km/h
                }
            except Exception as e:
                logger.warning(f"Could not cache properties for edge {edge_id}: {e}")

    def should_log(self, current_time: float) -> bool:
        """
        Check if metrics should be logged at current simulation time.

        Args:
            current_time: Current simulation time in seconds

        Returns:
            True if metrics should be logged
        """
        if current_time - self.last_log_time >= self.interval_seconds:
            return True
        return False

    def log_metrics(self, current_time: float):
        """
        Collect and log all metrics at current simulation time.

        Args:
            current_time: Current simulation time in seconds
        """
        try:
            metrics = self._collect_all_metrics(current_time)
            self._write_metrics_to_csv(metrics)
            self.last_log_time = current_time
            logger.info(f"MetricLogger: Logged metrics at t={current_time}s")
        except Exception as e:
            logger.error(f"MetricLogger: Error logging metrics at t={current_time}s: {e}")

    def _collect_all_metrics(self, timestamp: float) -> Dict:
        """Collect all traffic metrics from TraCI."""
        metrics = {'timestamp': timestamp}

        # System-level metrics
        vehicle_ids = traci.vehicle.getIDList()
        metrics['total_vehicles'] = len(vehicle_ids)
        metrics['vehicles_waiting'] = sum(1 for vid in vehicle_ids if traci.vehicle.getWaitingTime(vid) > 0)
        metrics['total_waiting_time'] = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)

        # Average speed
        if vehicle_ids:
            metrics['avg_speed_kmh'] = sum(traci.vehicle.getSpeed(vid) * 3.6 for vid in vehicle_ids) / len(vehicle_ids)
        else:
            metrics['avg_speed_kmh'] = 0.0

        # Throughput (vehicles that have arrived)
        metrics['throughput'] = traci.simulation.getArrivedNumber()

        # Edge-level metrics
        edge_metrics = self._collect_edge_metrics()
        metrics.update(edge_metrics)

        # Performance indicators
        perf_metrics = self._collect_performance_metrics()
        metrics.update(perf_metrics)

        # Traffic light metrics
        tl_metrics = self._collect_traffic_light_metrics()
        metrics.update(tl_metrics)

        return metrics

    def _collect_edge_metrics(self) -> Dict:
        """Collect edge-level traffic metrics."""
        occupancies = []
        densities = []
        bottleneck_count = 0
        total_flow = 0
        travel_times = []

        for edge_id in self.edge_ids:
            try:
                # Occupancy (0-1)
                occupancy = traci.edge.getLastStepOccupancy(edge_id)
                occupancies.append(occupancy)

                # Density (vehicles/km)
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                if edge_id in self.edge_properties:
                    length_km = self.edge_properties[edge_id]['length'] / 1000.0
                    lanes = self.edge_properties[edge_id]['lanes']
                    density = vehicle_count / (length_km * lanes) if length_km > 0 else 0
                    densities.append(density)

                    # Check if edge is bottleneck (using Greenshields model)
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge_id) * 3.6  # m/s to km/h
                    free_flow_speed = self.edge_properties[edge_id]['free_flow_speed']
                    q_max_u = self._calculate_q_max_u(free_flow_speed, lanes)

                    if avg_speed < q_max_u:
                        bottleneck_count += 1

                # Flow (vehicles/hour)
                flow = traci.edge.getLastStepVehicleNumber(edge_id) * 3600  # Convert to vehicles/hour
                total_flow += flow

                # Travel time
                travel_time = traci.edge.getTraveltime(edge_id)
                if travel_time > 0:
                    travel_times.append(travel_time)

            except Exception as e:
                logger.warning(f"Error collecting metrics for edge {edge_id}: {e}")

        return {
            'avg_occupancy': sum(occupancies) / len(occupancies) if occupancies else 0.0,
            'avg_density': sum(densities) / len(densities) if densities else 0.0,
            'num_bottleneck_edges': bottleneck_count,
            'total_flow': total_flow,
            'avg_edge_travel_time': sum(travel_times) / len(travel_times) if travel_times else 0.0
        }

    def _collect_performance_metrics(self) -> Dict:
        """Collect performance indicators from simulation."""
        arrived_ids = traci.simulation.getArrivedIDList()

        # For trip duration and waiting time, we need to track vehicles over time
        # For now, use simulation-level aggregates
        vehicle_ids = traci.vehicle.getIDList()

        if vehicle_ids:
            avg_waiting = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids) / len(vehicle_ids)
        else:
            avg_waiting = 0.0

        return {
            'vehicles_arrived': len(arrived_ids),
            'avg_trip_duration': 0.0,  # Will need to track this separately
            'avg_waiting_per_vehicle': avg_waiting
        }

    def _collect_traffic_light_metrics(self) -> Dict:
        """Collect traffic light phase metrics."""
        green_phases = 0
        total_phases = 0
        queue_lengths = []

        for tl_id in self.traffic_light_ids:
            try:
                # Current phase
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)

                # Check if green phase (simple heuristic: phase state contains 'G')
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                if 'G' in state or 'g' in state:
                    green_phases += 1
                total_phases += 1

                # Queue length at controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane_id in controlled_lanes:
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    queue_lengths.append(queue)

            except Exception as e:
                logger.warning(f"Error collecting metrics for traffic light {tl_id}: {e}")

        return {
            'phase_green_ratio': green_phases / total_phases if total_phases > 0 else 0.0,
            'avg_queue_length': sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0
        }

    def _calculate_q_max_u(self, free_flow_speed_km_h: float, num_lanes: int) -> float:
        """
        Calculate optimal flow speed using Greenshields model.

        Args:
            free_flow_speed_km_h: Free flow speed in km/h
            num_lanes: Number of lanes

        Returns:
            Optimal flow speed in km/h
        """
        q_max = 0
        q_max_u = -1

        for k in range(MAX_DENSITY):
            u = self._calc_u_by_k(k, free_flow_speed_km_h)
            q = u * k * num_lanes
            if q > q_max:
                q_max = q
                q_max_u = u

        return q_max_u

    def _calc_u_by_k(self, current_density: int, free_flow_speed_km_h: float) -> int:
        """
        Calculate speed from density using Greenshields model.

        Args:
            current_density: Current traffic density
            free_flow_speed_km_h: Free flow speed in km/h

        Returns:
            Speed in km/h
        """
        return max(
            round(free_flow_speed_km_h * ((1 - (current_density / MAX_DENSITY) ** (L - 1)) ** (1 / (1 - M)))),
            MIN_VELOCITY
        )

    def _write_metrics_to_csv(self, metrics: Dict):
        """Write collected metrics to CSV file."""
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metrics['timestamp'],
                metrics['total_vehicles'],
                metrics['vehicles_waiting'],
                metrics['total_waiting_time'],
                metrics['avg_speed_kmh'],
                metrics['throughput'],
                metrics['avg_occupancy'],
                metrics['avg_density'],
                metrics['num_bottleneck_edges'],
                metrics['total_flow'],
                metrics['avg_edge_travel_time'],
                metrics['vehicles_arrived'],
                metrics['avg_trip_duration'],
                metrics['avg_waiting_per_vehicle'],
                metrics['phase_green_ratio'],
                metrics['avg_queue_length'],
                0.0  # total_vehicle_minutes_lost - will calculate in analysis
            ]
            writer.writerow(row)

    def calculate_total_vehicle_minutes_lost(self) -> float:
        """
        Calculate total vehicle-minutes lost due to bottlenecks (cost-based metric).

        Returns:
            Total vehicle-minutes lost across all bottleneck edges
        """
        total_cost = 0.0

        for edge_id in self.edge_ids:
            try:
                if edge_id not in self.edge_properties:
                    continue

                props = self.edge_properties[edge_id]
                current_speed = traci.edge.getLastStepMeanSpeed(edge_id) * 3.6  # m/s to km/h
                q_max_u = self._calculate_q_max_u(props['free_flow_speed'], props['lanes'])

                # Only consider edges below optimal speed (bottlenecks)
                if current_speed < q_max_u:
                    flow = traci.edge.getLastStepVehicleNumber(edge_id)
                    time_loss_per_vehicle = (q_max_u - current_speed) / 60.0  # Convert to minutes
                    edge_cost = flow * time_loss_per_vehicle
                    total_cost += edge_cost

            except Exception as e:
                logger.warning(f"Error calculating cost for edge {edge_id}: {e}")

        return total_cost
