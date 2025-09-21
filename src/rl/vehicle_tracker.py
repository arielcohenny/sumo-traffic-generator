"""
Vehicle Journey Tracking for RL Reward Computation.

This module implements individual vehicle tracking and reward computation
based on the design from RL_DISCUSSION.md.
"""

from typing import Dict, List, Set
import logging

from .constants import (
    MEASUREMENT_INTERVAL_STEPS, CREDIT_ASSIGNMENT_WINDOW_STEPS,
    WAITING_TIME_PENALTY_FACTOR, MIN_WAITING_TIME_THRESHOLD,
    COMPLETION_BONUS_PER_VEHICLE, VEHICLE_PENALTY_WEIGHT, THROUGHPUT_BONUS_WEIGHT,
    MAX_TRACKED_VEHICLES, MAX_DECISION_HISTORY, DECISION_CLEANUP_INTERVAL,
    STATISTICS_UPDATE_INTERVAL, DEFAULT_INITIAL_TIME, DEFAULT_INITIAL_PENALTY
)


class VehicleTracker:
    """
    Tracks individual vehicle journeys for RL reward computation.

    Implements the vehicle tracking and credit assignment strategy from
    RL_DISCUSSION.md:
    - Individual vehicle journey monitoring
    - Waiting time delta calculations
    - Time-windowed credit assignment to signal decisions
    """

    def __init__(self):
        """Initialize vehicle tracking system."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Vehicle journey tracking
        self.vehicle_histories: Dict[str, Dict] = {}  # vehicle_id -> journey data
        self.decision_timestamps: List[Dict] = []  # Log of signal control decisions
        self.last_measurement_time: int = DEFAULT_INITIAL_TIME

        # Performance metrics
        self.completed_vehicles: Set[str] = set()
        self.total_penalties: float = DEFAULT_INITIAL_PENALTY

    def update_vehicles(self, current_time: int):
        """Update vehicle states and compute penalties for increased waiting times.

        Args:
            current_time: Current simulation time step
        """
        import traci

        try:
            # Get current active vehicles
            active_vehicles = traci.vehicle.getIDList()

            # Update existing vehicles and add new ones
            for vehicle_id in active_vehicles:
                try:
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    route_id = traci.vehicle.getRouteID(vehicle_id)
                    edge_id = traci.vehicle.getRoadID(vehicle_id)

                    if vehicle_id not in self.vehicle_histories:
                        # New vehicle - initialize tracking
                        self.vehicle_histories[vehicle_id] = {
                            'start_time': current_time,
                            'route': route_id,
                            'last_waiting_time': waiting_time,
                            'total_penalty': DEFAULT_INITIAL_PENALTY,
                            'current_edge': edge_id
                        }
                    else:
                        # Existing vehicle - update waiting time
                        vehicle_data = self.vehicle_histories[vehicle_id]
                        prev_waiting = vehicle_data['last_waiting_time']

                        # Check if waiting time increased
                        if waiting_time > prev_waiting + MIN_WAITING_TIME_THRESHOLD:
                            waiting_increase = waiting_time - prev_waiting
                            penalty = waiting_increase * WAITING_TIME_PENALTY_FACTOR
                            vehicle_data['total_penalty'] += penalty
                            self.total_penalties += penalty

                        vehicle_data['last_waiting_time'] = waiting_time
                        vehicle_data['current_edge'] = edge_id

                except Exception as e:
                    # Skip vehicles that can't be accessed
                    continue

            # Check for completed vehicles
            try:
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for vehicle_id in arrived_vehicles:
                    if vehicle_id in self.vehicle_histories:
                        self.completed_vehicles.add(vehicle_id)
                        # Keep history for episode reward calculation
            except Exception:
                pass

            self.last_measurement_time = current_time

        except Exception as e:
            self.logger.warning(f"Vehicle tracking update failed: {e}")

    def record_decision(self, timestamp: int, actions: Dict):
        """Record signal timing decisions for credit assignment.

        Args:
            timestamp: Time when decision was made
            actions: Dictionary of actions taken (intersection_id -> (phase, duration))
        """
        decision_record = {
            'timestamp': timestamp,
            'actions': actions.copy()
        }
        self.decision_timestamps.append(decision_record)

        # Phase 2: Implement sliding window to limit memory usage
        # Keep only recent decisions within credit assignment window

    def compute_intermediate_rewards(self) -> float:
        """Calculate vehicle-based penalties using time-windowed credit assignment.

        Returns:
            float: Aggregated penalty signal for this measurement interval
        """
        # Simple implementation: return accumulated penalties since last call
        # scaled by vehicle penalty weight
        reward = self.total_penalties * VEHICLE_PENALTY_WEIGHT

        # Reset penalties for next interval (they're already accumulated)
        penalty_since_last = self.total_penalties
        self.total_penalties = DEFAULT_INITIAL_PENALTY

        return penalty_since_last * VEHICLE_PENALTY_WEIGHT

    def compute_episode_reward(self) -> float:
        """Calculate final episode reward based on throughput.

        Returns:
            float: Episode completion bonus based on total completed vehicles
        """
        # R_episode = α × total_completed_vehicles
        throughput_bonus = len(self.completed_vehicles) * COMPLETION_BONUS_PER_VEHICLE * THROUGHPUT_BONUS_WEIGHT
        return throughput_bonus

    def get_vehicle_statistics(self) -> Dict:
        """Get summary statistics about tracked vehicles.

        Returns:
            Dict: Vehicle performance statistics
        """
        return {
            'active_vehicles': len(self.vehicle_histories),
            'completed_vehicles': len(self.completed_vehicles),
            'total_penalties': self.total_penalties,
            'decisions_recorded': len(self.decision_timestamps)
        }

    def reset(self):
        """Reset tracking state for new episode."""
        self.vehicle_histories.clear()
        self.decision_timestamps.clear()
        self.completed_vehicles.clear()
        self.last_measurement_time = DEFAULT_INITIAL_TIME
        self.total_penalties = DEFAULT_INITIAL_PENALTY