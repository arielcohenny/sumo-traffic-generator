"""
Vehicle Journey Tracking for RL Reward Computation.

This module implements individual vehicle tracking and reward computation
based on the design from RL_DISCUSSION.md.
"""

from typing import Dict, List, Set
import logging

import traci

from .constants import (
    MEASUREMENT_INTERVAL_STEPS, CREDIT_ASSIGNMENT_WINDOW_STEPS,
    WAITING_TIME_PENALTY_FACTOR, MIN_WAITING_TIME_THRESHOLD,
    COMPLETION_BONUS_PER_VEHICLE, VEHICLE_PENALTY_WEIGHT, THROUGHPUT_BONUS_WEIGHT,
    MAX_TRACKED_VEHICLES, MAX_DECISION_HISTORY, DECISION_CLEANUP_INTERVAL,
    STATISTICS_UPDATE_INTERVAL, DEFAULT_INITIAL_TIME, DEFAULT_INITIAL_PENALTY,
    PROGRESSIVE_BONUS_ENABLED, IMMEDIATE_THROUGHPUT_BONUS_WEIGHT,
    PERFORMANCE_STREAK_BASE_BONUS, PERFORMANCE_STREAK_MULTIPLIER, PERFORMANCE_STREAK_THRESHOLD,
    SPEED_IMPROVEMENT_BONUS_FACTOR, CONGESTION_REDUCTION_BONUS, MILESTONE_COMPLETION_BONUSES,
    PERFORMANCE_STREAK_WINDOW_SIZE, SPEED_HISTORY_WINDOW_SIZE, CONGESTION_HISTORY_WINDOW_SIZE,
    MILESTONE_COMPLETION_THRESHOLDS
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

        # Progressive bonus tracking
        self.progressive_bonus_enabled = PROGRESSIVE_BONUS_ENABLED
        if self.progressive_bonus_enabled:
            # Performance streak tracking
            self.performance_history: List[float] = []  # Recent average waiting times
            self.current_streak: int = 0
            self.milestone_achieved: Set[float] = set()  # Achieved completion milestones

            # Network performance tracking
            self.speed_history: List[float] = []  # Recent average speeds
            self.congestion_history: List[int] = []  # Recent bottleneck counts

            # Completion tracking for immediate bonuses
            self.vehicles_completed_this_step: int = 0
            self.total_vehicles_expected: int = 0  # Will be set from config

    def update_vehicles(self, current_time: int):
        """Update vehicle states and compute penalties for increased waiting times.

        Args:
            current_time: Current simulation time step
        """
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
                    # Skip vehicles that can't be accessed, but log the issue
                    self.logger.debug(f"Could not access vehicle {vehicle_id}: {e}")
                    continue

            # Check for completed vehicles
            try:
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for vehicle_id in arrived_vehicles:
                    if vehicle_id in self.vehicle_histories:
                        self.completed_vehicles.add(vehicle_id)
                        # Count vehicles completed this step for progressive bonuses
                        if self.progressive_bonus_enabled:
                            self.vehicles_completed_this_step += 1
                        # Keep history for episode reward calculation
            except Exception as e:
                self.logger.debug(f"Error checking arrived vehicles: {e}")

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

    def set_total_vehicles_expected(self, total_vehicles: int):
        """Set the total number of vehicles expected for milestone calculations.

        Args:
            total_vehicles: Total number of vehicles in the simulation
        """
        if self.progressive_bonus_enabled:
            self.total_vehicles_expected = total_vehicles

    def update_network_performance(self, avg_speed: float, bottleneck_count: int):
        """Update network performance metrics for progressive bonuses.

        Args:
            avg_speed: Current average network speed in km/h
            bottleneck_count: Current number of detected bottlenecks
        """
        if not self.progressive_bonus_enabled:
            return

        # Update speed history
        self.speed_history.append(avg_speed)
        if len(self.speed_history) > SPEED_HISTORY_WINDOW_SIZE:
            self.speed_history.pop(0)

        # Update congestion history
        self.congestion_history.append(bottleneck_count)
        if len(self.congestion_history) > CONGESTION_HISTORY_WINDOW_SIZE:
            self.congestion_history.pop(0)

    def compute_progressive_bonuses(self, current_avg_waiting_time: float = 0.0) -> float:
        """Compute progressive bonuses for the current step.

        Args:
            current_avg_waiting_time: Current average waiting time across all vehicles

        Returns:
            float: Total progressive bonus for this step
        """
        if not self.progressive_bonus_enabled:
            return 0.0

        total_bonus = 0.0

        # 1. Immediate throughput bonuses (vehicles completed this step)
        immediate_bonus = self.vehicles_completed_this_step * IMMEDIATE_THROUGHPUT_BONUS_WEIGHT
        total_bonus += immediate_bonus

        # 2. Performance streak bonuses
        streak_bonus = self._compute_performance_streak_bonus(current_avg_waiting_time)
        total_bonus += streak_bonus

        # 3. Speed improvement bonuses
        speed_bonus = self._compute_speed_improvement_bonus()
        total_bonus += speed_bonus

        # 4. Congestion reduction bonuses
        congestion_bonus = self._compute_congestion_reduction_bonus()
        total_bonus += congestion_bonus

        # 5. Milestone achievement bonuses
        milestone_bonus = self._compute_milestone_bonuses()
        total_bonus += milestone_bonus

        # Reset step counters
        self.vehicles_completed_this_step = 0

        if total_bonus > 0 and hasattr(self, 'logger'):
            self.logger.debug(f"Progressive bonuses: immediate={immediate_bonus:.2f}, "
                             f"streak={streak_bonus:.2f}, speed={speed_bonus:.2f}, "
                             f"congestion={congestion_bonus:.2f}, milestone={milestone_bonus:.2f}, "
                             f"total={total_bonus:.2f}")

        return total_bonus

    def _compute_performance_streak_bonus(self, current_avg_waiting_time: float) -> float:
        """Compute bonus for sustained good performance.

        Args:
            current_avg_waiting_time: Current average waiting time

        Returns:
            float: Performance streak bonus
        """
        # Update performance history
        self.performance_history.append(current_avg_waiting_time)
        if len(self.performance_history) > PERFORMANCE_STREAK_WINDOW_SIZE:
            self.performance_history.pop(0)

        # Check if current performance is good (below threshold)
        if current_avg_waiting_time <= PERFORMANCE_STREAK_THRESHOLD:
            self.current_streak += 1
        else:
            self.current_streak = 0

        # Compute streak bonus with exponential scaling
        if self.current_streak > 0:
            return PERFORMANCE_STREAK_BASE_BONUS * (PERFORMANCE_STREAK_MULTIPLIER ** self.current_streak)

        return 0.0

    def _compute_speed_improvement_bonus(self) -> float:
        """Compute bonus for improving average network speeds.

        Returns:
            float: Speed improvement bonus
        """
        if len(self.speed_history) < 2:
            return 0.0

        # Compare current speed to recent average
        current_speed = self.speed_history[-1]
        recent_avg = sum(self.speed_history[:-1]) / len(self.speed_history[:-1])

        speed_improvement = current_speed - recent_avg
        if speed_improvement > 0:
            return speed_improvement * SPEED_IMPROVEMENT_BONUS_FACTOR

        return 0.0

    def _compute_congestion_reduction_bonus(self) -> float:
        """Compute bonus for reducing network congestion.

        Returns:
            float: Congestion reduction bonus
        """
        if len(self.congestion_history) < 2:
            return 0.0

        # Compare current congestion to recent average
        current_congestion = self.congestion_history[-1]
        recent_avg = sum(self.congestion_history[:-1]) / len(self.congestion_history[:-1])

        congestion_reduction = recent_avg - current_congestion
        if congestion_reduction > 0:
            return congestion_reduction * CONGESTION_REDUCTION_BONUS

        return 0.0

    def _compute_milestone_bonuses(self) -> float:
        """Compute bonuses for reaching completion milestones.

        Returns:
            float: Milestone achievement bonus
        """
        if self.total_vehicles_expected == 0:
            return 0.0

        completion_rate = len(self.completed_vehicles) / self.total_vehicles_expected
        bonus = 0.0

        # Check each milestone threshold
        for i, threshold in enumerate(MILESTONE_COMPLETION_THRESHOLDS):
            if completion_rate >= threshold and threshold not in self.milestone_achieved:
                bonus += MILESTONE_COMPLETION_BONUSES[i]
                self.milestone_achieved.add(threshold)
                if hasattr(self, 'logger'):
                    self.logger.info(f"Milestone achieved: {threshold*100:.0f}% completion "
                                   f"({len(self.completed_vehicles)}/{self.total_vehicles_expected} vehicles), "
                                   f"bonus: {MILESTONE_COMPLETION_BONUSES[i]:.1f}")

        return bonus

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