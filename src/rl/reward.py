"""Reward function for traffic signal control RL agent.

This module contains the reward calculation logic separated from the environment.
The RewardCalculator is a pure function that computes rewards based on provided metrics,
making it testable and maintainable.
"""

from typing import Optional, List
import csv

from src.rl.constants import (
    REWARD_THROUGHPUT_PER_VEHICLE,
    REWARD_EXCESSIVE_WAITING_PENALTY,
    REWARD_EXCESSIVE_WAITING_THRESHOLD,
    REWARD_SPEED_REWARD_FACTOR,
    REWARD_SPEED_NORMALIZATION,
    REWARD_BOTTLENECK_PENALTY_PER_EDGE,
    REWARD_INSERTION_BONUS,
    REWARD_INSERTION_THRESHOLD
)

# ═══════════════════════════════════════════════════════════
# EMPIRICAL REWARD NORMALIZATION STATISTICS
# ═══════════════════════════════════════════════════════════
# These statistics were derived from comparative analysis of
# Tree Method (expert) vs Fixed timing (baseline) across 5 episodes.
# Used for z-score normalization: (value - mean) / std
#
# Source: evaluation/comparative_analysis/metric_statistics.csv
# Date: 2025-12-06
# ═══════════════════════════════════════════════════════════

EMPIRICAL_NORM_STATS = {
    'avg_waiting_per_vehicle': {'mean': 38.01, 'std': 11.12},
    'avg_speed_kmh': {'mean': 5.74, 'std': 2.73},
    'avg_queue_length': {'mean': 2.38, 'std': 0.71}
    # REMOVED: avg_edge_travel_time - getTraveltime() returns pathological values
    # for edges with no recent traffic (100,000+ seconds), breaking z-score normalization
}

# Empirically derived weights based on Cohen's d (discriminative power)
# Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large
# Weights redistributed after removing broken travel time metric (was 0.25)
EMPIRICAL_WEIGHTS = {
    'waiting': 0.45,  # d=0.966 (very large effect) - was 0.35, +0.10
    'speed': 0.35,    # d=0.682 (medium effect) - was 0.25, +0.10
    'queue': 0.20     # d=0.505 (medium effect) - was 0.15, +0.05
    # REMOVED: 'travel': 0.25 (broken metric)
}


class RewardCalculator:
    """Calculates multi-objective rewards for traffic signal control.

    The reward function optimizes for:
    1. Throughput: Maximize vehicles completing their journeys
    2. Waiting time: Minimize vehicle waiting and congestion
    3. Network flow: Maintain good average speeds and reduce bottlenecks
    4. Insertion: Get vehicles into the network efficiently

    This class is designed to be pure - it takes metrics as inputs and returns
    rewards, with no dependencies on TraCI or environment state.
    """

    def __init__(self, log_file_path: Optional[str] = None):
        """Initialize reward calculator with optional CSV logging.

        Args:
            log_file_path: Path to CSV file for logging reward components.
                          If None, logging is disabled.
        """
        self.log_file = None
        self.log_writer = None

        if log_file_path:
            self.log_file = open(log_file_path, 'w', newline='')
            self.log_writer = csv.writer(self.log_file)

            # Write CSV header
            self.log_writer.writerow([
                'step',
                'avg_speed',
                'bottleneck_count',
                'active_vehicles',
                'waiting_to_insert',
                'vehicles_with_excessive_waiting',
                'vehicles_completed_this_step',
                'throughput_reward',
                'waiting_penalty',
                'excessive_waiting_penalty',
                'speed_reward',
                'bottleneck_penalty',
                'insertion_bonus',
                'total_reward'
            ])

    def compute_reward(
        self,
        vehicles_completed: int,
        waiting_penalty: float,
        avg_speed: float,
        bottleneck_count: int,
        active_vehicle_waiting_times: List[float],
        waiting_to_insert: int
    ) -> float:
        """Compute multi-objective reward based on traffic metrics.

        Args:
            vehicles_completed: Number of vehicles that completed this step
            waiting_penalty: Penalty from vehicle tracker (already computed)
            avg_speed: Average network speed in km/h
            bottleneck_count: Number of bottleneck edges detected
            active_vehicle_waiting_times: List of waiting times for active vehicles
            waiting_to_insert: Number of vehicles waiting to be inserted

        Returns:
            Total reward value combining all components
        """
        # ═══════════════════════════════════════════════════════════
        # 1. THROUGHPUT REWARDS (Primary Goal)
        # ═══════════════════════════════════════════════════════════
        # Immediate reward for every vehicle that completes this step
        throughput_reward = vehicles_completed * REWARD_THROUGHPUT_PER_VEHICLE

        # ═══════════════════════════════════════════════════════════
        # 2. WAITING TIME PENALTIES (Reduce Congestion)
        # ═══════════════════════════════════════════════════════════
        # waiting_penalty already computed by vehicle tracker

        # Additional penalty for vehicles with excessive waiting (>5 minutes)
        excessive_waiting_penalty = 0.0
        for waiting_time in active_vehicle_waiting_times:
            if waiting_time > REWARD_EXCESSIVE_WAITING_THRESHOLD:
                excessive_waiting_penalty -= REWARD_EXCESSIVE_WAITING_PENALTY

        # ═══════════════════════════════════════════════════════════
        # 3. NETWORK FLOW EFFICIENCY (Keep traffic moving)
        # ═══════════════════════════════════════════════════════════
        # Reward for maintaining good average speed
        speed_reward = (avg_speed / REWARD_SPEED_NORMALIZATION) * REWARD_SPEED_REWARD_FACTOR

        # Penalty for bottlenecks (edges with congestion)
        bottleneck_penalty = -bottleneck_count * REWARD_BOTTLENECK_PENALTY_PER_EDGE

        # ═══════════════════════════════════════════════════════════
        # 4. INSERTION RATE BONUS (Get vehicles into network)
        # ═══════════════════════════════════════════════════════════
        insertion_bonus = 0.0
        if waiting_to_insert < REWARD_INSERTION_THRESHOLD:
            insertion_bonus = REWARD_INSERTION_BONUS

        # ═══════════════════════════════════════════════════════════
        # TOTAL REWARD
        # ═══════════════════════════════════════════════════════════
        total_reward = (
            throughput_reward +           # +50 per completed vehicle (PRIMARY)
            waiting_penalty +              # Negative (from vehicle tracker)
            excessive_waiting_penalty +    # Extra penalty for long waits
            speed_reward +                 # Reward for good network speed
            bottleneck_penalty +           # Penalty for congested edges
            insertion_bonus                # Bonus for clearing insertion queue
        )

        return total_reward

    def compute_empirical_reward(
        self,
        avg_waiting_per_vehicle: float,
        avg_speed_kmh: float,
        avg_queue_length: float,
        vehicles_arrived_this_step: int = 0
    ) -> float:
        """Compute empirically validated reward based on comparative analysis.

        This reward function was designed through data-driven analysis comparing
        Tree Method (expert) vs Fixed timing (baseline) across 5 episodes.
        It uses z-score normalization and Cohen's d-based weights.

        Updated 2025-12-07: Removed avg_edge_travel_time (broken metric - SUMO's
        getTraveltime() returns pathological values for edges with no recent traffic).
        Added throughput bonus for positive reward signal.

        Expected reward range: -1 to +2
        - Good traffic control → positive reward (near 0 to +2)
        - Bad traffic control → negative reward (-1 to 0)

        Args:
            avg_waiting_per_vehicle: Average waiting time per vehicle (seconds)
            avg_speed_kmh: Average network speed (km/h)
            avg_queue_length: Average queue length (vehicles)
            vehicles_arrived_this_step: Number of vehicles that completed their trips

        Returns:
            Empirically validated reward value (higher is better)

        See:
            evaluation/comparative_analysis/metric_statistics.csv
        """
        # Z-score normalization helper
        def normalize(value: float, metric_name: str) -> float:
            """Normalize metric using z-score: (value - mean) / std"""
            stats = EMPIRICAL_NORM_STATS[metric_name]
            return (value - stats['mean']) / stats['std']

        # Normalize all metrics (3 metrics only - travel time removed)
        z_waiting = normalize(avg_waiting_per_vehicle, 'avg_waiting_per_vehicle')
        z_speed = normalize(avg_speed_kmh, 'avg_speed_kmh')
        z_queue = normalize(avg_queue_length, 'avg_queue_length')

        # Weighted combination based on discriminative power
        # Weights sum to 1.0 for interpretability
        reward = (
            -EMPIRICAL_WEIGHTS['waiting'] * z_waiting  # Minimize per-vehicle waiting
            + EMPIRICAL_WEIGHTS['speed'] * z_speed      # Maximize network speed
            - EMPIRICAL_WEIGHTS['queue'] * z_queue      # Minimize queue length
        )

        # Throughput bonus: positive signal for completing vehicle trips
        # This encourages the agent to actually move vehicles through the network
        throughput_bonus = vehicles_arrived_this_step * 0.1

        return reward + throughput_bonus

    def log_reward_components(
        self,
        step: int,
        avg_speed: float,
        bottleneck_count: int,
        active_vehicles: int,
        waiting_to_insert: int,
        vehicles_with_excessive_waiting: int,
        vehicles_completed_this_step: int,
        throughput_reward: float,
        waiting_penalty: float,
        excessive_waiting_penalty: float,
        speed_reward: float,
        bottleneck_penalty: float,
        insertion_bonus: float,
        total_reward: float
    ):
        """Log reward components to CSV file.

        Args:
            step: Current simulation step
            avg_speed: Average network speed (km/h)
            bottleneck_count: Number of bottleneck edges
            active_vehicles: Number of active vehicles in network
            waiting_to_insert: Number of vehicles waiting to be inserted
            vehicles_with_excessive_waiting: Count of vehicles waiting >5min
            vehicles_completed_this_step: Vehicles that completed this step
            throughput_reward: Throughput component of reward
            waiting_penalty: Waiting time penalty component
            excessive_waiting_penalty: Excessive waiting penalty component
            speed_reward: Speed reward component
            bottleneck_penalty: Bottleneck penalty component
            insertion_bonus: Insertion bonus component
            total_reward: Total reward value
        """
        if self.log_writer:
            self.log_writer.writerow([
                step,
                avg_speed,
                bottleneck_count,
                active_vehicles,
                waiting_to_insert,
                vehicles_with_excessive_waiting,
                vehicles_completed_this_step,
                throughput_reward,
                waiting_penalty,
                excessive_waiting_penalty,
                speed_reward,
                bottleneck_penalty,
                insertion_bonus,
                total_reward
            ])
            self.log_file.flush()

    def close(self):
        """Close CSV log file if open."""
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass
