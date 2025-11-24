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
