#!/usr/bin/env python3
"""
Reward Validation Training Script.

Runs 3 short training episodes with realistic parameters to validate reward function behavior.
Each episode generates a separate CSV log (reward_analysis_episode_X.csv) containing:
- Model inputs (observations): speed, bottlenecks, active vehicles, etc.
- Reward components with coefficients applied
- Final reward score

Usage:
    python scripts/run_reward_validation_training.py
"""

import numpy as np
from src.rl.environment import TrafficControlEnv
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_episode(episode_num, env_params):
    """Run a single training episode with random policy.

    Args:
        episode_num: Episode number (1, 2, or 3)
        env_params: Environment parameter string

    Returns:
        Number of steps completed
    """
    print(f"\n{'='*80}")
    print(f"EPISODE {episode_num}/3")
    print(f"{'='*80}\n")

    # Create environment with episode number for proper log file naming
    env = TrafficControlEnv(env_params, episode_number=episode_num)

    # Reset environment
    obs, info = env.reset()
    done = False
    step_count = 0
    cumulative_reward = 0.0

    print(f"Episode {episode_num} started. Will log reward data every 100 steps to reward_analysis_episode_{episode_num}.csv")

    # Run episode with random policy (for reward validation, policy doesn't matter)
    while not done:
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        cumulative_reward += reward
        step_count += 1

        # Progress indicator every 100 steps
        if step_count % 100 == 0:
            print(
                f"  Step {step_count:4d}: Cumulative Reward = {cumulative_reward:+10.2f}", end='\r')

    print(
        f"  Step {step_count:4d}: Cumulative Reward = {cumulative_reward:+10.2f}")

    # Get final statistics
    stats = env.get_final_statistics()

    print(f"\n{'-'*80}")
    print(f"Episode {episode_num} Statistics:")
    print(f"  Steps completed:     {step_count}")
    print(f"  Cumulative reward:   {cumulative_reward:+.2f}")
    print(f"  Completion rate:     {stats['completion_rate']*100:.1f}%")
    print(f"  Vehicles arrived:    {stats['vehicles_arrived']}")
    print(f"  Throughput:          {stats['throughput']:.1f} veh/h")
    print(f"  Avg waiting time:    {stats['avg_waiting_time']:.1f}s")
    print(f"{'-'*80}\n")

    # Clean up
    env.close()

    return step_count


def main():
    """Run 3 episodes with realistic traffic parameters for reward validation."""

    # Realistic traffic parameters (same as production training)
    env_params = (
        "--network-seed 42 "
        "--grid_dimension 5 "
        "--block_size_m 200 "
        "--lane_count realistic "
        "--step-length 1.0 "
        "--land_use_block_size_m 25.0 "
        "--attractiveness land_use "
        "--traffic_light_strategy opposites "
        "--num_vehicles 1500 "
        "--routing_strategy 'shortest 75 realtime 25' "
        "--vehicle_types 'passenger 95 public 5' "
        "--passenger-routes 'in 20 out 20 inner 10 pass 50' "
        "--public-routes 'in 0 out 0 inner 0 pass 100' "
        "--departure_pattern uniform "
        "--private-traffic-seed 418655 "
        "--public-traffic-seed 166903 "
        "--end-time 3600 "
        "--start_time_hour 8.0 "
        "--traffic_control rl"
    )

    print("\n" + "="*80)
    print("REWARD VALIDATION TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  Episodes:        3")
    print("  Duration:        3600s (1 hour) per episode")
    print("  Vehicles:        1500")
    print("  Grid:            5x5")
    print("  Logging:         Every 100 steps → reward_analysis_episode_X.csv")
    print("\nPurpose:")
    print("  Validate that reward function correctly scores good vs bad traffic states")
    print("  by logging reward components and network statistics at each step.")

    start_time = time.time()

    # Run 3 episodes
    for episode in range(1, 4):
        try:
            run_episode(episode, env_params)
        except Exception as e:
            print(f"\n❌ Episode {episode} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("REWARD VALIDATION TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("\nGenerated files:")
    print("  - reward_analysis_episode_1.csv")
    print("  - reward_analysis_episode_2.csv")
    print("  - reward_analysis_episode_3.csv")
    print("\nNext step:")
    print("  python scripts/analyze_reward_validation.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
