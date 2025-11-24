#!/usr/bin/env python3
"""
Reward Function Validation Script.

Tests different traffic control policies with the current reward function
to validate that reward correlates with actual performance metrics.

Usage:
    python scripts/evaluate_reward_function.py --episodes 3 --timesteps 3600
    python scripts/evaluate_reward_function.py --episodes 5 --timesteps 7200 --vehicles 1000
"""

import numpy as np
import pandas as pd
import argparse
import random
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.environment import TrafficControlEnv


def greedy_policy(obs, step, num_junctions):
    """Greedy policy: favor main traffic directions (phases 0 and 2)."""
    # Alternate between phase 0 and phase 2 for better flow
    return np.array([0 if i % 2 == 0 else 2 for i in range(num_junctions)])


def fair_rotation_policy(obs, step, num_junctions):
    """Fair rotation: cycle through all phases equally."""
    # Change phase every 30 steps (30 seconds with 1s step length)
    phase = (step // 30) % 4
    return np.array([phase for _ in range(num_junctions)])


def random_policy(obs, step, num_junctions):
    """Random baseline: random phase selection."""
    return np.array([random.randint(0, 3) for _ in range(num_junctions)])


def actuated_like_policy(obs, step, num_junctions):
    """Actuated-like policy: favor phases with higher traffic."""
    # Simple heuristic: if bottleneck ratio is high, use phase 1 or 3 (side streets)
    # Otherwise use main directions (0 or 2)
    # This is a simplified version without actual detector logic
    if step % 60 < 40:  # 40s for main directions
        return np.array([0 if i % 2 == 0 else 2 for i in range(num_junctions)])
    else:  # 20s for side streets
        return np.array([1 if i % 2 == 0 else 3 for i in range(num_junctions)])


def run_policy_test(env, policy_fn, policy_name, num_episodes=5):
    """Run policy and collect reward + statistics.

    Args:
        env: TrafficControlEnv instance
        policy_fn: Policy function taking (obs, step, num_junctions) → action
        policy_name: Name of the policy for logging
        num_episodes: Number of episodes to run

    Returns:
        List of result dictionaries
    """
    results = []
    num_junctions = env.num_intersections

    print(f"\n{'='*80}")
    print(f"Testing Policy: {policy_name}")
    print(f"{'='*80}")

    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}...")

        obs, info = env.reset()
        done = False
        cumulative_reward = 0.0
        reward_history = []
        step_count = 0

        while not done:
            # Get action from policy
            action = policy_fn(obs, step_count, num_junctions)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cumulative_reward += reward
            reward_history.append(reward)
            step_count += 1

            # Progress indicator every 100 steps
            if step_count % 100 == 0:
                print(f"    Step {step_count}, Cumulative Reward: {cumulative_reward:.2f}", end='\r')

        print(f"    Episode completed in {step_count} steps. Final reward: {cumulative_reward:.2f}")

        # Get final statistics from SUMO
        stats = env.get_final_statistics()

        results.append({
            'policy': policy_name,
            'episode': episode,
            'cumulative_reward': cumulative_reward,
            'avg_step_reward': np.mean(reward_history) if reward_history else 0.0,
            'completion_rate': stats['completion_rate'],
            'avg_waiting_time': stats['avg_waiting_time'],
            'avg_time_loss': stats['avg_time_loss'],
            'throughput': stats['throughput'],
            'vehicles_arrived': stats['vehicles_arrived'],
            'vehicles_loaded': stats['vehicles_loaded'],
            'insertion_rate': stats['insertion_rate']
        })

    return results


def analyze_results(results_df):
    """Analyze if reward correlates with good outcomes.

    Args:
        results_df: DataFrame with policy results

    Returns:
        Tuple of (policy_summary, correlations)
    """
    print("\n" + "="*80)
    print("REWARD FUNCTION VALIDATION ANALYSIS")
    print("="*80)

    # Group by policy
    by_policy = results_df.groupby('policy').mean()

    print("\n" + "-"*80)
    print("Average Results by Policy:")
    print("-"*80)
    print(by_policy[['cumulative_reward', 'completion_rate', 'avg_waiting_time',
                     'throughput', 'vehicles_arrived']].to_string())

    # Rank policies by different metrics
    print("\n" + "="*80)
    print("Policy Rankings:")
    print("="*80)

    metrics = {
        'Cumulative Reward (what agent optimizes)': ('cumulative_reward', False),
        'Completion Rate (actual goal)': ('completion_rate', False),
        'Throughput (vehicles/hour)': ('throughput', False),
        'Avg Waiting Time (lower=better)': ('avg_waiting_time', True)
    }

    for name, (col, ascending) in metrics.items():
        ranked = by_policy.sort_values(col, ascending=ascending)
        print(f"\n{name}:")
        for i, (policy, row) in enumerate(ranked.iterrows(), 1):
            marker = "✓" if i == 1 else " "
            print(f"  {marker} {i}. {policy:20s} = {row[col]:8.2f}")

    # Correlation analysis
    print("\n" + "="*80)
    print("Correlation: Reward vs Actual Outcomes")
    print("="*80)

    correlations = {
        'Completion Rate': results_df[['cumulative_reward', 'completion_rate']].corr().iloc[0, 1],
        'Throughput': results_df[['cumulative_reward', 'throughput']].corr().iloc[0, 1],
        'Waiting Time': results_df[['cumulative_reward', 'avg_waiting_time']].corr().iloc[0, 1],
    }

    for metric, corr in correlations.items():
        if 'Waiting' in metric:
            # For waiting time, we want negative correlation (higher reward = lower waiting)
            status = "✓ GOOD" if corr < -0.5 else "⚠ WEAK"
        else:
            # For completion/throughput, we want positive correlation
            status = "✓ GOOD" if corr > 0.5 else "⚠ WEAK"
        print(f"  {metric:20s}: {corr:+.3f}  {status}")

    # Validity check
    print("\n" + "="*80)
    print("REWARD FUNCTION VALIDITY CHECK:")
    print("="*80)

    # Best policy by reward should have best completion/throughput
    best_by_reward = by_policy['cumulative_reward'].idxmax()
    best_by_completion = by_policy['completion_rate'].idxmax()
    best_by_throughput = by_policy['throughput'].idxmax()
    best_by_waiting = by_policy['avg_waiting_time'].idxmin()  # Lower is better

    print(f"\nBest policy by different metrics:")
    print(f"  Cumulative Reward:  {best_by_reward}")
    print(f"  Completion Rate:    {best_by_completion}")
    print(f"  Throughput:         {best_by_throughput}")
    print(f"  Waiting Time:       {best_by_waiting}")

    # Check alignment
    matches = 0
    if best_by_reward == best_by_completion:
        matches += 1
    if best_by_reward == best_by_throughput:
        matches += 1

    print(f"\n{'='*80}")
    if matches >= 2:
        print("✓ PASS: Reward function aligns well with actual goals!")
        print(f"  The policy that maximizes reward also performs best on {matches}/2 key metrics.")
    elif matches == 1:
        print("⚠ PARTIAL: Reward aligns with some metrics but not all")
        print(f"  Consider adjusting reward weights to improve alignment.")
    else:
        print("✗ FAIL: Reward function does NOT align with actual goals!")
        print(f"  RECOMMENDATION: Adjust reward weights in src/rl/constants.py")
        print(f"  - If completion rate is most important, increase REWARD_THROUGHPUT_PER_VEHICLE")
        print(f"  - If waiting time is too high, increase REWARD_EXCESSIVE_WAITING_PENALTY")
        print(f"  - If speed matters more, increase REWARD_SPEED_REWARD_FACTOR")
    print(f"{'='*80}\n")

    return by_policy, correlations


def main():
    parser = argparse.ArgumentParser(description='Validate reward function with test policies')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per policy')
    parser.add_argument('--timesteps', type=int, default=3600, help='Simulation duration (seconds)')
    parser.add_argument('--vehicles', type=int, default=800, help='Number of vehicles')
    parser.add_argument('--grid-size', type=int, default=5, help='Grid dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("REWARD FUNCTION VALIDATION EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Grid Size:      {args.grid_size}x{args.grid_size}")
    print(f"  Vehicles:       {args.vehicles}")
    print(f"  Duration:       {args.timesteps}s ({args.timesteps/3600:.1f}h)")
    print(f"  Episodes/Policy: {args.episodes}")
    print(f"  Random Seed:    {args.seed}")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create environment (same config as training)
    env_args = (f"--grid_dimension {args.grid_size} "
                f"--num_vehicles {args.vehicles} "
                f"--end-time {args.timesteps} "
                f"--traffic_control rl "
                f"--seed {args.seed}")

    print(f"\nCreating environment...")
    env = TrafficControlEnv(env_args)

    # Test all policies
    all_results = []
    policies = [
        (greedy_policy, "Greedy_Throughput"),
        (fair_rotation_policy, "Fair_Rotation"),
        (actuated_like_policy, "Actuated_Like"),
        (random_policy, "Random_Baseline")
    ]

    for policy_fn, policy_name in policies:
        results = run_policy_test(env, policy_fn, policy_name, args.episodes)
        all_results.extend(results)

    # Analyze
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)
    df.to_csv('reward_function_validation.csv', index=False)
    print(f"\n✓ Raw results saved to: reward_function_validation.csv")

    by_policy, correlations = analyze_results(df)

    # Save analysis
    by_policy.to_csv('reward_function_analysis.csv')
    print(f"✓ Policy summary saved to: reward_function_analysis.csv")

    # Clean up
    env.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
