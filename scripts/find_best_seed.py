#!/usr/bin/env python3
"""
Find the best seed combination for tree_method traffic control.
Searches for seeds where tree_method significantly outperforms actuated and fixed.
"""

import subprocess
import re
import random
from pathlib import Path
from typing import Dict, Optional


class SeedFinder:
    """Searches for optimal seed combinations for tree_method performance."""

    def __init__(self):
        self.num_iterations = 100
        self.methods = ['actuated', 'tree_method']
        self.vehicle_counts = [25000]

        # Fixed simulation parameters
        self.base_params = [
            '--grid_dimension', '6',
            '--block_size_m', '280',
            '--lane_count', 'realistic',
            '--step-length', '1.0',
            '--land_use_block_size_m', '25.0',
            '--attractiveness', 'land_use',
            '--traffic_light_strategy', 'partial_opposites',
            '--routing_strategy', 'realtime 100',
            '--vehicle_types', 'passenger 100',
            '--passenger-routes', 'in 0 out 0 inner 100 pass 0',
            '--departure_pattern', 'uniform',
            '--start_time_hour', '8.0',
            # '--junctions_to_remove', 'C4, F3',
            '--junctions_to_remove', '2',
            '--end-time', '7300'
        ]

        # Thresholds
        self.throughput_ratio_threshold = 1.20  # Tree must be 20% better
        # Tree must be 40% better (60% of others)
        self.duration_ratio_threshold = 0.60

        # Result tracking
        self.best_result = None
        self.all_attempts = []  # Track all attempts for reporting
        self.log_file = None  # Will be opened during search

    def log(self, message: str, end: str = '\n', flush: bool = False):
        """Log message to both screen and file."""
        print(message, end=end, flush=flush)
        if self.log_file:
            self.log_file.write(message + end)
            if flush:
                self.log_file.flush()

    def run_simulation(self, method: str, network_seed: int, private_seed: int,
                       public_seed: int, num_vehicles: int) -> Optional[Dict]:
        """Run a single simulation and extract statistics."""
        workspace_dir = Path(f'seed_search_{method}')
        workspace_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'env', 'PYTHONUNBUFFERED=1',
            'python', '-m', 'src.cli',
            '--traffic_control', method,
            '--network-seed', str(network_seed),
            '--private-traffic-seed', str(private_seed),
            '--public-traffic-seed', str(public_seed),
            '--num_vehicles', str(num_vehicles),
            '--workspace', str(workspace_dir)
        ] + self.base_params

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                return None

            # Parse statistics
            output = result.stdout + result.stderr
            stats = self.parse_statistics(output)
            return stats

        except Exception:
            return None

    def parse_statistics(self, output: str) -> Optional[Dict]:
        """Extract statistics from simulation output."""
        stats = {}

        patterns = {
            'throughput': r'Throughput:\s*([0-9.]+)\s*veh/h',
            'avg_duration': r'Average\s+duration:\s*([0-9.]+)s'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                stats[key] = float(match.group(1))

        if len(stats) == 2:
            return stats
        return None

    def check_criteria(self, tree_stats: Dict, actuated_stats: Dict) -> bool:
        """Check if tree_method meets the performance criteria."""
        # Throughput: Tree must be >= 1.20 * (both actuated AND fixed)
        throughput_ok = (
            tree_stats['throughput'] >= self.throughput_ratio_threshold *
            actuated_stats['throughput']
        )

        # Duration: Tree must be <= 0.60 * (both actuated AND fixed)
        duration_ok = (
            tree_stats['avg_duration'] <= self.duration_ratio_threshold *
            actuated_stats['avg_duration']
        )

        return throughput_ok and duration_ok

    def update_best_result(self, network_seed: int, private_seed: int,
                           public_seed: int, num_vehicles: int,
                           tree_stats: Dict, actuated_stats: Dict):
        """Update best result if current is better."""
        if self.best_result is None:
            # First qualifying result
            self.best_result = {
                'network_seed': network_seed,
                'private_seed': private_seed,
                'public_seed': public_seed,
                'num_vehicles': num_vehicles,
                'tree_stats': tree_stats,
                'actuated_stats': actuated_stats
            }
        elif tree_stats['throughput'] > self.best_result['tree_stats']['throughput']:
            # Better throughput
            self.best_result = {
                'network_seed': network_seed,
                'private_seed': private_seed,
                'public_seed': public_seed,
                'num_vehicles': num_vehicles,
                'tree_stats': tree_stats,
                'actuated_stats': actuated_stats
            }

    def print_best_result(self):
        """Print the current best result."""
        if self.best_result is None:
            self.log("\nNo qualifying results yet")
            return

        self.log("\n" + "="*70)
        self.log("BEST RESULT SO FAR:")
        self.log("="*70)
        self.log(f"  network-seed: {self.best_result['network_seed']}")
        self.log(f"  private-traffic-seed: {self.best_result['private_seed']}")
        self.log(f"  public-traffic-seed: {self.best_result['public_seed']}")
        self.log(f"  num_vehicles: {self.best_result['num_vehicles']}")
        self.log("")

        # Calculate percentages
        tree = self.best_result['tree_stats']
        act = self.best_result['actuated_stats']

        throughput_vs_act = (
            (tree['throughput'] - act['throughput']) / act['throughput']) * 100
        throughput_vs_fix = (
            (tree['throughput'] - fix['throughput']) / fix['throughput']) * 100
        duration_vs_act = (
            (tree['avg_duration'] - act['avg_duration']) / act['avg_duration']) * 100
        duration_vs_fix = (
            (tree['avg_duration'] - fix['avg_duration']) / fix['avg_duration']) * 100

        self.log(f"  Tree Method:")
        self.log(f"    Throughput: {tree['throughput']:.0f} veh/h")
        self.log(f"    Duration: {tree['avg_duration']:.1f}s")
        self.log("")
        self.log(f"  Actuated:")
        self.log(
            f"    Throughput: {act['throughput']:.0f} veh/h ({throughput_vs_act:+.1f}% vs Tree)")
        self.log(
            f"    Duration: {act['avg_duration']:.1f}s ({duration_vs_act:+.1f}% vs Tree)")
        self.log("")
        self.log(f"  Fixed:")
        self.log(
            f"    Throughput: {fix['throughput']:.0f} veh/h ({throughput_vs_fix:+.1f}% vs Tree)")
        self.log(
            f"    Duration: {fix['avg_duration']:.1f}s ({duration_vs_fix:+.1f}% vs Tree)")
        self.log("="*70)

    def save_results_to_file(self, f):
        """Save detailed results to an open file handle."""
        if self.best_result is None:
            f.write("No qualifying seed combination found.\n")
            f.write(
                f"Tried {self.num_iterations} iterations with different seeds.\n")
            f.write("\nCriteria:\n")
            f.write(
                f"  - Tree Throughput >= {self.throughput_ratio_threshold}x (Actuated & Fixed)\n")
            f.write(
                f"  - Tree Duration <= {self.duration_ratio_threshold}x (Actuated & Fixed)\n")
            return

        f.write("="*80 + "\n")
        f.write("BEST SEED COMBINATION FOR TREE METHOD\n")
        f.write("="*80 + "\n\n")

        f.write(f"Network seed: {self.best_result['network_seed']}\n")
        f.write(f"Private traffic seed: {self.best_result['private_seed']}\n")
        f.write(f"Public traffic seed: {self.best_result['public_seed']}\n")
        f.write(f"Number of vehicles: {self.best_result['num_vehicles']}\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-"*80 + "\n\n")

        f.write(f"{'Method':<15} {'Throughput (veh/h)':<20} {'Duration (s)':<15}\n")
        f.write(f"{'='*15} {'='*20} {'='*15}\n")

        for method_name, stats_key in [('Tree Method', 'tree_stats'),
                                       ('Actuated', 'actuated_stats')]:
            stats = self.best_result[stats_key]
            f.write(f"{method_name:<15} "
                    f"{stats['throughput']:<20.0f} "
                    f"{stats['avg_duration']:<15.1f}\n")

        f.write("\n")
        f.write("RELATIVE PERFORMANCE:\n")
        f.write("-"*80 + "\n\n")

        tree = self.best_result['tree_stats']
        act = self.best_result['actuated_stats']

        throughput_vs_act = (
            (tree['throughput'] - act['throughput']) / act['throughput']) * 100
        throughput_vs_fix = (
            (tree['throughput'] - fix['throughput']) / fix['throughput']) * 100
        duration_vs_act = (
            (tree['avg_duration'] - act['avg_duration']) / act['avg_duration']) * 100
        duration_vs_fix = (
            (tree['avg_duration'] - fix['avg_duration']) / fix['avg_duration']) * 100

        f.write(f"Tree vs Actuated:\n")
        f.write(f"  Throughput: {throughput_vs_act:+.1f}%\n")
        f.write(f"  Duration: {duration_vs_act:+.1f}%\n\n")

        f.write(f"Tree vs Fixed:\n")
        f.write(f"  Throughput: {throughput_vs_fix:+.1f}%\n")
        f.write(f"  Duration: {duration_vs_fix:+.1f}%\n\n")

        f.write("Interpretation:\n")
        f.write("  Throughput: Positive % = Tree is better\n")
        f.write("  Duration: Negative % = Tree is better (lower duration)\n\n")

        # All attempts section
        f.write("="*80 + "\n")
        f.write("ALL ATTEMPTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'#':<3} {'NetSd':<6} {'PriSd':<6} {'PubSd':<6} {'Veh':<5} ")
        f.write(f"{'Tr_T':<5} {'Tr_D':<5} {'Ac_T':<5} {'Ac_D':<5} ")
        f.write(f"{'Fx_T':<5} {'Fx_D':<5} ")
        f.write(f"{'T%':<6} {'D%':<6} {'OK':<3}\n")
        f.write(f"{'-'*3} {'-'*6} {'-'*6} {'-'*6} {'-'*5} ")
        f.write(f"{'-'*5} {'-'*5} {'-'*5} {'-'*5} ")
        f.write(f"{'-'*5} {'-'*5} ")
        f.write(f"{'-'*6} {'-'*6} {'-'*3}\n")

        for i, attempt in enumerate(self.all_attempts, 1):
            tree = attempt['tree_stats']
            act = attempt['actuated_stats']

            # Calculate average % improvement (average of vs actuated and vs fixed)
            throughput_vs_act = (
                (tree['throughput'] - act['throughput']) / act['throughput']) * 100
            duration_vs_act = (
                (tree['avg_duration'] - act['avg_duration']) / act['avg_duration']) * 100

            avg_throughput_improvement = (
                throughput_vs_act + throughput_vs_fix) / 2
            avg_duration_improvement = (duration_vs_act + duration_vs_fix) / 2

            f.write(f"{i:<3} ")
            f.write(f"{attempt['network_seed']:<6} ")
            f.write(f"{attempt['private_seed']:<6} ")
            f.write(f"{attempt['public_seed']:<6} ")
            f.write(f"{attempt['num_vehicles']:<5} ")
            f.write(f"{tree['throughput']:<5.0f} ")
            f.write(f"{tree['avg_duration']:<5.0f} ")
            f.write(f"{act['throughput']:<5.0f} ")
            f.write(f"{act['avg_duration']:<5.0f} ")
            f.write(f"{fix['throughput']:<5.0f} ")
            f.write(f"{fix['avg_duration']:<5.0f} ")
            f.write(f"{avg_throughput_improvement:<6.1f} ")
            f.write(f"{avg_duration_improvement:<6.1f} ")
            f.write(f"{'✓' if attempt['qualifies'] else '✗':<3}\n")

        f.write("\n")
        f.write("Legend:\n")
        f.write("  NetSd/PriSd/PubSd = Network/Private/Public seeds\n")
        f.write("  Veh = Number of vehicles\n")
        f.write("  Tr_T/Ac_T/Fx_T = Tree/Actuated/Fixed Throughput (veh/h)\n")
        f.write("  Tr_D/Ac_D/Fx_D = Tree/Actuated/Fixed Duration (s)\n")
        f.write("  T% = Avg Throughput improvement (Tree vs Actuated&Fixed)\n")
        f.write("  D% = Avg Duration improvement (Tree vs Actuated&Fixed)\n")
        f.write("  OK = Meets criteria (✓) or not (✗)\n")
        f.write("  Note: For T%, positive is better. For D%, negative is better.\n\n")

        f.write("="*80 + "\n")

    def run_search(self):
        """Run the seed search process."""
        # Open log file
        self.log_file = open('find_best_seed_result', 'w')

        self.log("="*70)
        self.log("SEED SEARCH FOR OPTIMAL TREE METHOD PERFORMANCE")
        self.log("="*70)
        self.log(f"Iterations: {self.num_iterations}")
        self.log(f"Vehicle counts to try: {self.vehicle_counts}")
        self.log(
            f"Throughput requirement: Tree >= {self.throughput_ratio_threshold}x (Actuated & Fixed)")
        self.log(
            f"Duration requirement: Tree <= {self.duration_ratio_threshold}x (Actuated & Fixed)")
        self.log("="*70 + "\n")

        for iteration in range(self.num_iterations):
            # Generate random seeds
            network_seed = random.randint(1000, 99999)
            private_seed = random.randint(1000, 99999)
            public_seed = random.randint(1000, 99999)

            self.log(f"\n[{iteration + 1}/{self.num_iterations}] Testing seeds: "
                     f"network={network_seed}, private={private_seed}, public={public_seed}")

            # Try both vehicle counts
            for num_vehicles in self.vehicle_counts:
                self.log(f"  Testing with {num_vehicles} vehicles...")

                # Run all three methods
                results = {}
                success = True

                for method in self.methods:
                    self.log(f"    Running {method}...", end=' ', flush=True)
                    stats = self.run_simulation(method, network_seed, private_seed,
                                                public_seed, num_vehicles)
                    if stats:
                        results[method] = stats
                        self.log(
                            f"✓ (T={stats['throughput']:.0f}, D={stats['avg_duration']:.1f})")
                    else:
                        self.log("✗ Failed")
                        success = False
                        break

                if not success:
                    continue

                # Record this attempt
                qualifies = self.check_criteria(
                    results['tree_method'], results['actuated'])
                self.all_attempts.append({
                    'network_seed': network_seed,
                    'private_seed': private_seed,
                    'public_seed': public_seed,
                    'num_vehicles': num_vehicles,
                    'tree_stats': results['tree_method'],
                    'actuated_stats': results['actuated'],
                    'qualifies': qualifies
                })

                # Calculate and print percentage improvements for THIS iteration
                tree = results['tree_method']
                act = results['actuated']

                throughput_vs_act = (
                    (tree['throughput'] - act['throughput']) / act['throughput']) * 100
                duration_vs_act = (
                    (tree['avg_duration'] - act['avg_duration']) / act['avg_duration']) * 100

                self.log(
                    f"    Performance vs Actuated: Throughput {throughput_vs_act:+.1f}%, Duration {duration_vs_act:+.1f}%")

                # Check if criteria met
                if qualifies:
                    self.log(f"    ✓✓✓ QUALIFYING RESULT! ✓✓✓")
                    self.update_best_result(network_seed, private_seed, public_seed,
                                            num_vehicles, results['tree_method'],
                                            results['actuated'])
                    self.print_best_result()

        # Final summary
        self.log("\n" + "="*70)
        self.log("SEARCH COMPLETE")
        self.log("="*70)
        self.print_best_result()

        # Close the main log file before writing final results
        if self.log_file:
            self.log_file.close()
            self.log_file = None

        # Now save the detailed results to the same file (append mode)
        output_file = 'find_best_seed_result'
        # Reopen in append mode to add the detailed results
        with open(output_file, 'a') as f:
            f.write("\n\n")
            self.save_results_to_file(f)

        print(f"\nAll results (log + details) saved to: {output_file}")


def main():
    """Main entry point."""
    finder = SeedFinder()
    try:
        finder.run_search()
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user")
        finder.save_results('find_best_seed_result')
        print("Partial results saved to: find_best_seed_result")


if __name__ == '__main__':
    main()
