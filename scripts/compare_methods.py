#!/usr/bin/env python3
"""
Compare traffic control methods (tree_method, actuated, fixed) over multiple runs.
Generates a comparison_results text file with statistics and averages.
"""

import subprocess
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ComparisonRunner:
    """Runs traffic control method comparisons and collects statistics."""

    def __init__(self):
        self.methods = ['tree_method', 'actuated', 'fixed']
        self.num_runs = 20
        # Set to True to delete workspace folders after completion
        self.cleanup_workspaces = False
        self.private_seed_start = 72632
        self.public_seed_start = 27031
        self.seed_increment = 4535

        # Fixed simulation parameters
        self.base_params = [
            '--network-seed', '24208',
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
            '--num_vehicles', '22000',
            '--end-time', '7300',
            '--junctions_to_remove', '2'
        ]

        # Storage for results
        self.results: Dict[str, List[Dict[str, float]]] = {
            'tree_method': [],
            'actuated': [],
            'fixed': []
        }

    def run_simulation(self, method: str, private_seed: int, public_seed: int,
                       network_seed: int, run_number: int) -> Optional[Dict]:
        """Run a single simulation and extract statistics."""

        # Create workspace directory
        workspace_dir = Path(f'comparison_results_{method}_run_{run_number}')
        workspace_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'env', 'PYTHONUNBUFFERED=1',
            'python', '-m', 'src.cli',
            '--traffic_control', method,
            '--private-traffic-seed', str(private_seed),
            '--public-traffic-seed', str(public_seed),
            '--workspace', str(workspace_dir)
        ] + self.base_params

        try:
            print(f"  Running simulation... ", end='', flush=True)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"FAILED (exit code {result.returncode})")
                print(f"  Error: {result.stderr[:200]}")
                return None

            # Parse output for statistics
            output = result.stdout + result.stderr
            stats = self.parse_statistics(output)

            if stats:
                print("SUCCESS")
                return {
                    'stats': stats,
                    'seeds': {
                        'network': network_seed,
                        'private': private_seed,
                        'public': public_seed
                    }
                }
            else:
                print("FAILED (could not parse statistics)")
                return None

        except subprocess.TimeoutExpired:
            print("FAILED (timeout)")
            return None
        except Exception as e:
            print(f"FAILED ({str(e)})")
            return None

    def parse_statistics(self, output: str) -> Optional[Dict[str, float]]:
        """Extract statistics from simulation output."""
        stats = {}

        # Patterns to match statistics in output
        # Format: "Throughput: 6 veh/h", "Vehicles arrived: 2", etc.
        patterns = {
            'throughput': r'Throughput:\s*([0-9.]+)\s*veh/h',
            'vehicles_arrived': r'Vehicles\s+arrived:\s*([0-9]+)',
            'avg_duration': r'Average\s+duration:\s*([0-9.]+)s',
            'avg_waiting_time': r'Average\s+waiting\s+time:\s*([0-9.]+)s'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                stats[key] = float(match.group(1))

        # Return stats only if we found all metrics
        if len(stats) == 4:
            return stats
        return None

    def run_all_comparisons(self):
        """Run all comparisons for all methods."""
        total_runs = len(self.methods) * self.num_runs
        current_run = 0

        # Extract network seed from base_params
        network_seed = None
        for i in range(len(self.base_params) - 1):
            if self.base_params[i] == '--network-seed':
                network_seed = int(self.base_params[i + 1])
                break

        print(f"\n{'='*70}")
        print(f"Starting Traffic Control Method Comparison")
        print(f"{'='*70}")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Runs per method: {self.num_runs}")
        print(f"Total simulations: {total_runs}")
        print(f"{'='*70}\n")

        for method in self.methods:
            print(f"\n{'─'*70}")
            print(f"METHOD: {method.upper()}")
            print(f"{'─'*70}")

            for run in range(self.num_runs):
                current_run += 1
                private_seed = self.private_seed_start + \
                    (run * self.seed_increment)
                public_seed = self.public_seed_start + \
                    (run * self.seed_increment)

                progress = f"[{current_run}/{total_runs}]"
                print(
                    f"\n{progress} Run {run + 1}/{self.num_runs} - Seeds: network={network_seed}, private={private_seed}, public={public_seed}")

                result = self.run_simulation(
                    method, private_seed, public_seed, network_seed, run + 1)

                if result:
                    self.results[method].append(result)
                    stats = result['stats']
                    print(f"  → Throughput: {stats['throughput']:.2f}, "
                          f"Arrived: {int(stats['vehicles_arrived'])}, "
                          f"Duration: {stats['avg_duration']:.2f}s, "
                          f"Waiting: {stats['avg_waiting_time']:.2f}s")

        print(f"\n{'='*70}")
        print(f"All simulations completed!")
        print(f"{'='*70}\n")

    def calculate_averages(self, method: str) -> Dict[str, float]:
        """Calculate average statistics for a method."""
        if not self.results[method]:
            return {}

        num_results = len(self.results[method])
        averages = {
            'throughput': 0.0,
            'vehicles_arrived': 0.0,
            'avg_duration': 0.0,
            'avg_waiting_time': 0.0
        }

        for result in self.results[method]:
            stats = result['stats']
            for key in averages:
                averages[key] += stats[key]

        for key in averages:
            averages[key] /= num_results

        return averages

    def generate_report(self, output_file: str):
        """Generate text report with results."""
        # Parse base_params to extract values
        params_dict = {}
        for i in range(0, len(self.base_params), 2):
            if i + 1 < len(self.base_params):
                key = self.base_params[i].lstrip(
                    '-').replace('_', ' ').replace('-', ' ')
                params_dict[key] = self.base_params[i + 1]

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAFFIC CONTROL METHOD COMPARISON RESULTS\n")
            f.write("="*80 + "\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Runs per method: {self.num_runs}\n")
            f.write(
                f"Network seed: {params_dict.get('network seed', 'N/A')}\n")
            f.write(
                f"Grid dimension: {params_dict.get('grid dimension', 'N/A')}\n")
            f.write(
                f"Block size (m): {params_dict.get('block size m', 'N/A')}\n")
            f.write(f"Lane count: {params_dict.get('lane count', 'N/A')}\n")
            f.write(f"Step length: {params_dict.get('step length', 'N/A')}\n")
            f.write(
                f"Land use block size (m): {params_dict.get('land use block size m', 'N/A')}\n")
            f.write(
                f"Attractiveness: {params_dict.get('attractiveness', 'N/A')}\n")
            f.write(
                f"Traffic light strategy: {params_dict.get('traffic light strategy', 'N/A')}\n")
            f.write(
                f"Routing strategy: {params_dict.get('routing strategy', 'N/A')}\n")
            f.write(
                f"Vehicle types: {params_dict.get('vehicle types', 'N/A')}\n")
            f.write(
                f"Passenger routes: {params_dict.get('passenger routes', 'N/A')}\n")
            f.write(
                f"Public routes: {params_dict.get('public routes', 'N/A')}\n")
            f.write(
                f"Departure pattern: {params_dict.get('departure pattern', 'N/A')}\n")
            f.write(
                f"Start time (hour): {params_dict.get('start time hour', 'N/A')}\n")
            f.write(
                f"Number of vehicles: {params_dict.get('num vehicles', 'N/A')}\n")
            f.write(f"End time (s): {params_dict.get('end time', 'N/A')}\n")
            f.write(f"Private traffic seed range: {self.private_seed_start} - "
                    f"{self.private_seed_start + (self.num_runs-1)*self.seed_increment} "
                    f"(increment: {self.seed_increment})\n")
            f.write(f"Public traffic seed range: {self.public_seed_start} - "
                    f"{self.public_seed_start + (self.num_runs-1)*self.seed_increment} "
                    f"(increment: {self.seed_increment})\n")
            f.write("="*80 + "\n\n")

            # Combined table showing all methods side-by-side for each run
            f.write("\n" + "─"*80 + "\n")
            f.write("DETAILED RUN COMPARISON\n")
            f.write("─"*80 + "\n\n")

            # Check if we have results for all methods
            max_runs = max(len(self.results[method])
                           for method in self.methods)

            if max_runs > 0:
                # Table header
                f.write(f"{'Run':<5} ")
                f.write(f"{'Net Seed':<10} {'Priv Seed':<10} {'Pub Seed':<10} ")
                f.write(f"{'Fix Thr':<10} {'Fix Dur':<10} ")
                f.write(f"{'Act Thr':<10} {'Act Dur':<10} ")
                f.write(f"{'Tree Thr':<10} {'Tree Dur':<10}\n")

                f.write(f"{'='*5} ")
                f.write(f"{'='*10} {'='*10} {'='*10} ")
                f.write(f"{'='*10} {'='*10} ")
                f.write(f"{'='*10} {'='*10} ")
                f.write(f"{'='*10} {'='*10}\n")

                # Data rows - one row per run showing all three methods
                for run_idx in range(max_runs):
                    f.write(f"{run_idx + 1:<5} ")

                    # Get seeds from first available method result
                    seeds = None
                    for method in self.methods:
                        if run_idx < len(self.results[method]):
                            seeds = self.results[method][run_idx]['seeds']
                            break

                    # Write seeds
                    if seeds:
                        f.write(
                            f"{seeds['network']:<10} {seeds['private']:<10} {seeds['public']:<10} ")
                    else:
                        f.write(f"{'N/A':<10} {'N/A':<10} {'N/A':<10} ")

                    # Fixed results
                    if run_idx < len(self.results['fixed']):
                        stats = self.results['fixed'][run_idx]['stats']
                        f.write(
                            f"{stats['throughput']:<10.0f} {stats['avg_duration']:<10.1f} ")
                    else:
                        f.write(f"{'N/A':<10} {'N/A':<10} ")

                    # Actuated results
                    if run_idx < len(self.results['actuated']):
                        stats = self.results['actuated'][run_idx]['stats']
                        f.write(
                            f"{stats['throughput']:<10.0f} {stats['avg_duration']:<10.1f} ")
                    else:
                        f.write(f"{'N/A':<10} {'N/A':<10} ")

                    # Tree method results
                    if run_idx < len(self.results['tree_method']):
                        stats = self.results['tree_method'][run_idx]['stats']
                        f.write(
                            f"{stats['throughput']:<10.0f} {stats['avg_duration']:<10.1f}")
                    else:
                        f.write(f"{'N/A':<10} {'N/A':<10}")

                    f.write("\n")

                f.write("\n")
            else:
                f.write("No successful runs for any method\n\n")

            # Summary comparison
            f.write("\n" + "="*80 + "\n")
            f.write("AVERAGE STATISTICS COMPARISON\n")
            f.write("="*80 + "\n\n")

            # Calculate averages
            averages = {}
            for method in self.methods:
                averages[method] = self.calculate_averages(method)

            # Comparison table - only Throughput and Duration
            f.write(f"{'Method':<15} {'Throughput':<15} {'Avg Duration':<15}\n")
            f.write(f"{'='*15} {'='*15} {'='*15}\n")

            for method in self.methods:
                if averages[method]:
                    avg = averages[method]
                    f.write(f"{method:<15} "
                            f"{avg['throughput']:<15.2f} "
                            f"{avg['avg_duration']:<15.2f}\n")
                else:
                    f.write(f"{method:<15} {'N/A':<15} {'N/A':<15}\n")

            f.write("\n")

            # Relative performance comparison (if tree_method has results)
            if averages.get('tree_method'):
                f.write("\n" + "─"*80 + "\n")
                f.write("RELATIVE PERFORMANCE (compared to Tree Method)\n")
                f.write("─"*80 + "\n\n")

                tree_avg = averages['tree_method']

                f.write(
                    f"{'Method':<15} {'Throughput':<25} {'Avg Duration':<25}\n")
                f.write(f"{'='*15} {'='*25} {'='*25}\n")

                for method in ['fixed', 'actuated']:
                    if averages.get(method):
                        avg = averages[method]

                        # Calculate percentage differences
                        throughput_diff = (
                            (avg['throughput'] - tree_avg['throughput']) / tree_avg['throughput'] * 100)
                        duration_diff = (
                            (avg['avg_duration'] - tree_avg['avg_duration']) / tree_avg['avg_duration'] * 100)

                        f.write(f"{method:<15} "
                                f"{throughput_diff:>+8.2f}%{'':<16} "
                                f"{duration_diff:>+8.2f}%{'':<16}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")


def main():
    """Main entry point."""
    runner = ComparisonRunner()

    try:
        # Run all comparisons
        runner.run_all_comparisons()

        # Generate report in current directory
        output_file = Path.cwd() / 'comparison_results'
        runner.generate_report(str(output_file))

        print(f"\n✓ Results saved to: {output_file}")
        print(f"\nSummary:")
        for method in runner.methods:
            num_successful = len(runner.results[method])
            print(f"  {method}: {num_successful}/{runner.num_runs} successful runs")

        # Cleanup workspace folders if requested
        if runner.cleanup_workspaces:
            print(f"\nCleaning up workspace folders...")
            import shutil
            for method in runner.methods:
                for run in range(runner.num_runs):
                    workspace_dir = Path(
                        f'comparison_results_{method}_run_{run + 1}')
                    if workspace_dir.exists():
                        shutil.rmtree(workspace_dir)
                        print(f"  Deleted: {workspace_dir}")
            print("✓ Cleanup complete")
        else:
            print(f"\nWorkspace folders preserved for inspection.")
            print(f"Set cleanup_workspaces=True in the script to auto-delete them.")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
