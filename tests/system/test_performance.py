"""
Performance and regression tests.

These tests validate performance characteristics and detect
regressions in simulation speed and output quality.
"""

import pytest
import time
import json
from pathlib import Path
from tests.utils.test_helpers import run_cli_command, get_simulation_metrics, compare_metrics
from tests.utils.assertions import SystemTestAssertions

def get_workspace_dir():
    """Get the actual workspace directory."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "workspace"


class TestPerformanceBaselines:
    """Test performance against established baselines."""
    
    @pytest.mark.scenario
    def test_3x3_grid_performance_baseline(self):
        """Test 3x3 grid performance against golden master."""
        start_time = time.time()
        
        result = run_cli_command([
            "--grid_dimension", "3",
            "--block_size_m", "100",
            "--num_vehicles", "50",
            "--end-time", "60",
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"Performance test failed: {result.stderr}"
        assert execution_time < 30, f"Test took too long: {execution_time:.1f}s (max 30s)"
        
        # Load golden master
        fixtures_dir = Path(__file__).parent / "fixtures"
        golden_master_file = fixtures_dir / "golden_master_3x3.json"
        
        if golden_master_file.exists():
            with open(golden_master_file) as f:
                golden_master = json.load(f)
            
            metrics = get_simulation_metrics(get_workspace_dir())
            baseline = golden_master["baseline_metrics"]
            
            # Performance should be within 15% of baseline
            assert compare_metrics(metrics, baseline, tolerance=0.15), (
                f"Performance regression detected: {metrics} vs baseline {baseline}"
            )

    @pytest.mark.scenario  
    def test_5x5_grid_performance_baseline(self):
        """Test 5x5 grid performance against golden master."""
        start_time = time.time()
        
        result = run_cli_command([
            "--grid_dimension", "5",
            "--block_size_m", "150",
            "--junctions_to_remove", "1",
            "--num_vehicles", "200",
            "--end-time", "300",
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"Performance test failed: {result.stderr}"
        assert execution_time < 120, f"Test took too long: {execution_time:.1f}s (max 2min)"
        
        # Load golden master and compare
        fixtures_dir = Path(__file__).parent / "fixtures"
        golden_master_file = fixtures_dir / "golden_master_5x5.json"
        
        if golden_master_file.exists():
            with open(golden_master_file) as f:
                golden_master = json.load(f)
            
            metrics = get_simulation_metrics(get_workspace_dir())
            baseline = golden_master["baseline_metrics"]
            
            assertions = SystemTestAssertions(get_workspace_dir())
            assertions.assert_performance_regression(baseline, tolerance=0.15)


class TestTrafficControlPerformance:
    """Test relative performance of traffic control methods."""
    
    @pytest.mark.scenario
    @pytest.mark.parametrize("control_method", ["tree_method", "actuated", "fixed"])
    def test_traffic_control_performance(self, control_method):
        """Test performance of different traffic control methods."""
        start_time = time.time()
        
        result = run_cli_command([
            "--grid_dimension", "4",
            "--num_vehicles", "150",
            "--end-time", "240",     # 4 minutes
            "--traffic_control", control_method,
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"{control_method} failed: {result.stderr}"
        assert execution_time < 90, f"{control_method} too slow: {execution_time:.1f}s"
        
        metrics = get_simulation_metrics(get_workspace_dir())
        
        # Store performance metrics for comparison
        perf_data = {
            "control_method": control_method,
            "execution_time": execution_time,
            "metrics": metrics
        }
        
        # Write to file for cross-test comparison
        perf_file = get_workspace_dir() / f"performance_{control_method}.json"
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f, indent=2)

    @pytest.mark.scenario
    def test_tree_method_vs_baseline_performance(self):
        """Test Tree Method performance improvement vs baseline methods."""
        # Test with identical conditions across methods
        base_args = [
            "--grid_dimension", "4",
            "--num_vehicles", "200",
            "--end-time", "300",
            "--seed", "123"  # Fixed seed for fair comparison
        ]
        
        methods_results = {}
        
        # Test each method
        for method in ["tree_method", "actuated", "fixed"]:
            args = base_args + ["--traffic_control", method]
            
            start_time = time.time()
            result = run_cli_command(args)
            execution_time = time.time() - start_time
            
            assert result.returncode == 0, f"{method} failed: {result.stderr}"
            
            metrics = get_simulation_metrics(get_workspace_dir())
            methods_results[method] = {
                "execution_time": execution_time,
                "metrics": metrics
            }
            
            # Clean workspace for next method
            workspace = get_workspace_dir()
            for file in workspace.glob("*.xml"):
                file.unlink()
            for file in workspace.glob("*.cfg"):
                file.unlink()
        
        # Compare Tree Method against baselines
        tree_method_travel_time = methods_results["tree_method"]["metrics"]["average_travel_time"]
        actuated_travel_time = methods_results["actuated"]["metrics"]["average_travel_time"]
        fixed_travel_time = methods_results["fixed"]["metrics"]["average_travel_time"]
        
        # Tree Method should perform better (lower travel times) - avoid division by zero
        tree_vs_actuated_improvement = 0.0
        if actuated_travel_time > 0:
            tree_vs_actuated_improvement = (actuated_travel_time - tree_method_travel_time) / actuated_travel_time
            
        tree_vs_fixed_improvement = 0.0
        if fixed_travel_time > 0:
            tree_vs_fixed_improvement = (fixed_travel_time - tree_method_travel_time) / fixed_travel_time
        
        # Store results for analysis
        comparison_results = {
            "tree_vs_actuated_improvement": tree_vs_actuated_improvement,
            "tree_vs_fixed_improvement": tree_vs_fixed_improvement,
            "methods_results": methods_results
        }
        
        results_file = get_workspace_dir() / "performance_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Performance expectations (Tree Method should show improvement)
        print(f"Tree Method vs Actuated improvement: {tree_vs_actuated_improvement:.1%}")
        print(f"Tree Method vs Fixed improvement: {tree_vs_fixed_improvement:.1%}")


class TestScalabilityPerformance:
    """Test performance scaling with network size and vehicle count."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("grid_size", [3, 4, 5])
    def test_network_size_scaling(self, grid_size):
        """Test performance scaling with network size."""
        # Scale vehicles with network size
        vehicles_per_grid = {3: 50, 4: 100, 5: 200}
        num_vehicles = vehicles_per_grid[grid_size]
        
        start_time = time.time()
        
        result = run_cli_command([
            "--grid_dimension", str(grid_size),
            "--num_vehicles", str(num_vehicles),
            "--end-time", "120",     # Fixed simulation time
            "--traffic_control", "tree_method",
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"Grid {grid_size}x{grid_size} failed: {result.stderr}"
        
        # Performance bounds based on network size
        max_times = {3: 45, 4: 75, 5: 120}  # seconds
        assert execution_time < max_times[grid_size], (
            f"Grid {grid_size}x{grid_size} too slow: {execution_time:.1f}s "
            f"(max {max_times[grid_size]}s)"
        )
        
        metrics = get_simulation_metrics(get_workspace_dir())
        
        # Store scaling data
        scaling_data = {
            "grid_size": grid_size,
            "num_vehicles": num_vehicles,
            "execution_time": execution_time,
            "metrics": metrics
        }
        
        scaling_file = get_workspace_dir() / f"scaling_grid_{grid_size}.json"
        with open(scaling_file, 'w') as f:
            json.dump(scaling_data, f, indent=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("vehicle_count", [100, 200, 400])
    def test_vehicle_count_scaling(self, vehicle_count):
        """Test performance scaling with vehicle count."""
        start_time = time.time()
        
        result = run_cli_command([
            "--grid_dimension", "5",  # Fixed network size
            "--num_vehicles", str(vehicle_count),
            "--end-time", "180",      # 3 minutes
            "--traffic_control", "tree_method",
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"{vehicle_count} vehicles failed: {result.stderr}"
        
        # Performance should scale reasonably with vehicle count
        max_time_per_vehicle = 0.3  # 0.3 seconds per vehicle max
        max_time = max(60, vehicle_count * max_time_per_vehicle)
        
        assert execution_time < max_time, (
            f"{vehicle_count} vehicles too slow: {execution_time:.1f}s "
            f"(max {max_time:.1f}s)"
        )


class TestMemoryPerformance:
    """Test memory usage and resource efficiency."""
    
    @pytest.mark.scenario
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = run_cli_command([
            "--grid_dimension", "5", 
            "--num_vehicles", "300",
            "--end-time", "240",
            "--traffic_control", "tree_method",
            "--seed", "42"
        ])
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        assert result.returncode == 0, f"Memory test failed: {result.stderr}"
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.1f}MB increase"

    @pytest.mark.scenario
    def test_tree_method_sample_performance(self, sample_networks_path):
        """Test Tree Method sample performance."""
        start_time = time.time()
        
        result = run_cli_command([
            "--tree_method_sample", sample_networks_path,
            "--traffic_control", "tree_method",
            "--end-time", "120",     # 2 minutes instead of full simulation
            "--seed", "42"
        ])
        
        execution_time = time.time() - start_time
        
        assert result.returncode == 0, f"Sample performance test failed: {result.stderr}"
        assert execution_time < 90, f"Sample test too slow: {execution_time:.1f}s"
        
        assertions = SystemTestAssertions(get_workspace_dir())
        assertions.assert_simulation_completed_successfully()
        assertions.assert_sample_network_imported()