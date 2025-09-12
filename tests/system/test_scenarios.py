"""
Main scenario tests for system validation.

These tests validate complete pipeline execution using shortened versions
of the proven scenarios from CLAUDE.md. Each test runs a full pipeline
but with reduced parameters for faster execution.
"""

import pytest
from pathlib import Path
import subprocess
import os
from tests.utils.test_helpers import run_cli_command, get_simulation_metrics
from tests.utils.assertions import SystemTestAssertions


def get_workspace_assertions():
    """Helper to get SystemTestAssertions with correct workspace."""
    project_root = Path(__file__).parent.parent.parent
    workspace_dir = project_root / "workspace"
    return SystemTestAssertions(workspace_dir)


class TestSyntheticGridScenarios:
    """Test synthetic grid network scenarios."""

    @pytest.mark.smoke
    def test_minimal_grid_smoke(self):
        """
        Smoke test: Minimal 3x3 grid, 50 vehicles, 1 minute.

        Validates basic pipeline functionality without errors.
        """
        result = run_cli_command([
            "--grid_dimension", "3",
            "--block_size_m", "100",
            "--num_vehicles", "50",
            "--end-time", "60",
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Validate basic file generation
        project_root = Path(__file__).parent.parent.parent
        workspace_dir = project_root / "workspace"

        essential_files = ["grid.net.xml", "vehicles.rou.xml", "grid.sumocfg"]
        for filename in essential_files:
            filepath = workspace_dir / filename
            assert filepath.exists(), f"Missing essential file: {filename}"

        # Check simulation metrics
        metrics = get_simulation_metrics(workspace_dir)
        assert metrics["vehicles_departed"] > 0
        assert metrics["simulation_completed"] is True

    @pytest.mark.scenario
    def test_morning_rush_scenario(self):
        """
        Scenario 1 (Modified): Morning rush hour pattern.

        Based on CLAUDE.md Scenario 1 but shortened for testing.
        """
        result = run_cli_command([
            "--grid_dimension", "5",
            "--block_size_m", "150",
            "--junctions_to_remove", "1",
            "--num_vehicles", "200",  # Reduced from 800
            "--step-length", "1.0",
            "--departure_pattern", "six_periods",
            "--start_time_hour", "7.0",
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # System assertions
        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_all_files_generated()
        assertions.assert_network_properties(
            expected_edges_min=40, expected_junctions_min=20)
        assertions.assert_vehicle_metrics_within_bounds(
            min_departed=150, max_travel_time=600)

    @pytest.mark.scenario
    def test_evening_light_traffic(self):
        """
        Scenario 2 (Modified): Light evening traffic.

        Tests uniform departure pattern with evening start time.
        """
        result = run_cli_command([
            "--grid_dimension", "5",
            "--block_size_m", "150",
            "--junctions_to_remove", "0",
            "--num_vehicles", "150",  # Reduced from 500
            "--step-length", "1.0",
            "--end-time", "300",      # 5 minutes instead of 1.5 hours
            "--departure_pattern", "uniform",
            "--start_time_hour", "20.0",
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # System assertions
        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_vehicle_metrics_within_bounds(min_departed=100)

    @pytest.mark.scenario
    def test_multi_modal_traffic_mix(self):
        """
        Scenario 10 (Modified): Multi-modal traffic with vehicle types.

        Tests vehicle type distribution and routing strategies.
        """
        result = run_cli_command([
            "--grid_dimension", "5",
            "--block_size_m", "150",
            "--junctions_to_remove", "1",
            "--num_vehicles", "200",  # Reduced from 850
            "--step-length", "1.0",
            "--end-time", "300",      # 5 minutes instead of 4.5 hours
            "--departure_pattern", "uniform",
            "--vehicle_types", "passenger 50 commercial 40 public 10",
            "--attractiveness", "poisson",
            "--routing_strategy", "shortest 70 realtime 30",
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_vehicle_types_generated(
            ["passenger", "commercial", "public"])


class TestTreeMethodSample:
    """Test Tree Method sample scenarios."""

    @pytest.mark.scenario
    def test_tree_method_sample_basic(self):
        """
        Tree Method Sample Test: Pre-built network validation.

        Uses Tree Method sample data from evaluation/datasets/networks/
        """
        result = run_cli_command([
            "--tree_method_sample", "evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1/",
            "--traffic_control", "tree_method",
            "--end-time", "180",      # 3 minutes instead of full 2 hours
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        # Tree Method samples don't generate all pipeline files, just copied network files
        essential_files = ["grid.net.xml", "vehicles.rou.xml", "grid.sumocfg"]
        assertions.assert_all_files_generated(essential_files)
        assertions.assert_tree_method_active()

    @pytest.mark.scenario
    @pytest.mark.parametrize("traffic_control", ["tree_method", "actuated", "fixed"])
    def test_tree_method_sample_comparison(self, traffic_control):
        """
        Compare traffic control methods on Tree Method sample network.

        Tests all three control methods on identical pre-built network.
        """
        result = run_cli_command([
            "--tree_method_sample", "evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1/",
            "--traffic_control", traffic_control,
            "--end-time", "120",      # 2 minutes for comparison
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_traffic_control_applied(traffic_control)


class TestTrafficControlComparison:
    """Test different traffic control methods."""

    @pytest.mark.scenario
    @pytest.mark.parametrize("traffic_control", ["fixed", "actuated", "tree_method"])
    def test_traffic_control_methods(self, traffic_control):
        """
        Compare traffic control methods with identical conditions.

        Validates that all control methods work without errors.
        """
        result = run_cli_command([
            "--grid_dimension", "5",
            "--num_vehicles", "200",
            "--end-time", "180",      # 3 minutes
            "--traffic_control", traffic_control,
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_traffic_control_applied(traffic_control)

    @pytest.mark.scenario
    def test_tree_method_integration(self):
        """
        Specific test for Tree Method integration.

        Ensures Tree Method algorithm integrates correctly with pipeline.
        """
        result = run_cli_command([
            "--grid_dimension", "4",
            "--num_vehicles", "150",
            "--end-time", "240",      # 4 minutes
            "--traffic_control", "tree_method",
            "--routing_strategy", "shortest 60 realtime 40",
            "--seed", "42"
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        assertions = get_workspace_assertions()
        assertions.assert_simulation_completed_successfully()
        assertions.assert_tree_method_active()


class TestRegressionScenarios:
    """Regression tests for performance and consistency."""

    @pytest.mark.scenario
    def test_performance_bounds_regression(self):
        """
        Regression test for performance metrics.

        Ensures performance stays within expected bounds over time.
        """
        result = run_cli_command([
            "--grid_dimension", "5",
            "--num_vehicles", "300",
            "--end-time", "300",
            "--traffic_control", "tree_method",
            "--seed", "42"  # Fixed seed for reproducibility
        ])

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        project_root = Path(__file__).parent.parent.parent
        workspace_dir = project_root / "workspace"
        metrics = get_simulation_metrics(workspace_dir)

        # Performance bounds (adjust based on baseline measurements)
        assert metrics["average_travel_time"] < 400, "Travel time regression detected"
        assert metrics["completion_rate"] > 0.7, "Completion rate regression detected"
        assert metrics["vehicles_departed"] > 250, "Departure rate regression detected"

    @pytest.mark.scenario
    def test_reproducibility(self):
        """
        Test simulation reproducibility with fixed seed.

        Runs same scenario twice and compares key metrics.
        """
        base_args = [
            "--grid_dimension", "4",
            "--num_vehicles", "100",
            "--end-time", "180",
            "--seed", "12345"
        ]

        # First run
        result1 = run_cli_command(base_args)
        assert result1.returncode == 0
        project_root = Path(__file__).parent.parent.parent
        workspace_dir = project_root / "workspace"
        metrics1 = get_simulation_metrics(workspace_dir)

        # Clean workspace
        for file in workspace_dir.glob("*.xml"):
            file.unlink()
        for file in workspace_dir.glob("*.cfg"):
            file.unlink()

        # Second run
        result2 = run_cli_command(base_args)
        assert result2.returncode == 0
        metrics2 = get_simulation_metrics(workspace_dir)

        # Compare key metrics (should be identical with same seed)
        assert metrics1["vehicles_departed"] == metrics2["vehicles_departed"]
        assert abs(metrics1["average_travel_time"] -
                   metrics2["average_travel_time"]) < 1.0
