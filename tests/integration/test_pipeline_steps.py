"""
Integration tests for individual pipeline steps.

These tests validate that each step of the 7-step pipeline
works correctly in isolation and in sequence.
"""

import pytest
from pathlib import Path
import sys
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tests.utils.test_helpers import run_cli_command, get_simulation_metrics
from tests.utils.assertions import SystemTestAssertions


class TestNetworkGeneration:
    """Test Step 1: Network generation."""
    
    @pytest.mark.integration
    def test_synthetic_grid_generation(self, temp_workspace):
        """Test synthetic grid network generation (Step 1)."""
        result = run_cli_command([
            "--grid_dimension", "4",
            "--block_size_m", "120",
            "--num_vehicles", "1",  # Minimal vehicles for network-only test
            "--end-time", "1",      # Minimal simulation
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Network generation failed: {result.stderr}"
        
        # Check network files exist
        network_files = ["grid.net.xml", "grid.nod.xml", "grid.edg.xml", "grid.con.xml"]
        for filename in network_files:
            filepath = temp_workspace / filename
            assert filepath.exists(), f"Network file missing: {filename}"
            assert filepath.stat().st_size > 0, f"Empty network file: {filename}"
        
        # Validate network structure
        assertions = SystemTestAssertions(temp_workspace)
        assertions.assert_network_properties(expected_edges_min=20, expected_junctions_min=12)


class TestEdgeSplittingWithLanes:
    """Test Step 2: Integrated edge splitting with lane assignment."""
    
    @pytest.mark.integration
    def test_edge_splitting_realistic_lanes(self, temp_workspace):
        """Test edge splitting with realistic lane assignment."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--lane_count", "realistic",
            "--num_vehicles", "10",
            "--end-time", "30",
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Edge splitting failed: {result.stderr}"
        
        # Check that network has variable lane counts
        net_file = temp_workspace / "grid.net.xml"
        net_content = net_file.read_text()
        assert "numLanes=" in net_content, "No lane assignments found"
        
        # Should have some variety in lane counts with realistic mode
        assert 'numLanes="1"' in net_content or 'numLanes="2"' in net_content

    @pytest.mark.integration
    @pytest.mark.parametrize("lane_mode", ["fixed:2", "random", "realistic"])
    def test_lane_assignment_modes(self, temp_workspace, lane_mode):
        """Test different lane assignment modes."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--lane_count", lane_mode,
            "--num_vehicles", "5",
            "--end-time", "30",
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Lane assignment {lane_mode} failed: {result.stderr}"
        
        net_file = temp_workspace / "grid.net.xml"
        net_content = net_file.read_text()
        assert "numLanes=" in net_content, f"No lane assignments for {lane_mode}"


class TestTrafficGeneration:
    """Test Steps 4-6: Traffic generation pipeline."""
    
    @pytest.mark.integration
    def test_vehicle_route_generation(self, temp_workspace):
        """Test vehicle and route generation."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--routing_strategy", "shortest 100",
            "--vehicle_types", "passenger 80 commercial 20",
            "--end-time", "60",
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Traffic generation failed: {result.stderr}"
        
        # Check route file
        rou_file = temp_workspace / "vehicles.rou.xml"
        assert rou_file.exists(), "Route file not generated"
        
        rou_content = rou_file.read_text()
        assert "<vehicle" in rou_content, "No vehicles in route file"
        assert 'type="passenger"' in rou_content, "Passenger vehicles not found"
        assert 'type="commercial"' in rou_content, "Commercial vehicles not found"

    @pytest.mark.integration
    @pytest.mark.parametrize("departure_pattern", ["uniform", "six_periods"])
    def test_departure_patterns(self, temp_workspace, departure_pattern):
        """Test different departure patterns."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "15",
            "--departure_pattern", departure_pattern,
            "--end-time", "60",
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Departure pattern {departure_pattern} failed: {result.stderr}"
        
        rou_file = temp_workspace / "vehicles.rou.xml"
        rou_content = rou_file.read_text()
        assert 'depart=' in rou_content, f"No departure times for {departure_pattern}"


class TestTrafficLightInjection:
    """Test Step 5: Traffic light injection."""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("tl_strategy", ["opposites", "incoming"])
    def test_traffic_light_strategies(self, temp_workspace, tl_strategy):
        """Test different traffic light strategies."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--traffic_light_strategy", tl_strategy,
            "--num_vehicles", "10",
            "--end-time", "30",
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Traffic light strategy {tl_strategy} failed: {result.stderr}"
        
        # Check traffic light file
        tll_file = temp_workspace / "grid.tll.xml"
        assert tll_file.exists(), "Traffic light file not generated"
        
        tll_content = tll_file.read_text()
        assert "<tlLogic" in tll_content, "No traffic light logic found"


class TestSimulationExecution:
    """Test Step 7: Dynamic simulation execution."""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("traffic_control", ["fixed", "actuated", "tree_method"])
    def test_simulation_with_traffic_control(self, temp_workspace, traffic_control):
        """Test simulation with different traffic control methods."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "15",
            "--end-time", "90",     # 1.5 minutes
            "--traffic_control", traffic_control,
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Simulation with {traffic_control} failed: {result.stderr}"
        
        assertions = SystemTestAssertions(temp_workspace)
        assertions.assert_simulation_completed_successfully()
        assertions.assert_traffic_control_applied(traffic_control)


class TestTreeMethodSampleIntegration:
    """Test Tree Method sample integration."""
    
    @pytest.mark.integration
    @pytest.mark.tree_method_sample
    def test_sample_network_processing(self, temp_workspace, sample_networks_path):
        """Test Tree Method sample network processing."""
        result = run_cli_command([
            "--tree_method_sample", sample_networks_path,
            "--traffic_control", "tree_method",
            "--end-time", "60",     # 1 minute
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Sample processing failed: {result.stderr}"
        
        assertions = SystemTestAssertions(temp_workspace)
        assertions.assert_simulation_completed_successfully()
        assertions.assert_sample_network_imported()
        assertions.assert_tree_method_active()


class TestPipelineSequence:
    """Test complete pipeline sequence validation."""
    
    @pytest.mark.integration
    def test_complete_pipeline_synthetic(self, temp_workspace):
        """Test complete 7-step pipeline for synthetic networks."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--block_size_m", "100",
            "--junctions_to_remove", "0",
            "--lane_count", "realistic",
            "--num_vehicles", "25",
            "--routing_strategy", "shortest 70 realtime 30",
            "--vehicle_types", "passenger 60 commercial 30 public 10",
            "--departure_pattern", "six_periods",
            "--traffic_light_strategy", "opposites",
            "--traffic_control", "tree_method",
            "--end-time", "120",    # 2 minutes
            "--seed", "42"
        ], workspace=temp_workspace)
        
        assert result.returncode == 0, f"Complete pipeline failed: {result.stderr}"
        
        # Comprehensive validation
        assertions = SystemTestAssertions(temp_workspace)
        assertions.assert_simulation_completed_successfully()
        assertions.assert_all_files_generated()
        assertions.assert_network_properties(expected_edges_min=10, expected_junctions_min=6)
        assertions.assert_vehicle_metrics_within_bounds(min_departed=15, max_travel_time=300)
        assertions.assert_vehicle_types_generated(["passenger", "commercial", "public"])
        assertions.assert_tree_method_active()

    @pytest.mark.integration
    def test_pipeline_reproducibility(self, temp_workspace):
        """Test pipeline reproducibility with identical parameters."""
        # Common parameters
        args = [
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--end-time", "90",
            "--seed", "12345"  # Fixed seed
        ]
        
        # First run
        result1 = run_cli_command(args, workspace=temp_workspace)
        assert result1.returncode == 0, "First run failed"
        metrics1 = get_simulation_metrics(temp_workspace)
        
        # Clean workspace
        for file in temp_workspace.glob("*.xml"):
            file.unlink()
        for file in temp_workspace.glob("*.cfg"):
            file.unlink()
            
        # Second run with same parameters
        result2 = run_cli_command(args, workspace=temp_workspace)
        assert result2.returncode == 0, "Second run failed"
        
        assertions = SystemTestAssertions(temp_workspace)
        assertions.assert_reproducibility(metrics1)