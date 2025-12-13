"""
Integration tests for CLI arguments.

Tests each CLI argument with multiple values to ensure network generation
completes successfully (steps 1-7). Does NOT run SUMO simulation.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from tests.utils.test_helpers import run_file_generation, validate_output_files


# Base arguments used in all tests (minimal working configuration)
BASE_ARGS = [
    "--grid_dimension", "3",
    "--num_vehicles", "20",
    "--seed", "42",
]


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_cli_args_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def get_workspace_dir(temp_workspace: Path) -> Path:
    """Get the workspace subdirectory where files are generated."""
    return temp_workspace / "workspace"


class TestNetworkArguments:
    """Tests for network generation arguments."""

    @pytest.mark.integration
    @pytest.mark.parametrize("grid_dim", [3, 4, 5, 7])
    def test_grid_dimension_values(self, temp_workspace, grid_dim):
        """Test different grid dimensions.

        Note: Starting from 3x3 because 2x2 grids have limited edge availability
        that can cause route generation issues with typical vehicle counts.
        """
        result = run_file_generation([
            "--grid_dimension", str(grid_dim),
            "--num_vehicles", "30",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with grid_dimension={grid_dim}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("block_size", [50, 100, 200, 300])
    def test_block_size_values(self, temp_workspace, block_size):
        """Test different block sizes."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--block_size_m", str(block_size),
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with block_size_m={block_size}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("junctions", ["0", "1", "2"])
    def test_junctions_to_remove_values(self, temp_workspace, junctions):
        """Test different junction removal counts."""
        result = run_file_generation([
            "--grid_dimension", "4",  # Larger grid to allow junction removal
            "--junctions_to_remove", junctions,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with junctions_to_remove={junctions}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("lane_count", ["2", "3", "random", "realistic"])
    def test_lane_count_values(self, temp_workspace, lane_count):
        """Test different lane count configurations."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--lane_count", lane_count,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with lane_count={lane_count}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))


class TestTrafficArguments:
    """Tests for traffic generation arguments."""

    @pytest.mark.integration
    @pytest.mark.parametrize("num_vehicles", [10, 50, 100, 200])
    def test_num_vehicles_values(self, temp_workspace, num_vehicles):
        """Test different vehicle counts."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", str(num_vehicles),
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with num_vehicles={num_vehicles}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("strategy", [
        "shortest 100",
        "realtime 100",
        "fastest 100",
        "shortest 70 realtime 30",
    ])
    def test_routing_strategy_values(self, temp_workspace, strategy):
        """Test different routing strategies."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--routing_strategy", strategy,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with routing_strategy='{strategy}': {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("vehicle_types", [
        "passenger 100",
        "passenger 90 public 10",
        "passenger 50 public 50",
    ])
    def test_vehicle_types_values(self, temp_workspace, vehicle_types):
        """Test different vehicle type distributions."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--vehicle_types", vehicle_types,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with vehicle_types='{vehicle_types}': {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("pattern", ["uniform", "six_periods"])
    def test_departure_pattern_values(self, temp_workspace, pattern):
        """Test different departure patterns."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--departure_pattern", pattern,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with departure_pattern={pattern}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("routes", [
        "in 30 out 30 inner 25 pass 15",
        "in 50 out 50 inner 0 pass 0",
        "in 25 out 25 inner 25 pass 25",
    ])
    def test_passenger_routes_values(self, temp_workspace, routes):
        """Test different passenger route patterns."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--passenger-routes", routes,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with passenger-routes='{routes}': {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("routes", [
        "in 25 out 25 inner 35 pass 15",
        "in 50 out 50 inner 0 pass 0",
        "in 25 out 25 inner 25 pass 25",
    ])
    def test_public_routes_values(self, temp_workspace, routes):
        """Test different public route patterns."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--public-routes", routes,
            "--vehicle_types", "passenger 90 public 10",
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with public-routes='{routes}': {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))


class TestSimulationArguments:
    """Tests for simulation configuration arguments."""

    @pytest.mark.integration
    @pytest.mark.parametrize("seed", [1, 42, 12345, 99999])
    def test_seed_values(self, temp_workspace, seed):
        """Test different seed values."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--seed", str(seed),
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with seed={seed}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("network_seed", [1, 100, 5000])
    def test_network_seed_values(self, temp_workspace, network_seed):
        """Test different network seed values."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--network-seed", str(network_seed),
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with network-seed={network_seed}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("private_seed", [1, 100, 5000])
    def test_private_traffic_seed_values(self, temp_workspace, private_seed):
        """Test different private traffic seed values."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--private-traffic-seed", str(private_seed),
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with private-traffic-seed={private_seed}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("public_seed", [1, 100, 5000])
    def test_public_traffic_seed_values(self, temp_workspace, public_seed):
        """Test different public traffic seed values."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--vehicle_types", "passenger 90 public 10",
            "--public-traffic-seed", str(public_seed),
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with public-traffic-seed={public_seed}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("step_length", [1.0, 2.0, 5.0])
    def test_step_length_values(self, temp_workspace, step_length):
        """Test different step length values (valid range: 1.0-10.0)."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--step-length", str(step_length),
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with step-length={step_length}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("end_time", [60, 300, 1800, 3600])
    def test_end_time_values(self, temp_workspace, end_time):
        """Test different end time values."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--end-time", str(end_time),
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with end-time={end_time}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))


class TestZoneArguments:
    """Tests for zone and attractiveness arguments."""

    @pytest.mark.integration
    @pytest.mark.parametrize("block_size", [10.0, 25.0, 50.0, 100.0])
    def test_land_use_block_size_values(self, temp_workspace, block_size):
        """Test different land use block sizes."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--land_use_block_size_m", str(block_size),
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with land_use_block_size_m={block_size}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("attractiveness", ["land_use", "poisson", "iac"])
    def test_attractiveness_values(self, temp_workspace, attractiveness):
        """Test different attractiveness methods."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--attractiveness", attractiveness,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with attractiveness={attractiveness}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("start_hour", [0.0, 7.0, 12.0, 18.0])
    def test_start_time_hour_values(self, temp_workspace, start_hour):
        """Test different start time hours."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--start_time_hour", str(start_hour),
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with start_time_hour={start_hour}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))


class TestTrafficControlArguments:
    """Tests for traffic control arguments."""

    @pytest.mark.integration
    @pytest.mark.parametrize("strategy", ["opposites", "incoming", "partial_opposites"])
    def test_traffic_light_strategy_values(self, temp_workspace, strategy):
        """Test different traffic light strategies."""
        # partial_opposites requires 2+ lanes
        lane_count = "2" if strategy == "partial_opposites" else "realistic"
        result = run_file_generation([
            "--grid_dimension", "3",
            "--traffic_light_strategy", strategy,
            "--lane_count", lane_count,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with traffic_light_strategy={strategy}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("control", ["tree_method", "actuated", "fixed"])
    def test_traffic_control_values(self, temp_workspace, control):
        """Test different traffic control methods."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--traffic_control", control,
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with traffic_control={control}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("interval", [30, 60, 90, 180])
    def test_tree_method_interval_values(self, temp_workspace, interval):
        """Test different tree method intervals."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--tree-method-interval", str(interval),
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with tree-method-interval={interval}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    @pytest.mark.parametrize("interval", [15, 30, 60, 120])
    def test_bottleneck_detection_interval_values(self, temp_workspace, interval):
        """Test different bottleneck detection intervals."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--bottleneck-detection-interval", str(interval),
            "--num_vehicles", "20",
            "--seed", "42",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Failed with bottleneck-detection-interval={interval}: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))


class TestCombinedArguments:
    """Tests for combined argument scenarios."""

    @pytest.mark.integration
    def test_full_configuration(self, temp_workspace):
        """Test a comprehensive configuration with many arguments."""
        result = run_file_generation([
            "--grid_dimension", "4",
            "--block_size_m", "150",
            "--lane_count", "2",
            "--num_vehicles", "100",
            "--routing_strategy", "shortest 70 realtime 30",
            "--vehicle_types", "passenger 90 public 10",
            "--departure_pattern", "uniform",
            "--seed", "42",
            "--end-time", "1800",
            "--attractiveness", "land_use",
            "--traffic_light_strategy", "partial_opposites",
            "--traffic_control", "actuated",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Full configuration failed: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    def test_minimal_configuration(self, temp_workspace):
        """Test minimal required configuration."""
        # Note: Using 3x3 grid minimum because very small grids (2x2) with few vehicles
        # can cause route generation issues due to limited edge availability
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "20",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Minimal configuration failed: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))

    @pytest.mark.integration
    def test_separate_seeds(self, temp_workspace):
        """Test configuration with separate seeds for network and traffic."""
        result = run_file_generation([
            "--grid_dimension", "3",
            "--num_vehicles", "50",
            "--network-seed", "100",
            "--private-traffic-seed", "200",
            "--public-traffic-seed", "300",
            "--vehicle_types", "passenger 80 public 20",
            "--workspace", str(temp_workspace),
        ])
        assert result.returncode == 0, f"Separate seeds configuration failed: {result.stderr}"
        validate_output_files(get_workspace_dir(temp_workspace))
