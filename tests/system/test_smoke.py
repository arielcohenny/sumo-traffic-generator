"""
Smoke tests for quick validation.

These tests provide fast feedback on basic system functionality.
All tests should complete in under 30 seconds each.
"""

import pytest
from pathlib import Path
from tests.utils.test_helpers import run_cli_command

def get_workspace_dir():
    """Get the actual workspace directory."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "workspace"


class TestQuickValidation:
    """Ultra-fast validation tests for CI/CD."""
    
    @pytest.mark.smoke
    def test_cli_help(self):
        """Test CLI help command works."""
        result = run_cli_command(["--help"])
        assert result.returncode == 0
        assert "SUMO Traffic Generator" in result.stdout or "usage:" in result.stdout

    @pytest.mark.smoke  
    def test_minimal_synthetic_network(self, temp_workspace):
        """
        Minimal synthetic network generation.
        
        Tests network generation without simulation.
        """
        from pathlib import Path
        
        result = run_cli_command([
            "--grid_dimension", "3",
            "--block_size_m", "100",
            "--num_vehicles", "10",
            "--end-time", "30",  # 30 seconds
            "--seed", "1"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Check essential files exist in actual workspace directory
        project_root = Path(__file__).parent.parent.parent
        workspace_dir = project_root / "workspace"
        
        essential_files = ["grid.net.xml", "vehicles.rou.xml", "grid.sumocfg"]
        for filename in essential_files:
            filepath = workspace_dir / filename
            assert filepath.exists(), f"Missing essential file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"

    @pytest.mark.smoke
    def test_tree_method_sample_smoke(self, temp_workspace):
        """
        Test Tree Method sample import without full simulation.
        
        Quick validation of sample data processing pipeline.
        """
        result = run_cli_command([
            "--tree_method_sample", "evaluation/datasets/networks/",
            "--traffic_control", "tree_method",
            "--end-time", "30",    # 30 seconds
            "--seed", "1"
        ])
        
        # Skip if sample data doesn't exist
        if result.returncode != 0 and "not found" in result.stderr.lower():
            pytest.skip("Tree Method sample data not available")
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Verify sample was processed
        workspace_dir = get_workspace_dir()
        net_file = workspace_dir / "grid.net.xml"
        assert net_file.exists(), "Network file not generated from sample"
        
        # Basic content validation
        net_content = net_file.read_text()
        assert "<net" in net_content, "Invalid network XML structure"
        assert "<edge" in net_content, "No edges found in network"

    @pytest.mark.smoke
    def test_configuration_validation(self, temp_workspace):
        """
        Test configuration parameter validation.
        
        Ensures invalid parameters are caught early.
        """
        # Test invalid grid dimension
        result = run_cli_command([
            "--grid_dimension", "0",  # Invalid
            "--num_vehicles", "10"
        ], workspace=temp_workspace)
        
        assert result.returncode != 0, "Should reject invalid grid dimension"

    @pytest.mark.smoke
    def test_vehicle_types_validation(self, temp_workspace):
        """
        Test vehicle type distribution validation.
        
        Quick check of parameter parsing.
        """
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "10",
            "--end-time", "30",
            "--vehicle_types", "passenger 70 commercial 20 public 10",
            "--seed", "1"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Check vehicle file contains expected types
        workspace_dir = get_workspace_dir()
        rou_file = workspace_dir / "vehicles.rou.xml"
        assert rou_file.exists(), "Route file not generated"
        
        rou_content = rou_file.read_text()
        assert 'type="passenger"' in rou_content, "Passenger vehicles not found"


class TestPipelineSteps:
    """Test individual pipeline steps in isolation."""
    
    @pytest.mark.smoke
    def test_network_generation_step(self, temp_workspace):
        """Test network generation step completes."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--end-time", "30",  # Increased from 1 second
            "--num_vehicles", "10",  # Increased from 1 vehicle
            "--seed", "1"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Network files should exist
        workspace_dir = get_workspace_dir()
        network_files = ["grid.net.xml", "grid.nod.xml", "grid.edg.xml"]
        for filename in network_files:
            filepath = workspace_dir / filename
            assert filepath.exists(), f"Network file missing: {filename}"

    @pytest.mark.smoke
    def test_traffic_generation_step(self, temp_workspace):
        """Test traffic generation step completes."""
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "5",
            "--end-time", "30",
            "--routing_strategy", "shortest 100",
            "--seed", "1"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Route file should exist and contain vehicles
        workspace_dir = get_workspace_dir()
        rou_file = workspace_dir / "vehicles.rou.xml"
        assert rou_file.exists(), "Route file not generated"
        
        rou_content = rou_file.read_text()
        assert "<vehicle" in rou_content, "No vehicles in route file"
        assert '<route edges=' in rou_content, "No routes assigned to vehicles"


class TestErrorHandling:
    """Test error handling and graceful failures."""
    
    @pytest.mark.smoke
    def test_missing_sample_directory(self, temp_workspace):
        """Test graceful handling of missing Tree Method sample directory."""
        result = run_cli_command([
            "--tree_method_sample", "/nonexistent/path/missing_dir/",
            "--num_vehicles", "10"
        ], workspace=temp_workspace)
        
        assert result.returncode != 0, "Should fail with missing sample directory"
        assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()

    @pytest.mark.smoke
    def test_invalid_parameter_combinations(self, temp_workspace):
        """Test validation of parameter combinations."""
        # Vehicle types that don't sum to 100
        result = run_cli_command([
            "--grid_dimension", "3",
            "--vehicle_types", "passenger 50 commercial 30",  # Missing 20%
            "--num_vehicles", "10"
        ], workspace=temp_workspace)
        
        assert result.returncode != 0, "Should reject invalid vehicle type percentages"

    @pytest.mark.smoke 
    def test_workspace_permissions(self, temp_workspace):
        """Test handling of workspace permission issues."""
        # This test validates workspace access
        result = run_cli_command([
            "--grid_dimension", "3",
            "--num_vehicles", "5",
            "--end-time", "30",
            "--seed", "1"
        ], workspace=temp_workspace)
        
        # Should succeed with proper temp workspace
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Verify we can read generated files
        for file in temp_workspace.glob("*.xml"):
            assert file.is_file(), f"Cannot access generated file: {file.name}"
            content = file.read_text()
            assert len(content) > 0, f"Generated file is empty: {file.name}"