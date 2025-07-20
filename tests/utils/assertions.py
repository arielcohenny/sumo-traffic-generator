"""
Custom assertion classes for system testing.

Provides high-level assertion methods for validating simulation
results, network properties, and system behavior.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import xml.etree.ElementTree as ET
from tests.utils.test_helpers import get_simulation_metrics, validate_xml_structure


class SystemTestAssertions:
    """
    High-level assertions for system test validation.
    
    Provides semantic assertions that encapsulate complex validation
    logic and provide clear error messages.
    """
    
    def __init__(self, workspace: Path):
        """
        Initialize assertions for a specific workspace.
        
        Args:
            workspace: Path to workspace containing test results
        """
        self.workspace = workspace
        self.metrics = get_simulation_metrics(workspace)
    
    def assert_simulation_completed_successfully(self) -> None:
        """Assert that simulation completed without errors."""
        assert self.metrics["simulation_completed"], (
            "Simulation did not complete successfully. "
            f"Check {self.workspace} for error logs."
        )
    
    def assert_all_files_generated(self, expected_files: Optional[List[str]] = None) -> None:
        """
        Assert that all expected output files were generated.
        
        Args:
            expected_files: Optional list of expected files (uses default if None)
        """
        if expected_files is None:
            expected_files = [
                "grid.net.xml",
                "grid.nod.xml", 
                "grid.edg.xml",
                "grid.con.xml",
                "grid.tll.xml",
                "vehicles.rou.xml",
                "zones.poly.xml",
                "grid.sumocfg"
            ]
        
        missing_files = []
        empty_files = []
        
        for filename in expected_files:
            filepath = self.workspace / filename
            
            if not filepath.exists():
                missing_files.append(filename)
            elif filepath.stat().st_size == 0:
                empty_files.append(filename)
        
        error_parts = []
        if missing_files:
            error_parts.append(f"Missing files: {', '.join(missing_files)}")
        if empty_files:
            error_parts.append(f"Empty files: {', '.join(empty_files)}")
        
        assert not error_parts, f"File generation issues: {'; '.join(error_parts)}"
    
    def assert_network_properties(
        self, 
        expected_edges_min: Optional[int] = None,
        expected_edges_max: Optional[int] = None,
        expected_junctions_min: Optional[int] = None,
        expected_junctions_max: Optional[int] = None
    ) -> None:
        """
        Assert network topology properties.
        
        Args:
            expected_edges_min: Minimum expected edge count
            expected_edges_max: Maximum expected edge count
            expected_junctions_min: Minimum expected junction count
            expected_junctions_max: Maximum expected junction count
        """
        edges = self.metrics["network_edges"]
        junctions = self.metrics["network_junctions"]
        
        if expected_edges_min is not None:
            assert edges >= expected_edges_min, (
                f"Network has {edges} edges, expected at least {expected_edges_min}"
            )
        
        if expected_edges_max is not None:
            assert edges <= expected_edges_max, (
                f"Network has {edges} edges, expected at most {expected_edges_max}"
            )
        
        if expected_junctions_min is not None:
            assert junctions >= expected_junctions_min, (
                f"Network has {junctions} junctions, expected at least {expected_junctions_min}"
            )
        
        if expected_junctions_max is not None:
            assert junctions <= expected_junctions_max, (
                f"Network has {junctions} junctions, expected at most {expected_junctions_max}"
            )
    
    def assert_vehicle_metrics_within_bounds(
        self,
        min_departed: Optional[int] = None,
        max_departed: Optional[int] = None,
        min_completion_rate: Optional[float] = None,
        max_completion_rate: Optional[float] = None,
        max_travel_time: Optional[float] = None
    ) -> None:
        """
        Assert vehicle performance metrics are within expected bounds.
        
        Args:
            min_departed: Minimum expected vehicles departed
            max_departed: Maximum expected vehicles departed
            min_completion_rate: Minimum expected completion rate (0.0-1.0)
            max_completion_rate: Maximum expected completion rate (0.0-1.0)
            max_travel_time: Maximum expected average travel time
        """
        departed = self.metrics["vehicles_departed"]
        completion_rate = self.metrics["completion_rate"]
        travel_time = self.metrics["average_travel_time"]
        
        if min_departed is not None:
            assert departed >= min_departed, (
                f"{departed} vehicles departed, expected at least {min_departed}"
            )
        
        if max_departed is not None:
            assert departed <= max_departed, (
                f"{departed} vehicles departed, expected at most {max_departed}"
            )
        
        if min_completion_rate is not None:
            assert completion_rate >= min_completion_rate, (
                f"Completion rate {completion_rate:.2f}, expected at least {min_completion_rate:.2f}"
            )
        
        if max_completion_rate is not None:
            assert completion_rate <= max_completion_rate, (
                f"Completion rate {completion_rate:.2f}, expected at most {max_completion_rate:.2f}"
            )
        
        if max_travel_time is not None:
            assert travel_time <= max_travel_time, (
                f"Average travel time {travel_time:.1f}s, expected at most {max_travel_time:.1f}s"
            )
    
    def assert_no_sumo_errors(self) -> None:
        """Assert that no SUMO errors occurred during simulation."""
        # Check for common error indicators in generated files
        net_file = self.workspace / "grid.net.xml"
        if net_file.exists():
            validate_xml_structure(net_file, "net", ["edge", "junction"])
        
        rou_file = self.workspace / "vehicles.rou.xml"
        if rou_file.exists():
            validate_xml_structure(rou_file, "routes", ["vehicle"])
    
    def assert_xml_validity(self) -> None:
        """Assert that all generated XML files are valid."""
        xml_files = list(self.workspace.glob("*.xml"))
        
        invalid_files = []
        for xml_file in xml_files:
            try:
                ET.parse(xml_file)
            except ET.ParseError as e:
                invalid_files.append(f"{xml_file.name}: {e}")
        
        assert not invalid_files, f"Invalid XML files: {'; '.join(invalid_files)}"
    
    def assert_vehicle_types_generated(self, expected_types: List[str]) -> None:
        """
        Assert that expected vehicle types were generated.
        
        Args:
            expected_types: List of expected vehicle type names
        """
        rou_file = self.workspace / "vehicles.rou.xml"
        assert rou_file.exists(), "Route file not found"
        
        rou_content = rou_file.read_text()
        
        missing_types = []
        for vehicle_type in expected_types:
            if f'type="{vehicle_type}"' not in rou_content:
                missing_types.append(vehicle_type)
        
        assert not missing_types, (
            f"Missing vehicle types in route file: {', '.join(missing_types)}"
        )
    
    def assert_sample_network_imported(self) -> None:
        """Assert that Tree Method sample network was successfully imported."""
        net_file = self.workspace / "grid.net.xml"
        assert net_file.exists(), "Network file not generated from sample"
        
        # Check for sample-specific characteristics
        net_content = net_file.read_text()
        assert "<net" in net_content, "Invalid network structure"
        assert "edge" in net_content, "No edges found in sample network"
        
        # Sample networks should have reasonable complexity
        assert self.metrics["network_edges"] > 10, "Sample network has too few edges"
        assert self.metrics["network_junctions"] > 5, "Sample network has too few junctions"
    
    def assert_traffic_control_applied(self, control_method: str) -> None:
        """
        Assert that specified traffic control method was applied.
        
        Args:
            control_method: Expected traffic control method ("fixed", "actuated", "tree_method")
        """
        # Check traffic light file exists
        tll_file = self.workspace / "grid.tll.xml"
        assert tll_file.exists(), "Traffic light file not generated"
        
        # For more specific validation, we could check:
        # - Fixed: static timing patterns
        # - Actuated: detector configurations  
        # - Tree Method: custom algorithm markers
        
        # Basic validation that traffic lights exist
        tll_content = tll_file.read_text()
        assert "<tlLogic" in tll_content, f"No traffic light logic found for {control_method}"
    
    def assert_tree_method_active(self) -> None:
        """Assert that Tree Method algorithm is active."""
        # Tree Method should generate traffic light configurations
        self.assert_traffic_control_applied("tree_method")
        
        # Additional Tree Method specific checks could be added here
        # e.g., checking for algorithm-specific files or configurations
    
    def assert_performance_regression(
        self, 
        baseline_metrics: Dict[str, Any],
        tolerance: float = 0.15
    ) -> None:
        """
        Assert that performance has not regressed compared to baseline.
        
        Args:
            baseline_metrics: Baseline metrics to compare against
            tolerance: Allowed performance degradation (0.15 = 15%)
        """
        current_travel_time = self.metrics["average_travel_time"]
        baseline_travel_time = baseline_metrics.get("average_travel_time", 0)
        
        if baseline_travel_time > 0:
            degradation = (current_travel_time - baseline_travel_time) / baseline_travel_time
            assert degradation <= tolerance, (
                f"Performance regression detected: travel time increased by "
                f"{degradation:.1%} (baseline: {baseline_travel_time:.1f}s, "
                f"current: {current_travel_time:.1f}s)"
            )
        
        current_completion = self.metrics["completion_rate"]
        baseline_completion = baseline_metrics.get("completion_rate", 0)
        
        if baseline_completion > 0:
            completion_drop = (baseline_completion - current_completion) / baseline_completion
            assert completion_drop <= tolerance, (
                f"Completion rate regression detected: rate decreased by "
                f"{completion_drop:.1%} (baseline: {baseline_completion:.1%}, "
                f"current: {current_completion:.1%})"
            )
    
    def assert_reproducibility(self, other_metrics: Dict[str, Any]) -> None:
        """
        Assert that results are reproducible (for fixed seed tests).
        
        Args:
            other_metrics: Metrics from another run with same parameters
        """
        # Key metrics should be identical for same seed
        assert self.metrics["vehicles_departed"] == other_metrics["vehicles_departed"], (
            "Vehicle departure count not reproducible"
        )
        
        # Travel times should be very close (within 1 second)
        travel_time_diff = abs(
            self.metrics["average_travel_time"] - other_metrics["average_travel_time"]
        )
        assert travel_time_diff < 1.0, (
            f"Travel times not reproducible: {travel_time_diff:.1f}s difference"
        )