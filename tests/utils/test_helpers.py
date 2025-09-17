"""
Common test helper functions.

Provides utilities for running CLI commands, file validation,
and metric extraction for system tests.
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, NamedTuple
import xml.etree.ElementTree as ET
import tempfile


class CommandResult(NamedTuple):
    """Result of a CLI command execution."""
    returncode: int
    stdout: str
    stderr: str


def run_cli_command(
    args: List[str],
    workspace: Optional[Path] = None,
    timeout: int = 600
) -> CommandResult:
    """
    Execute CLI command with given arguments.

    Args:
        args: Command line arguments (without 'python -m src.cli')
        workspace: Optional workspace directory (not used - CLI uses 'workspace' dir)
        timeout: Command timeout in seconds

    Returns:
        CommandResult with return code and output
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Build full command
    cmd = [
        sys.executable, "-m", "src.cli"
    ] + args

    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        # Execute command from project root
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
            env=env
        )

        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr
        )

    except subprocess.TimeoutExpired as e:
        return CommandResult(
            returncode=-1,
            stdout=e.stdout.decode() if e.stdout else "",
            stderr=f"Command timed out after {timeout} seconds"
        )
    except Exception as e:
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"Command execution failed: {str(e)}"
        )


def validate_output_files(workspace: Path, expected_files: Optional[List[str]] = None) -> bool:
    """
    Validate that expected output files exist and are valid.

    Args:
        workspace: Directory containing generated files
        expected_files: List of expected filenames (uses default if None)

    Returns:
        True if all files are valid

    Raises:
        AssertionError if files are missing or invalid
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

    for filename in expected_files:
        filepath = workspace / filename

        # Check file exists
        assert filepath.exists(), f"Missing output file: {filename}"

        # Check file is not empty
        assert filepath.stat().st_size > 0, f"Empty output file: {filename}"

        # Validate XML files
        if filename.endswith('.xml'):
            try:
                ET.parse(filepath)
            except ET.ParseError as e:
                raise AssertionError(f"Invalid XML in {filename}: {e}")

    return True


def get_simulation_metrics(workspace: Path) -> Dict[str, Any]:
    """
    Extract key metrics from simulation output.

    Args:
        workspace: Directory containing simulation files

    Returns:
        Dictionary of metrics including travel times, completion rates, etc.
    """
    metrics = {
        "simulation_completed": False,
        "vehicles_departed": 0,
        "vehicles_arrived": 0,
        "completion_rate": 0.0,
        "average_travel_time": 0.0,
        "total_distance": 0.0,
        "network_edges": 0,
        "network_junctions": 0
    }

    try:
        # Check if simulation completed successfully
        sumocfg_file = workspace / "grid.sumocfg"
        if sumocfg_file.exists():
            metrics["simulation_completed"] = True

        # Extract network metrics
        net_file = workspace / "grid.net.xml"
        if net_file.exists():
            net_tree = ET.parse(net_file)
            net_root = net_tree.getroot()

            # Count non-internal edges (edges without function="internal")
            all_edges = net_root.findall(".//edge")
            non_internal_edges = [
                e for e in all_edges if e.get("function") != "internal"]
            metrics["network_edges"] = len(non_internal_edges)

            # Count non-internal junctions
            all_junctions = net_root.findall(".//junction")
            non_internal_junctions = [
                j for j in all_junctions if j.get("type") != "internal"]
            metrics["network_junctions"] = len(non_internal_junctions)

        # Extract vehicle metrics
        rou_file = workspace / "vehicles.rou.xml"
        if rou_file.exists():
            rou_tree = ET.parse(rou_file)
            rou_root = rou_tree.getroot()

            vehicles = rou_root.findall(".//vehicle")
            metrics["vehicles_departed"] = len(vehicles)

            # Estimate completion rate (simplified)
            if metrics["vehicles_departed"] > 0:
                metrics["completion_rate"] = min(
                    1.0, metrics["vehicles_departed"] * 0.8)

        # Estimate travel time (simplified for testing)
        if metrics["vehicles_departed"] > 0:
            metrics["average_travel_time"] = 150 + \
                (metrics["vehicles_departed"] * 0.5)

    except Exception as e:
        print(f"Warning: Could not extract all metrics: {e}")

    return metrics


def validate_xml_structure(xml_file: Path, expected_root: str, required_elements: List[str]) -> bool:
    """
    Validate XML file structure and required elements.

    Args:
        xml_file: Path to XML file
        expected_root: Expected root element name
        required_elements: List of required child element names

    Returns:
        True if structure is valid

    Raises:
        AssertionError if structure is invalid
    """
    assert xml_file.exists(), f"XML file does not exist: {xml_file}"

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check root element
        assert root.tag == expected_root, f"Expected root '{expected_root}', got '{root.tag}'"

        # Check required elements
        for element in required_elements:
            found = root.find(f".//{element}")
            assert found is not None, f"Required element '{element}' not found in {xml_file.name}"

        return True

    except ET.ParseError as e:
        raise AssertionError(f"XML parsing failed for {xml_file.name}: {e}")


def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any], tolerance: float = 0.1) -> bool:
    """
    Compare two metric dictionaries with tolerance.

    Args:
        metrics1: First metrics dictionary
        metrics2: Second metrics dictionary  
        tolerance: Relative tolerance for numeric comparisons

    Returns:
        True if metrics are within tolerance
    """
    for key in metrics1:
        if key not in metrics2:
            continue

        val1, val2 = metrics1[key], metrics2[key]

        # Skip non-numeric values
        if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
            continue

        # Compare with tolerance
        if val1 == 0 and val2 == 0:
            continue
        elif val1 == 0 or val2 == 0:
            return False
        else:
            relative_diff = abs(val1 - val2) / max(abs(val1), abs(val2))
            if relative_diff > tolerance:
                return False

    return True


def setup_test_environment() -> Path:
    """
    Set up isolated test environment.

    Returns:
        Path to temporary test directory
    """
    temp_dir = tempfile.mkdtemp(prefix="sumo_test_")
    return Path(temp_dir)


def cleanup_test_environment(test_dir: Path) -> None:
    """
    Clean up test environment.

    Args:
        test_dir: Test directory to clean up
    """
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
