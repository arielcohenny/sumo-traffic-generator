"""
Comparison runner for multi-run experiments with network reuse.

Orchestrates the generation of network files once and reuse across
multiple simulation runs with different traffic seeds and control methods.
"""

import json
import logging
import shutil
from argparse import Namespace
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import CONFIG
from src.constants import (
    COMPARISON_NETWORK_FILES,
    COMPARISON_DATE_FORMAT,
    SECONDS_PER_HOUR,
)
from src.orchestration.run_spec import RunSpec, RunMetrics
from src.orchestration.comparison_results import ComparisonResults
from src.orchestration.metrics_extractor import MetricsExtractor


# Additional optional network files (beyond the required ones in constants)
OPTIONAL_NETWORK_FILES = [
    "zones.geojson",
    "attractiveness_phases.json",
]


class ComparisonRunner:
    """Orchestrates multi-run comparisons with network reuse.

    This class manages:
    1. Generating network files once (steps 1-5)
    2. Saving network files for reuse
    3. Running multiple simulations with different traffic configs
    4. Collecting and aggregating metrics
    """

    def __init__(self, workspace: Path):
        """Initialize comparison runner.

        Args:
            workspace: Base workspace directory for the experiment
        """
        self.workspace = Path(workspace)
        self.network_path = self.workspace / "network"
        self.runs_path = self.workspace / "runs"
        self.logger = logging.getLogger(__name__)
        self._metrics_extractor = MetricsExtractor()
        self._network_config: Optional[Dict[str, Any]] = None

    def generate_network_only(self, args: Namespace) -> Path:
        """Generate network files (steps 1-5) and save for reuse.

        Args:
            args: Command line arguments with network configuration

        Returns:
            Path to the network directory
        """
        from src.pipeline.standard_pipeline import StandardPipeline
        from src.utils.multi_seed_utils import get_network_seed

        self.logger.info("Generating network files (steps 1-5)")

        # Create network directory
        self.network_path.mkdir(parents=True, exist_ok=True)

        # Create a temporary workspace for pipeline execution
        temp_workspace = self.workspace / "temp_network_gen"
        temp_workspace.mkdir(parents=True, exist_ok=True)

        # Modify args to use temp workspace
        modified_args = deepcopy(args)
        modified_args.workspace = str(temp_workspace)

        # Update CONFIG to use temp workspace
        CONFIG.update_workspace(str(temp_workspace))

        try:
            # Create pipeline and run file generation only
            pipeline = StandardPipeline(modified_args)
            pipeline.execute_file_generation_only()

            # Copy network files to network directory
            temp_output = temp_workspace / "workspace"
            all_network_files = list(COMPARISON_NETWORK_FILES) + OPTIONAL_NETWORK_FILES
            for filename in all_network_files:
                src_file = temp_output / filename
                if src_file.exists():
                    shutil.copy2(src_file, self.network_path / filename)
                    self.logger.debug(f"Copied {filename} to network directory")

            # Save network configuration
            self._save_network_config(args, get_network_seed(args))

            self.logger.info(f"Network files saved to: {self.network_path}")

        finally:
            # Cleanup temp directory
            if temp_workspace.exists():
                shutil.rmtree(temp_workspace)

        return self.network_path

    def _save_network_config(self, args: Namespace, network_seed: int):
        """Save network configuration for reproducibility.

        Args:
            args: Command line arguments used for generation
            network_seed: The network seed that was used
        """
        config = {
            "created_at": datetime.now().isoformat(),
            "network_seed": network_seed,
            "grid_dimension": getattr(args, "grid_dimension", None),
            "block_size_m": getattr(args, "block_size_m", None),
            "junctions_to_remove": getattr(args, "junctions_to_remove", None),
            "lane_count": getattr(args, "lane_count", None),
            "traffic_light_strategy": getattr(args, "traffic_light_strategy", None),
            "attractiveness": getattr(args, "attractiveness", None),
            "land_use_block_size_m": getattr(args, "land_use_block_size_m", None),
        }

        config_file = self.network_path / "network_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        self._network_config = config

    def load_existing_network(self, network_path: Path) -> Dict[str, Any]:
        """Load and validate an existing network directory.

        Args:
            network_path: Path to network directory with saved files

        Returns:
            Network configuration dictionary

        Raises:
            FileNotFoundError: If required files are missing
        """
        network_path = Path(network_path)

        if not network_path.exists():
            raise FileNotFoundError(f"Network directory not found: {network_path}")

        # Check for required files
        required_files = ["grid.net.xml", "network_config.json"]
        for filename in required_files:
            if not (network_path / filename).exists():
                raise FileNotFoundError(f"Required file missing: {filename}")

        # Load configuration
        config_file = network_path / "network_config.json"
        with open(config_file, "r") as f:
            self._network_config = json.load(f)

        self.network_path = network_path
        self.logger.info(f"Loaded network from: {network_path}")

        return self._network_config

    def run_single(
        self,
        run_spec: RunSpec,
        base_args: Namespace,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunMetrics:
        """Run a single simulation with the specified configuration.

        Args:
            run_spec: Specification for this run
            base_args: Base command line arguments
            progress_callback: Optional callback(status, progress) for updates

        Returns:
            RunMetrics with results from this run
        """
        from src.traffic.builder import execute_route_generation
        from src.sumo_integration.sumo_utils import execute_config_generation
        from src.orchestration.simulator import execute_standard_simulation

        self.logger.info(f"Starting run: {run_spec.name}")

        # Create run directory
        run_dir = self.runs_path / run_spec.name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create workspace subdirectory for this run
        run_workspace = run_dir / "workspace"
        run_workspace.mkdir(parents=True, exist_ok=True)

        # Copy network files to run workspace
        all_network_files = list(COMPARISON_NETWORK_FILES) + OPTIONAL_NETWORK_FILES
        for filename in all_network_files:
            src_file = self.network_path / filename
            if src_file.exists():
                shutil.copy2(src_file, run_workspace / filename)

        # Prepare args for this run
        run_args = deepcopy(base_args)
        run_args.workspace = str(run_dir)

        # Override seeds and traffic control
        run_args.private_traffic_seed = run_spec.private_traffic_seed
        run_args.public_traffic_seed = run_spec.public_traffic_seed
        run_args.traffic_control = run_spec.traffic_control

        # Clear single seed to use individual seeds
        if hasattr(run_args, 'seed'):
            run_args.seed = None

        # Update CONFIG for this run
        CONFIG.update_workspace(str(run_dir))

        if progress_callback:
            progress_callback(f"Generating traffic for {run_spec.name}", 0.1)

        # Step 6: Generate vehicle routes
        execute_route_generation(run_args)

        if progress_callback:
            progress_callback(f"Generating config for {run_spec.name}", 0.2)

        # Step 7: Generate SUMO configuration
        execute_config_generation(run_args)

        if progress_callback:
            progress_callback(f"Running simulation for {run_spec.name}", 0.3)

        # Step 8: Run simulation
        execute_standard_simulation(run_args)

        if progress_callback:
            progress_callback(f"Extracting metrics for {run_spec.name}", 0.9)

        # Extract metrics
        metrics = self._metrics_extractor.extract_from_run(run_workspace, run_spec)

        if progress_callback:
            progress_callback(f"Completed {run_spec.name}", 1.0)

        self.logger.info(f"Completed run: {run_spec.name}")
        return metrics

    def run_comparison(
        self,
        run_specs: List[RunSpec],
        base_args: Namespace,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> ComparisonResults:
        """Execute all comparison runs sequentially.

        Args:
            run_specs: List of run specifications
            base_args: Base command line arguments
            progress_callback: Optional callback(status, current, total) for updates

        Returns:
            ComparisonResults with all run metrics
        """
        self.logger.info(f"Starting comparison with {len(run_specs)} runs")

        # Ensure network exists
        if not self.network_path.exists() or not (self.network_path / "grid.net.xml").exists():
            self.logger.info("Network not found, generating...")
            self.generate_network_only(base_args)

        # Load network config if not already loaded
        if self._network_config is None:
            self.load_existing_network(self.network_path)

        # Create runs directory
        self.runs_path.mkdir(parents=True, exist_ok=True)

        # Initialize results
        results = ComparisonResults(
            network_config=self._network_config or {},
            comparison_name=f"comparison_{datetime.now().strftime(COMPARISON_DATE_FORMAT)}"
        )

        # Run each specification
        for i, spec in enumerate(run_specs):
            if progress_callback:
                progress_callback(f"Run {i+1}/{len(run_specs)}: {spec.name}", i, len(run_specs))

            try:
                metrics = self.run_single(spec, base_args)
                results.add_run(metrics)
            except Exception as e:
                self.logger.error(f"Run {spec.name} failed: {e}")
                # Add empty metrics for failed run
                failed_metrics = RunMetrics(
                    name=spec.name,
                    traffic_control=spec.traffic_control,
                    private_traffic_seed=spec.private_traffic_seed,
                    public_traffic_seed=spec.public_traffic_seed,
                )
                results.add_run(failed_metrics)

        # Save results
        results_file = self.workspace / "comparison_results.json"
        results.to_json(results_file)
        self.logger.info(f"Results saved to: {results_file}")

        if progress_callback:
            progress_callback("Comparison complete", len(run_specs), len(run_specs))

        return results

    def get_network_config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded network configuration."""
        return self._network_config

    def has_network(self) -> bool:
        """Check if network files exist."""
        return (
            self.network_path.exists() and
            (self.network_path / "grid.net.xml").exists()
        )
