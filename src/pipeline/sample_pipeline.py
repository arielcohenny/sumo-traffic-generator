"""
Tree Method sample pipeline for bypass mode.

This pipeline skips steps 1-8 and goes directly to simulation using
pre-built Tree Method sample networks.
"""

from .base_pipeline import BasePipeline
from src.orchestration.simulator import execute_sample_simulation
from src.network.tree_method_samples import setup_tree_method_samples


class SamplePipeline(BasePipeline):
    """Pipeline for Tree Method sample testing (bypass mode)."""
    
    def __init__(self, args):
        super().__init__(args)
        self.sample_folder = args.tree_method_sample
    
    def execute(self) -> None:
        """Execute sample pipeline (bypass Steps 1-8, go to Step 9)."""
        self._validate_output_directory()
        
        self.logger.info(f"Tree Method Sample Mode: Using pre-built network from {self.sample_folder}")
        self.logger.info("Skipping Steps 1-8, going directly to Step 9 (Dynamic Simulation)")
        
        # Setup sample files
        setup_tree_method_samples(self.args, self.sample_folder)
        
        # Step 9: Dynamic Simulation
        self._log_step(9, "Dynamic Simulation (Tree Method Sample)")
        execute_sample_simulation(self.args)
    
