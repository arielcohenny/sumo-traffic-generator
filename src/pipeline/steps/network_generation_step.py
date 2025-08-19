"""
Network generation pipeline step.

Handles Step 1 of the pipeline: generating synthetic grid networks.
"""

from typing import Any

from .base_step import BaseStep
from src.network.generate_grid import generate_grid_network
from src.validate.validate_network import verify_generate_grid_network


class NetworkGenerationStep(BaseStep):
    """Pipeline step for network generation."""
    
    def execute(self) -> None:
        """Execute synthetic grid network generation."""
        self._generate_grid_network()
    
    def validate(self) -> None:
        """Validate network generation results."""
        verify_generate_grid_network(
            self._get_seed(),
            int(self.args.grid_dimension),
            int(self.args.block_size_m),
            self.args.junctions_to_remove,
            self.args.lane_count,
            self.args.traffic_light_strategy
        )
    
    def _generate_grid_network(self) -> None:
        """Generate synthetic grid network."""
        self.logger.info("Generating SUMO orthogonal grid network...")
        generate_grid_network(
            self._get_seed(),
            int(self.args.grid_dimension),
            int(self.args.block_size_m),
            self.args.junctions_to_remove,
            self.args.lane_count,
            self.args.traffic_light_strategy
        )
        self.logger.info("Generated grid successfully")
    
    def _get_seed(self) -> int:
        """Get the random seed, generating one if not provided."""
        if hasattr(self.args, '_seed'):
            return self.args._seed
        
        import random
        seed = self.args.seed if self.args.seed is not None else random.randint(0, 2**32 - 1)
        self.args._seed = seed  # Cache for other steps
        self.logger.info(f"Using seed: {seed}")
        return seed