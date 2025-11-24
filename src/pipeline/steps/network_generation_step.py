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
            self._get_network_seed(),
            int(self.args.grid_dimension),
            int(self.args.block_size_m),
            self.args.junctions_to_remove,
            self.args.lane_count,
            self.args.traffic_light_strategy,
            getattr(self.args, 'traffic_control', 'tree_method')
        )

    def _generate_grid_network(self) -> None:
        """Generate synthetic grid network."""
        # self.logger.info("Generating SUMO orthogonal grid network...")
        generate_grid_network(
            self._get_network_seed(),
            int(self.args.grid_dimension),
            int(self.args.block_size_m),
            self.args.junctions_to_remove,
            self.args.lane_count,
            self.args.traffic_light_strategy,
            getattr(self.args, 'traffic_control', None)
        )
        # self.logger.info("Generated grid successfully")

    def _get_network_seed(self) -> int:
        """Get the network seed for network generation."""
        from src.utils.multi_seed_utils import get_network_seed
        seed = get_network_seed(self.args)
        # self.logger.info(f"Using network seed: {seed}")
        return seed

    def _get_seed(self) -> int:
        """Get the random seed, generating one if not provided. (Backward compatibility)"""
        return self._get_network_seed()
