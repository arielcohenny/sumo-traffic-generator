"""
Zone generation pipeline step.

Handles Step 2 of the pipeline: generating land use zones for synthetic grid networks.
"""

from pathlib import Path

from .base_step import BaseStep
from src.network.zones import extract_zones_from_junctions
from src.validate.validate_network import verify_extract_zones_from_junctions
from src.config import CONFIG


class ZoneGenerationStep(BaseStep):
    """Pipeline step for zone generation."""

    def execute(self) -> None:
        """Execute zone generation for synthetic grid networks."""
        self._generate_synthetic_zones()

    def validate(self) -> None:
        """Validate zone generation results."""
        verify_extract_zones_from_junctions(
            self.args.land_use_block_size_m,
            seed=self._get_network_seed(),
            fill_polygons=True,
            inset=0.0
        )

    def _generate_synthetic_zones(self) -> None:
        """Generate zones for synthetic grid networks."""
        # self.logger.info("Generating synthetic network zones...")
        extract_zones_from_junctions(
            self.args.land_use_block_size_m,
            seed=self._get_network_seed(),
            fill_polygons=True,
            inset=0.0
        )
        # self.logger.info(
        #     f"Extracted land use zones successfully with {self.args.land_use_block_size_m}m blocks")

    def _get_network_seed(self) -> int:
        """Get the network seed for zone generation."""
        from src.utils.multi_seed_utils import get_network_seed
        return get_network_seed(self.args)

    def _get_seed(self) -> int:
        """Get the cached random seed (backward compatibility)."""
        return self._get_network_seed()
