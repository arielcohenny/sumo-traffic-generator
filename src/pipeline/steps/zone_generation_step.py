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
            seed=self._get_seed(),
            fill_polygons=True,
            inset=0.0
        )

    def _generate_synthetic_zones(self) -> None:
        """Generate zones for synthetic grid networks."""
        self.logger.info("Generating synthetic network zones...")
        extract_zones_from_junctions(
            self.args.land_use_block_size_m,
            seed=self._get_seed(),
            fill_polygons=True,
            inset=0.0
        )
        self.logger.info(
            f"Extracted land use zones successfully with {self.args.land_use_block_size_m}m blocks")

    def _get_seed(self) -> int:
        """Get the cached random seed from network generation step."""
        if hasattr(self.args, '_seed'):
            return self.args._seed

        import random
        seed = self.args.seed if self.args.seed is not None else random.randint(
            0, 2**32 - 1)
        self.args._seed = seed
        return seed
