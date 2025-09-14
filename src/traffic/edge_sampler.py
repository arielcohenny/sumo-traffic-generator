# src/traffic/edge_sampler.py
from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import List

class EdgeSampler(ABC):
    """Abstract interface for edge sampling strategies."""

    @abstractmethod
    def sample_start_edges(self, edges: List, n: int) -> List[str]:
        """Sample n start edges from the given edge list."""
        pass
    
    @abstractmethod
    def sample_end_edges(self, edges: List, n: int) -> List[str]:
        """Sample n end edges from the given edge list."""
        pass


class AttractivenessBasedEdgeSampler(EdgeSampler):
    """
    Samples edges proportionally to phase-specific temporal attributes:
      • {phase}_depart_attractiveness  – for start edges
      • {phase}_arrive_attractiveness  – for end edges
    Falls back to uniform sampling if weights are all zero / missing.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    # ---------- helpers ----------
    def _get_phase_attr(self, edge, direction: str):
        """Get the appropriate phase-specific attractiveness attribute for an edge."""
        # Get current phase from edge attributes (set during network generation)
        current_phase = getattr(edge, 'current_phase', 'morning_peak')
        attr_name = f"{current_phase}_{direction}_attractiveness"
        return float(getattr(edge, attr_name, 0.0) or 0.0)

    def _weights(self, edges, direction: str):
        """Get weights for edges using phase-specific attributes."""
        w = [self._get_phase_attr(e, direction) for e in edges]
        return w if any(w) else [1.0] * len(edges)

    def _choose(self, edges, weights, k):
        return [e.getID() for e in self.rng.choices(edges, weights=weights, k=k)]

    # ---------- API ----------
    def sample_start_edges(self, edges, n):
        return self._choose(edges, self._weights(edges, "depart"), n)

    def sample_end_edges(self, edges, n):
        return self._choose(edges, self._weights(edges, "arrive"), n)
