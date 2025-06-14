# src/traffic/edge_sampler.py
from __future__ import annotations
import random
from typing import List

class EdgeSampler:
    """Abstract interface."""

    def sample_start_edges(self, edges: List, n: int) -> List[str]: ...
    def sample_end_edges(self, edges: List, n: int) -> List[str]: ...


class AttractivenessBasedEdgeSampler(EdgeSampler):
    """
    Samples edges proportionally to custom attributes:
      • depart_attractiveness  – for start edges
      • arrive_attractiveness  – for end edges
    Falls back to uniform sampling if weights are all zero / missing.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    # ---------- helpers ----------
    @staticmethod
    def _weights(edges, attr):
        w = [float(getattr(e, attr, 0.0) or 0.0) for e in edges]
        return w if any(w) else [1.0] * len(edges)

    def _choose(self, edges, weights, k):
        return [e.getID() for e in self.rng.choices(edges, weights=weights, k=k)]

    # ---------- API ----------
    def sample_start_edges(self, edges, n):
        return self._choose(edges, self._weights(edges, "depart_attractiveness"), n)

    def sample_end_edges(self, edges, n):
        return self._choose(edges, self._weights(edges, "arrive_attractiveness"), n)
