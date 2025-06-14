# src/traffic/routing.py
from __future__ import annotations
from typing import List

class RoutingStrategy:
    """Abstract interface."""
    def compute_route(self, start_edge: str, end_edge: str) -> List[str]: ...


class ShortestPathRoutingStrategy(RoutingStrategy):
    """
    Uses SUMO-lib’s built-in Dijkstra.
    """

    def __init__(self, net):
        self.net = net  # sumolib.net.Net

    def compute_route(self, start_edge, end_edge):
        result = self.net.getShortestPath(
            self.net.getEdge(start_edge),
            self.net.getEdge(end_edge)
        )
        if not result or result[0] is None:
            return []
        edge_objs, _ = result
        return [e.getID() for e in edge_objs]
