"""
T6 Enhanced bottleneck detection implementing research proposal objectives.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from .vh_calculator import VHCostCalculator
from .prioritization import GlobalPrioritizer


@dataclass
class BottleneckData:
    """Container for T6 bottleneck detection results."""
    prioritized_bottlenecks: List[Any]
    enhanced_costs: Dict[str, float]
    step: int

    @classmethod
    def empty(cls):
        """Create empty bottleneck data."""
        return cls([], {}, 0)


class BottleneckDetector:
    """Enhanced bottleneck detection implementing research proposal T6 objectives."""

    def __init__(self, graph, network_data):
        self.graph = graph                           # REUSE existing Tree Method graph
        self.network_data = network_data             # REUSE existing network data
        self.vh_calculator = VHCostCalculator()      # → Used by detect_and_prioritize()
        # → Used by detect_and_prioritize()
        self.prioritizer = GlobalPrioritizer()

    def detect_and_prioritize(self, step: int, graph, iteration_trees) -> BottleneckData:
        """MAIN T6 METHOD - Called by ATLCSController.update() every 10s."""

        # REUSE existing Tree Method IterationTrees (no duplication)
        current_trees = iteration_trees[-1] if iteration_trees else None

        # ENHANCE with research-grade VH cost calculation (works with or without Tree Method data)
        vh_cost_data = self.vh_calculator.calculate_research_grade_vh(
            graph.all_links, current_trees
        )

        # NEW: Global prioritization algorithm (T6 research objective)
        prioritized_bottlenecks = self.prioritizer.prioritize_globally(
            vh_cost_data.vh_costs, graph.all_links
        )

        return BottleneckData(prioritized_bottlenecks, vh_cost_data.vh_costs, step)
