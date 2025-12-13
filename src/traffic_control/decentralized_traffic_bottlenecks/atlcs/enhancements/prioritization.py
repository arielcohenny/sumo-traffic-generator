"""
Global impact assessment and bottleneck prioritization (T6 enhancement).
"""

from typing import Dict, List
from dataclasses import dataclass

from ...constants import MIN_SEVERITY_THRESHOLD, MIN_MAX_BOTTLENECKS


@dataclass
class PrioritizedBottleneck:
    """Container for prioritized bottleneck information."""
    edge_id: str
    severity: float
    direction: str
    global_impact: float


class GlobalPrioritizer:
    """Global impact assessment for bottleneck prioritization (T6 research objective)."""

    def prioritize_globally(self, enhanced_costs: Dict[str, float], all_links) -> List[PrioritizedBottleneck]:
        """USED BY: BottleneckDetector.detect_and_prioritize()"""

        prioritized_bottlenecks = []

        # Calculate global impact for each bottleneck
        for link in all_links:
            if link.link_id in enhanced_costs:
                vh_cost = enhanced_costs[link.link_id]

                # Calculate severity based on VH cost relative to other links
                max_cost = max(enhanced_costs.values()
                               ) if enhanced_costs else 1.0
                severity = vh_cost / max_cost if max_cost > 0 else 0.0

                # Calculate global impact (simplified algorithm)
                global_impact = severity * link.lanes * link.distance_meters

                bottleneck = PrioritizedBottleneck(
                    edge_id=link.link_id,
                    severity=severity,
                    direction="unknown",  # Would need traffic flow direction analysis
                    global_impact=global_impact
                )

                prioritized_bottlenecks.append(bottleneck)

        # Sort by global impact (highest first)
        prioritized_bottlenecks.sort(
            key=lambda b: b.global_impact, reverse=True)

        # FILTER: Apply minimum severity threshold to reduce noise
        filtered_bottlenecks = [
            b for b in prioritized_bottlenecks if b.severity > MIN_SEVERITY_THRESHOLD]

        # LIMIT: Keep only top bottlenecks to focus on real problems (max 25% of edges)
        max_bottlenecks = max(MIN_MAX_BOTTLENECKS, len(all_links) // 4)
        final_bottlenecks = filtered_bottlenecks[:max_bottlenecks]

        return final_bottlenecks
