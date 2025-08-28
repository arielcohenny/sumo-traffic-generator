"""
Research-grade Vehicle Hours cost calculation (T6 enhancement).
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class VHCostData:
    """Container for VH cost calculation results."""
    vh_costs: Dict[str, float]
    step: int


class VHCostCalculator:
    """Research-grade Vehicle Hours cost calculation (T6 enhancement)."""

    def calculate_research_grade_vh(self, all_links, current_trees) -> VHCostData:
        """USED BY: BottleneckDetector.detect_and_prioritize()"""

        # Enhanced VH calculation using research proposal formula:
        # C(t) = d_ij * (1/v(t) - 1/v_qmax) * q(t)*N / (60/T)
        vh_costs = {}

        # ENHANCEMENT: Use Tree Method bottleneck data to focus calculation (when available)
        tree_bottleneck_links = set()
        tree_method_available = current_trees and hasattr(
            current_trees, 'all_trees_per_iteration')

        if tree_method_available:
            # Get links that are part of Tree Method's bottleneck trees
            tree_bottleneck_links = set(
                current_trees.all_trees_per_iteration.keys())

        # Calculate VH costs for all loaded links, with emphasis on Tree Method bottlenecks
        bottleneck_count = 0
        loaded_links = 0
        processed_links = 0

        for link in all_links:
            # ENHANCED: Combine Tree Method detection with intelligent analysis
            tree_method_bottleneck = link.is_loaded if hasattr(
                link, 'is_loaded') else False
            intelligent_bottleneck = self._is_actually_congested(link)

            # A link is a bottleneck if either Tree Method or intelligent analysis detects it
            is_bottleneck = tree_method_bottleneck or intelligent_bottleneck

            if is_bottleneck:
                loaded_links += 1

            if is_bottleneck:
                distance = link.distance_meters  # d_ij
                # v(t)
                current_speed = link.calc_iteration_speed[-1] if link.calc_iteration_speed else link.free_flow_v_km_h
                max_speed = link.free_flow_v_km_h  # v_qmax
                # q(t)
                flow_rate = link.iteration_data[-1].current_flow_per_iter if link.iteration_data else 0

                # Research formula implementation
                speed_factor = (1/current_speed - 1 /
                                max_speed) if current_speed > 0 else 0
                base_vh_cost = distance * speed_factor * flow_rate

                # ENHANCEMENT: Apply multiplier for Tree Method identified bottlenecks (when available)
                if tree_method_available and link.link_id in tree_bottleneck_links:
                    # Amplify VH cost for Tree Method bottlenecks (research enhancement)
                    vh_cost = base_vh_cost * 1.5  # 50% increase for Tree Method bottlenecks
                    bottleneck_count += 1
                else:
                    vh_cost = base_vh_cost

                processed_links += 1

                # Only include links with meaningful cost
                if vh_cost > 0.01:  # Threshold to avoid noise
                    vh_costs[link.link_id] = vh_cost

        return VHCostData(vh_costs, 0)  # step will be set by caller

    def _is_actually_congested(self, link) -> bool:
        """Check if link is actually congested (not just has traffic)."""

        # Check if link has meaningful traffic data
        if not hasattr(link, 'iteration_data') or not link.iteration_data or len(link.iteration_data) == 0:
            return False

        current_iteration = link.iteration_data[-1]

        # Must have some traffic flow
        if current_iteration.current_flow_per_iter <= 0:
            return False

        # Calculate current speed
        current_speed = link.calc_iteration_speed[-1] if hasattr(
            link, 'calc_iteration_speed') and link.calc_iteration_speed else link.free_flow_v_km_h
        free_flow_speed = link.free_flow_v_km_h

        # SIMPLIFIED: Any speed reduction with decent flow = congestion
        speed_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
        has_flow = current_iteration.current_flow_per_iter > 100.0
        has_speed_reduction = speed_ratio < 0.98  # Even 2% reduction counts

        is_congested = has_speed_reduction and has_flow

        return is_congested
