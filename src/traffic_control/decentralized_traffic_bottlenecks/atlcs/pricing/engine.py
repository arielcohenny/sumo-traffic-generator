"""
Dynamic road pricing engine implementing ATLCS objectives.
"""

from typing import Dict, List
from dataclasses import dataclass

# ATLCS is for traffic light control, not vehicle routing


@dataclass
class PricingUpdates:
    """Container for pricing update information."""
    edge_prices: Dict[str, float]
    route_updates: Dict[str, List[str]]
    step: int


class PricingEngine:
    """Dynamic road pricing engine implementing ATLCS objectives."""

    def __init__(self):
        # → Updated by calculate_dynamic_prices()
        self.base_prices = {}
        # → Used for traffic light priority decisions
        self.signal_priority_calculator = None

    def calculate_dynamic_prices(self, bottleneck_data, step: int) -> PricingUpdates:
        """MAIN ATLCS METHOD - Called by ATLCSController.update() every 30s."""

        # Use enhanced bottleneck detection data to calculate congestion-based pricing
        edge_prices = self._calculate_congestion_pricing(
            bottleneck_data.prioritized_bottlenecks)

        # Generate signal priority data for traffic lights
        signal_priorities = self._generate_signal_priorities(edge_prices, step)

        return PricingUpdates(edge_prices, signal_priorities, step)

    def _calculate_congestion_pricing(self, prioritized_bottlenecks) -> Dict[str, float]:
        """USED BY: calculate_dynamic_prices()"""

        edge_prices = {}
        for bottleneck in prioritized_bottlenecks:
            # Dynamic pricing based on bottleneck severity (INCREASED PRICING)
            base_price = 5.0  # Base price per km (INCREASED from 1.0)
            # 1x to 6x pricing (INCREASED)
            congestion_multiplier = 1.0 + (bottleneck.severity * 5.0)
            edge_prices[bottleneck.edge_id] = base_price * \
                congestion_multiplier

        return edge_prices

    def _generate_signal_priorities(self, edge_prices: Dict[str, float], step: int) -> Dict[str, Dict]:
        """USED BY: calculate_dynamic_prices() - Generate traffic light priority data"""

        import logging
        logger = logging.getLogger(self.__class__.__name__)

        # Generate signal priority data based on edge pricing
        signal_priorities = {}

        if not edge_prices:
            return signal_priorities

        # Calculate priority levels based on price thresholds
        max_price = max(edge_prices.values()) if edge_prices else 1.0

        for edge_id, price in edge_prices.items():
            # Normalize price to priority level (0.0 to 1.0)
            priority_level = price / max_price if max_price > 0 else 0.0

            # Determine signal control action based on priority
            if priority_level >= 0.8:  # High priority
                action = 'extend_major'  # Significant green time extension
                extension_seconds = 15
            elif priority_level >= 0.5:  # Medium priority
                action = 'extend_minor'  # Moderate green time extension
                extension_seconds = 8
            elif priority_level >= 0.3:  # Low priority
                action = 'maintain'     # Maintain current timing
                extension_seconds = 0
            else:
                action = 'reduce'       # Slightly reduce green time
                extension_seconds = -3

            signal_priorities[edge_id] = {
                'priority_level': priority_level,
                'action': action,
                'extension_seconds': extension_seconds,
                'price': price
            }

        if signal_priorities:
            logger.debug(
                f"ATLCS: Generated signal priorities for {len(signal_priorities)} edges at step {step}")

        return signal_priorities
