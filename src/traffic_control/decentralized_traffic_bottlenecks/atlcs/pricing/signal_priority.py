"""
Traffic light signal priority calculation for T7 ATLCS implementation.
Replaces route_choice.py - T7 is about traffic light control, not vehicle routing.
"""

import logging
from typing import Dict, List
import traci


class SignalPriorityCalculator:
    """Calculate traffic light signal priorities based on dynamic pricing data."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # How responsive to price changes (0-1)
        self.priority_sensitivity = 0.7
        # Minimum green time extension (seconds)
        self.min_extension = 5
        # Maximum green time extension (seconds)
        self.max_extension = 20

    def _calculate_edge_priority(self, edge_id: str, price: float, max_price: float) -> Dict:
        """Calculate priority data for a specific edge."""

        # Normalize price to priority level (0.0 to 1.0)
        priority_level = (price / max_price) if max_price > 0 else 0.0

        # Apply sensitivity factor
        adjusted_priority = priority_level * self.priority_sensitivity

        # Determine signal control action
        if adjusted_priority >= 0.8:
            action = 'extend_major'
            extension_seconds = self.max_extension
            description = 'High congestion - major green extension'
        elif adjusted_priority >= 0.6:
            action = 'extend_moderate'
            extension_seconds = int(self.max_extension * 0.6)
            description = 'Moderate congestion - moderate extension'
        elif adjusted_priority >= 0.4:
            action = 'extend_minor'
            extension_seconds = self.min_extension
            description = 'Light congestion - minor extension'
        elif adjusted_priority >= 0.2:
            action = 'maintain'
            extension_seconds = 0
            description = 'Normal flow - maintain timing'
        else:
            action = 'optimize'
            extension_seconds = -2
            description = 'Low usage - slight optimization'

        return {
            'edge_id': edge_id,
            'priority_level': priority_level,
            'adjusted_priority': adjusted_priority,
            'action': action,
            'extension_seconds': extension_seconds,
            'description': description,
            'price': price
        }
