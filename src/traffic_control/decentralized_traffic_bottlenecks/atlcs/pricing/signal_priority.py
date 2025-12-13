"""
Traffic light signal priority calculation for ATLCS implementation.
Replaces route_choice.py - ATLCS is about traffic light control, not vehicle routing.
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
