"""
Price-based route choice for vehicles (T7 component).
"""

import random
import logging
from typing import List, Dict
import traci


class PriceBasedRouteChoice:
    """Price-based route choice for vehicles (T7 research objective)."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # 80% of vehicles consider price-based rerouting (INCREASED)
        self.reroute_probability = 0.8
        # How much vehicles care about price (0-1) (INCREASED sensitivity)
        self.price_sensitivity = 0.5
