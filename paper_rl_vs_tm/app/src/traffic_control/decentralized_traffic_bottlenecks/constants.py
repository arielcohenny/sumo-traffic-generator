"""
Constants for the decentralized traffic bottlenecks module.

Centralizes hardcoded values from:
- atlcs/pricing/engine.py
- atlcs/pricing/demand_supply_coordinator.py
- atlcs/enhancements/prioritization.py
- atlcs/enhancements/vh_calculator.py
"""

# =============================================================================
# PRICING THRESHOLDS
# =============================================================================

BASE_PRICE = 5.0                      # Base price for edge pricing
HIGH_PRIORITY_THRESHOLD = 0.8         # Priority threshold for high priority
MEDIUM_PRIORITY_THRESHOLD = 0.5       # Priority threshold for medium priority
LOW_PRIORITY_THRESHOLD = 0.3          # Priority threshold for low priority
HIGH_PRIORITY_EXTENSION = 15          # Green time extension for high priority
MEDIUM_PRIORITY_EXTENSION = 8         # Green time extension for medium priority
LOW_PRIORITY_EXTENSION = 0            # Green time extension for low priority
NEGATIVE_PRIORITY_EXTENSION = -3      # Negative extension for low priority

# =============================================================================
# SEVERITY THRESHOLDS
# =============================================================================

SEVERITY_HIGH = 0.8                   # High severity threshold
SEVERITY_MEDIUM = 0.5                 # Medium severity threshold
SEVERITY_LOW = 0.3                    # Low severity threshold
EXTENSION_HIGH = 25                   # Extension seconds for high severity
EXTENSION_MEDIUM = 15                 # Extension seconds for medium severity
EXTENSION_LOW = 10                    # Extension seconds for low severity
PRICE_MULTIPLIER = 1.5                # Multiplier for price calculations

# =============================================================================
# PRIORITIZATION CONSTANTS
# =============================================================================

MIN_SEVERITY_THRESHOLD = 0.2          # Minimum severity threshold for bottleneck detection
MIN_MAX_BOTTLENECKS = 5               # Minimum number of bottlenecks to consider

# =============================================================================
# VH CALCULATOR CONSTANTS
# =============================================================================

TREE_METHOD_MULTIPLIER = 1.5          # Multiplier for Tree Method calculations
VH_COST_THRESHOLD = 0.01              # Threshold for VH cost calculations
FLOW_THRESHOLD = 100.0                # Flow threshold in vehicles/hour
SPEED_RATIO_THRESHOLD = 0.98          # Speed ratio threshold for congestion detection
