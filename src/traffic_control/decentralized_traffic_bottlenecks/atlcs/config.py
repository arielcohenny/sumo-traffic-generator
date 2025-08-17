"""
ATLCS (Adaptive Traffic Light Control System) Configuration

This module contains configuration constants for the ATLCS implementation,
including pricing thresholds and signal extension parameters.

Note: Bottleneck detection and ATLCS timing intervals are now configured via CLI arguments:
--bottleneck-detection-interval and --atlcs-interval (both in seconds)
"""


class ATLCSConfig:
    """Configuration constants for ATLCS (Adaptive Traffic Light Control System)"""

    # ---------- Dynamic Pricing Thresholds ----------
    # High congestion price threshold - indicates severe bottleneck
    HIGH_PRIORITY_THRESHOLD: float = 10.0

    # Medium congestion price threshold - indicates moderate bottleneck
    MEDIUM_PRIORITY_THRESHOLD: float = 5.0

    # ---------- Signal Extension Parameters ----------
    # Extension duration for high priority edges (seconds)
    # Applied when price > HIGH_PRIORITY_THRESHOLD
    HIGH_PRIORITY_EXTENSION_SEC: int = 15

    # Extension duration for medium priority edges (seconds)
    # Applied when price > MEDIUM_PRIORITY_THRESHOLD
    MEDIUM_PRIORITY_EXTENSION_SEC: int = 8


# Global configuration instance
CONFIG = ATLCSConfig()
