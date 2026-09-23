# src/traffic/vehicle_types.py
from typing import Dict


def parse_vehicle_types(vehicle_types_arg: str) -> Dict[str, float]:
    """
    Parse vehicle types argument into percentages dictionary.

    Examples:
    - "passenger 90 public 10" -> {"passenger": 90.0, "public": 10.0}
    - "passenger 80 public 20" -> {"passenger": 80.0, "public": 20.0}
    - "passenger 100" -> {"passenger": 100.0}

    Args:
        vehicle_types_arg: Space-separated vehicle type names and percentages

    Returns:
        Dictionary mapping vehicle type names to percentages

    Raises:
        ValueError: If percentages don't sum to 100 or invalid format
    """
    if not vehicle_types_arg.strip():
        from src.config import CONFIG
        return CONFIG.default_vehicle_distribution.copy()

    tokens = vehicle_types_arg.strip().split()
    if len(tokens) % 2 != 0:
        raise ValueError(
            "Vehicle types format: 'type1 percentage1 type2 percentage2 ...'")

    valid_types = {"passenger", "public"}
    percentages = {}

    for i in range(0, len(tokens), 2):
        vehicle_type = tokens[i]
        try:
            percentage = float(tokens[i + 1])
        except ValueError:
            raise ValueError(f"Invalid percentage value: {tokens[i + 1]}")

        if vehicle_type not in valid_types:
            raise ValueError(
                f"Unknown vehicle type: {vehicle_type}. Valid options: {valid_types}")

        if percentage < 0 or percentage > 100:
            raise ValueError(
                f"Percentage must be between 0 and 100, got {percentage}")

        percentages[vehicle_type] = percentage

    # Validate sum
    total = sum(percentages.values())
    if abs(total - 100.0) > 0.01:
        raise ValueError(
            f"Vehicle type percentages must sum to 100, got {total}")

    return percentages


def get_vehicle_weights(vehicle_distribution: Dict[str, float]) -> tuple:
    """
    Convert vehicle type percentages to weights list in consistent order.

    Args:
        vehicle_distribution: Dictionary with vehicle type percentages

    Returns:
        Tuple of (vehicle_type_names, weights) in consistent order
    """
    # Ensure consistent ordering
    ordered_types = ["passenger", "public"]

    vehicle_names = []
    weights = []

    for vehicle_type in ordered_types:
        if vehicle_type in vehicle_distribution:
            vehicle_names.append(vehicle_type)
            # Convert percentage to weight
            weights.append(vehicle_distribution[vehicle_type] / 100.0)

    return vehicle_names, weights
