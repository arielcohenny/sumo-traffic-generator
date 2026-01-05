"""
Command-line argument validation for SUMO traffic generator.

This module provides comprehensive validation for all CLI arguments,
ensuring consistency and format correctness before pipeline execution.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from src.config import CONFIG
from src.constants import (
    DEPARTURE_PATTERN_SIX_PERIODS, DEPARTURE_PATTERN_UNIFORM, UNIFORM_DEPARTURE_PATTERN, FIXED_START_TIME_HOUR, MAX_END_TIME,
    DEFAULT_START_TIME_HOUR, DEFAULT_END_TIME,
    SENTINEL_START_TIME_HOUR, SENTINEL_END_TIME,
    MAX_GRID_DIMENSION, MIN_BLOCK_SIZE_M, MAX_BLOCK_SIZE_VALIDATION,
    MAX_NUM_VEHICLES_VALIDATION, MIN_STEP_LENGTH, MAX_STEP_LENGTH,
    MIN_LANE_COUNT, MAX_LANE_COUNT,
    JUNCTION_ID_PATTERN, EDGE_ID_PATTERN,
    MIN_LANES_FOR_TL_STRATEGY,
    CUSTOM_PATTERN_PREFIX,
    CUSTOM_WINDOW_SEPARATOR,
    CUSTOM_TIME_PERCENT_SEPARATOR,
    CUSTOM_TIME_RANGE_SEPARATOR,
)
from src.utils.logging import get_logger
from typing import Tuple, List, Dict, Any

try:
    from .errors import ValidationError
except ImportError:
    class ValidationError(RuntimeError):
        pass

__all__ = ["validate_arguments"]


def validate_arguments(args) -> None:
    """
    Validate all command-line arguments for consistency and format correctness.

    Args:
        args: Parsed arguments from argparse

    Raises:
        ValidationError: If any argument is invalid
    """

    # FIRST: Smart parameter fixing for departure pattern constraints
    _smart_fix_departure_pattern_constraints(args)

    # Individual argument validations
    _validate_numeric_ranges(args)
    _validate_routing_strategy(args.routing_strategy)
    _validate_vehicle_types(args.vehicle_types)
    _validate_route_args(args)
    _validate_departure_pattern(args.departure_pattern)
    _validate_junctions_to_remove(args.junctions_to_remove)
    _validate_lane_count(args.lane_count)

    # NEW: Custom lanes validation
    _validate_custom_lanes(getattr(args, 'custom_lanes', None))
    _validate_custom_lanes_file(getattr(args, 'custom_lanes_file', None))

    # Cross-argument validations
    _validate_cross_arguments(args)
    _validate_sample_arguments(args)
    _validate_traffic_light_lane_compatibility(args)

    # NEW: Custom lanes cross-validation
    _validate_custom_lanes_cross_arguments(args)


def _validate_numeric_ranges(args) -> None:
    """Validate numeric argument ranges."""

    # Grid dimension validation
    if args.grid_dimension <= 0:
        raise ValidationError(
            f"Grid dimension must be > 0, got {args.grid_dimension}")
    if args.grid_dimension > MAX_GRID_DIMENSION:
        raise ValidationError(
            f"Grid dimension must be ≤ {MAX_GRID_DIMENSION} for performance, got {args.grid_dimension}")

    # Block size validation
    if args.block_size_m <= 0:
        raise ValidationError(
            f"Block size must be > 0, got {args.block_size_m}")
    if args.block_size_m < MIN_BLOCK_SIZE_M or args.block_size_m > MAX_BLOCK_SIZE_VALIDATION:
        raise ValidationError(
            f"Block size should be {MIN_BLOCK_SIZE_M}-{MAX_BLOCK_SIZE_VALIDATION}m for realism, got {args.block_size_m}")

    # Step length validation
    if args.step_length <= 0:
        raise ValidationError(
            f"Step length must be > 0, got {args.step_length}")
    if args.step_length < MIN_STEP_LENGTH or args.step_length > MAX_STEP_LENGTH:
        raise ValidationError(
            f"Step length should be {MIN_STEP_LENGTH}-{MAX_STEP_LENGTH} seconds, got {args.step_length}")

    # End time validation
    if args.end_time <= 0:
        raise ValidationError(f"End time must be > 0, got {args.end_time}")

    # Start time hour validation
    if args.start_time_hour < 0 or args.start_time_hour >= 24:
        raise ValidationError(
            f"Start time hour must be 0-24, got {args.start_time_hour}")

    # Tree Method interval validation (only validate if using tree_method)
    if hasattr(args, 'tree_method_interval') and args.tree_method_interval is not None:
        if args.tree_method_interval <= 0:
            raise ValidationError(
                f"Tree Method interval must be > 0, got {args.tree_method_interval}")
        if args.tree_method_interval < CONFIG.TREE_METHOD_MIN_INTERVAL_SEC or args.tree_method_interval > CONFIG.TREE_METHOD_MAX_INTERVAL_SEC:
            raise ValidationError(
                f"Tree Method interval should be {CONFIG.TREE_METHOD_MIN_INTERVAL_SEC}-{CONFIG.TREE_METHOD_MAX_INTERVAL_SEC} seconds, got {args.tree_method_interval}")

    # Land use block size validation
    if args.land_use_block_size_m <= 0:
        raise ValidationError(
            f"Land use block size must be > 0, got {args.land_use_block_size_m}")
    if args.land_use_block_size_m < 10 or args.land_use_block_size_m > 100:
        raise ValidationError(
            f"Land use block size should be 10-100m (research paper methodology), got {args.land_use_block_size_m}")


def _validate_routing_strategy(routing_strategy: str) -> None:
    """Validate routing strategy format and percentages."""

    parts = routing_strategy.strip().split()
    if len(parts) % 2 != 0:
        raise ValidationError(
            f"Routing strategy must be pairs of strategy + percentage, got: {routing_strategy}")

    valid_strategies = {"shortest", "realtime", "fastest", "attractiveness"}
    total_percentage = 0.0

    for i in range(0, len(parts), 2):
        strategy = parts[i]
        try:
            percentage = float(parts[i + 1])
        except (ValueError, IndexError):
            raise ValidationError(
                f"Invalid percentage in routing strategy: {routing_strategy}")

        if strategy not in valid_strategies:
            raise ValidationError(
                f"Invalid routing strategy '{strategy}'. Valid strategies: {valid_strategies}")

        if percentage < 0 or percentage > 100:
            raise ValidationError(
                f"Routing strategy percentage must be 0-100, got {percentage} for {strategy}")

        total_percentage += percentage

    if abs(total_percentage - 100.0) > 0.01:
        raise ValidationError(
            f"Routing strategy percentages must sum to 100, got {total_percentage}")


def _validate_vehicle_types(vehicle_types: str) -> None:
    """Validate vehicle types format and percentages."""

    parts = vehicle_types.strip().split()
    if len(parts) % 2 != 0:
        raise ValidationError(
            f"Vehicle types must be pairs of type + percentage, got: {vehicle_types}")

    valid_types = {"passenger", "public"}
    total_percentage = 0.0

    for i in range(0, len(parts), 2):
        vehicle_type = parts[i]
        try:
            percentage = float(parts[i + 1])
        except (ValueError, IndexError):
            raise ValidationError(
                f"Invalid percentage in vehicle types: {vehicle_types}")

        if vehicle_type not in valid_types:
            raise ValidationError(
                f"Invalid vehicle type '{vehicle_type}'. Valid types: {valid_types}")

        if percentage < 0 or percentage > 100:
            raise ValidationError(
                f"Vehicle type percentage must be 0-100, got {percentage} for {vehicle_type}")

        total_percentage += percentage

    if abs(total_percentage - 100.0) > 0.01:
        raise ValidationError(
            f"Vehicle type percentages must sum to 100, got {total_percentage}")


def _validate_route_args(args) -> None:
    """Validate route pattern format and percentages for passenger and public vehicles."""

    # Note: CLI args with hyphens become underscores in argparse
    # Validate passenger routes
    if hasattr(args, 'passenger_routes') and getattr(args, 'passenger_routes', None):
        _validate_single_route_pattern(args.passenger_routes, "passenger")

    # Validate public routes
    if hasattr(args, 'public_routes') and getattr(args, 'public_routes', None):
        _validate_single_route_pattern(args.public_routes, "public")


def _validate_single_route_pattern(route_pattern: str, vehicle_type: str) -> None:
    """Validate a single route pattern format and percentages."""

    parts = route_pattern.strip().split()
    if len(parts) != 8:
        raise ValidationError(
            f"{vehicle_type} route pattern must be 4 pairs of pattern + percentage, got: {route_pattern}")

    valid_patterns = {"in", "out", "inner", "pass"}
    total_percentage = 0.0
    found_patterns = set()

    for i in range(0, len(parts), 2):
        pattern = parts[i]
        try:
            percentage = float(parts[i + 1])
        except (ValueError, IndexError):
            raise ValidationError(
                f"Invalid percentage in {vehicle_type} route pattern: {route_pattern}")

        if pattern not in valid_patterns:
            raise ValidationError(
                f"Invalid route pattern '{pattern}'. Valid patterns: {valid_patterns}")

        if pattern in found_patterns:
            raise ValidationError(
                f"Duplicate route pattern '{pattern}' in {vehicle_type} routes: {route_pattern}")
        found_patterns.add(pattern)

        if percentage < 0 or percentage > 100:
            raise ValidationError(
                f"Route pattern percentage must be 0-100, got {percentage} for {pattern}")

        total_percentage += percentage

    # Check that all 4 patterns are present
    if found_patterns != valid_patterns:
        missing = valid_patterns - found_patterns
        raise ValidationError(
            f"{vehicle_type} route pattern must include all 4 patterns (in, out, inner, pass), missing: {missing}")

    if abs(total_percentage - 100.0) > 0.01:
        raise ValidationError(
            f"{vehicle_type} route pattern percentages must sum to 100, got {total_percentage}")


def _validate_departure_pattern(departure_pattern: str, start_hour: float = None, end_time: int = None) -> None:
    """Validate departure pattern format."""

    # Check for basic patterns
    if departure_pattern in [DEPARTURE_PATTERN_SIX_PERIODS, DEPARTURE_PATTERN_UNIFORM]:
        return

    # Check for custom pattern
    if departure_pattern.startswith(CUSTOM_PATTERN_PREFIX):
        if start_hour is None or end_time is None:
            # Basic syntax check only (used during initial arg parsing)
            _parse_custom_pattern(departure_pattern)
        else:
            # Full validation with simulation bounds
            _validate_custom_pattern(departure_pattern, start_hour, end_time)
        return

    raise ValidationError(
        f"Invalid departure pattern: {departure_pattern}. "
        f"Valid patterns: 'six_periods', 'uniform', 'custom:HH:MM-HH:MM,percent;...'"
    )


def _parse_custom_pattern(pattern: str) -> list:
    """
    Parse custom departure pattern into list of time windows.

    Args:
        pattern: Pattern string like "custom:9:00-9:30,40;10:00-10:45,30"

    Returns:
        List of dicts: [{"start": (h, m), "end": (h, m), "percent": int}, ...]
    """
    if not pattern.startswith(CUSTOM_PATTERN_PREFIX):
        raise ValidationError(f"Custom pattern must start with '{CUSTOM_PATTERN_PREFIX}'")

    pattern_body = pattern[len(CUSTOM_PATTERN_PREFIX):]
    windows = []

    # Split by semicolon, filter empty parts (handles trailing semicolon)
    parts = [p.strip() for p in pattern_body.split(CUSTOM_WINDOW_SEPARATOR) if p.strip()]

    for part in parts:
        # Split time range from percentage: "9:00-9:30,40"
        if CUSTOM_TIME_PERCENT_SEPARATOR not in part:
            raise ValidationError(
                f"Invalid window format '{part}': expected 'HH:MM-HH:MM,percent'"
            )

        time_range, percent_str = part.rsplit(CUSTOM_TIME_PERCENT_SEPARATOR, 1)

        # Parse percentage
        try:
            percent = int(percent_str)
        except ValueError:
            raise ValidationError(f"Invalid percentage '{percent_str}': must be an integer")

        if percent <= 0:
            raise ValidationError(f"Percentage must be positive, got {percent}")

        # Split time range: "9:00-9:30"
        if CUSTOM_TIME_RANGE_SEPARATOR not in time_range:
            raise ValidationError(
                f"Invalid time range '{time_range}': expected 'HH:MM-HH:MM'"
            )

        start_str, end_str = time_range.split(CUSTOM_TIME_RANGE_SEPARATOR, 1)

        # Parse times
        start_time = _parse_time_hhmm(start_str)
        end_time = _parse_time_hhmm(end_str)

        windows.append({
            "start": start_time,
            "end": end_time,
            "percent": percent
        })

    return windows


def _parse_time_hhmm(time_str: str) -> tuple:
    """
    Parse HH:MM time string to (hour, minute) tuple.

    Args:
        time_str: Time string like "9:00" or "14:30"

    Returns:
        Tuple of (hour, minute) as integers
    """
    time_str = time_str.strip()

    if ":" not in time_str:
        raise ValidationError(f"Invalid time format '{time_str}': expected HH:MM")

    parts = time_str.split(":")
    if len(parts) != 2:
        raise ValidationError(f"Invalid time format '{time_str}': expected HH:MM")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        raise ValidationError(f"Invalid time format '{time_str}': hour and minute must be integers")

    if hour < 0 or hour > 23:
        raise ValidationError(f"Invalid time format '{time_str}': hour must be 0-23")

    if minute < 0 or minute > 59:
        raise ValidationError(f"Invalid time format '{time_str}': minutes must be 0-59")

    return (hour, minute)


def _validate_custom_pattern(pattern: str, start_hour: float, end_time: int) -> None:
    """
    Validate custom departure pattern against simulation bounds.

    Args:
        pattern: Pattern string like "custom:9:00-9:30,40;10:00-10:45,30"
        start_hour: Simulation start time in hours (e.g., 8.0 for 8:00 AM)
        end_time: Simulation duration in seconds

    Raises:
        ValidationError: If pattern is invalid
    """
    windows = _parse_custom_pattern(pattern)

    if not windows:
        raise ValidationError("Custom pattern must have at least one time window")

    # Calculate simulation bounds
    sim_start_hour = start_hour
    sim_start_min = int((start_hour % 1) * 60)
    sim_end_hour = start_hour + (end_time / 3600)

    # Validate each window
    total_percent = 0
    for window in windows:
        start_h, start_m = window["start"]
        end_h, end_m = window["end"]
        percent = window["percent"]

        total_percent += percent

        # Convert to decimal hours for comparison
        window_start_decimal = start_h + start_m / 60
        window_end_decimal = end_h + end_m / 60

        # Check start before end
        if window_start_decimal >= window_end_decimal:
            raise ValidationError(
                f"Invalid window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}: "
                f"start time must be before end time"
            )

        # Check within simulation bounds
        if window_start_decimal < sim_start_hour:
            raise ValidationError(
                f"Window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} is outside "
                f"simulation range {int(sim_start_hour):02d}:{int(sim_start_min):02d}-"
                f"{int(sim_end_hour):02d}:{int((sim_end_hour % 1) * 60):02d}"
            )

        if window_end_decimal > sim_end_hour:
            raise ValidationError(
                f"Window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} is outside "
                f"simulation range {int(sim_start_hour):02d}:{int(sim_start_min):02d}-"
                f"{int(sim_end_hour):02d}:{int((sim_end_hour % 1) * 60):02d}"
            )

    # Check total percentage
    if total_percent > 100:
        raise ValidationError(
            f"Specified percentages sum to {total_percent}%, must be <= 100%"
        )

    # Check for overlaps
    _check_window_overlaps(windows)


def _check_window_overlaps(windows: list) -> None:
    """
    Check that no windows overlap.

    Args:
        windows: List of window dicts with start/end tuples

    Raises:
        ValidationError: If any windows overlap
    """
    # Convert to decimal hours for easier comparison
    ranges = []
    for w in windows:
        start = w["start"][0] + w["start"][1] / 60
        end = w["end"][0] + w["end"][1] / 60
        ranges.append((start, end, w))

    # Sort by start time
    ranges.sort(key=lambda x: x[0])

    # Check adjacent pairs for overlap
    for i in range(len(ranges) - 1):
        _, end1, w1 = ranges[i]
        start2, _, w2 = ranges[i + 1]

        if end1 > start2:
            s1, e1 = w1["start"], w1["end"]
            s2, e2 = w2["start"], w2["end"]
            raise ValidationError(
                f"Windows {s1[0]:02d}:{s1[1]:02d}-{e1[0]:02d}:{e1[1]:02d} and "
                f"{s2[0]:02d}:{s2[1]:02d}-{e2[0]:02d}:{e2[1]:02d} overlap"
            )


def _validate_junctions_to_remove(junctions_to_remove: str) -> None:
    """Validate junctions to remove format."""

    if junctions_to_remove == "0":
        return

    # Check if it's a number
    try:
        count = int(junctions_to_remove)
        if count < 0:
            raise ValidationError(
                f"Number of junctions to remove must be ≥ 0, got {count}")
        if count > 100:
            raise ValidationError(
                f"Number of junctions to remove must be ≤ 100, got {count}")
        return
    except ValueError:
        pass

    # Check if it's comma-separated junction IDs
    if "," in junctions_to_remove:
        junction_ids = [j.strip() for j in junctions_to_remove.split(",")]
        junction_pattern = re.compile(JUNCTION_ID_PATTERN)

        for junction_id in junction_ids:
            if not junction_pattern.match(junction_id):
                raise ValidationError(f"Invalid junction ID format: {junction_id}. "
                                      f"Must be format like 'A1', 'B2', 'C10', etc.")
    else:
        # Single junction ID
        junction_pattern = re.compile(JUNCTION_ID_PATTERN)
        if not junction_pattern.match(junctions_to_remove):
            raise ValidationError(f"Invalid junction ID format: {junctions_to_remove}. "
                                  f"Must be format like 'A1', 'B2', 'C10', etc.")


def _validate_lane_count(lane_count: str) -> None:
    """Validate lane count format."""

    # Check for valid algorithm names
    if lane_count in ["realistic", "random"]:
        return

    # Check if it's a fixed integer count
    try:
        count = int(lane_count)
        if count < MIN_LANE_COUNT or count > MAX_LANE_COUNT:
            raise ValidationError(
                f"Fixed lane count must be {MIN_LANE_COUNT}-{MAX_LANE_COUNT}, got {count}")
    except ValueError:
        raise ValidationError(f"Invalid lane count: {lane_count}. "
                              f"Must be 'realistic', 'random', or integer {MIN_LANE_COUNT}-{MAX_LANE_COUNT}")


def _validate_traffic_light_lane_compatibility(args) -> None:
    """Validate lane count is compatible with traffic light strategy.

    partial_opposites strategy requires minimum 2 lanes per edge to separate:
    - Lane 0 (rightmost): straight + right movements
    - Lane 1+ (leftmost): left + u-turn movements
    """
    if args.traffic_light_strategy == "partial_opposites":
        # Check if lane assignment is disabled (should not happen with defaults)
        if args.lane_count == "0":
            raise ValidationError(
                "partial_opposites strategy requires lane assignment. "
                "Cannot use --lane_count 0"
            )

        # Check for explicit fixed lane configuration
        if not args.lane_count.startswith("fixed") and args.lane_count not in ["realistic", "random"]:
            # Try to parse as integer
            try:
                lane_value = int(args.lane_count)
                if lane_value < 2:
                    raise ValidationError(
                        f"partial_opposites strategy requires minimum 2 lanes per edge. "
                        f"You specified: {args.lane_count}. "
                        f"Use '--lane_count 2' or higher, or use 'realistic'/'random' algorithms."
                    )
            except ValueError:
                # Not an integer, validation will be caught elsewhere
                pass


def _validate_cross_arguments(args) -> None:
    """Validate cross-argument constraints."""

    if args.end_time < 1:
        raise ValidationError("end_time must be positive")

    # --hide-zones requires --gui
    if getattr(args, 'hide_zones', False) and not args.gui:
        raise ValidationError(
            "--hide-zones requires --gui to be enabled. "
            "Zones are only visible in GUI mode."
        )

    # Grid dimension vs junctions to remove capacity limits
    max_removable = max(0, int(args.grid_dimension - 2)
                        ** 2)  # Interior junctions only

    # Check if junctions_to_remove is a number
    try:
        count = int(args.junctions_to_remove)
        if count > max_removable:
            raise ValidationError(f"Cannot remove {count} junctions from {args.grid_dimension}x{args.grid_dimension} grid. "
                                  f"Maximum removable interior junctions: {max_removable}")
    except ValueError:
        pass  # It's junction IDs, not a count

    # Traffic light strategy compatibility (currently no restrictions)
    # Could add future constraints here if needed


def _validate_custom_lanes(custom_lanes: str) -> None:
    """Validate custom lanes argument format and values."""
    if not custom_lanes:
        return

    # Split by semicolon to get individual edge configurations
    edge_configs = [config.strip()
                    for config in custom_lanes.split(';') if config.strip()]

    for config in edge_configs:
        _validate_single_edge_config(config)


def _validate_custom_lanes_file(custom_lanes_file: str) -> None:
    """Validate custom lanes file argument."""
    if not custom_lanes_file:
        return

    # Check file existence
    file_path = Path(custom_lanes_file)
    if not file_path.exists():
        raise ValidationError(
            f"Custom lanes file does not exist: {custom_lanes_file}")

    # Check file readability
    if not file_path.is_file():
        raise ValidationError(
            f"Custom lanes path is not a file: {custom_lanes_file}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except PermissionError:
        raise ValidationError(
            f"Cannot read custom lanes file: {custom_lanes_file}")
    except UnicodeDecodeError:
        raise ValidationError(
            f"Custom lanes file must be UTF-8 encoded: {custom_lanes_file}")
    except Exception as e:
        raise ValidationError(
            f"Error reading custom lanes file {custom_lanes_file}: {e}")

    # Validate file content line by line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        try:
            _validate_single_edge_config(line)
        except ValidationError as e:
            raise ValidationError(
                f"Invalid custom lanes syntax on line {line_num}: {e}")


def _validate_single_edge_config(config: str) -> None:
    """Validate a single edge configuration string."""
    # Split by semicolon for multiple edges in one config
    edge_configs = [cfg.strip() for cfg in config.split(';') if cfg.strip()]

    for edge_config in edge_configs:
        # Pattern: EdgeID=tail:N,head:ToEdge:N,ToEdge2:N OR EdgeID=tail:N OR EdgeID=head:ToEdge:N
        if '=' not in edge_config:
            raise ValidationError(
                f"Missing '=' in configuration: {edge_config}")

        edge_id, specification = edge_config.split('=', 1)
        edge_id = edge_id.strip()
        specification = specification.strip()

        # Validate edge ID format (A1B1, B2C2, etc.)
        edge_pattern = re.compile(EDGE_ID_PATTERN)
        if not edge_pattern.match(edge_id):
            raise ValidationError(
                f"Invalid edge ID format: {edge_id} - Must match pattern A1B1, B2C2, etc.")

        # Parse specification - need to handle tail: and head: properly
        tail_found = False
        head_found = False

        # Split by comma but be careful about head: movements
        tail_part = None
        head_part = None

        # Look for tail: and head: sections
        if 'tail:' in specification:
            tail_found = True
            tail_start = specification.find('tail:')
            tail_end = specification.find(',head:', tail_start)
            if tail_end == -1:
                tail_end = len(specification)
            tail_part = specification[tail_start:tail_end].strip()

            tail_value = tail_part[5:].strip()  # Remove 'tail:'
            if not tail_value:
                raise ValidationError(
                    f"Empty tail specification in: {edge_config}")

            try:
                tail_lanes = int(tail_value)
                if tail_lanes < MIN_LANE_COUNT or tail_lanes > 3:
                    raise ValidationError(
                        f"Tail lanes must be {MIN_LANE_COUNT}-3, got {tail_lanes} in: {edge_config}")
            except ValueError:
                raise ValidationError(
                    f"Invalid tail lane count '{tail_value}' in: {edge_config}")

        if 'head:' in specification:
            if head_found:
                raise ValidationError(
                    f"Duplicate head specification in: {edge_config}")
            head_found = True

            head_start = specification.find('head:')
            head_part = specification[head_start:].strip()

            head_value = head_part[5:].strip()  # Remove 'head:'

            # Handle dead-end case (empty head:)
            if not head_value:
                pass  # Valid dead-end syntax
            else:
                # Parse movement specifications: ToEdge1:N,ToEdge2:M
                movements = [mov.strip()
                             for mov in head_value.split(',') if mov.strip()]

                for movement in movements:
                    if ':' not in movement:
                        raise ValidationError(
                            f"Invalid movement format '{movement}' - must be ToEdge:N")

                    to_edge, lane_count = movement.split(':', 1)
                    to_edge = to_edge.strip()
                    lane_count = lane_count.strip()

                    # Validate destination edge ID
                    if not edge_pattern.match(to_edge):
                        raise ValidationError(
                            f"Invalid destination edge ID: {to_edge}")

                    # Validate lane count
                    try:
                        lanes = int(lane_count)
                        if lanes < MIN_LANE_COUNT or lanes > 3:
                            raise ValidationError(
                                f"Movement lanes must be {MIN_LANE_COUNT}-3, got {lanes} for {to_edge}")
                    except ValueError:
                        raise ValidationError(
                            f"Invalid lane count '{lane_count}' for movement {to_edge}")

        # Ensure at least one specification (tail or head)
        if not tail_found and not head_found:
            raise ValidationError(
                f"Configuration must specify at least tail: or head: - got: {edge_config}")


def _validate_custom_lanes_cross_arguments(args) -> None:
    """Validate cross-argument constraints for custom lanes."""

    # Mutually exclusive: cannot use both --custom_lanes and --custom_lanes_file
    if getattr(args, 'custom_lanes', None) and getattr(args, 'custom_lanes_file', None):
        raise ValidationError(
            "Cannot use both --custom_lanes and --custom_lanes_file simultaneously")

    # Custom lanes validation (synthetic grids only)
    custom_lanes_provided = getattr(args, 'custom_lanes', None) or getattr(
        args, 'custom_lanes_file', None)

    # Custom lanes override --lane_count but both can be specified
    # (custom lanes take precedence for specified edges, --lane_count for others)


def _validate_sample_arguments(args) -> None:
    """Validate arguments when using --tree_method_sample.

    Args:
        args: Parsed command line arguments

    Raises:
        ValidationError: If incompatible arguments are used with --tree_method_sample
    """
    if not args.tree_method_sample:
        return

    # Check for incompatible arguments (network generation related)
    incompatible = []
    if args.grid_dimension != 5:  # non-default
        incompatible.append('--grid_dimension')
    if args.block_size_m != 200:  # non-default
        incompatible.append('--block_size_m')
    if args.junctions_to_remove != "0":
        incompatible.append('--junctions_to_remove')
    if args.lane_count != "realistic":
        incompatible.append('--lane_count')

    if incompatible:
        raise ValidationError(
            f"--tree_method_sample incompatible with: {', '.join(incompatible)}")


def _smart_fix_departure_pattern_constraints(args) -> None:
    """Smart parameter fixing for departure pattern constraints.

    Auto-fixes sentinel values (not explicitly provided by user) but validates
    explicit values and shows errors if they are incorrect.

    Args:
        args: Parsed command line arguments (modified in place)

    Raises:
        ValidationError: If user explicitly provided incorrect values
    """
    logger = get_logger(__name__)

    if args.departure_pattern != UNIFORM_DEPARTURE_PATTERN:
        # Handle start_time_hour
        if args.start_time_hour == SENTINEL_START_TIME_HOUR:
            # User didn't provide this parameter, auto-fix it
            args.start_time_hour = FIXED_START_TIME_HOUR
            logger.info(
                f"Auto-setting start_time_hour to {FIXED_START_TIME_HOUR} for '{args.departure_pattern}' pattern")
        elif args.start_time_hour != FIXED_START_TIME_HOUR:
            # User explicitly provided wrong value
            raise ValidationError(
                f"start_time_hour must be {FIXED_START_TIME_HOUR} for '{args.departure_pattern}' "
                f"departure pattern (got {args.start_time_hour}). "
                f"Only '{UNIFORM_DEPARTURE_PATTERN}' pattern allows custom start times."
            )

        # Handle end_time
        if args.end_time == SENTINEL_END_TIME:
            # User didn't provide this parameter, auto-fix it
            args.end_time = MAX_END_TIME
            logger.info(
                f"Auto-setting end_time to {MAX_END_TIME} for '{args.departure_pattern}' pattern")
        elif args.end_time != MAX_END_TIME:
            # User explicitly provided wrong value
            raise ValidationError(
                f"end_time must be {MAX_END_TIME} seconds (24 hours) for '{args.departure_pattern}' "
                f"departure pattern (got {args.end_time}). "
                f"Only '{UNIFORM_DEPARTURE_PATTERN}' pattern allows custom durations."
            )
    else:
        # For uniform pattern, convert sentinel values to real defaults
        if args.start_time_hour == SENTINEL_START_TIME_HOUR:
            args.start_time_hour = DEFAULT_START_TIME_HOUR
        if args.end_time == SENTINEL_END_TIME:
            args.end_time = DEFAULT_END_TIME
