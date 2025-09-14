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
    UNIFORM_DEPARTURE_PATTERN, FIXED_START_TIME_HOUR, FIXED_END_TIME,
    DEFAULT_START_TIME_HOUR, DEFAULT_END_TIME,
    SENTINEL_START_TIME_HOUR, SENTINEL_END_TIME
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
    _validate_departure_pattern(args.departure_pattern)
    _validate_junctions_to_remove(args.junctions_to_remove)
    _validate_lane_count(args.lane_count)

    # NEW: Custom lanes validation
    _validate_custom_lanes(getattr(args, 'custom_lanes', None))
    _validate_custom_lanes_file(getattr(args, 'custom_lanes_file', None))

    # Cross-argument validations
    _validate_cross_arguments(args)
    _validate_sample_arguments(args)

    # NEW: Custom lanes cross-validation
    _validate_custom_lanes_cross_arguments(args)


def _validate_numeric_ranges(args) -> None:
    """Validate numeric argument ranges."""

    # Grid dimension validation
    if args.grid_dimension <= 0:
        raise ValidationError(
            f"Grid dimension must be > 0, got {args.grid_dimension}")
    if args.grid_dimension > 20:
        raise ValidationError(
            f"Grid dimension must be ≤ 20 for performance, got {args.grid_dimension}")

    # Block size validation
    if args.block_size_m <= 0:
        raise ValidationError(
            f"Block size must be > 0, got {args.block_size_m}")
    if args.block_size_m < 50 or args.block_size_m > 1000:
        raise ValidationError(
            f"Block size should be 50-1000m for realism, got {args.block_size_m}")

    # Number of vehicles validation
    if args.num_vehicles <= 0:
        raise ValidationError(
            f"Number of vehicles must be > 0, got {args.num_vehicles}")
    if args.num_vehicles > 10000:
        raise ValidationError(
            f"Number of vehicles must be ≤ 10000 for performance, got {args.num_vehicles}")

    # Step length validation
    if args.step_length <= 0:
        raise ValidationError(
            f"Step length must be > 0, got {args.step_length}")
    if args.step_length < 0.1 or args.step_length > 10.0:
        raise ValidationError(
            f"Step length should be 0.1-10.0 seconds, got {args.step_length}")

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

    valid_types = {"passenger", "commercial", "public"}
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


def _validate_departure_pattern(departure_pattern: str) -> None:
    """Validate departure pattern format."""

    # Check for basic patterns
    if departure_pattern in ["six_periods", "uniform"]:
        return

    # Check for rush_hours pattern
    if departure_pattern.startswith("rush_hours:"):
        _validate_rush_hours_pattern(departure_pattern)
        return

    raise ValidationError(f"Invalid departure pattern: {departure_pattern}. "
                          f"Valid patterns: 'six_periods', 'uniform', 'rush_hours:...'")


def _validate_rush_hours_pattern(pattern: str) -> None:
    """Validate rush_hours pattern format."""

    # Pattern: rush_hours:7-9:40,17-19:30,rest:10
    pattern_part = pattern[len("rush_hours:"):]
    parts = pattern_part.split(",")

    if len(parts) < 2:
        raise ValidationError(
            f"Rush hours pattern must have at least one time range and 'rest' percentage")

    # Check that last part is 'rest:XX'
    rest_part = parts[-1]
    if not rest_part.startswith("rest:"):
        raise ValidationError(
            f"Rush hours pattern must end with 'rest:XX', got: {rest_part}")

    try:
        rest_percentage = float(rest_part[5:])
        if rest_percentage < 0 or rest_percentage > 100:
            raise ValidationError(
                f"Rest percentage must be 0-100, got {rest_percentage}")
    except ValueError:
        raise ValidationError(f"Invalid rest percentage in: {rest_part}")

    # Validate time ranges
    total_percentage = rest_percentage
    for part in parts[:-1]:
        if ":" not in part:
            raise ValidationError(f"Invalid time range format: {part}")

        time_range, percentage_str = part.rsplit(":", 1)
        try:
            percentage = float(percentage_str)
            if percentage < 0 or percentage > 100:
                raise ValidationError(
                    f"Percentage must be 0-100, got {percentage}")
            total_percentage += percentage
        except ValueError:
            raise ValidationError(f"Invalid percentage in: {part}")

        # Validate time range format (e.g., "7-9")
        if "-" not in time_range:
            raise ValidationError(f"Invalid time range format: {time_range}")

        start_hour, end_hour = time_range.split("-", 1)
        try:
            start = float(start_hour)
            end = float(end_hour)
            if start < 0 or start >= 24 or end < 0 or end >= 24:
                raise ValidationError(f"Hours must be 0-24, got {start}-{end}")
            if start >= end:
                raise ValidationError(
                    f"Start hour must be < end hour, got {start}-{end}")
        except ValueError:
            raise ValidationError(f"Invalid hour values in: {time_range}")

    if abs(total_percentage - 100.0) > 0.01:
        raise ValidationError(
            f"Rush hours percentages must sum to 100, got {total_percentage}")


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
        junction_pattern = re.compile(r"^[A-Z]+\d+$")

        for junction_id in junction_ids:
            if not junction_pattern.match(junction_id):
                raise ValidationError(f"Invalid junction ID format: {junction_id}. "
                                      f"Must be format like 'A1', 'B2', 'C10', etc.")
    else:
        # Single junction ID
        junction_pattern = re.compile(r"^[A-Z]+\d+$")
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
        if count < 1 or count > 5:
            raise ValidationError(f"Fixed lane count must be 1-5, got {count}")
    except ValueError:
        raise ValidationError(f"Invalid lane count: {lane_count}. "
                              f"Must be 'realistic', 'random', or integer 1-5")


def _validate_cross_arguments(args) -> None:
    """Validate cross-argument constraints."""

    if args.end_time < 1:
        raise ValidationError("end_time must be positive")

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
    import re

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
        edge_pattern = re.compile(r'^[A-Z]+\d+[A-Z]+\d+$')
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
                if tail_lanes < 1 or tail_lanes > 3:
                    raise ValidationError(
                        f"Tail lanes must be 1-3, got {tail_lanes} in: {edge_config}")
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
                        if lanes < 1 or lanes > 3:
                            raise ValidationError(
                                f"Movement lanes must be 1-3, got {lanes} for {to_edge}")
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
            args.end_time = FIXED_END_TIME
            logger.info(
                f"Auto-setting end_time to {FIXED_END_TIME} for '{args.departure_pattern}' pattern")
        elif args.end_time != FIXED_END_TIME:
            # User explicitly provided wrong value
            raise ValidationError(
                f"end_time must be {FIXED_END_TIME} seconds (24 hours) for '{args.departure_pattern}' "
                f"departure pattern (got {args.end_time}). "
                f"Only '{UNIFORM_DEPARTURE_PATTERN}' pattern allows custom durations."
            )
    else:
        # For uniform pattern, convert sentinel values to real defaults
        if args.start_time_hour == SENTINEL_START_TIME_HOUR:
            args.start_time_hour = DEFAULT_START_TIME_HOUR
        if args.end_time == SENTINEL_END_TIME:
            args.end_time = DEFAULT_END_TIME
