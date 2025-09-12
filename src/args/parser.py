"""
Command line argument parser for SUMO traffic generator.

This module handles all argument parsing configuration, keeping the CLI
focused solely on delegation.
"""

import argparse
from src.config import CONFIG
from src.constants import (
    DEFAULT_GRID_DIMENSION, DEFAULT_BLOCK_SIZE_M, DEFAULT_JUNCTIONS_TO_REMOVE,
    DEFAULT_LANE_COUNT, DEFAULT_NUM_VEHICLES, DEFAULT_ROUTING_STRATEGY,
    DEFAULT_VEHICLE_TYPES, DEFAULT_DEPARTURE_PATTERN, DEFAULT_STEP_LENGTH,
    DEFAULT_END_TIME, DEFAULT_LAND_USE_BLOCK_SIZE_M, DEFAULT_ATTRACTIVENESS,
    DEFAULT_START_TIME_HOUR, DEFAULT_TRAFFIC_LIGHT_STRATEGY, 
    DEFAULT_TRAFFIC_CONTROL, DEFAULT_BOTTLENECK_DETECTION_INTERVAL,
    DEFAULT_ATLCS_INTERVAL, DEFAULT_TREE_METHOD_INTERVAL,
    MIN_TREE_METHOD_INTERVAL, MAX_TREE_METHOD_INTERVAL, DEFAULT_WORKSPACE_DIR
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Generate and simulate a SUMO orthogonal grid network with dynamic traffic-light control."
    )

    # Network Generation Arguments
    _add_network_arguments(parser)

    # Traffic Generation Arguments
    _add_traffic_arguments(parser)

    # Simulation Arguments
    _add_simulation_arguments(parser)

    # Zone and Attractiveness Arguments
    _add_zone_arguments(parser)

    # Traffic Control Arguments
    _add_traffic_control_arguments(parser)

    # Sample Testing Arguments
    _add_sample_arguments(parser)

    return parser


def _add_network_arguments(parser: argparse.ArgumentParser) -> None:
    """Add network generation arguments."""
    parser.add_argument(
        "--grid_dimension",
        type=float,
        default=DEFAULT_GRID_DIMENSION,
        help=f"The grid's number of rows and columns. Default is {DEFAULT_GRID_DIMENSION} ({DEFAULT_GRID_DIMENSION}x{DEFAULT_GRID_DIMENSION} grid)."
    )
    parser.add_argument(
        "--block_size_m",
        type=int,
        default=DEFAULT_BLOCK_SIZE_M,
        help=f"Block size in meters. Default is {DEFAULT_BLOCK_SIZE_M}m."
    )
    parser.add_argument(
        "--junctions_to_remove",
        type=str,
        default=DEFAULT_JUNCTIONS_TO_REMOVE,
        help=f"Number of junctions to remove from the grid (e.g., '5') or comma-separated list of specific junction IDs (e.g., 'A0,B1,C2'). Default is {DEFAULT_JUNCTIONS_TO_REMOVE}."
    )
    parser.add_argument(
        "--lane_count",
        type=str,
        default=DEFAULT_LANE_COUNT,
        help=f"Lane count algorithm: '{DEFAULT_LANE_COUNT}' (default, zone-based), 'random', or integer (fixed count for all edges)."
    )
    parser.add_argument(
        "--custom_lanes",
        type=str,
        help="Custom lane definitions for specific edges (format: 'EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=...')"
    )
    parser.add_argument(
        "--custom_lanes_file",
        type=str,
        help="File containing custom lane definitions (one configuration per line, same format as --custom_lanes)"
    )


def _add_traffic_arguments(parser: argparse.ArgumentParser) -> None:
    """Add traffic generation arguments."""
    parser.add_argument(
        "--num_vehicles",
        type=int,
        default=DEFAULT_NUM_VEHICLES,
        help=f"Number of vehicles to generate. Default is {DEFAULT_NUM_VEHICLES}."
    )
    parser.add_argument(
        "--routing_strategy",
        type=str,
        default=DEFAULT_ROUTING_STRATEGY,
        help=f"Routing strategy with percentages (e.g., 'shortest 70 realtime 30' or 'shortest 20 realtime 30 fastest 45 attractiveness 5'). Default: '{DEFAULT_ROUTING_STRATEGY}'"
    )
    parser.add_argument(
        "--vehicle_types",
        type=str,
        default=DEFAULT_VEHICLE_TYPES,
        help=f"Vehicle types with percentages (e.g., 'passenger 70 commercial 20 public 10'). Default: '{DEFAULT_VEHICLE_TYPES}'"
    )
    parser.add_argument(
        "--departure_pattern",
        type=str,
        default=DEFAULT_DEPARTURE_PATTERN,
        help=f"Vehicle departure pattern: '{DEFAULT_DEPARTURE_PATTERN}' (default, even distribution), 'six_periods' (research-based), 'rush_hours:7-9:40,17-19:30,rest:10'"
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add simulation control arguments."""
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for generating randomness. If not provided, a random seed will be used. "
             "Sets all seeds (network, private-traffic, public-traffic) to the same value."
    )
    parser.add_argument(
        "--network-seed",
        type=int,
        help="Seed for network structure generation (junction removal, lane assignment, "
             "land use, edge attractiveness). If not provided, uses --seed or random seed."
    )
    parser.add_argument(
        "--private-traffic-seed",
        type=int,
        help="Seed for private traffic generation (passenger and commercial vehicles). "
             "If not provided, uses --seed or random seed."
    )
    parser.add_argument(
        "--public-traffic-seed",
        type=int,
        help="Seed for public traffic generation (public vehicles). "
             "If not provided, uses --seed or random seed."
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=DEFAULT_STEP_LENGTH,
        help=f"Simulation step length in seconds (for TraCI loop). Default is {DEFAULT_STEP_LENGTH}."
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=DEFAULT_END_TIME,
        help=f"Total simulation duration in seconds. Default is {DEFAULT_END_TIME} (2 hours)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO in GUI mode (sumo-gui) instead of headless sumo"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=DEFAULT_WORKSPACE_DIR,
        help=f"Parent directory where 'workspace' folder will be created for simulation output files. Default is '{DEFAULT_WORKSPACE_DIR}' (creates './workspace/')."
    )


def _add_zone_arguments(parser: argparse.ArgumentParser) -> None:
    """Add zone and attractiveness arguments."""
    parser.add_argument(
        "--land_use_block_size_m",
        type=float,
        default=DEFAULT_LAND_USE_BLOCK_SIZE_M,
        help=f"Size of land use zone grid blocks in meters. Default: {DEFAULT_LAND_USE_BLOCK_SIZE_M}m (following research paper methodology). Controls resolution of zone generation."
    )
    parser.add_argument(
        "--attractiveness",
        type=str,
        default=DEFAULT_ATTRACTIVENESS,
        choices=["poisson", "land_use", "gravity", "iac", "hybrid"],
        help=f"Edge attractiveness method: '{DEFAULT_ATTRACTIVENESS}' (default), 'land_use', 'gravity', 'iac', or 'hybrid'."
    )
    parser.add_argument(
        "--time_dependent",
        action="store_true",
        help="Apply 4-phase time-of-day variations to the selected attractiveness method"
    )
    parser.add_argument(
        "--start_time_hour",
        type=float,
        default=DEFAULT_START_TIME_HOUR,
        help=f"Real-world hour when simulation starts (0-24, default: {DEFAULT_START_TIME_HOUR} for midnight)"
    )


def _add_traffic_control_arguments(parser: argparse.ArgumentParser) -> None:
    """Add traffic control arguments."""
    parser.add_argument(
        "--traffic_light_strategy",
        type=str,
        default=DEFAULT_TRAFFIC_LIGHT_STRATEGY,
        choices=["opposites", "incoming"],
        help=f"Traffic light phasing strategy: '{DEFAULT_TRAFFIC_LIGHT_STRATEGY}' (default, opposing directions together) or 'incoming' (each edge gets own phase)"
    )
    parser.add_argument(
        "--traffic_control",
        type=str,
        default=DEFAULT_TRAFFIC_CONTROL,
        choices=["tree_method", "atlcs", "actuated", "fixed"],
        help=f"Traffic control method: '{DEFAULT_TRAFFIC_CONTROL}' (default, Tree Method algorithm), 'atlcs' (Adaptive Traffic Light Control System with enhanced bottleneck detection and ATLCS), 'actuated' (SUMO gap-based), or 'fixed' (static timing)."
    )
    parser.add_argument(
        "--bottleneck-detection-interval",
        type=int,
        default=DEFAULT_BOTTLENECK_DETECTION_INTERVAL,
        help=f"Enhanced bottleneck detection interval in seconds for ATLCS (default: {DEFAULT_BOTTLENECK_DETECTION_INTERVAL})."
    )
    parser.add_argument(
        "--atlcs-interval",
        type=int,
        default=DEFAULT_ATLCS_INTERVAL,
        help=f"ATLCS pricing update interval in seconds for ATLCS (default: {DEFAULT_ATLCS_INTERVAL})."
    )
    parser.add_argument(
        "--tree-method-interval",
        type=int,
        default=DEFAULT_TREE_METHOD_INTERVAL,
        metavar="SECONDS",
        help=f"Tree Method calculation interval in seconds (default: {DEFAULT_TREE_METHOD_INTERVAL}). Controls how often Tree Method algorithm runs its optimization calculations. Valid range: {MIN_TREE_METHOD_INTERVAL}-{MAX_TREE_METHOD_INTERVAL} seconds."
    )


def _add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sample testing arguments."""
    parser.add_argument(
        "--tree_method_sample",
        type=str,
        metavar="FOLDER_PATH",
        help="Use pre-built Tree Method sample from specified folder (skips steps 1-8, goes directly to simulation)"
    )
