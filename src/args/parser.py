"""
Command line argument parser for SUMO traffic generator.

This module handles all argument parsing configuration, keeping the CLI
focused solely on delegation.
"""

import argparse
from src.config import CONFIG


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
        default=5,
        help="The grid's number of rows and columns. Default is 5 (5x5 grid)."
    )
    parser.add_argument(
        "--block_size_m",
        type=int,
        default=200,
        help="Block size in meters. Default is 200m."
    )
    parser.add_argument(
        "--junctions_to_remove",
        type=str,
        default="0",
        help="Number of junctions to remove from the grid (e.g., '5') or comma-separated list of specific junction IDs (e.g., 'A0,B1,C2'). Default is 0."
    )
    parser.add_argument(
        "--lane_count",
        type=str,
        default="realistic",
        help="Lane count algorithm: 'realistic' (default, zone-based), 'random', or integer (fixed count for all edges)."
    )
    parser.add_argument(
        "--osm_file",
        type=str,
        help="Path to OSM file to use instead of generating synthetic grid network"
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
        default=CONFIG.DEFAULT_NUM_VEHICLES,
        help=f"Number of vehicles to generate. Default is {CONFIG.DEFAULT_NUM_VEHICLES}."
    )
    parser.add_argument(
        "--routing_strategy",
        type=str,
        default="shortest 100",
        help="Routing strategy with percentages (e.g., 'shortest 70 realtime 30' or 'shortest 20 realtime 30 fastest 45 attractiveness 5'). Default: 'shortest 100'"
    )
    parser.add_argument(
        "--vehicle_types",
        type=str,
        default=CONFIG.DEFAULT_VEHICLE_TYPES,
        help="Vehicle types with percentages (e.g., 'passenger 70 commercial 20 public 10'). Default: 'passenger 60 commercial 30 public 10'"
    )
    parser.add_argument(
        "--departure_pattern",
        type=str,
        default="six_periods",
        help="Vehicle departure pattern: 'six_periods' (default, research-based), 'uniform', 'rush_hours:7-9:40,17-19:30,rest:10', or 'hourly:7:25,8:35,rest:5'"
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add simulation control arguments."""
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for generating randomness. If not provided, a random seed will be used."
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=1.0,
        help="Simulation step length in seconds (for TraCI loop)."
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=86400,
        help="Total simulation duration in seconds. Default is 86400 (24 hours/full day)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO in GUI mode (sumo-gui) instead of headless sumo"
    )


def _add_zone_arguments(parser: argparse.ArgumentParser) -> None:
    """Add zone and attractiveness arguments."""
    parser.add_argument(
        "--land_use_block_size_m",
        type=float,
        default=25.0,
        help="Size of land use zone grid blocks in meters. Default: 25.0m (following research paper methodology). Controls resolution of zone generation."
    )
    parser.add_argument(
        "--attractiveness",
        type=str,
        default="poisson",
        choices=["poisson", "land_use", "gravity", "iac", "hybrid"],
        help="Edge attractiveness method: 'poisson' (default), 'land_use', 'gravity', 'iac', or 'hybrid'."
    )
    parser.add_argument(
        "--time_dependent",
        action="store_true",
        help="Apply 4-phase time-of-day variations to the selected attractiveness method"
    )
    parser.add_argument(
        "--start_time_hour",
        type=float,
        default=0.0,
        help="Real-world hour when simulation starts (0-24, default: 0.0 for midnight)"
    )


def _add_traffic_control_arguments(parser: argparse.ArgumentParser) -> None:
    """Add traffic control arguments."""
    parser.add_argument(
        "--traffic_light_strategy",
        type=str,
        default="opposites",
        choices=["opposites", "incoming"],
        help="Traffic light phasing strategy: 'opposites' (default, opposing directions together) or 'incoming' (each edge gets own phase)"
    )
    parser.add_argument(
        "--traffic_control",
        type=str,
        default="tree_method",
        choices=["tree_method", "atlcs", "actuated", "fixed"],
        help="Traffic control method: 'tree_method' (default, Tree Method algorithm), 'atlcs' (Adaptive Traffic Light Control System with T6/T7 research), 'actuated' (SUMO gap-based), or 'fixed' (static timing)."
    )
    parser.add_argument(
        "--t6_interval",
        type=int,
        default=10,
        help="T6 bottleneck detection interval in seconds for ATLCS (default: 10)."
    )
    parser.add_argument(
        "--t7_interval", 
        type=int,
        default=5,
        help="T7 pricing update interval in seconds for ATLCS (default: 5)."
    )
    parser.add_argument(
        "--tree-method-interval",
        type=int,
        default=90,
        metavar="SECONDS",
        help="Tree Method calculation interval in seconds (default: 90). Controls how often Tree Method algorithm runs its optimization calculations. Valid range: 30-300 seconds."
    )


def _add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sample testing arguments."""
    parser.add_argument(
        "--tree_method_sample",
        type=str,
        metavar="FOLDER_PATH",
        help="Use pre-built Tree Method sample from specified folder (skips steps 1-8, goes directly to simulation)"
    )