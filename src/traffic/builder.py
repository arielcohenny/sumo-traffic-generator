# src/traffic/builder.py
from __future__ import annotations
import random
from pathlib import Path
from sumolib.net import readNet

from ..config import CONFIG
from src.constants import DEFAULT_VEHICLE_TYPES
from .edge_sampler import AttractivenessBasedEdgeSampler
from .routing import RoutingMixStrategy, parse_routing_strategy
from .vehicle_types import parse_vehicle_types, get_vehicle_weights
from .xml_writer import write_routes

# Constants for departure time generation
MAX_ROUTE_RETRIES = 20
SIMULATION_END_FACTOR = 0.9  # Use 90% of simulation time for departures
SIMULATION_END_FACTOR_SIX_PERIODS = 0.95  # Use 95% for six periods pattern
# 25% in evening part (10pm-12am), 75% early morning
NIGHT_EVENING_RATIO = 0.25

# Six periods time constants (in hours)
MORNING_START = 6.0
MORNING_END = 12.0
MORNING_RUSH_START = 7.5
MORNING_RUSH_END = 9.5
NOON_START = 12.0
NOON_END = 17.0
EVENING_RUSH_START = 17.0
EVENING_RUSH_END = 19.0
EVENING_START = 19.0
EVENING_END = 22.0
NIGHT_START = 22.0
NIGHT_END = 30.0  # Wraps to next day (6am)
EARLY_MORNING_END = 6.0

# Six periods weights
MORNING_WEIGHT = 20
MORNING_RUSH_WEIGHT = 30
NOON_WEIGHT = 25
EVENING_RUSH_WEIGHT = 20
EVENING_WEIGHT = 4
NIGHT_WEIGHT = 1


def generate_vehicle_routes(net_file: str | Path,
                            output_file: str | Path,
                            num_vehicles: int,
                            private_traffic_seed: int = CONFIG.RNG_SEED,
                            public_traffic_seed: int = CONFIG.RNG_SEED,
                            routing_strategy: str = "shortest 100",
                            vehicle_types: str = DEFAULT_VEHICLE_TYPES,
                            end_time: int = 7200,
                            departure_pattern: str = "six_periods") -> None:
    """
    Orchestrates vehicle creation and writes a .rou.xml.

    Args:
        net_file: Path to SUMO network file
        output_file: Output route file path
        num_vehicles: Number of vehicles to generate
        private_traffic_seed: Random seed for private traffic (passenger only)
        public_traffic_seed: Random seed for public traffic
        routing_strategy: Routing strategy specification (e.g., "shortest 70 realtime 30")
        vehicle_types: Vehicle types specification (e.g., "passenger 90 public 10")
        end_time: Total simulation duration in seconds for temporal distribution
        departure_pattern: Departure pattern ("six_periods", "uniform", "rush_hours:7-9:40,17-19:30")
    """
    # Create separate RNGs for private and public traffic
    private_rng = random.Random(private_traffic_seed)
    public_rng = random.Random(public_traffic_seed)
    
    net = readNet(str(net_file))
    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]

    # Create samplers for both traffic types  
    private_sampler = AttractivenessBasedEdgeSampler(private_rng)
    public_sampler = AttractivenessBasedEdgeSampler(public_rng)

    # Parse and initialize routing strategies
    strategy_percentages = parse_routing_strategy(routing_strategy)
    private_routing_mix = RoutingMixStrategy(net, strategy_percentages)
    public_routing_mix = RoutingMixStrategy(net, strategy_percentages)

    # Parse and initialize vehicle types
    vehicle_distribution = parse_vehicle_types(vehicle_types)
    vehicle_names, vehicle_weights = get_vehicle_weights(vehicle_distribution)

    print(f"Using routing strategies: {strategy_percentages}")
    print(f"Using vehicle types: {vehicle_distribution}")
    print(f"Using private traffic seed: {private_traffic_seed}")
    print(f"Using public traffic seed: {public_traffic_seed}")

    # Create a master RNG for vehicle type assignment to maintain distribution
    # This ensures the vehicle type percentages are respected across the fleet
    master_rng = random.Random(private_traffic_seed + public_traffic_seed)
    
    vehicles = []
    for vid in range(num_vehicles):
        # Choose vehicle type using master RNG to maintain distribution
        vtype = master_rng.choices(
            population=vehicle_names,
            weights=vehicle_weights,
            k=1
        )[0]

        # Select appropriate RNG, sampler, and routing mix based on vehicle type
        if vtype == 'passenger':
            # Private traffic
            current_rng = private_rng
            current_sampler = private_sampler
            current_routing_mix = private_routing_mix
        else:  # vtype == 'public'
            # Public traffic
            current_rng = public_rng
            current_sampler = public_sampler
            current_routing_mix = public_routing_mix

        # Assign routing strategy using appropriate RNG
        assigned_strategy = current_routing_mix.assign_strategy_to_vehicle(
            f"veh{vid}", current_rng)

        route_edges = []
        for _ in range(MAX_ROUTE_RETRIES):                       # retry up to 20 times
            start_edge = current_sampler.sample_start_edges(edges, 1)[0]
            end_edge = current_sampler.sample_end_edges(edges, 1)[0]
            if end_edge == start_edge:
                continue
            route_edges = current_routing_mix.compute_route(
                assigned_strategy, start_edge, end_edge)
            if route_edges:
                break
        else:
            print(
                f"⚠️  Could not find a path for vehicle {vid} using {assigned_strategy} strategy; skipping.")
            continue

        # make 100% sure we have a list of edges
        if not route_edges:
            print(f"⚠️  Empty route for vehicle {vid}; skipping.")
            continue
            
        # Generate departure time using appropriate RNG
        departure_time = _generate_departure_time(
            current_rng, departure_pattern, end_time)

        vehicles.append({
            "id":              f"veh{vid}",
            "type":            vtype,
            "depart":          int(departure_time),
            "from_edge":       start_edge,
            "to_edge":         end_edge,
            "route_edges":     route_edges,
            "routing_strategy": assigned_strategy,
        })

    # Sort vehicles by departure time (SUMO requirement)
    vehicles.sort(key=lambda v: v["depart"])

    write_routes(output_file, vehicles, CONFIG.vehicle_types)


def _generate_departure_time(rng, departure_pattern: str, end_time: int) -> int:
    """
    Generate departure time based on the specified pattern.

    Args:
        rng: Random number generator
        departure_pattern: Pattern specification
        end_time: Total simulation duration in seconds

    Returns:
        Departure time in seconds
    """
    if departure_pattern == "uniform":
        return int(rng.uniform(0, end_time * SIMULATION_END_FACTOR))

    elif departure_pattern == "six_periods":
        return _generate_six_periods_departure(rng, end_time)

    elif departure_pattern.startswith("rush_hours:"):
        return _generate_rush_hours_departure(rng, departure_pattern, end_time)


    else:
        # Default to six_periods if unknown pattern
        return _generate_six_periods_departure(rng, end_time)


def _generate_six_periods_departure(rng, end_time: int) -> int:
    """
    Generate departure time using the 6-period system from research paper.

    Time periods (assuming 24-hour simulation):
    - Morning (6am-12pm): 20%
    - Morning Rush (7:30am-9:30am): 30% 
    - Noon (12pm-5pm): 25%
    - Evening Rush (5pm-7pm): 20%
    - Evening (7pm-10pm): 4%
    - Night (10pm-6am): 1%
    """
    # Scale to simulation time (24 hours = 86400 seconds)
    scale_factor = end_time / 86400

    # Define periods in seconds (24-hour format)
    periods = [
        {"name": "morning", "start": MORNING_START*3600,
            "end": MORNING_END*3600, "weight": MORNING_WEIGHT},
        {"name": "morning_rush", "start": MORNING_RUSH_START*3600,
            "end": MORNING_RUSH_END*3600, "weight": MORNING_RUSH_WEIGHT},
        {"name": "noon", "start": NOON_START*3600,
            "end": NOON_END*3600, "weight": NOON_WEIGHT},
        {"name": "evening_rush", "start": EVENING_RUSH_START*3600,
            "end": EVENING_RUSH_END*3600, "weight": EVENING_RUSH_WEIGHT},
        {"name": "evening", "start": EVENING_START*3600,
            "end": EVENING_END*3600, "weight": EVENING_WEIGHT},
        {"name": "night", "start": NIGHT_START*3600,
            "end": NIGHT_END*3600, "weight": NIGHT_WEIGHT},
    ]

    # Choose period based on weights
    weights = [p["weight"] for p in periods]
    chosen_period = rng.choices(periods, weights=weights)[0]

    # Generate time within chosen period
    start_time = chosen_period["start"] * scale_factor
    end_time_period = chosen_period["end"] * scale_factor

    # Handle night period wrapping to next day
    if chosen_period["name"] == "night":
        if rng.random() < NIGHT_EVENING_RATIO:  # 25% in evening part (10pm-12am)
            departure_time = rng.uniform(
                NIGHT_START*3600*scale_factor, 24*3600*scale_factor)
        else:  # 75% in early morning part (12am-6am)
            departure_time = rng.uniform(
                0, EARLY_MORNING_END*3600*scale_factor)
    else:
        departure_time = rng.uniform(start_time, end_time_period)

    return int(min(departure_time, end_time * SIMULATION_END_FACTOR_SIX_PERIODS))


def _generate_rush_hours_departure(rng, pattern: str, end_time: int) -> int:
    """
    Generate departure time using rush hours pattern.
    Format: "rush_hours:7-9:40,17-19:30,rest:10"
    """
    # Parse pattern
    parts = pattern.split(":", 1)[1].split(",")
    rush_periods = []
    rest_weight = 10

    for part in parts:
        if part.startswith("rest:"):
            rest_weight = int(part.split(":")[1])
        else:
            time_range, weight = part.split(":")
            start_hour, end_hour = map(float, time_range.split("-"))
            rush_periods.append({
                "start": start_hour * 3600,
                "end": end_hour * 3600,
                "weight": int(weight)
            })

    # Calculate total weight
    total_rush_weight = sum(p["weight"] for p in rush_periods)
    total_weight = total_rush_weight + rest_weight

    # Choose rush hour or rest time
    if rng.random() < total_rush_weight / total_weight:
        # Choose rush period
        weights = [p["weight"] for p in rush_periods]
        chosen_period = rng.choices(rush_periods, weights=weights)[0]

        scale_factor = end_time / 86400
        start_time = chosen_period["start"] * scale_factor
        end_time_period = chosen_period["end"] * scale_factor
        departure_time = rng.uniform(start_time, end_time_period)
    else:
        # Rest time (uniform distribution outside rush hours)
        departure_time = rng.uniform(0, end_time * SIMULATION_END_FACTOR)

    return int(departure_time)




def execute_route_generation(args) -> None:
    """Execute vehicle route generation."""
    import logging
    from src.utils.multi_seed_utils import get_private_traffic_seed, get_public_traffic_seed
    from src.config import CONFIG
    from src.validate.validate_traffic import verify_generate_vehicle_routes
    from src.validate.errors import ValidationError

    logger = logging.getLogger(__name__)

    generate_vehicle_routes(
        net_file=CONFIG.network_file,
        output_file=CONFIG.routes_file,
        num_vehicles=args.num_vehicles,
        private_traffic_seed=get_private_traffic_seed(args),
        public_traffic_seed=get_public_traffic_seed(args),
        routing_strategy=args.routing_strategy,
        vehicle_types=args.vehicle_types,
        end_time=args.end_time,
        departure_pattern=args.departure_pattern
    )
    try:
        verify_generate_vehicle_routes(
            net_file=CONFIG.network_file,
            output_file=CONFIG.routes_file,
            num_vehicles=args.num_vehicles,
            seed=get_private_traffic_seed(args),  # Keep validation using private seed for backward compatibility
        )
    except ValidationError as ve:
        logger.error(f"Failed to generate vehicle routes: {ve}")
        raise
    logger.info("Generated vehicle routes successfully")
