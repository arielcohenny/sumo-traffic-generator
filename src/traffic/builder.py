# src/traffic/builder.py
from __future__ import annotations
import logging
import random
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
from sumolib.net import readNet

from ..config import CONFIG
from src.utils.multi_seed_utils import get_private_traffic_seed, get_public_traffic_seed
from src.validate.validate_traffic import verify_generate_vehicle_routes
from src.validate.errors import ValidationError
from src.traffic.routing import convert_tail_to_head_edge
from src.constants import (
    DEFAULT_VEHICLE_TYPES, SECONDS_IN_DAY, DEFAULT_VEHICLES_DAILY_PER_ROUTE,
    RUSH_HOUR_MORNING_START, RUSH_HOUR_MORNING_END, RUSH_HOUR_EVENING_START,
    RUSH_HOUR_EVENING_END, TEMPORAL_BIAS_STRENGTH,
    MAX_ROUTE_RETRIES, SIMULATION_END_FACTOR, SIMULATION_END_FACTOR_SIX_PERIODS,
    NIGHT_EVENING_RATIO, SIX_PERIODS_MORNING_START, SIX_PERIODS_MORNING_END,
    SIX_PERIODS_MORNING_RUSH_START, SIX_PERIODS_MORNING_RUSH_END,
    SIX_PERIODS_NOON_START, SIX_PERIODS_NOON_END, SIX_PERIODS_EVENING_RUSH_START,
    SIX_PERIODS_EVENING_RUSH_END, SIX_PERIODS_EVENING_START, SIX_PERIODS_EVENING_END,
    SIX_PERIODS_NIGHT_START, SIX_PERIODS_NIGHT_END, SIX_PERIODS_EARLY_MORNING_END,
    SIX_PERIODS_MORNING_WEIGHT, SIX_PERIODS_MORNING_RUSH_WEIGHT, SIX_PERIODS_NOON_WEIGHT,
    SIX_PERIODS_EVENING_RUSH_WEIGHT, SIX_PERIODS_EVENING_WEIGHT, SIX_PERIODS_NIGHT_WEIGHT,
    NIGHT_LOW_START,
    SECONDS_TO_HOURS_DIVISOR, HOURS_IN_DAY, SECONDS_IN_24_HOURS,
    ROUTE_PATTERN_PAIRS_COUNT, ROUTE_PATTERN_EXPECTED_PAIRS,
    INITIAL_VEHICLE_ID_COUNTER, SINGLE_SAMPLE_COUNT, EDGE_SAMPLE_SLICE_LIMIT,
    MAX_PERCENTAGE, DEFAULT_DEPARTURE_TIME_FALLBACK,
    DEFAULT_ROUTE_WEIGHT, MINIMUM_ROUTE_COUNT,
    MINIMUM_VEHICLES_PER_ROUTE, TEMPORAL_BIAS_INVERSE_FACTOR,
    ARRAY_FIRST_ELEMENT_INDEX, RANGE_STEP_INCREMENT, SINGLE_INCREMENT,
    ROUTE_ID_INCREMENT, VEHICLE_INCREMENT, VEHICLES_FOR_ROUTE_INCREMENT,
    DEFAULT_EDGE_ATTRIBUTE_VALUE, FALLBACK_ATTRIBUTE_VALUE,
    TIME_RANGE_START, TIME_RANGE_END_24H, ATTRACTIVENESS_PHASES_COUNT,
    # String literals
    PHASE_MORNING_PEAK, PHASE_MIDDAY_OFFPEAK, PHASE_EVENING_PEAK, PHASE_NIGHT_LOW,
    PATTERN_IN, PATTERN_OUT, PATTERN_INNER, PATTERN_PASS,
    DIRECTION_DEPART, DIRECTION_ARRIVE, VEHICLE_ID_PREFIX,
    VEHICLE_TYPE_PASSENGER, VEHICLE_TYPE_PUBLIC, ROUTING_SHORTEST,
    DEPARTURE_PATTERN_UNIFORM, DEPARTURE_PATTERN_SIX_PERIODS,
    PERIOD_MORNING, PERIOD_MORNING_RUSH, PERIOD_NOON, PERIOD_EVENING_RUSH,
    PERIOD_EVENING, PERIOD_NIGHT, ATTR_NAME, ATTR_WEIGHT, ATTR_START, ATTR_END,
    DEFAULT_PASSENGER_ROUTE_PATTERN, DEFAULT_PUBLIC_ROUTE_PATTERN,
    ATTR_DEPART, ATTR_ATTRACTIVENESS,
    ATTR_CURRENT_PHASE, FIELD_PASSENGER_ROUTES, FIELD_PUBLIC_ROUTES, ATTR_TYPE,
    ATTR_FROM_EDGE, ATTR_TO_EDGE, ATTR_ROUTE_EDGES, ATTR_ROUTING_STRATEGY,
    FUNCTION_INTERNAL, ATTR_ID, SUFFIX_ATTRACTIVENESS,
    CUSTOM_PATTERN_PREFIX,
)
from .routing import RoutingMixStrategy, parse_routing_strategy
from .vehicle_types import parse_vehicle_types, get_vehicle_weights
from .xml_writer import write_routes
from ..network.generate_grid import classify_edges
from src.validate.validate_arguments import _parse_custom_pattern

# Constants for departure time generation - now imported from src.constants


def top(x: float) -> int:
    """
    'top' function as specified in ROUTES2.md - equivalent to ceiling function.

    Args:
        x: Number to round up

    Returns:
        Smallest integer greater than or equal to x
    """
    return math.ceil(x)


def determine_attractiveness_phase_from_departure_time(departure_time_seconds: int) -> str:
    """
    Determine attractiveness phase based on departure time for phase-specific edge sampling.

    As specified in ROUTES2.md: "attractiveness has {ATTRACTIVENESS_PHASES_COUNT} time phases and based on the 
    vehicle's departure time we'll know which one to use"

    Args:
        departure_time_seconds: Vehicle departure time in seconds since midnight

    Returns:
        Phase name for attractiveness sampling: "morning_peak", "midday_offpeak", "evening_peak", or "night_low"
    """
    # Convert seconds to hours (0-24 format)
    hour = (departure_time_seconds / SECONDS_TO_HOURS_DIVISOR) % HOURS_IN_DAY

    # Define phase boundaries based on realistic traffic patterns
    # Morning peak: 6am-10am (rush hour commuting)
    if RUSH_HOUR_MORNING_START <= hour < RUSH_HOUR_MORNING_END:
        return PHASE_MORNING_PEAK
    # Evening peak: 4pm-8pm (evening rush hour)
    elif RUSH_HOUR_EVENING_START <= hour < RUSH_HOUR_EVENING_END:
        return PHASE_EVENING_PEAK
    # Night low: 10pm-6am (overnight hours)
    elif hour >= NIGHT_LOW_START or hour < RUSH_HOUR_MORNING_START:
        return PHASE_NIGHT_LOW
    # Midday off-peak: 10am-4pm (daytime non-rush)
    else:
        return PHASE_MIDDAY_OFFPEAK


# Route Pattern Parsing Functions
def parse_route_patterns(route_pattern_str: str) -> dict:
    """
    Parse route pattern string into percentages dictionary.

    Args:
        route_pattern_str: String like "in 30 out 30 inner 25 pass 15"

    Returns:
        Dictionary mapping pattern names to percentages

    Example:
        "in 30 out 30 inner 25 pass 15" -> {"in": 30.0, "out": 30.0, "inner": 25.0, "pass": 15.0}
    """
    parts = route_pattern_str.strip().split()
    if len(parts) != ROUTE_PATTERN_PAIRS_COUNT:
        raise ValueError(
            f"Route pattern must have {ROUTE_PATTERN_EXPECTED_PAIRS} pairs, got: {route_pattern_str}")

    patterns = {}
    for i in range(ARRAY_FIRST_ELEMENT_INDEX, len(parts), RANGE_STEP_INCREMENT):
        pattern = parts[i]
        percentage = float(parts[i + SINGLE_INCREMENT])
        patterns[pattern] = percentage

    return patterns


def assign_route_pattern_to_vehicles(num_vehicles: int, route_patterns: dict, rng: random.Random) -> List[str]:
    """
    Assign route patterns to vehicles based on percentages.

    Args:
        num_vehicles: Total number of vehicles to assign patterns to
        route_patterns: Dictionary of pattern percentages (e.g., {"in": 30, "out": 30, "inner": 25, "pass": 15})
        rng: Random number generator for consistent assignment

    Returns:
        List of pattern names assigned to each vehicle
    """
    pattern_names = list(route_patterns.keys())
    pattern_weights = list(route_patterns.values())

    # Assign patterns to vehicles using weighted random selection
    assigned_patterns = []
    for _ in range(num_vehicles):
        pattern = rng.choices(pattern_names, weights=pattern_weights, k=SINGLE_SAMPLE_COUNT)[
            ARRAY_FIRST_ELEMENT_INDEX]
        assigned_patterns.append(pattern)

    return assigned_patterns


def assign_route_pattern_with_temporal_bias(departure_times: List[int], route_patterns: dict, rng: random.Random) -> List[str]:
    """
    Assign route patterns to vehicles with temporal bias for realistic traffic flows.

    As specified in ROUTES2.md: "Morning rush hours favor in-bound routes to business zones, 
    evening rush hours favor out-bound routes from residential areas"

    Args:
        departure_times: List of departure times in seconds for each vehicle
        route_patterns: Base route pattern percentages
        rng: Random number generator for consistent assignment

    Returns:
        List of pattern names assigned to each vehicle with temporal bias
    """
    assigned_patterns = []
    pattern_names = list(route_patterns.keys())

    for departure_time in departure_times:
        # Get temporal bias factors based on departure time
        biased_weights = _apply_temporal_bias_to_route_patterns(
            departure_time, route_patterns)

        # Assign pattern using temporally biased weights
        pattern = rng.choices(pattern_names, weights=biased_weights, k=SINGLE_SAMPLE_COUNT)[
            ARRAY_FIRST_ELEMENT_INDEX]
        assigned_patterns.append(pattern)

    return assigned_patterns


def _apply_temporal_bias_to_route_patterns(departure_time_seconds: int, base_patterns: dict) -> List[float]:
    """
    Apply temporal bias to route pattern weights based on departure time.

    Args:
        departure_time_seconds: Departure time in seconds since midnight
        base_patterns: Base route pattern percentages

    Returns:
        List of biased weights corresponding to pattern order
    """
    # Convert to hour (0-24)
    hour = (departure_time_seconds / SECONDS_TO_HOURS_DIVISOR) % HOURS_IN_DAY

    # Start with base weights
    weights = list(base_patterns.values())
    pattern_names = list(base_patterns.keys())

    # Apply temporal bias using constants
    if RUSH_HOUR_MORNING_START <= hour < RUSH_HOUR_MORNING_END:
        # Morning rush: favor in-bound routes (commuting to work/business zones)
        for i, pattern in enumerate(pattern_names):
            if pattern == PATTERN_IN:
                weights[i] *= TEMPORAL_BIAS_STRENGTH
            elif pattern == PATTERN_OUT:
                weights[i] *= (TEMPORAL_BIAS_INVERSE_FACTOR -
                               TEMPORAL_BIAS_STRENGTH)  # Reduce out-bound

    elif RUSH_HOUR_EVENING_START <= hour < RUSH_HOUR_EVENING_END:
        # Evening rush: favor out-bound routes (commuting from work back home)
        for i, pattern in enumerate(pattern_names):
            if pattern == PATTERN_OUT:
                weights[i] *= TEMPORAL_BIAS_STRENGTH
            elif pattern == PATTERN_IN:
                weights[i] *= (TEMPORAL_BIAS_INVERSE_FACTOR -
                               TEMPORAL_BIAS_STRENGTH)  # Reduce in-bound

    # No bias for off-peak hours - use base patterns as-is
    return weights


class PhaseSpecificEdgeSampler:
    """
    Phase-specific edge sampler that determines attractiveness phase from departure time.

    As specified in ROUTES2.md: "For each vehicle, based on the departure time we can use the 
    attractiveness values to decide of the start/end point"
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def sample_edges_with_phase(self, edges, direction: str, departure_time: int, n: int):
        """Sample edges using phase-specific attractiveness based on departure time."""
        # Determine phase from departure time
        phase = determine_attractiveness_phase_from_departure_time(
            departure_time)

        # Get phase-specific weights
        weights = self._get_phase_weights(edges, direction, phase)

        # Sample edges using phase-specific weights
        return [e.getID() for e in self.rng.choices(edges, weights=weights, k=n)]

    def sample_edges_with_route_pattern_targeting(self, edges, direction: str, departure_time: int, route_pattern: str, n: int):
        """
        Sample edges using phase-specific attractiveness with route pattern targeting.

        As specified in ROUTES2.md: "In-bound routes target high-arrival attractiveness inner edges, 
        out-bound routes originate from high-departure attractiveness inner edges"

        Args:
            edges: Edge pool to sample from
            direction: Base direction ("depart" or "arrive")
            departure_time: Departure time for phase determination
            route_pattern: Route pattern ("in", "out", "inner", "pass")
            n: Number of edges to sample

        Returns:
            List of edge IDs with route-pattern-aware attractiveness targeting
        """
        # Determine phase from departure time
        phase = determine_attractiveness_phase_from_departure_time(
            departure_time)

        # Apply route pattern targeting to determine the effective attractiveness direction
        effective_direction = self._get_effective_attractiveness_direction(
            direction, route_pattern)

        # Get phase-specific weights with route pattern targeting
        weights = self._get_phase_weights_with_targeting(
            edges, effective_direction, phase, route_pattern)

        # Sample edges using targeted weights
        return [e.getID() for e in self.rng.choices(edges, weights=weights, k=n)]

    def _get_effective_attractiveness_direction(self, base_direction: str, route_pattern: str) -> str:
        """
        Determine the effective attractiveness direction based on route pattern targeting.

        For in-bound and out-bound routes, we want to target business zones (high attractiveness),
        so we need to choose the direction that represents business zone activity.
        """
        if route_pattern == PATTERN_IN:
            # In-bound routes: We want to target high-arrival zones (business districts) for END edges
            # Start edges use normal depart logic, end edges target high arrival attractiveness
            return base_direction  # Use the provided direction

        elif route_pattern == PATTERN_OUT:
            # Out-bound routes: We want to originate from high-departure zones (business districts) for START edges
            # Start edges target high departure attractiveness, end edges use normal arrive logic
            return base_direction  # Use the provided direction

        else:
            # Inner and pass routes: Use standard direction logic
            return base_direction

    def _get_phase_weights_with_targeting(self, edges, direction: str, phase: str, route_pattern: str):
        """
        Get phase-specific weights with route pattern targeting following ROUTES2.md specification.

        Attractiveness rules:
        - In-bound routes: start boundary (irrelevant), end inner (relevant)
        - Out-bound routes: start inner (relevant), end boundary (irrelevant)  
        - Inner routes: start inner (relevant), end inner (relevant)
        - Pass-through routes: start boundary (irrelevant), end boundary (irrelevant)
        """
        # Determine if attractiveness should be applied based on route pattern and direction
        use_attractiveness = self._should_use_attractiveness(
            route_pattern, direction)

        if not use_attractiveness:
            # Return uniform weights when attractiveness is irrelevant (boundary edges)
            return [DEFAULT_ROUTE_WEIGHT] * len(edges)

        # Apply phase-specific attractiveness when relevant (inner edges)
        weights = []
        for edge in edges:
            # Get phase-specific attractiveness attribute
            attr_name = f"{phase}_{direction}{SUFFIX_ATTRACTIVENESS}"
            weight = float(getattr(
                edge, attr_name, DEFAULT_EDGE_ATTRIBUTE_VALUE) or FALLBACK_ATTRIBUTE_VALUE)
            weights.append(weight)

        # Fallback to uniform if all weights are zero
        return weights if any(weights) else [DEFAULT_ROUTE_WEIGHT] * len(edges)

    def _should_use_attractiveness(self, route_pattern: str, direction: str) -> bool:
        """
        Determine if attractiveness should be used based on route pattern and direction.

        Returns True when sampling inner edges (attractiveness relevant),
        False when sampling boundary edges (attractiveness irrelevant).
        """
        if route_pattern == PATTERN_IN:
            # In-bound: start boundary (irrelevant), end inner (relevant)
            return direction == DIRECTION_ARRIVE
        elif route_pattern == PATTERN_OUT:
            # Out-bound: start inner (relevant), end boundary (irrelevant)
            return direction == DIRECTION_DEPART
        elif route_pattern == PATTERN_INNER:
            # Inner: both start and end inner (relevant for both)
            return True
        elif route_pattern == PATTERN_PASS:
            # Pass-through: both start and end boundary (irrelevant for both)
            return False
        else:
            # Default to using attractiveness
            return True

    def _get_phase_weights(self, edges, direction: str, phase: str):
        """Get phase-specific weights for edges."""
        weights = []
        for edge in edges:
            # Get phase-specific attractiveness attribute
            attr_name = f"{phase}_{direction}{SUFFIX_ATTRACTIVENESS}"
            weight = float(getattr(
                edge, attr_name, DEFAULT_EDGE_ATTRIBUTE_VALUE) or FALLBACK_ATTRIBUTE_VALUE)
            weights.append(weight)

        # Fallback to uniform if all weights are zero
        return weights if any(weights) else [DEFAULT_ROUTE_WEIGHT] * len(edges)


def load_attractiveness_attributes_from_xml(edges, net_file_path):
    """
    Load attractiveness attributes from XML network file and attach them to edge objects.

    sumolib.net.readNet() doesn't automatically parse custom attributes like attractiveness,
    so we need to manually load them from the XML and attach to the edge objects.
    """
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()

        # Create a mapping of edge ID to XML attributes
        edge_attrs = {}
        for edge_elem in root.findall('edge'):
            edge_id = edge_elem.get('id')
            if edge_id:
                # Extract all attractiveness attributes
                attrs = {}
                for attr_name in edge_elem.attrib:
                    if 'attractiveness' in attr_name or 'current_phase' in attr_name:
                        attrs[attr_name] = edge_elem.get(attr_name)
                edge_attrs[edge_id] = attrs

        # Attach attributes to edge objects
        for edge in edges:
            edge_id = edge.getID()
            if edge_id in edge_attrs:
                for attr_name, attr_value in edge_attrs[edge_id].items():
                    setattr(edge, attr_name, attr_value)

        # Optionally log the number of edges loaded
        # print(f"DEBUG: Loaded attractiveness attributes for {len(edge_attrs)} edges")

    except Exception as e:
        print(
            f"WARNING: Failed to load attractiveness attributes from XML: {e}")


def map_simulation_time_to_real_time(simulation_time_seconds: int, start_time_hour: float) -> int:
    """
    Map simulation time to real time for attractiveness phase determination.

    Args:
        simulation_time_seconds: Time within simulation ({TIME_RANGE_START} to end_time)
        start_time_hour: Real time hour when simulation starts (e.g., {EXAMPLE_START_TIME_8AM} for 8 AM)

    Returns:
        Real time in seconds since midnight for attractiveness phase calculations

    Example:
        map_simulation_time_to_real_time({TIME_RANGE_START}, {EXAMPLE_START_TIME_8AM}) -> {EXAMPLE_RESULT_28800} (8:00 AM)
        map_simulation_time_to_real_time({SECONDS_TO_HOURS_DIVISOR}, {EXAMPLE_START_TIME_8AM}) -> {EXAMPLE_RESULT_32400} (9:00 AM)
    """
    return int(start_time_hour * SECONDS_TO_HOURS_DIVISOR + simulation_time_seconds)


def generate_vehicle_routes(net_file: str | Path,
                            output_file: str | Path,
                            num_vehicles: int,
                            private_traffic_seed: int,
                            public_traffic_seed: int,
                            routing_strategy: str,
                            vehicle_types: str,
                            passenger_routes: str,
                            public_routes: str,
                            end_time: int,
                            departure_pattern: str,
                            start_time_hour: float,
                            grid_dimension: int) -> None:
    """
    Orchestrates vehicle creation with route patterns and writes a .rou.xml.

    This is the new route generation system that implements the {ROUTE_PATTERN_EXPECTED_PAIRS}-pattern route system
    (in-bound, out-bound, inner, pass-through) with deterministic departure timing and
    separate handling for passenger and public vehicles.

    Args:
        net_file: Path to SUMO network file
        output_file: Output route file path
        num_vehicles: Total number of vehicles to generate
        private_traffic_seed: Random seed for private traffic (passenger vehicles)
        public_traffic_seed: Random seed for public traffic
        routing_strategy: Routing strategy specification (e.g., "shortest 70 realtime 30")
        vehicle_types: Vehicle types specification (e.g., "passenger 90 public 10")
        passenger_routes: Passenger route patterns (e.g., "in 30 out 30 inner 25 pass 15")
        public_routes: Public route patterns (e.g., "in 25 out 25 inner 35 pass 15")
        end_time: Total simulation duration in seconds for temporal distribution
        departure_pattern: Departure pattern ("six_periods", "uniform", "custom:...")
        start_time_hour: Start time in hours ({TIME_RANGE_START}-{TIME_RANGE_END_24H})
        grid_dimension: Grid dimension for edge classification
    """
    # Load network and filter edges
    net = readNet(str(net_file))
    all_edges = [e for e in net.getEdges() if e.getFunction()
                 != FUNCTION_INTERNAL]

    # Load attractiveness attributes from XML and attach to edge objects
    # (sumolib.net.readNet doesn't automatically parse custom attributes)
    load_attractiveness_attributes_from_xml(all_edges, net_file)

    # Filter to get only tail edges (exclude head edges with _H_ suffix)
    # In multi-head edge architecture: tail edges like "A1B1" connect to intermediate nodes,
    # head edges like "A1B1_H_left" connect from intermediate nodes to junctions
    # Routing should use tail edges as origin/destination - SUMO automatically includes head edges in paths
    tail_edges_only = [e for e in all_edges if '_H_' not in e.getID()]

    # Classify edges into boundary and inner for route pattern restrictions
    boundary_edges, inner_edges = classify_edges(grid_dimension)

    # Convert edge ID lists to edge objects - only use tail edges (without _H_s or _H_node suffixes)
    # as specified in ROUTES2.md: "we only examine the tail part of edges, namely edges without suffix _H_s or _H_node"
    boundary_edge_objs = [e for e in tail_edges_only if e.getID() in boundary_edges]
    inner_edge_objs = [e for e in tail_edges_only if e.getID() in inner_edges]
    # For inner routes: combine boundary and inner tail edges (all possible tail edges)
    all_tail_edge_objs = boundary_edge_objs + inner_edge_objs
    all_edge_objs = tail_edges_only  # Use only tail edges for routing

    # print(
    #     f"Classified {len(boundary_edge_objs)} boundary edges and {len(inner_edge_objs)} inner edges")

    # Debug: Print a few examples of boundary vs inner edges
    # print(
    #     f"Sample boundary edges: {[e.getID() for e in boundary_edge_objs[:EDGE_SAMPLE_SLICE_LIMIT]]}")
    # print(
    #     f"Sample inner edges: {[e.getID() for e in inner_edge_objs[:EDGE_SAMPLE_SLICE_LIMIT]]}")

    # Parse configurations
    vehicle_distribution = parse_vehicle_types(vehicle_types)
    passenger_route_patterns = parse_route_patterns(passenger_routes)
    public_route_patterns = parse_route_patterns(public_routes)
    strategy_percentages = parse_routing_strategy(routing_strategy)

    # print(f"Using routing strategies: {strategy_percentages}")
    # print(f"Using vehicle types: {vehicle_distribution}")
    # print(f"Using passenger route patterns: {passenger_route_patterns}")
    # print(f"Using public route patterns: {public_route_patterns}")
    # print(f"Using private traffic seed: {private_traffic_seed}")
    # print(f"Using public traffic seed: {public_traffic_seed}")

    # Calculate vehicle counts by type
    vehicle_names, vehicle_weights = get_vehicle_weights(vehicle_distribution)
    passenger_count = int((vehicle_distribution.get(
        VEHICLE_TYPE_PASSENGER, 0) / MAX_PERCENTAGE) * num_vehicles)
    public_count = num_vehicles - passenger_count

    # print(
    #     f"Generating {passenger_count} passenger vehicles and {public_count} public vehicles")

    # Create RNGs
    private_rng = random.Random(private_traffic_seed)
    public_rng = random.Random(public_traffic_seed)

    # Generate vehicles
    vehicles = []
    # Unified counter for all vehicles to ensure 'veh' prefix compatibility
    vehicle_id_counter = INITIAL_VEHICLE_ID_COUNTER

    # ========== PASSENGER VEHICLE GENERATION ==========
    if passenger_count > 0:
        passenger_vehicles, vehicle_id_counter = _generate_passenger_vehicles(
            passenger_count, private_rng, net, all_edge_objs, boundary_edge_objs, inner_edge_objs, all_tail_edge_objs,
            passenger_route_patterns, strategy_percentages, departure_pattern, start_time_hour, end_time, vehicle_id_counter
        )
        vehicles.extend(passenger_vehicles)

    # ========== PUBLIC VEHICLE GENERATION ==========
    if public_count > 0:
        public_vehicles, vehicle_id_counter = _generate_public_vehicles(
            public_count, public_rng, net, all_edge_objs, boundary_edge_objs, inner_edge_objs, all_tail_edge_objs,
            public_route_patterns, departure_pattern, start_time_hour, end_time, vehicle_id_counter
        )
        vehicles.extend(public_vehicles)

    # Sort vehicles by departure time (SUMO requirement)
    vehicles.sort(key=lambda v: v[ATTR_DEPART])

    # print(f"Generated {len(vehicles)} total vehicles")

    # Write routes to file
    write_routes(output_file, vehicles, CONFIG.vehicle_types)


def _generate_passenger_vehicles(passenger_count: int, private_rng: random.Random, net,
                                 all_edge_objs, boundary_edge_objs, inner_edge_objs, all_tail_edge_objs,
                                 route_patterns: dict, strategy_percentages: dict,
                                 departure_pattern: str, start_time_hour: float, end_time: int,
                                 vehicle_id_counter: int) -> tuple[List[dict], int]:
    """Generate passenger vehicles with route pattern support."""
    vehicles = []

    # Create routing mix and phase-specific sampler for passenger vehicles
    routing_mix = RoutingMixStrategy(net, strategy_percentages)
    phase_sampler = PhaseSpecificEdgeSampler(private_rng)

    # Calculate deterministic departure times for all passenger vehicles
    departure_times = calculate_temporal_departure_times(
        passenger_count, departure_pattern, start_time_hour, end_time)

    # Assign route patterns to vehicles with temporal bias for realistic traffic flows
    # Convert simulation departure times to real times for temporal bias calculations
    real_departure_times = [map_simulation_time_to_real_time(
        dt, start_time_hour) for dt in departure_times]
    assigned_patterns = assign_route_pattern_with_temporal_bias(
        real_departure_times, route_patterns, private_rng)

    # Generate individual passenger vehicles
    for i in range(passenger_count):
        vid = f"{VEHICLE_ID_PREFIX}{vehicle_id_counter}"
        # Reserve ID immediately to prevent duplicates
        vehicle_id_counter += VEHICLE_INCREMENT
        route_pattern = assigned_patterns[i]
        departure_time = departure_times[i] if i < len(departure_times) else (
            departure_times[-1] if departure_times else DEFAULT_DEPARTURE_TIME_FALLBACK)

        # Assign routing strategy (passenger vehicles use configured strategies)
        assigned_strategy = routing_mix.assign_strategy_to_vehicle(
            vid, private_rng)

        # Select edge pools based on route pattern
        start_edge_pool, end_edge_pool = _get_edge_pools_for_pattern(
            route_pattern, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs)

        # Generate route with pattern restrictions
        route_edges = []
        start_edge = None
        end_edge = None

        for _ in range(MAX_ROUTE_RETRIES):
            # Use route-pattern-aware attractiveness targeting (ROUTES2.md line {ROUTES_MD_LINE_18})
            # "In-bound routes target high-arrival attractiveness inner edges,
            #  out-bound routes originate from high-departure attractiveness inner edges"
            # Map simulation time to real time for attractiveness phase determination
            real_time = map_simulation_time_to_real_time(
                departure_time, start_time_hour)
            start_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                start_edge_pool, DIRECTION_DEPART, real_time, route_pattern, SINGLE_SAMPLE_COUNT)
            end_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                end_edge_pool, DIRECTION_ARRIVE, real_time, route_pattern, SINGLE_SAMPLE_COUNT)

            start_edge = start_edge_ids[ARRAY_FIRST_ELEMENT_INDEX]
            end_edge = end_edge_ids[ARRAY_FIRST_ELEMENT_INDEX]

            # Skip if same edge ID
            if end_edge == start_edge:
                continue

            # Skip if start and end junctions are the same (prevents B1→B1 routes)
            # start_edge is a tail edge (e.g., "B1B2" from B1)
            # end_edge after head conversion will end at a junction
            try:
                start_junction = net.getEdge(start_edge).getFromNode().getID()
                # Convert end_edge to head edge to get the actual destination junction
                end_edge_head = convert_tail_to_head_edge(end_edge, net)
                end_junction = net.getEdge(end_edge_head).getToNode().getID()

                if start_junction == end_junction:
                    continue
            except (KeyError, RuntimeError, AttributeError):
                # Edge lookup failed, skip this combination
                continue

            route_edges = routing_mix.compute_route(
                assigned_strategy, start_edge, end_edge)
            if route_edges:
                break
        else:
            print(
                f"⚠️  Could not find a path for passenger vehicle {vid} using {assigned_strategy} strategy; skipping.")
            continue

        if not route_edges:
            print(f"⚠️  Empty route for passenger vehicle {vid}; skipping.")
            continue

        vehicles.append({
            ATTR_ID: vid,
            ATTR_TYPE: VEHICLE_TYPE_PASSENGER,
            ATTR_DEPART: int(departure_time),
            ATTR_FROM_EDGE: start_edge,
            ATTR_TO_EDGE: end_edge,
            ATTR_ROUTE_EDGES: route_edges,
            ATTR_ROUTING_STRATEGY: assigned_strategy,
        })

    # print(f"Generated {len(vehicles)} passenger vehicles")
    return vehicles, vehicle_id_counter


def _generate_public_vehicles(public_count: int, public_rng: random.Random, net,
                              all_edge_objs, boundary_edge_objs, inner_edge_objs, all_tail_edge_objs,
                              route_patterns: dict, departure_pattern: str,
                              start_time_hour: float, end_time: int,
                              vehicle_id_counter: int) -> tuple[List[dict], int]:
    """Generate public vehicles with fixed routes using shortest path only."""
    vehicles = []

    if public_count == 0:
        return vehicles, vehicle_id_counter

    # Calculate route structure for public vehicles using constants from ROUTES2.md specification
    ideal_num_vehicles_per_route = (
        end_time / SECONDS_IN_DAY) * DEFAULT_VEHICLES_DAILY_PER_ROUTE
    if ideal_num_vehicles_per_route <= 0:
        ideal_num_vehicles_per_route = MINIMUM_VEHICLES_PER_ROUTE

    num_public_routes = top(public_count / ideal_num_vehicles_per_route)
    base_vehicles_per_route = public_count // num_public_routes
    extra_vehicles = public_count % num_public_routes  # Distribute remaining vehicles

    # print(f"Creating {num_public_routes} public routes with ~{base_vehicles_per_route}-{base_vehicles_per_route + 1} vehicles each")

    # Calculate route distribution based on percentages
    route_counts = {}
    for pattern, percentage in route_patterns.items():
        route_counts[pattern] = max(MINIMUM_ROUTE_COUNT, int(
            (percentage / MAX_PERCENTAGE) * num_public_routes))

    # Handle case where there are not enough vehicles for all route types
    total_planned_routes = sum(route_counts.values())
    if total_planned_routes > num_public_routes:
        # Use only the highest percentage pattern
        max_pattern = max(route_patterns.keys(),
                          key=lambda k: route_patterns[k])
        route_counts = {max_pattern: num_public_routes}
        # print(
        #     f"⚠️  Not enough public vehicles for all route types. Using only '{max_pattern}' routes.")

    # Create routing strategy for public vehicles (always shortest path)
    public_routing_strategy = {ROUTING_SHORTEST: MAX_PERCENTAGE}
    routing_mix = RoutingMixStrategy(net, public_routing_strategy)
    phase_sampler = PhaseSpecificEdgeSampler(public_rng)

    # Calculate departure times for ALL public vehicles globally, then distribute to routes
    # This ensures proper temporal spread across the entire simulation time
    all_departure_times = calculate_temporal_departure_times(
        public_count, departure_pattern, start_time_hour, end_time)

    route_id = INITIAL_VEHICLE_ID_COUNTER
    # Track how many vehicles we've created locally
    vehicles_created = INITIAL_VEHICLE_ID_COUNTER

    # Generate routes for each pattern type
    for pattern, route_count in route_counts.items():
        for route_idx in range(route_count):
            # Select edge pools for this route pattern
            start_edge_pool, end_edge_pool = _get_edge_pools_for_pattern(
                pattern, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs)

            # Generate route definition
            route_edges = []
            start_edge = None
            end_edge = None

            for _ in range(MAX_ROUTE_RETRIES):
                # Use phase-specific attractiveness for public routes based on first departure time
                # As specified in ROUTES2.md line {ROUTES_MD_LINE_99}: "For each vehicle, based on the departure time we can use the attractiveness values"
                first_departure_time = all_departure_times[vehicles_created] if vehicles_created < len(
                    all_departure_times) else DEFAULT_DEPARTURE_TIME_FALLBACK
                # Map simulation time to real time for attractiveness phase determination
                real_first_departure_time = map_simulation_time_to_real_time(
                    first_departure_time, start_time_hour)

                start_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                    start_edge_pool, DIRECTION_DEPART, real_first_departure_time, pattern, SINGLE_SAMPLE_COUNT)
                end_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                    end_edge_pool, DIRECTION_ARRIVE, real_first_departure_time, pattern, SINGLE_SAMPLE_COUNT)

                start_edge = start_edge_ids[ARRAY_FIRST_ELEMENT_INDEX]
                end_edge = end_edge_ids[ARRAY_FIRST_ELEMENT_INDEX]

                # Skip if same edge ID
                if end_edge == start_edge:
                    continue

                # Skip if start and end junctions are the same (prevents B1→B1 routes)
                # start_edge is a tail edge (e.g., "B1B2" from B1)
                # end_edge after head conversion will end at a junction
                try:
                    start_junction = net.getEdge(start_edge).getFromNode().getID()
                    # Convert end_edge to head edge to get the actual destination junction
                    end_edge_head = convert_tail_to_head_edge(end_edge, net)
                    end_junction = net.getEdge(end_edge_head).getToNode().getID()

                    if start_junction == end_junction:
                        continue
                except (KeyError, RuntimeError, AttributeError):
                    # Edge lookup failed, skip this combination
                    continue

                route_edges = routing_mix.compute_route(
                    ROUTING_SHORTEST, start_edge, end_edge)
                if route_edges:
                    break
            else:
                print(
                    f"⚠️  Could not find a path for public route {route_id} with pattern {pattern}; skipping.")
                continue

            if not route_edges:
                print(
                    f"⚠️  Empty route for public route {route_id}; skipping.")
                continue

            # Generate vehicles for this route - distribute extra vehicles among first routes
            vehicles_for_this_route = base_vehicles_per_route
            if route_id < extra_vehicles:  # First 'extra_vehicles' routes get one extra vehicle
                vehicles_for_this_route += VEHICLES_FOR_ROUTE_INCREMENT

            actual_vehicles = min(vehicles_for_this_route, len(
                all_departure_times) - vehicles_created)
            # Safety check to not exceed total vehicle count
            if vehicles_created + actual_vehicles > public_count:
                actual_vehicles = public_count - vehicles_created

            for vehicle_idx in range(actual_vehicles):
                if vehicles_created >= public_count:
                    break

                departure_time = all_departure_times[vehicles_created]
                vid = f"{VEHICLE_ID_PREFIX}{vehicle_id_counter}"

                vehicles.append({
                    ATTR_ID: vid,
                    ATTR_TYPE: VEHICLE_TYPE_PUBLIC,
                    ATTR_DEPART: int(departure_time),
                    ATTR_FROM_EDGE: start_edge,
                    ATTR_TO_EDGE: end_edge,
                    ATTR_ROUTE_EDGES: route_edges,
                    ATTR_ROUTING_STRATEGY: ROUTING_SHORTEST,  # Public always uses shortest
                })

                vehicle_id_counter += VEHICLE_INCREMENT
                vehicles_created += VEHICLE_INCREMENT

            route_id += ROUTE_ID_INCREMENT

    # print(
    #     f"Generated {len(vehicles)} public vehicles across {route_id} routes")
    return vehicles, vehicle_id_counter


def _get_edge_pools_for_pattern(pattern: str, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs) -> tuple:
    """Get start and end edge pools based on route pattern."""
    if pattern == PATTERN_IN:
        # In-bound: start from boundary, end at inner edges
        return boundary_edge_objs, inner_edge_objs
    elif pattern == PATTERN_OUT:
        # Out-bound: start from inner edges, end at boundary
        return inner_edge_objs, boundary_edge_objs
    elif pattern == PATTERN_INNER:
        # Inner: start and end from inner edges only
        return inner_edge_objs, inner_edge_objs
    elif pattern == PATTERN_PASS:
        # Pass-through: start from boundary, end at boundary
        return boundary_edge_objs, boundary_edge_objs
    else:
        # Default to inner pattern
        return inner_edge_objs, inner_edge_objs


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
    if departure_pattern == DEPARTURE_PATTERN_UNIFORM:
        return int(rng.uniform(TIME_RANGE_START, end_time * SIMULATION_END_FACTOR))

    elif departure_pattern == DEPARTURE_PATTERN_SIX_PERIODS:
        return _generate_six_periods_departure(rng, end_time)

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
    # Scale to simulation time
    scale_factor = end_time / SECONDS_IN_24_HOURS

    # Define periods in seconds (24-hour format)
    periods = [
        {ATTR_NAME: PERIOD_MORNING, ATTR_START: SIX_PERIODS_MORNING_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_MORNING_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_MORNING_WEIGHT},
        {ATTR_NAME: PERIOD_MORNING_RUSH, ATTR_START: SIX_PERIODS_MORNING_RUSH_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_MORNING_RUSH_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_MORNING_RUSH_WEIGHT},
        {ATTR_NAME: PERIOD_NOON, ATTR_START: SIX_PERIODS_NOON_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_NOON_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_NOON_WEIGHT},
        {ATTR_NAME: PERIOD_EVENING_RUSH, ATTR_START: SIX_PERIODS_EVENING_RUSH_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_EVENING_RUSH_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_EVENING_RUSH_WEIGHT},
        {ATTR_NAME: PERIOD_EVENING, ATTR_START: SIX_PERIODS_EVENING_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_EVENING_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_EVENING_WEIGHT},
        {ATTR_NAME: PERIOD_NIGHT, ATTR_START: SIX_PERIODS_NIGHT_START*SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_NIGHT_END*SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_NIGHT_WEIGHT},
    ]

    # Choose period based on weights
    weights = [p[ATTR_WEIGHT] for p in periods]
    chosen_period = rng.choices(periods, weights=weights)[
        ARRAY_FIRST_ELEMENT_INDEX]

    # Generate time within chosen period
    start_time = chosen_period[ATTR_START] * scale_factor
    end_time_period = chosen_period[ATTR_END] * scale_factor

    # Handle night period wrapping to next day
    if chosen_period[ATTR_NAME] == PERIOD_NIGHT:
        if rng.random() < NIGHT_EVENING_RATIO:  # 25% in evening part (10pm-12am)
            departure_time = rng.uniform(
                SIX_PERIODS_NIGHT_START*SECONDS_TO_HOURS_DIVISOR*scale_factor, HOURS_IN_DAY*SECONDS_TO_HOURS_DIVISOR*scale_factor)
        else:  # 75% in early morning part (12am-6am)
            departure_time = rng.uniform(
                TIME_RANGE_START, SIX_PERIODS_EARLY_MORNING_END*SECONDS_TO_HOURS_DIVISOR*scale_factor)
    else:
        departure_time = rng.uniform(start_time, end_time_period)

    return int(min(departure_time, end_time * SIMULATION_END_FACTOR_SIX_PERIODS))


def calculate_temporal_departure_times(num_vehicles: int, departure_pattern: str, start_time: float, end_time: int) -> List[int]:
    """
    Calculate deterministic departure times for all vehicles based on departure pattern.

    This function pre-calculates all departure times to ensure exact percentage distribution
    and even temporal spacing within each period, replacing the current random individual
    selection approach.

    Args:
        num_vehicles: Total number of vehicles to generate departure times for
        departure_pattern: Pattern specification ("uniform", "six_periods", "custom:...")
        start_time: Start time in hours (0-24)
        end_time: Total simulation duration in seconds

    Returns:
        List of departure times in seconds, sorted chronologically

    Note:
        This function ensures deterministic distribution with exact percentages
        and even spacing within time periods, ideal for coordinated systems
        like public transit and reproducible research.
    """
    departure_times = []

    if departure_pattern == DEPARTURE_PATTERN_UNIFORM:
        # Even distribution across simulation duration
        # start_time_hour is used for time window mapping to real time for attractiveness/bias calculations
        if num_vehicles <= 0:
            return departure_times

        interval = end_time / num_vehicles
        for i in range(num_vehicles):
            departure_time = i * interval  # Simulation time (0 to end_time)
            departure_times.append(int(departure_time))

    elif departure_pattern == DEPARTURE_PATTERN_SIX_PERIODS:
        # Research-based 6-period distribution with exact percentages
        departure_times = _calculate_six_periods_deterministic(
            num_vehicles, end_time)

    elif departure_pattern.startswith(CUSTOM_PATTERN_PREFIX):
        # Custom pattern handling
        departure_times = _calculate_custom_deterministic(
            num_vehicles, departure_pattern, start_time, end_time)

    else:
        # Default to six_periods for unknown patterns
        departure_times = _calculate_six_periods_deterministic(
            num_vehicles, end_time)

    return sorted(departure_times)


def _calculate_six_periods_deterministic(num_vehicles: int, end_time: int) -> List[int]:
    """Calculate deterministic six periods departure times with exact percentages."""
    departure_times = []

    # Scale to simulation time
    scale_factor = end_time / SECONDS_IN_24_HOURS

    # Six periods with exact percentages
    periods = [
        {ATTR_NAME: PERIOD_MORNING, ATTR_START: SIX_PERIODS_MORNING_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_MORNING_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_MORNING_WEIGHT},
        {ATTR_NAME: PERIOD_MORNING_RUSH, ATTR_START: SIX_PERIODS_MORNING_RUSH_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_MORNING_RUSH_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_MORNING_RUSH_WEIGHT},
        {ATTR_NAME: PERIOD_NOON, ATTR_START: SIX_PERIODS_NOON_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_NOON_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_NOON_WEIGHT},
        {ATTR_NAME: PERIOD_EVENING_RUSH, ATTR_START: SIX_PERIODS_EVENING_RUSH_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_EVENING_RUSH_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_EVENING_RUSH_WEIGHT},
        {ATTR_NAME: PERIOD_EVENING, ATTR_START: SIX_PERIODS_EVENING_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_EVENING_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_EVENING_WEIGHT},
        {ATTR_NAME: PERIOD_NIGHT, ATTR_START: SIX_PERIODS_NIGHT_START * SECONDS_TO_HOURS_DIVISOR,
            ATTR_END: SIX_PERIODS_NIGHT_END * SECONDS_TO_HOURS_DIVISOR, ATTR_WEIGHT: SIX_PERIODS_NIGHT_WEIGHT},
    ]

    total_weight = sum(p[ATTR_WEIGHT] for p in periods)

    # Calculate vehicles per period with rounding correction
    period_vehicle_counts = []
    for period in periods:
        period_vehicles = int(
            (period[ATTR_WEIGHT] / total_weight) * num_vehicles)
        period_vehicle_counts.append(period_vehicles)

    # Add missing vehicles due to rounding (distribute to periods with highest weights)
    assigned_total = sum(period_vehicle_counts)
    missing_vehicles = num_vehicles - assigned_total
    if missing_vehicles > 0:
        # Sort periods by weight descending to add missing vehicles to largest periods
        period_indices_by_weight = sorted(
            range(len(periods)), key=lambda i: periods[i][ATTR_WEIGHT], reverse=True)
        for i in range(missing_vehicles):
            period_vehicle_counts[period_indices_by_weight[i %
                                                           len(period_indices_by_weight)]] += 1

    for period_idx, period in enumerate(periods):
        period_vehicles = period_vehicle_counts[period_idx]

        if period_vehicles > 0:
            if period[ATTR_NAME] == PERIOD_NIGHT:
                # Handle night period wrapping (10pm-6am next day)
                # 25% in evening part (10pm-12am), 75% in early morning (12am-6am)
                evening_vehicles = int(period_vehicles * NIGHT_EVENING_RATIO)
                morning_vehicles = period_vehicles - evening_vehicles

                # Evening part (10pm-12am)
                if evening_vehicles > 0:
                    evening_start = SIX_PERIODS_NIGHT_START * \
                        SECONDS_TO_HOURS_DIVISOR * scale_factor
                    evening_end = HOURS_IN_DAY * SECONDS_TO_HOURS_DIVISOR * scale_factor
                    evening_interval = (
                        evening_end - evening_start) / evening_vehicles
                    for i in range(evening_vehicles):
                        departure_time = evening_start + i * evening_interval
                        departure_times.append(int(departure_time))

                # Early morning part (12am-6am)
                if morning_vehicles > 0:
                    morning_start = TIME_RANGE_START
                    morning_end = SIX_PERIODS_EARLY_MORNING_END * \
                        SECONDS_TO_HOURS_DIVISOR * scale_factor
                    morning_interval = (
                        morning_end - morning_start) / morning_vehicles
                    for i in range(morning_vehicles):
                        departure_time = morning_start + i * morning_interval
                        departure_times.append(int(departure_time))
            else:
                # Regular period with even distribution
                period_start = period[ATTR_START] * scale_factor
                period_end = period[ATTR_END] * scale_factor
                period_interval = (period_end - period_start) / period_vehicles

                for i in range(period_vehicles):
                    departure_time = period_start + i * period_interval
                    departure_times.append(int(departure_time))

    return departure_times


def _calculate_custom_deterministic(
    num_vehicles: int,
    pattern: str,
    start_hour: float,
    end_time: int
) -> List[int]:
    """
    Calculate deterministic departure times for custom pattern.

    Args:
        num_vehicles: Total number of vehicles
        pattern: Custom pattern string like "custom:9:00-9:30,40;10:00-10:45,30"
        start_hour: Simulation start time in hours (e.g., 8.0)
        end_time: Simulation duration in seconds

    Returns:
        List of departure times in seconds from simulation start, sorted
    """
    departure_times = []

    # Parse pattern
    windows = _parse_custom_pattern(pattern)

    # Calculate simulation bounds in seconds from midnight
    sim_start_seconds = int(start_hour * 3600)
    sim_end_seconds = sim_start_seconds + end_time

    # Convert windows to seconds and calculate totals
    window_specs = []
    total_specified_percent = 0

    for w in windows:
        start_h, start_m = w["start"]
        end_h, end_m = w["end"]

        window_start = start_h * 3600 + start_m * 60
        window_end = end_h * 3600 + end_m * 60

        window_specs.append({
            "start": window_start,
            "end": window_end,
            "percent": w["percent"],
            "duration": window_end - window_start
        })
        total_specified_percent += w["percent"]

    rest_percent = 100 - total_specified_percent

    # Calculate rest windows (gaps between specified windows and simulation bounds)
    rest_windows = _compute_rest_windows_custom(window_specs, sim_start_seconds, sim_end_seconds)
    rest_total_duration = sum(w["duration"] for w in rest_windows)

    # Allocate vehicles to specified windows
    vehicles_assigned = 0
    for spec in window_specs:
        window_vehicles = int((spec["percent"] / 100) * num_vehicles)
        vehicles_assigned += window_vehicles

        if window_vehicles > 0:
            # Generate uniformly spaced departure times
            interval = spec["duration"] / window_vehicles
            for i in range(window_vehicles):
                # Convert from absolute time to simulation time (offset from start)
                abs_time = spec["start"] + i * interval
                sim_time = abs_time - sim_start_seconds
                departure_times.append(int(sim_time))

    # Allocate vehicles to rest windows proportionally
    rest_vehicles = num_vehicles - vehicles_assigned

    if rest_vehicles > 0 and rest_total_duration > 0:
        for rest_window in rest_windows:
            # Proportional allocation by duration
            window_vehicles = int((rest_window["duration"] / rest_total_duration) * rest_vehicles)

            if window_vehicles > 0:
                interval = rest_window["duration"] / window_vehicles
                for i in range(window_vehicles):
                    abs_time = rest_window["start"] + i * interval
                    sim_time = abs_time - sim_start_seconds
                    departure_times.append(int(sim_time))

    # Handle rounding: add any missing vehicles to largest window
    while len(departure_times) < num_vehicles:
        # Add to middle of simulation
        departure_times.append(end_time // 2)

    return sorted(departure_times)


def _compute_rest_windows_custom(
    specified_windows: List[dict],
    sim_start: int,
    sim_end: int
) -> List[dict]:
    """
    Compute rest windows (gaps) between specified windows.

    Args:
        specified_windows: List of window specs with start/end in seconds
        sim_start: Simulation start in seconds from midnight
        sim_end: Simulation end in seconds from midnight

    Returns:
        List of rest window specs
    """
    if not specified_windows:
        return [{"start": sim_start, "end": sim_end, "duration": sim_end - sim_start}]

    # Sort windows by start time
    sorted_windows = sorted(specified_windows, key=lambda w: w["start"])

    rest_windows = []

    # Gap before first window
    if sorted_windows[0]["start"] > sim_start:
        rest_windows.append({
            "start": sim_start,
            "end": sorted_windows[0]["start"],
            "duration": sorted_windows[0]["start"] - sim_start
        })

    # Gaps between windows
    for i in range(len(sorted_windows) - 1):
        gap_start = sorted_windows[i]["end"]
        gap_end = sorted_windows[i + 1]["start"]
        if gap_end > gap_start:
            rest_windows.append({
                "start": gap_start,
                "end": gap_end,
                "duration": gap_end - gap_start
            })

    # Gap after last window
    if sorted_windows[-1]["end"] < sim_end:
        rest_windows.append({
            "start": sorted_windows[-1]["end"],
            "end": sim_end,
            "duration": sim_end - sorted_windows[-1]["end"]
        })

    return rest_windows


def execute_route_generation(args) -> None:
    """Execute vehicle route generation."""
    logger = logging.getLogger(__name__)

    generate_vehicle_routes(
        net_file=CONFIG.network_file,
        output_file=CONFIG.routes_file,
        num_vehicles=args.num_vehicles,
        private_traffic_seed=get_private_traffic_seed(args),
        public_traffic_seed=get_public_traffic_seed(args),
        routing_strategy=args.routing_strategy,
        vehicle_types=args.vehicle_types,
        passenger_routes=getattr(
            args, FIELD_PASSENGER_ROUTES, DEFAULT_PASSENGER_ROUTE_PATTERN),
        public_routes=getattr(args, FIELD_PUBLIC_ROUTES,
                              DEFAULT_PUBLIC_ROUTE_PATTERN),
        end_time=args.end_time,
        departure_pattern=args.departure_pattern,
        start_time_hour=args.start_time_hour,
        grid_dimension=int(args.grid_dimension)
    )
    try:
        verify_generate_vehicle_routes(
            net_file=CONFIG.network_file,
            output_file=CONFIG.routes_file,
            num_vehicles=args.num_vehicles,
            # Keep validation using private seed for backward compatibility
            seed=get_private_traffic_seed(args),
        )
    except ValidationError as ve:
        logger.error(f"Failed to generate vehicle routes: {ve}")
        raise
    # logger.info("Generated vehicle routes successfully")
