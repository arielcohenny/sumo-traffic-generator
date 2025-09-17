# src/traffic/builder.py
from __future__ import annotations
import random
import math
from pathlib import Path
from typing import List, Tuple
from sumolib.net import readNet

from ..config import CONFIG
from src.constants import (DEFAULT_VEHICLE_TYPES, SECONDS_IN_DAY, DEFAULT_VEHICLES_DAILY_PER_ROUTE,
                        MORNING_RUSH_START, MORNING_RUSH_END, EVENING_RUSH_START, 
                        EVENING_RUSH_END, TEMPORAL_BIAS_STRENGTH)
from .routing import RoutingMixStrategy, parse_routing_strategy
from .vehicle_types import parse_vehicle_types, get_vehicle_weights
from .xml_writer import write_routes
from ..network.generate_grid import classify_edges

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
    
    As specified in ROUTES2.md: "attractiveness has 4 time phases and based on the 
    vehicle's departure time we'll know which one to use"
    
    Args:
        departure_time_seconds: Vehicle departure time in seconds since midnight
        
    Returns:
        Phase name for attractiveness sampling: "morning_peak", "midday_offpeak", "evening_peak", or "night_low"
    """
    # Convert seconds to hours (0-24 format)
    hour = (departure_time_seconds / 3600) % 24
    
    # Define phase boundaries based on realistic traffic patterns
    # Morning peak: 6am-10am (rush hour commuting)
    if 6 <= hour < 10:
        return "morning_peak"
    # Evening peak: 4pm-8pm (evening rush hour)
    elif 16 <= hour < 20:
        return "evening_peak"
    # Night low: 10pm-6am (overnight hours)
    elif hour >= 22 or hour < 6:
        return "night_low"
    # Midday off-peak: 10am-4pm (daytime non-rush)
    else:
        return "midday_offpeak"


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
    if len(parts) != 8:
        raise ValueError(f"Route pattern must have 4 pairs, got: {route_pattern_str}")
    
    patterns = {}
    for i in range(0, len(parts), 2):
        pattern = parts[i]
        percentage = float(parts[i + 1])
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
        pattern = rng.choices(pattern_names, weights=pattern_weights, k=1)[0]
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
        biased_weights = _apply_temporal_bias_to_route_patterns(departure_time, route_patterns)
        
        # Assign pattern using temporally biased weights
        pattern = rng.choices(pattern_names, weights=biased_weights, k=1)[0]
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
    hour = (departure_time_seconds / 3600) % 24
    
    # Start with base weights
    weights = list(base_patterns.values())
    pattern_names = list(base_patterns.keys())
    
    # Apply temporal bias using constants
    if MORNING_RUSH_START <= hour < MORNING_RUSH_END:
        # Morning rush: favor in-bound routes (commuting to work/business zones)
        for i, pattern in enumerate(pattern_names):
            if pattern == "in":
                weights[i] *= TEMPORAL_BIAS_STRENGTH
            elif pattern == "out":
                weights[i] *= (2.0 - TEMPORAL_BIAS_STRENGTH)  # Reduce out-bound
                
    elif EVENING_RUSH_START <= hour < EVENING_RUSH_END:
        # Evening rush: favor out-bound routes (commuting from work back home)
        for i, pattern in enumerate(pattern_names):
            if pattern == "out":
                weights[i] *= TEMPORAL_BIAS_STRENGTH
            elif pattern == "in":
                weights[i] *= (2.0 - TEMPORAL_BIAS_STRENGTH)  # Reduce in-bound
    
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
        phase = determine_attractiveness_phase_from_departure_time(departure_time)
        
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
        phase = determine_attractiveness_phase_from_departure_time(departure_time)
        
        # Apply route pattern targeting to determine the effective attractiveness direction
        effective_direction = self._get_effective_attractiveness_direction(direction, route_pattern)
        
        # Get phase-specific weights with route pattern targeting
        weights = self._get_phase_weights_with_targeting(edges, effective_direction, phase, route_pattern)
        
        # Sample edges using targeted weights
        return [e.getID() for e in self.rng.choices(edges, weights=weights, k=n)]
    
    def _get_effective_attractiveness_direction(self, base_direction: str, route_pattern: str) -> str:
        """
        Determine the effective attractiveness direction based on route pattern targeting.
        
        For in-bound and out-bound routes, we want to target business zones (high attractiveness),
        so we need to choose the direction that represents business zone activity.
        """
        if route_pattern == "in":
            # In-bound routes: We want to target high-arrival zones (business districts) for END edges
            # Start edges use normal depart logic, end edges target high arrival attractiveness
            return base_direction  # Use the provided direction
            
        elif route_pattern == "out":
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
        use_attractiveness = self._should_use_attractiveness(route_pattern, direction)
        
        if not use_attractiveness:
            # Return uniform weights when attractiveness is irrelevant (boundary edges)
            return [1.0] * len(edges)
        
        # Apply phase-specific attractiveness when relevant (inner edges)
        weights = []
        for edge in edges:
            # Get phase-specific attractiveness attribute  
            attr_name = f"{phase}_{direction}_attractiveness"
            weight = float(getattr(edge, attr_name, 0.0) or 0.0)
            weights.append(weight)
        
        # Fallback to uniform if all weights are zero
        return weights if any(weights) else [1.0] * len(edges)
    
    def _should_use_attractiveness(self, route_pattern: str, direction: str) -> bool:
        """
        Determine if attractiveness should be used based on route pattern and direction.
        
        Returns True when sampling inner edges (attractiveness relevant),
        False when sampling boundary edges (attractiveness irrelevant).
        """
        if route_pattern == "in":
            # In-bound: start boundary (irrelevant), end inner (relevant)
            return direction == "arrive"
        elif route_pattern == "out":
            # Out-bound: start inner (relevant), end boundary (irrelevant)
            return direction == "depart"
        elif route_pattern == "inner":
            # Inner: both start and end inner (relevant for both)
            return True
        elif route_pattern == "pass":
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
            attr_name = f"{phase}_{direction}_attractiveness"
            weight = float(getattr(edge, attr_name, 0.0) or 0.0)
            weights.append(weight)
        
        # Fallback to uniform if all weights are zero
        return weights if any(weights) else [1.0] * len(edges)


def load_attractiveness_attributes_from_xml(edges, net_file_path):
    """
    Load attractiveness attributes from XML network file and attach them to edge objects.
    
    sumolib.net.readNet() doesn't automatically parse custom attributes like attractiveness,
    so we need to manually load them from the XML and attach to the edge objects.
    """
    import xml.etree.ElementTree as ET
    
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
        print(f"WARNING: Failed to load attractiveness attributes from XML: {e}")


def map_simulation_time_to_real_time(simulation_time_seconds: int, start_time_hour: float) -> int:
    """
    Map simulation time to real time for attractiveness phase determination.
    
    Args:
        simulation_time_seconds: Time within simulation (0 to end_time)
        start_time_hour: Real time hour when simulation starts (e.g., 8.0 for 8 AM)
        
    Returns:
        Real time in seconds since midnight for attractiveness phase calculations
        
    Example:
        map_simulation_time_to_real_time(0, 8.0) -> 28800 (8:00 AM)
        map_simulation_time_to_real_time(3600, 8.0) -> 32400 (9:00 AM)
    """
    return int(start_time_hour * 3600 + simulation_time_seconds)


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
    
    This is the new route generation system that implements the 4-pattern route system
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
        departure_pattern: Departure pattern ("six_periods", "uniform", "rush_hours:...")
        start_time_hour: Start time in hours (0-24)
        grid_dimension: Grid dimension for edge classification
    """
    # Load network and filter edges
    net = readNet(str(net_file))
    all_edges = [e for e in net.getEdges() if e.getFunction() != "internal"]
    
    # Load attractiveness attributes from XML and attach to edge objects
    # (sumolib.net.readNet doesn't automatically parse custom attributes)
    load_attractiveness_attributes_from_xml(all_edges, net_file)
    
    # Classify edges into boundary and inner for route pattern restrictions
    boundary_edges, inner_edges = classify_edges(grid_dimension)
    
    # Convert edge ID lists to edge objects - only use tail edges (without _H_s or _H_node suffixes)
    # as specified in ROUTES2.md: "we only examine the tail part of edges, namely edges without suffix _H_s or _H_node"
    boundary_edge_objs = [e for e in all_edges if e.getID() in boundary_edges]
    inner_edge_objs = [e for e in all_edges if e.getID() in inner_edges]
    # For inner routes: combine boundary and inner tail edges (all possible tail edges)
    all_tail_edge_objs = boundary_edge_objs + inner_edge_objs
    all_edge_objs = all_edges
    
    print(f"Classified {len(boundary_edge_objs)} boundary edges and {len(inner_edge_objs)} inner edges")
    
    # Debug: Print a few examples of boundary vs inner edges
    print(f"Sample boundary edges: {[e.getID() for e in boundary_edge_objs[:5]]}")
    print(f"Sample inner edges: {[e.getID() for e in inner_edge_objs[:5]]}")
    
    # Parse configurations
    vehicle_distribution = parse_vehicle_types(vehicle_types)
    passenger_route_patterns = parse_route_patterns(passenger_routes)
    public_route_patterns = parse_route_patterns(public_routes)
    strategy_percentages = parse_routing_strategy(routing_strategy)
    
    print(f"Using routing strategies: {strategy_percentages}")
    print(f"Using vehicle types: {vehicle_distribution}")
    print(f"Using passenger route patterns: {passenger_route_patterns}")
    print(f"Using public route patterns: {public_route_patterns}")
    print(f"Using private traffic seed: {private_traffic_seed}")
    print(f"Using public traffic seed: {public_traffic_seed}")
    
    # Calculate vehicle counts by type
    vehicle_names, vehicle_weights = get_vehicle_weights(vehicle_distribution)
    passenger_count = int((vehicle_distribution.get('passenger', 0) / 100.0) * num_vehicles)
    public_count = num_vehicles - passenger_count
    
    print(f"Generating {passenger_count} passenger vehicles and {public_count} public vehicles")
    
    # Create RNGs
    private_rng = random.Random(private_traffic_seed)
    public_rng = random.Random(public_traffic_seed)
    
    # Generate vehicles
    vehicles = []
    vehicle_id_counter = 0  # Unified counter for all vehicles to ensure 'veh' prefix compatibility
    
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
    vehicles.sort(key=lambda v: v["depart"])
    
    print(f"Generated {len(vehicles)} total vehicles")
    
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
    departure_times = calculate_temporal_departure_times(passenger_count, departure_pattern, start_time_hour, end_time)
    
    # Assign route patterns to vehicles with temporal bias for realistic traffic flows
    # Convert simulation departure times to real times for temporal bias calculations
    real_departure_times = [map_simulation_time_to_real_time(dt, start_time_hour) for dt in departure_times]
    assigned_patterns = assign_route_pattern_with_temporal_bias(real_departure_times, route_patterns, private_rng)
    
    # Generate individual passenger vehicles
    for i in range(passenger_count):
        vid = f"veh{vehicle_id_counter}"
        vehicle_id_counter += 1  # Reserve ID immediately to prevent duplicates
        route_pattern = assigned_patterns[i]
        departure_time = departure_times[i] if i < len(departure_times) else departure_times[-1]
        
        # Assign routing strategy (passenger vehicles use configured strategies)
        assigned_strategy = routing_mix.assign_strategy_to_vehicle(vid, private_rng)
        
        # Select edge pools based on route pattern
        start_edge_pool, end_edge_pool = _get_edge_pools_for_pattern(route_pattern, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs)
        
        # Generate route with pattern restrictions
        route_edges = []
        start_edge = None
        end_edge = None
        
        for _ in range(MAX_ROUTE_RETRIES):
            # Use route-pattern-aware attractiveness targeting (ROUTES2.md line 18)
            # "In-bound routes target high-arrival attractiveness inner edges, 
            #  out-bound routes originate from high-departure attractiveness inner edges"
            # Map simulation time to real time for attractiveness phase determination
            real_time = map_simulation_time_to_real_time(departure_time, start_time_hour)
            start_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                start_edge_pool, "depart", real_time, route_pattern, 1)
            end_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                end_edge_pool, "arrive", real_time, route_pattern, 1)
            
            start_edge = start_edge_ids[0]
            end_edge = end_edge_ids[0] 
            if end_edge == start_edge:
                continue
            route_edges = routing_mix.compute_route(assigned_strategy, start_edge, end_edge)
            if route_edges:
                break
        else:
            print(f"⚠️  Could not find a path for passenger vehicle {vid} using {assigned_strategy} strategy; skipping.")
            continue
        
        if not route_edges:
            print(f"⚠️  Empty route for passenger vehicle {vid}; skipping.")
            continue
        
        vehicles.append({
            "id": vid,
            "type": "passenger",
            "depart": int(departure_time),
            "from_edge": start_edge,
            "to_edge": end_edge,
            "route_edges": route_edges,
            "routing_strategy": assigned_strategy,
        })
    
    print(f"Generated {len(vehicles)} passenger vehicles")
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
    ideal_num_vehicles_per_route = (end_time / SECONDS_IN_DAY) * DEFAULT_VEHICLES_DAILY_PER_ROUTE
    if ideal_num_vehicles_per_route <= 0:
        ideal_num_vehicles_per_route = 1.0
    
    num_public_routes = top(public_count / ideal_num_vehicles_per_route)
    base_vehicles_per_route = public_count // num_public_routes
    extra_vehicles = public_count % num_public_routes  # Distribute remaining vehicles
    
    print(f"Creating {num_public_routes} public routes with ~{base_vehicles_per_route}-{base_vehicles_per_route + 1} vehicles each")
    
    # Calculate route distribution based on percentages
    route_counts = {}
    for pattern, percentage in route_patterns.items():
        route_counts[pattern] = max(1, int((percentage / 100.0) * num_public_routes))
    
    # Handle case where there are not enough vehicles for all route types
    total_planned_routes = sum(route_counts.values())
    if total_planned_routes > num_public_routes:
        # Use only the highest percentage pattern
        max_pattern = max(route_patterns.keys(), key=lambda k: route_patterns[k])
        route_counts = {max_pattern: num_public_routes}
        print(f"⚠️  Not enough public vehicles for all route types. Using only '{max_pattern}' routes.")
    
    # Create routing strategy for public vehicles (always shortest path)
    public_routing_strategy = {"shortest": 100.0}
    routing_mix = RoutingMixStrategy(net, public_routing_strategy)
    phase_sampler = PhaseSpecificEdgeSampler(public_rng)
    
    # Calculate departure times for ALL public vehicles globally, then distribute to routes
    # This ensures proper temporal spread across the entire simulation time
    all_departure_times = calculate_temporal_departure_times(public_count, departure_pattern, start_time_hour, end_time)
    
    route_id = 0
    vehicles_created = 0  # Track how many vehicles we've created locally
    
    # Generate routes for each pattern type
    for pattern, route_count in route_counts.items():
        for route_idx in range(route_count):
            # Select edge pools for this route pattern
            start_edge_pool, end_edge_pool = _get_edge_pools_for_pattern(pattern, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs)
            
            # Generate route definition
            route_edges = []
            start_edge = None
            end_edge = None
            
            for _ in range(MAX_ROUTE_RETRIES):
                # Use phase-specific attractiveness for public routes based on first departure time
                # As specified in ROUTES2.md line 99: "For each vehicle, based on the departure time we can use the attractiveness values"
                first_departure_time = all_departure_times[vehicles_created] if vehicles_created < len(all_departure_times) else 0
                # Map simulation time to real time for attractiveness phase determination
                real_first_departure_time = map_simulation_time_to_real_time(first_departure_time, start_time_hour)
                
                start_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                    start_edge_pool, "depart", real_first_departure_time, pattern, 1)
                end_edge_ids = phase_sampler.sample_edges_with_route_pattern_targeting(
                    end_edge_pool, "arrive", real_first_departure_time, pattern, 1)
                
                start_edge = start_edge_ids[0]
                end_edge = end_edge_ids[0]
                if end_edge == start_edge:
                    continue
                route_edges = routing_mix.compute_route("shortest", start_edge, end_edge)
                if route_edges:
                    break
            else:
                print(f"⚠️  Could not find a path for public route {route_id} with pattern {pattern}; skipping.")
                continue
            
            if not route_edges:
                print(f"⚠️  Empty route for public route {route_id}; skipping.")
                continue
            
            # Generate vehicles for this route - distribute extra vehicles among first routes
            vehicles_for_this_route = base_vehicles_per_route
            if route_id < extra_vehicles:  # First 'extra_vehicles' routes get one extra vehicle
                vehicles_for_this_route += 1
            
            actual_vehicles = min(vehicles_for_this_route, len(all_departure_times) - vehicles_created)
            # Safety check to not exceed total vehicle count
            if vehicles_created + actual_vehicles > public_count:
                actual_vehicles = public_count - vehicles_created
            
            for vehicle_idx in range(actual_vehicles):
                if vehicles_created >= public_count:
                    break
                    
                departure_time = all_departure_times[vehicles_created]
                vid = f"veh{vehicle_id_counter}"
                
                vehicles.append({
                    "id": vid,
                    "type": "public",
                    "depart": int(departure_time),
                    "from_edge": start_edge,
                    "to_edge": end_edge,
                    "route_edges": route_edges,
                    "routing_strategy": "shortest",  # Public always uses shortest
                })
                
                vehicle_id_counter += 1
                vehicles_created += 1
            
            route_id += 1
    
    print(f"Generated {len(vehicles)} public vehicles across {route_id} routes")
    return vehicles, vehicle_id_counter


def _get_edge_pools_for_pattern(pattern: str, all_tail_edge_objs, boundary_edge_objs, inner_edge_objs) -> tuple:
    """Get start and end edge pools based on route pattern."""
    if pattern == "in":
        # In-bound: start from boundary, end at inner edges
        return boundary_edge_objs, inner_edge_objs
    elif pattern == "out":
        # Out-bound: start from inner edges, end at boundary
        return inner_edge_objs, boundary_edge_objs
    elif pattern == "inner":
        # Inner: start and end from inner edges only
        return inner_edge_objs, inner_edge_objs
    elif pattern == "pass":
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


def calculate_temporal_departure_times(num_vehicles: int, departure_pattern: str, start_time: float, end_time: int) -> List[int]:
    """
    Calculate deterministic departure times for all vehicles based on departure pattern.
    
    This function pre-calculates all departure times to ensure exact percentage distribution
    and even temporal spacing within each period, replacing the current random individual
    selection approach.
    
    Args:
        num_vehicles: Total number of vehicles to generate departure times for
        departure_pattern: Pattern specification ("uniform", "six_periods", "rush_hours:...")
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
    
    if departure_pattern == "uniform":
        # Even distribution across simulation duration
        # start_time_hour is used for time window mapping to real time for attractiveness/bias calculations
        if num_vehicles <= 0:
            return departure_times
            
        interval = end_time / num_vehicles
        for i in range(num_vehicles):
            departure_time = i * interval  # Simulation time (0 to end_time)
            departure_times.append(int(departure_time))
            
    elif departure_pattern == "six_periods":
        # Research-based 6-period distribution with exact percentages
        departure_times = _calculate_six_periods_deterministic(num_vehicles, end_time)
        
    elif departure_pattern.startswith("rush_hours:"):
        # Custom rush hours pattern with exact percentages
        departure_times = _calculate_rush_hours_deterministic(num_vehicles, departure_pattern, end_time)
        
    else:
        # Default to six_periods for unknown patterns
        departure_times = _calculate_six_periods_deterministic(num_vehicles, end_time)
    
    return sorted(departure_times)


def _calculate_six_periods_deterministic(num_vehicles: int, end_time: int) -> List[int]:
    """Calculate deterministic six periods departure times with exact percentages."""
    departure_times = []
    
    # Scale to simulation time
    scale_factor = end_time / 86400  # 24 hours = 86400 seconds
    
    # Six periods with exact percentages
    periods = [
        {"name": "morning", "start": MORNING_START * 3600, "end": MORNING_END * 3600, "weight": MORNING_WEIGHT},
        {"name": "morning_rush", "start": MORNING_RUSH_START * 3600, "end": MORNING_RUSH_END * 3600, "weight": MORNING_RUSH_WEIGHT},
        {"name": "noon", "start": NOON_START * 3600, "end": NOON_END * 3600, "weight": NOON_WEIGHT},
        {"name": "evening_rush", "start": EVENING_RUSH_START * 3600, "end": EVENING_RUSH_END * 3600, "weight": EVENING_RUSH_WEIGHT},
        {"name": "evening", "start": EVENING_START * 3600, "end": EVENING_END * 3600, "weight": EVENING_WEIGHT},
        {"name": "night", "start": NIGHT_START * 3600, "end": NIGHT_END * 3600, "weight": NIGHT_WEIGHT},
    ]
    
    total_weight = sum(p["weight"] for p in periods)
    
    for period in periods:
        # Calculate exact number of vehicles for this period
        period_vehicles = int((period["weight"] / total_weight) * num_vehicles)
        
        if period_vehicles > 0:
            if period["name"] == "night":
                # Handle night period wrapping (10pm-6am next day)
                # 25% in evening part (10pm-12am), 75% in early morning (12am-6am)
                evening_vehicles = int(period_vehicles * NIGHT_EVENING_RATIO)
                morning_vehicles = period_vehicles - evening_vehicles
                
                # Evening part (10pm-12am)
                if evening_vehicles > 0:
                    evening_start = NIGHT_START * 3600 * scale_factor
                    evening_end = 24 * 3600 * scale_factor
                    evening_interval = (evening_end - evening_start) / evening_vehicles
                    for i in range(evening_vehicles):
                        departure_time = evening_start + i * evening_interval
                        departure_times.append(int(departure_time))
                
                # Early morning part (12am-6am)
                if morning_vehicles > 0:
                    morning_start = 0
                    morning_end = EARLY_MORNING_END * 3600 * scale_factor
                    morning_interval = (morning_end - morning_start) / morning_vehicles
                    for i in range(morning_vehicles):
                        departure_time = morning_start + i * morning_interval
                        departure_times.append(int(departure_time))
            else:
                # Regular period with even distribution
                period_start = period["start"] * scale_factor
                period_end = period["end"] * scale_factor
                period_interval = (period_end - period_start) / period_vehicles
                
                for i in range(period_vehicles):
                    departure_time = period_start + i * period_interval
                    departure_times.append(int(departure_time))
    
    return departure_times


def _calculate_rush_hours_deterministic(num_vehicles: int, pattern: str, end_time: int) -> List[int]:
    """Calculate deterministic rush hours departure times with exact percentages."""
    departure_times = []
    
    # Parse pattern (e.g., "rush_hours:7-9:40,17-19:30,rest:10")
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
    
    # Calculate vehicles for each period
    total_rush_weight = sum(p["weight"] for p in rush_periods)
    total_weight = total_rush_weight + rest_weight
    
    scale_factor = end_time / 86400
    
    # Distribute vehicles to rush hour periods
    for period in rush_periods:
        period_vehicles = int((period["weight"] / total_weight) * num_vehicles)
        if period_vehicles > 0:
            period_start = period["start"] * scale_factor
            period_end = period["end"] * scale_factor
            period_interval = (period_end - period_start) / period_vehicles
            
            for i in range(period_vehicles):
                departure_time = period_start + i * period_interval
                departure_times.append(int(departure_time))
    
    # Distribute vehicles to rest time (outside rush hours)
    rest_vehicles = int((rest_weight / total_weight) * num_vehicles)
    if rest_vehicles > 0:
        # Calculate rest windows (24 hours minus rush hours)
        rest_windows = _compute_rest_windows(rush_periods, end_time)
        rest_total_time = sum(window[1] - window[0] for window in rest_windows)
        
        if rest_total_time > 0:
            for window_start, window_end in rest_windows:
                window_time = window_end - window_start
                window_vehicles = int((window_time / rest_total_time) * rest_vehicles)
                
                if window_vehicles > 0:
                    window_interval = (window_end - window_start) / window_vehicles
                    for i in range(window_vehicles):
                        departure_time = window_start + i * window_interval
                        departure_times.append(int(departure_time))
    
    return departure_times


def _compute_rest_windows(rush_periods: List[dict], end_time: int) -> List[Tuple[float, float]]:
    """Compute time windows outside of rush hour periods."""
    scale_factor = end_time / 86400
    
    # Create occupied time ranges from rush periods
    occupied_ranges = []
    for period in rush_periods:
        start = period["start"] * scale_factor
        end = period["end"] * scale_factor
        occupied_ranges.append((start, end))
    
    # Sort by start time
    occupied_ranges.sort()
    
    # Find gaps between occupied ranges
    rest_windows = []
    simulation_start = 0
    simulation_end = end_time * SIMULATION_END_FACTOR
    
    current_pos = simulation_start
    for start, end in occupied_ranges:
        if current_pos < start:
            # Gap found
            rest_windows.append((current_pos, start))
        current_pos = max(current_pos, end)
    
    # Check for gap at the end
    if current_pos < simulation_end:
        rest_windows.append((current_pos, simulation_end))
    
    return rest_windows


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
        passenger_routes=getattr(args, 'passenger_routes', 'in 30 out 30 inner 25 pass 15'),
        public_routes=getattr(args, 'public_routes', 'in 25 out 25 inner 35 pass 15'),
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
            seed=get_private_traffic_seed(args),  # Keep validation using private seed for backward compatibility
        )
    except ValidationError as ve:
        logger.error(f"Failed to generate vehicle routes: {ve}")
        raise
    logger.info("Generated vehicle routes successfully")
