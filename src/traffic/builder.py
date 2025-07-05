# src/traffic/builder.py
from __future__ import annotations
import random
from pathlib import Path
from sumolib.net import readNet

from ..config import CONFIG
from .edge_sampler import AttractivenessBasedEdgeSampler
from .routing import RoutingMixStrategy, parse_routing_strategy
from .vehicle_types import parse_vehicle_types, get_vehicle_weights
from .xml_writer import write_routes


def generate_vehicle_routes(net_file: str | Path,
                            output_file: str | Path,
                            num_vehicles: int,
                            seed: int = CONFIG.RNG_SEED,
                            routing_strategy: str = "shortest 100",
                            vehicle_types: str = CONFIG.DEFAULT_VEHICLE_TYPES) -> None:
    """
    Orchestrates vehicle creation and writes a .rou.xml.
    
    Args:
        net_file: Path to SUMO network file
        output_file: Output route file path
        num_vehicles: Number of vehicles to generate
        seed: Random seed for reproducibility
        routing_strategy: Routing strategy specification (e.g., "shortest 70 realtime 30")
        vehicle_types: Vehicle types specification (e.g., "passenger 70 commercial 20 public 10")
    """
    rng = random.Random(seed)
    net = readNet(str(net_file))

    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]

    sampler = AttractivenessBasedEdgeSampler(rng)
    
    # Parse and initialize routing strategies
    strategy_percentages = parse_routing_strategy(routing_strategy)
    routing_mix = RoutingMixStrategy(net, strategy_percentages)
    
    # Parse and initialize vehicle types
    vehicle_distribution = parse_vehicle_types(vehicle_types)
    vehicle_names, vehicle_weights = get_vehicle_weights(vehicle_distribution)
    
    print(f"Using routing strategies: {strategy_percentages}")
    print(f"Using vehicle types: {vehicle_distribution}")

    vehicles = []
    for vid in range(num_vehicles):
        vtype = rng.choices(
            population=vehicle_names,
            weights=vehicle_weights,
            k=1
        )[0]

        # Assign routing strategy to this vehicle
        assigned_strategy = routing_mix.assign_strategy_to_vehicle(f"veh{vid}", rng)
        
        route_edges = []
        for _ in range(20):                       # retry up to 20 times
            start_edge = sampler.sample_start_edges(edges, 1)[0]
            end_edge   = sampler.sample_end_edges(edges, 1)[0]
            if end_edge == start_edge:
                continue
            route_edges = routing_mix.compute_route(assigned_strategy, start_edge, end_edge)
            if route_edges:
                break
        else:
            print(f"⚠️  Could not find a path for vehicle {vid} using {assigned_strategy} strategy; skipping.")
            continue

        # make 100% sure we have a list of edges
        if not route_edges:
            print(f"⚠️  Empty route for vehicle {vid}; skipping.")
            continue
        vehicles.append({
            "id":              f"veh{vid}",
            "type":            vtype,
            "depart":          vid,
            "from_edge":       start_edge,
            "to_edge":         end_edge,
            "route_edges":     route_edges,
            "routing_strategy": assigned_strategy,
        })

    write_routes(output_file, vehicles, CONFIG.vehicle_types)
    print(f"Wrote {len(vehicles)} vehicles → {output_file}")
