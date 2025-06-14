# src/traffic/builder.py
from __future__ import annotations
import random
from pathlib import Path
from sumolib.net import readNet

from ..config import CONFIG
from .edge_sampler import AttractivenessBasedEdgeSampler
from .routing import ShortestPathRoutingStrategy
from .xml_writer import write_routes


def generate_vehicle_routes(net_file: str | Path,
                            output_file: str | Path,
                            num_vehicles: int,
                            seed: int = CONFIG.RNG_SEED) -> None:
    """
    Orchestrates vehicle creation and writes a .rou.xml.
    """
    rng = random.Random(seed)
    net = readNet(str(net_file))

    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]

    sampler = AttractivenessBasedEdgeSampler(rng)
    router = ShortestPathRoutingStrategy(net)

    vehicles = []
    for vid in range(num_vehicles):
        vtype = rng.choices(
            population=list(CONFIG.vehicle_types.keys()),
            weights=CONFIG.vehicle_weights,
            k=1
        )[0]

        route_edges = []
        for _ in range(20):                       # retry up to 20 times
            start_edge = sampler.sample_start_edges(edges, 1)[0]
            end_edge   = sampler.sample_end_edges(edges, 1)[0]
            if end_edge == start_edge:
                continue
            route_edges = router.compute_route(start_edge, end_edge)
            if route_edges:
                break
        else:
            print(f"⚠️  Could not find a path for vehicle {vid}; skipping.")
            continue

        # make 100% sure we have a list of edges
        if not route_edges:
            print(f"⚠️  Empty route for vehicle {vid}; skipping.")
            continue
        vehicles.append({
            "id":          f"veh{vid}",
            "type":        vtype,
            "depart":      vid,
            "from_edge":   start_edge,
            "to_edge":     end_edge,
            "route_edges": route_edges,
        })

    write_routes(output_file, vehicles, CONFIG.vehicle_types)
    print(f"Wrote {len(vehicles)} vehicles → {output_file}")
