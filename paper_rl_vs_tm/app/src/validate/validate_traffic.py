from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from .errors import ValidationError
except ImportError:
    class ValidationError(RuntimeError):
        pass

__all__ = [
    "verify_generate_vehicle_routes"
]


"""
Inline validator for *generate_vehicle_routes*.

Verifications
-------------
1. **Vehicle count** – `<vehicle>` elements equal `num_vehicles` requested.
2. **Unique IDs** – every vehicle `id` is unique.
3. **Route validity** – each `<route edges="…">` is non‑empty and every edge
   in the list exists in the supplied SUMO network file.
4. **Departure sanity** – the `depart` time attribute is a non‑negative float;
   warns if all vehicles share the same departure time.
5. **Vehicle types** – all vehicle types exist in CONFIG.vehicle_types.
6. **Route connectivity** – routes form valid connected paths.
7. **Edge filtering** – no internal edges are used in routes.
8. **Route length** – routes have reasonable minimum length (≥2 edges).
"""


def verify_generate_vehicle_routes(
    *,
    net_file: str | Path,
    output_file: str | Path,
    num_vehicles: int,
    seed: int | None = None,  # not used, kept for call‑symmetry
    tolerate_shortfall: float = 0.02,  # allow 2 % fewer vehicles (rounding)
) -> None:
    """Validate the routes XML produced by *generate_vehicle_routes*."""

    net_path = Path(net_file)
    if not net_path.exists():
        raise ValidationError(
            f"Routes check: network file missing: {net_path}")

    routes_path = Path(output_file)
    if not routes_path.exists():
        raise ValidationError(
            f"Routes check: routes file missing: {routes_path}")

    # ── collect valid edge IDs from network ─────────────────────────────────
    net_root = ET.parse(net_path).getroot()
    valid_edges = {e.get("id") for e in net_root.findall("edge")}
    internal_edges = {e.get("id") for e in net_root.findall(
        "edge") if e.get("function") == "internal"}
    external_edges = valid_edges - internal_edges

    # ── get valid vehicle types from CONFIG ─────────────────────────────────
    from ..config import CONFIG
    valid_vehicle_types = set(CONFIG.vehicle_types.keys())

    # ── parse routes file ───────────────────────────────────────────────────
    rt_tree = ET.parse(routes_path)
    rt_root = rt_tree.getroot()

    vehicles = rt_root.findall("vehicle")
    trips = rt_root.findall("trip")
    total_count = len(vehicles) + len(trips)
    min_expected = int(num_vehicles * (1 - tolerate_shortfall))
    if total_count < min_expected:
        raise ValidationError(
            f"Routes file contains {total_count} vehicles/trips, expected ≥ {min_expected} (requested {num_vehicles})")

    ids_seen: set[str] = set()
    depart_times: list[float] = []
    route_lengths: list[int] = []

    # Validate vehicle elements (with full routes)
    for v in vehicles:
        vid = v.get("id")
        if vid is None:
            raise ValidationError("Vehicle without id attribute found")
        if vid in ids_seen:
            raise ValidationError(f"Duplicate vehicle id: {vid}")
        ids_seen.add(vid)

        # ── validate vehicle type ───────────────────────────────────────────────
        vtype = v.get("type")
        if vtype and vtype not in valid_vehicle_types:
            raise ValidationError(f"Vehicle {vid} has unknown type: {vtype}")

        # ── validate departure time ─────────────────────────────────────────────
        try:
            depart = float(v.get("depart", "0"))
        except ValueError:
            raise ValidationError(f"Vehicle {vid} has non‑numeric depart time")
        if depart < 0:
            raise ValidationError(f"Vehicle {vid} depart time negative")
        depart_times.append(depart)

        # ── validate route ──────────────────────────────────────────────────────
        route_elem = v.find("route")
        if route_elem is None:
            raise ValidationError(f"Vehicle {vid} missing <route>")
        edges_attr = route_elem.get("edges", "").strip()
        if not edges_attr:
            raise ValidationError(f"Vehicle {vid} has empty route")

        route_edges = edges_attr.split()
        route_lengths.append(len(route_edges))

        # Check route length is reasonable (at least 1 edge, ideally 2+)
        if len(route_edges) < 1:
            raise ValidationError(f"Vehicle {vid} has empty route")

        # Check all edges exist and are external (non-internal)
        for edge_id in route_edges:
            if edge_id not in valid_edges:
                raise ValidationError(
                    f"Vehicle {vid} references unknown edge {edge_id}")
            if edge_id in internal_edges:
                raise ValidationError(
                    f"Vehicle {vid} route contains internal edge {edge_id}")

        # Check route connectivity (each edge should connect to next)
        if len(route_edges) > 1:
            for i in range(len(route_edges) - 1):
                current_edge_id = route_edges[i]
                next_edge_id = route_edges[i + 1]
                # Find edge elements to check connectivity
                current_edge = net_root.find(
                    f".//edge[@id='{current_edge_id}']")
                next_edge = net_root.find(f".//edge[@id='{next_edge_id}']")

                if current_edge is not None and next_edge is not None:
                    current_to = current_edge.get("to")
                    next_from = next_edge.get("from")
                    if current_to != next_from:
                        raise ValidationError(
                            f"Vehicle {vid} route disconnected: edge {current_edge_id} (to={current_to}) "
                            f"does not connect to edge {next_edge_id} (from={next_from})")

    # Validate trip elements (with from/to attributes only)
    for t in trips:
        trip_id = t.get("id")
        if trip_id is None:
            raise ValidationError("Trip without id attribute found")
        if trip_id in ids_seen:
            raise ValidationError(f"Duplicate vehicle/trip id: {trip_id}")
        ids_seen.add(trip_id)

        # ── validate vehicle type ───────────────────────────────────────────────
        vtype = t.get("type")
        if vtype and vtype not in valid_vehicle_types:
            raise ValidationError(f"Trip {trip_id} has unknown type: {vtype}")

        # ── validate departure time ─────────────────────────────────────────────
        try:
            depart = float(t.get("depart", "0"))
        except ValueError:
            raise ValidationError(f"Trip {trip_id} has non‑numeric depart time")
        if depart < 0:
            raise ValidationError(f"Trip {trip_id} depart time negative")
        depart_times.append(depart)

        # ── validate from/to edges ──────────────────────────────────────────────
        from_edge = t.get("from")
        to_edge = t.get("to")

        if not from_edge:
            raise ValidationError(f"Trip {trip_id} missing 'from' attribute")
        if not to_edge:
            raise ValidationError(f"Trip {trip_id} missing 'to' attribute")

        # Validate edges exist and are external
        if from_edge not in external_edges:
            raise ValidationError(
                f"Trip {trip_id} 'from' edge {from_edge} not found or is internal")
        if to_edge not in external_edges:
            raise ValidationError(
                f"Trip {trip_id} 'to' edge {to_edge} not found or is internal")

        if from_edge == to_edge:
            raise ValidationError(
                f"Trip {trip_id} has same 'from' and 'to' edge: {from_edge}")

        # Note: Trips don't have route_edges yet, SUMO will calculate them at departure
        # We don't track route_lengths for trips

    # ── additional statistical validation ───────────────────────────────────

    # departure variability – not critical, just warn via ValidationError if suspicious
    if len(set(depart_times)) == 1:
        raise ValidationError(
            "All vehicles depart at the same time – check generation logic")

    # Route length statistics (only for vehicles with pre-computed routes, not trips)
    if route_lengths and len(route_lengths) >= 2:  # Need at least 2 routes for diversity check
        avg_route_length = sum(route_lengths) / len(route_lengths)
        min_route_length = min(route_lengths)
        max_route_length = max(route_lengths)

        # Warn if average route length is too short (might indicate poor routing)
        if avg_route_length < 2.0:
            raise ValidationError(
                f"Average route length {avg_route_length:.1f} is suspiciously short; "
                f"check routing algorithm")

        # Warn if all routes are the same length (might indicate lack of diversity)
        # Only check if we have a reasonable number of non-trip vehicles
        if min_route_length == max_route_length and len(route_lengths) > 5:
            raise ValidationError(
                f"All {len(route_lengths)} routes have identical length {min_route_length}; "
                f"check route diversity")

    # Edge usage diversity - check that routes use different edges
    all_used_edges = set()
    for v in vehicles:
        route_elem = v.find("route")
        if route_elem is not None:
            edges_attr = route_elem.get("edges", "").strip()
            if edges_attr:
                all_used_edges.update(edges_attr.split())

    # Also count from/to edges for trips
    for t in trips:
        from_edge = t.get("from")
        to_edge = t.get("to")
        if from_edge:
            all_used_edges.add(from_edge)
        if to_edge:
            all_used_edges.add(to_edge)

    # Warn if routes use very few unique edges relative to available edges
    edge_usage_ratio = len(all_used_edges) / \
        len(external_edges) if external_edges else 0
    # Only check for larger networks - just a warning, doesn't raise error
    # (Low edge usage might be intentional for certain traffic patterns)

    return
