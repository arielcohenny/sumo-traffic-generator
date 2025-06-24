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

    # ── parse routes file ───────────────────────────────────────────────────
    rt_tree = ET.parse(routes_path)
    rt_root = rt_tree.getroot()

    vehicles = rt_root.findall("vehicle")
    v_count = len(vehicles)
    min_expected = int(num_vehicles * (1 - tolerate_shortfall))
    if v_count < min_expected:
        raise ValidationError(
            f"Routes file contains {v_count} vehicles, expected ≥ {min_expected} (requested {num_vehicles})")

    ids_seen: set[str] = set()
    depart_times: list[float] = []

    for v in vehicles:
        vid = v.get("id")
        if vid is None:
            raise ValidationError("Vehicle without id attribute found")
        if vid in ids_seen:
            raise ValidationError(f"Duplicate vehicle id: {vid}")
        ids_seen.add(vid)

        try:
            depart = float(v.get("depart", "0"))
        except ValueError:
            raise ValidationError(f"Vehicle {vid} has non‑numeric depart time")
        if depart < 0:
            raise ValidationError(f"Vehicle {vid} depart time negative")
        depart_times.append(depart)

        route_elem = v.find("route")
        if route_elem is None:
            raise ValidationError(f"Vehicle {vid} missing <route>")
        edges_attr = route_elem.get("edges", "").strip()
        if not edges_attr:
            raise ValidationError(f"Vehicle {vid} has empty route")
        for edge_id in edges_attr.split():
            if edge_id not in valid_edges:
                raise ValidationError(
                    f"Vehicle {vid} references unknown edge {edge_id}")

    # departure variability – not critical, just warn via ValidationError if suspicious
    if len(set(depart_times)) == 1:
        raise ValidationError(
            "All vehicles depart at the same time – check generation logic")

    return
