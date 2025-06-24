# src/validate/validate_network.py
"""
Runtime invariants for network generation.

Each `verify_*` function is designed to be called **inline** in the production
pipeline (e.g. right after `generate_grid_network` in `cli.py`). They run in a
few milliseconds and abort the run early if an invariant is broken.

If any check fails the function raises a `ValidationError`, which is a plain
sub‑class of `RuntimeError`, so it will propagate up the call‑stack unless the
caller catches it.
"""

from __future__ import annotations

import re
from pathlib import Path

import sumolib  # SUMO's Python helper library

import json
from xml.etree import ElementTree as ET

__all__ = [
    "ValidationError",
    "verify_generate_grid_network",
]


class ValidationError(RuntimeError):
    """Raised when an inline runtime check discovers a violation."""


# ---------------------------------------------------------------------------
#  Grid‑network verification (inline after generate_grid_network)
# ---------------------------------------------------------------------------

def _theoretical_max_edges(grid_dim: int) -> int:
    """Return the maximum *directed* edge count for a full grid with no removals.

    For an *N*×*N* block grid there are ``(N+1)×N`` street *segments* per axis.
    Each segment becomes two directed edges (A→B and B→A), hence:

    ``max_edges = 2 (axis) × 2 (direction) × N × (N+1)``
    """
    return 4 * grid_dim * (grid_dim - 1)


def verify_generate_grid_network(
    seed: int,
    grid_dim: int,
    block_size_m: int,
    network_file: str | Path,
    blocks_removed: int,
) -> None:
    """Cheap invariants for a freshly generated orthogonal grid network.

    Parameters and exceptions are identical to the previous version, but the
    *edge* check is now smarter: it validates that the number of directed edges
    is *bounded* by the theoretical maximum, and—if no blocks were removed—
    exactly equals that maximum.
    """

    path = Path(network_file)
    if not path.exists():
        raise ValidationError(f"Network file missing: {path}")

    # 1 ── parse the network ---------------------------------------------------
    try:
        net = sumolib.net.readNet(str(path))
    except Exception as exc:  # pragma: no cover – malformed XML or similar
        raise ValidationError(
            f"Failed to parse network file {path}: {exc}") from exc

    # 2 ── visible junction count matches expectation -------------------------
    visible_nodes = [
        n for n in net.getNodes() if not n.getID().startswith(":")]
    expected_nodes = grid_dim ** 2 - blocks_removed
    if len(visible_nodes) != expected_nodes:
        raise ValidationError(
            f"junction‑count mismatch: expected {expected_nodes}, got {len(visible_nodes)}"
        )

    # 3 ── ID scheme sanity: <Letter><Digits> or similar -----------------------
    id_pattern = re.compile(r"^[A-Za-z]\d+$")
    malformed = [n.getID()
                 for n in visible_nodes if not id_pattern.match(n.getID())]
    if malformed:
        raise ValidationError(
            f"unexpected junction IDs: {', '.join(malformed)}")

    # 4 ── directed edge count within theoretical bounds ----------------------
    edge_num = len(net.getEdges())
    max_edges = _theoretical_max_edges(grid_dim)

    if blocks_removed == 0:
        if edge_num != max_edges:
            raise ValidationError(
                f"edge‑count mismatch: expected {max_edges}, got {edge_num}"
            )
    else:
        # after removing blocks, the graph must have *fewer* edges, but never 0
        if not (0 < edge_num < max_edges):
            raise ValidationError(
                f"edge‑count {edge_num} outside expected range (1…{max_edges - 1})"
            )

    # 5 ── bounding‑box within expected limits --------------------------------
    if hasattr(net, "getBBox"):
        xmin, ymin, xmax, ymax = net.getBBox()
    else:
        xmin, ymin, xmax, ymax = net.getBoundary()
    tol = 1e-3
    max_coord = (grid_dim - 1) * block_size_m + tol
    if abs(xmin) > tol or abs(ymin) > tol:
        raise ValidationError("Grid should start at (0,0) – got shifted bbox.")
    if xmax > max_coord + tol or ymax > max_coord + tol:
        raise ValidationError(
            f"Bounding box too large ({xmax:.2f}×{ymax:.2f} m); expected ≤{max_coord:.2f} m"
        )

    # Passed all checks – nothing to return
    return


# ---------------------------------------------------------------------------
#  Zone extraction verification (inline after extract_zones_from_junctions)
# ---------------------------------------------------------------------------
def verify_extract_zones_from_junctions(
    net_file: str | Path,
    out_dir: str | Path,
) -> None:
    """Validate that *zones.geojson* agrees with the junction grid in *net_file*.

    Cheap checks (≪ 10 ms):
    1. *zones.geojson* exists and is valid JSON.
    2. Number of features equals ``(Nx‑1) × (Ny‑1)``, where ``Nx``/``Ny`` are the
       counts of *visible* junction x/y coordinates in the SUMO net.
    3. Every feature has a unique ``zone_id`` and a Polygon geometry.
    """

    net_file = Path(net_file)
    out_dir = Path(out_dir)
    geojson_path = out_dir / "zones.geojson"

    if not geojson_path.exists():
        raise ValidationError(f"zones.geojson not found in {out_dir}")

    # 1 ── parse GeoJSON -------------------------------------------------------
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {geojson_path}: {exc}") from exc

    if geo.get("type") != "FeatureCollection":
        raise ValidationError("zones.geojson must be a FeatureCollection")

    features = geo.get("features", [])
    if not features:
        raise ValidationError("zones.geojson contains no features")

    # check uniqueness and geometry type
    seen_ids: set[str] = set()
    for feat in features:
        props = feat.get("properties", {})
        zid = props.get("zone_id")
        if zid is None:
            raise ValidationError("Feature without zone_id in zones.geojson")
        if zid in seen_ids:
            raise ValidationError(f"Duplicate zone_id {zid} in zones.geojson")
        seen_ids.add(zid)
        if feat.get("geometry", {}).get("type") != "Polygon":
            raise ValidationError(f"zone {zid} geometry is not Polygon")

    # 2 ── derive expected zone count from the network ------------------------
    tree = ET.parse(net_file)
    root = tree.getroot()

    xs: set[float] = set()
    ys: set[float] = set()
    for j in root.findall("junction"):
        jid = j.get("id")
        if jid.startswith(":"):
            continue
        xs.add(float(j.get("x")))
        ys.add(float(j.get("y")))

    nx, ny = len(xs), len(ys)
    expected = (nx - 1) * (ny - 1)
    actual = len(features)

    if actual != expected:
        raise ValidationError(
            f"zones.geojson has {actual} features; expected {expected} "
            f"((|X|-1)*(|Y|-1) from junction grid)"
        )

    # all good
    return


def verify_set_lane_counts(
    net_file_out: str | Path,
    *,
    min_lanes: int = 1,
    max_lanes: int = 3,
) -> None:
    """Validate lane counts and *practical* connectivity after mutation."""

    path = Path(net_file_out)
    if not path.exists():
        raise ValidationError(
            f"Lane‑count check: network file missing: {path}")

    try:
        net = sumolib.net.readNet(str(path))
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse network file {path}: {exc}") from exc

    counts: list[int] = []
    unreachable: list[str] = []

    for edge in net.getEdges():
        func = edge.getFunction() if hasattr(
            edge, "getFunction") else edge.func  # type: ignore
        if func == "internal":
            continue
        lanes = edge.getLanes()
        counts.append(len(lanes))
        for lane in lanes:
            if not lane.getIncoming():
                unreachable.append(lane.getID())

    if not counts:
        raise ValidationError(
            "Lane‑count check: no driveable edges found in network")

    # Bound check ----------------------------------------------------------------
    bad = [c for c in counts if c < min_lanes or c > max_lanes]
    if bad:
        raise ValidationError(
            f"{len(bad)} edges have lane count outside [{min_lanes}, {max_lanes}]"
        )

    # Distribution check ---------------------------------------------------------
    if all(c == counts[0] for c in counts):
        raise ValidationError(
            "All edges share the same lane number; randomisation may have failed")
    if min_lanes not in counts:
        raise ValidationError(
            f"No edge ended up with the minimum lane count ({min_lanes})")
    if max_lanes not in counts:
        raise ValidationError(
            f"No edge ended up with the maximum lane count ({max_lanes})")

    # Connectivity check ---------------------------------------------------------
    if unreachable:
        # geometry helpers
        xmin, ymin, xmax, ymax = net.getBoundary()
        tol = 1e-3

        def on_border(node) -> bool:
            x, y = node.getCoord()
            return abs(x - xmin) < tol or abs(x - xmax) < tol or abs(y - ymin) < tol or abs(y - ymax) < tol

        interior_violations: list[str] = []
        for lane_id in unreachable:
            lane_obj = net.getLane(lane_id)
            edge = lane_obj.getEdge()

            # perimeter tolerance
            if on_border(edge.getFromNode()) and on_border(edge.getToNode()):
                continue

            # upstream‑lane‑count tolerance
            from_node = edge.getFromNode()
            max_in_lanes = max((len(e.getLanes())
                               for e in from_node.getIncoming()), default=0)
            try:
                lane_idx = int(lane_id.rsplit("_", 1)[-1])
            except ValueError:
                lane_idx = -1  # cannot parse, treat as interior

            if lane_idx >= max_in_lanes:
                continue  # tolerated: no upstream lane with this index exists

            interior_violations.append(lane_id)

        if interior_violations:
            sample = ", ".join(interior_violations[:5])
            more = "" if len(
                interior_violations) <= 5 else f" … and {len(interior_violations)-5} more"
            raise ValidationError(
                f"{len(interior_violations)} interior lanes have no incoming connection: {sample}{more}"
            )

    # all good
    return

# ---------------------------------------------------------------------------
#  Edge attractiveness verification (inline after assign_edge_attractiveness)
# ---------------------------------------------------------------------------

    """Validate *assign_edge_attractiveness* output.

Checks performed
----------------
1. **Presence**   Every non‑internal `<edge>` carries *both* XML attributes
   `depart_attractiveness` and `arrive_attractiveness`.
2. **Type & Range**   Values are non‑negative integers (netconvert often wraps a
   single‑lane value as `[3]`; brackets are stripped first).
3. **Distribution sanity**   The sample mean of each attribute must sit within
   ±50 % (default *tolerance*) of the corresponding Poisson λ used when the
   helper was called (`lambda_depart`, `lambda_arrive`).  This weeds out cases
   where the attribute was written but the random draw silently failed.
4. **Variability**   Rejects the degenerate case where every edge got the same
   value, indicating the random generator wasn’t invoked.
"""


def _mean(vals: list[int]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _to_float(val):
    """Coerce CLI values that sometimes arrive as tuple / str → float."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    return float(val)


def verify_assign_edge_attractiveness(
    seed: int,
    net_file: str | Path,
    *,
    lambda_depart: float | str | tuple,
    lambda_arrive: float | str | tuple,
    tolerance: float = 0.5,  # ±50 % of λ is acceptable
) -> None:
    """Validate that attractiveness attributes exist and are plausible."""

    lambda_depart_f = _to_float(lambda_depart)
    lambda_arrive_f = _to_float(lambda_arrive)

    path = Path(net_file)
    if not path.exists():
        raise ValidationError(
            f"Attractiveness check: network file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    dep_vals: list[int] = []
    arr_vals: list[int] = []

    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue

        dep = edge.get("depart_attractiveness")
        arr = edge.get("arrive_attractiveness")
        if dep is None or arr is None:
            raise ValidationError(
                f"Edge {edge.get('id')} missing attractiveness attributes")

        dep = dep.strip("[]")
        arr = arr.strip("[]")
        try:
            dep_i = int(dep)
            arr_i = int(arr)
        except ValueError:
            raise ValidationError(
                f"Edge {edge.get('id')} has non‑integer attractiveness value")
        if dep_i < 0 or arr_i < 0:
            raise ValidationError(
                f"Edge {edge.get('id')} has negative attractiveness value")

        dep_vals.append(dep_i)
        arr_vals.append(arr_i)

    # sample‑mean sanity ------------------------------------------------------
    dep_mean = _mean(dep_vals)
    arr_mean = _mean(arr_vals)

    if not (lambda_depart_f * (1 - tolerance) <= dep_mean <= lambda_depart_f * (1 + tolerance)):
        raise ValidationError(
            f"Depart attractiveness mean {dep_mean:.2f} outside ±{tolerance*100:.0f}% of λ={lambda_depart_f}")
    if not (lambda_arrive_f * (1 - tolerance) <= arr_mean <= lambda_arrive_f * (1 + tolerance)):
        raise ValidationError(
            f"Arrive attractiveness mean {arr_mean:.2f} outside ±{tolerance*100:.0f}% of λ={lambda_arrive_f}")

    # variability check -------------------------------------------------------
    if len(set(dep_vals)) <= 1 and len(set(arr_vals)) <= 1:
        raise ValidationError(
            "Attractiveness values appear constant; distribution sampling may have failed")

    return


def verify_inject_traffic_lights(net_file: str | Path) -> None:
    path = Path(net_file)
    if not path.exists():
        raise ValidationError(f"TL validation: network file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # --- collect traffic-light objects ---------------------------------------
    tl_elems = {tl.get("id"): tl for tl in root.findall("tlLogic")}
    if not tl_elems:
        raise ValidationError(
            "No <tlLogic> elements found after inject_traffic_lights")

    # 1) each junction that *should* have a traffic light actually has one
    junction_tls = [j.get("id") for j in root.findall("junction")
                    if j.get("type") == "traffic_light"]
    missing = [jid for jid in junction_tls if jid not in tl_elems]
    if missing:
        raise ValidationError(
            f"Junction(s) missing tlLogic: {', '.join(missing)}")

    # 2) every connection’s tl attribute refers to a real tlLogic
    conn_bad: list[str] = []
    conn_map: dict[str, list[ET.Element]] = {}      # tlID -> connections
    for conn in root.findall("connection"):
        tl = conn.get("tl")
        if tl is None:
            continue
        if tl not in tl_elems:
            conn_bad.append(tl)
        conn_map.setdefault(tl, []).append(conn)

    if conn_bad:
        uniq = sorted(set(conn_bad))
        raise ValidationError(
            f"Connections reference non-existent tlLogic: {', '.join(uniq)}")

    # 3) phase length must equal controlled connections
    for tl_id, tl_elem in tl_elems.items():
        phases = tl_elem.findall("phase")
        if not phases:
            raise ValidationError(f"tlLogic '{tl_id}' has no <phase> elements")

        conn_count = len(conn_map.get(tl_id, []))
        # Some SUMO builds include internal lanes in <phase state> – that is fine
        for ph in phases:
            state = ph.get("state", "")
            if len(state) != conn_count:
                raise ValidationError(
                    f"tlLogic '{tl_id}' phase length {len(state)} "
                    f"≠ number of controlled connections {conn_count}"
                )

    # 4) duplicate IDs already impossible in XML; still, sanity:
    if len(tl_elems) != len(set(tl_elems)):
        raise ValidationError("Duplicate <tlLogic> IDs detected")

    # all good
    return
