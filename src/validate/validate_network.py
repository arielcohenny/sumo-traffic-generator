"""
Network validation functions for the SUMO traffic generator.

This module provides validation functions for each step of the network generation pipeline,
ensuring that network files are properly structured and contain expected data.
"""

import sumolib
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree.ElementTree import Element

from src.config import CONFIG
from src.validate.errors import ValidationError


__all__ = [
    "verify_generate_grid_network",
    "verify_extract_zones_from_junctions", 
    "verify_insert_split_edges",
    "verify_rebuild_network",
    "verify_set_lane_counts",
    "verify_assign_edge_attractiveness",
    "verify_generate_sumo_conf_file",
]


def verify_generate_grid_network(
    seed: int,
    dimension: int,
    block_size_m: int,
    junctions_to_remove_input: str,
    lane_count_arg: str,
) -> None:
    """Validate grid network generation with junction removal.
    
    This function validates the generate_grid_network function by checking:
    1. All required XML files exist and are valid
    2. Junction coordinates follow expected grid pattern
    3. Edge structure is consistent with grid topology
    4. Traffic light logic is properly configured
    5. Junction removal was applied correctly
    """
    
    # Required files that should exist after grid generation
    required_files = [
        CONFIG.network_nod_file,
        CONFIG.network_edg_file,
        CONFIG.network_con_file,
        CONFIG.network_tll_file,
    ]
    
    # 1 ── check file existence ------------------------------------------------
    for file_path in required_files:
        if not Path(file_path).exists():
            raise ValidationError(f"Required file missing: {file_path}")
    
    # 2 ── parse and validate XML structure ------------------------------------
    try:
        nod_tree = ET.parse(CONFIG.network_nod_file)
        nod_root = nod_tree.getroot()
        
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        
        con_tree = ET.parse(CONFIG.network_con_file)
        con_root = con_tree.getroot()
        
        tll_tree = ET.parse(CONFIG.network_tll_file)
        tll_root = tll_tree.getroot()
        
    except ET.ParseError as e:
        raise ValidationError(f"XML parsing error: {e}")
    
    # 3 ── validate junction structure -----------------------------------------
    grid_nodes = []
    h_nodes = []
    node_coords = {}
    
    for node in nod_root.findall("node"):
        node_id = node.get("id")
        x = float(node.get("x"))
        y = float(node.get("y"))
        node_coords[node_id] = (x, y)
        
        if "_H_node" in node_id:
            h_nodes.append(node_id)
        else:
            grid_nodes.append(node_id)
    
    # Check that we have grid nodes (some may be removed)
    if not grid_nodes:
        raise ValidationError("No grid nodes found in network")
    
    # 4 ── validate edge structure ---------------------------------------------
    edge_count = len(edg_root.findall("edge"))
    if edge_count == 0:
        raise ValidationError("No edges found in network")
    
    # At this stage (before edge splitting), there should be no head edges
    head_edges = 0
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id.endswith("_H"):
            head_edges += 1
    
    # Before edge splitting, there should be no head edges
    if head_edges > 0:
        raise ValidationError(f"Found {head_edges} head edges before edge splitting step")
    
    # 5 ── validate traffic light configuration --------------------------------
    tl_logics = {}
    for tl_logic in tll_root.findall("tlLogic"):
        tl_id = tl_logic.get("id")
        phases = tl_logic.findall("phase")
        if not phases:
            raise ValidationError(f"Traffic light {tl_id} has no phases")
        tl_logics[tl_id] = len(phases)
    
    # Check that traffic lights exist for grid nodes
    grid_tl_nodes = [node.get("id") for node in nod_root.findall("node") 
                     if node.get("type") == "traffic_light"]
    
    if not grid_tl_nodes:
        raise ValidationError("No traffic light nodes found")
    
    # 6 ── validate bounding box from node coordinates ------------------------
    if node_coords:
        xs = [coord[0] for coord in node_coords.values()]
        ys = [coord[1] for coord in node_coords.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        tol = 1e-3
        max_coord = (dimension - 1) * block_size_m + tol
        
        if xmax > max_coord or ymax > max_coord:
            raise ValidationError(f"Bounding box exceeds expected: max=({xmax}, {ymax}), expected≤{max_coord}")
    
    # 7 ── validate junction removal -------------------------------------------
    if junctions_to_remove_input.strip():
        # Parse junction removal input
        try:
            num_to_remove = int(junctions_to_remove_input)
            # Random removal - we can't predict which specific junctions were removed
            # Just verify that some junctions are missing from the full grid
            expected_total = dimension * dimension
            if len(grid_nodes) >= expected_total:
                raise ValidationError(f"Expected junction removal but found {len(grid_nodes)} >= {expected_total}")
        except ValueError:
            # Specific junction IDs provided
            junctions_to_remove = [j.strip() for j in junctions_to_remove_input.split(",")]
            for junction_id in junctions_to_remove:
                if junction_id in [n.get("id") for n in nod_root.findall("node")]:
                    raise ValidationError(f"Junction {junction_id} should have been removed but still exists")
    
    # 8 ── validate edge connectivity ------------------------------------------
    edge_from_to = {}
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        edge_from_to[edge_id] = (from_node, to_node)
    
    # Check that all referenced nodes exist
    all_node_ids = set(node_coords.keys())
    for edge_id, (from_node, to_node) in edge_from_to.items():
        if from_node not in all_node_ids:
            raise ValidationError(f"Edge {edge_id} references non-existent from node: {from_node}")
        if to_node not in all_node_ids:
            raise ValidationError(f"Edge {edge_id} references non-existent to node: {to_node}")
    
    # 9 ── validate connections consistency ------------------------------------
    connection_count = len(con_root.findall("connection"))
    if connection_count == 0:
        raise ValidationError("No connections found in network")
    
    # Check that connections reference existing edges
    edge_ids = set(edge.get("id") for edge in edg_root.findall("edge"))
    for conn in con_root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        if from_edge not in edge_ids:
            raise ValidationError(f"Connection references non-existent from edge: {from_edge}")
        if to_edge not in edge_ids:
            raise ValidationError(f"Connection references non-existent to edge: {to_edge}")
    
    # 10 ── validate lane assignments -----------------------------------------
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        num_lanes = int(edge.get("numLanes", 1))
        
        # Check numLanes is within reasonable bounds
        if num_lanes < 1 or num_lanes > 5:
            raise ValidationError(f"Edge {edge_id} has unreasonable lane count: {num_lanes}")
        
        # Check that lanes exist as child elements or numLanes attribute
        lane_elements = edge.findall("lane")
        if lane_elements:
            if len(lane_elements) != num_lanes:
                raise ValidationError(f"Edge {edge_id} numLanes={num_lanes} but has {len(lane_elements)} lane elements")
    
    # 11 ── validate lane indexing in connections -----------------------------
    for conn in con_root.findall("connection"):
        from_lane = conn.get("fromLane")
        to_lane = conn.get("toLane")
        
        if from_lane is not None:
            try:
                from_lane_idx = int(from_lane)
                if from_lane_idx < 0:
                    raise ValidationError(f"Connection has negative fromLane: {from_lane}")
            except ValueError:
                raise ValidationError(f"Connection has invalid fromLane: {from_lane}")
        
        if to_lane is not None:
            try:
                to_lane_idx = int(to_lane)
                if to_lane_idx < 0:
                    raise ValidationError(f"Connection has negative toLane: {to_lane}")
            except ValueError:
                raise ValidationError(f"Connection has invalid toLane: {to_lane}")
    
    # 12 ── validate file sizes are reasonable ---------------------------------
    for file_path in required_files:
        file_size = Path(file_path).stat().st_size
        if file_size < 100:  # Minimum reasonable size for XML files
            raise ValidationError(f"File {file_path} too small ({file_size} bytes), may be empty or corrupt")
    
    # 13 ── validate no internal nodes remain after generation ----------------
    internal_nodes = [node.get("id") for node in nod_root.findall("node") 
                      if node.get("id").startswith(":")]
    if internal_nodes:
        raise ValidationError(f"Internal nodes found after generation: {', '.join(internal_nodes[:5])}"
                             + (f" ... and {len(internal_nodes)-5} more" if len(internal_nodes) > 5 else ""))
    
    # 14 ── validate traffic light phase timing --------------------------------
    for tl_logic in tll_root.findall("tlLogic"):
        tl_id = tl_logic.get("id")
        total_cycle_time = 0
        green_time = 0
        
        for phase in tl_logic.findall("phase"):
            try:
                duration = float(phase.get("duration", 0))
                state = phase.get("state", "")
                
                total_cycle_time += duration
                if "G" in state or "g" in state:  # Green phases
                    green_time += duration
                
                # Check reasonable phase duration (1-120 seconds)
                if duration < 1 or duration > 120:
                    raise ValidationError(f"Traffic light {tl_id} has unreasonable phase duration: {duration}s")
            except ValueError:
                raise ValidationError(f"Traffic light {tl_id} has invalid phase duration")
        
        # Check reasonable cycle time (10-300 seconds)
        if total_cycle_time < 10 or total_cycle_time > 300:
            raise ValidationError(f"Traffic light {tl_id} has unreasonable cycle time: {total_cycle_time}s")
        
        # Check minimum green time (at least 20% of cycle)
        if green_time < 0.2 * total_cycle_time:
            raise ValidationError(f"Traffic light {tl_id} has insufficient green time: {green_time}/{total_cycle_time}s")

    return


def verify_insert_split_edges() -> None:
    """Validate edge splitting results.
    
    This function validates the insert_split_edges function by checking:
    1. All original edges are split into body + head segments
    2. H_nodes are created at correct positions
    3. Body edges connect to H_nodes, head edges connect from H_nodes
    4. Geometric consistency is maintained
    """
    
    # Required files should exist after edge splitting
    required_files = [
        CONFIG.network_nod_file,
        CONFIG.network_edg_file,
        CONFIG.network_con_file,
    ]
    
    # 1 ── check file existence ------------------------------------------------
    for file_path in required_files:
        if not Path(file_path).exists():
            raise ValidationError(f"Required file missing: {file_path}")
    
    # 2 ── parse XML files -----------------------------------------------------
    try:
        nod_tree = ET.parse(CONFIG.network_nod_file)
        nod_root = nod_tree.getroot()
        
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        
        con_tree = ET.parse(CONFIG.network_con_file)
        con_root = con_tree.getroot()
        
    except ET.ParseError as e:
        raise ValidationError(f"XML parsing error: {e}")
    
    # 3 ── classify nodes and edges --------------------------------------------
    grid_nodes = []
    h_nodes = []
    node_coords = {}
    
    for node in nod_root.findall("node"):
        node_id = node.get("id")
        x = float(node.get("x"))
        y = float(node.get("y"))
        node_coords[node_id] = (x, y)
        
        if "_H_node" in node_id:
            h_nodes.append(node_id)
        else:
            grid_nodes.append(node_id)
    
    body_edges = {}
    head_edges = {}
    
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        
        if edge_id.endswith("_H"):
            head_edges[edge_id] = (from_node, to_node)
        else:
            body_edges[edge_id] = (from_node, to_node)
    
    # 4 ── validate edge splitting consistency ---------------------------------
    if len(body_edges) != len(head_edges):
        raise ValidationError(f"Body/head edge count mismatch: {len(body_edges)} body, {len(head_edges)} head")
    
    # Check body-head edge pairing
    for body_id, (body_from, body_to) in body_edges.items():
        head_id = body_id + "_H"
        if head_id not in head_edges:
            raise ValidationError(f"Body edge {body_id} missing corresponding head edge {head_id}")
        
        head_from, head_to = head_edges[head_id]
        
        # Body edge should connect to H_node, head edge should connect from H_node
        if not body_to.endswith("_H_node"):
            raise ValidationError(f"Body edge {body_id} should connect to H_node, but connects to {body_to}")
        
        if head_from != body_to:
            raise ValidationError(f"Head edge {head_id} should start from {body_to}, but starts from {head_from}")
    
    # 5 ── validate H_node creation --------------------------------------------
    expected_h_nodes = len(body_edges)
    if len(h_nodes) != expected_h_nodes:
        raise ValidationError(f"H_node count mismatch: expected {expected_h_nodes}, found {len(h_nodes)}")
    
    # 6 ── validate geometric consistency --------------------------------------
    for body_id, (body_from, body_to) in body_edges.items():
        head_id = body_id + "_H"
        if head_id not in head_edges:
            continue
        
        head_from, head_to = head_edges[head_id]
        
        # Get coordinates
        body_from_coord = node_coords[body_from]
        h_node_coord = node_coords[body_to]  # body_to is the H_node
        head_to_coord = node_coords[head_to]
        
        # Check that H_node is between body_from and head_to
        # Using simple distance check: distance(from, h_node) + distance(h_node, to) ≈ distance(from, to)
        def distance(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        
        d_direct = distance(body_from_coord, head_to_coord)
        d_via_h = distance(body_from_coord, h_node_coord) + distance(h_node_coord, head_to_coord)
        
        # Allow small tolerance for floating point errors
        if abs(d_direct - d_via_h) > 1e-6:
            raise ValidationError(f"H_node {body_to} not on line between {body_from} and {head_to}")
    
    # 7 ── validate lane count consistency -------------------------------------
    for body_id in body_edges:
        head_id = body_id + "_H"
        if head_id not in head_edges:
            continue
        
        # Find corresponding edge elements
        body_edge = None
        head_edge = None
        for edge in edg_root.findall("edge"):
            if edge.get("id") == body_id:
                body_edge = edge
            elif edge.get("id") == head_id:
                head_edge = edge
        
        if body_edge is None or head_edge is None:
            raise ValidationError(f"Could not find edge elements for {body_id}/{head_id}")
        
        # Check lane count consistency
        body_lanes_attr = body_edge.get("numLanes")
        head_lanes_attr = head_edge.get("numLanes")
        
        if body_lanes_attr is not None and head_lanes_attr is not None:
            if body_lanes_attr != head_lanes_attr:
                raise ValidationError(f"Lane count mismatch between {body_id} ({body_lanes_attr}) and {head_id} ({head_lanes_attr})")
        
        # Check lane elements if present
        body_lane_elements = body_edge.findall("lane")
        head_lane_elements = head_edge.findall("lane")
        
        if body_lane_elements and head_lane_elements:
            if len(body_lane_elements) != len(head_lane_elements):
                raise ValidationError(f"Lane element count mismatch between {body_id} and {head_id}")
    
    # 8 ── validate connection updates ----------------------------------------
    # Check that connections are properly updated for split edges
    connection_edges = set()
    for conn in con_root.findall("connection"):
        from_edge = conn.get("from")
        to_edge = conn.get("to")
        connection_edges.add(from_edge)
        connection_edges.add(to_edge)
    
    # All edges referenced in connections should exist
    all_edge_ids = set(body_edges.keys()) | set(head_edges.keys())
    for edge_id in connection_edges:
        if edge_id not in all_edge_ids:
            raise ValidationError(f"Connection references non-existent edge: {edge_id}")
    
    # 9 ── validate edge naming pattern ---------------------------------------
    for body_id in body_edges:
        if not body_id.replace("_", "").isalnum():
            raise ValidationError(f"Body edge has invalid naming pattern: {body_id}")
        
        # Check that head edge follows naming convention
        expected_head_id = body_id + "_H"
        if expected_head_id not in head_edges:
            raise ValidationError(f"Head edge naming doesn't follow convention: expected {expected_head_id}")
    
    # 10 ── validate H_node naming pattern ------------------------------------
    for h_node_id in h_nodes:
        if not h_node_id.endswith("_H_node"):
            raise ValidationError(f"H_node naming doesn't follow convention: {h_node_id}")
    
    # 11 ── validate no orphaned nodes ----------------------------------------
    # All nodes should be referenced by at least one edge
    referenced_nodes = set()
    for edge in edg_root.findall("edge"):
        referenced_nodes.add(edge.get("from"))
        referenced_nodes.add(edge.get("to"))
    
    all_node_ids = set(node_coords.keys())
    orphaned_nodes = all_node_ids - referenced_nodes
    if orphaned_nodes:
        raise ValidationError(f"Found orphaned nodes: {', '.join(list(orphaned_nodes)[:5])}")
    
    # 12 ── validate edge lengths ---------------------------------------------
    # After splitting, body edges should be longer than head edges
    for body_id in body_edges:
        head_id = body_id + "_H"
        if head_id not in head_edges:
            continue
        
        body_from, body_to = body_edges[body_id]
        head_from, head_to = head_edges[head_id]
        
        body_length = distance(node_coords[body_from], node_coords[body_to])
        head_length = distance(node_coords[head_from], node_coords[head_to])
        
        # Head segment should be CONFIG.HEAD_DISTANCE (30m by default)
        expected_head_length = CONFIG.HEAD_DISTANCE
        if abs(head_length - expected_head_length) > 1e-3:
            raise ValidationError(f"Head edge {head_id} has incorrect length: {head_length:.3f}m, expected {expected_head_length}m")
        
        # Body segment should be original_length - head_length
        # Original edges are block_size_m long, so body should be block_size_m - head_distance
        expected_body_length = 150 - CONFIG.HEAD_DISTANCE  # 150m is original edge length
        if abs(body_length - expected_body_length) > 1e-3:
            raise ValidationError(f"Body edge {body_id} has incorrect length: {body_length:.3f}m, expected {expected_body_length}m")

    return


def verify_extract_zones_from_junctions(
    cell_size: Optional[float],
    seed: int,
    fill_polygons: bool,
    inset: float,
) -> None:
    """Validate zone extraction from junctions.
    
    This function validates the extract_zones_from_junctions function by checking:
    1. Zone files are generated correctly
    2. Zone count matches expected (n-1)×(n-1) for n×n grid
    3. Zone properties are valid
    4. Land use assignment is reasonable
    """
    
    # Zone files that should exist  
    zone_files = [
        CONFIG.zones_file,
    ]
    
    # 1 ── check file existence ------------------------------------------------
    for file_path in zone_files:
        if not Path(file_path).exists():
            raise ValidationError(f"Zone file missing: {file_path}")
    
    # 2 ── validate SUMO polygon file ------------------------------------------
    try:
        tree = ET.parse(CONFIG.zones_file)
        root = tree.getroot()
        
        polygons = root.findall("poly")
        if not polygons:
            raise ValidationError("SUMO polygon file has no polygons")
        
        # Validate polygon structure
        for poly in polygons:
            poly_id = poly.get("id")
            if not poly_id:
                raise ValidationError("Polygon missing ID")
            
            shape = poly.get("shape")
            if not shape:
                raise ValidationError(f"Polygon {poly_id} missing shape")
            
            # Validate shape format (should be x1,y1 x2,y2 x3,y3 x4,y4)
            shape_coords = shape.split()
            if len(shape_coords) < 3:
                raise ValidationError(f"Polygon {poly_id} has too few coordinates")
            
            for coord in shape_coords:
                try:
                    x, y = coord.split(",")
                    float(x)
                    float(y)
                except ValueError:
                    raise ValidationError(f"Polygon {poly_id} has invalid coordinate: {coord}")
    
    except ET.ParseError as e:
        raise ValidationError(f"Error parsing SUMO polygon file: {e}")
    
    # 4 ── validate zone count -------------------------------------------------
    # For n×n grid, we should have (n-1)×(n-1) zones
    # Need to determine grid size from network nodes
    try:
        nod_tree = ET.parse(CONFIG.network_nod_file)
        nod_root = nod_tree.getroot()
        
        # Count grid nodes (excluding H_nodes)
        grid_coords = []
        for node in nod_root.findall("node"):
            node_id = node.get("id")
            if "_H_node" not in node_id:
                x = float(node.get("x"))
                y = float(node.get("y"))
                grid_coords.append((x, y))
        
        # Determine grid dimensions
        xs = [coord[0] for coord in grid_coords]
        ys = [coord[1] for coord in grid_coords]
        unique_xs = sorted(set(xs))
        unique_ys = sorted(set(ys))
        
        grid_x_size = len(unique_xs)
        grid_y_size = len(unique_ys)
        
        expected_zones = (grid_x_size - 1) * (grid_y_size - 1)
        actual_zones = len(polygons)
        
        if actual_zones != expected_zones:
            raise ValidationError(f"Zone count mismatch: expected {expected_zones}, found {actual_zones}")
    
    except Exception as e:
        raise ValidationError(f"Error determining grid size: {e}")

    return


def verify_rebuild_network() -> None:
    """Validate network rebuild from separate XML files.
    
    This function validates that the network rebuild process successfully
    creates a valid compiled network file from separate XML components.
    """
    
    # 1 ── check compiled network file exists ----------------------------------
    if not Path(CONFIG.network_file).exists():
        raise ValidationError(f"Compiled network file missing: {CONFIG.network_file}")
    
    # 2 ── validate XML structure ----------------------------------------------
    try:
        tree = ET.parse(CONFIG.network_file)
        root = tree.getroot()
        
        if root.tag != "net":
            raise ValidationError("Network file root element is not 'net'")
        
    except ET.ParseError as e:
        raise ValidationError(f"Network file XML parsing error: {e}")
    
    # 3 ── validate SUMO can load the network ---------------------------------
    try:
        net = sumolib.net.readNet(CONFIG.network_file)
    except Exception as e:
        raise ValidationError(f"SUMO cannot load network file: {e}")
    
    # 4 ── validate network has reasonable content -----------------------------
    edges = net.getEdges()
    if not edges:
        raise ValidationError("Network has no edges")
    
    nodes = net.getNodes()
    if not nodes:
        raise ValidationError("Network has no nodes")
    
    # 5 ── validate traffic light logic is preserved -------------------------
    # Check that traffic lights exist in the compiled network
    tl_nodes = [node for node in nodes if node.getType() == "traffic_light"]
    if not tl_nodes:
        raise ValidationError("No traffic light nodes found in compiled network")

    return


def verify_set_lane_counts(
    *,
    min_lanes: int = 1,
    max_lanes: int = 3,
    algorithm: str = 'realistic',
) -> None:
    """Validate lane counts and distribution after lane assignment."""

    # Work with separate XML files since network not rebuilt yet
    edg_file = CONFIG.network_edg_file
    if not Path(edg_file).exists():
        raise ValidationError(f"Edges file missing: {edg_file}")

    try:
        tree = ET.parse(edg_file)
        root = tree.getroot()
    except ET.ParseError as exc:
        raise ValidationError(f"Failed to parse edges file: {exc}") from exc

    counts: list[int] = []
    
    # Extract lane counts from edges
    for edge in root.findall("edge"):
        # Get numLanes attribute
        num_lanes_str = edge.get("numLanes")
        if num_lanes_str is None:
            raise ValidationError(f"Edge {edge.get('id')} missing numLanes attribute")
        
        try:
            num_lanes = int(num_lanes_str)
            counts.append(num_lanes)
        except ValueError:
            raise ValidationError(f"Invalid numLanes value for edge {edge.get('id')}: {num_lanes_str}")

    if not counts:
        raise ValidationError("No edges found in edges file")

    # Bound check ----------------------------------------------------------------
    bad = [c for c in counts if c < min_lanes or c > max_lanes]
    if bad:
        raise ValidationError(
            f"{len(bad)} edges have lane count outside [{min_lanes}, {max_lanes}]"
        )

    # Distribution check ---------------------------------------------------------
    if all(c == counts[0] for c in counts):
        # Only raise error for random algorithm - fixed counts and realistic are allowed to be uniform
        if algorithm == 'random':
            raise ValidationError(
                "All edges share the same lane number; randomisation may have failed")
        elif algorithm.isdigit():
            # Fixed count algorithm - uniform distribution is expected
            expected_count = int(algorithm)
            if counts[0] != expected_count:
                raise ValidationError(
                    f"Fixed algorithm expected {expected_count} lanes but found {counts[0]}")
        # For 'realistic' algorithm, uniform distribution is acceptable (though unusual)
    
    # Only check min/max distribution for random algorithm
    if algorithm == 'random':
        if min_lanes not in counts:
            raise ValidationError(
                f"No edge ended up with the minimum lane count ({min_lanes})")
        if max_lanes not in counts:
            raise ValidationError(
                f"No edge ended up with the maximum lane count ({max_lanes})")

    return


# ---------------------------------------------------------------------------
#  Edge attractiveness verification (inline after assign_edge_attractiveness)
# ---------------------------------------------------------------------------


def _to_float(val):
    """Coerce CLI values that sometimes arrive as tuple / str → float."""
    if isinstance(val, (list, tuple)):
        val = val[0]
    return float(val)


def verify_assign_edge_attractiveness(
    seed: int,
    tolerance: float = 0.5,  # ±50 % of λ is acceptable
) -> None:
    """Validate that attractiveness attributes exist and are plausible.
    
    This function validates the assign_edge_attractiveness function by checking:
    1. All edges have depart_attractiveness and arrive_attractiveness attributes
    2. Values are non-negative integers from Poisson distribution
    3. Sample means are within tolerance of expected lambda values
    4. Values show proper variability (not constant)
    """
    
    # Work with rebuilt network file since attractiveness is added to compiled network
    net_file = CONFIG.network_file
    if not Path(net_file).exists():
        raise ValidationError(f"Network file missing: {net_file}")
    
    # Parse the XML directly to check for attractiveness attributes
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValidationError(f"Failed to parse network XML: {e}")
    
    depart_attrs = []
    arrive_attrs = []
    
    # Find all edge elements and check for attractiveness attributes
    for edge_elem in root.findall(".//edge"):
        edge_id = edge_elem.get("id")
        
        # Skip internal edges (start with ":")
        if edge_id and edge_id.startswith(":"):
            continue
        
        # Check for attractiveness attributes
        depart_attr = edge_elem.get("depart_attractiveness")
        arrive_attr = edge_elem.get("arrive_attractiveness")
        
        if depart_attr is None:
            raise ValidationError(f"Edge {edge_id} missing depart_attractiveness")
        if arrive_attr is None:
            raise ValidationError(f"Edge {edge_id} missing arrive_attractiveness")
        
        # Parse values (may be wrapped in brackets)
        try:
            depart_val = _to_float(depart_attr.strip("[]"))
            arrive_val = _to_float(arrive_attr.strip("[]"))
        except ValueError:
            raise ValidationError(f"Edge {edge_id} has invalid attractiveness values")
        
        # Check non-negative
        if depart_val < 0:
            raise ValidationError(f"Edge {edge_id} has negative depart_attractiveness: {depart_val}")
        if arrive_val < 0:
            raise ValidationError(f"Edge {edge_id} has negative arrive_attractiveness: {arrive_val}")
        
        depart_attrs.append(depart_val)
        arrive_attrs.append(arrive_val)
    
    if not depart_attrs:
        raise ValidationError("No edges with attractiveness attributes found")
    
    # Statistical validation
    depart_mean = statistics.mean(depart_attrs)
    arrive_mean = statistics.mean(arrive_attrs)
    
    expected_depart = CONFIG.LAMBDA_DEPART
    expected_arrive = CONFIG.LAMBDA_ARRIVE
    
    # Check means are within tolerance
    if abs(depart_mean - expected_depart) > tolerance * expected_depart:
        raise ValidationError(
            f"Depart attractiveness mean {depart_mean:.2f} outside tolerance of {expected_depart:.2f}")
    
    if abs(arrive_mean - expected_arrive) > tolerance * expected_arrive:
        raise ValidationError(
            f"Arrive attractiveness mean {arrive_mean:.2f} outside tolerance of {expected_arrive:.2f}")
    
    # Check variability
    if len(set(depart_attrs)) < 2:
        raise ValidationError("Depart attractiveness values lack variability")
    if len(set(arrive_attrs)) < 2:
        raise ValidationError("Arrive attractiveness values lack variability")

    return


# ---------------------------------------------------------------------------
#  SUMO configuration file verification
# ---------------------------------------------------------------------------


def verify_generate_sumo_conf_file() -> None:
    """Validate SUMO configuration file generation.
    
    This function validates the generate_sumo_conf_file function by checking:
    1. Configuration file exists and is valid XML
    2. All required sections are present
    3. File references are correct
    4. Time parameters are reasonable
    """
    
    # 1 ── check configuration file exists -------------------------------------
    if not Path(CONFIG.config_file).exists():
        raise ValidationError(f"SUMO configuration file missing: {CONFIG.config_file}")
    
    # 2 ── validate XML structure ----------------------------------------------
    try:
        tree = ET.parse(CONFIG.config_file)
        root = tree.getroot()
        
        if root.tag != "configuration":
            raise ValidationError("SUMO config root element is not 'configuration'")
        
    except ET.ParseError as e:
        raise ValidationError(f"SUMO config XML parsing error: {e}")
    
    # 3 ── validate required sections ------------------------------------------
    required_sections = ["input", "time"]
    for section in required_sections:
        if root.find(section) is None:
            raise ValidationError(f"SUMO config missing required section: {section}")
    
    # 4 ── validate input file references --------------------------------------
    input_section = root.find("input")
    
    # Check network file reference
    net_file_elem = input_section.find("net-file")
    if net_file_elem is None:
        raise ValidationError("SUMO config missing net-file reference")
    
    net_file_value = net_file_elem.get("value")
    if not net_file_value:
        raise ValidationError("SUMO config net-file has no value")
    
    # Check routes file reference  
    route_files_elem = input_section.find("route-files")
    if route_files_elem is None:
        raise ValidationError("SUMO config missing route-files reference")
    
    route_files_value = route_files_elem.get("value")
    if not route_files_value:
        raise ValidationError("SUMO config route-files has no value")
    
    # 5 ── validate time parameters --------------------------------------------
    time_section = root.find("time")
    
    # Check for time parameters
    begin_elem = time_section.find("begin")
    end_elem = time_section.find("end")
    step_length_elem = time_section.find("step-length")
    
    if begin_elem is not None:
        try:
            begin_val = float(begin_elem.get("value", 0))
            if begin_val < 0:
                raise ValidationError(f"SUMO config begin time is negative: {begin_val}")
        except ValueError:
            raise ValidationError("SUMO config begin time is not a valid number")
    
    if end_elem is not None:
        try:
            end_val = float(end_elem.get("value", 0))
            if end_val <= 0:
                raise ValidationError(f"SUMO config end time is not positive: {end_val}")
        except ValueError:
            raise ValidationError("SUMO config end time is not a valid number")
    
    if step_length_elem is not None:
        try:
            step_val = float(step_length_elem.get("value", 1))
            if step_val <= 0:
                raise ValidationError(f"SUMO config step length is not positive: {step_val}")
        except ValueError:
            raise ValidationError("SUMO config step length is not a valid number")
    
    # 6 ── validate no template placeholders remain ----------------------------
    # Read file content to check for any remaining placeholders
    with open(CONFIG.config_file, 'r') as f:
        content = f.read()
    
    placeholders = ["#IFNET#", "#IFROUTE#", "#IFADDITIONAL#"]
    for placeholder in placeholders:
        if placeholder in content:
            raise ValidationError(f"SUMO config contains unresolved placeholder: {placeholder}")
    
    # 7 ── validate file references exist --------------------------------------
    # Check that referenced files actually exist (resolve relative to data directory)
    config_dir = Path(CONFIG.config_file).parent
    
    net_file_path = config_dir / net_file_value
    if not net_file_path.exists():
        raise ValidationError(f"SUMO config references non-existent network file: {net_file_value} (resolved to {net_file_path})")
    
    route_file_path = config_dir / route_files_value
    if not route_file_path.exists():
        raise ValidationError(f"SUMO config references non-existent route file: {route_files_value} (resolved to {route_file_path})")
    
    # 8 ── validate configuration is loadable by SUMO -------------------------
    # This is a basic check - we could extend it by actually trying to load the config
    # but that would require SUMO to be installed and configured
    
    return


# ---------------------------------------------------------------------------
#  Vehicle route generation verification
# ---------------------------------------------------------------------------


def verify_generate_vehicle_routes(
    num_vehicles: int,
    tolerance: float = 0.02,  # 2% shortfall tolerance
) -> None:
    """Validate vehicle route generation.
    
    This function validates the generate_vehicle_routes function by checking:
    1. Routes XML file exists and is valid
    2. Vehicle count is within tolerance
    3. Route structure is correct
    4. Departure times are reasonable
    """
    
    # 1 ── check routes file exists --------------------------------------------
    if not Path(CONFIG.routes_file).exists():
        raise ValidationError(f"Routes file missing: {CONFIG.routes_file}")
    
    # 2 ── validate XML structure ----------------------------------------------
    try:
        tree = ET.parse(CONFIG.routes_file)
        root = tree.getroot()
        
        if root.tag != "routes":
            raise ValidationError("Routes file root element is not 'routes'")
        
    except ET.ParseError as e:
        raise ValidationError(f"Routes file XML parsing error: {e}")
    
    # 3 ── validate vehicle types ----------------------------------------------
    vtypes = root.findall("vType")
    if not vtypes:
        raise ValidationError("Routes file has no vehicle types")
    
    # Check that vehicle types have required attributes
    for vtype in vtypes:
        vtype_id = vtype.get("id")
        if not vtype_id:
            raise ValidationError("Vehicle type missing ID")
        
        # Check for basic attributes
        if not vtype.get("vClass"):
            raise ValidationError(f"Vehicle type {vtype_id} missing vClass")
    
    # 4 ── validate vehicles and routes ----------------------------------------
    vehicles = root.findall("vehicle")
    
    # Check vehicle count
    actual_vehicles = len(vehicles)
    shortfall = num_vehicles - actual_vehicles
    
    if shortfall > tolerance * num_vehicles:
        raise ValidationError(
            f"Too few vehicles generated: {actual_vehicles}/{num_vehicles} "
            f"(shortfall {shortfall/num_vehicles:.1%} > {tolerance:.1%})")
    
    if not vehicles:
        raise ValidationError("Routes file has no vehicles")
    
    # 5 ── validate route connectivity -----------------------------------------
    # Load network to validate routes
    try:
        net = sumolib.net.readNet(CONFIG.network_file)
    except Exception as e:
        raise ValidationError(f"Failed to load network for route validation: {e}")
    
    all_edge_ids = set(edge.getID() for edge in net.getEdges())
    
    # Check each vehicle's route
    for i, vehicle in enumerate(vehicles):
        vehicle_id = vehicle.get("id")
        route_elem = vehicle.find("route")
        
        if route_elem is None:
            raise ValidationError(f"Vehicle {vehicle_id} has no route")
        
        edges = route_elem.get("edges", "").split()
        if not edges:
            raise ValidationError(f"Vehicle {vehicle_id} has empty route")
        
        # Check that all edges exist in network
        for edge_id in edges:
            if edge_id not in all_edge_ids:
                raise ValidationError(f"Vehicle {vehicle_id} route references non-existent edge: {edge_id}")
    
    # 6 ── validate departure times --------------------------------------------
    departure_times = []
    for vehicle in vehicles:
        depart_str = vehicle.get("depart")
        if depart_str is None:
            raise ValidationError(f"Vehicle {vehicle.get('id')} missing departure time")
        
        try:
            depart_time = float(depart_str)
            departure_times.append(depart_time)
        except ValueError:
            raise ValidationError(f"Vehicle {vehicle.get('id')} has invalid departure time: {depart_str}")
    
    # Check departure time distribution
    if departure_times:
        min_depart = min(departure_times)
        max_depart = max(departure_times)
        
        if min_depart < 0:
            raise ValidationError(f"Negative departure time found: {min_depart}")
        
        # Check for reasonable spread
        if max_depart - min_depart < 1:
            raise ValidationError("Departure times lack proper spread")
    
    # 7 ── validate edge filtering ---------------------------------------------
    # Check that internal edges are not used in routes
    for vehicle in vehicles:
        route_elem = vehicle.find("route")
        if route_elem is not None:
            edges = route_elem.get("edges", "").split()
            for edge_id in edges:
                if edge_id.startswith(":"):
                    raise ValidationError(f"Vehicle {vehicle.get('id')} route contains internal edge: {edge_id}")
    
    # 8 ── validate statistical properties ------------------------------------
    # Check that vehicle generation shows expected randomness
    if len(vehicles) >= 10:  # Only for sufficiently large samples
        # Check vehicle type distribution
        type_counts = {}
        for vehicle in vehicles:
            vtype = vehicle.get("type")
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        # Should have some diversity in vehicle types
        if len(type_counts) < 2 and len(vtypes) > 1:
            raise ValidationError("Vehicle type assignment lacks diversity")

    return