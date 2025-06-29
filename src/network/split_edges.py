# =============================================================================
# Edge‐Splitting Specification for SUMO Dumps
#
# This script transforms the plain‐dump files (.nod, .edg, .con, .tll)
# by splitting each original edge E into:
#   • a body segment (E)
#   • one head segment per movement (E_H_F, where F is the target edge)
# using Nimrod’s “_H_” naming convention.
#
# 1. .nod (nodes)
#    - For each original edge E with outgoing connections, add a divergence node:
#        id:        E_H_node
#        coordinates: X meters upstream from E’s to-node along E’s shape
#    - Preserve all original nodes unchanged.
#
# 2. .edg (edges)
#    - Remove every <edge id="E"> that appears as a “from" in .con.
#    - Add the body segment for E:
#        <edge id="E"
#              from="orig-u-node"
#              to="E_H_node"
#              shape="first X meters of E’s polyline"
#              (other attributes unchanged) />
#    - For each original connection E→F, add a head segment:
#        <edge id="E_H_F"
#              from="E_H_node"
#              to="orig-v-node"
#              shape="remaining segment of E’s polyline"
#              (lanes and attributes as appropriate) />
#
# 3. .con (connections)
#    - Remove all <connection from="E" to="F"> entries for split edges.
#    - For each original E→F, emit:
#        <connection from="E_H_F" to="F" (other attrs unchanged) />
#    - Leave all other connections intact.
#
# 4. .tll (traffic‐light logic)
#    - Within each <tlLogic> block, locate connections from E→F.
#    - Replace each with:
#        <connection from="E_H_F" to="F" (other attrs unchanged) />
#    - Preserve all other signal logic (phases, timings, etc.) unchanged.
#
# Naming Conventions Summary:
#    Divergence node:    E_H_node
#    Body edge:          E
#    Head edge:          E_H_F
#    Connections:        from="E_H_F" to="F"
# =============================================================================

import xml.etree.ElementTree as ET
from shapely.geometry import LineString, GeometryCollection
from shapely.ops import split as split_line
from typing import List, Tuple, Dict


def parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    """
    Parse a SUMO shape string into a list of (x, y) tuples.
    """
    coords: List[Tuple[float, float]] = []
    for pair in shape_str.strip().split():
        x_str, y_str = pair.split(',')
        coords.append((float(x_str), float(y_str)))
    return coords


def shape_to_str(coords: List[Tuple[float, float]]) -> str:
    """
    Convert a list of (x, y) tuples back into a SUMO shape string.
    """
    return " ".join(f"{x},{y}" for x, y in coords)


def split_shape(coords: List[Tuple[float, float]], split_dist: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Split a polyline at a fixed distance upstream from its end.
    Returns (body_coords, head_coords).
    """
    # Straight-line shortcut for two-point shapes
    if len(coords) <= 2:
        line = LineString(coords)
        total_length = line.length
        split_pos = total_length - split_dist
        if split_pos <= 0 or split_pos >= total_length:
            raise ValueError(
                f"split distance {split_dist} out of bounds for line length {total_length}")
        split_pt = line.interpolate(split_pos)
        xp, yp = float(split_pt.x), float(split_pt.y)
        return [coords[0], (xp, yp)], [(xp, yp), coords[-1]]

    # General case for multi-point shapes
    line = LineString(coords)
    total_length = line.length
    split_pos = total_length - split_dist
    if split_pos <= 0 or split_pos >= total_length:
        raise ValueError(
            f"split distance {split_dist} out of bounds for line length {total_length}")
    split_pt = line.interpolate(split_pos)

    parts = split_line(line, split_pt)
    # shapely 2.x returns a GeometryCollection, so extract geoms
    candidates = []
    if isinstance(parts, GeometryCollection):
        candidates = [g for g in parts.geoms if isinstance(g, LineString)]
    else:
        candidates = [g for g in parts if isinstance(g, LineString)]

    if len(candidates) < 2:
        raise RuntimeError("Failed to split line into two segments")
    # Identify body vs head by start coordinate
    body_seg = min(candidates, key=lambda s: s.coords[0] != coords[0])
    head_seg = max(candidates, key=lambda s: s.coords[0] != coords[0])

    return list(body_seg.coords), list(head_seg.coords)


def insert_split_edges(
    nod_path: str,
    edg_path: str,
    con_path: str,
    tll_path: str,
    split_distance: float
) -> None:
    """
    Split edges in SUMO plain-dump files.
    """
    # Parse input files
    nod_tree = ET.parse(nod_path)
    nod_root = nod_tree.getroot()
    node_coords: Dict[str, Tuple[float, float]] = {
        node.get('id'): (float(node.get('x')), float(node.get('y')))
        for node in nod_root.findall('node')
    }

    edg_tree = ET.parse(edg_path)
    edg_root = edg_tree.getroot()
    con_tree = ET.parse(con_path)
    con_root = con_tree.getroot()
    tll_tree = ET.parse(tll_path)
    tll_root = tll_tree.getroot()

    # Collect edges and connections
    edge_elems = {e.get('id'): e for e in edg_root.findall('edge')}
    original_cons: List[Tuple[str, str, Dict[str, str]]] = [
        (c.get('from'), c.get('to'), c.attrib.copy()) for c in con_root.findall('connection')
    ]
    edges_to_split = {E for E, _, _ in original_cons}

    # Compute split info
    split_info = {}
    for E in edges_to_split:
        edge = edge_elems.get(E)
        if edge is None:
            raise KeyError(f"Edge '{E}' not found in {edg_path}")
        shape_str = edge.get('shape')
        coords = parse_shape(shape_str) if shape_str else [
            node_coords[edge.get('from')], node_coords[edge.get('to')]
        ]
        body_coords, head_coords = split_shape(coords, split_distance)
        split_info[E] = {
            'orig_from': edge.get('from'),
            'orig_to': edge.get('to'),
            'body_coords': body_coords,
            'head_coords': head_coords
        }

    # 1. .nod – add divergence nodes
    for E, info in split_info.items():
        div_id = f"{E}_H_node"
        x, y = info['head_coords'][0]
        # set radius 18.00 (the smallest one I found) to avoid
        # sumo warnings telling that, geometrically, two of the
        # left-turn head-links overlap inside the default junction circle.
        # Splitting the edge exactly X meters back created
        # two distinct inbound lanes whose curved turn paths still
        # cross within the junction’s small radius (typically 1 m).
        # SUMO flags this because, in reality, cars couldn’t follow
        # both turn trajectories without collision.
        ET.SubElement(nod_root, 'node', {
                      'id': div_id, 'x': str(x), 'y': str(y), 'radius': "18.00"})

    # 2. .edg – remove originals and add new edges
    for E in edges_to_split:
        edg_root.remove(edge_elems[E])
    for E, info in split_info.items():
        base_attrib = edge_elems[E].attrib.copy()
        # Body edge
        body_attrib = base_attrib.copy()
        body_attrib.update(
            {'from': info['orig_from'], 'to': f"{E}_H_node", 'shape': shape_to_str(info['body_coords'])})
        ET.SubElement(edg_root, 'edge', body_attrib)
        # Head edges
        for orig_E, F, _ in original_cons:
            if orig_E != E:
                continue
            head_attrib = base_attrib.copy()
            head_attrib.update({'id': f"{E}_H_{F}", 'from': f"{E}_H_node",
                               'to': info['orig_to'], 'shape': shape_to_str(info['head_coords'])})
            ET.SubElement(edg_root, 'edge', head_attrib)

    # 3. .con – rewrite connections
    for c in list(con_root.findall('connection')):
        if c.get('from') in edges_to_split:
            con_root.remove(c)
    for E, F, attrib in original_cons:
        if E in edges_to_split:
            new_attrib = attrib.copy()
            new_attrib['from'] = f"{E}_H_{F}"
            ET.SubElement(con_root, 'connection', new_attrib)

    # 4. .tll – update traffic‐light logic connections
    #    - Find all <connection> tags (handling possible namespaces) and update “from” attributes
    for conn in tll_root.findall('.//{*}connection'):
        E = conn.attrib.get('from')
        F = conn.attrib.get('to')
        # If this was an original edge to split, point it to the head-edge ID
        if E in edges_to_split:
            conn.attrib['from'] = f"{E}_H_{F}"

    # Write updated XML back to files
        nod_tree.write(nod_path, encoding='UTF-8', xml_declaration=True)
        edg_tree.write(edg_path, encoding='UTF-8', xml_declaration=True)
        con_tree.write(con_path, encoding='UTF-8', xml_declaration=True)
        tll_tree.write(tll_path, encoding='UTF-8', xml_declaration=True)
