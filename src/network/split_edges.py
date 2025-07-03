import xml.etree.ElementTree as ET
from shapely.geometry import LineString, GeometryCollection
from shapely.ops import split as split_line
from typing import List, Tuple, Dict

from src.config import CONFIG


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
    # shapely 2.x returns a GeometryCollection
    candidates = []
    if isinstance(parts, GeometryCollection):
        candidates = [g for g in parts.geoms if isinstance(g, LineString)]
    else:
        candidates = [g for g in parts if isinstance(g, LineString)]

    if len(candidates) < 2:
        raise RuntimeError("Failed to split line into two segments")
    # Identify body vs head by start coordinate
    if list(candidates[0].coords)[0] == coords[0]:
        body_seg, head_seg = candidates[0], candidates[1]
    else:
        body_seg, head_seg = candidates[1], candidates[0]

    return list(body_seg.coords), list(head_seg.coords)


def insert_split_edges() -> None:
    # Parse input files
    nod_tree = ET.parse(CONFIG.network_nod_file)
    nod_root = nod_tree.getroot()
    node_coords: Dict[str, Tuple[float, float]] = {
        node.get('id'): (float(node.get('x')), float(node.get('y')))
        for node in nod_root.findall('node')
    }

    edg_tree = ET.parse(CONFIG.network_edg_file)
    edg_root = edg_tree.getroot()
    con_tree = ET.parse(CONFIG.network_con_file)
    con_root = con_tree.getroot()
    tll_tree = ET.parse(CONFIG.network_tll_file)
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
            raise KeyError(
                f"Edge '{E}' not found in {CONFIG.network_edg_file}")
        shape_str = edge.get('shape')
        coords = parse_shape(shape_str) if shape_str else [
            node_coords[edge.get('from')], node_coords[edge.get('to')]
        ]
        body_coords, head_coords = split_shape(coords, CONFIG.HEAD_DISTANCE)
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
        ET.SubElement(nod_root, 'node', {
                      'id': div_id, 'x': str(x), 'y': str(y), 'radius': str(CONFIG.DEFAULT_JUNCTION_RADIUS)})

    # 2. .edg – remove originals and add new edges
    for E in edges_to_split:
        edg_root.remove(edge_elems[E])
    for E, info in split_info.items():
        base_attrib = edge_elems[E].attrib.copy()
        # Body edge: original id, up to new node
        body_attrib = base_attrib.copy()
        body_attrib.update({
            'from': info['orig_from'],
            'to': f"{E}_H_node",
            'shape': shape_to_str(info['body_coords'])
        })
        ET.SubElement(edg_root, 'edge', body_attrib)
        # Head edge: from new node to original 'to'
        head_attrib = base_attrib.copy()
        head_attrib.update({
            'id': f"{E}_H",
            'from': f"{E}_H_node",
            'to': info['orig_to'],
            'shape': shape_to_str(info['head_coords'])
        })
        ET.SubElement(edg_root, 'edge', head_attrib)
        # insert connection from body to head for all lanes
        num_lanes = int(edge_elems[E].get('numLanes', '1'))
        for lane in range(num_lanes):
            ET.SubElement(con_root, 'connection', {
                'from': E,
                'to':   f"{E}_H",
                'fromLane': str(lane),
                'toLane':   str(lane)
            })

    # 3. .con – rewrite connections
    for c in list(con_root.findall('connection')):
        # remove old junction connections but keep the new internal pre→post ones
        if c.get('from') in edges_to_split and not c.get('to', '').endswith('_H'):
            con_root.remove(c)
    for E, F, attrib in original_cons:
        if E in edges_to_split:
            new_attrib = attrib.copy()
            new_attrib['from'] = f"{E}_H"
            ET.SubElement(con_root, 'connection', new_attrib)

    # 4. .tll – update traffic‐light logic connections
    for conn in tll_root.findall('.//{*}connection'):
        E = conn.attrib.get('from')
        F = conn.attrib.get('to')
        if E in edges_to_split:
            conn.attrib['from'] = f"{E}_H"

    # Write updated XML back to files
    ET.indent(edg_root)
    ET.indent(nod_root)
    ET.indent(con_root)
    ET.indent(tll_root)
    nod_tree.write(CONFIG.network_nod_file,
                   encoding='UTF-8', xml_declaration=True)
    edg_tree.write(CONFIG.network_edg_file,
                   encoding='UTF-8', xml_declaration=True)
    con_tree.write(CONFIG.network_con_file,
                   encoding='UTF-8', xml_declaration=True)
    tll_tree.write(CONFIG.network_tll_file,
                   encoding='UTF-8', xml_declaration=True)
