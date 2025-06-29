import xml.etree.ElementTree as ET
import math
from pathlib import Path


def parse_shape(shape_str):
    pts = []
    for coord in shape_str.strip().split():
        x_str, y_str = coord.split(',')
        pts.append((float(x_str), float(y_str)))
    return pts


def shape_to_str(pts):
    return ' '.join(f'{x:.6f},{y:.6f}' for x, y in pts)


def compute_length(pts):
    return sum(math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]) for i in range(len(pts)-1))


def find_split_index(coords, dist):
    rem = dist
    for i in range(len(coords)-1, 0, -1):
        x1, y1 = coords[i]
        x0, y0 = coords[i-1]
        seg = math.hypot(x1-x0, y1-y0)
        if rem <= seg:
            ratio = rem/seg
            sx = x1 + (x0 - x1) * ratio
            sy = y1 + (y0 - y1) * ratio
            return i-1, (sx, sy)
        rem -= seg
    return None, None


def insert_split_edges(net_file: str, dist: float) -> None:
    """
    Splits each non-internal edge in a SUMO net.xml into 'body' and 'head'
    segments at distance `dist` from the downstream end. Preserves all
    other elements (internal edges, junctions, types, location) and
    correctly retains and updates <connection> elements.
    """
    path = Path(net_file)
    tree = ET.parse(str(path))
    root = tree.getroot()

    # --- Snapshot and remove existing connections ---
    conn_nodes = [c for c in root.findall('connection')]
    for c in conn_nodes:
        root.remove(c)

    new_children = []
    split_info = []  # list of tuples: (orig_id, body_id, head_id)

    # Rebuild children without <connection>
    for node in list(root):
        # Identify splittable edges
        if (node.tag == 'edge' and 'from' in node.attrib and 'to' in node.attrib
                and node.get('function') != 'internal'):
            eid = node.get('id')
            frm, to_node = node.get('from'), node.get('to')
            lane = node.find('lane')
            if lane is None or not lane.get('shape'):
                new_children.append(node)
                continue

            coords = parse_shape(lane.get('shape'))
            idx, split_pt = find_split_index(coords, dist)
            if idx is None:
                new_children.append(node)
                continue

            # Compute split shapes & lengths
            body_coords = coords[:idx+1] + [split_pt]
            head_coords = [split_pt] + coords[idx+1:]
            body_len = compute_length(body_coords)
            head_len = compute_length(head_coords)
            lane_idx = lane.get('index', '0')

            # Create split-point internal junction
            junct = ET.Element('junction', {
                'id': f'{eid}_split', 'type': 'internal',
                'x': f'{split_pt[0]:.6f}', 'y': f'{split_pt[1]:.6f}',
                'incLanes': f'{eid}_body_{lane_idx}',
                'intLanes': f'{eid}_head_{lane_idx}'
            })

            # Build body edge
            body_attrib = node.attrib.copy()
            body_attrib.update(
                {'id': f'{eid}_body', 'from': frm, 'to': f'{eid}_split'})
            body_edge = ET.Element('edge', body_attrib)
            body_lane = ET.Element('lane', lane.attrib.copy())
            body_lane.set('id', f'{eid}_body_{lane_idx}')
            body_lane.set('shape', shape_to_str(body_coords))
            body_lane.set('length', f'{body_len:.6f}')
            body_edge.append(body_lane)

            # Build head edge
            head_attrib = node.attrib.copy()
            head_attrib.update(
                {'id': f'{eid}_head', 'from': f'{eid}_split', 'to': to_node})
            head_edge = ET.Element('edge', head_attrib)
            head_lane = ET.Element('lane', lane.attrib.copy())
            head_lane.set('id', f'{eid}_head_{lane_idx}')
            head_lane.set('shape', shape_to_str(head_coords))
            head_lane.set('length', f'{head_len:.6f}')
            head_edge.append(head_lane)

            # Append new junction and edges
            new_children.extend([junct, body_edge, head_edge])
            split_info.append((eid, f'{eid}_body', f'{eid}_head'))
        else:
            # Preserve all other nodes (including internal edges, junctions, types, location)
            new_children.append(node)

    # Replace root children (without connections)
    root[:] = new_children

    # --- Re-add and update connections ---
    for orig, body_id, head_id in split_info:
        for conn in conn_nodes:
            # Update edge references
            if conn.get('from') == orig:
                conn.set('from', body_id)
            if conn.get('to') == orig:
                conn.set('to', head_id)
            # Update lane references
            for attr in ('fromLane', 'toLane'):
                val = conn.get(attr)
                if val and orig in val:
                    new_val = val.replace(
                        orig, body_id if attr == 'fromLane' else head_id)
                    conn.set(attr, new_val)

    # Append connections back into the tree
    for conn in conn_nodes:
        root.append(conn)

    # Write modified network back to file
    tree.write(str(path), encoding='utf-8', xml_declaration=True)
