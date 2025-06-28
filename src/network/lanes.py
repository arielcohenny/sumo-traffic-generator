import xml.etree.ElementTree as ET
import random
from pathlib import Path


def repair_connectivity_in_place(net_path: str | Path) -> None:
    """
    For every <connection> that goes from lane 0→0, clone it so that
    lane i of the *from* edge connects to lane i of the *to* edge for all
    i = 1 … min(n_from, n_to)-1.  Overwrites the .net.xml in place.
    """
    net_path = Path(net_path)
    tree = ET.parse(net_path)
    root = tree.getroot()

    new_conns = []
    for conn in root.findall('connection'):
        fn = conn.get('from')
        tn = conn.get('to')
        fl = conn.get('fromLane')
        tl = conn.get('toLane')
        if fl.endswith('0') and tl.endswith('0'):
            # count lanes on each edge
            n_from = max(int(l.get('index'))
                         for l in root.findall(f"edge[@id='{fn}']/lane")) + 1
            n_to = max(int(l.get('index'))
                       for l in root.findall(f"edge[@id='{tn}']/lane")) + 1
            for i in range(1, min(n_from, n_to)):
                new = ET.Element('connection', conn.attrib.copy())
                new.set('fromLane', fl[:-1] + str(i))
                new.set('toLane', tl[:-1] + str(i))
                new_conns.append(new)
    # append cloned connections
    for nc in new_conns:
        root.append(nc)
    tree.write(net_path, encoding='utf-8', xml_declaration=True)


def set_lane_counts(net_file_in: str | Path,
                    net_file_out: str | Path,
                    seed: int,
                    min_lanes: int,
                    max_lanes: int) -> None:
    """
    Rewrites every <lane> element on each edge (including internal) in the .net.xml
    to a random count between min_lanes and max_lanes, preserving all other elements
    (junctions, connections, types, location).
    """
    net_file_in = Path(net_file_in)
    net_file_out = Path(net_file_out)

    # parse network
    tree = ET.parse(net_file_in)
    root = tree.getroot()

    rng = random.Random(seed)
    new_children = []

    # rebuild network, updating lanes for all edges with existing lanes
    for node in list(root):
        if node.tag == 'edge' and node.find('lane') is not None:
            edge = node
            old_lanes = edge.findall('lane')
            # remove all existing lanes
            for l in old_lanes:
                edge.remove(l)
            # decide random lane count
            num_lanes = rng.randint(min_lanes, max_lanes)
            # preserve original lane attributes
            speed = old_lanes[0].get('speed')
            length = old_lanes[0].get('length')
            shape = old_lanes[0].get('shape') or '0.00,0.00 1.00,0.00'
            # add new lanes
            for i in range(num_lanes):
                lane_attrs = {
                    'id': f"{edge.get('id')}_{i}",
                    'index': str(i),
                    'speed': speed,
                    'length': length,
                    'shape': shape
                }
                ET.SubElement(edge, 'lane', lane_attrs)
        # always preserve the node
        new_children.append(node)

    # replace all children with updated list
    root[:] = new_children

    # write modified network
    net_file_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(net_file_out, encoding='utf-8', xml_declaration=True)

    # repair connectivity in-place (cloning lane connections)
    repair_connectivity_in_place(net_file_out)
