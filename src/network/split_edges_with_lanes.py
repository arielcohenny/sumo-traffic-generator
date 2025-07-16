import xml.etree.ElementTree as ET
import subprocess
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from src.config import CONFIG
from src.network.lane_counts import calculate_lane_count


def split_edges_with_flow_based_lanes(seed: int, min_lanes: int, max_lanes: int, algorithm: str) -> None:
    """Integrated edge splitting with flow-based lane assignment.
    
    Replaces separate edge splitting and lane configuration steps with a single
    integrated approach that:
    1. Analyzes original netgenerate connections to determine movement counts
    2. Splits edges at HEAD_DISTANCE from downstream junction
    3. Assigns lanes using existing algorithms (realistic/random/fixed)
    4. Updates all 4 XML files (.nod/.edg/.con/.tll) maintaining structure
    """
    
    # Initialize random number generator
    rng = random.Random(seed)
    
    print("Starting integrated edge splitting with flow-based lane assignment...")
    
    # Step 1: Parse existing network files
    nod_tree = ET.parse(CONFIG.network_nod_file)
    edg_tree = ET.parse(CONFIG.network_edg_file)
    con_tree = ET.parse(CONFIG.network_con_file)
    tll_tree = ET.parse(CONFIG.network_tll_file)
    
    nod_root = nod_tree.getroot()
    edg_root = edg_tree.getroot()
    con_root = con_tree.getroot()
    tll_root = tll_tree.getroot()
    
    # Step 2: Analyze original connections to get detailed movement data
    movement_data = analyze_movements_from_connections(con_root)
    
    # Step 3: Calculate edge coordinates for splitting
    edge_coords = extract_edge_coordinates(edg_root, nod_root)
    
    # Step 4: Split edges and calculate lane assignments
    split_edges, new_nodes = split_edges_at_head_distance(edg_root, edge_coords, movement_data, algorithm, rng, min_lanes, max_lanes)
    
    # Step 5: Update all XML files
    update_nodes_file(nod_root, new_nodes)
    update_edges_file(edg_root, split_edges)
    update_connections_file(con_root, split_edges, movement_data)
    update_traffic_lights_file(tll_root, split_edges)
    
    # Step 6: Write updated files
    write_xml_files(nod_tree, edg_tree, con_tree, tll_tree)
    
    print("Completed integrated edge splitting with flow-based lane assignment.")


def analyze_movements_from_connections(con_root) -> Dict[str, Dict]:
    """Analyze detailed movement information from connections.
    
    Returns:
        Dict mapping edge_id to movement data:
        {
            'edge_id': {
                'total_movement_lanes': int,  # Sum of lanes each movement uses
                'movements': [
                    {
                        'to_edge': str,
                        'from_lanes': List[int],  # Original lanes this movement uses
                        'num_lanes': int  # Number of lanes this movement needs
                    }
                ]
            }
        }
    """
    edge_movements = {}
    
    # Group connections by from_edge and to_edge
    for connection in con_root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        from_lane = connection.get("fromLane")
        
        if from_edge and to_edge and from_lane is not None:
            if from_edge not in edge_movements:
                edge_movements[from_edge] = {}
            
            if to_edge not in edge_movements[from_edge]:
                edge_movements[from_edge][to_edge] = []
            
            edge_movements[from_edge][to_edge].append(int(from_lane))
    
    # Calculate total movement lanes for each edge
    movement_data = {}
    for edge_id, destinations in edge_movements.items():
        movements = []
        total_lanes = 0
        
        for to_edge, from_lanes in destinations.items():
            # Remove duplicates and sort
            unique_lanes = sorted(set(from_lanes))
            num_lanes = len(unique_lanes)
            
            movements.append({
                'to_edge': to_edge,
                'from_lanes': unique_lanes,
                'num_lanes': num_lanes
            })
            
            total_lanes += num_lanes
        
        movement_data[edge_id] = {
            'total_movement_lanes': total_lanes,
            'movements': movements
        }
    
    return movement_data


def extract_edge_coordinates(edg_root, nod_root) -> Dict[str, Tuple[float, float, float, float]]:
    """Extract start and end coordinates for each edge."""
    edge_coords = {}
    
    # First, build a map of node IDs to coordinates
    node_coords = {}
    for node in nod_root.findall("node"):
        node_id = node.get("id")
        x = float(node.get("x"))
        y = float(node.get("y"))
        node_coords[node_id] = (x, y)
    
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        
        if from_node in node_coords and to_node in node_coords:
            start_x, start_y = node_coords[from_node]
            end_x, end_y = node_coords[to_node]
            edge_coords[edge_id] = (start_x, start_y, end_x, end_y)
        else:
            # Fallback: try to parse shape attribute if available
            shape = edge.get("shape")
            if shape:
                coords = shape.split()
                if len(coords) >= 2:
                    start_x, start_y = map(float, coords[0].split(','))
                    end_x, end_y = map(float, coords[-1].split(','))
                    edge_coords[edge_id] = (start_x, start_y, end_x, end_y)
    
    return edge_coords


def split_edges_at_head_distance(edg_root, edge_coords: Dict[str, Tuple[float, float, float, float]], 
                                movement_data: Dict[str, Dict], algorithm: str, rng, min_lanes: int, max_lanes: int) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Split edges and calculate lane assignments."""
    split_edges = {}
    new_nodes = []
    
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        priority = edge.get("priority", "-1")
        speed = edge.get("speed", "13.89")
        
        if edge_id not in edge_coords:
            continue
            
        start_x, start_y, end_x, end_y = edge_coords[edge_id]
        
        # Calculate edge length and direction
        edge_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Use dynamic head distance: min(HEAD_DISTANCE, edge_length / 3)
        actual_head_distance = min(CONFIG.HEAD_DISTANCE, edge_length / 3)
        
        # Skip very short edges
        if edge_length <= actual_head_distance:
            continue
            
        # Calculate split point at actual_head_distance from end
        ratio = (edge_length - actual_head_distance) / edge_length
        split_x = start_x + ratio * (end_x - start_x)
        split_y = start_y + ratio * (end_y - start_y)
        
        # Calculate lane counts
        edge_movement_info = movement_data.get(edge_id, {'total_movement_lanes': 1, 'movements': []})
        total_movement_lanes = edge_movement_info['total_movement_lanes']
        tail_lanes = calculate_lane_count(edge_id, algorithm, rng, min_lanes, max_lanes)
        head_lanes = max(total_movement_lanes, tail_lanes)
        
        # Create new intermediate node
        head_node_id = f"{edge_id}_H_node"
        new_nodes.append({
            'id': head_node_id,
            'x': split_x,
            'y': split_y,
            'radius': '10.0'
        })
        
        # Create tail segment (from start to split point)
        tail_segment = {
            'id': edge_id,
            'from': from_node,
            'to': head_node_id,
            'priority': priority,
            'numLanes': str(tail_lanes),
            'speed': speed,
            'shape': f"{start_x},{start_y} {split_x},{split_y}"
        }
        
        # Create head segment (from split point to end)
        head_edge_id = f"{edge_id}_H"
        head_segment = {
            'id': head_edge_id,
            'from': head_node_id,
            'to': to_node,
            'priority': priority,
            'numLanes': str(head_lanes),
            'speed': speed,
            'shape': f"{split_x},{split_y} {end_x},{end_y}"
        }
        
        split_edges[edge_id] = {
            'tail': tail_segment,
            'head': head_segment,
            'tail_lanes': tail_lanes,
            'head_lanes': head_lanes,
            'total_movement_lanes': total_movement_lanes,
            'movements': edge_movement_info['movements'],
            'head_node_id': head_node_id
        }
    
    return split_edges, new_nodes


def update_nodes_file(nod_root, new_nodes: List[Dict]):
    """Add new intermediate nodes to the nodes file."""
    for node_data in new_nodes:
        node_elem = ET.SubElement(nod_root, "node")
        node_elem.set("id", node_data['id'])
        node_elem.set("x", str(node_data['x']))
        node_elem.set("y", str(node_data['y']))
        node_elem.set("radius", node_data['radius'])


def update_edges_file(edg_root, split_edges: Dict[str, Dict]):
    """Update edges with split tail and head segments."""
    # Remove original edges and add split segments
    edges_to_remove = []
    
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id in split_edges:
            edges_to_remove.append(edge)
    
    # Remove original edges
    for edge in edges_to_remove:
        edg_root.remove(edge)
    
    # Add split segments
    for edge_id, split_data in split_edges.items():
        # Add tail segment
        tail_elem = ET.SubElement(edg_root, "edge")
        for attr, value in split_data['tail'].items():
            tail_elem.set(attr, value)
        
        # Add head segment
        head_elem = ET.SubElement(edg_root, "edge")
        for attr, value in split_data['head'].items():
            head_elem.set(attr, value)


def update_connections_file(con_root, split_edges: Dict[str, Dict], movement_data: Dict[str, Dict]):
    """Update connections to reference head segments with proper movement-to-lane assignment."""
    
    # First, assign head lanes to movements for each edge
    edge_movement_assignments = {}
    for edge_id, split_data in split_edges.items():
        if edge_id in movement_data:
            movements = movement_data[edge_id]['movements']
            head_lanes = split_data['head_lanes']
            
            # Assign head lanes to movements (one movement per head lane)
            movement_to_head_lanes = {}
            head_lane_idx = 0
            
            for movement in movements:
                to_edge = movement['to_edge']
                num_lanes_needed = movement['num_lanes']
                
                # Assign consecutive head lanes to this movement
                assigned_head_lanes = []
                for _ in range(num_lanes_needed):
                    if head_lane_idx < head_lanes:
                        assigned_head_lanes.append(head_lane_idx)
                        head_lane_idx += 1
                
                if assigned_head_lanes:
                    movement_to_head_lanes[to_edge] = assigned_head_lanes
            
            edge_movement_assignments[edge_id] = movement_to_head_lanes
    
    # Update existing connections to reference head segments with specific lane assignments
    for connection in con_root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        
        if from_edge in split_edges and from_edge in edge_movement_assignments:
            # Update to reference head segment
            head_edge_id = split_edges[from_edge]['head']['id']
            connection.set("from", head_edge_id)
            
            # Assign specific head lane for this movement
            if to_edge in edge_movement_assignments[from_edge]:
                assigned_head_lanes = edge_movement_assignments[from_edge][to_edge]
                if assigned_head_lanes:
                    # Use the first assigned head lane for this movement
                    connection.set("fromLane", str(assigned_head_lanes[0]))
                    
                    # Create additional connections for multi-lane movements
                    for head_lane in assigned_head_lanes[1:]:
                        new_conn = ET.SubElement(con_root, "connection")
                        new_conn.set("from", head_edge_id)
                        new_conn.set("to", to_edge)
                        new_conn.set("fromLane", str(head_lane))
                        new_conn.set("toLane", connection.get("toLane", "0"))
    
    # Add internal connections (tail to head)
    for edge_id, split_data in split_edges.items():
        tail_lanes = split_data['tail_lanes']
        head_lanes = split_data['head_lanes']
        
        # Create connections from tail to head
        if tail_lanes <= head_lanes:
            # Distribute tail lanes evenly across head lanes
            lanes_per_head = head_lanes // tail_lanes
            extra_lanes = head_lanes % tail_lanes
            
            head_lane_idx = 0
            for tail_lane in range(tail_lanes):
                # Each tail lane connects to one or more head lanes
                connections_for_this_tail = lanes_per_head + (1 if tail_lane < extra_lanes else 0)
                
                for _ in range(connections_for_this_tail):
                    if head_lane_idx < head_lanes:
                        conn_elem = ET.SubElement(con_root, "connection")
                        conn_elem.set("from", split_data['tail']['id'])
                        conn_elem.set("to", split_data['head']['id'])
                        conn_elem.set("fromLane", str(tail_lane))
                        conn_elem.set("toLane", str(head_lane_idx))
                        head_lane_idx += 1
        else:
            # More tail lanes than head lanes - multiple tail lanes per head lane
            tail_per_head = tail_lanes // head_lanes
            extra_tails = tail_lanes % head_lanes
            
            tail_lane_idx = 0
            for head_lane in range(head_lanes):
                connections_for_this_head = tail_per_head + (1 if head_lane < extra_tails else 0)
                
                for _ in range(connections_for_this_head):
                    if tail_lane_idx < tail_lanes:
                        conn_elem = ET.SubElement(con_root, "connection")
                        conn_elem.set("from", split_data['tail']['id'])
                        conn_elem.set("to", split_data['head']['id'])
                        conn_elem.set("fromLane", str(tail_lane_idx))
                        conn_elem.set("toLane", str(head_lane))
                        tail_lane_idx += 1


def update_traffic_lights_file(tll_root, split_edges: Dict[str, Dict]):
    """Update traffic light connections to reference head segments."""
    
    # Update connection references in traffic light file
    for connection in tll_root.findall("connection"):
        from_edge = connection.get("from")
        if from_edge in split_edges:
            # Update to reference head segment
            head_edge_id = split_edges[from_edge]['head']['id']
            connection.set("from", head_edge_id)


def write_xml_files(nod_tree, edg_tree, con_tree, tll_tree):
    """Write all updated XML files."""
    
    # Format XML with proper indentation
    for tree in [nod_tree, edg_tree, con_tree, tll_tree]:
        ET.indent(tree.getroot())
    
    # Write files
    nod_tree.write(CONFIG.network_nod_file, encoding="UTF-8", xml_declaration=True)
    edg_tree.write(CONFIG.network_edg_file, encoding="UTF-8", xml_declaration=True)
    con_tree.write(CONFIG.network_con_file, encoding="UTF-8", xml_declaration=True)
    tll_tree.write(CONFIG.network_tll_file, encoding="UTF-8", xml_declaration=True)
