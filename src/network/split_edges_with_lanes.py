import xml.etree.ElementTree as ET
import subprocess
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from src.config import CONFIG
from src.constants import MIN_LANE_COUNT, MAX_LANE_COUNT
from src.network.lane_counts import calculate_lane_count


def calculate_bearing(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    """Calculate bearing angle from start point to end point in radians."""
    return math.atan2(end_y - start_y, end_x - start_x)


def normalize_angle_degrees(angle_degrees: float) -> float:
    """Normalize angle to [-180, 180] degree range."""
    while angle_degrees > 180:
        angle_degrees -= 360
    while angle_degrees < -180:
        angle_degrees += 360
    return angle_degrees


def calculate_offset_shape(base_shape: str, lateral_offset_m: float) -> str:
    """
    Calculate offset shape for multi-head edges.

    Args:
        base_shape: Original shape string "x1,y1 x2,y2"
        lateral_offset_m: Lateral offset in meters (positive = right, negative = left)

    Returns:
        Offset shape string
    """
    # Parse shape coordinates
    coords = base_shape.strip().split()
    if len(coords) < 2:
        return base_shape  # Can't offset if not enough points

    start = coords[0].split(',')
    end = coords[1].split(',')
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])

    # Calculate direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)

    if length < 0.01:  # Avoid division by zero
        return base_shape

    # Calculate perpendicular unit vector (90° clockwise rotation to the right)
    perp_x = dy / length
    perp_y = -dx / length

    # Apply offset to both start and end points
    new_x1 = x1 + perp_x * lateral_offset_m
    new_y1 = y1 + perp_y * lateral_offset_m
    new_x2 = x2 + perp_x * lateral_offset_m
    new_y2 = y2 + perp_y * lateral_offset_m

    return f"{new_x1:.2f},{new_y1:.2f} {new_x2:.2f},{new_y2:.2f}"


def split_edges_with_flow_based_lanes(seed: int, min_lanes: int, max_lanes: int, algorithm: str, block_size_m: int = 200, traffic_light_strategy: str = "opposites") -> None:
    """Integrated edge splitting with flow-based lane assignment.

    Replaces separate edge splitting and lane configuration steps with a single
    integrated approach that:
    1. Analyzes original netgenerate connections to determine movement counts
    2. Splits edges at HEAD_DISTANCE from downstream junction
    3. Assigns lanes using existing algorithms (realistic/random/fixed)
    4. Updates all 4 XML files (.nod/.edg/.con/.tll) maintaining structure

    Args:
        seed: Random seed for reproducibility
        min_lanes: Minimum number of lanes
        max_lanes: Maximum number of lanes
        algorithm: Lane count algorithm ('realistic', 'random', or fixed number)
        block_size_m: Block size in meters
        traffic_light_strategy: Traffic light strategy (affects minimum lanes)
    """

    # Initialize random number generator
    rng = random.Random(seed)

    # print("Starting integrated edge splitting with flow-based lane assignment...")

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
    split_edges, new_nodes = split_edges_at_head_distance(
        edg_root, edge_coords, movement_data, algorithm, rng, min_lanes, max_lanes, block_size_m, traffic_light_strategy)

    # Step 4.5: Post-process to create multi-head structure
    split_edges = post_process_to_multi_head_edges(
        split_edges, movement_data, edge_coords)

    # Step 5: Update all XML files
    update_nodes_file(nod_root, new_nodes)
    update_edges_file(edg_root, split_edges)
    update_connections_file(con_root, split_edges, movement_data, edge_coords)
    update_traffic_lights_file(tll_root, split_edges, con_root)

    # Step 6: Write updated files
    write_xml_files(nod_tree, edg_tree, con_tree, tll_tree)

    # print("Completed integrated edge splitting with flow-based lane assignment.")


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


def calculate_movement_angles(from_edge_coords: Tuple[float, float, float, float],
                              movements: List[Dict],
                              edge_coords: Dict[str, Tuple[float, float, float, float]]) -> List[Dict]:
    """
    Calculate actual turn angles for movements without arbitrary classification.

    Args:
        from_edge_coords: (start_x, start_y, end_x, end_y) of incoming edge
        movements: List of movement dictionaries with 'to_edge' field
        edge_coords: Dictionary mapping edge_id -> (start_x, start_y, end_x, end_y)

    Returns:
        List of movements with added angle information:
        [
            {
                'to_edge': str,
                'from_lanes': List[int],
                'num_lanes': int,
                'turn_angle': float,  # NEW: actual turn angle in degrees [-180, 180]
            },
            ...
        ]
    """
    movements_with_angles = []

    # Calculate incoming edge direction vector
    from_start_x, from_start_y, from_end_x, from_end_y = from_edge_coords
    incoming_angle_rad = calculate_bearing(
        from_start_x, from_start_y, from_end_x, from_end_y)

    # Process each movement
    for movement in movements:
        to_edge = movement['to_edge']

        # Get outgoing edge coordinates
        if to_edge in edge_coords:
            to_start_x, to_start_y, to_end_x, to_end_y = edge_coords[to_edge]

            # Calculate outgoing edge direction vector
            outgoing_angle_rad = calculate_bearing(
                to_start_x, to_start_y, to_end_x, to_end_y)

            # Calculate turn angle relative to incoming direction
            turn_angle_rad = outgoing_angle_rad - incoming_angle_rad
            turn_angle_deg = turn_angle_rad * 180 / math.pi

            # Normalize to [-180, 180] range
            turn_angle_deg = normalize_angle_degrees(turn_angle_deg)

            # Create enhanced movement data
            enhanced_movement = {
                **movement,  # Copy all original fields
                'turn_angle': turn_angle_deg
            }

            movements_with_angles.append(enhanced_movement)
        else:
            # Handle missing edge coordinates (fallback)
            # Keep original movement but add default angle
            enhanced_movement = {
                **movement,
                'turn_angle': 0.0  # Default to straight
            }
            movements_with_angles.append(enhanced_movement)

    return movements_with_angles


def assign_lanes_by_angle(movements_with_angles: List[Dict], head_lanes: int) -> Dict[str, List[int]]:
    """
    Assign head lanes to movements based on actual turn angles.

    Spatial Logic (SUMO lane indexing: lane 0 = rightmost, lane n-1 = leftmost):
    - Left turns (positive angles): Get leftmost lanes (high lane indices)
    - Straight (near 0°): Get middle lanes  
    - Right turns (negative angles): Get rightmost lanes (low lane indices)
    - U-turns (±180°): Get leftmost lanes (like left turns)

    Args:
        movements_with_angles: List of movements with turn_angle field
        head_lanes: Total number of head lanes available

    Returns:
        Dictionary mapping to_edge -> list of assigned head lane indices
    """
    if not movements_with_angles:
        return {}

    # Sort movements by turn angle (leftmost to rightmost)
    # Key insight: Sort by NEGATIVE turn_angle so:
    # - Left turns (positive angles) come first → get leftmost lanes
    # - Straight (0°) comes middle → get middle lanes
    # - Right turns (negative angles) come last → get rightmost lanes
    # SPECIAL HANDLING: U-turns (±180°) should always be treated as leftmost
    def sort_key(movement):
        angle = movement['turn_angle']
        # Treat U-turns (both +180° and -180°) as extreme left turns
        if abs(angle) >= 179.0:  # U-turn threshold
            # Lower than any normal left turn (comes first in sorted list)
            return -200.0
        else:
            return -angle  # Normal sorting by negative angle

    sorted_movements = sorted(movements_with_angles, key=sort_key)

    # Initialize lane assignment tracking
    movement_to_head_lanes = {}
    head_lane_idx = 0

    # Assign consecutive lanes to each movement
    # IMPORTANT: SUMO lane indexing is right-to-left (lane 0 = rightmost, lane n-1 = leftmost)
    # So we need to reverse the lane assignments to match SUMO's convention
    for movement in sorted_movements:
        to_edge = movement['to_edge']
        num_lanes_needed = movement['num_lanes']

        # Assign consecutive head lanes (but reverse them for SUMO's right-to-left indexing)
        assigned_head_lanes = []
        for _ in range(num_lanes_needed):
            if head_lane_idx < head_lanes:
                # Reverse the lane index: rightmost lane (0) for right turns, leftmost lane (n-1) for left turns
                sumo_lane_idx = head_lanes - 1 - head_lane_idx
                assigned_head_lanes.append(sumo_lane_idx)
                head_lane_idx += 1
            else:
                # Handle insufficient lanes (shouldn't happen with proper head_lanes calculation)
                break

        # Store assignment if lanes were assigned
        if assigned_head_lanes:
            movement_to_head_lanes[to_edge] = assigned_head_lanes

    return movement_to_head_lanes


def split_edges_at_head_distance(edg_root, edge_coords: Dict[str, Tuple[float, float, float, float]],
                                 movement_data: Dict[str, Dict], algorithm: str, rng, min_lanes: int, max_lanes: int, block_size_m: int, traffic_light_strategy: str = "opposites") -> Tuple[Dict[str, Dict], List[Dict]]:
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
        edge_movement_info = movement_data.get(
            edge_id, {'total_movement_lanes': 1, 'movements': []})
        total_movement_lanes = edge_movement_info['total_movement_lanes']
        tail_lanes = calculate_lane_count(
            edge_id, algorithm, rng, min_lanes, max_lanes, block_size_m, traffic_light_strategy)
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
        head_edge_id = f"{edge_id}_H_s"
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


def post_process_to_multi_head_edges(split_edges: Dict[str, Dict],
                                     movement_data: Dict[str, Dict],
                                     edge_coords: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Dict]:
    """
    Post-process single-head edges to create multi-head structure.

    Takes the output of split_edges_at_head_distance() and splits each single head edge
    into multiple movement-specific head edges based on actual lane assignments.

    Args:
        split_edges: Dictionary from split_edges_at_head_distance()
        movement_data: Movement data from analyze_movements_from_connections()
        edge_coords: Edge coordinates for turn angle calculation

    Returns:
        Updated split_edges dictionary with multi-head structure
    """
    updated_split_edges = {}
    edges_processed = 0
    edges_with_multi_head = 0

    for edge_id, split_data in split_edges.items():
        edges_processed += 1

        # Get movement data
        if edge_id not in movement_data or not movement_data[edge_id]['movements']:
            updated_split_edges[edge_id] = split_data
            continue

        movements = movement_data[edge_id]['movements']

        # If only one movement, keep single head
        if len(movements) <= 1:
            updated_split_edges[edge_id] = split_data
            continue

        edges_with_multi_head += 1

        # Calculate turn angles and classify movements
        from_edge_coords = edge_coords.get(edge_id)
        movements_with_angles = []

        if from_edge_coords:
            from_start_x, from_start_y, from_end_x, from_end_y = from_edge_coords
            incoming_angle_rad = calculate_bearing(
                from_start_x, from_start_y, from_end_x, from_end_y)

            for movement in movements:
                dest_edge = movement['to_edge']
                if dest_edge in edge_coords:
                    to_start_x, to_start_y, to_end_x, to_end_y = edge_coords[dest_edge]
                    outgoing_angle_rad = calculate_bearing(
                        to_start_x, to_start_y, to_end_x, to_end_y)
                    turn_angle_rad = outgoing_angle_rad - incoming_angle_rad
                    turn_angle_deg = turn_angle_rad * 180 / math.pi
                    turn_angle_deg = normalize_angle_degrees(turn_angle_deg)
                    movements_with_angles.append({
                        **movement,
                        'turn_angle': turn_angle_deg
                    })
                else:
                    movements_with_angles.append({
                        **movement,
                        'turn_angle': 0.0
                    })

        # Group movements by movement type
        # movement_type -> {'lanes': [...], 'to_edges': [...]}
        movement_groups = {}

        for movement in movements_with_angles:
            turn_angle = movement['turn_angle']

            # Inline movement type classification (no hardcoded geometry)
            abs_angle = abs(turn_angle)
            if abs_angle >= 179.0:
                movement_type = 'uturn'
            elif turn_angle < -45:
                movement_type = 'right'
            elif turn_angle > 45:
                movement_type = 'left'
            else:
                movement_type = 'straight'

            dest_edge = movement['to_edge']
            from_lanes = movement['from_lanes']

            if movement_type not in movement_groups:
                movement_groups[movement_type] = {
                    'lanes': [],
                    'to_edges': []
                }
            movement_groups[movement_type]['lanes'].extend(from_lanes)
            movement_groups[movement_type]['to_edges'].append(dest_edge)

        # Create multi-head structure with proper geometry offsets
        multi_heads = {}
        original_head = split_data['head']
        total_head_lanes = split_data['head_lanes']
        LANE_WIDTH = 3.2  # Standard SUMO lane width in meters

        # Sort movement types: uturn, left, straight, right
        # U-turn (idx=0) gets lowest position → most negative offset → leftmost lane
        # Right turn (idx=last) gets highest position → most positive offset → rightmost lane
        movement_order = ['uturn', 'left', 'straight', 'right']
        sorted_movements = []
        for mt in movement_order:
            if mt in movement_groups:
                sorted_movements.append((mt, movement_groups[mt]))

        # Calculate spacing: spread movements evenly across total_head_lanes
        num_movements = len(sorted_movements)

        for idx, (movement_type, group_data) in enumerate(sorted_movements):
            # Original lane indices from single head
            lanes = sorted(set(group_data['lanes']))
            num_lanes = len(lanes)
            to_edges = group_data['to_edges']

            # For simplicity, use first destination edge (they should all be same movement type)
            primary_dest = to_edges[0] if to_edges else None

            # Calculate position for this movement within the total head lanes
            # idx=0 (uturn) → position=0 → most negative offset → leftmost lane
            # idx=last (right) → position=total_head_lanes-1 → most positive offset → rightmost lane
            if num_movements > 1:
                # Position this movement proportionally across the total lanes
                movement_position = idx * \
                    (total_head_lanes - 1) / (num_movements - 1)
            else:
                # Single movement: center it
                movement_position = (total_head_lanes - 1) / 2.0

            # Center of all lanes
            center_pos = (total_head_lanes - 1) / 2.0

            # Offset from center in lane units
            lane_offset = movement_position - center_pos

            # Convert to meters
            # Negative offset = right side (right of travel)
            # Positive offset = left side (left of travel)
            movement_offset_m = lane_offset * LANE_WIDTH

            # Add direction-based base offset to separate opposite directions
            # The perpendicular calculation ALREADY handles direction correctly!
            # Both directions get positive base offset:
            # - Northbound: perpendicular = East, +offset goes East (right side)
            # - Southbound: perpendicular = West, +offset goes West (right side, opposite side of road!)
            tail_lanes = split_data['tail_lanes']

            if (tail_lanes == 2):
                direction_base_offset_m = (tail_lanes * LANE_WIDTH)
            else:
                direction_base_offset_m = (tail_lanes * LANE_WIDTH) / 2.0

            # Total offset: base offset (separates directions) + movement offset (separates movements)
            total_offset_m = direction_base_offset_m + movement_offset_m

            # Calculate offset shape
            offset_shape = calculate_offset_shape(
                original_head['shape'], total_offset_m)

            # Create new head edge for this movement type
            new_head_id = f"{edge_id}_H_{movement_type}"

            new_head_edge = {
                'id': new_head_id,
                'from': original_head['from'],
                'to': original_head['to'],
                'priority': original_head['priority'],
                'numLanes': str(num_lanes),
                'speed': original_head['speed'],
                'shape': offset_shape  # Properly calculated offset based on lane positions
            }

            multi_heads[movement_type] = {
                'edge': new_head_edge,
                'original_lanes': lanes,  # Original lane indices from single head
                'num_lanes': num_lanes,
                'to_edge': primary_dest  # Primary destination for this movement type
            }

        # Update split_edges entry with multi-head structure
        updated_split_edges[edge_id] = {
            **split_data,
            'heads': multi_heads  # Add multi-head structure
        }

    print(
        f"Post-processing: {edges_processed} edges processed, {edges_with_multi_head} converted to multi-head")
    return updated_split_edges


def update_nodes_file(nod_root, new_nodes: List[Dict]):
    """Add new intermediate nodes to the nodes file."""
    for node_data in new_nodes:
        node_elem = ET.SubElement(nod_root, "node")
        node_elem.set("id", node_data['id'])
        node_elem.set("x", str(node_data['x']))
        node_elem.set("y", str(node_data['y']))
        node_elem.set("radius", node_data['radius'])


def update_edges_file(edg_root, split_edges: Dict[str, Dict]):
    """Update edges with split tail and head segments (supports multi-head structure)."""
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

        # Add head segment(s)
        if 'heads' in split_data:
            # Multi-head structure: write multiple head edges
            for movement_type, head_data in split_data['heads'].items():
                head_elem = ET.SubElement(edg_root, "edge")
                for attr, value in head_data['edge'].items():
                    head_elem.set(attr, value)
        else:
            # Single-head structure (fallback for edges not processed)
            head_elem = ET.SubElement(edg_root, "edge")
            for attr, value in split_data['head'].items():
                head_elem.set(attr, value)


def update_connections_file(con_root, split_edges: Dict[str, Dict], movement_data: Dict[str, Dict], edge_coords: Dict[str, Tuple[float, float, float, float]]):
    """Update connections to reference head segments (supports multi-head structure)."""

    # Remove all existing connections (we'll regenerate them)
    connections_to_remove = list(con_root.findall("connection"))
    for conn in connections_to_remove:
        con_root.remove(conn)

    # Regenerate connections for each edge
    for edge_id, split_data in split_edges.items():
        if 'heads' in split_data:
            # Multi-head structure: create connections for each head
            tail_id = split_data['tail']['id']
            tail_lanes = split_data['tail_lanes']

            # 1. INTERNAL CONNECTIONS: tail → heads
            # Distribute tail lanes across all head edges evenly
            # Each tail lane connects to one or more head lanes (across different heads)

            total_head_lanes = sum(h['num_lanes']
                                   for h in split_data['heads'].values())

            # Flatten all head lanes with their movement types for even distribution
            # [(movement_type, head_edge_id, head_lane_idx), ...]
            head_lanes_list = []
            for movement_type, head_data in split_data['heads'].items():
                head_edge_id = f"{edge_id}_H_{movement_type}"
                num_lanes = head_data['num_lanes']
                for lane_idx in range(num_lanes):
                    head_lanes_list.append(
                        (movement_type, head_edge_id, lane_idx))

            # Distribute tail lanes evenly across all available head lanes
            if tail_lanes <= total_head_lanes:
                # Each tail lane connects to multiple head lanes (1 or more heads)
                lanes_per_tail = total_head_lanes // tail_lanes
                extra_lanes = total_head_lanes % tail_lanes

                head_idx = 0
                for tail_lane in range(tail_lanes):
                    # How many head lanes this tail lane should connect to
                    connections_count = lanes_per_tail + \
                        (1 if tail_lane < extra_lanes else 0)

                    for _ in range(connections_count):
                        if head_idx < len(head_lanes_list):
                            movement_type, head_edge_id, head_lane_idx = head_lanes_list[head_idx]

                            conn_elem = ET.SubElement(con_root, "connection")
                            conn_elem.set("from", tail_id)
                            conn_elem.set("to", head_edge_id)
                            conn_elem.set("fromLane", str(tail_lane))
                            conn_elem.set("toLane", str(head_lane_idx))

                            head_idx += 1
            else:
                # More tail lanes than head lanes - multiple tail lanes connect to each head lane
                tails_per_head = tail_lanes // total_head_lanes
                extra_tails = tail_lanes % total_head_lanes

                tail_idx = 0
                for head_idx, (movement_type, head_edge_id, head_lane_idx) in enumerate(head_lanes_list):
                    # How many tail lanes should connect to this head lane
                    connections_count = tails_per_head + \
                        (1 if head_idx < extra_tails else 0)

                    for _ in range(connections_count):
                        if tail_idx < tail_lanes:
                            conn_elem = ET.SubElement(con_root, "connection")
                            conn_elem.set("from", tail_id)
                            conn_elem.set("to", head_edge_id)
                            conn_elem.set("fromLane", str(tail_idx))
                            conn_elem.set("toLane", str(head_lane_idx))

                            tail_idx += 1

            # 2. EXTERNAL CONNECTIONS: heads → destinations
            for movement_type, head_data in split_data['heads'].items():
                head_edge_id = f"{edge_id}_H_{movement_type}"
                dest_edge = head_data['to_edge']
                num_lanes = head_data['num_lanes']

                # Each head lane connects to destination (simple 1:1 for now)
                for lane_idx in range(num_lanes):
                    conn_elem = ET.SubElement(con_root, "connection")
                    conn_elem.set("from", head_edge_id)
                    conn_elem.set("to", dest_edge)
                    conn_elem.set("fromLane", str(lane_idx))
                    # Simplified - all go to lane 0 of destination
                    conn_elem.set("toLane", "0")

        else:
            # Single-head structure (fallback)
            tail_id = split_data['tail']['id']
            head_id = split_data['head']['id']
            tail_lanes = split_data['tail_lanes']
            head_lanes = split_data['head_lanes']

            # Internal connections: tail → head
            if tail_lanes <= head_lanes:
                lanes_per_tail = head_lanes // tail_lanes
                extra_lanes = head_lanes % tail_lanes

                head_lane_idx = 0
                for tail_lane in range(tail_lanes):
                    connections_for_this_tail = lanes_per_tail + \
                        (1 if tail_lane < extra_lanes else 0)

                    for _ in range(connections_for_this_tail):
                        if head_lane_idx < head_lanes:
                            conn_elem = ET.SubElement(con_root, "connection")
                            conn_elem.set("from", tail_id)
                            conn_elem.set("to", head_id)
                            conn_elem.set("fromLane", str(tail_lane))
                            conn_elem.set("toLane", str(head_lane_idx))
                            head_lane_idx += 1
            else:
                tail_per_head = tail_lanes // head_lanes
                extra_tails = tail_lanes % head_lanes

                tail_lane_idx = 0
                for head_lane in range(head_lanes):
                    connections_for_this_head = tail_per_head + \
                        (1 if head_lane < extra_tails else 0)

                    for _ in range(connections_for_this_head):
                        if tail_lane_idx < tail_lanes:
                            conn_elem = ET.SubElement(con_root, "connection")
                            conn_elem.set("from", tail_id)
                            conn_elem.set("to", head_id)
                            conn_elem.set("fromLane", str(tail_lane_idx))
                            conn_elem.set("toLane", str(head_lane))
                            tail_lane_idx += 1

            # External connections: head → destinations (from movement_data)
            if edge_id in movement_data:
                for movement in movement_data[edge_id]['movements']:
                    dest_edge = movement['to_edge']
                    from_lanes = movement['from_lanes']

                    for lane in from_lanes:
                        conn_elem = ET.SubElement(con_root, "connection")
                        conn_elem.set("from", head_id)
                        conn_elem.set("to", dest_edge)
                        conn_elem.set("fromLane", str(lane))
                        conn_elem.set("toLane", "0")


def update_traffic_lights_file(tll_root, split_edges: Dict[str, Dict], con_root):
    """Update traffic light connections to reference head segments (supports multi-head structure)."""

    # Remove all traffic light connections - let netconvert regenerate them
    # This is simpler and more reliable for multi-head structure
    connections_to_remove = list(tll_root.findall("connection"))
    for conn in connections_to_remove:
        tll_root.remove(conn)

    # Note: netconvert will regenerate traffic light connections based on
    # the connections in .con.xml file, so we don't need to manually create them


def write_xml_files(nod_tree, edg_tree, con_tree, tll_tree):
    """Write all updated XML files."""

    # Format XML with proper indentation
    for tree in [nod_tree, edg_tree, con_tree, tll_tree]:
        ET.indent(tree.getroot())

    # Write files
    nod_tree.write(CONFIG.network_nod_file,
                   encoding="UTF-8", xml_declaration=True)
    edg_tree.write(CONFIG.network_edg_file,
                   encoding="UTF-8", xml_declaration=True)
    con_tree.write(CONFIG.network_con_file,
                   encoding="UTF-8", xml_declaration=True)
    tll_tree.write(CONFIG.network_tll_file,
                   encoding="UTF-8", xml_declaration=True)


def execute_edge_splitting(args) -> None:
    """Execute edge splitting with lane assignment."""
    import logging
    from src.utils.multi_seed_utils import get_network_seed
    from src.validate.validate_split_edges_with_lanes import verify_split_edges_with_flow_based_lanes
    from src.validate.errors import ValidationError

    logger = logging.getLogger(__name__)

    if args.lane_count != "0" and not (args.lane_count.isdigit() and args.lane_count == "0"):
        split_edges_with_flow_based_lanes(
            seed=get_network_seed(args),
            min_lanes=MIN_LANE_COUNT,
            max_lanes=MAX_LANE_COUNT,
            algorithm=args.lane_count,
            block_size_m=args.block_size_m,
            traffic_light_strategy=args.traffic_light_strategy
        )
        # logger.info(
        #     "Successfully completed integrated edge splitting with lane assignment")

        # Validate the split edges
        # TEMPORARILY DISABLED: Validation needs to be updated for multi-head structure
        # try:
        #     verify_split_edges_with_flow_based_lanes(
        #         connections_file=str(CONFIG.network_con_file),
        #         edges_file=str(CONFIG.network_edg_file),
        #         nodes_file=str(CONFIG.network_nod_file)
        #     )
        # except (ValidationError, ValueError) as ve:
        #     logger.error(f"Split edges validation failed: {ve}")
        #     raise
    else:
        logger.info("Skipping lane assignment (lane_count is 0)")
