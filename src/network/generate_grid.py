import os
import subprocess
import xml.etree.ElementTree as ET
import random
import re
from typing import List, Set, Tuple
from pathlib import Path
from src.config import CONFIG


# --- Function Definitions ---
# Generate a full grid network using netgenerate

def generate_full_grid_network(dimension, block_size_m, lane_count_arg, traffic_light_strategy="opposites", traffic_control=None):
    # Run netgenerate to create the grid network
    netgenerate_cmd = [
        "netgenerate", "--grid",
        f"--grid.x-number={dimension}",
        f"--grid.y-number={dimension}",
        f"--grid.x-length={block_size_m}",
        f"--grid.y-length={block_size_m}",
        "--default-junction-type=traffic_light",
        # no-turnarounds.geometry=false is used to make sure we have u-turns in corners
        "--no-turnarounds.geometry=false",
        # default.junctions.radius: takes care of the “Intersecting left turns…” warnings.
        # By default netgenerate uses a radius of 4m; raising it eliminates those conflicts
        f"--default.junctions.radius={CONFIG.DEFAULT_JUNCTION_RADIUS}",
        # Always use opposites for netgenerate (incoming will be post-processed)
        f"--tls.layout={traffic_light_strategy}",
        # aggregate-warnings: aggregates warnings of the same type whenever more than 1 occur
        # warnings can be removed completely by using --no-warnings=true
        # "--aggregate-warnings=1",
        "-o", CONFIG.network_file
    ]
    # Always use 1 lane from netgenerate - lane assignment will be handled by integrated step
    netgenerate_cmd.append("--default.lanenumber=1")

    try:
        subprocess.run(netgenerate_cmd, check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netgenerate execution:", e.stderr)

    # Plain-dump that .net.xml into nod/edg/con/tll files
    netconvert_cmd = [
        "netconvert",
        "--sumo-net-file", str(CONFIG.network_file),
        "--plain-output-prefix", str(CONFIG.network_prefix),
    ]
    try:
        subprocess.run(netconvert_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netconvert execution:", e.stderr)

    # Fix traffic lights for RL control only
    if traffic_control == 'rl':
        fix_traffic_lights_for_rl_control()


def pick_random_junction_ids(seed: int, num_junctions_to_remove: int, dimension: int) -> List[str]:
    if dimension < 3:
        raise ValueError(
            "Grid must be at least 3×3 to have removable interior nodes")

    row_labels = [chr(ord('A') + i) for i in range(0, dimension)]
    col_labels = list(range(0, dimension))

    candidates = [f"{row}{col}" for row in row_labels for col in col_labels]

    if num_junctions_to_remove > len(candidates):
        raise ValueError(
            f"Too many junctions requested: only {len(candidates)} interior junctions available.")

    rng = random.Random(seed)
    return rng.sample(candidates, num_junctions_to_remove)


# Regular expression to match edge IDs like 'A0B0' and 'B0C0'
_SPLIT_EDGE = re.compile(r"^([A-Z]+\d*)([A-Z]+\d*)$")


def get_cons_to_wipe(node_ids: list[str], root) -> Set[Tuple[str, str]]:
    cons_to_remove: Set[Tuple[str, str]] = set()
    # iterate over all node_ids
    for node_id in node_ids:
        # collect all connections. e.g. <connection from="A0A1" to="A1B1" fromLane="0" toLane="0"/>
        for j in root.findall("connection"):
            jfrom = j.get("from")
            jto = j.get("to")
            mfr = _SPLIT_EDGE.match(jfrom)
            mto = _SPLIT_EDGE.match(jto)
            # if the connection is part of the crossing, mark it for deletion
            #  <connection from="A0B0" to="B0C0" fromLane="0" toLane="0"/>
            # if from or to starts or ends with the node_id, it's part of the crossing
            # be carefull with prefix, e.g. for B11 don't match B1
            if (mfr and node_id in mfr.groups()) or (mto and node_id in mto.groups()):
                cons_to_remove.add((jfrom, jto))
    return cons_to_remove


def wipe_crossing_from_con(node_ids: list[str]) -> None:
    tree = ET.parse(Path(str(CONFIG.network_con_file)))
    root = tree.getroot()

    cons_to_remove: Set[Tuple[str, str]] = get_cons_to_wipe(node_ids, root)

    # remove cons that are marked for deletion
    connections_to_remove = []
    for j in root.findall("connection"):
        key = (j.get("from"), j.get("to"))
        if key in cons_to_remove:
            connections_to_remove.append(j)

    for j in connections_to_remove:
        root.remove(j)

    ET.indent(root)
    tree.write(Path(str(CONFIG.network_con_file)),
               encoding="UTF-8", xml_declaration=True)


def wipe_crossing_from_edg(node_ids: list[str]) -> None:
    tree = ET.parse(Path(str(CONFIG.network_edg_file)))
    root = tree.getroot()

    edgs_to_remove: Set[str] = set()

    # iterate over all node_ids
    for node_id in node_ids:
        # collect all edges
        for j in root.findall("edge"):
            jid = j.get("id")
            jfrom = j.get("from")
            jto = j.get("to")
            # if the edge is part of the crossing, mark it for deletion
            if jfrom == node_id or jto == node_id:
                edgs_to_remove.add(jid)

    # remove edges that are marked for deletion
    edges_to_remove_elements = []
    for j in root.findall("edge"):
        if j.get("id") in edgs_to_remove:
            edges_to_remove_elements.append(j)

    for j in edges_to_remove_elements:
        root.remove(j)

    tree.write(Path(str(CONFIG.network_edg_file)),
               encoding="UTF-8", xml_declaration=True)


def wipe_crossing_from_nod(node_ids: list[str]) -> None:
    tree = ET.parse(Path(str(CONFIG.network_nod_file)))
    root = tree.getroot()

    nods_to_remove: Set[str] = set()

    # iterate over all node_ids
    for node_id in node_ids:
        # collect all nodes
        for j in root.findall("node"):
            jid = j.get("id")
            if jid == node_id:
                nods_to_remove.add(jid)

    # remove nodes that are marked for deletion
    nodes_to_remove_elements = []
    for j in root.findall("node"):
        if j.get("id") in nods_to_remove:
            nodes_to_remove_elements.append(j)

    for j in nodes_to_remove_elements:
        root.remove(j)

    tree.write(Path(str(CONFIG.network_nod_file)),
               encoding="UTF-8", xml_declaration=True)


def wipe_crossing_from_tll(node_ids: list[str]) -> None:
    # Parse the TLS definition file
    tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    root = tree.getroot()

    # 1. Identify and remove connections to wipe
    cons_to_remove: Set[Tuple[str, str]] = get_cons_to_wipe(node_ids, root)
    # Collect (tl_id, linkIndex) for each removed connection
    states_to_remove: List[Tuple[str, int]] = []

    connections_to_remove_elements = []
    for conn in root.findall("connection"):
        key = (conn.get("from"), conn.get("to"))
        if key in cons_to_remove:
            tl_id = conn.get("tl")
            link_idx = int(conn.get("linkIndex"))
            states_to_remove.append((tl_id, link_idx))
            connections_to_remove_elements.append(conn)

    for conn in connections_to_remove_elements:
        root.remove(conn)

    # 2. Remove entire TLs for the specified node_ids
    tlls_to_remove = set(node_ids)

    # 3. Update remaining TL logic phases by dropping removed indices
    tls_to_remove_elements = []
    for tl in root.findall("tlLogic"):
        jid = tl.get("id")
        if jid in tlls_to_remove:
            tls_to_remove_elements.append(tl)
            continue

        # Collect indices to remove for this TL, in descending order
        indices = sorted(
            (idx for tl_id, idx in states_to_remove if tl_id == jid), reverse=True)
        if not indices:
            continue

        # Remove the characters at those indices in each phase's state
        for phase in tl.findall("phase"):
            state = phase.get("state", "")
            for idx in indices:
                if 0 <= idx < len(state):
                    state = state[:idx] + state[idx+1:]
            phase.set("state", state)

        # If all connections were removed, remove the entire traffic light logic
        if all(phase.get("state", "") == "" for phase in tl.findall("phase")):
            tls_to_remove_elements.append(tl)

    # Remove marked traffic lights
    for tl in tls_to_remove_elements:
        root.remove(tl)

    # 4. Reindex remaining connections so linkIndex starts at 0 and is contiguous per TL
    tl_ids = {conn.get("tl") for conn in root.findall("connection")}
    for tid in tl_ids:
        conns = [c for c in root.findall("connection") if c.get("tl") == tid]
        for new_idx, conn in enumerate(conns):
            old_idx = int(conn.get("linkIndex"))
            if old_idx != new_idx:
                conn.set("linkIndex", str(new_idx))

    # 5. Write changes back to file
    tree.write(
        Path(str(CONFIG.network_tll_file)),
        encoding="UTF-8",
        xml_declaration=True
    )


def wipe_crossing(node_ids: list[str]) -> None:
    wipe_crossing_from_con(node_ids)
    wipe_crossing_from_nod(node_ids)
    wipe_crossing_from_edg(node_ids)
    wipe_crossing_from_tll(node_ids)


def parse_junctions_to_remove(junctions_input: str) -> tuple[bool, list[str], int]:
    """Parse junctions_to_remove input and return (is_list, junction_ids, count)"""
    if not junctions_input or junctions_input == "0":
        return False, [], 0

    # Try to parse as integer first
    try:
        count = int(junctions_input)
        return False, [], count
    except ValueError:
        # Parse as comma-separated list
        junction_ids = [j.strip()
                        for j in junctions_input.split(',') if j.strip()]
        return True, junction_ids, len(junction_ids)


def convert_to_incoming_strategy():
    """Convert traffic light layout from opposites to incoming strategy"""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    # Parse the TLS definition file (for phases)
    tll_tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    tll_root = tll_tree.getroot()

    # Parse the network file (for connections - connections are in net.xml, not tll.xml!)
    net_tree = ET.parse(Path(str(CONFIG.network_net_file)))
    net_root = net_tree.getroot()

    # For each traffic light logic
    for tl in tll_root.findall("tlLogic"):
        tl_id = tl.get("id")

        # Get all connections for this traffic light from net.xml, sorted by linkIndex
        connections = [c for c in net_root.findall(
            "connection") if c.get("tl") == tl_id]
        if not connections:
            continue

        # Sort connections by linkIndex to ensure correct ordering
        connections.sort(key=lambda c: int(c.get("linkIndex")))
        total_links = len(connections)

        # Group connections by incoming edge (from)
        incoming_edges = {}
        for conn in connections:
            from_edge = conn.get("from")
            link_idx = int(conn.get("linkIndex"))
            if from_edge not in incoming_edges:
                incoming_edges[from_edge] = []
            incoming_edges[from_edge].append(link_idx)

        # Sort link indices for each incoming edge
        for edge in incoming_edges:
            incoming_edges[edge].sort()

        # Create new phases - one for each incoming edge
        new_phases = []
        seen_states = set()

        for edge, link_indices in incoming_edges.items():
            # Create green phase for this incoming edge
            state = ['r'] * total_links
            for idx in link_indices:
                if idx < total_links:  # Safety check
                    state[idx] = 'G'
            green_state = ''.join(state)

            # Create yellow phase for this incoming edge
            state = ['r'] * total_links
            for idx in link_indices:
                if idx < total_links:  # Safety check
                    state[idx] = 'y'
            yellow_state = ''.join(state)

            # Add phases only if not already seen (avoid duplicates)
            if green_state not in seen_states:
                new_phases.append(("30", green_state))  # 30 second green
                new_phases.append(("3", yellow_state))   # 3 second yellow
                seen_states.add(green_state)
                seen_states.add(yellow_state)

        # Remove old phases
        phases_to_remove = list(tl.findall("phase"))
        for phase in phases_to_remove:
            tl.remove(phase)

        # Add new phases
        for duration, state in new_phases:
            phase_elem = ET.SubElement(tl, "phase")
            phase_elem.set("duration", duration)
            phase_elem.set("state", state)

    # Write changes back to file
    ET.indent(tll_root)
    tll_tree.write(
        Path(str(CONFIG.network_tll_file)),
        encoding="UTF-8",
        xml_declaration=True
    )


def convert_to_partial_opposites_strategy():
    """Convert traffic light layout to partial_opposites strategy

    Creates 4-phase system where:
    - Phase 1: N/S straight + right turns
    - Phase 2: N/S left turns + U-turns
    - Phase 3: E/W straight + right turns
    - Phase 4: E/W left turns + U-turns

    Phase durations are calculated dynamically by convert_to_green_only_phases():
    - Interior junctions (4 phases): 90s / 4 = 22.5s per phase
    - Corner junctions (2-3 phases): 90s / N phases (equal durations)

    Total cycle: 90 seconds for all junctions (matches Tree Method interval)
    Yellow transitions (3s) are automatically added by SUMO between conflicting phases

    Requires: Minimum 2 lanes per edge to separate movement groups
    """
    import xml.etree.ElementTree as ET
    import re
    from pathlib import Path

    def get_edge_direction(edge_id: str) -> str:
        """Determine which cardinal direction an edge is heading"""
        base_edge = edge_id.split('_')[0]
        junctions = re.findall(r'[A-Z]\d+', base_edge)

        if len(junctions) != 2:
            return 'unknown'

        from_junction, to_junction = junctions
        # CORRECTED: Letter = column, Number = row
        from_col = from_junction[0]  # Letter (A, B, C, ...) = column
        from_row = int(from_junction[1:])  # Number (0, 1, 2, ...) = row
        to_col = to_junction[0]  # Letter (A, B, C, ...) = column
        to_row = int(to_junction[1:])  # Number (0, 1, 2, ...) = row

        # Determine direction based on junction change
        if from_col == to_col:  # Vertical movement (same column)
            if to_row > from_row:
                return 'N'  # Moving north (row increases upward: 0→1→2)
            elif to_row < from_row:
                return 'S'  # Moving south (row decreases downward: 2→1→0)
        elif from_row == to_row:  # Horizontal movement (same row)
            if to_col > from_col:
                return 'E'  # Moving east (column increases rightward: A→B→C)
            elif to_col < from_col:
                return 'W'  # Moving west (column decreases leftward: C→B→A)

        return 'unknown'

    def get_edge_orientation(edge_id: str) -> str:
        """Determine if edge is vertical (N/S) or horizontal (E/W)"""
        direction = get_edge_direction(edge_id)
        if direction in ['N', 'S']:
            return 'NS'
        elif direction in ['E', 'W']:
            return 'EW'
        else:
            return 'EW'  # Default fallback

    def calculate_turn_angle(from_dir: str, to_dir: str) -> float:
        """Calculate turn angle from one direction to another"""
        direction_angles = {'N': 0, 'E': 90, 'S': 180, 'W': 270}

        if from_dir not in direction_angles or to_dir not in direction_angles:
            return 0

        from_angle = direction_angles[from_dir]
        to_angle = direction_angles[to_dir]

        # Calculate relative turn angle
        turn = to_angle - from_angle

        # Normalize to -180 to +180 range
        while turn > 180:
            turn -= 360
        while turn < -180:
            turn += 360

        return turn

    def classify_movement(turn_angle: float) -> str:
        """Classify movement type for partial_opposites strategy

        The turn angle is calculated as the difference between outgoing and incoming edge directions:
            0° = straight (continuing in same direction, e.g., westbound→westbound)
            +90° = right turn (e.g., westbound→northbound)
            -90° = left turn (e.g., westbound→southbound)
            ±180° = u-turn (complete reversal, e.g., westbound→eastbound)

        For partial_opposites strategy:
            - 'straight_right': straight movements (0°) and right turns (+90°)
            - 'left_uturn': left turns (-90°/+270°) and u-turns (±180°)

        Returns:
            'straight_right' for straight (0°) or right turns (+90°)
            'left_uturn' for left turns (-90°) or u-turns (±180°)
        """
        # Normalize angle to 0-360 range for easier classification
        angle = turn_angle % 360

        # Straight: ~0° OR Right turn: ~90°
        if (angle < 10 or angle > 350) or (80 <= angle <= 100):
            return 'straight_right'
        # Left turn: ~270° (same as -90°) OR U-turn: ~180°
        else:
            return 'left_uturn'

    # Parse the traffic light logic file (for phases)
    tll_tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    tll_root = tll_tree.getroot()

    # Parse the network file (for connections - connections are in net.xml, not tll.xml!)
    net_tree = ET.parse(Path(str(CONFIG.network_file)))
    net_root = net_tree.getroot()

    # Build connection database with movement classification
    connections_db = {}  # tl_id -> list of connections with metadata

    for tl_logic in tll_root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        connections_db[tl_id] = []

        # Find all connections controlled by this traffic light (from net.xml)
        for connection in net_root.findall('connection'):
            if connection.get('tl') == tl_id:
                from_edge = connection.get('from')
                to_edge = connection.get('to')
                link_index = int(connection.get('linkIndex', -1))

                # Check if this is a multi-head edge (has _H_ suffix)
                if '_H_' in from_edge:
                    # Extract movement type directly from edge name
                    # e.g., "A0A1_H_right" → "right"
                    movement_suffix = from_edge.split('_H_')[-1]

                    # Map movement suffix to phase group
                    if movement_suffix in ['right', 'straight']:
                        movement_type = 'straight_right'
                    elif movement_suffix in ['left', 'uturn']:
                        movement_type = 'left_uturn'
                    else:
                        # Fallback to turn angle calculation
                        from_direction = get_edge_direction(from_edge)
                        to_direction = get_edge_direction(to_edge)
                        turn_angle = calculate_turn_angle(from_direction, to_direction)
                        movement_type = classify_movement(turn_angle)

                    # Determine approach orientation from base edge name
                    approach_orientation = get_edge_orientation(from_edge)
                    turn_angle = 0.0  # Not needed for multi-head edges
                    from_direction = get_edge_direction(from_edge)
                    to_direction = 'unknown'
                else:
                    # Original single-head edge logic
                    # Determine edge directions from junction IDs
                    from_direction = get_edge_direction(from_edge)
                    to_direction = get_edge_direction(to_edge)

                    # Calculate turn angle
                    turn_angle = calculate_turn_angle(from_direction, to_direction)

                    # Classify movement type based on turn angle
                    movement_type = classify_movement(turn_angle)

                    # Determine approach orientation (N/S or E/W)
                    approach_orientation = get_edge_orientation(from_edge)

                # DEBUG: Print orientation for B1 connections
                if tl_id == 'B1':
                    print(f"  DEBUG [{link_index}] {from_edge}: orientation={approach_orientation}, movement={movement_type}")

                connections_db[tl_id].append({
                    'from': from_edge,
                    'to': to_edge,
                    'link_index': link_index,
                    'from_direction': from_direction,
                    'to_direction': to_direction,
                    'turn_angle': turn_angle,
                    'movement_type': movement_type,
                    'approach_orientation': approach_orientation
                })

    # Rebuild traffic light phases for each intersection
    for tl_logic in tll_root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        connections = connections_db.get(tl_id, [])

        if not connections:
            continue

        # Group connections by approach direction and movement type
        ns_straight_right = []  # North/South straight+right
        ns_left_uturn = []      # North/South left+u-turn
        ew_straight_right = []  # East/West straight+right
        ew_left_uturn = []      # East/West left+u-turn

        for conn in connections:
            approach_orientation = conn['approach_orientation']
            movement = conn['movement_type']
            link_idx = conn['link_index']

            # Group by approach orientation and movement type
            if approach_orientation == 'NS':
                if movement == 'straight_right':
                    ns_straight_right.append(link_idx)
                else:
                    ns_left_uturn.append(link_idx)
            elif approach_orientation == 'EW':
                if movement == 'straight_right':
                    ew_straight_right.append(link_idx)
                else:
                    ew_left_uturn.append(link_idx)

        # DEBUG: Print grouping results for B1
        if tl_id == 'B1':
            print(f"\n=== B1 Grouping Results ===")
            print(f"NS straight_right: {ns_straight_right}")
            print(f"NS left_uturn: {ns_left_uturn}")
            print(f"EW straight_right: {ew_straight_right}")
            print(f"EW left_uturn: {ew_left_uturn}")

        # Determine state string length (number of connections)
        num_links = max([c['link_index'] for c in connections]) + 1

        # Build phase states
        def build_state_string(num_links: int, green_link_indices: list) -> str:
            """Build SUMO traffic light state string"""
            state = ['r'] * num_links
            for idx in green_link_indices:
                if 0 <= idx < num_links:
                    state[idx] = 'G'
            return ''.join(state)

        def convert_to_yellow(green_state: str) -> str:
            """Convert green state to yellow transition state"""
            return green_state.replace('G', 'y').replace('g', 'y')

        phases = []

        # Phase 1: N/S straight+right green
        state1 = build_state_string(num_links, ns_straight_right)
        phases.append({'duration': 1, 'state': state1})  # Placeholder, recalculated by convert_to_green_only_phases()

        # Phase 2: N/S left+u-turn green
        state2 = build_state_string(num_links, ns_left_uturn)
        phases.append({'duration': 1, 'state': state2})  # Placeholder, recalculated by convert_to_green_only_phases()

        # Phase 3: E/W straight+right green
        state3 = build_state_string(num_links, ew_straight_right)
        phases.append({'duration': 1, 'state': state3})  # Placeholder, recalculated by convert_to_green_only_phases()

        # Phase 4: E/W left+u-turn green
        state4 = build_state_string(num_links, ew_left_uturn)
        phases.append({'duration': 1, 'state': state4})  # Placeholder, recalculated by convert_to_green_only_phases()

        # DEBUG: Print phases for B1
        if tl_id == 'B1':
            print(f"\n=== B1 Phases ===")
            print(f"Phase 1 (NS straight+right): {state1}")
            print(f"Phase 2 (NS left+uturn): {state2}")
            print(f"Phase 3 (EW straight+right): {state3}")
            print(f"Phase 4 (EW left+uturn): {state4}")

        # Filter out phases with no green lights (all 'r')
        # This happens at corner junctions that don't have all movement groups
        phases = [p for p in phases if 'G' in p['state']]

        # Replace phases in traffic light logic
        for phase in tl_logic.findall('phase'):
            tl_logic.remove(phase)

        for phase_data in phases:
            phase_elem = ET.SubElement(tl_logic, 'phase')
            phase_elem.set('duration', str(phase_data['duration']))
            phase_elem.set('state', phase_data['state'])

    # Write modified traffic light file
    ET.indent(tll_root)
    tll_tree.write(
        Path(str(CONFIG.network_tll_file)),
        encoding='UTF-8',
        xml_declaration=True
    )


def convert_traffic_lights_for_control_mode(traffic_control):
    """Configure traffic lights based on traffic control mode.

    Sets traffic light type and parameters based on the selected traffic control mode:
    - 'fixed': type="static", no minDur/maxDur (static timing)
    - 'actuated': type="actuated", minDur/maxDur bounds, gap-based detection params
    - 'tree_method': type="static", no minDur/maxDur (Tree Method overrides via TraCI)

    Must be called AFTER traffic light strategy conversions (incoming, partial_opposites)
    and AFTER convert_to_green_only_phases() (which sets equal durations).

    Args:
        traffic_control: Traffic control mode ('fixed', 'actuated', or 'tree_method')

    For actuated mode, parameters match original decentralized-traffic-bottlenecks repository:
    - max-gap: 3.0s (maximum time gap to extend phase)
    - detector-gap: 1.0s (detector placement distance)
    - passing-time: 10.0s (vehicle headway estimate)
    - freq: 300s (detector data aggregation)
    - show-detectors: true (visible in GUI)
    - minDur: 10s (minimum phase duration)
    - maxDur: 70s (maximum phase duration)
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from src.constants import (
        ACTUATED_MAX_GAP,
        ACTUATED_DETECTOR_GAP,
        ACTUATED_PASSING_TIME,
        ACTUATED_FREQ,
        ACTUATED_SHOW_DETECTORS,
        ACTUATED_MIN_DUR,
        ACTUATED_MAX_DUR
    )

    # Parse the traffic light logic file
    tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    root = tree.getroot()

    # Configure each traffic light based on control mode
    for tl_logic in root.findall('tlLogic'):
        # Clear existing param elements (if any)
        for param in tl_logic.findall('param'):
            tl_logic.remove(param)

        # Remove minDur/maxDur from all phases first (clean slate)
        for phase in tl_logic.findall('phase'):
            if 'minDur' in phase.attrib:
                del phase.attrib['minDur']
            if 'maxDur' in phase.attrib:
                del phase.attrib['maxDur']

        if traffic_control == 'actuated':
            # Actuated mode: gap-based detection with dynamic phase extensions
            tl_logic.set('type', 'actuated')

            # Add actuated control parameters
            param_max_gap = ET.Element('param')
            param_max_gap.set('key', 'max-gap')
            param_max_gap.set('value', str(ACTUATED_MAX_GAP))
            tl_logic.append(param_max_gap)

            param_detector_gap = ET.Element('param')
            param_detector_gap.set('key', 'detector-gap')
            param_detector_gap.set('value', str(ACTUATED_DETECTOR_GAP))
            tl_logic.append(param_detector_gap)

            param_passing_time = ET.Element('param')
            param_passing_time.set('key', 'passing-time')
            param_passing_time.set('value', str(ACTUATED_PASSING_TIME))
            tl_logic.append(param_passing_time)

            param_freq = ET.Element('param')
            param_freq.set('key', 'freq')
            param_freq.set('value', str(ACTUATED_FREQ))
            tl_logic.append(param_freq)

            param_show_detectors = ET.Element('param')
            param_show_detectors.set('key', 'show-detectors')
            param_show_detectors.set('value', str(ACTUATED_SHOW_DETECTORS).lower())
            tl_logic.append(param_show_detectors)

            # Add minDur and maxDur only to phases with green lights
            for phase in tl_logic.findall('phase'):
                state = phase.get('state', '')
                # Only add minDur/maxDur if phase contains green lights ('G')
                if 'G' in state:
                    phase.set('minDur', str(ACTUATED_MIN_DUR))
                    phase.set('maxDur', str(ACTUATED_MAX_DUR))

        elif traffic_control in ['fixed', 'tree_method']:
            # Fixed or Tree Method mode: static timing (no gap-based detection)
            # Fixed uses static timing directly, Tree Method overrides via TraCI
            tl_logic.set('type', 'static')
            # No additional parameters needed - phases use their duration attribute

        else:
            # Default to static for unknown modes
            tl_logic.set('type', 'static')

    # Write modified traffic light file
    ET.indent(root)
    tree.write(
        Path(str(CONFIG.network_tll_file)),
        encoding='UTF-8',
        xml_declaration=True
    )


def convert_to_green_only_phases():
    """Convert traffic lights to green-only phases with equal durations.

    Removes yellow transition phases, normalizes states, and recalculates equal phase durations.
    This ensures fair Fixed baseline and proper Tree Method optimization.

    Must be called AFTER traffic light strategy conversions (incoming, partial_opposites)
    but BEFORE convert_traffic_lights_for_control_mode().

    Transformation:
    - Removes pure yellow phases (contain 'y' but no 'G')
    - Normalizes state strings to use only 'G' (green) and 'r' (red)
    - Recalculates equal durations: CYCLE_TIME / num_phases for each junction
    - Ensures all junctions have same total cycle time (90s) regardless of phase count

    This matches the original decentralized-traffic-bottlenecks repository format
    where Tree Method only optimizes green phases and SUMO automatically inserts
    3-second yellow transitions between conflicting phases.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    # Standard cycle time (matching Tree Method interval)
    # TODO: Make this configurable via --tree-method-interval argument
    CYCLE_TIME = 90  # seconds

    # Parse TLL file
    tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    root = tree.getroot()

    # Process each traffic light
    for tl_logic in root.findall('tlLogic'):
        phases = list(tl_logic.findall('phase'))
        green_only_phases = []

        for phase in phases:
            state = phase.get('state', '')

            # Skip yellow transition phases (contain 'y' character)
            # These will be automatically inserted by SUMO at 3 seconds
            if 'y' in state:
                continue

            # Normalize state to green-only format (only 'G' and 'r')
            # Convert lowercase 'g' (permissive green) to uppercase 'G'
            # Convert everything else to 'r' (red)
            normalized_state = ''
            for char in state:
                if char in ['G', 'g']:  # Both uppercase and lowercase green
                    normalized_state += 'G'
                else:
                    # Convert 'r' and any other character to 'r'
                    normalized_state += 'r'

            green_only_phases.append({
                'state': normalized_state
            })

        # Calculate equal duration for all phases
        # This ensures fair Fixed baseline: all movements get equal green time
        num_phases = len(green_only_phases)
        if num_phases > 0:
            equal_duration = CYCLE_TIME / num_phases
        else:
            equal_duration = CYCLE_TIME  # Fallback (shouldn't happen)

        # Replace phases with equal durations
        for phase in phases:
            tl_logic.remove(phase)

        for phase_data in green_only_phases:
            phase_elem = ET.SubElement(tl_logic, 'phase')
            phase_elem.set('duration', str(equal_duration))
            phase_elem.set('state', phase_data['state'])

    # Write modified traffic light file
    ET.indent(root)
    tree.write(Path(str(CONFIG.network_tll_file)), encoding='UTF-8', xml_declaration=True)


def generate_grid_network(seed, dimension, block_size_m, junctions_to_remove_input, lane_count_arg, traffic_light_strategy="opposites", traffic_control=None):
    try:
        is_list, junction_ids, count = parse_junctions_to_remove(
            junctions_to_remove_input)

        if count > 0:
            # generate the full grid network first
            generate_full_grid_network(
                dimension, block_size_m, lane_count_arg, "opposites", traffic_control)

            if is_list:
                # use the provided list of junction IDs
                junctions_to_remove = junction_ids
            else:
                # pick random junctions to remove
                junctions_to_remove = pick_random_junction_ids(
                    seed, count, dimension)

            # remove the selected junctions
            wipe_crossing(junctions_to_remove)
        else:
            generate_full_grid_network(
                dimension, block_size_m, lane_count_arg, "opposites", traffic_control)

        # NOTE: convert_to_green_only_phases() and convert_traffic_lights_for_control_mode()
        # are now called AFTER traffic light strategy is applied in standard_pipeline.py
        # This ensures proper phase durations for all strategies (opposites, incoming, partial_opposites)

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netgenerate execution:", e.stderr)

    if not os.path.exists(CONFIG.network_file):
        raise Exception(
            f"Error: Network file '{CONFIG.network_file}' was not created.")


def classify_edges(dimension: int) -> Tuple[List[str], List[str]]:
    """
    Classify edges into boundary and inner edges for route pattern generation.

    Args:
        dimension: Grid dimension (e.g., 5 for 5x5 grid)

    Returns:
        Tuple of (boundary_edges, inner_edges) containing edge IDs

    Note:
        Only examines tail edges (without _H_s or _H_node suffixes)
        Boundary edges require BOTH endpoints to be on grid boundary

    Example for 5x5 grid:
        - Boundary edges: A0A1, A1A0, A0B0, B0A0, E0E1, E1E0, A4B4, B4A4, E3E4, E4E3, etc.
        - Inner edges: B1B2, B2B1, B1C1, C1B1, E2D2, D2E2, B1A1, A1B1, etc.
    """
    boundary_edges = []
    inner_edges = []

    # Generate row and column labels
    row_labels = [chr(ord('A') + i) for i in range(dimension)]

    # Define boundary detection patterns
    # For a dimension x dimension grid (0-indexed):
    # - First row: A (index 0)
    # - Last row: dimension-1 (e.g., E for 5x5)
    # - First column: 0
    # - Last column: dimension-1

    first_row = row_labels[0]  # 'A'
    last_row = row_labels[dimension - 1]  # e.g., 'E' for 5x5
    first_col = 0
    last_col = dimension - 1

    # Generate all possible edge IDs and classify them
    for from_row in range(dimension):
        for from_col in range(dimension):
            from_node = f"{row_labels[from_row]}{from_col}"

            # Check all possible connections (horizontal and vertical)
            directions = [
                (0, 1),   # right
                (0, -1),  # left
                (1, 0),   # down
                (-1, 0)   # up
            ]

            for dr, dc in directions:
                to_row = from_row + dr
                to_col = from_col + dc

                # Check if destination is within grid bounds
                if 0 <= to_row < dimension and 0 <= to_col < dimension:
                    to_node = f"{row_labels[to_row]}{to_col}"
                    edge_id = f"{from_node}{to_node}"

                    # Classify as boundary or inner edge
                    # An edge is boundary if BOTH endpoints are on the grid boundary
                    # This ensures pass-through routes truly connect boundary-to-boundary
                    from_is_boundary = (from_row == 0 or from_row == dimension - 1 or
                                        from_col == 0 or from_col == dimension - 1)
                    to_is_boundary = (to_row == 0 or to_row == dimension - 1 or
                                      to_col == 0 or to_col == dimension - 1)

                    if from_is_boundary and to_is_boundary:
                        boundary_edges.append(edge_id)
                    else:
                        inner_edges.append(edge_id)

    # Remove duplicates and sort for consistency
    boundary_edges = sorted(list(set(boundary_edges)))
    inner_edges = sorted(list(set(inner_edges)))

    return boundary_edges, inner_edges


def fix_traffic_lights_for_rl_control() -> None:
    """
    Fix traffic light configuration to enable RL control.

    Changes all traffic lights from 'static' type to 'actuated' type and
    sets long durations that RL agent can override via TraCI commands.
    This is required for proper RL traffic signal control.
    """
    tll_file = CONFIG.network_tll_file

    if not tll_file.exists():
        print(f"Traffic light file {tll_file} not found, skipping RL fix")
        return

    # print("Fixing traffic light configuration for RL control...")

    try:
        tree = ET.parse(tll_file)
        root = tree.getroot()

        # Fix all traffic light logics
        for tl_logic in root.findall("tlLogic"):
            # Keep type as "static" but with minimal durations for RL override
            # Static type allows setPhaseDuration() to work when called before phase starts
            tl_logic.set("type", "static")

            # Fix all phase durations to be very short so RL can take control immediately
            for phase in tl_logic.findall("phase"):
                # Set very short duration (1 second) - RL will override before phase completes
                # This allows RL to take control at the start of each phase
                phase.set("duration", "1")

        # Save the modified file
        tree.write(tll_file, encoding="UTF-8", xml_declaration=True)
        # print("✅ Traffic light configuration fixed for RL control")
        # print("   - Type: 'static' (compatible with setPhaseDuration)")
        # print("   - Changed durations: hardcoded → 1s (RL takes control immediately)")

    except Exception as e:
        print(f"❌ Error fixing traffic lights for RL control: {e}")
        raise
