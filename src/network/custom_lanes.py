"""
Custom edge lane definition system for SUMO traffic generator.

This module provides functionality to manually specify lane configurations
for specific edges in synthetic grid networks, overriding automatic lane
assignment algorithms. It uses a shared code architecture to reuse existing
spatial logic functions while providing complete bidirectional impact management.
"""

import logging
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Set, Tuple, Optional

from src.config import CONFIG, CustomLaneConfig
from src.network.split_edges_with_lanes import (
    analyze_movements_from_connections,
    calculate_movement_angles,
    assign_lanes_by_angle,
    extract_edge_coordinates
)


def apply_custom_lane_configs(custom_lane_config: CustomLaneConfig) -> None:
    """
    Apply custom lane configurations using shared spatial logic.

    This is the main entry point that orchestrates the complete custom lane
    application process with bidirectional impact management.

    Args:
        custom_lane_config: Configuration object with parsed custom lane specifications
    """
    if not custom_lane_config.edge_configs:
        return  # No custom configurations to apply

    print("Applying custom lane configurations...")

    # Phase 1: Load and analyze current XML files
    print("  Phase 1: Loading XML files and analyzing current network structure...")
    nod_tree = ET.parse(CONFIG.network_nod_file)
    edg_tree = ET.parse(CONFIG.network_edg_file)
    con_tree = ET.parse(CONFIG.network_con_file)
    tll_tree = ET.parse(CONFIG.network_tll_file)

    nod_root = nod_tree.getroot()
    edg_root = edg_tree.getroot()
    con_root = con_tree.getroot()
    tll_root = tll_tree.getroot()

    # Analyze existing movement data and edge coordinates
    movement_data = analyze_movements_from_connections(con_root)
    edge_coords = extract_edge_coordinates(edg_root, nod_root)

    # Phase 2: Calculate bidirectional impact scope
    print("  Phase 2: Calculating bidirectional impact scope...")
    affected_junctions = _calculate_affected_junctions(
        custom_lane_config.edge_configs)

    # Phase 3: Update XML files with complete deletion/regeneration
    print("  Phase 3: Updating XML files...")

    # 3a. Update edges file (modify numLanes)
    _update_edges_file(edg_root, custom_lane_config)

    # 3b. Complete regeneration of connections file
    _regenerate_connections_file(
        con_root, edg_root, custom_lane_config, movement_data, edge_coords)

    # 3c. Complete regeneration of traffic lights file
    _regenerate_traffic_lights_file(tll_root, con_root, affected_junctions)

    # Phase 4: Write updated files
    print("  Phase 4: Writing updated XML files...")
    _write_xml_files(nod_tree, edg_tree, con_tree, tll_tree)

    print(
        f"Successfully applied custom lane configurations for {len(custom_lane_config.edge_configs)} edges")


def _calculate_affected_junctions(edge_configs: Dict[str, Dict]) -> Set[str]:
    """
    Calculate all junctions affected by custom lane configurations.

    Custom lane changes have bidirectional impact - they affect both the
    downstream junction (where the edge ends) and upstream junction (where it starts).
    """
    affected_junctions = set()

    for edge_id in edge_configs.keys():
        # Extract junction IDs from edge ID (e.g., A1B1 -> A1 and B1)
        match = re.match(r'^([A-Z]+\d+)([A-Z]+\d+)$', edge_id)
        if match:
            upstream_junction = match.group(1)    # A1 (where edge starts)
            downstream_junction = match.group(2)  # B1 (where edge ends)

            affected_junctions.add(upstream_junction)
            affected_junctions.add(downstream_junction)

    return affected_junctions


def _update_edges_file(edg_root, custom_lane_config: CustomLaneConfig) -> None:
    """
    Update edges file by modifying numLanes attribute for customized edges.

    Only the numLanes attribute is modified. All other attributes (speed, priority, shape)
    remain unchanged to preserve existing network geometry and traffic characteristics.
    """
    for edge in edg_root.findall("edge"):
        edge_id = edge.get("id")

        if not edge_id:
            continue

        # Handle both tail and head segments
        # Remove head segment suffix
        base_edge_id = edge_id.replace("_H_s", "")

        if custom_lane_config.has_custom_config(base_edge_id):
            if edge_id.endswith("_H_s"):
                # Head segment - update based on movement specifications
                movements = custom_lane_config.get_movements(base_edge_id)
                if movements is not None:
                    if not movements:  # Dead-end case
                        head_lanes = 0
                    else:
                        head_lanes = sum(movements.values())
                    edge.set("numLanes", str(head_lanes))
            else:
                # Tail segment - update based on tail specification only
                tail_lanes = custom_lane_config.get_tail_lanes(base_edge_id)
                if tail_lanes is not None:
                    edge.set("numLanes", str(tail_lanes))


def _regenerate_connections_file(con_root, edg_root, custom_lane_config: CustomLaneConfig,
                                 movement_data: Dict[str, Dict],
                                 edge_coords: Dict[str, Tuple[float, float, float, float]]) -> None:
    """
    Complete regeneration of connections file using deletion/regeneration strategy.

    This function:
    1. Deletes ALL connections for affected edges (bidirectional)
    2. Regenerates internal tail→head connections
    3. Regenerates head→downstream connections using shared spatial logic + custom overrides
    4. Regenerates upstream→tail connections with lane redistribution
    """
    # Step 1: Delete all connections for affected edges
    _delete_connections_for_affected_edges(con_root, custom_lane_config)

    # Step 2: Regenerate connections for each customized edge
    for edge_id, edge_config in custom_lane_config.edge_configs.items():
        _regenerate_edge_connections(
            con_root, edg_root, edge_id, edge_config, movement_data, edge_coords)

    # Step 3: Validate all connections have valid lane indices
    _validate_all_connections(con_root, edg_root)


def _delete_connections_for_affected_edges(con_root, custom_lane_config: CustomLaneConfig) -> None:
    """
    Delete connections that need regeneration due to custom configurations.

    SURGICAL DELETION STRATEGY - only delete connections directly related to customized edges:
    - Delete ALL connections FROM customized edges (both tail and head segments) 
    - Delete connections TO customized edges (only tail segments, not affecting other edges)
    - Preserve internal connections for non-customized edges (e.g., A1A0 -> A1A0_H_s)
    """
    connections_to_remove = []

    for connection in con_root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")

        # Remove connections FROM customized edges (both tail and head segments)
        if from_edge:
            base_from_edge = from_edge.replace("_H_s", "")
            if custom_lane_config.has_custom_config(base_from_edge):
                connections_to_remove.append(connection)
                continue

        # Remove connections TO customized edges (only tail segments to avoid affecting other edges)
        if to_edge:
            # Only delete connections TO the tail segment of customized edges
            # This preserves internal connections like A1A0 -> A1A0_H_s for non-customized edges
            if custom_lane_config.has_custom_config(to_edge):
                connections_to_remove.append(connection)

    # Remove connections
    for connection in connections_to_remove:
        con_root.remove(connection)


def _regenerate_edge_connections(con_root, edg_root, edge_id: str, edge_config: Dict,
                                 movement_data: Dict[str, Dict],
                                 edge_coords: Dict[str, Tuple[float, float, float, float]]) -> None:
    """Regenerate all connections for a single customized edge."""

    # Get configuration
    tail_lanes = edge_config.get('tail_lanes')
    custom_movements = edge_config.get('movements')

    # Determine actual lane counts
    if edge_id in movement_data:
        original_movements = movement_data[edge_id]['movements']
        original_total_lanes = movement_data[edge_id]['total_movement_lanes']
    else:
        original_movements = []
        original_total_lanes = 1

    # Calculate head lanes
    if custom_movements is not None:
        # Custom head movements specified
        if not custom_movements:  # Dead-end case
            head_lanes = 0
            movements_to_use = []
        else:
            head_lanes = sum(custom_movements.values())
            # Convert custom movements to movement format
            movements_to_use = []
            for to_edge, lane_count in custom_movements.items():
                movements_to_use.append({
                    'to_edge': to_edge,
                    'num_lanes': lane_count,
                    'from_lanes': list(range(lane_count))  # Will be reassigned
                })
    else:
        # Preserve original movements
        head_lanes = max(original_total_lanes, tail_lanes if tail_lanes else 1)
        movements_to_use = original_movements

    # Use original tail lanes if not specified (head-only configuration)
    if tail_lanes is None:
        # For head-only configurations, preserve original tail lane count
        tail_lanes = _get_edge_lane_count(edg_root, edge_id)

    # Generate internal tail→head connections
    _generate_internal_connections(con_root, edge_id, tail_lanes, head_lanes)

    # Generate head→downstream connections using shared spatial logic
    if movements_to_use:
        _generate_downstream_connections(con_root, edg_root, edge_id, movements_to_use,
                                         head_lanes, edge_coords)

    # Generate upstream→tail connections (bidirectional impact)
    _generate_upstream_connections(con_root, edge_id, tail_lanes)


def _generate_internal_connections(con_root, edge_id: str, tail_lanes: int, head_lanes: int) -> None:
    """Generate internal tail→head connections with proper lane distribution."""
    if head_lanes == 0:  # Dead-end case
        return

    tail_segment_id = edge_id
    head_segment_id = f"{edge_id}_H_s"

    if tail_lanes <= head_lanes:
        # Distribute tail lanes evenly across head lanes
        lanes_per_head = head_lanes // tail_lanes
        extra_lanes = head_lanes % tail_lanes

        head_lane_idx = 0
        for tail_lane in range(tail_lanes):
            # Each tail lane connects to one or more head lanes
            connections_for_this_tail = lanes_per_head + \
                (1 if tail_lane < extra_lanes else 0)

            for _ in range(connections_for_this_tail):
                if head_lane_idx < head_lanes:
                    conn_elem = ET.SubElement(con_root, "connection")
                    conn_elem.set("from", tail_segment_id)
                    conn_elem.set("to", head_segment_id)
                    conn_elem.set("fromLane", str(tail_lane))
                    conn_elem.set("toLane", str(head_lane_idx))
                    head_lane_idx += 1
    else:
        # More tail lanes than head lanes - multiple tail lanes per head lane
        tail_per_head = tail_lanes // head_lanes
        extra_tails = tail_lanes % head_lanes

        tail_lane_idx = 0
        for head_lane in range(head_lanes):
            connections_for_this_head = tail_per_head + \
                (1 if head_lane < extra_tails else 0)

            for _ in range(connections_for_this_head):
                if tail_lane_idx < tail_lanes:
                    conn_elem = ET.SubElement(con_root, "connection")
                    conn_elem.set("from", tail_segment_id)
                    conn_elem.set("to", head_segment_id)
                    conn_elem.set("fromLane", str(tail_lane_idx))
                    conn_elem.set("toLane", str(head_lane))
                    tail_lane_idx += 1


def _generate_downstream_connections(con_root, edg_root, edge_id: str, movements: List[Dict],
                                     head_lanes: int, edge_coords: Dict) -> None:
    """Generate head→downstream connections using shared spatial logic."""
    if not movements or head_lanes == 0:
        return

    head_segment_id = f"{edge_id}_H_s"

    # Use shared spatial logic to assign lanes
    if edge_id in edge_coords:
        from_edge_coords = edge_coords[edge_id]

        # Calculate actual turn angles for movements
        movements_with_angles = calculate_movement_angles(
            from_edge_coords, movements, edge_coords
        )

        # Assign lanes based on spatial logic
        movement_to_head_lanes = assign_lanes_by_angle(
            movements_with_angles, head_lanes)

        # Create connections based on spatial assignments
        for movement in movements_with_angles:
            to_edge = movement['to_edge']
            if to_edge in movement_to_head_lanes:
                assigned_head_lanes = movement_to_head_lanes[to_edge]

                for head_lane_idx, head_lane in enumerate(assigned_head_lanes):
                    # Get actual lane count for destination edge
                    destination_lanes = _get_edge_lane_count(edg_root, to_edge)
                    # Distribute connections across available destination lanes
                    target_lane = head_lane_idx % max(1, destination_lanes)

                    conn_elem = ET.SubElement(con_root, "connection")
                    conn_elem.set("from", head_segment_id)
                    conn_elem.set("to", to_edge)
                    conn_elem.set("fromLane", str(head_lane))
                    conn_elem.set("toLane", str(target_lane))


def _generate_upstream_connections(con_root, edge_id: str, tail_lanes: int) -> None:
    """Generate upstream→tail connections with proper lane redistribution."""
    # Find all edges that connect TO this edge by scanning all connections
    upstream_connections = []

    for conn_elem in con_root.findall("connection"):
        to_edge = conn_elem.get("to")
        if to_edge == edge_id:
            from_edge = conn_elem.get("from")
            from_lane = int(conn_elem.get("fromLane"))
            upstream_connections.append({
                'from_edge': from_edge,
                'from_lane': from_lane,
                'connection_element': conn_elem
            })

    if not upstream_connections:
        return  # No upstream connections to regenerate

    # Group by upstream edge to handle multiple lanes from same edge
    upstream_edges = {}
    for conn in upstream_connections:
        from_edge = conn['from_edge']
        if from_edge not in upstream_edges:
            upstream_edges[from_edge] = []
        upstream_edges[from_edge].append(conn)

    # Remove existing upstream connections (they were already identified above)
    for conn in upstream_connections:
        conn_elem = conn['connection_element']
        if conn_elem.getparent() is not None:
            con_root.remove(conn_elem)

    # Regenerate upstream connections with lane redistribution
    for from_edge, edge_connections in upstream_edges.items():
        # Sort connections by from_lane for consistent ordering
        edge_connections.sort(key=lambda x: x['from_lane'])

        # Create new connections with even distribution
        for i, conn in enumerate(edge_connections):
            # Distribute upstream lanes evenly across available tail lanes
            target_tail_lane = i % tail_lanes

            # Create new connection element
            new_conn = ET.SubElement(con_root, "connection")
            new_conn.set("from", from_edge)
            new_conn.set("to", edge_id)
            new_conn.set("fromLane", str(conn['from_lane']))
            new_conn.set("toLane", str(target_tail_lane))


def _regenerate_traffic_lights_file(tll_root, con_root, affected_junctions: Set[str]) -> None:
    """
    Complete regeneration of traffic light logic for affected junctions.

    This function:
    1. Deletes traffic light logic for affected junctions
    2. Deletes traffic light connections for affected junctions  
    3. Recalculates connection counts and state strings
    4. Recreates traffic light logic with new state strings
    5. Recreates traffic light connections with sequential linkIndex
    """
    # Step 1: Delete existing traffic light logic for affected junctions
    tl_logics_to_remove = []
    for tl_logic in tll_root.findall("tlLogic"):
        junction_id = tl_logic.get("id")
        if junction_id in affected_junctions:
            tl_logics_to_remove.append(tl_logic)

    for tl_logic in tl_logics_to_remove:
        tll_root.remove(tl_logic)

    # Step 2: Delete existing traffic light connections for affected junctions
    tl_connections_to_remove = []
    for connection in tll_root.findall("connection"):
        tl_id = connection.get("tl")
        if tl_id in affected_junctions:
            tl_connections_to_remove.append(connection)

    for connection in tl_connections_to_remove:
        tll_root.remove(connection)

    # Step 3: Recreate traffic light logic and connections for each affected junction
    for junction_id in affected_junctions:
        _recreate_junction_traffic_light(tll_root, con_root, junction_id)


def _recreate_junction_traffic_light(tll_root, con_root, junction_id: str) -> None:
    """Recreate complete traffic light logic for a single junction."""

    # Find all connections going into this junction (from head segments ending at junction)
    junction_connections = []
    for connection in con_root.findall("connection"):
        from_edge = connection.get("from")
        # Include connections FROM head segments that end at this junction
        if from_edge and from_edge.endswith("_H_s"):
            # Extract base edge ID (e.g., A1B1_H_s → A1B1)
            base_edge = from_edge.replace("_H_s", "")
            # Check if this head segment ends at the target junction (e.g., A1B1 ends at B1)
            if base_edge.endswith(junction_id):
                junction_connections.append(connection)

    if not junction_connections:
        return  # No connections to this junction

    # Calculate total connection count
    connection_count = len(junction_connections)

    # Generate traffic light state strings based on connection count
    # This is a simplified 4-phase system - more sophisticated logic could be added
    if connection_count > 0:
        # Create 4-phase traffic light logic
        state_green_ns = "G" * (connection_count // 2) + \
            "r" * (connection_count - connection_count // 2)
        state_yellow_ns = "y" * (connection_count // 2) + \
            "r" * (connection_count - connection_count // 2)
        state_green_ew = "r" * (connection_count // 2) + \
            "G" * (connection_count - connection_count // 2)
        state_yellow_ew = "r" * (connection_count // 2) + \
            "y" * (connection_count - connection_count // 2)

        # Create traffic light logic element
        tl_logic = ET.SubElement(tll_root, "tlLogic")
        tl_logic.set("id", junction_id)
        # Changed from "static" to allow TraCI control
        tl_logic.set("type", "actuated")
        tl_logic.set("programID", "0")
        tl_logic.set("offset", "0")

        # Add phases with long default durations that RL can override
        phase1 = ET.SubElement(tl_logic, "phase")
        phase1.set("duration", "1000")  # Long duration - RL will override this
        phase1.set("state", state_green_ns)

        phase2 = ET.SubElement(tl_logic, "phase")
        phase2.set("duration", "3")  # Keep short yellow phases
        phase2.set("state", state_yellow_ns)

        phase3 = ET.SubElement(tl_logic, "phase")
        phase3.set("duration", "1000")  # Long duration - RL will override this
        phase3.set("state", state_green_ew)

        phase4 = ET.SubElement(tl_logic, "phase")
        phase4.set("duration", "3")  # Keep short yellow phases
        phase4.set("state", state_yellow_ew)

    # Create traffic light connections with sequential linkIndex
    for link_idx, connection in enumerate(junction_connections):
        tl_connection = ET.SubElement(tll_root, "connection")
        tl_connection.set("from", connection.get("from"))
        tl_connection.set("to", connection.get("to"))
        tl_connection.set("fromLane", connection.get("fromLane"))
        tl_connection.set("toLane", connection.get("toLane"))
        tl_connection.set("tl", junction_id)
        tl_connection.set("linkIndex", str(link_idx))


def _get_edge_lane_count(edg_root, edge_id: str) -> int:
    """Get the current lane count for an edge from the edges file."""
    for edge in edg_root.findall("edge"):
        if edge.get("id") == edge_id:
            return int(edge.get("numLanes", "1"))
    return 1  # Default fallback


def _validate_all_connections(con_root, edg_root):
    """Validate all connections have valid lane indices."""
    for connection in con_root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        from_lane = int(connection.get("fromLane", "0"))
        to_lane = int(connection.get("toLane", "0"))

        # Get actual lane counts
        from_lanes = _get_edge_lane_count(edg_root, from_edge)
        to_lanes = _get_edge_lane_count(edg_root, to_edge)

        # Validate indices
        if from_lane >= from_lanes:
            raise ValueError(
                f"Invalid fromLane {from_lane} for edge {from_edge} (has {from_lanes} lanes)")
        if to_lane >= to_lanes:
            raise ValueError(
                f"Invalid toLane {to_lane} for edge {to_edge} (has {to_lanes} lanes)")


def _write_xml_files(nod_tree, edg_tree, con_tree, tll_tree) -> None:
    """Write all updated XML files with proper formatting."""

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


# Factory function for pipeline integration
def create_custom_lane_config_from_args(args) -> Optional[CustomLaneConfig]:
    """
    Create CustomLaneConfig from command line arguments.

    This function handles both --custom_lanes and --custom_lanes_file arguments,
    with proper precedence and validation.
    """
    if hasattr(args, 'custom_lanes') and args.custom_lanes:
        return CustomLaneConfig.parse_custom_lanes(args.custom_lanes)
    elif hasattr(args, 'custom_lanes_file') and args.custom_lanes_file:
        return CustomLaneConfig.parse_custom_lanes_file(args.custom_lanes_file)
    else:
        return None


def execute_custom_lanes(args) -> None:
    """Execute custom lane configuration application."""
    logger = logging.getLogger(__name__)

    # Create custom lane configuration from arguments
    custom_lane_config = create_custom_lane_config_from_args(args)

    if custom_lane_config and custom_lane_config.edge_configs:
        logger.info("Applying custom lane configurations...")
        apply_custom_lane_configs(custom_lane_config)
        logger.info(
            f"Successfully applied custom lane configurations for {len(custom_lane_config.edge_configs)} edges")
