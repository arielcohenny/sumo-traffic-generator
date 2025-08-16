"""
Validation functions for split edges with flow-based lane assignment.

This module provides comprehensive validation for the edge splitting and lane assignment
process to ensure proper network topology and lane connectivity.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple
from pathlib import Path


def verify_split_edges_with_flow_based_lanes(
    connections_file: str,
    edges_file: str,
    nodes_file: str
) -> None:
    """
    Comprehensive validation for split edges with flow-based lane assignment.

    Validates the XML files that are modified by split_edges_with_flow_based_lanes():
    1. Tail lanes equal original edge lanes
    2. Head lanes equal original total_movement_lanes
    3. All head lanes have exactly one outgoing direction
    4. All head lanes have incoming connections from tail lanes
    5. All tail lanes lead to head lanes
    6. Additional structural validations

    Args:
        connections_file: Path to connections file (.con.xml)
        edges_file: Path to edges file (.edg.xml)
        nodes_file: Path to nodes file (.nod.xml)
    """
    print("Starting comprehensive split edges validation...")

    # Load and parse all required files
    edge_lanes = _parse_edge_lanes_from_xml(edges_file)
    connections = _parse_connections(connections_file)
    movements = _analyze_movements_from_connections(connections_file)

    # Extract edge information from XML files instead of network
    edge_data = _extract_edge_info_from_xml(edge_lanes, movements)

    # Track validation results
    validation_errors = []
    edges_validated = 0

    # Validate each edge
    for edge_id, edge_info in edge_data.items():
        try:
            _validate_single_edge(
                edge_id, edge_info, edge_lanes, connections, movements, validation_errors
            )
            edges_validated += 1
        except Exception as e:
            validation_errors.append(
                f"Error validating edge {edge_id}: {str(e)}")

    # Additional structural validations
    _validate_network_structure(
        edge_data, edge_lanes, connections, validation_errors)

    # Report results
    if validation_errors:
        print(f"\n❌ VALIDATION FAILED: {len(validation_errors)} errors found:")
        for error in validation_errors:
            print(f"  - {error}")
        raise ValueError(
            f"Split edges validation failed with {len(validation_errors)} errors")


def _extract_edge_info_from_xml(edge_lanes: Dict[str, int], movements: Dict[str, Dict]) -> Dict:
    """
    Extract edge information from edges XML file.

    Args:
        edge_lanes: Dictionary of edge_id -> lane count
        movements: Dictionary of edge_id -> movement information

    Returns:
        Dictionary of edge_id -> edge info (with 'lanes' key for original lane count)
    """
    edge_data = {}

    # Find all base edges (non-head edges)
    base_edges = set()
    for edge_id in edge_lanes.keys():
        if not edge_id.endswith('_H_s') and not edge_id.startswith(':'):
            base_edges.add(edge_id)

    # Create edge info for each base edge
    for edge_id in base_edges:
        if edge_id in edge_lanes:
            edge_data[edge_id] = {
                'id': edge_id,
                # Original lane count from tail segment
                'lanes': edge_lanes[edge_id],
                'heads': [f"{edge_id}_H_s"] if f"{edge_id}_H_s" in edge_lanes else []
            }

    return edge_data


def _parse_edge_lanes_from_xml(edges_file: str) -> Dict[str, int]:
    """Parse edges XML file to extract lane count for each edge."""
    tree = ET.parse(edges_file)
    root = tree.getroot()

    lanes_by_edge = {}

    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        num_lanes_str = edge.get('numLanes')

        if edge_id and num_lanes_str:
            lanes_by_edge[edge_id] = int(num_lanes_str)

    return lanes_by_edge


def _parse_connections(connections_file: str) -> Dict[str, List[Dict]]:
    """Parse connections file to extract lane-to-lane connections."""
    tree = ET.parse(connections_file)
    root = tree.getroot()

    connections = {}

    for conn in root.findall('connection'):
        from_edge = conn.get('from')
        to_edge = conn.get('to')
        from_lane_str = conn.get('fromLane')
        to_lane_str = conn.get('toLane')

        # Skip connections that don't have required lane attributes
        if from_lane_str is None or to_lane_str is None:
            continue

        from_lane = int(from_lane_str)
        to_lane = int(to_lane_str)

        if from_edge not in connections:
            connections[from_edge] = []

        connections[from_edge].append({
            'to_edge': to_edge,
            'from_lane': from_lane,
            'to_lane': to_lane
        })

    return connections


def _analyze_movements_from_connections(connections_file: str) -> Dict[str, Dict]:
    """Analyze movements from connections to calculate total movement lanes."""
    tree = ET.parse(connections_file)
    root = tree.getroot()

    movements_by_edge = {}

    for conn in root.findall('connection'):
        from_edge = conn.get('from')
        to_edge = conn.get('to')
        from_lane_str = conn.get('fromLane')

        # Skip connections that don't have required lane attributes
        if from_lane_str is None:
            continue

        from_lane = int(from_lane_str)

        # Only analyze head edges (ending with _H)
        if from_edge.endswith('_H_s'):
            base_edge = from_edge[:-4]  # Remove _H_s suffix

            if base_edge not in movements_by_edge:
                movements_by_edge[base_edge] = {
                    'movements': {},
                    'total_movement_lanes': 0
                }

            if to_edge not in movements_by_edge[base_edge]['movements']:
                movements_by_edge[base_edge]['movements'][to_edge] = {
                    'from_lanes': [],
                    'num_lanes': 0
                }

            movements_by_edge[base_edge]['movements'][to_edge]['from_lanes'].append(
                from_lane)
            movements_by_edge[base_edge]['movements'][to_edge]['num_lanes'] += 1
            movements_by_edge[base_edge]['total_movement_lanes'] += 1

    return movements_by_edge


def _validate_single_edge(
    edge_id: str,
    edge_info: Dict,
    edge_lanes: Dict[str, int],
    connections: Dict[str, List[Dict]],
    movements: Dict[str, Dict],
    validation_errors: List[str]
) -> None:
    """Validate a single edge for all requirements."""

    # 1. Validate tail lanes equal original edge lanes
    original_lane_count = edge_info['lanes']
    if edge_id in edge_lanes:
        actual_tail_lanes = edge_lanes[edge_id]
        if actual_tail_lanes != original_lane_count:
            validation_errors.append(
                f"Edge {edge_id}: tail lanes ({actual_tail_lanes}) != original lanes ({original_lane_count})"
            )
    else:
        validation_errors.append(
            f"Edge {edge_id}: tail segment not found in edges file")

    # 2. Validate head lanes equal max(lane_count, total_movement_lanes)
    head_edge_id = f"{edge_id}_H_s"
    if head_edge_id in edge_lanes:
        actual_head_lanes = edge_lanes[head_edge_id]
        if edge_id in movements:
            lane_count = edge_info['lanes']
            total_movement_lanes = movements[edge_id]['total_movement_lanes']
            expected_head_lanes = max(lane_count, total_movement_lanes)
            if actual_head_lanes != expected_head_lanes:
                validation_errors.append(
                    f"Edge {edge_id}: head lanes ({actual_head_lanes}) != max(lane_count, total_movement_lanes) ({expected_head_lanes})"
                )
        else:
            validation_errors.append(f"Edge {edge_id}: no movement data found")
    else:
        validation_errors.append(
            f"Edge {edge_id}: head segment {head_edge_id} not found in edges file")

    # 3. Validate all head lanes have exactly one outgoing direction
    if head_edge_id in connections:
        head_connections = connections[head_edge_id]
        lane_connections = {}

        for conn in head_connections:
            from_lane = conn['from_lane']
            if from_lane not in lane_connections:
                lane_connections[from_lane] = []
            lane_connections[from_lane].append(conn)

        for lane_idx, lane_conns in lane_connections.items():
            if len(lane_conns) != 1:
                validation_errors.append(
                    f"Edge {edge_id}: head lane {lane_idx} has {len(lane_conns)} connections (expected 1)"
                )

    # 4. Validate all head lanes have incoming connections from tail lanes
    if head_edge_id in edge_lanes:
        head_lane_count = edge_lanes[head_edge_id]

        # Check if tail connects to head
        if edge_id in connections:
            tail_connections = connections[edge_id]
            connected_head_lanes = set()

            for conn in tail_connections:
                if conn['to_edge'] == head_edge_id:
                    connected_head_lanes.add(conn['to_lane'])

            # Check if all head lanes are connected
            expected_head_lanes = set(range(head_lane_count))
            unconnected_head_lanes = expected_head_lanes - connected_head_lanes

            if unconnected_head_lanes:
                validation_errors.append(
                    f"Edge {edge_id}: head lanes {unconnected_head_lanes} have no incoming connections from tail"
                )

    # 5. Validate all tail lanes lead to head lanes
    if edge_id in connections and edge_id in edge_lanes:
        tail_lane_count = edge_lanes[edge_id]
        tail_connections = connections[edge_id]

        connected_tail_lanes = set()
        head_connections = []

        for conn in tail_connections:
            connected_tail_lanes.add(conn['from_lane'])
            if conn['to_edge'] == head_edge_id:
                head_connections.append(conn)

        # Check if all tail lanes are connected
        expected_tail_lanes = set(range(tail_lane_count))
        unconnected_tail_lanes = expected_tail_lanes - connected_tail_lanes

        if unconnected_tail_lanes:
            validation_errors.append(
                f"Edge {edge_id}: tail lanes {unconnected_tail_lanes} have no outgoing connections"
            )

        # Check if tail connects to head
        if not head_connections:
            validation_errors.append(
                f"Edge {edge_id}: no connections from tail to head segment"
            )


def _validate_network_structure(
    edge_data: Dict,
    edge_lanes: Dict[str, int],
    connections: Dict[str, List[Dict]],
    validation_errors: List[str]
) -> None:
    """Perform additional structural validations."""

    # Check for orphaned edges
    all_edges = set(edge_data.keys())
    edges_with_lanes = set(edge_lanes.keys())
    edges_with_head_lanes = set()

    for edge_id in all_edges:
        head_edge_id = f"{edge_id}_H_s"
        if head_edge_id in edge_lanes:
            edges_with_head_lanes.add(edge_id)

    # Check for missing tail segments
    missing_tail_segments = all_edges - edges_with_lanes
    if missing_tail_segments:
        validation_errors.append(
            f"Missing tail segments in edges file: {missing_tail_segments}"
        )

    # Check for missing head segments
    missing_head_segments = all_edges - edges_with_head_lanes
    if missing_head_segments:
        validation_errors.append(
            f"Missing head segments in edges file: {missing_head_segments}"
        )

    # Check for connection consistency
    for from_edge, conns in connections.items():
        if from_edge in edge_lanes:
            from_lane_count = edge_lanes[from_edge]

            for conn in conns:
                from_lane = conn['from_lane']
                if from_lane >= from_lane_count:
                    validation_errors.append(
                        f"Connection from {from_edge} lane {from_lane} exceeds lane count ({from_lane_count})"
                    )
