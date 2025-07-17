import xml.etree.ElementTree as ET
from src.config import CONFIG
from src.network.edge_attrs import load_zones_data, find_adjacent_zones


def is_perimeter_edge(edge_id: str, edg_root) -> bool:
    """Check if edge is on the perimeter of the grid"""
    # Extract junction coordinates from edge endpoints
    edge = edg_root.find(f"edge[@id='{edge_id}']")
    if edge is None:
        return False

    from_node = edge.get('from')
    to_node = edge.get('to')

    # Simple heuristic: if either node contains A, C, 0, or 2 (grid boundaries)
    # This assumes a 3x3 grid with junctions A0, A1, A2, B0, B1, B2, C0, C1, C2
    boundary_markers = ['A', 'C', '0', '2']
    return any(marker in from_node or marker in to_node for marker in boundary_markers)


def calculate_lane_count_realistic(edge_id: str, edg_root, block_size_m: int = 200) -> int:
    """Calculate lane count based on land use zones and edge characteristics"""
    zones_data = load_zones_data()
    adjacent_zones = find_adjacent_zones(
        edge_id, zones_data, edg_root, block_size_m)

    if not adjacent_zones:
        # Fallback: return moderate lane count if no zone data
        return 2

    # Calculate traffic demand score based on adjacent zones
    demand_score = 0.0

    # Traffic generation weights by land use type
    land_use_weights = {
        'Mixed': 1,                    # Highest traffic generation
        'Employment': 1,               # High peak traffic
        'Entertainment/Retail': 1,    # High commercial traffic
        'Public Buildings': 0.5,        # Moderate institutional traffic
        'Residential': 0.3,              # Moderate residential traffic
        'Public Open Space': 0.2         # Lower recreational traffic
    }

    for i, zone in enumerate(adjacent_zones):
        land_use = zone.get('land_use', 'Residential')
        attractiveness = zone.get('attractiveness', 0.5)

        # Weight by land use and attractiveness
        weight = land_use_weights.get(land_use, 1.5)
        zone_contribution = weight * attractiveness
        demand_score += zone_contribution

    # Apply edge position modifier
    if is_perimeter_edge(edge_id, edg_root):
        demand_score *= 0.8  # Perimeter edges typically have less internal traffic

    # Convert demand score to lane count (1-3 lanes)
    if demand_score < 1.0:
        lanes = 1
        return lanes
    elif demand_score < 2.5:
        lanes = 2
        return lanes
    else:
        lanes = 3
        return lanes


def calculate_lane_count(edge_id: str, algorithm: str, rng, min_lanes: int, max_lanes: int, block_size_m: int = 200) -> int:
    """Calculate lane count based on specified algorithm"""
    if algorithm == 'random':
        return rng.randint(min_lanes, max_lanes)
    elif algorithm == 'realistic':
        # Parse edge files to get edge root
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        realistic_lanes = calculate_lane_count_realistic(
            edge_id, edg_root, block_size_m)
        # Clamp to min/max bounds
        return max(min_lanes, min(max_lanes, realistic_lanes))
    elif algorithm.isdigit():
        # Fixed number of lanes
        fixed_lanes = int(algorithm)
        return max(min_lanes, min(max_lanes, fixed_lanes))
    else:
        # Default to realistic if unknown algorithm
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        realistic_lanes = calculate_lane_count_realistic(edge_id, edg_root)
        return max(min_lanes, min(max_lanes, realistic_lanes))
