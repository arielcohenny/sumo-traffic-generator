import xml.etree.ElementTree as ET
from src.config import CONFIG
from src.constants import MIN_LANES_FOR_TL_STRATEGY
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


def calculate_lane_count_realistic(edge_id: str, edg_root, block_size_m: int = 200, traffic_light_strategy: str = "opposites") -> int:
    """Calculate lane count based on land use zones and edge characteristics

    Args:
        edge_id: ID of the edge
        edg_root: Root element of edge XML
        block_size_m: Block size in meters
        traffic_light_strategy: Traffic light strategy (affects minimum lanes)

    Returns:
        Lane count (1-3 for opposites/incoming, 2-3 for partial_opposites)
    """
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
        'Entertainment/Retail': 1,    # High traffic
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

    # Enforce minimum based on traffic light strategy
    min_lanes = MIN_LANES_FOR_TL_STRATEGY.get(traffic_light_strategy, 1)

    # Convert demand score to lane count (enforce min_lanes to 3)
    if demand_score < 1.0:
        lanes = max(min_lanes, 1)
    elif demand_score < 2.5:
        lanes = max(min_lanes, 2)
    else:
        lanes = 3

    return lanes


def calculate_lane_count(edge_id: str, algorithm: str, rng, min_lanes: int, max_lanes: int, block_size_m: int = 200, traffic_light_strategy: str = "opposites") -> int:
    """Calculate lane count based on specified algorithm

    Args:
        edge_id: ID of the edge
        algorithm: Lane count algorithm ('random', 'realistic', or fixed number)
        rng: Random number generator
        min_lanes: Minimum number of lanes
        max_lanes: Maximum number of lanes
        block_size_m: Block size in meters
        traffic_light_strategy: Traffic light strategy (affects minimum lanes)

    Returns:
        Lane count for the edge
    """
    # Enforce minimum based on traffic light strategy
    strategy_min_lanes = MIN_LANES_FOR_TL_STRATEGY.get(traffic_light_strategy, 1)
    effective_min = max(min_lanes, strategy_min_lanes)

    if algorithm == 'random':
        return rng.randint(effective_min, max_lanes)
    elif algorithm == 'realistic':
        # Parse edge files to get edge root
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        realistic_lanes = calculate_lane_count_realistic(
            edge_id, edg_root, block_size_m, traffic_light_strategy)
        # Clamp to min/max bounds
        return max(effective_min, min(max_lanes, realistic_lanes))
    elif algorithm.isdigit():
        # Fixed number of lanes
        fixed_lanes = int(algorithm)
        return max(effective_min, min(max_lanes, fixed_lanes))
    else:
        # Default to realistic if unknown algorithm
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        realistic_lanes = calculate_lane_count_realistic(edge_id, edg_root, block_size_m, traffic_light_strategy)
        return max(effective_min, min(max_lanes, realistic_lanes))
