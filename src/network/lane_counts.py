import xml.etree.ElementTree as ET
from src.config import CONFIG
from shapely.geometry import Point, Polygon
from src.network.edge_attrs import load_zones_data


def find_adjacent_zones(edge_id: str, zones_data: list, edg_root) -> list:
    """Find zones adjacent to a given edge"""
    if not zones_data:
        return []

    # Get edge geometry
    edge = edg_root.find(f"edge[@id='{edge_id}']")
    if edge is None or 'shape' not in edge.attrib:
        return []

    try:
        # Parse edge shape coordinates
        coords = [tuple(map(float, p.split(',')))
                  for p in edge.get('shape').split()]
        if len(coords) < 2:
            return []

        # Create a buffer around the edge line to find intersecting zones
        edge_line = Point(coords[0]).buffer(
            50).union(Point(coords[-1]).buffer(50))

        adjacent_zones = []
        for zone_feature in zones_data:
            try:
                # Create polygon from zone coordinates
                zone_coords = zone_feature['geometry']['coordinates'][0]
                zone_polygon = Polygon(zone_coords)

                # Check if edge intersects or is near the zone
                if edge_line.intersects(zone_polygon) or edge_line.distance(zone_polygon) < 10:
                    adjacent_zones.append(zone_feature['properties'])
            except (KeyError, ValueError, TypeError):
                continue

        return adjacent_zones
    except (ValueError, TypeError):
        return []


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


def calculate_lane_count_realistic(edge_id: str, edg_root) -> int:
    """Calculate lane count based on land use zones and edge characteristics"""
    zones_data = load_zones_data()
    adjacent_zones = find_adjacent_zones(edge_id, zones_data, edg_root)

    if not adjacent_zones:
        # Fallback: return moderate lane count if no zone data
        return 2

    # Calculate traffic demand score based on adjacent zones
    demand_score = 0.0

    # Traffic generation weights by land use type
    land_use_weights = {
        'Mixed': 3.0,                    # Highest traffic generation
        'Employment': 2.5,               # High peak traffic
        'Entertainment/Retail': 2.5,    # High commercial traffic
        'Public Buildings': 2.0,        # Moderate institutional traffic
        'Residential': 1.5,              # Moderate residential traffic
        'Public Open Space': 1.0         # Lower recreational traffic
    }

    for zone in adjacent_zones:
        land_use = zone.get('land_use', 'Residential')
        attractiveness = zone.get('attractiveness', 0.5)

        # Weight by land use and attractiveness
        weight = land_use_weights.get(land_use, 1.5)
        demand_score += weight * attractiveness

    # Apply edge position modifier
    if is_perimeter_edge(edge_id, edg_root):
        demand_score *= 0.8  # Perimeter edges typically have less internal traffic

    # Convert demand score to lane count (1-3 lanes)
    if demand_score < 1.5:
        return 1
    elif demand_score < 3.0:
        return 2
    else:
        return 3


def calculate_lane_count(edge_id: str, algorithm: str, rng, min_lanes: int, max_lanes: int) -> int:
    """Calculate lane count based on specified algorithm"""
    if algorithm == 'random':
        return rng.randint(min_lanes, max_lanes)
    elif algorithm == 'realistic':
        # Parse edge files to get edge root
        edg_tree = ET.parse(CONFIG.network_edg_file)
        edg_root = edg_tree.getroot()
        realistic_lanes = calculate_lane_count_realistic(edge_id, edg_root)
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