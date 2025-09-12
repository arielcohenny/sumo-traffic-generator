import xml.etree.ElementTree as ET
import json
from pathlib import Path
from src.config import CONFIG

try:
    import numpy as np
    from shapely.geometry import Polygon, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely/numpy not available, using fallback geometry functions")


def _point_in_polygon(x, y, polygon_coords):
    """Simple point-in-polygon test using ray casting"""
    n = len(polygon_coords)
    inside = False

    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _line_intersects_polygon_fallback(line_coords, polygon_coords):
    """Check if line intersects polygon without shapely"""
    # Check if any line endpoint is inside polygon
    for x, y in line_coords:
        if _point_in_polygon(x, y, polygon_coords):
            return True

    # Check if line is very close to polygon boundary
    min_dist = float('inf')
    for lx, ly in line_coords:
        for px, py in polygon_coords:
            dist = ((lx - px)**2 + (ly - py)**2) ** 0.5
            min_dist = min(min_dist, dist)

    return min_dist < 50  # Within 50 units


def load_zones_data():
    """Load zones data from the generated GeoJSON file"""
    zones_geojson_path = Path(CONFIG.output_dir) / "zones.geojson"
    if not zones_geojson_path.exists():
        return []

    try:
        with open(zones_geojson_path, 'r') as f:
            geojson_data = json.load(f)
        features = geojson_data.get('features', [])
        if features:
            sample_zone = features[0]
        return features
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def find_adjacent_zones(edge_id: str, zones_data: list, edg_root, block_size_m: int = 200) -> list:
    """Find zones adjacent to a given edge using node coordinates when shape is not available"""
    if not zones_data:
        return []

    # Find the edge in the edg_root
    edge_elem = None
    for edge in edg_root.findall('edge'):
        if edge.get('id') == edge_id:
            edge_elem = edge
            break

    if edge_elem is None:
        return []

    # Try to get shape coordinates first
    shape_str = edge_elem.get('shape', '')
    coords = []

    if shape_str:
        # Parse shape coordinates if available
        try:
            for coord_pair in shape_str.split():
                x, y = map(float, coord_pair.split(','))
                coords.append((x, y))
        except (ValueError, TypeError):
            coords = []

    # If no shape coordinates, derive from node positions
    if not coords or len(coords) < 2:
        # Get from and to nodes
        from_node = edge_elem.get('from')
        to_node = edge_elem.get('to')

        if not from_node or not to_node:
            return []

        # Find node coordinates from the edg_root's parent or use fallback based on node names
        from_coords = _get_node_coordinates(from_node, block_size_m)
        to_coords = _get_node_coordinates(to_node, block_size_m)

        if from_coords and to_coords:
            coords = [from_coords, to_coords]
        else:
            return []

    if len(coords) < 2:
        return []

    try:
        # Find adjacent zones
        adjacent_zones = []
        for zone_feature in zones_data:
            zone_coords = zone_feature['geometry']['coordinates'][0]

            if SHAPELY_AVAILABLE:
                # Use shapely for precise geometry
                edge_line = LineString(coords)
                zone_polygon = Polygon(zone_coords)
                intersects = edge_line.intersects(zone_polygon)
                distance = edge_line.distance(zone_polygon)
                is_adjacent = intersects or distance < 10
            else:
                # Use fallback geometry functions
                is_adjacent = _line_intersects_polygon_fallback(
                    coords, zone_coords)

            if is_adjacent:
                zone_props = zone_feature.get('properties', {})
                # Return just properties, not full feature
                adjacent_zones.append(zone_props)
            else:
                # Show some non-adjacent zones for comparison
                zone_props = zone_feature.get('properties', {})

        return adjacent_zones

    except (ValueError, KeyError):
        return []


def _get_node_coordinates(node_id: str, block_size_m: int = 200) -> tuple:
    """Get coordinates for a node based on its ID using standard grid naming"""
    # Parse node ID like A0, B1, C2, etc.
    if len(node_id) < 2:
        return None

    # Extract column (letter) and row (number)
    col_char = node_id[0]
    row_char = node_id[1]

    try:
        # Convert column letter to x coordinate (A=0, B=block_size, C=2*block_size, etc.)
        col_index = ord(col_char) - ord('A')
        x = col_index * block_size_m

        # Convert row number to y coordinate
        row_index = int(row_char)
        y = row_index * block_size_m

        return (float(x), float(y))
    except (ValueError, TypeError):
        return None


def get_current_phase(current_hour: float) -> str:
    """Get current traffic phase based on hour of day (0-24)

    Research-based 4-phase system:
    - morning_peak: 6:00-9:30 (Extended morning rush, commuter-heavy)
    - midday_offpeak: 9:30-16:00 (Lower baseline traffic)
    - evening_peak: 16:00-19:00 (Extended evening rush, higher volumes)
    - night_low: 19:00-6:00 (Reduced overnight traffic)
    """
    if 6.0 <= current_hour < 9.5:
        return "morning_peak"
    elif 9.5 <= current_hour < 16.0:
        return "midday_offpeak"
    elif 16.0 <= current_hour < 19.0:
        return "evening_peak"
    else:
        return "night_low"


def get_phase_multipliers(phase: str) -> dict:
    """Get phase-specific multipliers for attractiveness

    Based on research showing bimodal traffic patterns with:
    - High outbound traffic during morning (home→work)
    - High inbound traffic during evening (work→home)
    """
    PHASE_MULTIPLIERS = {
        # High outbound (home→work)
        "morning_peak": {"depart": 1.4, "arrive": 0.7},
        "midday_offpeak": {"depart": 1.0, "arrive": 1.0},  # Balanced baseline
        # High inbound (work→home)
        "evening_peak": {"depart": 0.7, "arrive": 1.5},
        "night_low": {"depart": 0.4, "arrive": 0.4}        # Minimal activity
    }
    return PHASE_MULTIPLIERS.get(phase, {"depart": 1.0, "arrive": 1.0})


def get_time_multipliers(time_dependent: bool, hour_of_day: float = 12.0) -> dict:
    """Get time-dependent multipliers for attractiveness (legacy function)"""
    if not time_dependent:
        return {'depart': 1.0, 'arrive': 1.0}

    # Use new phase-based system
    phase = get_current_phase(hour_of_day)
    return get_phase_multipliers(phase)


def calculate_attractiveness_poisson(seed: int, time_dependent: bool = False) -> tuple:
    """Calculate attractiveness using Poisson distribution"""
    depart_base = np.random.poisson(lam=CONFIG.LAMBDA_DEPART)
    arrive_base = np.random.poisson(lam=CONFIG.LAMBDA_ARRIVE)

    if time_dependent:
        # Use midday as default for base calculation
        time_mult = get_time_multipliers(True, 12)
        depart_base = max(1, int(depart_base * time_mult['depart']))
        arrive_base = max(1, int(arrive_base * time_mult['arrive']))

    return depart_base, arrive_base


def calculate_attractiveness_land_use(edge_id: str, zones_data: list, edg_root, time_dependent: bool = False) -> tuple:
    """Calculate attractiveness based on land use zones"""
    # Land use multipliers based on research
    land_use_multipliers = {
        'Residential': {'depart': 0.8, 'arrive': 1.4},
        'Employment': {'depart': 1.3, 'arrive': 0.9},
        'Mixed': {'depart': 1.1, 'arrive': 1.1},
        'Entertainment/Retail': {'depart': 0.7, 'arrive': 1.3},
        'Public Buildings': {'depart': 0.9, 'arrive': 1.0},
        'Public Open Space': {'depart': 0.6, 'arrive': 0.8}
    }

    # Get base Poisson values
    depart_base = np.random.poisson(lam=CONFIG.LAMBDA_DEPART)
    arrive_base = np.random.poisson(lam=CONFIG.LAMBDA_ARRIVE)

    # Find adjacent zones
    adjacent_zones = find_adjacent_zones(edge_id, zones_data, edg_root)

    if adjacent_zones:
        # Calculate weighted average based on zone attractiveness
        total_weight = 0
        weighted_depart = 0
        weighted_arrive = 0

        for zone in adjacent_zones:
            zone_type = zone.get('type', 'Mixed')
            attractiveness = float(
                zone.get('attractiveness', 0.5))

            multipliers = land_use_multipliers.get(
                zone_type, {'depart': 1.0, 'arrive': 1.0})

            weighted_depart += multipliers['depart'] * attractiveness
            weighted_arrive += multipliers['arrive'] * attractiveness
            total_weight += attractiveness

        if total_weight > 0:
            avg_depart_mult = weighted_depart / total_weight
            avg_arrive_mult = weighted_arrive / total_weight
        else:
            avg_depart_mult = 1.0
            avg_arrive_mult = 1.0
    else:
        # Default multipliers if no zones found
        avg_depart_mult = 1.0
        avg_arrive_mult = 1.0

    # Apply multipliers
    depart_attr = max(1, int(depart_base * avg_depart_mult))
    arrive_attr = max(1, int(arrive_base * avg_arrive_mult))

    # Apply time dependency if enabled
    if time_dependent:
        time_mult = get_time_multipliers(True, 12)
        depart_attr = max(1, int(depart_attr * time_mult['depart']))
        arrive_attr = max(1, int(arrive_attr * time_mult['arrive']))

    return depart_attr, arrive_attr




def calculate_attractiveness_iac(edge_id: str, zones_data: list, edg_root, network_tree, time_dependent: bool = False) -> tuple:
    """Calculate attractiveness using Integrated Attraction Coefficient (IAC)"""
    # IAC parameters from research
    d_param = 0.95
    g_param = 1.02

    # Get land use components
    depart_land, arrive_land = calculate_attractiveness_land_use(
        edge_id, zones_data, edg_root, False)

    # Base attractiveness (theta)
    theta = max(0.1, np.random.normal(1.0, 0.3))

    # Random mood factor
    m_rand = max(0.5, np.random.normal(1.0, 0.2))

    # Spatial preference (simplified)
    f_spatial = 1.0

    # Calculate IAC using only land use
    land_norm = (depart_land + arrive_land) / \
        (2 * CONFIG.LAMBDA_DEPART + 2 * CONFIG.LAMBDA_ARRIVE)

    iac_factor = land_norm * theta * m_rand * f_spatial

    depart_attr = max(1, int(CONFIG.LAMBDA_DEPART * iac_factor))
    arrive_attr = max(1, int(CONFIG.LAMBDA_ARRIVE * iac_factor))

    # Apply time dependency if enabled
    if time_dependent:
        time_mult = get_time_multipliers(True, 12)
        depart_attr = max(1, int(depart_attr * time_mult['depart']))
        arrive_attr = max(1, int(arrive_attr * time_mult['arrive']))

    return depart_attr, arrive_attr




def assign_edge_attractiveness(seed: int, method: str = "poisson", time_dependent: bool = False, start_time_hour: float = 0.0) -> None:
    """
    Adds attractiveness attributes to each edge in the .net.xml file.

    When time_dependent=True, generates 4 phase-specific profiles:
    - Base attributes: depart_attractiveness, arrive_attractiveness
    - Phase-specific: [phase]_depart_attractiveness, [phase]_arrive_attractiveness

    Parameters
    ----------
    seed : random seed for reproducibility
    method : attractiveness calculation method ('poisson', 'land_use', 'iac')
    time_dependent : whether to generate 4-phase time-of-day profiles
    start_time_hour : real-world hour when simulation starts (0-24)
    """
    np.random.seed(seed)

    net_file = CONFIG.network_file
    tree = ET.parse(net_file)
    root = tree.getroot()

    # Load zone data if needed
    zones_data = []
    if method in ['land_use', 'iac']:
        zones_data = load_zones_data()

    # Load edge data if needed for spatial methods
    edg_root = None
    if method in ['land_use', 'iac']:
        try:
            edg_file = CONFIG.network_edg_file
            edg_tree = ET.parse(edg_file)
            edg_root = edg_tree.getroot()
        except:
            edg_root = None

    # Get all phases for time-dependent mode
    if time_dependent:
        phases = ["morning_peak", "midday_offpeak",
                  "evening_peak", "night_low"]
        # Determine starting phase based on start_time_hour
        current_phase = get_current_phase(start_time_hour)
    else:
        phases = [None]  # Single calculation without time dependency
        current_phase = None

    for edge in root.findall("edge"):
        edge_id = edge.get('id')

        # Skip internal edges (start with ":")
        if edge_id and edge_id.startswith(":"):
            continue

        if time_dependent:
            # Generate base attractiveness without time dependency
            if method == "poisson":
                base_depart, base_arrive = calculate_attractiveness_poisson(
                    seed, False)
            elif method == "land_use":
                base_depart, base_arrive = calculate_attractiveness_land_use(
                    edge_id, zones_data, edg_root, False)
            elif method == "iac":
                base_depart, base_arrive = calculate_attractiveness_iac(
                    edge_id, zones_data, edg_root, tree, False)
            else:
                base_depart, base_arrive = calculate_attractiveness_poisson(
                    seed, False)

            # Generate phase-specific profiles
            for phase in phases:
                multipliers = get_phase_multipliers(phase)
                phase_depart = max(1, int(base_depart * multipliers['depart']))
                phase_arrive = max(1, int(base_arrive * multipliers['arrive']))

                edge.set(f"{phase}_depart_attractiveness", str(phase_depart))
                edge.set(f"{phase}_arrive_attractiveness", str(phase_arrive))

            # Set current phase as active attributes
            current_multipliers = get_phase_multipliers(current_phase)
            current_depart = max(
                1, int(base_depart * current_multipliers['depart']))
            current_arrive = max(
                1, int(base_arrive * current_multipliers['arrive']))
            edge.set("depart_attractiveness", str(current_depart))
            edge.set("arrive_attractiveness", str(current_arrive))
            edge.set("current_phase", current_phase)

        else:
            # Non-time-dependent: single calculation
            if method == "poisson":
                depart_attr, arrive_attr = calculate_attractiveness_poisson(
                    seed, False)
            elif method == "land_use":
                depart_attr, arrive_attr = calculate_attractiveness_land_use(
                    edge_id, zones_data, edg_root, False)
            elif method == "iac":
                depart_attr, arrive_attr = calculate_attractiveness_iac(
                    edge_id, zones_data, edg_root, tree, False)
            else:
                depart_attr, arrive_attr = calculate_attractiveness_poisson(
                    seed, False)

            edge.set("depart_attractiveness", str(depart_attr))
            edge.set("arrive_attractiveness", str(arrive_attr))

    tree.write(net_file, encoding="utf-8")


def execute_attractiveness_assignment(args) -> None:
    """Execute edge attractiveness assignment."""
    import logging
    from src.utils.multi_seed_utils import get_network_seed
    from src.validate.validate_network import verify_assign_edge_attractiveness
    from src.validate.errors import ValidationError

    logger = logging.getLogger(__name__)

    assign_edge_attractiveness(
        get_network_seed(args),
        args.attractiveness,
        args.time_dependent,
        args.start_time_hour
    )
    try:
        verify_assign_edge_attractiveness(
            get_network_seed(args),
            args.attractiveness,
            args.time_dependent
        )
    except ValidationError as ve:
        logger.error(f"Failed to assign edge attractiveness: {ve}")
        raise
