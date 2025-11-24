import xml.etree.ElementTree as ET
import json
import random
from pathlib import Path
from src.config import CONFIG
from src.constants import (
    ATTR_CURRENT_PHASE, RUSH_HOUR_MORNING_START, RUSH_HOUR_MORNING_END,
    RUSH_HOUR_EVENING_START, RUSH_HOUR_EVENING_END,
    PHASE_MORNING_PEAK_DEPART_MULTIPLIER, PHASE_MORNING_PEAK_ARRIVE_MULTIPLIER,
    PHASE_MIDDAY_OFFPEAK_DEPART_MULTIPLIER, PHASE_MIDDAY_OFFPEAK_ARRIVE_MULTIPLIER,
    PHASE_EVENING_PEAK_DEPART_MULTIPLIER, PHASE_EVENING_PEAK_ARRIVE_MULTIPLIER,
    PHASE_NIGHT_LOW_DEPART_MULTIPLIER, PHASE_NIGHT_LOW_ARRIVE_MULTIPLIER,
    MIN_ATTRACTIVENESS_VALUE, MAX_ATTRACTIVENESS_VALUE,
    GEOMETRY_DISTANCE_THRESHOLD, ZONE_ADJACENCY_DISTANCE_THRESHOLD
)

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

    return min_dist < GEOMETRY_DISTANCE_THRESHOLD


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
            zone_props = zone_feature.get('properties', {})
            # Changed from 'type' to 'land_use'
            zone_type = zone_props.get('land_use', 'Mixed')
            # Changed from 'id' to 'zone_id'
            zone_id = zone_props.get('zone_id', 'Unknown')

            if SHAPELY_AVAILABLE:
                # Use shapely for precise geometry
                edge_line = LineString(coords)
                zone_polygon = Polygon(zone_coords)
                intersects = edge_line.intersects(zone_polygon)
                distance = edge_line.distance(zone_polygon)
                is_adjacent = intersects or distance < ZONE_ADJACENCY_DISTANCE_THRESHOLD
            else:
                # Use fallback geometry functions
                is_adjacent = _line_intersects_polygon_fallback(
                    coords, zone_coords)

            if is_adjacent:
                # Return properties with corrected field names for consistency
                corrected_props = zone_props.copy()
                corrected_props['type'] = zone_props.get('land_use', 'Mixed')
                corrected_props['id'] = zone_props.get('zone_id', 'Unknown')
                adjacent_zones.append(corrected_props)

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
    if RUSH_HOUR_MORNING_START <= current_hour < RUSH_HOUR_MORNING_END:
        return "morning_peak"
    elif RUSH_HOUR_MORNING_END <= current_hour < RUSH_HOUR_EVENING_START:
        return "midday_offpeak"
    elif RUSH_HOUR_EVENING_START <= current_hour < RUSH_HOUR_EVENING_END:
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
        "morning_peak": {"depart": PHASE_MORNING_PEAK_DEPART_MULTIPLIER, "arrive": PHASE_MORNING_PEAK_ARRIVE_MULTIPLIER},
        # Balanced baseline
        "midday_offpeak": {"depart": PHASE_MIDDAY_OFFPEAK_DEPART_MULTIPLIER, "arrive": PHASE_MIDDAY_OFFPEAK_ARRIVE_MULTIPLIER},
        # High inbound (work→home)
        "evening_peak": {"depart": PHASE_EVENING_PEAK_DEPART_MULTIPLIER, "arrive": PHASE_EVENING_PEAK_ARRIVE_MULTIPLIER},
        # Minimal activity
        "night_low": {"depart": PHASE_NIGHT_LOW_DEPART_MULTIPLIER, "arrive": PHASE_NIGHT_LOW_ARRIVE_MULTIPLIER}
    }
    return PHASE_MULTIPLIERS.get(phase, {"depart": 1.0, "arrive": 1.0})


def get_time_multipliers(hour_of_day: float = 12.0) -> dict:
    """Get time-dependent multipliers for attractiveness (legacy function)"""
    # Use phase-based system
    phase = get_current_phase(hour_of_day)
    return get_phase_multipliers(phase)


def calculate_attractiveness_poisson(seed: int, edge_id: str = "") -> dict:
    """Calculate attractiveness using Poisson distribution for all 4 time periods.

    Returns 8 values (departure/arrival × 4 phases) in 1-20 range.
    """
    # Edge-specific variation for reproducible but varied results
    edge_hash = hash(edge_id) % 10000 if edge_id else 0

    phases = {}
    phase_names = ["morning_peak", "midday_offpeak",
                   "evening_peak", "night_low"]

    for phase in phase_names:
        # Different lambda values for each phase to create variety
        phase_hash = hash(phase) % 1000

        # Generate values with phase and edge-specific variation
        depart_lambda = CONFIG.LAMBDA_DEPART + \
            (edge_hash * 0.001) + (phase_hash * 0.01)
        arrive_lambda = CONFIG.LAMBDA_ARRIVE + \
            (edge_hash * 0.001) + (phase_hash * 0.01)

        # Generate Poisson values and clamp to 1-20 range
        depart_raw = np.random.poisson(lam=depart_lambda)
        arrive_raw = np.random.poisson(lam=arrive_lambda)

        phases[phase] = {
            'depart': max(MIN_ATTRACTIVENESS_VALUE, min(MAX_ATTRACTIVENESS_VALUE, depart_raw)),
            'arrive': max(MIN_ATTRACTIVENESS_VALUE, min(MAX_ATTRACTIVENESS_VALUE, arrive_raw))
        }

    return phases


def calculate_attractiveness_land_use(edge_id: str, zones_data: list, edg_root, seed: int) -> dict:
    """Calculate attractiveness based on land use zones with realistic temporal patterns.

    Returns 8 values (departure/arrival × 4 phases) in 1-20 range based on zone types.

    Parameters
    ----------
    edge_id : Edge identifier
    zones_data : Land use zone data
    edg_root : XML root for edge definitions
    seed : Random seed for deterministic edge variation
    """
    # Realistic temporal patterns for each zone type (1-20 range)
    land_use_temporal_patterns = {
        'Residential': {
            # People leaving for work
            'morning_peak': {'depart': 17, 'arrive': 1},
            'midday_offpeak': {'depart': 6, 'arrive': 4},   # Moderate activity
            # People coming home
            'evening_peak': {'depart': 3, 'arrive': 18},
            'night_low': {'depart': 1, 'arrive': 4}         # Very low activity
        },
        'Employment': {
            # People arriving for work
            'morning_peak': {'depart': 1, 'arrive': 18},
            'midday_offpeak': {'depart': 6, 'arrive': 6},   # Lunch, meetings
            # People leaving work
            'evening_peak': {'depart': 18, 'arrive': 1},
            # Security, cleaning
            'night_low': {'depart': 1, 'arrive': 1}
        },
        'Mixed': {
            'morning_peak': {'depart': 10, 'arrive': 10},   # Balanced activity
            # Peak mixed activity
            'midday_offpeak': {'depart': 10, 'arrive': 10},
            'evening_peak': {'depart': 10, 'arrive': 10},   # Balanced activity
            # Restaurants, services
            'night_low': {'depart': 4, 'arrive': 4}
        },
        'Entertainment/Retail': {
            # Opening, staff arriving
            'morning_peak': {'depart': 1, 'arrive': 5},
            # Shopping, services
            'midday_offpeak': {'depart': 9, 'arrive': 12},
            # Peak entertainment
            'evening_peak': {'depart': 12, 'arrive': 16},
            'night_low': {'depart': 12, 'arrive': 3}        # Bars, restaurants
        },
        'Public Buildings': {
            'morning_peak': {'depart': 2, 'arrive': 8},    # Services opening
            # Peak service hours
            'midday_offpeak': {'depart': 7, 'arrive': 6},
            'evening_peak': {'depart': 7, 'arrive': 2},     # Reduced activity
            'night_low': {'depart': 1, 'arrive': 1}         # Mostly closed
        },
        'Public Open Space': {
            # Exercise, commuting
            'morning_peak': {'depart': 6, 'arrive': 6},
            'midday_offpeak': {'depart': 8, 'arrive': 8},   # Recreation
            # After-work activities
            'evening_peak': {'depart': 10, 'arrive': 10},
            # Limited night activity
            'night_low': {'depart': 1, 'arrive': 1}
        }
    }

    # Find adjacent zones
    adjacent_zones = find_adjacent_zones(edge_id, zones_data, edg_root)

    phases = {}
    phase_names = ["morning_peak", "midday_offpeak",
                   "evening_peak", "night_low"]

    # Edge-specific base variation using seeded random generator
    # Combine seed with edge_id for deterministic per-edge variation
    # Use string concatenation to create unique seed per edge
    edge_seed_str = f"{seed}_{edge_id}"
    edge_rng = random.Random(edge_seed_str)
    edge_variation_value = edge_rng.randint(0, 99)
    base_variation = 1 + (edge_variation_value * 0.01)  # 1.0 to 2.0 multiplier

    for phase in phase_names:
        if adjacent_zones:
            # Calculate weighted average based on zone types and attractiveness
            total_weight = 0
            weighted_depart = 0
            weighted_arrive = 0

            for zone in adjacent_zones:
                zone_type = zone.get('type', 'Mixed')
                attractiveness = float(zone.get('attractiveness', 0.5))

                pattern = land_use_temporal_patterns.get(
                    zone_type, land_use_temporal_patterns['Mixed'])

                weighted_depart += pattern[phase]['depart'] * attractiveness
                weighted_arrive += pattern[phase]['arrive'] * attractiveness
                total_weight += attractiveness

            if total_weight > 0:
                avg_depart = weighted_depart / total_weight
                avg_arrive = weighted_arrive / total_weight
            else:
                # Fallback to Mixed pattern
                avg_depart = land_use_temporal_patterns['Mixed'][phase]['depart']
                avg_arrive = land_use_temporal_patterns['Mixed'][phase]['arrive']
        else:
            # No zones found - use Mixed pattern as default
            avg_depart = land_use_temporal_patterns['Mixed'][phase]['depart']
            avg_arrive = land_use_temporal_patterns['Mixed'][phase]['arrive']

        # Apply edge variation and ensure attractiveness value range
        final_depart = max(MIN_ATTRACTIVENESS_VALUE, min(
            MAX_ATTRACTIVENESS_VALUE, int(avg_depart * base_variation)))
        final_arrive = max(MIN_ATTRACTIVENESS_VALUE, min(
            MAX_ATTRACTIVENESS_VALUE, int(avg_arrive * base_variation)))

        phases[phase] = {
            'depart': final_depart,
            'arrive': final_arrive
        }

    return phases


def calculate_attractiveness_iac(edge_id: str, zones_data: list, edg_root, network_tree) -> dict:
    """Calculate attractiveness using Integrated Attraction Coefficient (IAC) with temporal patterns.

    Returns 8 values (departure/arrival × 4 phases) in 1-20 range.
    Applies spatial IAC factors to land use temporal patterns.
    """
    # Get land use temporal patterns as base
    land_use_patterns = calculate_attractiveness_land_use(
        edge_id, zones_data, edg_root)

    # IAC parameters from research (unused variables removed to avoid warnings)
    # Base attractiveness factor (theta)
    edge_hash = hash(edge_id) % 1000
    theta = max(0.5, min(2.0, 1.0 + (edge_hash * 0.001)))  # 0.5-2.0 range

    # Random mood factor with edge-specific seed
    np.random.seed(hash(edge_id) % 10000)
    m_rand = max(0.7, min(1.5, np.random.normal(1.0, 0.2)))  # 0.7-1.5 range

    # Spatial preference based on edge connectivity (simplified)
    # In real implementation, this would analyze network topology
    # 0.8-1.3 range
    f_spatial = max(0.8, min(1.3, 1.0 + ((edge_hash % 100) * 0.005)))

    # Combined IAC factor
    iac_factor = theta * m_rand * f_spatial  # Typical range: 0.3-4.0

    # Apply IAC factor to land use patterns
    phases = {}
    for phase, values in land_use_patterns.items():
        # Apply IAC factor and ensure 1-20 range
        depart_iac = values['depart'] * iac_factor
        arrive_iac = values['arrive'] * iac_factor

        phases[phase] = {
            'depart': max(MIN_ATTRACTIVENESS_VALUE, min(MAX_ATTRACTIVENESS_VALUE, int(depart_iac))),
            'arrive': max(MIN_ATTRACTIVENESS_VALUE, min(MAX_ATTRACTIVENESS_VALUE, int(arrive_iac)))
        }

    return phases


def assign_edge_attractiveness(seed: int, method: str = "poisson", start_time_hour: float = 0.0) -> None:
    """
    Adds attractiveness attributes to each edge in the .net.xml file.

    Generates 4 phase-specific profiles:
    - Base attributes: depart_attractiveness, arrive_attractiveness
    - Phase-specific: [phase]_depart_attractiveness, [phase]_arrive_attractiveness

    Parameters
    ----------
    seed : random seed for reproducibility
    method : attractiveness calculation method ('poisson', 'land_use', 'iac')
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
    phases = ["morning_peak", "midday_offpeak",
              "evening_peak", "night_low"]
    # Determine starting phase based on start_time_hour
    current_phase = get_current_phase(start_time_hour)

    for edge in root.findall("edge"):
        edge_id = edge.get('id')

        # Skip internal edges (start with ":")
        if edge_id and edge_id.startswith(":"):
            continue

        # Generate all 8 attractiveness values directly
        if method == "poisson":
            phase_values = calculate_attractiveness_poisson(seed)
        elif method == "land_use":
            phase_values = calculate_attractiveness_land_use(
                edge_id, zones_data, edg_root, seed)
        elif method == "iac":
            phase_values = calculate_attractiveness_iac(
                edge_id, zones_data, edg_root, tree)
        else:
            phase_values = calculate_attractiveness_poisson(seed)

        # Set phase-specific attributes directly from calculated values
        for phase in phases:
            depart_val = phase_values[phase]['depart']
            arrive_val = phase_values[phase]['arrive']

            edge.set(f"{phase}_depart_attractiveness", str(depart_val))
            edge.set(f"{phase}_arrive_attractiveness", str(arrive_val))

        # Store current phase for dynamic lookup by other components
        edge.set(ATTR_CURRENT_PHASE, current_phase)

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
        args.start_time_hour
    )
    try:
        verify_assign_edge_attractiveness(
            get_network_seed(args),
            args.attractiveness
        )
    except ValidationError as ve:
        logger.error(f"Failed to assign edge attractiveness: {ve}")
        raise
