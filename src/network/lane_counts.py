import xml.etree.ElementTree as ET
from src.config import CONFIG
import random
import re
import math
import json
from collections import defaultdict
from pathlib import Path
from shapely.geometry import Point, Polygon


def load_zones_data():
    """Load zones data from the generated GeoJSON file"""
    zones_geojson_path = Path(CONFIG.output_dir) / "zones.geojson"
    if not zones_geojson_path.exists():
        return []

    try:
        with open(zones_geojson_path, 'r') as f:
            geojson_data = json.load(f)
        return geojson_data.get('features', [])
    except (json.JSONDecodeError, FileNotFoundError):
        return []


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


def set_lane_counts(seed: int, min_lanes: int, max_lanes: int, algorithm: str = 'realistic'):
    rng = random.Random(seed)

    # Parse all SUMO network files
    edg_tree = ET.parse(CONFIG.network_edg_file)
    edg_root = edg_tree.getroot()
    con_tree = ET.parse(CONFIG.network_con_file)
    con_root = con_tree.getroot()
    tll_tree = ET.parse(CONFIG.network_tll_file)
    tll_root = tll_tree.getroot()

    # Step 1: Assign lane counts to each logical edge group (base and its _H variant)
    edge_groups = defaultdict(list)
    for edge in edg_root.findall('edge'):
        eid = edge.get('id')
        base = eid[:-2] if eid.endswith('_H') else eid
        edge_groups[base].append(edge)

    lane_counts = {}
    for base in sorted(edge_groups):
        edges = edge_groups[base]
        lanes = calculate_lane_count(
            base, algorithm, rng, min_lanes, max_lanes)
        lane_counts[base] = lanes
        for e in edges:
            e.set('numLanes', str(lanes))  # update edge element

    # Step 2: Build mapping from each edge id to its raw connection elements
    out_conns = defaultdict(list)
    for conn in con_root.findall('connection'):
        out_conns[conn.get('from')].append(conn)

    # Helper: infer direction (left/right/straight) by geometry
    def infer_direction(from_edge_id: str, to_edge_id: str):
        # find edge shapes
        fe = edg_root.find(f"edge[@id='{from_edge_id}']")
        te = edg_root.find(f"edge[@id='{to_edge_id}']")
        if fe is None or te is None or 'shape' not in fe.attrib or 'shape' not in te.attrib:
            return 'straight'
        # parse coordinates
        coords_f = [tuple(map(float, p.split(',')))
                    for p in fe.get('shape').split()]
        coords_t = [tuple(map(float, p.split(',')))
                    for p in te.get('shape').split()]
        if len(coords_f) < 2 or len(coords_t) < 2:
            return 'straight'
        # vector into intersection and out of intersection
        in_vec = (coords_f[-1][0]-coords_f[-2][0],
                  coords_f[-1][1]-coords_f[-2][1])
        out_vec = (coords_t[1][0]-coords_t[0][0],
                   coords_t[1][1]-coords_t[0][1])
        # cross and dot
        cross = in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0]
        dot = in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]
        ang = math.degrees(math.atan2(cross, dot))
        if abs(ang) < 15:
            return 'straight'
        return 'left' if ang > 0 else 'right'

    # Helper: map from-lanes to to-lanes based on counts R->T
    # def map_to_lanes(R: int, to_edge_id: str):
    #     # find T = lanes on destination edge
    #     te = edg_root.find(f"edge[@id='{to_edge_id}']")
    #     T = int(te.get('numLanes', '1')) if te is not None else 1
    #     mapping = {}
    #     if R <= T:
    #         # map each incoming lane to distinct outgoing lanes from right to left
    #         for k in range(R):
    #             mapping[k] = T-1 - k
    #     else:
    #         # map top T lanes one-to-one, rest to leftmost (0)
    #         for k in range(R):
    #             if k >= R-T:
    #                 offset = k - (R-T)
    #                 mapping[k] = T-1 - offset
    #             else:
    #                 mapping[k] = 0
    #     return mapping

    def map_to_lanes(R: int, to_edge_id: str):
        # find T = lanes on destination edge
        te = edg_root.find(f"edge[@id='{to_edge_id}']")
        T = int(te.get('numLanes', '1')) if te is not None else 1
        # initialize mapping of each incoming lane to a list of outgoing lanes
        mapping = {k: [] for k in range(R)}
        if R <= T:
            # one-to-one mapping from right to left
            for k in range(R):
                mapping[k] = [T-1 - k]
        else:
            # top T incoming lanes one-to-one, rest are leftovers
            for k in range(R-T, R):
                mapping[k] = [T-1 - (k - (R-T))]
            leftovers = list(range(R-T))
            # spread leftover lanes evenly across T targets
            per = math.ceil(len(leftovers) / T)
            for i, t in enumerate(range(T)):
                chunk = leftovers[i*per:(i+1)*per]
                for fl in chunk:
                    mapping[fl].append(T-1 - t)
        return mapping

    # Step 3: Redistribute connections in con file per lane and direction rules
    for base, lanes in lane_counts.items():
        head = base + '_H'
        conns = out_conns.get(head, [])
        if not conns:
            continue
        # determine reverse id for u-turn detection
        parts = re.findall(r"[A-Z][^A-Z]*", base)
        rev = parts[1] + parts[0] if len(parts) >= 2 else ''
        # group by movement
        groups = {'uturn': [], 'left': [], 'right': [], 'straight': []}
        for c in conns:
            tgt = c.get('to')
            if tgt == rev:
                groups['uturn'].append(c)
            else:
                groups[infer_direction(head, tgt)].append(c)
        # remove old
        for c in conns:
            con_root.remove(c)
        # store new connections
        new_conns = []
        # u-turn: always lane (lanes - 1)
        # for c in groups['uturn']:
        #     fl = lanes - 1
        #     tl = map_to_lanes(lanes, c.get('to'))[fl]
        #     a = c.attrib.copy()
        #     a['fromLane'] = str(fl)
        #     a['toLane'] = str(tl)
        #     new_conns.append(ET.Element('connection', a))
        for c in groups['uturn']:
            fl = lanes - 1
            # Original connection was from 1 lane (index 0)
            for tl in map_to_lanes(1, c.get('to'))[0]:
                a = c.attrib.copy()
                a['fromLane'] = str(fl)
                a['toLane'] = str(tl)
                new_conns.append(ET.Element('connection', a))
        # determine non-u movements
        moves = [d for d in ['left', 'straight', 'right'] if groups[d]]
        assign = {}

        # CORRECT APPROACH: Assign movements left-to-right in logical order
        # Left lanes get left turns, middle lanes get straight, right lanes get right turns
        # This prevents crossing conflicts while preserving all movement options

        if not moves:
            # no movements available, assign straight as fallback
            for i in range(lanes):
                assign[i] = ['straight']
        elif len(moves) == 1:
            # only one movement type - assign to all lanes
            for i in range(lanes):
                assign[i] = moves
        elif set(moves) == {'left', 'right'}:
            # Left and right only (no straight) - CORRECTED: lane 0 = rightmost, higher = leftmost
            if lanes == 1:
                # prefer right for single lane (lane 0 = rightmost)
                assign[0] = ['right']
            elif lanes == 2:
                assign[0] = ['right']  # lane 0 = rightmost lane: right turn
                assign[1] = ['left']   # lane 1 = leftmost lane: left turn
            else:  # 3+ lanes
                mid = lanes // 2
                for i in range(mid):
                    # right half (lower numbers): right turns
                    assign[i] = ['right']
                for i in range(mid, lanes):
                    # left half (higher numbers): left turns
                    assign[i] = ['left']
        elif set(moves) == {'left', 'straight'}:
            # Left and straight - CORRECTED: assign left to highest lanes, straight to others
            if lanes == 1:
                assign[0] = ['straight']  # prefer straight for single lane
            else:
                # highest lane = leftmost: left turn
                assign[lanes-1] = ['left']
                for i in range(lanes-1):
                    assign[i] = ['straight']  # other lanes: straight
        elif set(moves) == {'straight', 'right'}:
            # Straight and right - CORRECTED: assign straight to higher lanes, right to lane 0
            if lanes == 1:
                assign[0] = ['straight']  # prefer straight for single lane
            else:
                assign[0] = ['right']       # lane 0 = rightmost: right turn
                for i in range(1, lanes):
                    assign[i] = ['straight']  # higher lanes: straight
        elif set(moves) == {'left', 'straight', 'right'}:
            # All three movements - CORRECTED assignment
            if lanes == 1:
                assign[0] = ['straight']  # prefer straight for single lane
            elif lanes == 2:
                assign[0] = ['right']     # lane 0 = rightmost: right turn
                assign[1] = ['left']      # lane 1 = leftmost: left turn
            else:  # 3+ lanes - CORRECTED perfect assignment
                # lane 0 = rightmost: right turn
                assign[0] = ['right']
                for i in range(1, lanes-1):
                    assign[i] = ['straight']    # middle lanes: straight
                # highest lane = leftmost: left turn
                assign[lanes-1] = ['left']
        else:
            # fallback: straight for all
            for i in range(lanes):
                assign[i] = ['straight']
        # create connections per lane and direction
        # for fl, dirs in assign.items():
        #     tl_map = map_to_lanes(lanes, None)  # here None will default T=1
        #     # but we need per direction per connection
        #     for d in dirs:
        #         for c in groups[d]:
        #             tl = map_to_lanes(lanes, c.get('to'))[fl]
        #             a = c.attrib.copy()
        #             a['fromLane'] = str(fl)
        #             a['toLane'] = str(tl)
        #             new_conns.append(ET.Element('connection', a))
        # # append all new
        # for nc in new_conns:
        #     con_root.append(nc)
        # create connections per lane and direction - PRESERVE ALL MOVEMENTS
        # Ensure every lane gets a connection and all original movements are preserved

        # Track which destinations have been used for each direction to avoid duplicates
        used_by_direction = {direction: set()
                             for direction in ['left', 'straight', 'right']}

        for fl, dirs in assign.items():
            if not dirs:
                continue

            # Each lane should have exactly ONE direction assigned
            direction = dirs[0]
            if direction not in groups or not groups[direction]:
                continue

            # Find an unused connection for this direction
            available_connections = [c for c in groups[direction]
                                     if c.get('to') not in used_by_direction[direction]]

            # If no unused connections, use any connection for this direction (preserve movements)
            if not available_connections:
                available_connections = groups[direction]

            if not available_connections:
                continue

            # Use the first available connection
            c = available_connections[0]
            used_by_direction[direction].add(c.get('to'))

            # Get target edge lane count and map to exactly ONE target lane
            lane_mapping = map_to_lanes(lanes, c.get('to'))
            if fl in lane_mapping and lane_mapping[fl]:
                # Use only the FIRST target lane to avoid multiple connections
                tl = lane_mapping[fl][0]
                a = c.attrib.copy()
                a['fromLane'] = str(fl)
                a['toLane'] = str(tl)
                new_conns.append(ET.Element('connection', a))

        # ENSURE ALL MOVEMENTS ARE PRESERVED: Add missing movements to available lanes
        for direction in ['left', 'straight', 'right']:
            if direction in groups and groups[direction]:
                # Check if this direction is covered by any lane
                direction_covered = any(direction in assign.get(
                    fl, []) for fl in range(lanes))

                if not direction_covered:
                    # Find a lane that could accommodate this direction
                    # Prefer appropriate lanes: left->high lanes, right->lane 0, straight->middle
                    if direction == 'left' and lanes > 1:
                        target_lane = lanes - 1  # leftmost lane
                    elif direction == 'right':
                        target_lane = 0  # rightmost lane
                    else:  # straight
                        target_lane = lanes // 2  # middle lane

                    # Add this direction to the target lane
                    if target_lane not in assign:
                        assign[target_lane] = []
                    assign[target_lane].append(direction)

                    # Create connection for the missing movement
                    c = groups[direction][0]
                    lane_mapping = map_to_lanes(lanes, c.get('to'))
                    if target_lane in lane_mapping and lane_mapping[target_lane]:
                        tl = lane_mapping[target_lane][0]
                        a = c.attrib.copy()
                        a['fromLane'] = str(target_lane)
                        a['toLane'] = str(tl)
                        new_conns.append(ET.Element('connection', a))

        # append all new
        for nc in new_conns:
            con_root.append(nc)

    # Step 4: Place updated connections after all tlLogic elements
    # Remove any <connection> children under each tlLogic
    for tl_logic in tll_root.findall('tlLogic'):
        for c in list(tl_logic.findall('connection')):
            tl_logic.remove(c)
    # Remove any existing <connection> elements directly under the root
    for c in list(tll_root.findall('connection')):
        tll_root.remove(c)

    new_tll_conns = []
    # iterate each traffic light and gather its own incoming links
    for tl_logic in tll_root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        idx = 0
        # for each connection, include only if its incoming edge ends at this junction
        for c in con_root.findall('connection'):
            fe = edg_root.find(f"edge[@id='{c.get('from')}']")
            if fe is None or fe.get('to') != tl_id:
                continue
            # Create a traffic light connection for EVERY actual connection
            a = c.attrib.copy()
            a['tl'] = tl_id
            a['linkIndex'] = str(idx)
            idx += 1
            new_tll_conns.append(ET.Element('connection', a))

    # Append all new connections to tll_root, so they appear after all tlLogic definitions
    for conn in new_tll_conns:
        tll_root.append(conn)

    # Step 4.5: Update traffic light state strings to match new connection count
    def classify_connection_direction(conn):
        """Classify connection as NS (North-South) or EW (East-West) based on edge names"""
        from_edge = conn.get('from')
        to_edge = conn.get('to')

        # Extract junction names from edge IDs (e.g., "A0B0_H" -> from A0 to B0)
        from_parts = from_edge.replace('_H', '').split(
            '_')[0] if '_H' in from_edge else from_edge
        to_parts = to_edge.split('_')[0] if '_' in to_edge else to_edge

        # Determine direction based on junction coordinate pattern
        # Grid uses pattern: A0, A1, B0, B1 where letter = row, number = column
        if len(from_parts) >= 2 and len(to_parts) >= 2:
            from_row, from_col = from_parts[0], from_parts[1]
            to_row, to_col = to_parts[0], to_parts[1]

            # NS movement: same column, different row
            if from_col == to_col and from_row != to_row:
                return 'ns'
            # EW movement: same row, different column
            elif from_row == to_row and from_col != to_col:
                return 'ew'

        # Default to straight/turn based on edge direction pattern
        return 'ns'  # Default assumption

    # Update each traffic light's state strings
    for tl_logic in tll_root.findall('tlLogic'):
        tl_id = tl_logic.get('id')

        # Get all connections for this traffic light
        tl_connections = [c for c in tll_root.findall(
            'connection') if c.get('tl') == tl_id]
        tl_connections.sort(key=lambda x: int(x.get('linkIndex', '0')))

        if not tl_connections:
            continue

        # Build new state strings based on connection directions
        green_duration = 42
        yellow_duration = 3

        # Phase 1: NS green, EW red
        state1 = "".join("G" if classify_connection_direction(
            c) == 'ns' else "r" for c in tl_connections)
        # Phase 2: NS yellow, EW red
        state2 = "".join("y" if classify_connection_direction(
            c) == 'ns' else "r" for c in tl_connections)
        # Phase 3: NS red, EW green
        state3 = "".join("r" if classify_connection_direction(
            c) == 'ns' else "G" for c in tl_connections)
        # Phase 4: NS red, EW yellow
        state4 = "".join("r" if classify_connection_direction(
            c) == 'ns' else "y" for c in tl_connections)

        # Update the phase elements
        phases = tl_logic.findall('phase')
        if len(phases) >= 4:
            phases[0].set('state', state1)
            phases[1].set('state', state2)
            phases[2].set('state', state3)
            phases[3].set('state', state4)

    # Step 4.7: Fix base edge to _H edge connections (these were created before lane counts were set)
    # Remove old base→_H connections with only lane 0
    for c in list(con_root.findall('connection')):
        if c.get('to', '').endswith('_H') and not c.get('to', '').endswith('_H_node'):
            # This is a base→_H connection, remove it so we can recreate with all lanes
            con_root.remove(c)

    # Create new base→_H connections for all lanes
    for base, lanes in lane_counts.items():
        # Find if this base edge exists in the connections
        base_edge_exists = any(
            c.get('from') == base for c in con_root.findall('connection'))
        if base_edge_exists or base in [edge.get('id') for edge in edg_root.findall('edge')]:
            # Create connections from base edge to its _H variant for all lanes
            for lane in range(lanes):
                new_conn = ET.Element('connection', {
                    'from': base,
                    'to': f'{base}_H',
                    'fromLane': str(lane),
                    'toLane': str(lane)
                })
                # Insert at beginning to keep them organized
                con_root.insert(0, new_conn)

    # Step 5: Write changes back
    ET.indent(edg_root)
    ET.indent(con_root)
    ET.indent(tll_root)
    edg_tree.write(CONFIG.network_edg_file,
                   encoding='UTF-8', xml_declaration=True)
    con_tree.write(CONFIG.network_con_file,
                   encoding='UTF-8', xml_declaration=True)
    tll_tree.write(CONFIG.network_tll_file,
                   encoding='UTF-8', xml_declaration=True)
