import xml.etree.ElementTree as ET
from src.config import CONFIG
import random
import re
import math
from collections import defaultdict


def set_lane_counts(seed: int, min_lanes: int, max_lanes: int):
    rng = random.Random(seed)

    # Parse all SUMO network files
    edg_tree = ET.parse(CONFIG.network_edg_file)
    edg_root = edg_tree.getroot()
    con_tree = ET.parse(CONFIG.network_con_file)
    con_root = con_tree.getroot()
    tll_tree = ET.parse(CONFIG.network_tll_file)
    tll_root = tll_tree.getroot()

    # Step 1: Assign random lane counts to each logical edge group (base and its _H variant)
    edge_groups = defaultdict(list)
    for edge in edg_root.findall('edge'):
        eid = edge.get('id')
        base = eid[:-2] if eid.endswith('_H') else eid
        edge_groups[base].append(edge)

    lane_counts = {}
    for base in sorted(edge_groups):
        edges = edge_groups[base]
        lanes = rng.randint(min_lanes, max_lanes)
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
            for tl in map_to_lanes(lanes, c.get('to'))[fl]:
                a = c.attrib.copy()
                a['fromLane'] = str(fl)
                a['toLane'] = str(tl)
                new_conns.append(ET.Element('connection', a))
        # determine non-u movements
        moves = [d for d in ['left', 'straight', 'right'] if groups[d]]
        assign = {}
        # apply rules 1-6
        if moves == ['right']:
            for i in range(lanes):
                assign[i] = ['right']
        elif moves == ['left']:
            for i in range(lanes):
                assign[i] = ['left']
        elif set(moves) == {'left', 'right'}:
            half = lanes//2
            for i in range(half):
                assign[i] = ['left']
            for i in range(half, lanes):
                assign[i] = ['right']
            if lanes % 2:
                assign[lanes//2] = ['left', 'right']
        elif set(moves) == {'straight', 'right'}:
            if lanes > 2:
                half = lanes//2
                for i in range(half):
                    assign[i] = ['right']
                for i in range(half, lanes):
                    assign[i] = ['straight']
                if lanes % 2:
                    assign[lanes//2].append('right')
            else:
                assign[0] = ['straight']
                assign[1] = ['straight', 'right'] if lanes > 1 else ['straight']
        elif set(moves) == {'straight', 'left'}:
            if lanes > 2:
                half = lanes//2
                for i in range(half):
                    assign[i] = ['left']
                for i in range(half, lanes):
                    assign[i] = ['straight']
                if lanes % 2:
                    assign[lanes//2].append('left')
            else:
                assign[0] = ['straight', 'left']
                assign[1] = ['straight'] if lanes > 1 else ['straight']
        elif set(moves) == {'left', 'straight', 'right'}:
            for i in range(lanes):
                assign[i] = ['straight']
            assign[0].append('left')
            assign[lanes-1].append('right')
        else:
            # only straight or no movement
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
        # create connections per lane and direction
        for fl, dirs in assign.items():
            for d in dirs:
                for c in groups[d]:
                    # one entry for each target lane in the list
                    for tl in map_to_lanes(lanes, c.get('to'))[fl]:
                        a = c.attrib.copy()
                        a['fromLane'] = str(fl)
                        a['toLane'] = str(tl)
                        new_conns.append(ET.Element('connection', a))

        # fallback: ensure every destination lane has at least one incoming connection
        by_dest = defaultdict(list)
        for c in new_conns:
            by_dest[c.get('to')].append(c)
        for dest_edge, conns_to_dest in by_dest.items():
            T = int(edg_root.find(
                f"edge[@id='{dest_edge}']").get('numLanes', '1'))
            assigned = {int(c.get('toLane')) for c in conns_to_dest}
            missing = [i for i in range(T) if i not in assigned]
            if not missing:
                continue
            # assign missing lanes from highest-index source lanes
            srcs = sorted({int(c.get('fromLane'))
                          for c in conns_to_dest}, reverse=True)
            for i, t in enumerate(missing):
                fl = srcs[i % len(srcs)]
                tpl = conns_to_dest[0].attrib.copy()
                tpl['fromLane'] = str(fl)
                tpl['toLane'] = str(t)
                new_conns.append(ET.Element('connection', tpl))

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
        seen = set()
        idx = 0
        # for each connection, include only if its incoming edge ends at this junction
        for c in con_root.findall('connection'):
            fe = edg_root.find(f"edge[@id='{c.get('from')}']")
            if fe is None or fe.get('to') != tl_id:
                continue
            key = (c.get('from'), c.get('to'))
            if key in seen:
                continue
            seen.add(key)
            a = c.attrib.copy()
            a['tl'] = tl_id
            a['linkIndex'] = str(idx)
            idx += 1
            new_tll_conns.append(ET.Element('connection', a))

    # Append all new connections to tll_root, so they appear after all tlLogic definitions
    for conn in new_tll_conns:
        tll_root.append(conn)

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
