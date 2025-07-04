import os
import subprocess
import xml.etree.ElementTree as ET
import random
import re
from typing import List
from alive_progress import *
from pathlib import Path
from typing import Set, Tuple
from src.sim.sumo_utils import *
from src.config import CONFIG


# --- Function Definitions ---
# Generate a full grid network using netgenerate

def generate_full_grid_network(dimension, block_size_m, lane_count_arg):
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
        # aggregate-warnings: aggregates warnings of the same type whenever more than 1 occur
        # warnings can be removed completely by using --no-warnings=true
        # "--aggregate-warnings=1",
        "-o", CONFIG.network_file
    ]
    # Only use netgenerate's fixed lane count if a specific integer is provided
    if lane_count_arg.isdigit() and int(lane_count_arg) > 0:
        netgenerate_cmd.append(f"--default.lanenumber={lane_count_arg}")

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
    for j in list(root.findall("connection")):
        key = (j.get("from"), j.get("to"))
        if key in cons_to_remove:
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
    for j in list(root.findall("edge")):
        if j.get("id") in edgs_to_remove:
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
    for j in list(root.findall("node")):
        if j.get("id") in nods_to_remove:
            root.remove(j)

    tree.write(Path(str(CONFIG.network_nod_file)),
               encoding="UTF-8", xml_declaration=True)


def wipe_crossing_from_tll(node_ids: list[str]) -> None:
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from typing import Set, Tuple, List

    # Parse the TLS definition file
    tree = ET.parse(Path(str(CONFIG.network_tll_file)))
    root = tree.getroot()

    # 1. Identify and remove connections to wipe
    cons_to_remove: Set[Tuple[str, str]] = get_cons_to_wipe(node_ids, root)
    # Collect (tl_id, linkIndex) for each removed connection
    states_to_remove: List[Tuple[str, int]] = []

    for conn in list(root.findall("connection")):
        key = (conn.get("from"), conn.get("to"))
        if key in cons_to_remove:
            tl_id = conn.get("tl")
            link_idx = int(conn.get("linkIndex"))
            states_to_remove.append((tl_id, link_idx))
            root.remove(conn)

    # 2. Remove entire TLs for the specified node_ids
    tlls_to_remove = set(node_ids)

    # 3. Update remaining TL logic phases by dropping removed indices
    for tl in list(root.findall("tlLogic")):
        jid = tl.get("id")
        if jid in tlls_to_remove:
            root.remove(tl)
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
        junction_ids = [j.strip() for j in junctions_input.split(',') if j.strip()]
        return True, junction_ids, len(junction_ids)


def generate_grid_network(seed, dimension, block_size_m, junctions_to_remove_input, lane_count_arg):
    try:
        is_list, junction_ids, count = parse_junctions_to_remove(junctions_to_remove_input)
        
        if count > 0:
            # generate the full grid network first
            generate_full_grid_network(
                dimension, block_size_m, lane_count_arg)

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
                dimension, block_size_m, lane_count_arg)

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netgenerate execution:", e.stderr)

    if not os.path.exists(CONFIG.network_file):
        raise Exception(
            f"Error: Network file '{CONFIG.network_file}' was not created.")
