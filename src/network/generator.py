import os
import subprocess
import xml.etree.ElementTree as ET
import random
from typing import List
from alive_progress import *
from pathlib import Path
from typing import Set
from src.sim.sumo_utils import *
from src.config import CONFIG

# --- Global Variables ---
# temporary files
network_file_temp = os.path.abspath(os.path.join(CONFIG.output_dir, "grid_temp.net.xml"))
network_file_prone = os.path.abspath(os.path.join(CONFIG.output_dir, "grid_prone.net.xml"))

# --- Function Definitions ---
# Generate a full grid network using netgenerate
def generate_full_grid_network(dimension, block_size_m, output_file):
    netgenerate_cmd = [
        "netgenerate", "--grid",
        f"--grid.x-number={dimension}",
        f"--grid.y-number={dimension}",
        f"--grid.x-length={block_size_m}",
        f"--grid.y-length={block_size_m}",
        "-o", output_file
    ]
    try:
        subprocess.run(netgenerate_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netgenerate execution:", e.stderr)

# Generate a list of random junction IDs to remove from the grid
def pick_random_junction_ids(seed: int, num_junctions_to_remove: int, dimension: int) -> List[str]:
    """
    Picks a list of junction IDs (e.g. 'B2') to remove from a square SUMO grid.

    Assumes the grid uses IDs like 'A0', 'B3', etc., where rows are 'A' to ...,
    and columns are 0 to (dimension - 1).

    Skips the outermost border (e.g. A* and D* in 4×4) to avoid breaking connectivity.

    Parameters
    ----------
    num_junctions_to_remove : int
        How many junctions to pick.
    dimension : int
        Size of the square grid (e.g. 4 → 4×4).
    seed : int | None
        Optional RNG seed to make the selection reproducible.

    Returns
    -------
    List[str]
        List of junction IDs like ['B1', 'C2', ...]
    """
    if dimension < 3:
        raise ValueError("Grid must be at least 3×3 to have removable interior nodes")

    row_labels = [chr(ord('A') + i) for i in range(0, dimension)]
    col_labels = list(range(0, dimension))

    candidates = [f"{row}{col}" for row in row_labels for col in col_labels]

    if num_junctions_to_remove > len(candidates):
        raise ValueError(f"Too many junctions requested: only {len(candidates)} interior junctions available.")

    rng = random.Random(seed)
    return rng.sample(candidates, num_junctions_to_remove)

# Remove a crossing and all its associated elements from the network file
def wipe_crossing(node_ids: list[str]) -> None:
    """
    Delete *node_id* completely together with every edge, lane, connection
    and micro-junction that belongs to it.  No axes are kept: the four
    roads that met at the crossing now end abruptly.
    """
    net_path, out_path = Path(str(network_file_temp)), Path(str(network_file_prone))
    tree = ET.parse(net_path)
    root = tree.getroot()

    # 0 ── sanity check
    for node_id in node_ids:
        if root.find(f".//junction[@id='{node_id}']") is None:
            raise ValueError(f"{node_id} not found in {net_path}")

    # ------------------------------------------------------------------ #
    junc_kill: Set[str] = set()
    edge_kill: Set[str] = set()
    lane_kill: Set[str] = set()

    # ------------------------------------------------------------------ #
    # 1  collect everything (micro stuff)                                #
    # ------------------------------------------------------------------ #
    # iterate over all node_ids, which are the junction IDs
    for node_id in node_ids:
        prefix = f":{node_id}_"

        # collect all micro-junctions, edges and lanes that are part of the crossing
        for j in root.findall("junction"):
            jid = j.get("id")
            if jid.startswith(prefix):
                junc_kill.add(jid)

        # collect all edges that are part of the crossing
        for e in root.findall("edge"):
            eid = e.get("id")
            fr, to = e.get("from"), e.get("to")

            # real arms that touch node_id OR internal edge of the crossing
            if fr == node_id or to == node_id or eid.startswith(prefix):
                edge_kill.add(eid)
                for ln in e.findall("lane"):
                    lane_kill.add(ln.get("id"))

        # don't forget the parent node itself
        junc_kill.add(node_id)

    # ------------------------------------------------------------------ #
    # 2  purge junctions & edges                                         #
    # ------------------------------------------------------------------ #
    # remove junctions and edges that are marked for deletion
    for j in list(root.findall("junction")):
        if j.get("id") in junc_kill:
            root.remove(j)

    # remove edges that are marked for deletion
    for e in list(root.findall("edge")):
        if e.get("id") in edge_kill:
            root.remove(e)

    # ------------------------------------------------------------------ #
    # 3  purge connections that still point at removed stuff             #
    # ------------------------------------------------------------------ #
    # remove connections that reference edges or lanes that are marked for deletion
    attrs = ("from", "to", "via", "viaLane", "fromLane", "toLane")
    for c in list(root.findall("connection")):
        if any(c.get(a) in edge_kill or c.get(a) in lane_kill for a in attrs):
            root.remove(c)

    # ------------------------------------------------------------------ #
    # 4  scrub incLanes / intLanes lists of remaining junctions          #
    # ------------------------------------------------------------------ #
    # for each remaining junction, remove lanes that were part of the crossing
    for j in root.findall("junction"):
        # check if the junction has incLanes or intLanes attributes
        for attr in ("incLanes", "intLanes"):
            if attr in j.attrib:
                kept = [ln for ln in j.attrib[attr].split() if ln not in lane_kill]
                if kept:
                    j.attrib[attr] = " ".join(kept)
                else:
                    j.attrib.pop(attr)

    # ------------------------------------------------------------------ #
    # 5  write cleaned net                                               #
    # ------------------------------------------------------------------ #
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    print(f"Removed junctions: {', '.join(node_ids)}")

def generate_grid_network(seed, dimension, block_size_m, network_file, num_junction_to_remove):
    try:
        if (num_junction_to_remove > 0):
            # generate the full grid network first
            generate_full_grid_network(dimension, block_size_m, network_file_temp)

            # then pick random junctions to remove
            junctions_to_remove = pick_random_junction_ids(seed, num_junction_to_remove, dimension)

            # remove the selected junctions
            wipe_crossing(junctions_to_remove)

            # run netconvert ONCE so it recomputes incLanes/intLanes, updates shapes, …
            subprocess.run([
                "netconvert",
                "--sumo-net-file", str(network_file_prone),
                "--output-file", str(network_file)
            ], check=True)

            # remove temporary files
            os.remove(network_file_temp)
            os.remove(network_file_prone)
        else:
            generate_full_grid_network(dimension, block_size_m, network_file)

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during netgenerate execution:", e.stderr)

    if not os.path.exists(network_file):
        raise Exception(f"Error: Network file '{network_file}' was not created.")
