import subprocess
import xml.etree.ElementTree as ET
from alive_progress import *
from pathlib import Path
from src.sim.sumo_utils import *


def inject_traffic_lights(
    net_file: str,
    program_id: str = "0",
    green_duration: int = 30,
    yellow_duration: int = 3,
) -> None:
    """
    Inject static traffic-light logic into an existing SUMO network file in place,
    ensuring proper linkIndex and connection-to-TL mappings to eliminate unused-state warnings.

    Workflow:
    1) Run netconvert to assign linkIndex attributes, producing a temporary link-indexed file.
    2) Parse the link-indexed file, remove any existing <tlLogic> blocks.
    3) For each junction, collect its controlled <connection> elements, assign 'tl' and linkIndex,
       then inject a new <tlLogic> with phases, using linkIndexByTLS to map indices correctly.
    4) Restore each connection's original 'state' attribute.
    5) Overwrite the original net_file with the modified network and clean up.

    Parameters:
        net_file: Path to the .net.xml file to modify in place.
        program_id: The programID attribute for each <tlLogic>.
        green_duration: Duration (seconds) for green phases.
        yellow_duration: Duration (seconds) for yellow phases.
    """
    path = Path(net_file)

    # Backup original connection states
    orig_tree = ET.parse(path)
    orig_root = orig_tree.getroot()
    state_map = {}
    for c in orig_root.findall("connection"):
        key = (c.get("from"), c.get("to"), c.get("via"), c.get("dir"))
        state_map[key] = c.get("state")

    # 1) Run netconvert to assign linkIndex
    linkindexed = path.with_suffix(".linkindexed.net.xml")
    subprocess.run([
        "netconvert",
        "--sumo-net-file", str(path),
        "--output-file",    str(linkindexed),
        "--aggregate-warnings=1",
    ], check=True)

    # 2) Parse link-indexed file and remove old tlLogic
    tree = ET.parse(linkindexed)
    root = tree.getroot()
    for old in root.findall("tlLogic"):
        root.remove(old)

    all_conns = root.findall("connection")

    # 3) Inject new tlLogic per junction
    # Keep tlLogic BEFORE any <connection> so SUMO sees the controller first
    # Determine insertion index: just after the <location> element if present,
    # otherwise at the very beginning of <net>
    insert_at = 0
    for i, elem in enumerate(root):
        if elem.tag == "location":
            insert_at = i + 1
            break

    for j in root.findall("junction"):
        jid = j.get("id")
        if jid.startswith(":"):
            continue  # skip internal junctions
        j.set("type", "traffic_light")

        # Controlled connections for this junction (all internal ones)
        conns_here = [
            c for c in all_conns
            if c.get("from", "").startswith(f":{jid}_")
        ]
        if not conns_here:
            continue

        # Assign 'tl' and linkIndex to each connection
        for idx, c in enumerate(conns_here):
            c.set("tl", jid)
            c.set("linkIndex", str(idx))

        # Helper to classify NS vs EW
        def is_ns(conn):
            return conn.get("dir") in ("s", "t")

        # Build phases, skipping all-red
        phases = [
            (green_duration,  "".join("G" if is_ns(c) else "r" for c in conns_here)),
            (yellow_duration, "".join("y" if is_ns(c) else "r" for c in conns_here)),
            (green_duration,  "".join("r" if is_ns(c) else "G" for c in conns_here)),
            (yellow_duration, "".join("r" if is_ns(c) else "y" for c in conns_here)),
        ]

        tl_attribs = {
            "id": jid,
            "type": "static",
            "programID": program_id,
            "offset": "0",
            "linkIndexByTLS": "true",
        }
        tl = ET.Element("tlLogic", **tl_attribs)
        for dur, state in phases:
            if all(ch == "r" for ch in state):
                continue
            ET.SubElement(tl, "phase", duration=str(dur), state=state)

        # Insert the tlLogic before any <connection> elements
        root.insert(insert_at, tl)
        insert_at += 1  # keep subsequent tlLogics in order

    # 4) Restore connection states
    for c in root.findall("connection"):
        key = (c.get("from"), c.get("to"), c.get("via"), c.get("dir"))
        orig_state = state_map.get(key)
        if orig_state is not None:
            c.set("state", orig_state)

    # 5) Write back to original file and cleanup
    tree.write(path, encoding="utf-8", xml_declaration=True)
    try:
        linkindexed.unlink()
    except:
        pass
