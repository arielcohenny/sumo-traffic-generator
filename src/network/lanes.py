import xml.etree.ElementTree as ET
import random
from pathlib import Path
import subprocess


def repair_connectivity_in_place(net_path: str | Path) -> None:
    """
    For every <connection …> that goes from lane 0→0, clone it so that
    lane i of the *from* edge connects to lane i of the *to* edge for all
    i = 1 … min(n_from, n_to)-1.  Overwrites the .net.xml in place.
    """
    net_path = Path(net_path)
    tree = ET.parse(net_path)
    root = tree.getroot()

    # cache number-of-lanes per edge
    lanes_per_edge: dict[str, int] = {
        e.get("id"): len(e.findall("lane"))
        for e in root.findall("edge")
        if e.get("function") != "internal"
    }

    new_conns: list[ET.Element] = []
    for conn in root.findall("connection"):
        # consider only templates that currently connect lane 0→0
        if conn.get("fromLane") != "0" or conn.get("toLane") != "0":
            continue
        frm, to = conn.get("from"), conn.get("to")
        n_from = lanes_per_edge.get(frm, 1)
        n_to = lanes_per_edge.get(to,   1)
        for i in range(1, min(n_from, n_to)):
            dup = ET.Element("connection", attrib=conn.attrib)
            dup.set("fromLane", str(i))
            dup.set("toLane",   str(i))
            new_conns.append(dup)

    # append the clones and write back
    root.extend(new_conns)
    tree.write(net_path, encoding="utf-8")


def set_lane_counts(net_file_in: str | Path,
                    net_file_out: str | Path,
                    seed: int,
                    min_lanes: int,
                    max_lanes: int) -> None:
    """
    Rewrites every edge in the .net.xml file to have a random number of lanes (1–3).

    Parameters
    ----------
    net_file_in : path to input .net.xml
    net_file_out : path to save modified net.xml
    seed : global random seed
    min_lanes : minimum lanes per edge (inclusive)
    max_lanes : maximum lanes per edge (inclusive)
    """
    tree = ET.parse(net_file_in)
    root = tree.getroot()

    rng = random.Random(seed)

    for edge in root.findall("edge"):
        if "function" in edge.attrib and edge.attrib["function"] == "internal":
            continue  # skip turn edges / internal edges

        old_lanes = edge.findall("lane")
        if not old_lanes:
            continue

        # choose new number of lanes
        num_lanes = rng.randint(min_lanes, max_lanes)
        speed = old_lanes[0].get("speed")
        length = old_lanes[0].get("length")

        # remove all existing <lane> elements
        for ln in old_lanes:
            edge.remove(ln)

        # keep the shape of the first lane
        shape = old_lanes[0].get("shape")

        # add new ones
        # print(f"Setting {num_lanes} lanes for edge {edge.get('id')}")
        # for i in range(num_lanes):
        #     print(f"Adding lane {i} to edge {edge.get('id')}")
        #     ET.SubElement(edge, "lane", {
        #         "id": f"{edge.get('id')}_{i}",
        #         "index": str(i),
        #         "speed": speed,
        #         "length": length,
        #         "shape": shape
        #     })

        # keep the shape of the first lane
        # fallback shape if missing
        shape = old_lanes[0].get("shape") or "0.00,0.00 1.00,0.00"

        for i in range(num_lanes):
            lane_attrs = {
                "id": f"{edge.get('id')}_{i}",
                "index": str(i),
                "speed": speed,
                "length": length,
                "shape": shape
            }
            ET.SubElement(edge, "lane", lane_attrs)

    Path(net_file_out).parent.mkdir(parents=True, exist_ok=True)
    tree.write(net_file_out, encoding="UTF-8", xml_declaration=True)

    repair_connectivity_in_place(net_file_out)

    subprocess.run([
        "netconvert",
        "--sumo-net-file", str(net_file_out),
        "--output-file", str(net_file_out),
        # aggregate-warnings: aggregates warnings of the same type whenever more than 1 occur
        # warnings can be removed completely by using --no-warnings=true
        "--aggregate-warnings=1",
    ], check=True)
