"""
Network helpers shared by the analysis scripts: which junction each edge leads to, and
free-flow data per edge, read from a run's outputs/grid.net.xml.gz.

Every road between two junctions consists of a tail edge (e.g. A1B1, from junction A1 to the
split node A1B1_H_node) and head edges per movement (A1B1_H_left, ... from the split node to
junction B1). Both belong to the approach of the downstream junction (B1).
"""

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def load_edges(run_dir: Path) -> pd.DataFrame:
    """One row per non-internal edge: edge, from, to, approach_junction, part (tail/head),
    n_lanes, length (m, first lane), speed (m/s, first lane), axis (NS/EW travel direction)."""
    with gzip.open(run_dir / "outputs" / "grid.net.xml.gz") as f:
        root = ET.parse(f).getroot()
    junction_type = {j.get("id"): j.get("type") for j in root.iter("junction")}
    xy = {j.get("id"): (float(j.get("x")), float(j.get("y"))) for j in root.iter("junction")}
    rows = []
    for e in root.iter("edge"):
        if e.get("function") == "internal":
            continue
        lanes = e.findall("lane")
        rows.append({"edge": e.get("id"), "from": e.get("from"), "to": e.get("to"), "n_lanes": len(lanes),
                     "length": float(lanes[0].get("length")), "speed": float(lanes[0].get("speed"))})
    df = pd.DataFrame(rows)

    # Head edges end at a junction; a tail edge ends at a split node whose head edges end at the junction
    head_to = df[df["from"].str.endswith("_H_node")].groupby("from")["to"].first()
    df["part"] = df["to"].str.endswith("_H_node").map({True: "tail", False: "head"})
    df["approach_junction"] = df.apply(lambda r: head_to.get(r["to"]) if r["part"] == "tail" else r["to"], axis=1)

    # Travel axis from the upstream junction of the road to the downstream junction
    def axis(r):
        start = r["from"] if r["part"] == "tail" else df.set_index("to").loc[r["from"], "from"]
        (x0, y0), (x1, y1) = xy[start], xy[r["approach_junction"]]
        return "EW" if abs(x1 - x0) >= abs(y1 - y0) else "NS"
    df["axis"] = df.apply(axis, axis=1)
    df["approach_is_signal"] = df["approach_junction"].map(lambda j: junction_type.get(j) == "traffic_light")
    return df
