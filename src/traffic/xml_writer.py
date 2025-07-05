# src/traffic/xml_writer.py
from __future__ import annotations
from xml.etree.ElementTree import Element, SubElement, ElementTree
from pathlib import Path
from typing import Dict, List

def write_routes(outfile: str | Path,
                 vehicles: List[dict],
                 vehicle_types: Dict[str, dict]) -> None:
    """
    Produce a SUMO-compatible .rou.xml.
    """
    root = Element("routes")

    # vType definitions
    for vt_id, attrs in vehicle_types.items():
        SubElement(root, "vType", id=vt_id,
                   **{k: str(v) for k, v in attrs.items()})

    # vehicle entries
    for v in vehicles:
        # Add routing_strategy as a vehicle attribute if present
        veh_attrs = {
            "id": v["id"], 
            "type": v["type"], 
            "depart": str(v["depart"])
        }
        if "routing_strategy" in v:
            veh_attrs["routing_strategy"] = v["routing_strategy"]
        
        veh = SubElement(root, "vehicle", **veh_attrs)
        SubElement(veh, "route", edges=" ".join(v["route_edges"]))

    ElementTree(root).write(outfile, encoding="utf-8", xml_declaration=True)
