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
        routing_strategy = v.get("routing_strategy", "shortest")

        # Base attributes common to both trips and vehicles
        base_attrs = {
            "id": v["id"],
            "type": v["type"],
            "depart": str(v["depart"])
        }

        # Realtime vehicles use <trip> with from/to attributes
        # SUMO automatically routes them at departure based on current traffic
        if routing_strategy == "realtime":
            trip_attrs = {
                **base_attrs,
                "from": v["from_edge"],
                "to": v["to_edge"]
            }
            trip_elem = SubElement(root, "trip", **trip_attrs)
            # Store routing strategy as param for TraCI to read
            SubElement(trip_elem, "param", key="routing_strategy", value=routing_strategy)

        # Other strategies (shortest, fastest, attractiveness) use <vehicle> with full routes
        else:
            veh_elem = SubElement(root, "vehicle", **base_attrs)
            SubElement(veh_elem, "route", edges=" ".join(v["route_edges"]))
            # Store routing strategy as param for TraCI to read
            SubElement(veh_elem, "param", key="routing_strategy", value=routing_strategy)

    ElementTree(root).write(outfile, encoding="utf-8", xml_declaration=True)
