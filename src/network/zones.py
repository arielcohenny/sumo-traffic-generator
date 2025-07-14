import os
import random
from typing import Protocol
from xml.etree import ElementTree as ET
from shapely.geometry import box, mapping
import geopandas as gpd
from collections import deque
from src.config import CONFIG


class ThetaGenerator(Protocol):
    def sample(self, cell_id: str, land_use: str, zone_id: str) -> float: ...


def assign_land_use_to_zones(features, seed):
    """
    Assigns land use to zones using clustering algorithm from the paper.
    Based on "A Simulation Model for Intra-Urban Movements" methodology.
    """
    rng = random.Random(seed)

    total_cells = len(features)
    land_use_targets = [
        {**lu, "target": round(total_cells * lu["percentage"] / 100)} for lu in CONFIG.land_uses
    ]

    grid = {(feat['properties']['i'], feat['properties']['j'])
             : feat for feat in features}
    available = set(grid.keys())

    def get_neighbors(cell):
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (x + dx, y + dy) in available]

    for lu in land_use_targets:
        remaining = lu["target"]
        while remaining > 0 and available:
            start = rng.choice(list(available))
            cluster_size = min(remaining, lu["max_size"])
            cluster = set()
            queue = deque([start])
            while queue and len(cluster) < cluster_size:
                cell = queue.popleft()
                if cell in available:
                    cluster.add(cell)
                    available.remove(cell)
                    queue.extend(get_neighbors(cell))
            for cell in cluster:
                grid[cell]['properties']['land_use'] = lu['name']
                grid[cell]['properties']['color'] = lu['color']
            remaining -= len(cluster)

    for cell in available:
        lu = rng.choice(land_use_targets)
        grid[cell]['properties']['land_use'] = lu['name']
        grid[cell]['properties']['color'] = lu['color']


def extract_zones_from_junctions(cell_size: float, seed: int, fill_polygons: bool = False, inset: float = 0.0) -> None:
    """
    Extracts zones from raw SUMO files (nod/edg/con/tll) following the methodology
    from "A Simulation Model for Intra-Urban Movements" paper.

    Creates cellular grid zones based on junction coordinates from raw network files.
    Each zone represents a cell as described in the paper.

    Args:
        cell_size: Size of each cell in meters (paper uses 25x25m cells)
        seed: Random seed for land use assignment
        fill_polygons: Whether to fill polygons in SUMO visualization
        inset: Inset distance to shrink zones from boundaries
    """

    # Parse junction coordinates from raw .nod.xml file
    tree = ET.parse(CONFIG.network_nod_file)
    root = tree.getroot()

    junctions = []
    for node in root.findall("node"):
        node_id = node.get("id")
        # Skip internal nodes and split edge nodes (H_node)
        if not node_id.startswith(":") and "_H_node" not in node_id:
            x = float(node.get("x"))
            y = float(node.get("y"))
            junctions.append((x, y, node_id))

    if len(junctions) < 4:
        raise ValueError("Need at least 4 junctions to form a meaningful grid")

    # Extract unique x and y coordinates
    xs = sorted(set(x for x, _, _ in junctions))
    ys = sorted(set(y for _, y, _ in junctions))

    # Determine cell size if not provided
    if cell_size is None:
        dxs = [b - a for a, b in zip(xs, xs[1:]) if b > a]
        dys = [b - a for a, b in zip(ys, ys[1:]) if b > a]
        cell_size = min(min(dxs), min(dys))

    # Create zones based on cellular grid methodology from the paper
    # Subdivide the area between junctions into smaller cells based on cell_size
    features = []

    # Get network bounds
    network_xmin, network_xmax = min(xs), max(xs)
    network_ymin, network_ymax = min(ys), max(ys)

    # Calculate number of cells based on cell_size
    num_x_cells = max(1, int((network_xmax - network_xmin) / cell_size))
    num_y_cells = max(1, int((network_ymax - network_ymin) / cell_size))
    
    # Recalculate actual cell size to fit exactly within network bounds
    actual_cell_x = (network_xmax - network_xmin) / num_x_cells
    actual_cell_y = (network_ymax - network_ymin) / num_y_cells

    for i in range(num_x_cells):
        for j in range(num_y_cells):
            # Calculate cell boundaries based on cell_size subdivision
            xmin = network_xmin + i * actual_cell_x
            xmax = network_xmin + (i + 1) * actual_cell_x
            ymin = network_ymin + j * actual_cell_y
            ymax = network_ymin + (j + 1) * actual_cell_y

            # Apply inset if specified
            if inset > 0.0:
                xmin += inset
                xmax -= inset
                ymin += inset
                ymax -= inset

                # Skip cells that become too small after inset
                if xmax <= xmin or ymax <= ymin:
                    continue

            # Create zone geometry
            geom = box(xmin, ymin, xmax, ymax)
            zone_id = f"Z_{i}_{j}"

            # Add cell properties following the paper's methodology
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "zone_id": zone_id,
                    "i": i,
                    "j": j,
                    "cell_size": cell_size,
                    "center_x": (xmin + xmax) / 2,
                    "center_y": (ymin + ymax) / 2,
                    "area": cell_size * cell_size
                },
            })

    # Assign land uses using the paper's clustering algorithm
    assign_land_use_to_zones(features, seed)

    # Add attractiveness values (θᵢ) following normal distribution as in the paper
    rng = random.Random(seed)
    for feat in features:
        # Assign random attractiveness value following normal distribution
        # Using mean=0.5, std=0.2 to keep values mostly in [0,1] range
        theta = max(0.0, min(1.0, rng.normalvariate(0.5, 0.2)))
        feat['properties']['attractiveness'] = theta

    # Write GeoJSON file
    geojson_path = os.path.join(CONFIG.output_dir, "zones.geojson")
    gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").to_file(
        geojson_path, driver="GeoJSON"
    )

    # Write SUMO .poly.xml file
    poly_path = os.path.join(CONFIG.output_dir, "zones.poly.xml")
    layer = "0" if fill_polygons else "-1"
    fill_attr = "1" if fill_polygons else "0"

    with open(poly_path, "w", encoding="utf-8") as f:
        f.write("<additional>\n")
        for feat in features:
            coords = feat["geometry"]["coordinates"][0]
            shape_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            color = feat['properties'].get('color', "#000000")
            land_use = feat['properties'].get('land_use', 'Unknown')
            attractiveness = feat['properties'].get('attractiveness', 0.0)

            f.write(
                f"  <poly id=\"{feat['properties']['zone_id']}\" "
                f"color=\"{color}\" fill=\"{fill_attr}\" layer=\"{layer}\" "
                f"shape=\"{shape_str}\" type=\"{land_use}\" "
                f"attractiveness=\"{attractiveness:.3f}\"/>\n"
            )
        f.write("</additional>\n")

    print(f"Created {len(features)} zones based on cellular grid methodology")


def create_minimal_zones_for_osm(network_file: str, zones_file: str):
    """
    Create minimal zones for OSM networks as fallback when intelligent zone generation fails
    
    Args:
        network_file: Path to SUMO network file
        zones_file: Path to output zones polygon file
    """
    import sumolib
    from shapely.geometry import Polygon
    
    # Load network to get bounds
    net = sumolib.net.readNet(network_file)
    min_x, min_y, max_x, max_y = net.getBBoxXY()
    
    # Create a simple 3x3 grid of zones
    grid_size = 3
    cell_width = (max_x - min_x) / grid_size
    cell_height = (max_y - min_y) / grid_size
    
    # Zone type configurations
    zone_configs = {
        'residential': {'color': '0,255,0'},    # Green
        'commercial': {'color': '255,0,0'},     # Red  
        'mixed': {'color': '128,255,128'}       # Light green
    }
    
    zone_types = ['residential', 'commercial', 'mixed']
    
    # Write zones polygon file
    with open(zones_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<polygons xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.20">\n')
        
        zone_count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate cell boundaries
                cell_min_x = min_x + i * cell_width
                cell_max_x = min_x + (i + 1) * cell_width
                cell_min_y = min_y + j * cell_height
                cell_max_y = min_y + (j + 1) * cell_height
                
                # Assign zone type (cycle through types)
                zone_type = zone_types[(i + j) % len(zone_types)]
                color = zone_configs[zone_type]['color']
                
                # Create polygon coordinates
                coords = f"{cell_min_x:.2f},{cell_min_y:.2f} {cell_max_x:.2f},{cell_min_y:.2f} {cell_max_x:.2f},{cell_max_y:.2f} {cell_min_x:.2f},{cell_max_y:.2f} {cell_min_x:.2f},{cell_min_y:.2f}"
                
                f.write(f'  <poly id="osm_fallback_zone_{i}_{j}" type="{zone_type}" color="{color}" fill="1" shape="{coords}"/>\n')
                zone_count += 1
        
        f.write('</polygons>\n')
    
    print(f"Created {zone_count} minimal fallback zones")
