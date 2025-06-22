import os
from typing import Protocol
from xml.etree import ElementTree as ET
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import random
from src.config import CONFIG
from collections import deque

# -----------------------------
#  Interface for pluggable θᵢ
# -----------------------------


class ThetaGenerator(Protocol):
    def sample(self, cell_id: str, land_use: str, zone_id: str) -> float: ...


def assign_land_use_to_zones(features, seed):
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

# ------------------------------------------------------------
#  Main helper: build zone outline polygons from junction grid
# ------------------------------------------------------------
# insert - shrink each square on all sides


def extract_zones_from_junctions(
    net_file: str,
    cell_size: float,
    out_dir: str,
    seed,
    fill_polygons: bool = False,
    inset: float = 0.0,
) -> None:

    # ----------------------
    # 1. Parse junction grid
    # ----------------------
    tree = ET.parse(net_file)
    root = tree.getroot()

    xs: set[float] = set()
    ys: set[float] = set()
    for j in root.findall("junction"):
        jid = j.get("id")
        if jid.startswith(":"):
            # internal helper nodes are *not* part of the visible grid
            continue
        xs.add(float(j.get("x")))
        ys.add(float(j.get("y")))

    xs = sorted(xs)
    ys = sorted(ys)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(
            "Need at least two distinct x and y coordinates to form zones.")

    # ------------------------------------------------------
    # 2. Infer cell size if caller didn’t supply one explicitly
    # ------------------------------------------------------
    if cell_size is None:
        dxs = [b - a for a, b in zip(xs, xs[1:]) if b > a]
        dys = [b - a for a, b in zip(ys, ys[1:]) if b > a]
        cell_size = min(min(dxs), min(dys))

    # ---------------------------------------------
    # 3. Build a square per *interval* (xs[i]→xs[i+1])
    # ---------------------------------------------
    features = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            xmin, xmax = xs[i], xs[i + 1]
            ymin, ymax = ys[j], ys[j + 1]

            if inset > 0.0:
                xmin += inset
                xmax -= inset
                ymin += inset
                ymax -= inset

            geom = box(xmin, ymin, xmax, ymax)
            zone_id = f"Z_{i}_{j}"

            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"zone_id": zone_id, "i": i, "j": j},
            })

    assign_land_use_to_zones(features, seed)

    # -----------------
    # 4. Write GeoJSON
    # -----------------
    geojson_path = os.path.join(out_dir, "zones.geojson")
    gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").to_file(
        geojson_path, driver="GeoJSON"
    )

    # -------------------------
    # 5. Write SUMO .poly.xml
    # -------------------------
    poly_path = os.path.join(out_dir, "zones.poly.xml")
    layer = "0" if fill_polygons else "-1"
    fill_attr = "1" if fill_polygons else "0"
    # colour = "#A6CEE3" if fill_polygons else "#000000"  # outline colour

    with open(poly_path, "w", encoding="utf-8") as f:
        f.write("<additional>\n")
        for feat in features:
            coords = feat["geometry"]["coordinates"][0]
            shape_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            color = feat['properties'].get('color', "#000000")
            f.write(
                f"  <poly id=\"{feat['properties']['zone_id']}\" "
                f"color=\"{color}\" fill=\"{fill_attr}\" layer=\"{layer}\" "
                f"shape=\"{shape_str}\"/>\n"
            )
        f.write("</additional>\n")
