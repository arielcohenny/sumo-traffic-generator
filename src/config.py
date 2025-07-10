# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class _Config:
    # ---------- paths ----------
    output_dir: Path = Path("data")
    network_prefix = output_dir / "grid"
    network_file: Path = f"{network_prefix}.net.xml"
    network_nod_file: Path = f"{network_prefix}.nod.xml"
    network_edg_file: Path = f"{network_prefix}.edg.xml"
    network_con_file: Path = f"{network_prefix}.con.xml"
    network_tll_file: Path = f"{network_prefix}.tll.xml"
    config_file: Path = output_dir / "grid.sumocfg"
    zones_file: Path = output_dir / "zones.poly.xml"
    routes_file: Path = output_dir / "vehicles.rou.xml"

    # ---------- land-use palette ----------
    land_uses: list[dict] = field(default_factory=lambda: [
        {"name": "Residential",           "percentage": 34,
            "max_size": 1000, "color": "#1f78b4"},
        {"name": "Employment",            "percentage": 10,
            "max_size":  500, "color": "#33a02c"},
        {"name": "Public Buildings",      "percentage": 12,
            "max_size":  200, "color": "#fb9a99"},
        {"name": "Mixed",                 "percentage": 24,
            "max_size":  300, "color": "#ff7f00"},
        {"name": "Entertainment/Retail",  "percentage":  8,
            "max_size":   40, "color": "#6a3d9a"},
        {"name": "Public Open Space",     "percentage": 12,
            "max_size":  100, "color": "#b2df8a"},
    ])

    # ---------- vehicle generation ----------
    vehicle_types: dict = field(default_factory=lambda: {
        "passenger":   {"length": 5.0,  "maxSpeed": 13.9, "accel": 2.6, "decel": 4.5, "sigma": 0.5},
        "commercial": {"length": 12.0, "maxSpeed": 10.0, "accel": 1.3, "decel": 3.5, "sigma": 0.5},
        "public":     {"length": 10.0, "maxSpeed": 11.1, "accel": 1.8, "decel": 4.0, "sigma": 0.5},
    })

    # Default vehicle type distribution (must sum to 100)
    default_vehicle_distribution: dict = field(
        default_factory=lambda: {"passenger": 60.0, "commercial": 30.0, "public": 10.0})

    DEFAULT_NUM_VEHICLES: int = 300
    RNG_SEED: int = 42

    # Default vehicle types string for CLI
    DEFAULT_VEHICLE_TYPES: str = "passenger 60 commercial 30 public 10"

    # ---------- simulation parameters ----------
    DEFAULT_JUNCTION_RADIUS: float = 10.0  # meters
    # ---------- head distance from the downstream end when splitting edges ----------
    HEAD_DISTANCE = 50
    # ---------- default number of lanes ---------
    MIN_LANES: int = 1
    MAX_LANES: int = 3
    # ---------- edge attractiveness ----------
    LAMBDA_DEPART = 3.5
    LAMBDA_ARRIVE = 2.0

    # ---------- simulation verification ----------
    # Verify algorithm every N simulation steps
    SIMULATION_VERIFICATION_FREQUENCY: int = 30


@dataclass
class OSMConfig:
    """Configuration for OSM import and processing"""
    osm_file_path: str = ""
    filter_highway_types: List[str] = field(default_factory=lambda: [
        "primary", "secondary", "tertiary", "residential", "unclassified"
    ])
    preserve_osm_lanes: bool = True
    min_edge_length: float = 20.0  # Minimum edge length for splitting
    zone_extraction_method: str = "osm_landuse"  # "osm_landuse", "hybrid", "cellular"
    traffic_light_strategy: str = "osm_preserve"  # "osm_preserve", "generate_all", "hybrid"


@dataclass
class NetworkConfig:
    """Unified config for both grid and OSM networks"""
    source_type: str = "grid"  # "grid" or "osm"
    osm_config: Optional[OSMConfig] = None


CONFIG = _Config()
