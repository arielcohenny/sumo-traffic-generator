# src/config.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class _Config:
    # ---------- paths ----------
    output_dir: Path = Path("data")
    network_file: Path = output_dir / "grid.net.xml"
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
        "car":   {"length": 5.0,  "maxSpeed": 13.9},
        "truck": {"length": 12.0, "maxSpeed": 10.0},
        "bus":   {"length": 10.0, "maxSpeed": 11.1},
    })

    # weights must align with keys order above
    vehicle_weights: list[float] = field(
        default_factory=lambda: [0.6, 0.3, 0.1])   # car, truck, bus

    DEFAULT_NUM_VEHICLES: int = 300
    RNG_SEED: int = 42

    # ---------- simulation parameters ----------
    DEFAULT_JUNCTION_RADIUS: float = 10.0  # meters


CONFIG = _Config()
