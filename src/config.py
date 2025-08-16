# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass(frozen=True)
class _Config:
    # ---------- paths ----------
    output_dir: Path = Path("workspace")
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
            "max_size": 1000, "color": "#FFA500"},
        {"name": "Employment",            "percentage": 10,
            "max_size":  500, "color": "#8B0000"},
        {"name": "Public Buildings",      "percentage": 12,
            "max_size":  200, "color": "#000080"},
        {"name": "Mixed",                 "percentage": 24,
            "max_size":  300, "color": "#FFFF00"},
        {"name": "Entertainment/Retail",  "percentage":  8,
            "max_size":   40, "color": "#006400"},
        {"name": "Public Open Space",     "percentage": 12,
            "max_size":  100, "color": "#90EE90"},
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

    # ---------- Tree Method algorithm timing ----------
    # Tree Method calculations occur every N seconds to balance efficiency with responsiveness
    # This is independent of traffic light cycle timing
    TREE_METHOD_ITERATION_INTERVAL_SEC: int = 90

    # Valid range for Tree Method interval (in seconds)
    TREE_METHOD_MIN_INTERVAL_SEC: int = 30   # Minimum for responsiveness
    TREE_METHOD_MAX_INTERVAL_SEC: int = 300  # Maximum for efficiency


@dataclass
class OSMConfig:
    """Configuration for OSM import and processing"""
    osm_file_path: str = ""
    filter_highway_types: List[str] = field(default_factory=lambda: [
        "primary", "secondary", "tertiary", "residential", "unclassified"
    ])
    preserve_osm_lanes: bool = True
    min_edge_length: float = 20.0  # Minimum edge length for splitting
    # "osm_landuse", "hybrid", "cellular"
    zone_extraction_method: str = "osm_landuse"
    # "osm_preserve", "generate_all", "hybrid"
    traffic_light_strategy: str = "osm_preserve"


@dataclass
class NetworkConfig:
    """Unified config for both grid and OSM networks"""
    source_type: str = "grid"  # "grid" or "osm"
    osm_config: Optional[OSMConfig] = None


@dataclass
class CustomLaneConfig:
    """Configuration for custom edge lane definitions"""
    edge_configs: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def parse_custom_lanes(cls, custom_lanes_str: str) -> 'CustomLaneConfig':
        """Parse custom lanes string into structured configuration."""
        if not custom_lanes_str:
            return cls()

        config = cls()

        # Split by semicolon to get individual edge configurations (handle trailing semicolon)
        edge_configs = [cfg.strip() for cfg in custom_lanes_str.rstrip(
            ';').split(';') if cfg.strip()]

        for edge_config in edge_configs:
            edge_id, edge_data = cls._parse_single_edge_config(edge_config)
            config.edge_configs[edge_id] = edge_data

        return config

    @classmethod
    def parse_custom_lanes_file(cls, file_path: str) -> 'CustomLaneConfig':
        """Parse custom lanes file into structured configuration."""
        if not file_path:
            return cls()

        config = cls()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(
                f"Error reading custom lanes file {file_path}: {e}")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            try:
                # Handle multiple edge configs on same line (separated by semicolon)
                edge_configs = [cfg.strip()
                                for cfg in line.split(';') if cfg.strip()]
                for edge_config in edge_configs:
                    edge_id, edge_data = cls._parse_single_edge_config(
                        edge_config)
                    config.edge_configs[edge_id] = edge_data
            except Exception as e:
                raise ValueError(
                    f"Invalid custom lanes syntax on line {line_num}: {e}")

        return config

    @classmethod
    def _parse_single_edge_config(cls, edge_config: str) -> tuple:
        """Parse a single edge configuration into structured data."""
        if '=' not in edge_config:
            raise ValueError(f"Missing '=' in configuration: {edge_config}")

        edge_id, specification = edge_config.split('=', 1)
        edge_id = edge_id.strip()
        specification = specification.strip()

        # Parse specification - need to handle tail: and head: properly
        edge_data = {}

        # Look for tail: section
        if 'tail:' in specification:
            tail_start = specification.find('tail:')
            tail_end = specification.find(',head:', tail_start)
            if tail_end == -1:
                tail_end = len(specification)
            tail_part = specification[tail_start:tail_end].strip()

            tail_value = tail_part[5:].strip()  # Remove 'tail:'
            edge_data['tail_lanes'] = int(tail_value)

        # Look for head: section
        if 'head:' in specification:
            head_start = specification.find('head:')
            head_part = specification[head_start:].strip()

            head_value = head_part[5:].strip()  # Remove 'head:'

            # Handle dead-end case (empty head:)
            if not head_value:
                edge_data['movements'] = {}  # Empty movements = dead-end
            else:
                # Parse movement specifications: ToEdge1:N,ToEdge2:M
                movements = {}
                for movement in head_value.split(','):
                    movement = movement.strip()
                    if ':' not in movement:
                        raise ValueError(
                            f"Invalid movement format '{movement}' - must be ToEdge:N")

                    to_edge, lane_count = movement.split(':', 1)
                    movements[to_edge.strip()] = int(lane_count.strip())

                edge_data['movements'] = movements

        return edge_id, edge_data

    def get_tail_lanes(self, edge_id: str) -> Optional[int]:
        """Get tail lane count for specific edge."""
        return self.edge_configs.get(edge_id, {}).get('tail_lanes')

    def get_movements(self, edge_id: str) -> Optional[Dict[str, int]]:
        """Get all movements for specific edge."""
        return self.edge_configs.get(edge_id, {}).get('movements')

    def has_custom_config(self, edge_id: str) -> bool:
        """Check if edge has custom configuration."""
        return edge_id in self.edge_configs


CONFIG = _Config()
