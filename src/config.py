# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from src.constants import (
    DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_TYPES, MIN_LANE_COUNT, MAX_LANE_COUNT,
    MIN_TREE_METHOD_INTERVAL, MAX_TREE_METHOD_INTERVAL, DEFAULT_WORKSPACE_DIR
)


class _Config:
    """Dynamic configuration class that allows workspace updates."""
    
    def __init__(self):
        # Initialize with default workspace
        self._output_dir = Path(DEFAULT_WORKSPACE_DIR)
        self._update_paths()
        
        # Initialize non-path attributes
        self._init_attributes()
    
    def _update_paths(self):
        """Update all dependent paths when workspace changes."""
        self.network_prefix = self._output_dir / "grid"
        self.network_file = Path(f"{self.network_prefix}.net.xml")
        self.network_nod_file = Path(f"{self.network_prefix}.nod.xml")
        self.network_edg_file = Path(f"{self.network_prefix}.edg.xml")
        self.network_con_file = Path(f"{self.network_prefix}.con.xml")
        self.network_tll_file = Path(f"{self.network_prefix}.tll.xml")
        self.config_file = self._output_dir / "grid.sumocfg"
        self.zones_file = self._output_dir / "zones.poly.xml"
        self.routes_file = self._output_dir / "vehicles.rou.xml"
    
    @property
    def output_dir(self) -> Path:
        """Get the current output directory."""
        return self._output_dir
    
    def update_workspace(self, workspace_path: str) -> None:
        """Update the workspace directory and regenerate all paths.
        
        The workspace_path specifies the parent directory where a 'workspace' 
        subdirectory will be created for simulation output files.
        """
        parent_dir = Path(workspace_path)
        self._output_dir = parent_dir / "workspace"
        self._update_paths()
    
    def _init_attributes(self) -> None:
        """Initialize non-path configuration attributes."""
        # ---------- land-use palette ----------
        self.land_uses = [
            {"name": "Residential", "percentage": 34, "max_size": 1000, "color": "#FFA500"},
            {"name": "Employment", "percentage": 10, "max_size": 500, "color": "#8B0000"},
            {"name": "Public Buildings", "percentage": 12, "max_size": 200, "color": "#000080"},
            {"name": "Mixed", "percentage": 24, "max_size": 300, "color": "#FFFF00"},
            {"name": "Entertainment/Retail", "percentage": 8, "max_size": 40, "color": "#006400"},
            {"name": "Public Open Space", "percentage": 12, "max_size": 100, "color": "#90EE90"},
        ]

        # ---------- vehicle generation ----------
        self.vehicle_types = {
            "passenger": {"length": 5.0, "maxSpeed": 13.9, "accel": 2.6, "decel": 4.5, "sigma": 0.5},
            "commercial": {"length": 12.0, "maxSpeed": 10.0, "accel": 1.3, "decel": 3.5, "sigma": 0.5},
            "public": {"length": 10.0, "maxSpeed": 11.1, "accel": 1.8, "decel": 4.0, "sigma": 0.5},
        }

        # Default vehicle type distribution (must sum to 100)
        self.default_vehicle_distribution = {"passenger": 60.0, "commercial": 30.0, "public": 10.0}

        # ---------- simulation parameters ----------
        self.RNG_SEED = 42
        self.DEFAULT_JUNCTION_RADIUS = 10.0  # meters
        self.HEAD_DISTANCE = 50  # head distance from the downstream end when splitting edges
        self.MIN_LANES = MIN_LANE_COUNT  # Backward compatibility alias
        self.MAX_LANES = MAX_LANE_COUNT  # Backward compatibility alias
        self.LAMBDA_DEPART = 3.5  # edge attractiveness
        self.LAMBDA_ARRIVE = 2.0

        # ---------- simulation verification ----------
        self.SIMULATION_VERIFICATION_FREQUENCY = 30  # Verify algorithm every N simulation steps

        # ---------- Tree Method algorithm timing ----------
        self.TREE_METHOD_ITERATION_INTERVAL_SEC = 90
        self.TREE_METHOD_MIN_INTERVAL_SEC = MIN_TREE_METHOD_INTERVAL  # Backward compatibility alias
        self.TREE_METHOD_MAX_INTERVAL_SEC = MAX_TREE_METHOD_INTERVAL  # Backward compatibility alias


@dataclass
class NetworkConfig:
    """Configuration for grid networks"""
    source_type: str = "grid"


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
