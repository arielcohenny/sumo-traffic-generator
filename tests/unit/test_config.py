"""
Unit tests for configuration system.

Tests individual configuration classes and validation logic
without external dependencies.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.config import CONFIG, OSMConfig, NetworkConfig


class TestGlobalConfig:
    """Test global CONFIG instance and defaults."""
    
    @pytest.mark.unit
    def test_config_instance_exists(self):
        """Test that CONFIG instance is accessible."""
        assert CONFIG is not None
        assert hasattr(CONFIG, 'output_dir')
        assert hasattr(CONFIG, 'vehicle_types')

    @pytest.mark.unit
    def test_vehicle_types_configuration(self):
        """Test vehicle types configuration."""
        assert 'passenger' in CONFIG.vehicle_types
        assert 'commercial' in CONFIG.vehicle_types
        assert 'public' in CONFIG.vehicle_types
        
        # Check passenger vehicle config
        passenger = CONFIG.vehicle_types['passenger']
        assert passenger['length'] == 5.0
        assert passenger['maxSpeed'] == 13.9

    @pytest.mark.unit
    def test_default_values(self):
        """Test default configuration values."""
        assert CONFIG.DEFAULT_NUM_VEHICLES == 300
        assert CONFIG.RNG_SEED == 42
        assert CONFIG.MIN_LANES == 1
        assert CONFIG.MAX_LANES == 3
        assert CONFIG.HEAD_DISTANCE == 50

    @pytest.mark.unit
    def test_file_paths(self):
        """Test configured file paths."""
        assert str(CONFIG.output_dir) == "workspace"
        assert "grid.net.xml" in str(CONFIG.network_file)
        assert "vehicles.rou.xml" in str(CONFIG.routes_file)
        assert "zones.poly.xml" in str(CONFIG.zones_file)


class TestOSMConfig:
    """Test OSM configuration class."""
    
    @pytest.mark.unit
    def test_osm_config_creation(self):
        """Test creation of OSM configuration."""
        osm_config = OSMConfig()
        
        assert osm_config.osm_file_path == ""
        assert isinstance(osm_config.filter_highway_types, list)
        assert osm_config.preserve_osm_lanes is True
        assert osm_config.min_edge_length == 20.0

    @pytest.mark.unit
    def test_osm_config_with_values(self):
        """Test OSM config with custom values."""
        osm_config = OSMConfig(
            osm_file_path="test.osm",
            preserve_osm_lanes=False,
            min_edge_length=30.0
        )
        
        assert osm_config.osm_file_path == "test.osm"
        assert osm_config.preserve_osm_lanes is False
        assert osm_config.min_edge_length == 30.0

    @pytest.mark.unit
    def test_highway_types_filter(self):
        """Test highway types filter configuration."""
        osm_config = OSMConfig()
        
        expected_types = ["primary", "secondary", "tertiary", "residential", "unclassified"]
        assert osm_config.filter_highway_types == expected_types


class TestNetworkConfig:
    """Test NetworkConfig class."""
    
    @pytest.mark.unit
    def test_network_config_grid(self):
        """Test network config for grid networks."""
        net_config = NetworkConfig(source_type="grid")
        
        assert net_config.source_type == "grid"
        assert net_config.osm_config is None

    @pytest.mark.unit
    def test_network_config_osm(self):
        """Test network config for OSM networks."""
        osm_config = OSMConfig(osm_file_path="test.osm")
        net_config = NetworkConfig(
            source_type="osm",
            osm_config=osm_config
        )
        
        assert net_config.source_type == "osm"
        assert net_config.osm_config is not None
        assert net_config.osm_config.osm_file_path == "test.osm"


class TestConfigIntegration:
    """Test configuration integration and combinations."""
    
    @pytest.mark.unit
    def test_global_config_with_network_config(self):
        """Test global CONFIG with network configurations."""
        # Test grid network config
        grid_net_config = NetworkConfig(source_type="grid")
        assert grid_net_config.source_type == "grid"
        
        # Test OSM network config
        osm_config = OSMConfig(osm_file_path="sample.osm")
        osm_net_config = NetworkConfig(
            source_type="osm", 
            osm_config=osm_config
        )
        assert osm_net_config.source_type == "osm"
        
        # Both should work with global CONFIG
        assert CONFIG.DEFAULT_NUM_VEHICLES == 300
        assert CONFIG.RNG_SEED == 42

    @pytest.mark.unit
    def test_vehicle_type_distribution(self):
        """Test default vehicle type distribution."""
        distribution = CONFIG.default_vehicle_distribution
        
        # Should sum to 100
        total = sum(distribution.values())
        assert total == 100.0
        
        # Should have all three types
        assert "passenger" in distribution
        assert "commercial" in distribution
        assert "public" in distribution

    @pytest.mark.unit
    def test_land_use_configuration(self):
        """Test land use configuration."""
        land_uses = CONFIG.land_uses
        
        # Should have expected land use types
        land_use_names = [lu["name"] for lu in land_uses]
        expected_names = ["Residential", "Employment", "Public Buildings", 
                         "Mixed", "Entertainment/Retail", "Public Open Space"]
        
        for name in expected_names:
            assert name in land_use_names
        
        # Percentages should sum to 100
        total_percentage = sum(lu["percentage"] for lu in land_uses)
        assert total_percentage == 100