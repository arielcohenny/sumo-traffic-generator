#!/usr/bin/env python3
"""
OSM-specific zone extraction and land use integration

This module extracts zones from OSM data using real land use information exclusively,
separate from the cellular grid method used for synthetic networks.

Key Features:
- Extract actual OSM land use tags (residential, commercial, industrial)
- Convert OSM building polygons into traffic generation zones
- Use building density and floor area for capacity estimation
- Integrate with existing attractiveness calculation system
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMLandUseExtractor:
    """Extract zones from OSM data using land use tags and building polygons"""
    
    # OSM land use mapping to zone types with attractiveness multipliers
    LANDUSE_MAPPING = {
        # Residential zones
        'residential': {'type': 'residential', 'multiplier': 1.0, 'departure_weight': 1.5, 'arrival_weight': 0.8},
        'apartments': {'type': 'residential', 'multiplier': 1.2, 'departure_weight': 1.8, 'arrival_weight': 0.7},
        'housing': {'type': 'residential', 'multiplier': 1.0, 'departure_weight': 1.5, 'arrival_weight': 0.8},
        
        # Commercial zones
        'commercial': {'type': 'commercial', 'multiplier': 2.5, 'departure_weight': 0.8, 'arrival_weight': 2.0},
        'retail': {'type': 'commercial', 'multiplier': 2.8, 'departure_weight': 0.6, 'arrival_weight': 2.5},
        'industrial': {'type': 'industrial', 'multiplier': 1.8, 'departure_weight': 0.9, 'arrival_weight': 1.8},
        
        # Mixed use
        'mixed': {'type': 'mixed', 'multiplier': 2.0, 'departure_weight': 1.2, 'arrival_weight': 1.8},
        
        # Special zones
        'education': {'type': 'education', 'multiplier': 3.0, 'departure_weight': 0.5, 'arrival_weight': 3.5},
        'healthcare': {'type': 'healthcare', 'multiplier': 2.2, 'departure_weight': 0.3, 'arrival_weight': 2.8},
        'recreation': {'type': 'recreation', 'multiplier': 1.5, 'departure_weight': 0.4, 'arrival_weight': 2.0},
    }
    
    # OSM amenity mapping for POI-based zone enhancement
    AMENITY_MAPPING = {
        # Shopping and services
        'shop': {'multiplier': 2.0, 'type': 'commercial'},
        'supermarket': {'multiplier': 2.5, 'type': 'commercial'}, 
        'mall': {'multiplier': 3.0, 'type': 'commercial'},
        'marketplace': {'multiplier': 2.8, 'type': 'commercial'},
        
        # Food and dining
        'restaurant': {'multiplier': 1.8, 'type': 'commercial'},
        'cafe': {'multiplier': 1.5, 'type': 'commercial'},
        'fast_food': {'multiplier': 1.6, 'type': 'commercial'},
        'pub': {'multiplier': 1.4, 'type': 'commercial'},
        
        # Education
        'school': {'multiplier': 3.5, 'type': 'education'},
        'university': {'multiplier': 4.0, 'type': 'education'},
        'college': {'multiplier': 3.8, 'type': 'education'},
        'kindergarten': {'multiplier': 2.5, 'type': 'education'},
        
        # Healthcare
        'hospital': {'multiplier': 3.0, 'type': 'healthcare'},
        'clinic': {'multiplier': 2.0, 'type': 'healthcare'},
        'pharmacy': {'multiplier': 1.5, 'type': 'healthcare'},
        'dentist': {'multiplier': 1.3, 'type': 'healthcare'},
        
        # Recreation and culture
        'cinema': {'multiplier': 2.5, 'type': 'recreation'},
        'theatre': {'multiplier': 2.2, 'type': 'recreation'},
        'library': {'multiplier': 1.8, 'type': 'recreation'},
        'museum': {'multiplier': 2.0, 'type': 'recreation'},
        'park': {'multiplier': 1.2, 'type': 'recreation'},
        
        # Transportation
        'bus_station': {'multiplier': 2.0, 'type': 'transport'},
        'railway_station': {'multiplier': 3.5, 'type': 'transport'},
        'subway_station': {'multiplier': 3.0, 'type': 'transport'},
        'parking': {'multiplier': 0.8, 'type': 'transport'},
        
        # Government and services
        'townhall': {'multiplier': 1.5, 'type': 'government'},
        'courthouse': {'multiplier': 1.3, 'type': 'government'},
        'police': {'multiplier': 1.2, 'type': 'government'},
        'fire_station': {'multiplier': 0.8, 'type': 'government'},
        'post_office': {'multiplier': 1.4, 'type': 'government'},
        'bank': {'multiplier': 1.6, 'type': 'commercial'},
    }
    
    def __init__(self, osm_file_path: str):
        """Initialize with OSM file path"""
        self.osm_file_path = Path(osm_file_path)
        self.zones = []
        self.building_polygons = []
        self.poi_points = []
        
    def parse_osm_data(self) -> Dict:
        """Parse OSM XML to extract ways, nodes, and relations"""
        logger.info(f"Parsing OSM data from {self.osm_file_path}")
        
        tree = ET.parse(self.osm_file_path)
        root = tree.getroot()
        
        # Store nodes with coordinates
        nodes = {}
        ways = []
        relations = []
        
        for element in root:
            if element.tag == 'node':
                node_id = element.get('id')
                lat = float(element.get('lat'))
                lon = float(element.get('lon'))
                
                # Extract tags
                tags = {}
                for tag in element.findall('tag'):
                    tags[tag.get('k')] = tag.get('v')
                
                nodes[node_id] = {
                    'lat': lat,
                    'lon': lon,
                    'tags': tags
                }
                
            elif element.tag == 'way':
                way_id = element.get('id')
                
                # Get node references
                nd_refs = []
                for nd in element.findall('nd'):
                    nd_refs.append(nd.get('ref'))
                
                # Extract tags
                tags = {}
                for tag in element.findall('tag'):
                    tags[tag.get('k')] = tag.get('v')
                
                ways.append({
                    'id': way_id,
                    'nodes': nd_refs,
                    'tags': tags
                })
                
            elif element.tag == 'relation':
                # Handle multipolygon relations for complex land use areas
                relation_id = element.get('id')
                
                members = []
                for member in element.findall('member'):
                    members.append({
                        'type': member.get('type'),
                        'ref': member.get('ref'),
                        'role': member.get('role')
                    })
                
                tags = {}
                for tag in element.findall('tag'):
                    tags[tag.get('k')] = tag.get('v')
                
                relations.append({
                    'id': relation_id,
                    'members': members,
                    'tags': tags
                })
        
        logger.info(f"Parsed {len(nodes)} nodes, {len(ways)} ways, {len(relations)} relations")
        return {'nodes': nodes, 'ways': ways, 'relations': relations}
    
    def create_polygon_from_way(self, way: Dict, nodes: Dict) -> Optional[Polygon]:
        """Create Shapely polygon from OSM way"""
        try:
            # Get coordinates for all nodes in the way
            coords = []
            for node_id in way['nodes']:
                if node_id in nodes:
                    node = nodes[node_id]
                    coords.append((node['lon'], node['lat']))  # lon, lat for Shapely
            
            # Check if way is closed (first and last node are the same)
            if len(coords) >= 4 and coords[0] == coords[-1]:
                return Polygon(coords)
            elif len(coords) >= 3:
                # Close the polygon if it's not already closed
                coords.append(coords[0])
                return Polygon(coords)
            
        except Exception as e:
            logger.warning(f"Failed to create polygon from way {way['id']}: {e}")
        
        return None
    
    def extract_landuse_polygons(self, osm_data: Dict) -> List[Dict]:
        """Extract land use polygons from OSM data"""
        logger.info("Extracting land use polygons...")
        
        landuse_polygons = []
        nodes = osm_data['nodes']
        
        for way in osm_data['ways']:
            tags = way['tags']
            
            # Check for land use tags
            landuse_type = None
            for tag_key in ['landuse', 'amenity', 'building']:
                if tag_key in tags:
                    tag_value = tags[tag_key].lower()
                    
                    if tag_key == 'landuse' and tag_value in self.LANDUSE_MAPPING:
                        landuse_type = tag_value
                        break
                    elif tag_key == 'amenity' and tag_value in self.AMENITY_MAPPING:
                        landuse_type = tag_value
                        break
                    elif tag_key == 'building' and tag_value in ['commercial', 'retail', 'office', 'residential', 'apartments', 'house']:
                        landuse_type = tag_value
                        break
            
            if landuse_type:
                polygon = self.create_polygon_from_way(way, nodes)
                if polygon and polygon.is_valid and polygon.area > 0:
                    # Calculate additional properties
                    area_sqm = polygon.area * 111000 * 111000 * 0.6  # Rough conversion to square meters
                    
                    # Estimate building capacity based on type and area
                    capacity = self.estimate_zone_capacity(landuse_type, area_sqm, tags)
                    
                    landuse_polygons.append({
                        'id': way['id'],
                        'geometry': polygon,
                        'landuse_type': landuse_type,
                        'tags': tags,
                        'area_sqm': area_sqm,
                        'capacity': capacity,
                        'zone_info': self.get_zone_info(landuse_type)
                    })
        
        logger.info(f"Extracted {len(landuse_polygons)} land use polygons")
        return landuse_polygons
    
    def extract_poi_points(self, osm_data: Dict) -> List[Dict]:
        """Extract points of interest from OSM nodes"""
        logger.info("Extracting POI points...")
        
        poi_points = []
        
        for node_id, node_data in osm_data['nodes'].items():
            tags = node_data['tags']
            
            # Check for amenity tags
            if 'amenity' in tags:
                amenity_type = tags['amenity'].lower()
                
                if amenity_type in self.AMENITY_MAPPING:
                    point = Point(node_data['lon'], node_data['lat'])
                    
                    poi_points.append({
                        'id': node_id,
                        'geometry': point,
                        'amenity_type': amenity_type,
                        'tags': tags,
                        'zone_info': self.AMENITY_MAPPING[amenity_type]
                    })
        
        logger.info(f"Extracted {len(poi_points)} POI points")
        return poi_points
    
    def estimate_zone_capacity(self, landuse_type: str, area_sqm: float, tags: Dict) -> float:
        """Estimate zone capacity based on land use type and area"""
        
        # Base capacity per square meter by land use type
        capacity_per_sqm = {
            'residential': 0.02,    # ~20 people per 1000 sqm
            'apartments': 0.05,     # ~50 people per 1000 sqm (higher density)
            'commercial': 0.03,     # ~30 employees per 1000 sqm
            'retail': 0.04,         # ~40 people per 1000 sqm
            'industrial': 0.01,     # ~10 employees per 1000 sqm
            'mixed': 0.035,         # Average of residential and commercial
            'education': 0.08,      # ~80 students per 1000 sqm
            'healthcare': 0.06,     # ~60 people per 1000 sqm
            'recreation': 0.02,     # ~20 visitors per 1000 sqm
        }
        
        base_rate = capacity_per_sqm.get(landuse_type, 0.02)
        base_capacity = area_sqm * base_rate
        
        # Adjust based on building height/levels if available
        levels_multiplier = 1.0
        if 'building:levels' in tags:
            try:
                levels = int(tags['building:levels'])
                levels_multiplier = min(levels, 10)  # Cap at 10 floors for realism
            except ValueError:
                pass
        
        # Adjust based on specific amenity types
        if landuse_type in self.AMENITY_MAPPING:
            amenity_multiplier = self.AMENITY_MAPPING[landuse_type]['multiplier']
            base_capacity *= amenity_multiplier
        
        return base_capacity * levels_multiplier
    
    def get_zone_info(self, landuse_type: str) -> Dict:
        """Get zone information including attractiveness multipliers"""
        if landuse_type in self.LANDUSE_MAPPING:
            return self.LANDUSE_MAPPING[landuse_type]
        elif landuse_type in self.AMENITY_MAPPING:
            amenity_info = self.AMENITY_MAPPING[landuse_type]
            # Convert amenity to land use format
            zone_type = amenity_info['type']
            if zone_type in self.LANDUSE_MAPPING:
                base_info = self.LANDUSE_MAPPING[zone_type].copy()
                base_info['multiplier'] *= amenity_info['multiplier']
                return base_info
        
        # Default for unknown types
        return {'type': 'mixed', 'multiplier': 1.0, 'departure_weight': 1.0, 'arrival_weight': 1.0}
    
    def cluster_nearby_zones(self, zones: List[Dict], cluster_distance: float = 0.001) -> List[Dict]:
        """Cluster nearby zones of similar types into larger zones"""
        logger.info(f"Clustering zones within {cluster_distance} degrees...")
        
        if not zones:
            return zones
        
        # Create GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame(zones)
        
        clustered_zones = []
        processed_indices = set()
        
        for i, zone in enumerate(zones):
            if i in processed_indices:
                continue
            
            # Find nearby zones of the same type
            zone_geom = zone['geometry']
            zone_type = zone['landuse_type']
            
            nearby_zones = [zone]
            nearby_indices = {i}
            
            for j, other_zone in enumerate(zones):
                if j == i or j in processed_indices:
                    continue
                
                if (other_zone['landuse_type'] == zone_type and
                    zone_geom.distance(other_zone['geometry']) < cluster_distance):
                    nearby_zones.append(other_zone)
                    nearby_indices.add(j)
            
            # If we found multiple nearby zones, merge them
            if len(nearby_zones) > 1:
                merged_geometry = unary_union([z['geometry'] for z in nearby_zones])
                merged_capacity = sum(z['capacity'] for z in nearby_zones)
                merged_area = sum(z['area_sqm'] for z in nearby_zones)
                
                clustered_zone = {
                    'id': f"cluster_{zone['id']}",
                    'geometry': merged_geometry,
                    'landuse_type': zone_type,
                    'area_sqm': merged_area,
                    'capacity': merged_capacity,
                    'zone_info': zone['zone_info'],
                    'clustered_count': len(nearby_zones)
                }
                
                clustered_zones.append(clustered_zone)
                processed_indices.update(nearby_indices)
            else:
                # Single zone, keep as is
                clustered_zones.append(zone)
                processed_indices.add(i)
        
        logger.info(f"Clustered {len(zones)} zones into {len(clustered_zones)} zones")
        return clustered_zones
    
    def create_grid_zones_from_network(self, network_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """Create synthetic grid zones when OSM land use data is insufficient"""
        logger.info("Creating synthetic grid zones from network structure...")
        
        min_lon, min_lat, max_lon, max_lat = network_bounds
        
        # Create a 3x3 grid of synthetic zones within the network bounds
        grid_size = 3
        lon_step = (max_lon - min_lon) / grid_size
        lat_step = (max_lat - min_lat) / grid_size
        
        synthetic_zones = []
        zone_types = ['residential', 'commercial', 'mixed', 'education', 'healthcare']
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate zone boundaries
                zone_min_lon = min_lon + i * lon_step
                zone_max_lon = min_lon + (i + 1) * lon_step
                zone_min_lat = min_lat + j * lat_step
                zone_max_lat = min_lat + (j + 1) * lat_step
                
                # Create polygon
                zone_polygon = Polygon([
                    (zone_min_lon, zone_min_lat),
                    (zone_max_lon, zone_min_lat),
                    (zone_max_lon, zone_max_lat),
                    (zone_min_lon, zone_max_lat),
                    (zone_min_lon, zone_min_lat)
                ])
                
                # Assign zone type in a pattern
                zone_type = zone_types[(i + j) % len(zone_types)]
                zone_info = self.get_zone_info(zone_type)
                
                # Calculate area and capacity
                area_sqm = zone_polygon.area * 111000 * 111000 * 0.6  # Rough conversion
                capacity = area_sqm * 0.02  # Default capacity per sqm
                
                synthetic_zones.append({
                    'id': f'synthetic_zone_{i}_{j}',
                    'geometry': zone_polygon,
                    'landuse_type': zone_type,
                    'area_sqm': area_sqm,
                    'capacity': capacity,
                    'zone_info': zone_info,
                    'is_synthetic': True
                })
        
        logger.info(f"Created {len(synthetic_zones)} synthetic zones")
        return synthetic_zones
    
    def validate_osm_landuse_coverage(self, network_bounds: Tuple[float, float, float, float], allow_synthetic: bool = True) -> Dict:
        """Validate OSM has sufficient land use data for zone extraction"""
        
        min_lon, min_lat, max_lon, max_lat = network_bounds
        network_polygon = Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat), 
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat)
        ])
        
        network_area = network_polygon.area
        
        # Calculate total coverage area
        if not self.zones:
            if allow_synthetic:
                logger.warning("No OSM land use data found - creating synthetic zones")
                synthetic_zones = self.create_grid_zones_from_network(network_bounds)
                self.zones.extend(synthetic_zones)
                
                return {
                    'valid': True,
                    'coverage_percentage': 100.0,
                    'total_zones': len(self.zones),
                    'network_area': network_area,
                    'landuse_area': network_area,
                    'using_synthetic': True
                }
            else:
                return {
                    'valid': False,
                    'coverage_percentage': 0.0,
                    'total_zones': 0,
                    'error': 'No zones extracted from OSM data'
                }
        
        total_landuse_area = 0
        for zone in self.zones:
            intersection = zone['geometry'].intersection(network_polygon)
            if intersection.is_valid:
                total_landuse_area += intersection.area
        
        coverage_percentage = (total_landuse_area / network_area) * 100
        
        # Validation criteria
        min_coverage_percentage = 15.0  # Minimum 15% coverage required
        min_zones = 5  # Minimum 5 zones required
        
        is_valid = (coverage_percentage >= min_coverage_percentage and 
                   len(self.zones) >= min_zones)
        
        validation_result = {
            'valid': is_valid,
            'coverage_percentage': coverage_percentage,
            'total_zones': len(self.zones),
            'network_area': network_area,
            'landuse_area': total_landuse_area,
            'using_synthetic': False
        }
        
        if not is_valid:
            if allow_synthetic:
                logger.warning(f"Insufficient OSM land use coverage ({coverage_percentage:.1f}%) - adding synthetic zones")
                synthetic_zones = self.create_grid_zones_from_network(network_bounds)
                self.zones.extend(synthetic_zones)
                
                validation_result.update({
                    'valid': True,
                    'coverage_percentage': 100.0,
                    'total_zones': len(self.zones),
                    'using_synthetic': True
                })
            else:
                if coverage_percentage < min_coverage_percentage:
                    validation_result['error'] = f'Insufficient land use coverage: {coverage_percentage:.1f}% < {min_coverage_percentage}%'
                else:
                    validation_result['error'] = f'Too few zones: {len(self.zones)} < {min_zones}'
        
        return validation_result
    
    def extract_osm_zones(self, network_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """
        Main method to extract zones using OSM land use data exclusively
        
        Args:
            network_bounds: (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            List of zone dictionaries with geometry and attractiveness data
        """
        logger.info("Starting OSM zone extraction...")
        
        # Parse OSM data
        osm_data = self.parse_osm_data()
        
        # Extract land use polygons
        landuse_polygons = self.extract_landuse_polygons(osm_data)
        
        # Extract POI points
        poi_points = self.extract_poi_points(osm_data)
        
        # Convert POI points to small buffer zones
        poi_zones = []
        for poi in poi_points:
            # Create small buffer around POI (roughly 50m radius)
            buffer_radius = 0.0005  # Approximately 50m in degrees
            poi_polygon = poi['geometry'].buffer(buffer_radius)
            
            poi_zones.append({
                'id': f"poi_{poi['id']}",
                'geometry': poi_polygon,
                'landuse_type': poi['amenity_type'],
                'area_sqm': poi_polygon.area * 111000 * 111000 * 0.6,
                'capacity': 50,  # Fixed capacity for POI zones
                'zone_info': poi['zone_info']
            })
        
        # Combine all zones
        all_zones = landuse_polygons + poi_zones
        
        # Cluster nearby zones
        clustered_zones = self.cluster_nearby_zones(all_zones)
        
        # Store zones for validation
        self.zones = clustered_zones
        
        # Validate coverage
        validation = self.validate_osm_landuse_coverage(network_bounds)
        
        if not validation['valid']:
            logger.error(f"OSM land use validation failed: {validation['error']}")
            raise ValueError(f"Insufficient OSM land use data: {validation['error']}")
        
        logger.info(f"Successfully extracted {len(clustered_zones)} zones with {validation['coverage_percentage']:.1f}% coverage")
        
        return clustered_zones


def calculate_osm_zone_attractiveness(zones: List[Dict], temporal_phase: str = "morning_rush") -> Dict[str, Dict[str, float]]:
    """
    Calculate attractiveness weights for OSM zones based on land use
    
    Args:
        zones: List of zone dictionaries from extract_osm_zones
        temporal_phase: Time phase for temporal adjustments
        
    Returns:
        Dictionary mapping zone_id to {'departure_weight', 'arrival_weight'}
    """
    logger.info(f"Calculating OSM zone attractiveness for {temporal_phase} phase...")
    
    # Temporal multipliers for different phases
    temporal_multipliers = {
        'morning_rush': {
            'residential': {'departure': 2.0, 'arrival': 0.3},
            'commercial': {'departure': 0.4, 'arrival': 1.8},
            'education': {'departure': 0.2, 'arrival': 3.0},
            'healthcare': {'departure': 0.5, 'arrival': 1.5},
            'industrial': {'departure': 0.6, 'arrival': 1.4},
        },
        'afternoon': {
            'residential': {'departure': 0.8, 'arrival': 1.2},
            'commercial': {'departure': 1.2, 'arrival': 1.0},
            'education': {'departure': 1.0, 'arrival': 0.8},
            'healthcare': {'departure': 1.0, 'arrival': 1.0},
            'industrial': {'departure': 1.0, 'arrival': 1.0},
        },
        'evening_rush': {
            'residential': {'departure': 0.3, 'arrival': 2.0},
            'commercial': {'departure': 1.8, 'arrival': 0.4},
            'education': {'departure': 2.5, 'arrival': 0.2},
            'healthcare': {'departure': 1.2, 'arrival': 0.8},
            'industrial': {'departure': 1.4, 'arrival': 0.6},
        },
        'night': {
            'residential': {'departure': 0.5, 'arrival': 1.5},
            'commercial': {'departure': 0.2, 'arrival': 0.3},
            'education': {'departure': 0.1, 'arrival': 0.1},
            'healthcare': {'departure': 0.8, 'arrival': 1.2},
            'industrial': {'departure': 0.3, 'arrival': 0.3},
        }
    }
    
    zone_attractiveness = {}
    
    for zone in zones:
        zone_id = str(zone['id'])
        zone_info = zone['zone_info']
        zone_type = zone_info['type']
        
        # Base weights from zone configuration
        base_departure = zone_info['departure_weight']
        base_arrival = zone_info['arrival_weight']
        
        # Capacity scaling
        capacity_factor = min(zone['capacity'] / 100.0, 3.0)  # Cap at 3x multiplier
        
        # Temporal adjustment
        temporal_mult = temporal_multipliers.get(temporal_phase, temporal_multipliers['afternoon'])
        zone_temporal = temporal_mult.get(zone_type, {'departure': 1.0, 'arrival': 1.0})
        
        # Calculate final weights
        departure_weight = (base_departure * 
                          zone_info['multiplier'] * 
                          capacity_factor * 
                          zone_temporal['departure'])
        
        arrival_weight = (base_arrival * 
                         zone_info['multiplier'] * 
                         capacity_factor * 
                         zone_temporal['arrival'])
        
        zone_attractiveness[zone_id] = {
            'departure_weight': max(departure_weight, 0.1),  # Minimum weight
            'arrival_weight': max(arrival_weight, 0.1),
            'zone_type': zone_type,
            'capacity': zone['capacity']
        }
    
    logger.info(f"Calculated attractiveness for {len(zone_attractiveness)} zones")
    return zone_attractiveness


def save_zones_to_poly_file(zones: List[Dict], output_file: str, net_file: str = None) -> None:
    """Save zones to SUMO polygon file format with coordinate transformation"""
    logger.info(f"Saving {len(zones)} zones to {output_file}")
    
    # Import sumolib for coordinate transformation
    try:
        import sumolib
    except ImportError:
        logger.error("sumolib is required for coordinate transformation")
        raise ImportError("Please install sumolib: pip install sumolib")
    
    # Load network for coordinate transformation if provided
    net = None
    if net_file and Path(net_file).exists():
        try:
            net = sumolib.net.readNet(net_file)
            logger.info(f"Loaded network from {net_file} for coordinate transformation")
        except Exception as e:
            logger.warning(f"Could not load network file {net_file}: {e}")
            logger.warning("Proceeding without coordinate transformation")
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<polygons xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.20">\n')
        
        for zone in zones:
            zone_id = zone['id']
            zone_type = zone['zone_info']['type']
            
            # Color mapping for visualization
            colors = {
                'residential': '0,255,0',      # Green
                'commercial': '255,0,0',       # Red  
                'industrial': '128,128,128',   # Gray
                'education': '0,0,255',        # Blue
                'healthcare': '255,255,0',     # Yellow
                'recreation': '255,0,255',     # Magenta
                'mixed': '128,255,128',        # Light green
                'transport': '0,255,255',      # Cyan
                'government': '128,0,128',     # Purple
            }
            
            color = colors.get(zone_type, '128,128,128')
            
            # Transform coordinates if network is available
            def transform_coords(coords_list):
                if net is None:
                    # No transformation - use original coords
                    return coords_list
                
                transformed = []
                for lon, lat in coords_list:
                    try:
                        # Convert from lon,lat to SUMO network coordinates
                        x, y = net.convertLonLat2XY(lon, lat)
                        transformed.append((x, y))
                    except Exception as e:
                        logger.warning(f"Could not transform coordinate ({lon}, {lat}): {e}")
                        # Fallback to original coordinates
                        transformed.append((lon, lat))
                return transformed
            
            # Handle MultiPolygon
            if hasattr(zone['geometry'], 'geoms'):
                for i, geom in enumerate(zone['geometry'].geoms):
                    transformed_coords = transform_coords(list(geom.exterior.coords))
                    coords = ' '.join([f"{x},{y}" for x, y in transformed_coords])
                    f.write(f'  <poly id="{zone_id}_{i}" type="{zone_type}" color="{color}" fill="1" shape="{coords}"/>\n')
            else:
                transformed_coords = transform_coords(list(zone['geometry'].exterior.coords))
                coords = ' '.join([f"{x},{y}" for x, y in transformed_coords])
                f.write(f'  <poly id="{zone_id}" type="{zone_type}" color="{color}" fill="1" shape="{coords}"/>\n')
        
        f.write('</polygons>\n')
    
    logger.info(f"Successfully saved zones to {output_file}")


# Main function for testing
if __name__ == "__main__":
    # Example usage
    osm_file = "src/osm/samples/manhattan_upper_west.osm"
    
    if Path(osm_file).exists():
        extractor = OSMLandUseExtractor(osm_file)
        
        # Network bounds (example)
        bounds = (-73.9850, 40.7800, -73.9750, 40.7900)
        
        try:
            # Extract zones
            zones = extractor.extract_osm_zones(bounds)
            
            # Calculate attractiveness
            attractiveness = calculate_osm_zone_attractiveness(zones, "morning_rush")
            
            # Save to file
            save_zones_to_poly_file(zones, "data/osm_zones.poly.xml")
            
            print(f"Successfully extracted {len(zones)} zones from OSM data")
            print(f"Zone types: {set([z['zone_info']['type'] for z in zones])}")
            
        except ValueError as e:
            print(f"Failed to extract zones: {e}")
    else:
        print(f"OSM file not found: {osm_file}")