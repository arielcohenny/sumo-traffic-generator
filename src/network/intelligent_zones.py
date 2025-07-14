#!/usr/bin/env python3
"""
Intelligent Land Use Zone Generation System

This module implements a sophisticated approach for generating land use zones that works
for both OSM and synthetic networks. When OSM land use data is available, it uses actual
land use tags. When data is insufficient, it combines network topology analysis,
accessibility metrics, and OSM infrastructure analysis to intelligently infer zone types.

Key Features:
- Real OSM land use data extraction and processing
- Intelligent inference using network topology + accessibility + infrastructure analysis
- Lane count analysis for commercial vs residential classification
- Configurable grid resolution via --land_use_block_size_m parameter
- Unified system that works for both OSM and synthetic networks
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from pathlib import Path
import logging
import numpy as np
import json
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    logging.warning("geopandas not available - some features may be limited")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("networkx not available - network analysis will be simplified")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logging.warning("pandas not available - using simplified data structures")

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ZoneScore:
    """Score for different zone types at a grid cell"""
    residential: float = 0.0
    commercial: float = 0.0
    industrial: float = 0.0
    education: float = 0.0
    healthcare: float = 0.0
    mixed: float = 0.0
    
    def get_highest_score_type(self) -> str:
        """Get zone type with highest score"""
        scores = {
            'residential': self.residential,
            'commercial': self.commercial,
            'industrial': self.industrial,
            'education': self.education,
            'healthcare': self.healthcare,
            'mixed': self.mixed
        }
        return max(scores.items(), key=lambda x: x[1])[0]

class IntelligentZoneGenerator:
    """
    Generates intelligent land use zones using multiple analysis methods
    """
    
    # Zone type configurations with attractiveness multipliers
    ZONE_CONFIGS = {
        'residential': {
            'color': '0,255,0',  # Green
            'multiplier': 1.0,
            'departure_weight': 1.5,
            'arrival_weight': 0.8
        },
        'commercial': {
            'color': '255,0,0',  # Red
            'multiplier': 2.5,
            'departure_weight': 0.8,
            'arrival_weight': 2.0
        },
        'industrial': {
            'color': '128,128,128',  # Gray
            'multiplier': 1.8,
            'departure_weight': 0.9,
            'arrival_weight': 1.8
        },
        'education': {
            'color': '0,0,255',  # Blue
            'multiplier': 3.0,
            'departure_weight': 0.5,
            'arrival_weight': 3.5
        },
        'healthcare': {
            'color': '255,255,0',  # Yellow
            'multiplier': 2.2,
            'departure_weight': 0.3,
            'arrival_weight': 2.8
        },
        'mixed': {
            'color': '128,255,128',  # Light green
            'multiplier': 2.0,
            'departure_weight': 1.2,
            'arrival_weight': 1.8
        }
    }
    
    def __init__(self, land_use_block_size_m: float = 200.0):
        """
        Initialize intelligent zone generator
        
        Args:
            land_use_block_size_m: Size of grid cells in meters (default 200m)
        """
        self.block_size_m = land_use_block_size_m
        self.network_graph = None
        self.edge_data = {}
        self.junction_data = {}
        self.osm_pois = []
        self.osm_zones = []
        
    def load_network_data(self, network_json_file: str, network_bounds: Tuple[float, float, float, float]):
        """
        Load network data for topology analysis
        
        Args:
            network_json_file: Path to grid.net.json file
            network_bounds: (min_lon, min_lat, max_lon, max_lat)
        """
        logger.info("Loading network data for topology analysis...")
        
        with open(network_json_file, 'r') as f:
            network_data = json.load(f)
        
        # Create network graph (if networkx available)
        if HAS_NETWORKX:
            self.network_graph = nx.DiGraph()
        else:
            self.network_graph = None
        
        # Add edges with attributes
        for edge in network_data['edges_list']:
            edge_id = edge['id']
            from_junction = edge['from_junction']
            to_junction = edge['to_junction']
            lanes = edge['lanes']
            distance = edge['distance']
            
            if HAS_NETWORKX and self.network_graph is not None:
                self.network_graph.add_edge(from_junction, to_junction, 
                                          edge_id=edge_id, lanes=lanes, distance=distance)
            
            self.edge_data[edge_id] = {
                'lanes': lanes,
                'distance': distance,
                'from_junction': from_junction,
                'to_junction': to_junction
            }
        
        # Store junction data
        if 'junctions_dict' in network_data:
            self.junction_data = network_data['junctions_dict']
        
        logger.info(f"Loaded {len(self.edge_data)} edges and {len(self.junction_data)} junctions")
    
    def load_osm_data(self, osm_file_path: str) -> Dict:
        """
        Load and parse OSM data for land use and POI extraction
        
        Args:
            osm_file_path: Path to OSM file
            
        Returns:
            Dictionary with parsed OSM data
        """
        if not Path(osm_file_path).exists():
            logger.warning(f"OSM file not found: {osm_file_path}")
            return {'nodes': {}, 'ways': [], 'relations': []}
        
        logger.info(f"Loading OSM data from {osm_file_path}")
        
        tree = ET.parse(osm_file_path)
        root = tree.getroot()
        
        nodes = {}
        ways = []
        relations = []
        
        for element in root:
            if element.tag == 'node':
                node_id = element.get('id')
                lat = float(element.get('lat'))
                lon = float(element.get('lon'))
                
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
                
                nd_refs = []
                for nd in element.findall('nd'):
                    nd_refs.append(nd.get('ref'))
                
                tags = {}
                for tag in element.findall('tag'):
                    tags[tag.get('k')] = tag.get('v')
                
                ways.append({
                    'id': way_id,
                    'nodes': nd_refs,
                    'tags': tags
                })
                
            elif element.tag == 'relation':
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
        
        logger.info(f"Loaded {len(nodes)} nodes, {len(ways)} ways, {len(relations)} relations")
        return {'nodes': nodes, 'ways': ways, 'relations': relations}
    
    def extract_osm_zones_and_pois(self, osm_data: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract land use zones and POIs from OSM data
        
        Args:
            osm_data: Parsed OSM data
            
        Returns:
            Tuple of (zones, pois) lists
        """
        logger.info("Extracting OSM zones and POIs...")
        
        zones = []
        pois = []
        nodes = osm_data['nodes']
        
        # Extract POI points from nodes
        for node_id, node_data in nodes.items():
            tags = node_data['tags']
            
            if 'amenity' in tags:
                amenity_type = tags['amenity'].lower()
                point = Point(node_data['lon'], node_data['lat'])
                
                pois.append({
                    'id': node_id,
                    'geometry': point,
                    'amenity_type': amenity_type,
                    'tags': tags
                })
        
        # Extract land use zones from ways
        for way in osm_data['ways']:
            tags = way['tags']
            
            # Check for land use, amenity, or building tags
            zone_type = None
            for tag_key in ['landuse', 'amenity', 'building']:
                if tag_key in tags:
                    tag_value = tags[tag_key].lower()
                    
                    # Map OSM values to our zone types
                    if tag_value in ['residential', 'apartments', 'housing']:
                        zone_type = 'residential'
                    elif tag_value in ['commercial', 'retail', 'shop', 'office']:
                        zone_type = 'commercial'
                    elif tag_value in ['industrial']:
                        zone_type = 'industrial'
                    elif tag_value in ['school', 'university', 'college', 'education']:
                        zone_type = 'education'
                    elif tag_value in ['hospital', 'clinic', 'healthcare']:
                        zone_type = 'healthcare'
                    elif tag_value in ['mixed']:
                        zone_type = 'mixed'
                    
                    if zone_type:
                        break
            
            if zone_type:
                # Create polygon from way nodes
                coords = []
                for node_id in way['nodes']:
                    if node_id in nodes:
                        node = nodes[node_id]
                        coords.append((node['lon'], node['lat']))
                
                if len(coords) >= 3:
                    # Close polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    try:
                        polygon = Polygon(coords)
                        if polygon.is_valid and polygon.area > 0:
                            zones.append({
                                'id': way['id'],
                                'geometry': polygon,
                                'zone_type': zone_type,
                                'tags': tags,
                                'area_sqm': polygon.area * 111000 * 111000 * 0.6  # Rough conversion
                            })
                    except Exception as e:
                        logger.warning(f"Failed to create polygon from way {way['id']}: {e}")
        
        logger.info(f"Extracted {len(zones)} zones and {len(pois)} POIs from OSM")
        return zones, pois
    
    def analyze_network_topology(self, network_bounds: Tuple[float, float, float, float]) -> Dict[Tuple[int, int], float]:
        """
        Analyze network topology to identify commercial vs residential areas
        
        Args:
            network_bounds: (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary mapping grid coordinates to commercial likelihood scores
        """
        logger.info("Analyzing network topology...")
        
        if not self.network_graph:
            logger.warning("No network graph loaded")
            return {}
        
        topology_scores = {}
        
        # Calculate edge betweenness centrality
        centrality = {}
        if HAS_NETWORKX and self.network_graph is not None:
            try:
                centrality = nx.edge_betweenness_centrality(self.network_graph, weight='distance')
            except Exception as e:
                logger.warning(f"Could not calculate centrality: {e}")
                centrality = {}
        else:
            logger.info("NetworkX not available - using simplified topology analysis")
        
        # Create grid - handle both geographic and projected coordinates
        min_x, min_y, max_x, max_y = network_bounds
        
        # Check if coordinates are geographic (lat/lon) or projected
        if abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90:
            # Geographic coordinates - convert block size to degrees
            grid_size_unit = self.block_size_m / 111000  # Convert meters to degrees (rough)
        else:
            # Projected coordinates - use meters directly
            grid_size_unit = self.block_size_m
        
        num_cols = max(1, int((max_x - min_x) / grid_size_unit))
        num_rows = max(1, int((max_y - min_y) / grid_size_unit))
        
        network_center_x = (min_x + max_x) / 2
        network_center_y = (min_y + max_y) / 2
        
        for i in range(num_cols):
            for j in range(num_rows):
                cell_center_x = min_x + (i + 0.5) * grid_size_unit
                cell_center_y = min_y + (j + 0.5) * grid_size_unit
                
                score = 0.0
                
                # Factor 1: Distance from center (closer = more commercial)
                distance_from_center = np.sqrt((cell_center_x - network_center_x)**2 + 
                                             (cell_center_y - network_center_y)**2)
                max_distance = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
                center_score = 1.0 - (distance_from_center / max_distance) if max_distance > 0 else 0.5
                score += center_score * 0.3
                
                # Factor 2: Nearby edge characteristics
                nearby_lanes = []
                nearby_centrality = []
                
                for edge_id, edge_info in self.edge_data.items():
                    # This is simplified - in reality we'd need to check if edge is near cell
                    # For now, add some randomness based on lane count
                    if np.random.random() < 0.1:  # Sample 10% of edges
                        nearby_lanes.append(edge_info['lanes'])
                        
                        # Get centrality for this edge
                        edge_centrality = 0.0
                        for (from_node, to_node), cent_value in centrality.items():
                            if (from_node == edge_info['from_junction'] and 
                                to_node == edge_info['to_junction']):
                                edge_centrality = cent_value
                                break
                        nearby_centrality.append(edge_centrality)
                
                # Factor 3: Average lane count (higher lanes = more commercial)
                if nearby_lanes:
                    avg_lanes = np.mean(nearby_lanes)
                    lane_score = min(avg_lanes / 4.0, 1.0)  # Normalize by 4 lanes max
                    score += lane_score * 0.4
                
                # Factor 4: Edge centrality (higher centrality = more commercial)
                if nearby_centrality:
                    avg_centrality = np.mean(nearby_centrality)
                    # Normalize centrality (values are typically small)
                    centrality_score = min(avg_centrality * 1000, 1.0)
                    score += centrality_score * 0.3
                
                topology_scores[(i, j)] = min(score, 1.0)
        
        logger.info(f"Calculated topology scores for {len(topology_scores)} grid cells")
        return topology_scores
    
    def analyze_accessibility(self, network_bounds: Tuple[float, float, float, float]) -> Dict[Tuple[int, int], float]:
        """
        Analyze accessibility to identify commercial vs residential areas
        
        Args:
            network_bounds: (min_x, min_y, max_x, max_y) - can be geographic or projected
            
        Returns:
            Dictionary mapping grid coordinates to accessibility scores
        """
        logger.info("Analyzing accessibility...")
        
        accessibility_scores = {}
        
        # Create grid - handle both geographic and projected coordinates
        min_x, min_y, max_x, max_y = network_bounds
        
        # Check if coordinates are geographic (lat/lon) or projected
        if abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90:
            # Geographic coordinates - convert block size to degrees
            grid_size_unit = self.block_size_m / 111000  # Convert meters to degrees (rough)
        else:
            # Projected coordinates - use meters directly
            grid_size_unit = self.block_size_m
        
        num_cols = max(1, int((max_x - min_x) / grid_size_unit))
        num_rows = max(1, int((max_y - min_y) / grid_size_unit))
        
        # Calculate junction connectivity
        junction_connectivity = {}
        if HAS_NETWORKX and self.network_graph is not None:
            for junction_id in self.network_graph.nodes():
                in_degree = self.network_graph.in_degree(junction_id)
                out_degree = self.network_graph.out_degree(junction_id)
                junction_connectivity[junction_id] = in_degree + out_degree
        else:
            # Simplified connectivity analysis using junction data
            for junction_id in self.junction_data.keys():
                # Estimate connectivity from edge data
                connectivity = sum(1 for edge in self.edge_data.values() 
                                 if edge['from_junction'] == junction_id or edge['to_junction'] == junction_id)
                junction_connectivity[junction_id] = connectivity
        
        max_connectivity = max(junction_connectivity.values()) if junction_connectivity else 1
        
        for i in range(num_cols):
            for j in range(num_rows):
                score = 0.0
                
                # Factor 1: Junction connectivity (simplified - use average connectivity)
                # In reality, we'd find nearby junctions and use their connectivity
                if junction_connectivity:
                    avg_connectivity = np.mean(list(junction_connectivity.values()))
                    connectivity_score = avg_connectivity / max_connectivity
                    score += connectivity_score * 0.5
                
                # Factor 2: Network density (simplified)
                # Areas with more nearby edges are more accessible
                density_score = min(len(self.edge_data) / 100.0, 1.0)  # Normalize by 100 edges
                score += density_score * 0.3
                
                # Factor 3: Random variation for areas we can't analyze precisely
                score += np.random.uniform(0, 0.2)
                
                accessibility_scores[(i, j)] = min(score, 1.0)
        
        logger.info(f"Calculated accessibility scores for {len(accessibility_scores)} grid cells")
        return accessibility_scores
    
    def analyze_osm_infrastructure(self, network_bounds: Tuple[float, float, float, float]) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Analyze OSM infrastructure to identify zone types
        
        Args:
            network_bounds: (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary mapping grid coordinates to zone type scores
        """
        logger.info("Analyzing OSM infrastructure...")
        
        infrastructure_scores = {}
        
        # Create grid - handle both geographic and projected coordinates
        min_x, min_y, max_x, max_y = network_bounds
        
        # Check if coordinates are geographic (lat/lon) or projected
        if abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90:
            # Geographic coordinates - convert block size to degrees
            grid_size_unit = self.block_size_m / 111000  # Convert meters to degrees (rough)
        else:
            # Projected coordinates - use meters directly
            grid_size_unit = self.block_size_m
        
        num_cols = max(1, int((max_x - min_x) / grid_size_unit))
        num_rows = max(1, int((max_y - min_y) / grid_size_unit))
        
        for i in range(num_cols):
            for j in range(num_rows):
                cell_center_x = min_x + (i + 0.5) * grid_size_unit
                cell_center_y = min_y + (j + 0.5) * grid_size_unit
                cell_center = Point(cell_center_x, cell_center_y)
                
                zone_scores = ZoneScore()
                
                # Analyze nearby POIs
                search_radius = grid_size_unit * 1.5  # Search radius in same units as grid
                
                for poi in self.osm_pois:
                    distance = cell_center.distance(poi['geometry'])
                    if distance <= search_radius:
                        amenity_type = poi['amenity_type']
                        
                        # Distance-weighted influence (closer = more influence)
                        influence = max(0, 1.0 - (distance / search_radius))
                        
                        # Map amenity types to zone scores
                        if amenity_type in ['shop', 'restaurant', 'cafe', 'bank', 'supermarket']:
                            zone_scores.commercial += influence * 0.5
                        elif amenity_type in ['school', 'university', 'college', 'library']:
                            zone_scores.education += influence * 0.7
                        elif amenity_type in ['hospital', 'clinic', 'pharmacy']:
                            zone_scores.healthcare += influence * 0.6
                        elif amenity_type in ['bus_station', 'subway_station']:
                            zone_scores.mixed += influence * 0.4
                            zone_scores.commercial += influence * 0.3
                        elif amenity_type in ['parking']:
                            zone_scores.commercial += influence * 0.2
                
                # Analyze nearby OSM zones
                for zone in self.osm_zones:
                    if cell_center.distance(zone['geometry'].centroid) <= search_radius:
                        zone_type = zone['zone_type']
                        influence = 0.8  # OSM zones have high influence
                        
                        if zone_type == 'residential':
                            zone_scores.residential += influence
                        elif zone_type == 'commercial':
                            zone_scores.commercial += influence
                        elif zone_type == 'industrial':
                            zone_scores.industrial += influence
                        elif zone_type == 'education':
                            zone_scores.education += influence
                        elif zone_type == 'healthcare':
                            zone_scores.healthcare += influence
                        elif zone_type == 'mixed':
                            zone_scores.mixed += influence
                
                infrastructure_scores[(i, j)] = {
                    'residential': zone_scores.residential,
                    'commercial': zone_scores.commercial,
                    'industrial': zone_scores.industrial,
                    'education': zone_scores.education,
                    'healthcare': zone_scores.healthcare,
                    'mixed': zone_scores.mixed
                }
        
        logger.info(f"Calculated infrastructure scores for {len(infrastructure_scores)} grid cells")
        return infrastructure_scores
    
    def combine_scores_and_classify(self, 
                                  topology_scores: Dict[Tuple[int, int], float],
                                  accessibility_scores: Dict[Tuple[int, int], float],
                                  infrastructure_scores: Dict[Tuple[int, int], Dict[str, float]],
                                  network_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """
        Combine all analysis scores to classify each grid cell
        
        Args:
            topology_scores: Network topology analysis results
            accessibility_scores: Accessibility analysis results  
            infrastructure_scores: OSM infrastructure analysis results
            network_bounds: Network boundaries
            
        Returns:
            List of zone dictionaries
        """
        logger.info("Combining scores and classifying zones...")
        
        zones = []
        
        # Grid parameters - handle both geographic and projected coordinates
        min_x, min_y, max_x, max_y = network_bounds
        
        # Check if coordinates are geographic (lat/lon) or projected
        if abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90:
            # Geographic coordinates - convert block size to degrees
            grid_size_unit = self.block_size_m / 111000  # Convert meters to degrees (rough)
        else:
            # Projected coordinates - use meters directly
            grid_size_unit = self.block_size_m
        
        num_cols = max(1, int((max_x - min_x) / grid_size_unit))
        num_rows = max(1, int((max_y - min_y) / grid_size_unit))
        
        for i in range(num_cols):
            for j in range(num_rows):
                # Get scores for this cell
                topology_score = topology_scores.get((i, j), 0.0)
                accessibility_score = accessibility_scores.get((i, j), 0.0)
                infra_scores = infrastructure_scores.get((i, j), {})
                
                # Initialize final zone scores
                final_scores = ZoneScore()
                
                # Commercial bias from topology and accessibility
                commercial_bias = (topology_score * 0.6 + accessibility_score * 0.4)
                
                # Start with base residential assumption
                final_scores.residential = 0.5
                
                # Add commercial bias - boost it significantly
                final_scores.commercial = commercial_bias * 2.0  # Double the commercial bias
                
                # Add some variety based on grid position
                center_i, center_j = num_cols // 2, num_rows // 2
                distance_from_center = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                max_distance = np.sqrt(center_i**2 + center_j**2)
                
                if max_distance > 0:
                    # Center areas more likely to be commercial
                    center_factor = 1.0 - (distance_from_center / max_distance)
                    final_scores.commercial += center_factor * 1.5
                    
                    # Edge areas might be industrial
                    edge_factor = distance_from_center / max_distance
                    final_scores.industrial += edge_factor * 0.8
                
                # Add some education zones near residential areas
                if final_scores.residential > 0.7:
                    final_scores.education += 0.3
                
                # Add mixed-use zones in medium-density areas
                if 0.3 < commercial_bias < 0.7:
                    final_scores.mixed += 0.5
                
                # Add infrastructure scores
                for zone_type, score in infra_scores.items():
                    if zone_type == 'residential':
                        final_scores.residential += score * 0.7
                    elif zone_type == 'commercial':
                        final_scores.commercial += score * 0.7
                    elif zone_type == 'industrial':
                        final_scores.industrial += score * 0.7
                    elif zone_type == 'education':
                        final_scores.education += score * 0.7
                    elif zone_type == 'healthcare':
                        final_scores.healthcare += score * 0.7
                    elif zone_type == 'mixed':
                        final_scores.mixed += score * 0.7
                
                # Add some industrial potential to edge areas
                edge_distance = min(i, j, num_cols - 1 - i, num_rows - 1 - j)
                edge_factor = max(0, 1.0 - edge_distance / min(num_cols, num_rows) * 4)
                final_scores.industrial += edge_factor * 0.3
                
                # Determine final zone type
                zone_type = final_scores.get_highest_score_type()
                
                # Create zone polygon
                cell_min_x = min_x + i * grid_size_unit
                cell_max_x = min_x + (i + 1) * grid_size_unit
                cell_min_y = min_y + j * grid_size_unit
                cell_max_y = min_y + (j + 1) * grid_size_unit
                
                zone_polygon = Polygon([
                    (cell_min_x, cell_min_y),
                    (cell_max_x, cell_min_y),
                    (cell_max_x, cell_max_y),
                    (cell_min_x, cell_max_y),
                    (cell_min_x, cell_min_y)
                ])
                
                # Calculate area and capacity
                area_sqm = (self.block_size_m ** 2)
                capacity = area_sqm * 0.02  # Base capacity per sqm
                
                zones.append({
                    'id': f'intelligent_zone_{i}_{j}',
                    'geometry': zone_polygon,
                    'zone_type': zone_type,
                    'area_sqm': area_sqm,
                    'capacity': capacity,
                    'zone_info': self.ZONE_CONFIGS[zone_type],
                    'is_intelligent': True,
                    'grid_coords': (i, j),
                    'scores': {
                        'topology': topology_score,
                        'accessibility': accessibility_score,
                        'infrastructure': infra_scores,
                        'final': final_scores.__dict__
                    }
                })
        
        logger.info(f"Generated {len(zones)} intelligent zones")
        return zones
    
    def generate_intelligent_zones(self, 
                                 network_json_file: str,
                                 network_bounds: Tuple[float, float, float, float],
                                 osm_file_path: Optional[str] = None) -> List[Dict]:
        """
        Main method to generate intelligent land use zones
        
        Args:
            network_json_file: Path to grid.net.json file
            network_bounds: (min_lon, min_lat, max_lon, max_lat)
            osm_file_path: Optional path to OSM file for enhanced analysis
            
        Returns:
            List of zone dictionaries with intelligent land use classification
        """
        logger.info("Starting intelligent zone generation...")
        
        # Load network data
        self.load_network_data(network_json_file, network_bounds)
        
        # Load OSM data if available
        if osm_file_path:
            osm_data = self.load_osm_data(osm_file_path)
            self.osm_zones, self.osm_pois = self.extract_osm_zones_and_pois(osm_data)
        
        # Perform analysis
        topology_scores = self.analyze_network_topology(network_bounds)
        accessibility_scores = self.analyze_accessibility(network_bounds)
        infrastructure_scores = self.analyze_osm_infrastructure(network_bounds)
        
        # Combine scores and classify zones
        zones = self.combine_scores_and_classify(
            topology_scores, accessibility_scores, infrastructure_scores, network_bounds
        )
        
        logger.info(f"Successfully generated {len(zones)} intelligent zones")
        return zones
    
    def generate_intelligent_zones_from_osm(self, 
                                          osm_file_path: str,
                                          geographic_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """
        Generate intelligent zones directly from OSM file without requiring SUMO network
        
        Args:
            osm_file_path: Path to OSM file
            geographic_bounds: (min_lon, min_lat, max_lon, max_lat) in geographic coordinates
            
        Returns:
            List of zone dictionaries with intelligent land use classification
        """
        logger.info("Starting OSM-based intelligent zone generation...")
        
        # Load OSM data
        osm_data = self.load_osm_data(osm_file_path)
        self.osm_zones, self.osm_pois = self.extract_osm_zones_and_pois(osm_data)
        
        # Create grid based on geographic bounds and block size
        min_lon, min_lat, max_lon, max_lat = geographic_bounds
        
        # Convert block size from meters to degrees (rough approximation)
        # 1 degree ≈ 111,000 meters at equator
        grid_size_degrees = self.block_size_m / 111000
        
        num_cols = max(1, int((max_lon - min_lon) / grid_size_degrees))
        num_rows = max(1, int((max_lat - min_lat) / grid_size_degrees))
        
        logger.info(f"Creating {num_cols}x{num_rows} grid with {grid_size_degrees:.6f} degree cells")
        
        zones = []
        
        for i in range(num_cols):
            for j in range(num_rows):
                # Calculate cell boundaries in geographic coordinates
                cell_min_lon = min_lon + i * grid_size_degrees
                cell_max_lon = min_lon + (i + 1) * grid_size_degrees
                cell_min_lat = min_lat + j * grid_size_degrees
                cell_max_lat = min_lat + (j + 1) * grid_size_degrees
                
                cell_center_lon = (cell_min_lon + cell_max_lon) / 2
                cell_center_lat = (cell_min_lat + cell_max_lat) / 2
                cell_center = Point(cell_center_lon, cell_center_lat)
                
                # Analyze OSM data for this cell
                zone_scores = ZoneScore()
                
                # Check for nearby OSM zones
                search_radius = grid_size_degrees * 1.5
                for zone in self.osm_zones:
                    if cell_center.distance(zone['geometry'].centroid) <= search_radius:
                        zone_type = zone['zone_type']
                        influence = 0.8
                        
                        if zone_type == 'residential':
                            zone_scores.residential += influence
                        elif zone_type == 'commercial':
                            zone_scores.commercial += influence
                        elif zone_type == 'industrial':
                            zone_scores.industrial += influence
                        elif zone_type == 'education':
                            zone_scores.education += influence
                        elif zone_type == 'healthcare':
                            zone_scores.healthcare += influence
                        elif zone_type == 'mixed':
                            zone_scores.mixed += influence
                
                # Check for nearby POIs
                for poi in self.osm_pois:
                    distance = cell_center.distance(poi['geometry'])
                    if distance <= search_radius:
                        amenity_type = poi['amenity_type']
                        influence = max(0, 1.0 - (distance / search_radius))
                        
                        if amenity_type in ['shop', 'restaurant', 'cafe', 'bank', 'supermarket']:
                            zone_scores.commercial += influence * 0.5
                        elif amenity_type in ['school', 'university', 'college', 'library']:
                            zone_scores.education += influence * 0.7
                        elif amenity_type in ['hospital', 'clinic', 'pharmacy']:
                            zone_scores.healthcare += influence * 0.6
                
                # Add default residential assumption if no OSM data
                if (zone_scores.residential + zone_scores.commercial + zone_scores.industrial + 
                    zone_scores.education + zone_scores.healthcare + zone_scores.mixed) < 0.1:
                    zone_scores.residential = 0.5
                
                # Determine final zone type
                zone_type = zone_scores.get_highest_score_type()
                
                # Create zone polygon in geographic coordinates
                zone_polygon = Polygon([
                    (cell_min_lon, cell_min_lat),
                    (cell_max_lon, cell_min_lat),
                    (cell_max_lon, cell_max_lat),
                    (cell_min_lon, cell_max_lat),
                    (cell_min_lon, cell_min_lat)
                ])
                
                # Calculate area and capacity
                area_sqm = (self.block_size_m ** 2)
                capacity = area_sqm * 0.02
                
                zones.append({
                    'id': f'osm_zone_{i}_{j}',
                    'geometry': zone_polygon,
                    'zone_type': zone_type,
                    'area_sqm': area_sqm,
                    'capacity': capacity,
                    'zone_info': self.ZONE_CONFIGS[zone_type],
                    'is_intelligent': True,
                    'grid_coords': (i, j),
                    'geographic_coordinates': True  # Flag to indicate these need coordinate conversion later
                })
        
        logger.info(f"Generated {len(zones)} OSM-based intelligent zones")
        return zones

def save_intelligent_zones_to_poly_file(zones: List[Dict], output_file: str, net_file: str = None) -> None:
    """
    Save intelligent zones to SUMO polygon file format
    
    Args:
        zones: List of zone dictionaries
        output_file: Output polygon file path
        net_file: Optional SUMO network file for coordinate transformation
    """
    logger.info(f"Saving {len(zones)} intelligent zones to {output_file}")
    
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
            logger.info(f"Attempting to load network file: {net_file}")
            net = sumolib.net.readNet(net_file)
            logger.info(f"Successfully loaded network from {net_file} for coordinate transformation")
            
            # Get network bounds and projection info
            bbox = net.getBBoxXY()
            logger.info(f"Network bounding box (projected): {bbox}")
            
            # Get location info
            loc = net.getLocation()
            if loc:
                logger.info(f"Network location info: netOffset={loc.get('netOffset', 'None')}")
                logger.info(f"Network original boundary: {loc.get('origBoundary', 'None')}")
                logger.info(f"Network projection: {loc.get('projParameter', 'None')}")
            else:
                logger.warning("No location information available in network")
                
        except Exception as e:
            logger.error(f"Could not load network file {net_file}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Proceeding without coordinate transformation")
    else:
        logger.error(f"Network file issue - provided: {net_file}, exists: {Path(net_file).exists() if net_file else 'N/A'}")
        logger.warning("Proceeding without coordinate transformation")
    
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<polygons xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.20">\n')
        
        for zone in zones:
            zone_id = zone['id']
            zone_type = zone['zone_type']
            color = zone['zone_info']['color']
            
            # Transform coordinates if network is available
            def transform_coords(coords_list):
                if net is None:
                    logger.warning("No network available for coordinate transformation")
                    return coords_list
                
                transformed = []
                for lon, lat in coords_list:
                    try:
                        # Convert from geographic (lat/lon) to projected coordinates (UTM)
                        # Note: convertLonLat2XY expects (lon, lat) order
                        x, y = net.convertLonLat2XY(lon, lat)
                        logger.info(f"Transformed ({lon:.6f}, {lat:.6f}) -> ({x:.2f}, {y:.2f})")
                        transformed.append((x, y))
                    except Exception as e:
                        logger.error(f"Could not transform coordinate ({lon}, {lat}): {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # Fallback: use original coordinates
                        logger.warning(f"Using original coordinates as fallback: ({lon}, {lat})")
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
    
    logger.info(f"Successfully saved intelligent zones to {output_file}")

def convert_zones_to_projected_coordinates(zones_file: str, net_file: str) -> None:
    """
    Convert zone coordinates from geographic (lat/lon) to projected coordinates
    
    Args:
        zones_file: Path to zones polygon file
        net_file: Path to SUMO network file
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    # Import sumolib for coordinate transformation
    try:
        import sumolib
    except ImportError:
        logger.error("sumolib is required for coordinate transformation")
        raise ImportError("Please install sumolib: pip install sumolib")
    
    # Load network for coordinate transformation
    net = sumolib.net.readNet(net_file)
    logger.info(f"Loaded network from {net_file} for coordinate transformation")
    
    # Parse existing zones file
    tree = ET.parse(zones_file)
    root = tree.getroot()
    
    zones_converted = 0
    
    # Convert each polygon's coordinates
    for poly in root.findall('poly'):
        shape = poly.get('shape')
        if shape:
            # Parse coordinate pairs
            coord_pairs = shape.split(' ')
            converted_coords = []
            
            for coord_pair in coord_pairs:
                if ',' in coord_pair:
                    try:
                        lon, lat = map(float, coord_pair.split(','))
                        # Convert from geographic to projected coordinates
                        x, y = net.convertLonLat2XY(lon, lat)
                        converted_coords.append(f"{x:.2f},{y:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not convert coordinate {coord_pair}: {e}")
                        converted_coords.append(coord_pair)  # Keep original
            
            # Update the shape attribute with converted coordinates
            poly.set('shape', ' '.join(converted_coords))
            zones_converted += 1
    
    # Write the updated zones file
    tree.write(zones_file, encoding='utf-8', xml_declaration=True)
    
    logger.info(f"Converted coordinates for {zones_converted} zones from geographic to projected")

# Main function for testing
if __name__ == "__main__":
    # Example usage
    generator = IntelligentZoneGenerator(land_use_block_size_m=200.0)
    
    # Test with synthetic network
    network_json = "data/grid.net.json"
    network_bounds = (-73.990, 40.740, -73.970, 40.760)  # Example Manhattan bounds
    osm_file = "src/osm/export.osm"
    
    if Path(network_json).exists():
        try:
            zones = generator.generate_intelligent_zones(
                network_json_file=network_json,
                network_bounds=network_bounds,
                osm_file_path=osm_file if Path(osm_file).exists() else None
            )
            
            # Save zones
            save_intelligent_zones_to_poly_file(zones, "data/intelligent_zones.poly.xml", "data/grid.net.xml")
            
            print(f"Successfully generated {len(zones)} intelligent zones")
            
            # Print zone type distribution
            zone_types = {}
            for zone in zones:
                zone_type = zone['zone_type']
                zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
            
            print("Zone type distribution:")
            for zone_type, count in sorted(zone_types.items()):
                print(f"  {zone_type}: {count} zones")
                
        except Exception as e:
            print(f"Failed to generate intelligent zones: {e}")
    else:
        print(f"Network JSON file not found: {network_json}")