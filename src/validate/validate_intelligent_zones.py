"""
Intelligent zones validation functions for the SUMO traffic generator.

This module provides validation functions for intelligent zone generation and processing,
specifically for OSM-based networks with coordinate transformations and coverage validation.
"""

import sumolib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from src.config import CONFIG
from src.validate.errors import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "verify_convert_zones_to_projected_coordinates",
]


def verify_convert_zones_to_projected_coordinates(zones_file: str, network_file: str) -> None:
    """
    Validate that intelligent zones adequately cover the network bounds after coordinate conversion.
    
    This function validates the convert_zones_to_projected_coordinates function by checking:
    1. Zone coordinates are properly transformed from geographic to projected coordinates
    2. Zone bounds adequately cover the network bounds (>= 90% coverage)
    3. Zone coordinate format is valid
    4. Network and zone files exist and are readable
    
    Args:
        zones_file: Path to zones polygon file
        network_file: Path to SUMO network file
        
    Raises:
        ValidationError: If validation fails
    """
    logger.info("Validating zone coverage against network bounds...")
    
    # Check that required files exist
    if not Path(zones_file).exists():
        raise ValidationError(f"Zones file not found: {zones_file}")
    
    if not Path(network_file).exists():
        raise ValidationError(f"Network file not found: {network_file}")
    
    try:
        # Load network for bounds checking
        net = sumolib.net.readNet(network_file)
        net_bounds = net.getBBoxXY()
        logger.info(f"Network bounds: {net_bounds}")
        
        # Parse zone coordinates
        tree = ET.parse(zones_file)
        root = tree.getroot()
        zone_coords = []
        
        for poly in root.findall('poly'):
            shape = poly.get('shape')
            if shape:
                for coord_pair in shape.split(' '):
                    if ',' in coord_pair:
                        try:
                            x, y = map(float, coord_pair.split(','))
                            zone_coords.append((x, y))
                        except ValueError as e:
                            logger.warning(f"Invalid coordinate format: {coord_pair}")
                            continue
        
        if not zone_coords:
            raise ValidationError("No valid zone coordinates found in zones file")
        
        # Calculate zone bounds
        zone_bounds = (
            min(c[0] for c in zone_coords),
            min(c[1] for c in zone_coords),
            max(c[0] for c in zone_coords),
            max(c[1] for c in zone_coords)
        )
        logger.info(f"Zone bounds: {zone_bounds}")
        
        # Validate coordinate system consistency
        # Both network and zones should be in projected coordinates (typically much larger than geographic)
        if (abs(net_bounds[0]) < 180 and abs(net_bounds[1]) < 180 and 
            abs(net_bounds[2]) < 180 and abs(net_bounds[3]) < 180):
            logger.warning("Network bounds appear to be in geographic coordinates - this may indicate transformation issues")
        
        if (abs(zone_bounds[0]) < 180 and abs(zone_bounds[1]) < 180 and 
            abs(zone_bounds[2]) < 180 and abs(zone_bounds[3]) < 180):
            logger.warning("Zone bounds appear to be in geographic coordinates - coordinate transformation may have failed")
        
        # Check coverage adequacy
        net_width = net_bounds[2] - net_bounds[0]
        net_height = net_bounds[3] - net_bounds[1]
        zone_width = zone_bounds[2] - zone_bounds[0]
        zone_height = zone_bounds[3] - zone_bounds[1]
        
        if net_width <= 0 or net_height <= 0:
            raise ValidationError(f"Invalid network bounds: width={net_width}, height={net_height}")
        
        if zone_width <= 0 or zone_height <= 0:
            raise ValidationError(f"Invalid zone bounds: width={zone_width}, height={zone_height}")
        
        # Calculate coverage ratios
        coverage_x = zone_width / net_width
        coverage_y = zone_height / net_height
        
        logger.info(f"Zone coverage - X: {coverage_x:.2f}, Y: {coverage_y:.2f}")
        
        # Validate coverage meets minimum requirements
        min_coverage = 0.9  # 90% minimum coverage
        
        if coverage_x < min_coverage:
            raise ValidationError(
                f"Insufficient X-axis coverage: {coverage_x:.2f} < {min_coverage:.2f}. "
                f"Consider using smaller --land_use_block_size_m or check coordinate transformation"
            )
        
        if coverage_y < min_coverage:
            raise ValidationError(
                f"Insufficient Y-axis coverage: {coverage_y:.2f} < {min_coverage:.2f}. "
                f"Consider using smaller --land_use_block_size_m or check coordinate transformation"
            )
        
        # Additional validation: check for coordinate transformation artifacts
        _validate_coordinate_transformation_quality(zone_coords, net_bounds)
        
        logger.info(f"Zone coverage validation passed - X: {coverage_x:.2f}, Y: {coverage_y:.2f}")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Zone coverage validation failed: {e}")


def _validate_coordinate_transformation_quality(zone_coords: List[Tuple[float, float]], 
                                              net_bounds: Tuple[float, float, float, float]) -> None:
    """
    Internal helper to validate coordinate transformation quality
    
    Args:
        zone_coords: List of (x, y) zone coordinates
        net_bounds: Network bounds (min_x, min_y, max_x, max_y)
    """
    # Check for obvious coordinate transformation issues
    
    # 1. Check for coordinates that are clearly outside reasonable bounds
    for x, y in zone_coords:
        if abs(x) > 1e8 or abs(y) > 1e8:
            logger.warning(f"Coordinate appears unreasonably large: ({x}, {y}) - possible transformation error")
        
        if x == 0 and y == 0:
            logger.warning("Found (0,0) coordinate - possible failed transformation")
    
    # 2. Check for clusters of identical coordinates (transformation failures)
    coord_counts = {}
    for coord in zone_coords:
        coord_counts[coord] = coord_counts.get(coord, 0) + 1
    
    duplicate_threshold = len(zone_coords) * 0.1  # More than 10% identical coordinates is suspicious
    max_duplicates = max(coord_counts.values())
    
    if max_duplicates > duplicate_threshold:
        logger.warning(f"High number of duplicate coordinates ({max_duplicates}) - possible transformation issue")
    
    # 3. Check coordinate distribution relative to network bounds
    zone_center_x = sum(x for x, y in zone_coords) / len(zone_coords)
    zone_center_y = sum(y for x, y in zone_coords) / len(zone_coords)
    
    net_center_x = (net_bounds[0] + net_bounds[2]) / 2
    net_center_y = (net_bounds[1] + net_bounds[3]) / 2
    
    center_distance = ((zone_center_x - net_center_x)**2 + (zone_center_y - net_center_y)**2)**0.5
    net_diagonal = ((net_bounds[2] - net_bounds[0])**2 + (net_bounds[3] - net_bounds[1])**2)**0.5
    
    if center_distance > net_diagonal:
        logger.warning(f"Zone center significantly offset from network center - possible coordinate system mismatch")


def verify_osm_zone_generation(osm_file: str, zones_file: str, 
                             expected_min_zones: int = 1) -> None:
    """
    Validate OSM-based intelligent zone generation
    
    Args:
        osm_file: Path to original OSM file
        zones_file: Path to generated zones file
        expected_min_zones: Minimum expected number of zones
        
    Raises:
        ValidationError: If validation fails
    """
    logger.info("Validating OSM-based intelligent zone generation...")
    
    # Check files exist
    if not Path(osm_file).exists():
        raise ValidationError(f"OSM file not found: {osm_file}")
    
    if not Path(zones_file).exists():
        raise ValidationError(f"Zones file not found: {zones_file}")
    
    try:
        # Parse zones file
        tree = ET.parse(zones_file)
        root = tree.getroot()
        zones = root.findall('poly')
        
        if len(zones) < expected_min_zones:
            raise ValidationError(f"Insufficient zones generated: {len(zones)} < {expected_min_zones}")
        
        # Validate zone properties
        zone_types = set()
        for zone in zones:
            zone_id = zone.get('id')
            zone_type = zone.get('type')
            color = zone.get('color')
            shape = zone.get('shape')
            
            if not zone_id:
                raise ValidationError("Zone missing ID attribute")
            
            if not zone_type:
                raise ValidationError(f"Zone {zone_id} missing type attribute")
            
            if not color:
                raise ValidationError(f"Zone {zone_id} missing color attribute")
            
            if not shape:
                raise ValidationError(f"Zone {zone_id} missing shape attribute")
            
            zone_types.add(zone_type)
        
        logger.info(f"Generated {len(zones)} zones with types: {sorted(zone_types)}")
        
        # Validate zone type diversity (warn if all zones are same type)
        if len(zone_types) == 1:
            logger.warning(f"All zones have the same type: {list(zone_types)[0]}. "
                         f"This may indicate insufficient OSM land use data for intelligent classification.")
        
        logger.info("OSM zone generation validation passed")
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"OSM zone generation validation failed: {e}")