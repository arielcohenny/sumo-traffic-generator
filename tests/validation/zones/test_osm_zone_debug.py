#!/usr/bin/env python3
"""
Debug OSM zone generation issue
"""

try:
    from src.network.intelligent_zones import IntelligentZoneGenerator
    
    print("Testing OSM zone generation with --land_use_block_size_m...")
    
    # Test parameters
    osm_file = "src/osm/samples/manhattan_upper_west.osm"
    land_use_block_size = 200.0
    
    # Check if OSM file exists
    from pathlib import Path
    if not Path(osm_file).exists():
        print(f"ERROR: OSM file not found: {osm_file}")
        exit(1)
    
    # Parse OSM bounds
    from xml.etree import ElementTree as ET
    tree = ET.parse(osm_file)
    root = tree.getroot()
    
    bounds_elem = root.find('bounds')
    if bounds_elem is not None:
        min_lat = float(bounds_elem.get('minlat'))
        min_lon = float(bounds_elem.get('minlon'))
        max_lat = float(bounds_elem.get('maxlat'))
        max_lon = float(bounds_elem.get('maxlon'))
    else:
        # Calculate bounds from nodes
        lats, lons = [], []
        for node in root.findall('node'):
            lats.append(float(node.get('lat')))
            lons.append(float(node.get('lon')))
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
    
    geographic_bounds = (min_lon, min_lat, max_lon, max_lat)
    print(f"Geographic bounds: {geographic_bounds}")
    
    # Test zone generation
    zone_generator = IntelligentZoneGenerator(land_use_block_size_m=land_use_block_size)
    print(f"Zone generator created with block size: {land_use_block_size}m")
    
    intelligent_zones = zone_generator.generate_intelligent_zones_from_osm(
        osm_file_path=osm_file,
        geographic_bounds=geographic_bounds
    )
    
    print(f"Successfully generated {len(intelligent_zones)} zones")
    
    # Check zone distribution
    zone_types = {}
    for zone in intelligent_zones:
        zone_type = zone['zone_type']
        zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
    
    print("Zone type distribution:")
    for zone_type, count in sorted(zone_types.items()):
        print(f"  {zone_type}: {count} zones")
        
    # Test saving
    from src.network.intelligent_zones import save_intelligent_zones_to_poly_file
    save_intelligent_zones_to_poly_file(intelligent_zones, "test_zones_output.poly.xml", None)
    print("Zones saved successfully to test_zones_output.poly.xml")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()