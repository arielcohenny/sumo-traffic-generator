#!/usr/bin/env python3
"""
Debug coordinate transformation directly
"""

from pathlib import Path
from src.network.intelligent_zones import save_intelligent_zones_to_poly_file

# Create a minimal test zone
test_zones = [{
    'id': 'test_zone_0_0',
    'zone_type': 'commercial',
    'zone_info': {'color': '255,0,0'},
    'geometry': None  # We'll create this manually
}]

# Create a test polygon in geographic coordinates
from shapely.geometry import Polygon
test_polygon = Polygon([
    (-73.9844643, 40.7802483),
    (-73.9826624981982, 40.7802483), 
    (-73.9826624981982, 40.7820501018018),
    (-73.9844643, 40.7820501018018),
    (-73.9844643, 40.7802483)
])

test_zones[0]['geometry'] = test_polygon

# Test coordinate transformation
print("Testing coordinate transformation...")
print(f"Input coordinates: {list(test_polygon.exterior.coords)[:2]}")

# Test with network file
net_file = "data/grid.net.xml"
output_file = "debug_zones.poly.xml"

try:
    save_intelligent_zones_to_poly_file(test_zones, output_file, net_file)
    print("Zone saved successfully!")
    
    # Read the output to see if coordinates were transformed
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            content = f.read()
        print("\nOutput file content:")
        print(content)
        
        # Check if coordinates look like they were transformed
        if "-73.98" in content:
            print("\nWARNING: Coordinates still in geographic format (lat/lon)")
        elif "," in content and any(c.isdigit() for c in content):
            print("\nSUCCESS: Coordinates appear to be transformed")
    else:
        print("ERROR: Output file was not created")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()