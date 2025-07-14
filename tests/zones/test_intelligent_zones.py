#!/usr/bin/env python3
"""
Test script for the intelligent zone generation system
"""

from src.network.intelligent_zones import IntelligentZoneGenerator, save_intelligent_zones_to_poly_file
from pathlib import Path
import json

def test_intelligent_zones():
    """Test the intelligent zone generation system"""
    print("Testing Intelligent Zone Generation System")
    print("=" * 50)
    
    # Test parameters
    generator = IntelligentZoneGenerator(land_use_block_size_m=200.0)
    network_json = 'data/grid.net.json'
    network_bounds = (-73.990, 40.740, -73.970, 40.760)  # Test Manhattan bounds
    osm_file = 'src/osm/export.osm'
    
    # Check if network JSON exists
    if not Path(network_json).exists():
        print(f"❌ Network JSON not found: {network_json}")
        print("Please run a simulation first to generate the network JSON file")
        return False
    
    print(f"✅ Found network JSON: {network_json}")
    
    # Check OSM file
    osm_file_path = osm_file if Path(osm_file).exists() else None
    if osm_file_path:
        print(f"✅ Found OSM file: {osm_file_path}")
    else:
        print(f"⚠️  OSM file not found: {osm_file} (will test without OSM data)")
    
    # Test zone generation
    try:
        print(f"\n🧠 Generating intelligent zones...")
        print(f"   Grid resolution: {generator.block_size_m}m blocks")
        print(f"   Network bounds: {network_bounds}")
        print(f"   OSM enhancement: {'Yes' if osm_file_path else 'No'}")
        
        zones = generator.generate_intelligent_zones(
            network_json_file=network_json,
            network_bounds=network_bounds,
            osm_file_path=osm_file_path
        )
        
        print(f"\n✅ Successfully generated {len(zones)} zones!")
        
        # Analyze zone distribution
        zone_types = {}
        for zone in zones:
            zone_type = zone['zone_type']
            zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
        
        print(f"\n📊 Zone Type Distribution:")
        total_zones = len(zones)
        for zone_type, count in sorted(zone_types.items()):
            percentage = (count / total_zones) * 100
            print(f"   {zone_type:>12}: {count:>3} zones ({percentage:>5.1f}%)")
        
        # Test polygon file generation
        print(f"\n💾 Testing polygon file generation...")
        test_poly_file = "data/test_intelligent_zones.poly.xml"
        save_intelligent_zones_to_poly_file(zones, test_poly_file, "data/grid.net.xml")
        
        if Path(test_poly_file).exists():
            file_size = Path(test_poly_file).stat().st_size
            print(f"✅ Generated polygon file: {test_poly_file} ({file_size} bytes)")
        else:
            print(f"❌ Failed to generate polygon file: {test_poly_file}")
        
        # Sample zone details
        print(f"\n🔍 Sample Zone Details:")
        for i, zone in enumerate(zones[:3]):  # Show first 3 zones
            print(f"   Zone {i+1}: {zone['id']}")
            print(f"     Type: {zone['zone_type']}")
            print(f"     Area: {zone['area_sqm']:.1f} sqm")
            print(f"     Grid coords: {zone.get('grid_coords', 'N/A')}")
            if 'scores' in zone:
                scores = zone['scores']
                print(f"     Topology score: {scores.get('topology', 0):.3f}")
                print(f"     Accessibility score: {scores.get('accessibility', 0):.3f}")
        
        print(f"\n🎉 Intelligent zone generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during zone generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_intelligent_zones()
    exit(0 if success else 1)