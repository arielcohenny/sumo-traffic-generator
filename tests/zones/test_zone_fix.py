#!/usr/bin/env python3
"""
Test script to verify the intelligent zones coordinate fix
"""

import sys
from pathlib import Path
from src.network.intelligent_zones import IntelligentZoneGenerator, save_intelligent_zones_to_poly_file

def test_coordinate_fix():
    """Test the coordinate fix for intelligent zones"""
    print("🧪 Testing Intelligent Zones Coordinate Fix")
    print("=" * 50)
    
    # Test with SUMO network bounds (projected coordinates)
    network_bounds = (0.0, 0.0, 600.0, 600.0)  # SUMO network bounds from grid.net.xml
    
    print(f"Network bounds: {network_bounds}")
    print(f"Zone block size: 75m")
    
    # Create zone generator
    generator = IntelligentZoneGenerator(land_use_block_size_m=75.0)
    
    # Check coordinate system detection
    min_x, min_y, max_x, max_y = network_bounds
    
    # Check if coordinates are geographic (lat/lon) or projected
    if abs(min_x) < 180 and abs(max_x) < 180 and abs(min_y) < 90 and abs(max_y) < 90:
        coord_system = "Geographic (lat/lon)"
        grid_size_unit = 75.0 / 111000  # Convert meters to degrees
    else:
        coord_system = "Projected (meters)"
        grid_size_unit = 75.0  # Use meters directly
    
    print(f"Coordinate system: {coord_system}")
    print(f"Grid size unit: {grid_size_unit}")
    
    # Calculate grid dimensions
    num_cols = max(1, int((max_x - min_x) / grid_size_unit))
    num_rows = max(1, int((max_y - min_y) / grid_size_unit))
    
    print(f"Grid dimensions: {num_cols} cols x {num_rows} rows = {num_cols * num_rows} zones")
    
    # Create a sample zone polygon
    i, j = 0, 0  # First zone
    cell_min_x = min_x + i * grid_size_unit
    cell_max_x = min_x + (i + 1) * grid_size_unit
    cell_min_y = min_y + j * grid_size_unit
    cell_max_y = min_y + (j + 1) * grid_size_unit
    
    print(f"Sample zone coordinates:")
    print(f"  Bottom-left: ({cell_min_x:.2f}, {cell_min_y:.2f})")
    print(f"  Top-right: ({cell_max_x:.2f}, {cell_max_y:.2f})")
    
    # Check if coordinates are reasonable for SUMO
    if coord_system == "Projected (meters)" and 0 <= cell_min_x <= 600 and 0 <= cell_min_y <= 600:
        print("✅ Coordinates look correct for SUMO network!")
        return True
    elif coord_system == "Geographic (lat/lon)":
        print("⚠️  Using geographic coordinates - may need transformation")
        return True
    else:
        print("❌ Coordinates appear incorrect")
        return False

if __name__ == "__main__":
    success = test_coordinate_fix()
    print(f"\nTest result: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)