# Zone Coordinate Conversion (OSM Mode Only)

## Purpose and Timing

**Why Zone Coordinate Conversion is Needed:**

- **Geographic vs Projected Coordinates**: OSM zones are initially created in geographic coordinates (latitude/longitude) from the original OSM file
- **Spatial Analysis Requirement**: Edge attractiveness assignment requires precise spatial analysis between edges and zones
- **Coordinate System Mismatch**: SUMO network uses projected coordinates (x,y in meters) while zones start in geographic coordinates
- **Distance Calculations**: Accurate distance measurements between zones and edges require both to be in the same coordinate system

**Why This Step Occurs After Network Rebuild:**

- **Dependency on Final Network**: The network rebuild (Step 4) establishes the final projected coordinate system used by SUMO
- **Coordinate System Authority**: Only after `netconvert` processes the network do we have the definitive coordinate transformation parameters
- **Edge Geometry Finalization**: Network rebuild finalizes all edge geometries and positions in projected coordinates
- **Spatial Reference Availability**: The rebuilt `grid.net.xml` contains the coordinate system information needed for accurate conversion

## OSM Mode Coordinate Conversion Process

- **Step**: Convert zone coordinates from geographic (lat/lon) to projected (x,y)
- **Function**: `convert_zones_to_projected_coordinates()` in `src/network/intelligent_zones.py`
- **Arguments Used**: Existing `workspace/zones.poly.xml` and `workspace/grid.net.xml`
- **Timing**: Only executed when `--osm_file` argument was used in Step 1

### Coordinate Transformation Process

- **Source Coordinates**: Geographic coordinates (latitude, longitude) from OSM data
- **Target Coordinates**: Projected coordinates (x, y in meters) matching SUMO network
- **Transformation Method**: Uses SUMO's coordinate system parameters from the rebuilt network
- **Precision**: Maintains geometric accuracy for spatial zone-edge analysis

### Zone File Update

- **Input File**: `workspace/zones.poly.xml` with geographic coordinates
- **Output File**: `workspace/zones.poly.xml` updated with projected coordinates (same filename, converted content)
- **Preservation**: Maintains all zone properties (type, color, attractiveness) while updating only coordinates
- **Validation**: Ensures all zone polygons remain valid after coordinate transformation

### Spatial Consistency Verification

- **Coordinate System Match**: Verify zones and network edges use identical coordinate systems
- **Geometric Integrity**: Ensure zone polygons maintain proper shapes after transformation
- **Network Bounds**: Confirm converted zones fall within reasonable bounds of the network area

## Non-OSM Mode (No Action Required)

For synthetic grid networks (when `--osm_file` is not provided):

- **No Conversion Needed**: Zones already created in projected coordinates during Step 2
- **Coordinate System Match**: Traditional zones use same coordinate system as synthetic network
- **Direct Spatial Analysis**: Zones ready for edge attractiveness assignment without conversion

## Zone Coordinate Conversion Completion

- **Step**: Confirm successful coordinate conversion (OSM mode only)
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Successfully converted zone coordinates to projected system."
- **Error Handling**: Falls back gracefully if conversion fails: "Zones will remain in geographic coordinates."
- **Ready for Next Step**: Zones are now in correct coordinate system for spatial edge attractiveness analysis