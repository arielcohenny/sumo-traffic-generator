# Zone Generation

## Universal Land Use Zone Types

Based on "A Simulation Model for Intra-Urban Movements" research methodology, the system uses exactly **six research-based land use types** for synthetic networks:

### Zone Type Definitions:

- **Residential**: 34% distribution, max 1000 cells, orange color (#FFA500)
  - Multiplier: 1.0, departure_weight: 1.5, arrival_weight: 0.8
- **Employment**: 10% distribution, max 500 cells, dark red color (#8B0000)
  - Multiplier: 1.8, departure_weight: 0.9, arrival_weight: 1.8
- **Public Buildings**: 12% distribution, max 200 cells, dark blue color (#000080)
  - Multiplier: 3.0, departure_weight: 0.5, arrival_weight: 3.5
- **Mixed** (Residential + Employment + Retail): 24% distribution, max 300 cells, yellow color (#FFFF00)
  - Multiplier: 2.0, departure_weight: 1.2, arrival_weight: 1.8
- **Entertainment/Retail**: 8% distribution, max 40 cells, dark green color (#006400)
  - Multiplier: 2.5, departure_weight: 0.8, arrival_weight: 2.0
- **Public Open Space**: 12% distribution, max 100 cells, light green color (#90EE90)
  - Multiplier: 1.0, departure_weight: 1.5, arrival_weight: 0.8

### Application:

- Uses clustering algorithm to assign these types to synthetic grid cells

## Zone Generation Process

- **Step**: Generate land use zones for synthetic grid networks
- **Function**: Traditional zone extraction (`src/network/zones.py`)
- **Process**: Execute zone extraction workflow for synthetic grids
- **Arguments Used**: `--land_use_block_size_m` (affects both modes)

## Synthetic Grid Zone Generation

### Geographic Bounds Extraction

- **Step**: Extract geographic boundaries from synthetic file for zone generation
- **Function**: Bounds extraction in `src/cli.py`
- **Arguments Used**: `--land_use_block_size_m`
- **Purpose**: Define geographic area for intelligent zone grid overlay
- **Process**:
  - Parse synthetic XML file using ElementTree
  - Extract bounds from `<bounds>` element if present
  - If no bounds element, calculate from all node coordinates:
    - Iterate through all `<node>` elements in synthetic file
    - Collect `lat` and `lon` attributes from each node
    - Calculate `min_lat = min(lats)`, `max_lat = max(lats)`, `min_lon = min(lons)`, `max_lon = max(lons)`
  - Create geographic bounds tuple: `(min_lon, min_lat, max_lon, max_lat)`
  - Print bounds for verification: `f"Using geographic bounds from synthetic: {geographic_bounds}"`

### Intelligent Zone Generation

- **Step**: Generate intelligent land use zones using real synthetic data and inference
- **Function**: `IntelligentZoneGenerator.generate_intelligent_zones_from_grid()` in `src/network/intelligent_zones.py`
- **Arguments Used**: `--land_use_block_size_m`
- **Process**: Uses sophisticated multi-layer analysis combining real synthetic data with intelligent inference algorithms

#### Zone Configuration System:

- **Zone Types**: Uses the six research-based land use types defined in Section 2.1
- **synthetic Mapping**: Maps synthetic land use tags to standard zone types
- **Intelligent Inference**: When synthetic data insufficient, infers zone types using network topology + accessibility + infrastructure analysis

#### synthetic Data Loading and Parsing:

**1. synthetic XML Processing** (`load_grid_data()`):

- **XML Parsing**: Uses ElementTree to parse synthetic file structure
- **Node Extraction**: Stores all nodes with lat/lon coordinates and tag attributes
- **Way Processing**: Extracts ways with node references and tag collections
- **Relation Support**: Processes synthetic relations with member structures
- **Tag Normalization**: Converts synthetic tags to standardized key-value pairs
- **Error Handling**: Graceful fallback when synthetic file is missing or corrupted

**2. Real Land Use Zone Extraction** (`extract_grid_zones_and_pois()`):

- **synthetic Tag Mapping**: Maps synthetic landuse/amenity/building tags to zone types:
  - `residential/apartments/housing` → residential
  - `commercial/retail/shop/office` → commercial
  - `industrial` → industrial
  - `school/university/college/education` → education
  - `hospital/clinic/healthcare` → healthcare
  - `mixed` → mixed
- **Polygon Generation**: Creates valid Shapely polygons from synthetic way node sequences
- **Area Calculation**: Estimates zone area using rough geographic-to-metric conversion
- **Polygon Validation**: Ensures polygons are valid and have positive area

**3. Point of Interest (POI) Processing**:

- **Amenity Extraction**: Identifies amenity nodes (shops, restaurants, schools, hospitals)
- **POI Categorization**: Groups amenities by influence on zone classification
- **Geographic Points**: Creates Shapely Point geometries for spatial analysis

#### Multi-Layer Analysis System:

**Layer 1: Network Topology Analysis** (`analyze_network_topology()`):

- **Network Graph Construction**: Uses NetworkX DiGraph with edges and lane attributes
- **Edge Betweenness Centrality**: Calculates centrality using `nx.edge_betweenness_centrality()` with distance weighting
- **Lane Count Analysis**: Higher lane counts (>2) indicate commercial/arterial roads vs residential streets
- **Junction Connectivity**: Measures in-degree + out-degree of network junctions
- **Distance from Center Scoring**: Central locations receive commercial bias (30% weight)
- **Commercial Likelihood**: Combines centrality (30%), lane count (40%), and center distance (30%)
- **Spatial Sampling**: Uses 10% edge sampling for computational efficiency
- **Coordinate System Handling**: Supports both geographic (lat/lon) and projected coordinates

**Layer 2: Accessibility Analysis** (`analyze_accessibility()`):

- **Junction Connectivity Scoring**: In-degree and out-degree analysis of network graph
- **Network Density Calculation**: Edge density per grid cell (normalized by 100 edges)
- **Accessibility Scoring**: Combines connectivity (50%), density (30%), and random variation (20%)
- **Maximum Connectivity Normalization**: Scales scores relative to most-connected junction
- **Simplified Connectivity**: Fallback algorithm when NetworkX unavailable

**Layer 3: synthetic Infrastructure Analysis** (`analyze_grid_infrastructure()`):

- **Grid Cell Analysis**: Processes each grid cell individually with configurable search radius
- **POI Proximity Analysis**: Distance-weighted influence within 1.5x grid cell radius
- **POI-to-Zone Mapping**:
  - `shop/restaurant/cafe/bank/supermarket` → commercial (0.5 influence)
  - `school/university/college/library` → education (0.7 influence)
  - `hospital/clinic/pharmacy` → healthcare (0.6 influence)
  - `bus_station/subway_station` → mixed (0.4) + commercial (0.3)
  - `parking` → commercial (0.2 influence)
- **synthetic Zone Integration**: Direct zone type scoring with 0.8 influence weight
- **Distance Decay**: `influence = max(0, 1.0 - (distance / search_radius))`

#### Grid System and Coordinate Handling:

**Grid Generation**:

- **Configurable Resolution**: Uses `--land_use_block_size_m` parameter (default 200m)
- **Geographic Conversion**: `grid_size_degrees = block_size_m / 111000` for lat/lon coordinates
- **Projected Coordinate Support**: Direct meter-based grid for UTM/projected coordinates
- **Grid Bounds**: `num_cols = int((max_x - min_x) / grid_size_unit)`
- **Cell Boundaries**: Precise cell boundary calculation for polygon generation

**Coordinate System Detection**:

- **Geographic Detection**: Checks if coordinates within ±180° longitude, ±90° latitude
- **Automatic Conversion**: Applies appropriate grid size units based on coordinate system
- **Mixed System Support**: Handles networks with different coordinate systems

#### Score Combination and Classification:

**Multi-Factor Scoring Algorithm** (`combine_scores_and_classify()`):

- **Commercial Bias Calculation**: `commercial_bias = topology_score * 0.6 + accessibility_score * 0.4`
- **Base Residential Assumption**: Default residential score of 0.5 for all cells
- **Commercial Boost**: `commercial_score = commercial_bias * 2.0` (doubled for emphasis)
- **Center Distance Factor**: Additional commercial bias for central locations (1.5x multiplier)
- **Edge Industrial Bias**: Peripheral areas receive industrial potential (0.8x factor)
- **Education Near Residential**: Residential areas (>0.7) get education bonus (0.3)
- **Mixed-Use Detection**: Medium-density areas (0.3-0.7 commercial bias) get mixed-use bonus (0.5)

**Infrastructure Score Integration**:

- **Weighted Addition**: Infrastructure scores added with 0.7 weight to final scores
- **Zone Type Preservation**: Direct zone type mapping maintains synthetic land use where available
- **Score Aggregation**: All factors combined before final classification

**Final Classification**:

- **Highest Score Selection**: `ZoneScore.get_highest_score_type()` determines final zone type
- **Polygon Generation**: Creates rectangular grid cell polygons with exact boundaries
- **Capacity Calculation**: `capacity = area_sqm * 0.02` (base capacity per square meter)
- **Metadata Storage**: Stores all analysis scores for debugging and verification

#### Fallback and Error Handling:

**Data Availability Checks**:

- **synthetic Data Validation**: Graceful handling of missing or corrupted synthetic files
- **Network Data Requirements**: Fallback algorithms when network analysis unavailable
- **Library Dependencies**: Optional NetworkX, GeoPandas, Pandas with simplified alternatives

**Default Assumptions**:

- **No synthetic Data**: Falls back to residential (0.5 score) when no synthetic zones/POIs found
- **Missing Network**: Uses simplified topology analysis without graph algorithms
- **Coordinate Transformation**: Maintains geographic coordinates when projection unavailable

#### Output and Integration:

**Zone Generation Results**:

- **Zone Metadata**: Each zone includes id, geometry, zone_type, area_sqm, capacity
- **Analysis Scores**: Stores topology, accessibility, infrastructure, and final scores
- **Grid Coordinates**: Maintains (i,j) grid position for reference
- **Intelligent Flag**: `is_intelligent: True` distinguishes from traditional zones

**Performance Characteristics**:

- **Computational Complexity**: O(n×m×p) where n=grid_cols, m=grid_rows, p=POI_count
- **Memory Usage**: Stores full synthetic data in memory for spatial analysis
- **Processing Time**: Varies with synthetic file size and grid resolution

### synthetic Zone File Creation

- **Step**: Save intelligent zones to polygon file in geographic coordinates
- **Function**: `save_intelligent_zones_to_poly_file()` in `src/network/intelligent_zones.py`
- **Process**:
  - Generate polygon shapes for each classified grid cell
  - Create SUMO polygon XML format with zone type, color, and coordinates
  - Save initially in geographic coordinates (lat/lon)
  - Zone count and type distribution logged for verification
- **Output**: `workspace/zones.poly.xml` with intelligent zone classification
- **Coordinates**: Geographic (lat/lon) format, converted to projected later in Step 5
- **Success Message**: `f"Generated and saved {len(intelligent_zones)} intelligent zones to {CONFIG.zones_file}"`

### synthetic Mode Validation

- **Step**: Verify intelligent zone generation success
- **Function**: Exception handling in `src/cli.py`
- **Process**:
  - Check zone file existence and size
  - Validate zone count is reasonable for network area
  - Verify zone type distribution is realistic
  - Confirm geographic coordinate format is valid
- **Error Handling**: Print failure message and exit with code 1 on validation failure
- **Failure Message**: `f"Failed to generate synthetic zones: {e}"`

## Traditional Zone Extraction (Synthetic Grids)

### Traditional Zone Extraction

- **Step**: Extract zones from junction-based cellular grid methodology
- **Function**: `extract_zones_from_junctions()` in `src/network/zones.py`
- **Arguments Used**: `--land_use_block_size_m`
- **Research Basis**: Based on "A Simulation Model for Intra-Urban Movements" cellular grid methodology

#### Junction Coordinate Parsing:

- **Source File**: Parse coordinates from `workspace/grid.nod.xml`
- **Node Filtering**: Exclude internal nodes, include only junction nodes
- **Coordinate Extraction**: Extract x,y coordinates for grid boundary calculation
- **Network Bounds**: Calculate `network_xmin`, `network_xmax`, `network_ymin`, `network_ymax`

#### Cellular Grid Creation:

- **Cell Size Configuration**: Uses `--land_use_block_size_m` parameter (default 25.0m for both network types)
- **Grid Subdivision**: `num_x_cells = int((xmax - xmin) / cell_size)`, same for y-axis
- **Polygon Generation**: Create rectangular zones using Shapely `box()` geometry
- **Zone Independence**: Block size independent of junction spacing for flexible resolution

#### Land Use Classification System:

- **Zone Types**: Uses the six research-based land use types defined in Section 2.1
- **Distribution**: Follows exact percentages and cluster sizes from research paper
- **Assignment**: Uses clustering algorithm with BFS to create contiguous land use clusters

#### Clustering Algorithm:

- **BFS Clustering**: Uses breadth-first search to create contiguous land use clusters
- **Size Constraints**: Respects maximum cluster size per land use type
- **Spatial Distribution**: Ensures realistic geographic distribution of land uses
- **Randomization**: Uses provided seed for reproducible zone assignment

### Traditional Zone Validation

- **Step**: Verify traditional zone extraction success
- **Function**: `verify_extract_zones_from_junctions()` in `src/validate/validate_network.py`

#### Zone Count Validation:

- **Expected Calculation**: `expected_zones = num_x_cells * num_y_cells`
- **Actual Count**: Parse and count polygons in generated zones file
- **Count Verification**: Ensure actual zones match expected subdivision count
- **Error Condition**: Raise `ValidationError` if counts don't match

#### Zone Structure Validation:

- **File Existence**: Verify `workspace/zones.poly.xml` was created
- **XML Structure**: Parse and validate polygon XML format
- **Coordinate Bounds**: Ensure all zone coordinates within network bounds
- **Land Use Distribution**: Verify land use percentages approximate target distribution
- **Polygon Integrity**: Check all polygons are valid and non-overlapping

#### Validation Constants:

- **Minimum Zone Size**: Cell size must be > 0
- **Maximum Zones**: Reasonable upper limit based on cell size and network area
- **Distribution Tolerance**: Allow ±5% deviation from target land use percentages

### Traditional Zone Completion

- **Step**: Confirm successful traditional zone generation
- **Function**: Success logging in `src/cli.py`
- **Output**: `workspace/zones.poly.xml` with traditional zone extraction
- **Success Message**: `f"Extracted land use zones successfully with {args.land_use_block_size_m}m blocks."`
- **Zone Information**: Log zone count, land use distribution, and file size

## Common Zone Generation Outputs

- **File Generated** (both modes): `workspace/zones.poly.xml`

## Zone Visibility in SUMO GUI

### Default Behavior

By default, when running with `--gui`, zone polygons are displayed as colored overlays in SUMO's visualization:

- Each zone type has a distinct color (e.g., Residential = orange, Employment = dark red)
- Zones are rendered as filled polygons on the network layer
- Zone overlay can sometimes obscure vehicle traffic visualization

### Hiding Zones from Display

Use the `--hide-zones` flag to suppress zone display while keeping zones fully functional:

```bash
# Show zones (default)
env PYTHONUNBUFFERED=1 python -m src.cli --gui

# Hide zones from display
env PYTHONUNBUFFERED=1 python -m src.cli --gui --hide-zones
```

**Important**: When `--hide-zones` is used:

- Zones are still **computed** during pipeline Step 2 (Zone Generation)
- Zones are still **used** for:
  - Edge attractiveness calculations
  - Lane count assignment (when using `realistic` algorithm)
  - Traffic routing decisions
- Only the **visual display** in SUMO GUI is suppressed
- The `zones.poly.xml` file is still generated in the workspace

### Web GUI Support

In the web GUI (`dbps`), a "Hide Zones" checkbox appears when "Launch SUMO GUI" is selected. This provides the same functionality as the `--hide-zones` CLI flag.

### Technical Implementation

The `--hide-zones` flag works by excluding `zones.poly.xml` from the SUMO configuration file (`grid.sumocfg`). Normally, zones are included via the `<additional-files>` element:

```xml
<!-- With zones visible (default) -->
<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="vehicles.rou.xml"/>
        <additional-files value="zones.poly.xml"/>
    </input>
    ...
</configuration>

<!-- With --hide-zones (zones hidden) -->
<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="vehicles.rou.xml"/>
        <!-- additional-files line omitted -->
    </input>
    ...
</configuration>
```
