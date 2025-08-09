# Network Generation

## Mode Selection Based on Arguments

- **Step**: Determine network generation approach
- **Function**: Pipeline factory pattern (`src/pipeline/pipeline_factory.py`)
- **Process**:
  - Check if `--tree_method_sample` argument is provided
  - If Tree Method sample provided: Execute research dataset workflow (Section 1.4)
  - Check if `--osm_file` argument is provided
  - If OSM file provided: Execute OSM import workflow (Section 1.2)
  - If no OSM file: Execute synthetic grid generation workflow (Section 1.3)
- **Arguments Used**: `--tree_method_sample`, `--osm_file`

## OSM Mode (`--osm_file` provided)

### OSM Network Import

- **Step**: Import real-world street network from OSM data
- **Function**: `import_osm_network()` in `src/network/import_osm.py`
- **Arguments Used**: `--osm_file`

#### OSM File Requirements:

- **Required Elements**: Nodes, ways with highway tags, bounding box
- **Highway Types**: Primary, secondary, tertiary, residential, unclassified streets
- **Lane Information**: Number of lanes per edge (preserved from OSM data)
- **Traffic Signals**: Original OSM traffic light locations (auto-generated if missing)
- **Minimum Content**: At least 10 highway ways for viable network
- **Area Limits**: Warning issued for areas > 5 km² (performance impact)

#### 12 Specialized netconvert Parameters:

- **Process**:
  - `--geometry.remove`: Remove unnecessary geometry points for cleaner network
  - `--roundabouts.guess`: Automatically detect roundabout structures
  - `--junctions.join`: Join nearby junctions to reduce complexity
  - `--tls.guess-signals`: Guess traffic signal locations from OSM data
  - `--tls.discard-simple`: Remove simple traffic lights that don't need control
  - `--ramps.guess`: Detect highway ramps and on/off connections
  - `--junctions.corner-detail 5`: Set junction corner detail level to 5 meters
  - `--output.street-names`: Preserve original street names from OSM
  - `--output.original-names`: Keep original OSM element names
  - `--keep-edges.by-vclass passenger`: Filter to keep only passenger vehicle infrastructure
  - `--remove-edges.by-vclass pedestrian`: Remove pedestrian-only paths
  - Command: `netconvert --osm-files {osm_file} --output-prefix workspace/grid/osm_network`
- **Failure**: System fails with error if OSM file lacks sufficient data for network generation
- **Output**: Intermediate files in `workspace/grid/` directory

### OSM File Organization

- **Step**: Move OSM-generated files to expected pipeline locations
- **Function**: File movement logic in `src/cli.py`
- **Process**:
  - Move files from `workspace/grid/osm_network.*` to `workspace/grid.*`
  - Handle file patterns: `*.nod.xml`, `*.edg.xml`, `*.con.xml`, `*.tll.xml`
  - Extract file extensions and rename to standard format
  - Clean up temporary `workspace/grid/` directory
  - Print movement confirmation for each file
- **Output**: `workspace/grid.nod.xml`, `workspace/grid.edg.xml`, `workspace/grid.con.xml`, `workspace/grid.tll.xml`

### Dead-End Street Handling (OSM-Specific)

- **Step**: Handle real-world street topology irregularities in OSM networks
- **Function**: Defensive programming in `src/traffic_control/decentralized_traffic_bottlenecks/classes/net_data_builder.py`
- **Scope**: Applied only to OSM networks - synthetic grids have no dead-ends since isolated junctions are completely removed

#### Dead-End Detection:

- **OSM Networks**: Contains natural dead-ends (cul-de-sacs, private roads, incomplete boundaries)
- **Synthetic Grids**: No dead-ends exist - junction removal process completely deletes isolated junctions and all connected edges
- **Algorithm Protection**: Uses `.get()` method instead of direct dictionary access to prevent crashes when edges have no outgoing connections

#### Handling Process:

- **Connection Building**: Build dictionary of edge-to-edge connections during network parsing
- **Missing Connections**: Dead-end streets have no entries in connections dictionary
- **Fallback Mechanism**: Return empty connection list for dead-end streets instead of crashing
- **Graph Integration**: Dead-end streets become terminal nodes in traffic control graph with no outgoing links

#### Network Topology Impact:

- **OSM Networks**: Dead-end streets preserved and handled gracefully by traffic control algorithm
- **Synthetic Grids**: Perfect topology with no special handling needed
- **Traffic Simulation**: Vehicles can enter dead-ends but must stop/turn around (handled by SUMO physics)
- **Signal Control**: Dead-end intersections receive traffic light control if signalized

### OSM Mode Validation

- **Step**: Verify OSM network import success
- **Function**: OSM validation in `src/network/import_osm.py`
- **Process**: Validate the following aspects, if any fails print a relevant message and exit.

#### Required Output Files:

- `workspace/grid.nod.xml`: Network nodes (junctions)
- `workspace/grid.edg.xml`: Network edges (streets)
- `workspace/grid.con.xml`: Connection definitions
- `workspace/grid.tll.xml`: Traffic light definitions

#### XML Structure Validation:

- **Parse XML Files**: Ensure valid XML structure and syntax
- **Element Validation**: Verify required elements (nodes, edges, connections, traffic lights)
- **Attribute Validation**: Check required attributes exist (id, coordinates, lane counts)
- **Reference Validation**: Ensure edge references valid nodes, connections reference valid edges

#### Network Connectivity and Integrity:

- **Node Coverage**: Verify all junctions are properly defined
- **Edge Coverage**: Count and validate highway ways (minimum 10 required)
- **Connection Coverage**: Ensure proper edge-to-edge connections
- **Traffic Light Coverage**: Validate signal placement and timing
- **Bounding Box**: Verify network area calculation and limits (warn if > 5 km²)
- **Highway Type Analysis**: Count primary, secondary, tertiary, residential, unclassified streets
- **Network Statistics**: Extract node count, edge count, connection count, traffic light count
- **Edge Length Statistics**: Calculate minimum, maximum, and average edge lengths

- **Success Message**: "Successfully imported OSM network."

### OSM Samples

- **Directory**: `osm_samples/` (relocated from `src/osm/`)
- **Purpose**: Verified working OSM areas for testing and development
- **Quality**: All samples tested with ≥294/300 vehicle route generation success rate

#### Available Samples:

- **Manhattan Upper West Side** (`manhattan_upper_west.osm`):

  - **Location**: New York City (40.7800, -73.9850, 40.7900, -73.9750)
  - **Type**: Classic Manhattan grid pattern
  - **Size**: 3,511 lines
  - **Performance**: 300/300 vehicles (100% success rate)

- **San Francisco Downtown** (`sf_downtown.osm`):

  - **Location**: San Francisco (37.7850, -122.4100, 37.7950, -122.4000)
  - **Type**: Downtown grid layout
  - **Size**: 6,541 lines
  - **Performance**: 298/300 vehicles (99.3% success rate)

- **Washington DC Downtown** (`dc_downtown.osm`):
  - **Location**: Washington DC (38.8950, -77.0350, 38.9050, -77.0250)
  - **Type**: Planned grid system
  - **Size**: 6,660 lines
  - **Performance**: 300/300 vehicles (100% success rate)

#### Sample Management:

- **Download Script**: `tools/scripts/download_osm_samples.py` using Overpass API
- **Query Filter**: Highway types (motorway, trunk, primary, secondary, tertiary, unclassified, residential) and traffic signals
- **Testing Verified**: Each sample validated for reliable traffic simulation with high vehicle route generation success rates

## Non-OSM Mode (synthetic grid)

### Grid Network Generation

- **Step**: Generate orthogonal grid network using SUMO netgenerate
- **Function**: `generate_grid_network()` in `src/network/generate_grid.py`

#### Arguments Used and Values:

- `--grid_dimension`: Number of rows and columns (e.g., 5 for 5×5 grid)
- `--block_size_m`: Block size in meters (e.g., 200m per block)
- `--junctions_to_remove`: Junction removal specification (e.g., "3" or "A1,B2,C3")
- `--lane_count`: Lane configuration algorithm ("realistic", "random", or integer like "2")
- `--traffic_light_strategy`: Traffic signal phasing strategy ("opposites" or "incoming")

#### Grid Generation Process:

- **Creates uniform grid** based on `--grid_dimension` and `--block_size_m`
- **Generates default 1-lane network** for consistent lane assignment
- **Applies traffic light strategy** using SUMO's `--tls.layout` functionality
- **Command**: `netgenerate --grid --grid.number {dimension} --grid.length {block_size}`

#### Junction Removal Process:

- **Step**: Remove specified junctions from generated grid
- **Function**: `wipe_crossing()` in `src/network/generate_grid.py`
- **Trigger**: When `--junctions_to_remove` is not "0"
- **Selection Methods**:
  - **Random Count**: Integer value (e.g., "3") selects random interior junctions
  - **Specific IDs**: Comma-separated list (e.g., "A1,B2,C3") removes exact junctions
- **Grid Requirements**: Minimum 3×3 grid required to have removable interior nodes
- **Junction ID Format**: Row letters (A, B, C...) + column numbers (0, 1, 2...)
- **Cascade Effect**: If a junction (not in the selected junctions to remove) becomes disconnected from all other junctions after removal, it is also automatically removed
- **Removal Process**:

  **1. Connection File (.con.xml) Updates (`wipe_crossing_from_con()`):**

  - Parse connections XML file using ElementTree
  - Identify connections involving target junctions using regex pattern `^([A-Z]+\d*)([A-Z]+\d*)$`
  - Mark connections for removal where `from` or `to` edge contains target junction ID
  - Example: For junction "B1", remove connections like `<connection from="A1B1" to="B1C1">`
  - Remove marked connection elements from XML tree
  - Write updated file with UTF-8 encoding and XML declaration

  **2. Node File (.nod.xml) Updates (`wipe_crossing_from_nod()`):**

  - Parse nodes XML file
  - Find all node elements with `id` matching target junction IDs
  - Remove entire node elements (e.g., `<node id="B1" x="200" y="200" type="traffic_light"/>`)
  - Write updated file back to disk

  **3. Edge File (.edg.xml) Updates (`wipe_crossing_from_edg()`):**

  - Parse edges XML file
  - Identify edges where `from` or `to` attribute equals target junction ID
  - Remove edge elements connected to target junctions
  - Example: For junction "B1", remove edges like `<edge id="A1B1" from="A1" to="B1"/>`
  - Write updated file back to disk

  **4. Traffic Light File (.tll.xml) Updates (`wipe_crossing_from_tll()`):**

  - **Connection Analysis**: Identify connections to remove using same logic as step 1
  - **Link Index Collection**: Collect `(tl_id, linkIndex)` pairs for each removed connection
  - **Connection Removal**: Remove connection elements involving target junctions
  - **Traffic Light Logic Removal**: Remove entire `<tlLogic>` elements for target junction IDs
  - **Phase State Updates**: For remaining traffic lights, remove characters from phase state strings at removed link indices (processed in descending order)
  - **Empty TL Cleanup**: Remove traffic lights with no remaining connections (empty state strings)
  - **Link Index Reindexing**: Reindex remaining connections so linkIndex starts at 0 and is contiguous per traffic light
  - **Example**: If linkIndex 1 is removed from state "GrGr", result is "GGr" with remaining indices 0,1,2

### Grid Network Validation

- **Step**: Verify synthetic grid generation
- **Function**: `verify_generate_grid_network()` in `src/validate/validate_network.py`

#### 14 Comprehensive Validation Steps:

1. **File Existence**: Check all 4 required XML files exist (.nod.xml, .edg.xml, .con.xml, .tll.xml)
2. **XML Structure**: Parse and validate XML structure, ensure minimum file size (100 bytes)
3. **Junction Structure**: Validate grid nodes vs H_nodes, collect coordinates within bounds
4. **Edge Structure**: Validate edge count, ensure no head edges exist before splitting
5. **Traffic Light Configuration**: Validate TL logic exists, phase durations (1-120s), cycle times (10-300s)
6. **Bounding Box**: Validate coordinates within expected grid bounds based on dimension and block size
7. **Junction Removal**: Verify specific junctions were removed correctly, count matches expectation
8. **Edge Connectivity**: Validate all referenced nodes exist, no orphaned edges
9. **Connections Consistency**: Validate connections reference existing edges, proper from/to relationships
10. **Lane Assignments**: Check lane counts within bounds (1-5 lanes per edge)
11. **Lane Indexing**: Validate connection lane indices are non-negative and sequential
12. **File Sizes**: Ensure files aren't corrupted, contain expected minimum content
13. **Internal Nodes**: Validate no internal nodes remain after generation (only junction and H-nodes)
14. **Traffic Light Timing**: Validate green time ratios (≥20% of cycle time), phase consistency

#### Key Validation Constants:

- **HEAD_DISTANCE**: 50 meters (from config.py)
- **Junction Radius**: 10.0 meters (DEFAULT_JUNCTION_RADIUS)
- **Lane Bounds**: 1-5 lanes per edge maximum
- **Phase Duration**: 1-120 seconds valid range
- **Cycle Time**: 10-300 seconds valid range
- **Minimum Green Time**: 20% of total cycle time

- **Error Handling**: Raises `ValidationError` on failure, exits with code 1

### Grid Generation Completion

- **Step**: Confirm successful grid generation
- **Function**: Success logging in `src/cli.py`
- **Output**: Same file structure as OSM mode: `workspace/grid.nod.xml`, `workspace/grid.edg.xml`, `workspace/grid.con.xml`, `workspace/grid.tll.xml`
- **Success Message**: "Generated grid successfully."

## Tree Method Research Dataset Mode (`--tree_method_sample` provided)

### Research Dataset Loading

- **Step**: Load pre-built Tree Method research networks for validation
- **Function**: `src/pipeline/sample_pipeline.py`
- **Arguments Used**: `--tree_method_sample`
- **Process**:
  - Validate sample folder exists and contains required files
  - Check for required files: `network.net.xml`, `vehicles.trips.xml`, `simulation.sumocfg.xml`
  - Copy sample files to workspace directory
  - Adapt file names to pipeline naming convention
- **Purpose**: Enable research validation against established Tree Method benchmarks

### File Management and Adaptation

- **Step**: Copy and rename sample files to match pipeline expectations
- **Function**: File copying logic in `src/pipeline/sample_pipeline.py`
- **Process**:
  - `network.net.xml` → `workspace/grid.net.xml`
  - `vehicles.trips.xml` → `workspace/vehicles.rou.xml`
  - `simulation.sumocfg.xml` → `workspace/grid.sumocfg`
  - Preserve original file content and structure
  - Update configuration file paths if needed
- **Validation**: Verify all required files copied successfully

### Research Network Characteristics

- **Dataset Source**: Original Tree Method research networks (decentralized-traffic-bottlenecks)
- **Network Complexity**: Complex urban topology with multi-lane edges and advanced traffic signals
- **Simulation Scale**: 946 vehicles over 2-hour simulation period (7300 seconds)
- **Research Context**: Pre-processed networks from original research (Experiment1-realistic-high-load)
- **Validation Purpose**: Test Tree Method implementation against published performance benchmarks

### Pipeline Bypass Behavior

- **Bypass Mode**: Skips Steps 1-8 entirely, proceeds directly to Step 9 (Dynamic Simulation)
- **Steps Skipped**: Network generation, zone extraction, edge splitting, attractiveness assignment, route generation
- **Benefits**: Immediate simulation execution, research validation, method comparison
- **Incompatible Arguments**: Cannot be used with network generation parameters (`--osm_file`, `--grid_dimension`, `--block_size_m`, `--junctions_to_remove`, `--lane_count`)

### Tree Method Research Dataset Completion

- **Step**: Confirm successful dataset loading and preparation
- **Function**: Success logging in `src/pipeline/sample_pipeline.py`
- **Success Message**: "Successfully loaded Tree Method research dataset."
- **Output**: Research network ready for immediate simulation execution
- **Ready for Step 9**: All components prepared for dynamic simulation with traffic control comparison

## Common Network Generation Outputs

- **Files Generated** (OSM and synthetic modes):
  - `workspace/grid.nod.xml`: Network nodes (junctions)
  - `workspace/grid.edg.xml`: Network edges (streets)
  - `workspace/grid.con.xml`: Connection definitions
  - `workspace/grid.tll.xml`: Traffic light definitions
- **Files Provided** (Tree Method research dataset mode):
  - `workspace/grid.net.xml`: Complete pre-built network
  - `workspace/vehicles.rou.xml`: Pre-generated vehicle routes
  - `workspace/grid.sumocfg`: Pre-configured simulation parameters
- **Purpose**: Provide foundation network structure for subsequent pipeline steps
- **Format**: SUMO-compatible XML network definition files