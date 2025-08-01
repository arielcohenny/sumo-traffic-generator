# SUMO Traffic Generator Specification

This document provides the formal specification for the SUMO Traffic Generator, detailing all available CLI arguments and the 9-step pipeline execution process.

## 0. Initialization and Setup

### 0.1 Data Directory Cleanup

- **Step**: Clean and prepare output directory
- **Function**: Directory cleanup via pipeline orchestration
- **Process**:
  - Remove existing `workspace/` directory if it exists (`shutil.rmtree(CONFIG.output_dir)`)
  - Create fresh `workspace/` directory (`CONFIG.output_dir.mkdir(exist_ok=True)`)
- **Purpose**: Ensures clean state for each simulation run

### 0.2 Command-Line Argument Parsing

- **Step**: Parse and validate all CLI arguments
- **Function**: `src/args/parser.py` via CLI orchestration
- **Process**: Parse 20 available arguments with defaults and validation
- **Available Arguments**: See section 0.3 below

### 0.3 Seed Initialization

- **Step**: Initialize random seed for reproducible simulations
- **Function**: Seed generation via pipeline orchestration
- **Process**:
  - Use provided `--seed` value if specified
  - Generate random seed if not provided: `random.randint(0, 2**32 - 1)`
  - Print seed value for reproduction: `print(f"Using seed: {seed}")`

### 0.4 Available Arguments

#### 0.4.1 `--grid_dimension` (float, default: 5)

Defines the grid's number of rows and columns for synthetic network generation. Not applicable when `--osm_file` is provided.

#### 0.4.2 `--block_size_m` (int, default: 200)

Sets block size in meters for grid network generation. Not applicable when `--osm_file` is provided.

#### 0.4.3 `--junctions_to_remove` (str, default: "0")

Number of junctions to remove or comma-separated list of specific junction IDs (e.g., "5" or "A0,B1,C2"). Not applicable when `--osm_file` is provided.

#### 0.4.4 `--lane_count` (str, default: "realistic")

Sets the lane count. 3 algorithms are available:

- `realistic`: Zone-based demand calculation
- `random`: Randomized within bounds (1-3 lanes)
- Integer value: Fixed count for all edges

#### 0.4.5 `--num_vehicles` (int, default: 300)

Total vehicles to generate.

#### 0.4.6 `--seed` (int, optional)

Controls randomization. If not provided, random seed is generated.

#### 0.4.7 `--step-length` (float, default: 1.0)

Simulation step length in seconds for TraCI control loop.

#### 0.4.8 `--end-time` (int, default: 86400)

Simulation duration in seconds.

#### 0.4.9 `--attractiveness` (str, default: "poisson")

Sets the departure and arrival attractiveness of each edge. Five methods available:

- `poisson`: Random distribution
- `land_use`: Zone-based calculation
- `gravity`: Distance-based model
- `iac`: Intersection accessibility calculation
- `hybrid`: Combined approach

#### 0.4.10 `--time_dependent` (flag)

Applies 4-phase variations to synthetic zone attractiveness.

#### 0.4.11 `--start_time_hour` (float, default: 0.0)

Real-world hour when simulation starts (0-24) for temporal attractiveness. Used with `--time_dependent` for phase calculation.

#### 0.4.12 `--departure_pattern` (str, default: "six_periods")

Vehicle departure timing. Four patterns available:

- `six_periods`: Research-based daily structure
- `uniform`: Even distribution
- `rush_hours:7-9:40,17-19:30,rest:10`: Custom rush hour definition
- `hourly:7:25,8:35,rest:5`: Granular hourly control

#### 0.4.13 `--routing_strategy` (str, default: "shortest 100")

Vehicle routing behavior. Four strategies with percentage mixing:

- `shortest`: Static shortest path
- `realtime`: 30-second dynamic rerouting
- `fastest`: 45-second fastest path rerouting
- `attractiveness`: Multi-criteria routing

#### 0.4.14 `--vehicle_types` (str, default: "passenger 60 commercial 30 public 10")

Vehicle type distribution. Three types with percentage assignment:

- `passenger`: Cars (5.0m length, 13.9 m/s max speed)
- `commercial`: Trucks (12.0m length, 10.0 m/s max speed)
- `public`: Buses (10.0m length, 11.1 m/s max speed)

#### 0.4.15 `--traffic_light_strategy` (str, default: "opposites")

Applied strategies for traffic lights. Two strategies available. Not applicable when `--osm_file` is provided.

- `opposites`: Opposing directions signal together
- `incoming`: Each edge gets separate phase

#### 0.4.16 `--traffic_control` (str, default: "tree_method")

Dynamic signal control. Three methods available:

- `tree_method`: Tree Method (Decentralized Bottleneck Prioritization Algorithm)
- `actuated`: SUMO gap-based control
- `fixed`: Static timing from configuration

#### 0.4.17 `--gui` (flag)

Launch SUMO GUI.

#### 0.4.18 `--osm_file` (str, optional)

Path to OSM file that replaces synthetic grid generation.

#### 0.4.19 `--land_use_block_size_m` (float, default: 25.0)

Zone cell size in meters for both OSM (intelligent zones) and non-OSM (traditional zones) mode.

**Default**: 25.0m for both network types (following research paper methodology from "A Simulation Model for Intra-Urban Movements")

**Purpose**: Controls the resolution of land use zone generation. Creates fine-grained cells for detailed spatial analysis independent of street block size.

#### 0.4.20 `--tree_method_sample` (str, optional)

Path to folder containing pre-built Tree Method sample files for bypass mode.

**Purpose**: Enables testing and validation using original research networks without generating new networks.

**Behavior**: 
- **Bypass Mode**: Skips Steps 1-8 entirely, goes directly to Step 9 (Dynamic Simulation)
- **File Requirements**: Folder must contain `network.net.xml`, `vehicles.trips.xml`, and `simulation.sumocfg.xml`
- **File Management**: Automatically copies and adapts sample files to our pipeline naming convention
- **Validation**: Tests our Tree Method implementation against established research benchmarks

**Incompatible Arguments**: Cannot be used with network generation arguments (`--osm_file`, `--grid_dimension`, `--block_size_m`, `--junctions_to_remove`, `--lane_count`)

**Usage Examples**:
```bash
# Basic Tree Method validation
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --gui

# Compare traffic control methods on identical network
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control actuated --gui
```

#### 0.4.21 `--custom_lanes` (str, optional)

Custom lane definitions for specific edges in synthetic grid networks.

**Format**: `"EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=..."`

**Supported Syntax**:
- **Full Specification**: `A1B1=tail:2,head:B1B0:1,B1C1:2` (custom tail + explicit head movements)
- **Tail-Only**: `A1B1=tail:2` (custom tail lanes, preserve existing movements)
- **Head-Only**: `A1B1=head:B1B0:1,B1C1:2` (automatic tail, custom movements)
- **Dead-End**: `A1B1=tail:2,head:` (create dead-end street)

**Multiple Edges**: Separate configurations with semicolons

**Constraints**: 
- Synthetic networks only (not compatible with `--osm_file`)
- Edge IDs must match grid pattern (A1B1, B2C2, etc.)
- Lane counts must be 1-3
- Mutually exclusive with `--custom_lanes_file`

#### 0.4.22 `--custom_lanes_file` (str, optional)

File containing custom lane definitions for complex scenarios.

**Format**: Same syntax as `--custom_lanes`, one configuration per line
**Features**: 
- Supports comments (lines starting with #)
- UTF-8 encoding required
- Line-specific error reporting
- Mutually exclusive with `--custom_lanes`

**Example File Content**:
```
# Custom lane configuration file
A1B1=tail:2,head:B1B0:1,B1C1:2
A2B2=tail:3,head:B2C2:2,B2B1:1
# Dead-end creation
D1E1=tail:2,head:
```

### 0.5 Argument Validation

- **Step**: Validate all CLI arguments before processing
- **Function**: Argument validation (to be implemented)
- **Process**: Comprehensive validation of all input parameters
- **Validations Required**:

#### 0.5.1 Routing Strategy Validation

- **Current**: Implemented in `parse_routing_strategy()` in `src/traffic/routing.py`
- **Checks**:
  - Format validation: Must be pairs of strategy name + percentage
  - Valid strategies: {"shortest", "realtime", "fastest", "attractiveness"}
  - Percentage range: 0-100 for each strategy
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

#### 0.5.2 Vehicle Types Validation

- **Current**: Implemented in `parse_vehicle_types()` in `src/traffic/vehicle_types.py`
- **Checks**:
  - Format validation: Must be pairs of vehicle type + percentage
  - Valid types: {"passenger", "commercial", "public"}
  - Percentage range: 0-100 for each type
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

#### 0.5.3 Departure Pattern Validation

- **Current**: Not implemented
- **Needed Checks**:
  - Valid pattern names: {"six_periods", "uniform"}
  - Format validation for "rush_hours:7-9:40,17-19:30,rest:10"
  - Format validation for "hourly:7:25,8:35,rest:5"
  - Hour range validation (0-24)
  - Percentage validation for custom patterns
  - Time range validation (start < end hours)

#### 0.5.4 OSM File Validation

- **Current**: Not implemented
- **Needed Checks**:
  - File existence verification
  - File format validation (XML structure)
  - OSM-specific validation (nodes, ways, bounds elements present)
  - File readability and permissions

#### 0.5.5 Numeric Range Validations

- **Current**: Basic type checking only
- **Needed Checks**:
  - `--grid_dimension`: > 0, reasonable upper bound (e.g., ≤ 20)
  - `--block_size_m`: > 0, reasonable range (50-1000m)
  - `--num_vehicles`: > 0, reasonable upper bound
  - `--step-length`: > 0, reasonable range (0.1-10.0 seconds)
  - `--end-time`: > 0
  - `--start_time_hour`: 0-24 range
  - `--land_use_block_size_m`: > 0, reasonable range (10-100m). **Default**: 25.0m for both network types (research paper methodology)

#### 0.5.6 Junctions to Remove Validation

- **Current**: String input only
- **Needed Checks**:
  - Format validation for comma-separated junction IDs
  - Numeric validation when integer count provided
  - Range validation (can't exceed grid capacity)
  - Junction ID format validation for specific removal

#### 0.5.7 Lane Count Validation

- **Current**: String acceptance only
- **Needed Checks**:
  - Valid algorithm names: {"realistic", "random"}
  - Integer validation for fixed count mode
  - Range validation for fixed counts (1-5 lanes)

#### 0.5.8 Cross-Argument Validation

- **Current**: Not implemented
- **Needed Checks**:
  - OSM file vs grid parameters (mutually exclusive usage)
  - Time-dependent features requiring appropriate end-time duration
  - Grid dimension vs junctions to remove capacity limits
  - Traffic light strategy compatibility with network type

#### 0.5.9 Choice Validations

- **Current**: Implemented via argparse choices
- **Existing Checks**:

  - `--attractiveness`: {"poisson", "land_use", "gravity", "iac", "hybrid"}
  - `--traffic_light_strategy`: {"opposites", "incoming"}
  - `--traffic_control`: {"tree_method", "actuated", "fixed"}

- **Error Handling**: Use existing `ValidationError` class for consistent error reporting

## 1. Network Generation

### 1.1 Mode Selection Based on Arguments

- **Step**: Determine network generation approach
- **Function**: Pipeline factory pattern (`src/pipeline/pipeline_factory.py`)
- **Process**:
  - Check if `--tree_method_sample` argument is provided
  - If Tree Method sample provided: Execute research dataset workflow (Section 1.4)
  - Check if `--osm_file` argument is provided
  - If OSM file provided: Execute OSM import workflow (Section 1.2)
  - If no OSM file: Execute synthetic grid generation workflow (Section 1.3)
- **Arguments Used**: `--tree_method_sample`, `--osm_file`

### 1.2 OSM Mode (`--osm_file` provided)

#### 1.2.1 OSM Network Import

- **Step**: Import real-world street network from OSM data
- **Function**: `import_osm_network()` in `src/network/import_osm.py`
- **Arguments Used**: `--osm_file`

##### OSM File Requirements:

- **Required Elements**: Nodes, ways with highway tags, bounding box
- **Highway Types**: Primary, secondary, tertiary, residential, unclassified streets
- **Lane Information**: Number of lanes per edge (preserved from OSM data)
- **Traffic Signals**: Original OSM traffic light locations (auto-generated if missing)
- **Minimum Content**: At least 10 highway ways for viable network
- **Area Limits**: Warning issued for areas > 5 km² (performance impact)

##### 12 Specialized netconvert Parameters:

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

#### 1.2.2 OSM File Organization

- **Step**: Move OSM-generated files to expected pipeline locations
- **Function**: File movement logic in `src/cli.py`
- **Process**:
  - Move files from `workspace/grid/osm_network.*` to `workspace/grid.*`
  - Handle file patterns: `*.nod.xml`, `*.edg.xml`, `*.con.xml`, `*.tll.xml`
  - Extract file extensions and rename to standard format
  - Clean up temporary `workspace/grid/` directory
  - Print movement confirmation for each file
- **Output**: `workspace/grid.nod.xml`, `workspace/grid.edg.xml`, `workspace/grid.con.xml`, `workspace/grid.tll.xml`

#### 1.2.3 Dead-End Street Handling (OSM-Specific)

- **Step**: Handle real-world street topology irregularities in OSM networks
- **Function**: Defensive programming in `src/traffic_control/decentralized_traffic_bottlenecks/classes/net_data_builder.py`
- **Scope**: Applied only to OSM networks - synthetic grids have no dead-ends since isolated junctions are completely removed

##### Dead-End Detection:

- **OSM Networks**: Contains natural dead-ends (cul-de-sacs, private roads, incomplete boundaries)
- **Synthetic Grids**: No dead-ends exist - junction removal process completely deletes isolated junctions and all connected edges
- **Algorithm Protection**: Uses `.get()` method instead of direct dictionary access to prevent crashes when edges have no outgoing connections

##### Handling Process:

- **Connection Building**: Build dictionary of edge-to-edge connections during network parsing
- **Missing Connections**: Dead-end streets have no entries in connections dictionary
- **Fallback Mechanism**: Return empty connection list for dead-end streets instead of crashing
- **Graph Integration**: Dead-end streets become terminal nodes in traffic control graph with no outgoing links

##### Network Topology Impact:

- **OSM Networks**: Dead-end streets preserved and handled gracefully by traffic control algorithm
- **Synthetic Grids**: Perfect topology with no special handling needed
- **Traffic Simulation**: Vehicles can enter dead-ends but must stop/turn around (handled by SUMO physics)
- **Signal Control**: Dead-end intersections receive traffic light control if signalized

#### 1.2.4 OSM Mode Validation

- **Step**: Verify OSM network import success
- **Function**: OSM validation in `src/network/import_osm.py`
- **Process**: Validate the following aspects, if any fails print a relevant message and exit.

##### Required Output Files:

- `workspace/grid.nod.xml`: Network nodes (junctions)
- `workspace/grid.edg.xml`: Network edges (streets)
- `workspace/grid.con.xml`: Connection definitions
- `workspace/grid.tll.xml`: Traffic light definitions

##### XML Structure Validation:

- **Parse XML Files**: Ensure valid XML structure and syntax
- **Element Validation**: Verify required elements (nodes, edges, connections, traffic lights)
- **Attribute Validation**: Check required attributes exist (id, coordinates, lane counts)
- **Reference Validation**: Ensure edge references valid nodes, connections reference valid edges

##### Network Connectivity and Integrity:

- **Node Coverage**: Verify all junctions are properly defined
- **Edge Coverage**: Count and validate highway ways (minimum 10 required)
- **Connection Coverage**: Ensure proper edge-to-edge connections
- **Traffic Light Coverage**: Validate signal placement and timing
- **Bounding Box**: Verify network area calculation and limits (warn if > 5 km²)
- **Highway Type Analysis**: Count primary, secondary, tertiary, residential, unclassified streets
- **Network Statistics**: Extract node count, edge count, connection count, traffic light count
- **Edge Length Statistics**: Calculate minimum, maximum, and average edge lengths

- **Success Message**: "Successfully imported OSM network."

#### 1.2.5 OSM Samples

- **Directory**: `osm_samples/` (relocated from `src/osm/`)
- **Purpose**: Verified working OSM areas for testing and development
- **Quality**: All samples tested with ≥294/300 vehicle route generation success rate

##### Available Samples:

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

##### Sample Management:

- **Download Script**: `tools/scripts/download_osm_samples.py` using Overpass API
- **Query Filter**: Highway types (motorway, trunk, primary, secondary, tertiary, unclassified, residential) and traffic signals
- **Testing Verified**: Each sample validated for reliable traffic simulation with high vehicle route generation success rates

### 1.3 Non-OSM Mode (synthetic grid)

#### 1.3.1 Grid Network Generation

- **Step**: Generate orthogonal grid network using SUMO netgenerate
- **Function**: `generate_grid_network()` in `src/network/generate_grid.py`

##### Arguments Used and Values:

- `--grid_dimension`: Number of rows and columns (e.g., 5 for 5×5 grid)
- `--block_size_m`: Block size in meters (e.g., 200m per block)
- `--junctions_to_remove`: Junction removal specification (e.g., "3" or "A1,B2,C3")
- `--lane_count`: Lane configuration algorithm ("realistic", "random", or integer like "2")
- `--traffic_light_strategy`: Traffic signal phasing strategy ("opposites" or "incoming")

##### Grid Generation Process:

- **Creates uniform grid** based on `--grid_dimension` and `--block_size_m`
- **Generates default 1-lane network** for consistent lane assignment
- **Applies traffic light strategy** using SUMO's `--tls.layout` functionality
- **Command**: `netgenerate --grid --grid.number {dimension} --grid.length {block_size}`

##### Junction Removal Process:

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

#### 1.3.2 Grid Network Validation

- **Step**: Verify synthetic grid generation
- **Function**: `verify_generate_grid_network()` in `src/validate/validate_network.py`

##### 14 Comprehensive Validation Steps:

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

##### Key Validation Constants:

- **HEAD_DISTANCE**: 50 meters (from config.py)
- **Junction Radius**: 10.0 meters (DEFAULT_JUNCTION_RADIUS)
- **Lane Bounds**: 1-5 lanes per edge maximum
- **Phase Duration**: 1-120 seconds valid range
- **Cycle Time**: 10-300 seconds valid range
- **Minimum Green Time**: 20% of total cycle time

- **Error Handling**: Raises `ValidationError` on failure, exits with code 1

#### 1.3.3 Grid Generation Completion

- **Step**: Confirm successful grid generation
- **Function**: Success logging in `src/cli.py`
- **Output**: Same file structure as OSM mode: `workspace/grid.nod.xml`, `workspace/grid.edg.xml`, `workspace/grid.con.xml`, `workspace/grid.tll.xml`
- **Success Message**: "Generated grid successfully."

### 1.4 Tree Method Research Dataset Mode (`--tree_method_sample` provided)

#### 1.4.1 Research Dataset Loading

- **Step**: Load pre-built Tree Method research networks for validation
- **Function**: `src/pipeline/sample_pipeline.py`
- **Arguments Used**: `--tree_method_sample`
- **Process**:
  - Validate sample folder exists and contains required files
  - Check for required files: `network.net.xml`, `vehicles.trips.xml`, `simulation.sumocfg.xml`
  - Copy sample files to workspace directory
  - Adapt file names to pipeline naming convention
- **Purpose**: Enable research validation against established Tree Method benchmarks

#### 1.4.2 File Management and Adaptation

- **Step**: Copy and rename sample files to match pipeline expectations
- **Function**: File copying logic in `src/pipeline/sample_pipeline.py`
- **Process**:
  - `network.net.xml` → `workspace/grid.net.xml`
  - `vehicles.trips.xml` → `workspace/vehicles.rou.xml`
  - `simulation.sumocfg.xml` → `workspace/grid.sumocfg`
  - Preserve original file content and structure
  - Update configuration file paths if needed
- **Validation**: Verify all required files copied successfully

#### 1.4.3 Research Network Characteristics

- **Dataset Source**: Original Tree Method research networks (decentralized-traffic-bottlenecks)
- **Network Complexity**: Complex urban topology with multi-lane edges and advanced traffic signals
- **Simulation Scale**: 946 vehicles over 2-hour simulation period (7300 seconds)
- **Research Context**: Pre-processed networks from original research (Experiment1-realistic-high-load)
- **Validation Purpose**: Test Tree Method implementation against published performance benchmarks

#### 1.4.4 Pipeline Bypass Behavior

- **Bypass Mode**: Skips Steps 1-8 entirely, proceeds directly to Step 9 (Dynamic Simulation)
- **Steps Skipped**: Network generation, zone extraction, edge splitting, attractiveness assignment, route generation
- **Benefits**: Immediate simulation execution, research validation, method comparison
- **Incompatible Arguments**: Cannot be used with network generation parameters (`--osm_file`, `--grid_dimension`, `--block_size_m`, `--junctions_to_remove`, `--lane_count`)

#### 1.4.5 Tree Method Research Dataset Completion

- **Step**: Confirm successful dataset loading and preparation
- **Function**: Success logging in `src/pipeline/sample_pipeline.py`
- **Success Message**: "Successfully loaded Tree Method research dataset."
- **Output**: Research network ready for immediate simulation execution
- **Ready for Step 9**: All components prepared for dynamic simulation with traffic control comparison

### 1.5 Common Network Generation Outputs

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

## 2. Zone Generation

### 2.1 Universal Land Use Zone Types (Both OSM and Non-OSM)

Based on "A Simulation Model for Intra-Urban Movements" research methodology, the system uses exactly **six research-based land use types** that apply to both OSM and synthetic networks:

#### Zone Type Definitions:

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

#### Universal Application:
- **Non-OSM Mode**: Uses clustering algorithm to assign these types to synthetic grid cells
- **OSM Mode**: Maps OSM land use tags to these standard types, falls back to intelligent inference when data insufficient

### 2.2 Mode Selection Based on Arguments

- **Step**: Determine zone generation approach based on network type
- **Function**: Pipeline step coordination (`src/pipeline/standard_pipeline.py`)
- **Process**:
  - Check if `--osm_file` argument was provided in Step 1
  - If OSM network: Execute intelligent zone generation workflow (Section 2.3)
  - If synthetic grid: Execute traditional zone extraction workflow (Section 2.4)
- **Arguments Used**: `--land_use_block_size_m` (affects both modes)

### 2.3 OSM Mode (intelligent zones)

#### 2.3.1 Geographic Bounds Extraction

- **Step**: Extract geographic boundaries from OSM file for zone generation
- **Function**: Bounds extraction in `src/cli.py`
- **Arguments Used**: `--osm_file`
- **Purpose**: Define geographic area for intelligent zone grid overlay
- **Process**:
  - Parse OSM XML file using ElementTree
  - Extract bounds from `<bounds>` element if present
  - If no bounds element, calculate from all node coordinates:
    - Iterate through all `<node>` elements in OSM file
    - Collect `lat` and `lon` attributes from each node
    - Calculate `min_lat = min(lats)`, `max_lat = max(lats)`, `min_lon = min(lons)`, `max_lon = max(lons)`
  - Create geographic bounds tuple: `(min_lon, min_lat, max_lon, max_lat)`
  - Print bounds for verification: `f"Using geographic bounds from OSM: {geographic_bounds}"`

#### 2.3.2 Intelligent Zone Generation

- **Step**: Generate intelligent land use zones using real OSM data and inference
- **Function**: `IntelligentZoneGenerator.generate_intelligent_zones_from_osm()` in `src/network/intelligent_zones.py`
- **Arguments Used**: `--land_use_block_size_m`
- **Process**: Uses sophisticated multi-layer analysis combining real OSM data with intelligent inference algorithms

##### Zone Configuration System:

- **Zone Types**: Uses the six research-based land use types defined in Section 2.1
- **OSM Mapping**: Maps OSM land use tags to standard zone types
- **Intelligent Inference**: When OSM data insufficient, infers zone types using network topology + accessibility + infrastructure analysis

##### OSM Data Loading and Parsing:

**1. OSM XML Processing** (`load_osm_data()`):

- **XML Parsing**: Uses ElementTree to parse OSM file structure
- **Node Extraction**: Stores all nodes with lat/lon coordinates and tag attributes
- **Way Processing**: Extracts ways with node references and tag collections
- **Relation Support**: Processes OSM relations with member structures
- **Tag Normalization**: Converts OSM tags to standardized key-value pairs
- **Error Handling**: Graceful fallback when OSM file is missing or corrupted

**2. Real Land Use Zone Extraction** (`extract_osm_zones_and_pois()`):

- **OSM Tag Mapping**: Maps OSM landuse/amenity/building tags to zone types:
  - `residential/apartments/housing` → residential
  - `commercial/retail/shop/office` → commercial
  - `industrial` → industrial
  - `school/university/college/education` → education
  - `hospital/clinic/healthcare` → healthcare
  - `mixed` → mixed
- **Polygon Generation**: Creates valid Shapely polygons from OSM way node sequences
- **Area Calculation**: Estimates zone area using rough geographic-to-metric conversion
- **Polygon Validation**: Ensures polygons are valid and have positive area

**3. Point of Interest (POI) Processing**:

- **Amenity Extraction**: Identifies amenity nodes (shops, restaurants, schools, hospitals)
- **POI Categorization**: Groups amenities by influence on zone classification
- **Geographic Points**: Creates Shapely Point geometries for spatial analysis

##### Multi-Layer Analysis System:

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

**Layer 3: OSM Infrastructure Analysis** (`analyze_osm_infrastructure()`):

- **Grid Cell Analysis**: Processes each grid cell individually with configurable search radius
- **POI Proximity Analysis**: Distance-weighted influence within 1.5x grid cell radius
- **POI-to-Zone Mapping**:
  - `shop/restaurant/cafe/bank/supermarket` → commercial (0.5 influence)
  - `school/university/college/library` → education (0.7 influence)
  - `hospital/clinic/pharmacy` → healthcare (0.6 influence)
  - `bus_station/subway_station` → mixed (0.4) + commercial (0.3)
  - `parking` → commercial (0.2 influence)
- **OSM Zone Integration**: Direct zone type scoring with 0.8 influence weight
- **Distance Decay**: `influence = max(0, 1.0 - (distance / search_radius))`

##### Grid System and Coordinate Handling:

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

##### Score Combination and Classification:

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
- **Zone Type Preservation**: Direct zone type mapping maintains OSM land use where available
- **Score Aggregation**: All factors combined before final classification

**Final Classification**:

- **Highest Score Selection**: `ZoneScore.get_highest_score_type()` determines final zone type
- **Polygon Generation**: Creates rectangular grid cell polygons with exact boundaries
- **Capacity Calculation**: `capacity = area_sqm * 0.02` (base capacity per square meter)
- **Metadata Storage**: Stores all analysis scores for debugging and verification

##### Fallback and Error Handling:

**Data Availability Checks**:

- **OSM Data Validation**: Graceful handling of missing or corrupted OSM files
- **Network Data Requirements**: Fallback algorithms when network analysis unavailable
- **Library Dependencies**: Optional NetworkX, GeoPandas, Pandas with simplified alternatives

**Default Assumptions**:

- **No OSM Data**: Falls back to residential (0.5 score) when no OSM zones/POIs found
- **Missing Network**: Uses simplified topology analysis without graph algorithms
- **Coordinate Transformation**: Maintains geographic coordinates when projection unavailable

##### Output and Integration:

**Zone Generation Results**:

- **Zone Metadata**: Each zone includes id, geometry, zone_type, area_sqm, capacity
- **Analysis Scores**: Stores topology, accessibility, infrastructure, and final scores
- **Grid Coordinates**: Maintains (i,j) grid position for reference
- **Intelligent Flag**: `is_intelligent: True` distinguishes from traditional zones

**Performance Characteristics**:

- **Computational Complexity**: O(n×m×p) where n=grid_cols, m=grid_rows, p=POI_count
- **Memory Usage**: Stores full OSM data in memory for spatial analysis
- **Processing Time**: Varies with OSM file size and grid resolution

#### 2.3.3 OSM Zone File Creation

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

#### 2.2.4 OSM Mode Validation

- **Step**: Verify intelligent zone generation success
- **Function**: Exception handling in `src/cli.py`
- **Process**:
  - Check zone file existence and size
  - Validate zone count is reasonable for network area
  - Verify zone type distribution is realistic
  - Confirm geographic coordinate format is valid
- **Error Handling**: Print failure message and exit with code 1 on validation failure
- **Failure Message**: `f"Failed to generate OSM zones: {e}"`

### 2.4 Non-OSM Mode (traditional zones)

#### 2.4.1 Traditional Zone Extraction

- **Step**: Extract zones from junction-based cellular grid methodology
- **Function**: `extract_zones_from_junctions()` in `src/network/zones.py`
- **Arguments Used**: `--land_use_block_size_m`
- **Research Basis**: Based on "A Simulation Model for Intra-Urban Movements" cellular grid methodology

##### Junction Coordinate Parsing:

- **Source File**: Parse coordinates from `workspace/grid.nod.xml`
- **Node Filtering**: Exclude internal nodes, include only junction nodes
- **Coordinate Extraction**: Extract x,y coordinates for grid boundary calculation
- **Network Bounds**: Calculate `network_xmin`, `network_xmax`, `network_ymin`, `network_ymax`

##### Cellular Grid Creation:

- **Cell Size Configuration**: Uses `--land_use_block_size_m` parameter (default 25.0m for both network types)
- **Grid Subdivision**: `num_x_cells = int((xmax - xmin) / cell_size)`, same for y-axis
- **Polygon Generation**: Create rectangular zones using Shapely `box()` geometry
- **Zone Independence**: Block size independent of junction spacing for flexible resolution

##### Land Use Classification System:

- **Zone Types**: Uses the six research-based land use types defined in Section 2.1
- **Distribution**: Follows exact percentages and cluster sizes from research paper
- **Assignment**: Uses clustering algorithm with BFS to create contiguous land use clusters

##### Clustering Algorithm:

- **BFS Clustering**: Uses breadth-first search to create contiguous land use clusters
- **Size Constraints**: Respects maximum cluster size per land use type
- **Spatial Distribution**: Ensures realistic geographic distribution of land uses
- **Randomization**: Uses provided seed for reproducible zone assignment

#### 2.4.2 Traditional Zone Validation

- **Step**: Verify traditional zone extraction success
- **Function**: `verify_extract_zones_from_junctions()` in `src/validate/validate_network.py`

##### Zone Count Validation:

- **Expected Calculation**: `expected_zones = num_x_cells * num_y_cells`
- **Actual Count**: Parse and count polygons in generated zones file
- **Count Verification**: Ensure actual zones match expected subdivision count
- **Error Condition**: Raise `ValidationError` if counts don't match

##### Zone Structure Validation:

- **File Existence**: Verify `workspace/zones.poly.xml` was created
- **XML Structure**: Parse and validate polygon XML format
- **Coordinate Bounds**: Ensure all zone coordinates within network bounds
- **Land Use Distribution**: Verify land use percentages approximate target distribution
- **Polygon Integrity**: Check all polygons are valid and non-overlapping

##### Validation Constants:

- **Minimum Zone Size**: Cell size must be > 0
- **Maximum Zones**: Reasonable upper limit based on cell size and network area
- **Distribution Tolerance**: Allow ±5% deviation from target land use percentages

#### 2.3.3 Traditional Zone Completion

- **Step**: Confirm successful traditional zone generation
- **Function**: Success logging in `src/cli.py`
- **Output**: `workspace/zones.poly.xml` with traditional zone extraction
- **Success Message**: `f"Extracted land use zones successfully using traditional method with {args.land_use_block_size_m}m blocks."`
- **Zone Information**: Log zone count, land use distribution, and file size

### 2.5 Common Zone Generation Outputs

- **File Generated** (both modes): `workspace/zones.poly.xml`
- **Format**: SUMO polygon XML with zone type, color, and coordinate information
- **Content**: Land use zones with attractiveness multipliers for traffic generation
- **Purpose**: Provide spatial context for edge attractiveness assignment in Step 6
- **Coordinate Systems**:
  - OSM mode: Geographic coordinates (converted to projected in Step 5)
  - Non-OSM mode: Projected coordinates (matches network coordinate system)

## 3. Integrated Edge Splitting with Lane Assignment

### 3.1 Unified Edge Splitting Process (Both OSM and Non-OSM)

#### 3.1.1 Network File Parsing

- **Step**: Parse all existing SUMO XML network files
- **Function**: `split_edges_with_flow_based_lanes()` in `src/network/split_edges_with_lanes.py`
- **Arguments Used**: `--lane_count`, `--seed`

##### File Loading Process:

- **Parse Nodes File**: Load `workspace/grid.nod.xml` for junction coordinates
- **Parse Edges File**: Load `workspace/grid.edg.xml` for edge definitions and geometry
- **Parse Connections File**: Load `workspace/grid.con.xml` for traffic movement analysis
- **Parse Traffic Lights File**: Load `workspace/grid.tll.xml` for signal connections
- **Validation**: Ensure all files exist and are valid XML structures

#### 3.1.2 Movement Analysis

- **Step**: Analyze traffic movements from connection data
- **Function**: `analyze_movements_from_connections()` in `src/network/split_edges_with_lanes.py`
- **Purpose**: Count total outgoing lane movements per edge to determine traffic demand

##### Movement Counting Process:

- **Connection Parsing**: Iterate through all `<connection>` elements in `.con.xml`
- **Lane-Level Counting**: Count total lanes used by all movements for each edge
- **Example**: Edge "A0B0" with connections:
  - To "B0C0" from 1 lane (lane 0)
  - To "B0A1" from 1 lane (lane 0)
  - To "B0B1" from 1 lane (lane 0)
  - **Total movement lanes**: 3 (each movement gets independent capacity after splitting)
- **Pre-Split State**: Before splitting, connections may share lanes (like all movements using lane 0)
- **Post-Split Result**: After edge splitting, each movement gets its own dedicated lane
- **Output**: Dictionary mapping edge IDs to total movement lane counts

#### 3.1.3 Edge Coordinate Extraction

- **Step**: Extract start and end coordinates for each edge
- **Function**: Coordinate extraction logic in `src/network/split_edges_with_lanes.py`

##### Coordinate Extraction Methods:

- **Primary Method**: Extract coordinates from node definitions in `.nod.xml`
- **Node Lookup**: Use edge `from` and `to` attributes to find corresponding nodes
- **Coordinate Assignment**: Assign `(x,y)` coordinates from node positions
- **Fallback Method**: Extract coordinates from edge `shape` attribute if nodes not found
- **Edge Length Calculation**: `edge_length = sqrt((end_x - start_x)² + (end_y - start_y)²)`

#### 3.1.4 Lane Count Determination

- **Step**: Determine lane count for each edge based on network type
- **Function**: Mode-specific lane count determination

##### OSM Mode:

- **Lane Count Source**: Extracted from OSM data during network import
- **OSM Lane Information**: Total number of lanes per edge preserved from original OSM data
- **Example**: If OSM data specifies edge has 2 lanes, `lane_count = 2`
- **No Algorithm**: Lane count is not calculated, only read from imported network

##### Non-OSM Mode (Three Lane Assignment Algorithms):

**1. Realistic Algorithm (`--lane_count realistic`):**

- **Zone Integration**: Load zone data from `workspace/zones.poly.xml`
- **Spatial Analysis**: Find zones adjacent to each edge using geometric intersection
- **Land Use Weights**: Apply traffic generation weights by zone type:
  - Mixed: 3.0 (highest traffic generation)
  - Employment: 2.5 (high peak traffic)
  - Entertainment/Retail: 2.5 (high commercial traffic)
  - Public Buildings: 2.0 (moderate institutional traffic)
  - Residential: 1.5 (moderate residential traffic)
  - Public Open Space: 1.0 (lower recreational traffic)
- **Demand Score Calculation**: Sum weighted zone attractiveness multipliers
- **Perimeter Modifier**: Apply 0.8x modifier for network perimeter edges
- **Lane Mapping**: Score < 1.5 → 1 lane, < 3.0 → 2 lanes, ≥ 3.0 → 3 lanes

**2. Random Algorithm (`--lane_count random`):**

- **Random Generation**: `rng.randint(min_lanes, max_lanes)` using seeded random number generator
- **Bounds**: Respects CONFIG.MIN_LANES (1) and CONFIG.MAX_LANES (3)
- **Reproducibility**: Uses provided seed for consistent results across runs

**3. Fixed Algorithm (`--lane_count [integer]`):**

- **Fixed Assignment**: Uses integer value as lane count for all edges
- **Bounds Checking**: `max(min_lanes, min(max_lanes, fixed_lanes))`
- **Example**: `--lane_count 2` assigns 2 lanes to all edges

#### 3.1.5 Edge Splitting Geometry Calculation

- **Step**: Calculate split points and segment geometry for each edge
- **Function**: Split point calculation in `src/network/split_edges_with_lanes.py`

##### Dynamic Head Distance Calculation:

- **OSM and Non-OSM Compatible**: `actual_head_distance = min(HEAD_DISTANCE, edge_length/3)`
- **HEAD_DISTANCE Constant**: 50 meters (CONFIG.HEAD_DISTANCE)
- **Short Street Protection**: Prevents geometric issues with short urban streets
- **All Edges Split**: Every edge is always split regardless of length

##### Split Point Geometry:

- **Split Ratio**: `ratio = (edge_length - actual_head_distance) / edge_length`
- **Split Coordinates**:
  - `split_x = start_x + ratio * (end_x - start_x)`
  - `split_y = start_y + ratio * (end_y - start_y)`
- **Segment Creation**:
  - **Tail Segment**: From original start to split point (uses `tail_lanes`)
  - **Head Segment**: From split point to original end (uses `head_lanes`)

##### Unified Lane Assignment Formula (Both OSM and Non-OSM):

- **Tail Lanes**: `lane_count` (from OSM data or calculated algorithm)
- **Head Lanes**: `max(lane_count, total_movement_lanes)`
- **Purpose**: Tail segment uses original lane count, head segment provides adequate capacity for all movements

#### 3.1.6 Movement-to-Head-Lane Assignment Algorithm

- **Step**: Assign each movement to a dedicated head lane (one movement per lane)
- **Function**: Movement assignment logic in `src/network/split_edges_with_lanes.py`
- **Purpose**: Ensure each traffic movement gets its own dedicated lane in the head segment

##### Movement Assignment Process:

**1. Movement Collection:**

- **Extract All Movements**: Parse all outgoing connections from original edge
- **Movement List**: Create ordered list of movements by destination
- **Multi-Lane Movements**: Each movement may use multiple pre-split lanes (e.g., straight movement from lanes 0,1,2)
- **Shared Lane Usage**: Multiple movements may share the same pre-split lanes (e.g., straight, right, and left all using lane 0)
- **Example**: Edge "A0B0" movements: [to_B0C0 from lanes 0,1,2], [to_B0A1 from lane 2], [to_B0B1 from lane 0]

**2. Head Lane Assignment:**

- **Individual Movement Assignment**: Each movement type gets dedicated head lanes equal to the number of pre-split lanes it used
- **Lane-to-Lane Mapping**: Each pre-split lane that served a movement gets its own dedicated head lane for that movement
- **No Lane Sharing**: After splitting, no two movements share the same head lane

**3. Connection Creation:**

- **Head-to-External Connections**: Each head lane connects to its assigned destination
- **Lane Specification**: Connection specifies exact fromLane and toLane indices
- **Example Connections**:
  - `<connection from="A0B0_H" to="B0C0" fromLane="0" toLane="0"/>`
  - `<connection from="A0B0_H" to="B0A1" fromLane="1" toLane="0"/>`
  - `<connection from="A0B0_H" to="B0B1" fromLane="2" toLane="0"/>`

##### Movement Priority Algorithm:

**Movement Ordering Principle:**

- **One Movement Per Head Lane**: Each movement gets exactly one dedicated head lane, regardless of how many lanes it used before splitting
- **Spatial Preference**: Movements are assigned based on turn direction when possible
- **Sequential Assignment**: When spatial preference conflicts occur, use sequential order (0, 1, 2...)

**Example Movement Assignments:**

**Scenario 1: 3 movements from 1 shared lane → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 0)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 0)** → Head lane 2

**Scenario 2: 2 movements from 2 different lanes → 2-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 1)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0

**Scenario 3: 3 movements from 3 lanes → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 1,2)** → Head lane 1
- **Movement 3: Right turn to B0B1 (originally from lane 0)** → Head lane 0

**Scenario 4: 4 movements from 3 lanes → 6-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 0,1,2)** → Head lanes 1,2,3
- **Movement 3: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 4: Left turn to B0A1 (originally from lane 2)** → Head lane 4
- **Movement 5: U-turn to B0A0 (originally from lane 2)** → Head lane 5

**Scenario 5: 3 movements from 3 lanes → 5-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 0,1,2)** → Head lane 1,2,3
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 2)** → Head lane 4

**Scenario 6: 3 movements from 2 lanes → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 0)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 1)** → Head lane 2

**Key Principle**: The pre-split lane usage is irrelevant - what matters is that we have N distinct movements that each need their own dedicated head lane. The algorithm counts movements, not the lanes they previously used.

#### 3.1.7 Tail-to-Head Lane Connection Algorithm

- **Step**: Create internal connections between tail and head segments
- **Function**: Internal connection generation logic
- **Purpose**: Connect each tail lane to appropriate head lanes for traffic flow continuity

##### Connection Distribution Cases:

**Case 1: Tail Lanes ≤ Head Lanes:**

- **Algorithm**: Each tail lane connects to multiple head lanes (even distribution)
- **Calculation**:
  - `connections_per_tail = head_lanes // tail_lanes`
  - `extra_connections = head_lanes % tail_lanes`
- **Distribution**:
  - Each tail lane gets `connections_per_tail` head lane connections
  - First `extra_connections` tail lanes get one additional head lane connection
- **Example**: 2 tail → 3 head:
  - Tail lane 0 → Head lanes 0, 1 (2 connections)
  - Tail lane 1 → Head lane 2 (1 connection)

**Case 2: Tail Lanes > Head Lanes:**

- **Algorithm**: Multiple tail lanes connect to each head lane
- **Calculation**:
  - `tails_per_head = tail_lanes // head_lanes`
  - `extra_tails = tail_lanes % head_lanes`
- **Distribution**:
  - Each head lane receives connections from `tails_per_head` tail lanes
  - First `extra_tails` head lanes receive one additional tail lane connection
- **Example**: 3 tail → 2 head:
  - Head lane 0 ← Tail lanes 0, 1 (2 tail lanes)
  - Head lane 1 ← Tail lane 2 (1 tail lane)

**Case 3: Tail Lanes = Head Lanes:**

- **Algorithm**: One-to-one mapping
- **Connection**: Tail lane i → Head lane i
- **Example**: 2 tail → 2 head: tail[0]→head[0], tail[1]→head[1]

##### Sequential Lane Assignment:

- **Left-to-Right Order**: Tail lanes connect to head lanes in sequential order (0,1,2...)
- **Continuous Mapping**: Maintains lane continuity for traffic flow
- **Example Detailed Mapping** (2 tail → 3 head):
  - Connection: tail_lane=0, from_lane=0 → head_lane=0, to_lane=0
  - Connection: tail_lane=0, from_lane=0 → head_lane=1, to_lane=0
  - Connection: tail_lane=1, from_lane=1 → head_lane=2, to_lane=0

#### 3.1.8 XML File Updates

- **Step**: Update all four SUMO XML files with split edges and lane assignments
- **Function**: File update functions in `src/network/split_edges_with_lanes.py`

##### 1. Nodes File (.nod.xml) Updates:

- **Add Split Nodes**: Create new intermediate nodes at calculated split points
- **Node Naming Convention**: Original edge "A0B0" creates intermediate node "A0B0_H_node"
- **Node Attributes**: Include coordinates (x,y at split point), junction type, and radius=10.0
- **XML Structure**: `<node id="A0B0_H_node" x="150.0" y="200.0" radius="10.0"/>`
- **Coordinate Calculation**: Split point positioned at HEAD_DISTANCE (50m) from downstream junction

##### 2. Edges File (.edg.xml) Updates:

- **Replace Original Edges**: Remove original single edges completely from XML
- **Add Tail Segments**: Create edges from original start to split point with `tail_lanes`
  - **Tail Naming**: Original edge "A0B0" becomes tail segment "A0B0" (keeps original ID)
  - **Tail Structure**: `<edge id="A0B0" from="A0" to="A0B0_H_node" numLanes="2" speed="13.89" priority="-1"/>`
  - **Tail Attributes**: Preserves original speed, priority; uses calculated tail lane count
- **Add Head Segments**: Create edges from split point to original end with `head_lanes`
  - **Head Naming**: Original edge "A0B0" becomes head segment "A0B0_H"
  - **Head Structure**: `<edge id="A0B0_H" from="A0B0_H_node" to="B0" numLanes="5" speed="13.89" priority="-1"/>`
  - **Head Attributes**: Preserves original speed, priority; uses calculated head lane count (≥ movement count)
- **Shape Coordinates**: Include explicit shape coordinates for geometric accuracy

##### 3. Connections File (.con.xml) Updates:

- **Update External Connections**: Modify existing connections to reference head segments with movement-to-lane assignments
  - **Original**: `<connection from="A0B0" to="B0C0" fromLane="0" toLane="0"/>`
  - **Updated**: `<connection from="A0B0_H" to="B0C0" fromLane="1" toLane="0"/>` (head segment with spatial lane assignment)
- **Add Internal Connections**: Create new connections between tail and head segments using distribution algorithm
  - **Tail-to-Head**: `<connection from="A0B0" to="A0B0_H" fromLane="0" toLane="0"/>`
  - **Distribution**: Each tail lane connects to multiple head lanes based on calculated ratios
- **Movement-Specific Lane Assignment**: Each movement gets dedicated head lanes based on spatial logic
  - **Right turns**: Rightmost head lanes (e.g., lane 0)
  - **Straight movements**: Middle head lanes (e.g., lanes 1,2,3)
  - **Left turns**: Leftmost head lanes (e.g., lane 4)
- **Lane Indexing**: Ensure proper `fromLane` and `toLane` indexing for movement distribution

##### 4. Traffic Lights File (.tll.xml) Updates:

- **Update Connection References**: Modify traffic light connections to reference head segments
  - **Original**: `<connection from="A0B0" to="B0C0" fromLane="0" toLane="0"/>`
  - **Updated**: `<connection from="A0B0_H" to="B0C0" fromLane="1" toLane="0"/>` (matches connections file)
- **Preserve Signal Logic**: Maintain original phase timing and state strings
  - **Phase Structure**: Keep existing `<phase duration="30" state="GGrr"/>` definitions
  - **TLS Programs**: Preserve traffic light program IDs and logic
- **Link Index Adjustment**: Update linkIndex values to match new connection structure
  - **Recalculate Indices**: Adjust linkIndex to correspond to new head segment connections
  - **State String Length**: Ensure state strings match total number of connections after splitting

#### 3.1.9 Edge Splitting Validation

- **Step**: Comprehensive validation of edge splitting and lane assignment
- **Function**: `verify_split_edges_with_flow_based_lanes()` in `src/validate/validate_split_edges_with_lanes.py`
- **Execution**: Runs after network rebuild (Step 4) to validate final network structure

##### Primary Validation Checks:

1. **Tail Lane Consistency**: Verify tail segments have same lane count as original edges
2. **Head Lane Accuracy**: Confirm head segments have correct total movement lanes
3. **Movement Integrity**: Ensure all head lanes have exactly one outgoing direction  
4. **Connectivity Validation**: Check all head lanes have incoming connections from tail lanes
5. **Complete Flow**: Verify all tail lanes lead to head lanes

##### Structural Validation Checks:

- **Network Completeness**: Validate all expected tail and head segments exist
- **Lane Index Consistency**: Ensure proper sequential lane numbering (0,1,2...)
- **Connection Bounds**: Verify all connections reference valid lane indices
- **Edge Mapping**: Confirm proper edge-to-movement-to-lane assignments

##### Validation Process:

- **Network Parsing**: Extract lane information from rebuilt `.net.xml` file
- **Connection Analysis**: Parse movement data from `.con.xml` file  
- **Cross-Validation**: Compare network structure against connection requirements
- **Error Reporting**: Detailed error messages for any validation failures
- **Success Confirmation**: "✅ VALIDATION PASSED: X edges validated successfully"

##### Validation Timing:

- **Pre-Network Rebuild**: Not executed (head segments don't exist yet)
- **Post-Network Rebuild**: Executed after Step 4 completion
- **Integration**: Seamlessly integrated into CLI pipeline with proper error handling

#### 3.1.10 Edge Splitting Completion

- **Step**: Confirm successful edge splitting with lane assignment
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Successfully completed integrated edge splitting with flow-based lane assignment."

### 3.2 Common Edge Splitting Outputs

- **Files Modified** (both modes):
  - `workspace/grid.nod.xml`: Network nodes with added split points
  - `workspace/grid.edg.xml`: Edges replaced with tail/head segments and lane counts
  - `workspace/grid.con.xml`: Connections updated with internal connections and lane distribution
  - `workspace/grid.tll.xml`: Traffic lights updated to reference head segments
- **Purpose**: Create optimized network structure with appropriate lane capacity for traffic movements
- **Integration**: Unified algorithm works seamlessly with both OSM and synthetic grid networks
- **Configuration Constants**:
  - HEAD_DISTANCE: 50 meters
  - MIN_LANES: 1
  - MAX_LANES: 3
  - Junction radius: 10.0 meters

## 4. Network Rebuild

### 4.1 Purpose and Necessity

After edge splitting and lane assignment modifications, the SUMO network must be rebuilt to:

1. **Consolidate XML Changes**: Integrate all modifications from the 4 separate XML files (.nod, .edg, .con, .tll) into a single coherent network file
2. **Generate Final Network**: Create the definitive `grid.net.xml` file that SUMO simulation engine will use
3. **Resolve Dependencies**: Process interdependencies between nodes, edges, connections, and traffic lights that were modified during splitting
4. **Validate Network Integrity**: Ensure all split edges, new intermediate nodes, and updated connections form a valid SUMO network
5. **Enable Simulation**: Produce a complete network file required for vehicle route generation and simulation execution

### 4.2 Network Rebuild Process

- **Step**: Rebuild SUMO network from all modified XML components
- **Function**: `rebuild_network()` in `src/sumo_integration/sumo_utils.py`
- **Tool Used**: SUMO's `netconvert` utility
- **Input Files**: Modified XML files from edge splitting process:
  - `workspace/grid.nod.xml` (nodes with new intermediate split nodes)
  - `workspace/grid.edg.xml` (edges with tail/head segments)
  - `workspace/grid.con.xml` (connections with movement-specific lane assignments)
  - `workspace/grid.tll.xml` (traffic lights with updated connection references)
- **Output**: `workspace/grid.net.xml` (complete SUMO network ready for simulation)

### 4.3 Technical Requirements

**Why Rebuilding is Essential:**

- **SUMO Architecture**: SUMO requires a compiled `.net.xml` file that contains the complete network definition
- **Dependency Resolution**: `netconvert` resolves all geometric calculations, lane geometries, and junction logic
- **Coordinate System**: Establishes final projected coordinate system for all network elements
- **Internal Links**: Generates internal lane connections within junctions that connect incoming and outgoing edges
- **Traffic Light Integration**: Properly links traffic light programs with the actual network topology

**Rebuild Command Process:**

- **Netconvert Execution**: Runs SUMO's `netconvert` tool with appropriate parameters
- **File Integration**: Combines all 4 XML input files into single network representation
- **Geometric Processing**: Calculates final lane geometries, junction shapes, and connection curves
- **Validation**: Ensures network is topologically sound and ready for simulation

### 4.4 Network Rebuild Validation

- **Step**: Verify network rebuild was successful
- **Function**: `verify_rebuild_network()` in `src/validate/validate_network.py`
- **Validation Checks**:
  - **File Existence**: Confirm `workspace/grid.net.xml` was generated successfully
  - **XML Validity**: Ensure output file is valid XML with proper SUMO network structure
  - **Edge Preservation**: Verify all split edges (tail and head segments) exist in final network
  - **Node Integration**: Confirm all intermediate split nodes are properly integrated
  - **Connection Consistency**: Validate all movement-specific lane assignments are preserved
  - **Geometric Integrity**: Check that edge geometries and junction shapes are correctly calculated

### 4.5 Network Rebuild Completion

- **Step**: Confirm successful network rebuild
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Rebuilt the network successfully."
- **Ready for Next Steps**: Network is now prepared for zone coordinate conversion (OSM mode) and edge attractiveness assignment

## 5. Zone Coordinate Conversion (OSM Mode Only)

### 5.1 Purpose and Timing

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

### 5.2 OSM Mode Coordinate Conversion Process

- **Step**: Convert zone coordinates from geographic (lat/lon) to projected (x,y)
- **Function**: `convert_zones_to_projected_coordinates()` in `src/network/intelligent_zones.py`
- **Arguments Used**: Existing `workspace/zones.poly.xml` and `workspace/grid.net.xml`
- **Timing**: Only executed when `--osm_file` argument was used in Step 1

#### 5.2.1 Coordinate Transformation Process

- **Source Coordinates**: Geographic coordinates (latitude, longitude) from OSM data
- **Target Coordinates**: Projected coordinates (x, y in meters) matching SUMO network
- **Transformation Method**: Uses SUMO's coordinate system parameters from the rebuilt network
- **Precision**: Maintains geometric accuracy for spatial zone-edge analysis

#### 5.2.2 Zone File Update

- **Input File**: `workspace/zones.poly.xml` with geographic coordinates
- **Output File**: `workspace/zones.poly.xml` updated with projected coordinates (same filename, converted content)
- **Preservation**: Maintains all zone properties (type, color, attractiveness) while updating only coordinates
- **Validation**: Ensures all zone polygons remain valid after coordinate transformation

#### 5.2.3 Spatial Consistency Verification

- **Coordinate System Match**: Verify zones and network edges use identical coordinate systems
- **Geometric Integrity**: Ensure zone polygons maintain proper shapes after transformation
- **Network Bounds**: Confirm converted zones fall within reasonable bounds of the network area

### 5.3 Non-OSM Mode (No Action Required)

For synthetic grid networks (when `--osm_file` is not provided):

- **No Conversion Needed**: Zones already created in projected coordinates during Step 2
- **Coordinate System Match**: Traditional zones use same coordinate system as synthetic network
- **Direct Spatial Analysis**: Zones ready for edge attractiveness assignment without conversion

### 5.4 Zone Coordinate Conversion Completion

- **Step**: Confirm successful coordinate conversion (OSM mode only)
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Successfully converted zone coordinates to projected system."
- **Error Handling**: Falls back gracefully if conversion fails: "Zones will remain in geographic coordinates."
- **Ready for Next Step**: Zones are now in correct coordinate system for spatial edge attractiveness analysis

## 6. Edge Attractiveness Assignment

### 6.1 Purpose and Process Overview

**Purpose of Edge Attractiveness Assignment:**

- **Traffic Generation Foundation**: Determines how many vehicles depart from and arrive at each edge during simulation
- **Realistic Traffic Patterns**: Creates non-uniform traffic distribution that reflects real-world urban traffic flow
- **Spatial Traffic Modeling**: Accounts for land use patterns, accessibility, and network topology in traffic generation
- **Temporal Traffic Variation**: Enables time-of-day traffic patterns (rush hours, off-peak periods)

**Universal Application**: Works for both OSM and non-OSM networks after proper zone coordinate alignment

### 6.2 Edge Attractiveness Assignment Process

- **Step**: Assign departure and arrival attractiveness values to all network edges
- **Function**: `assign_edge_attractiveness()` in `src/network/edge_attrs.py`
- **Arguments Used**: `--attractiveness`, `--time_dependent`, `--start_time_hour`, `--seed`
- **Input Files**:
  - `workspace/grid.net.xml` (rebuilt network with final edge definitions)
  - `workspace/zones.poly.xml` (zones with correct coordinate system from Steps 2/5)
  - `workspace/grid.edg.xml` (original edge file for spatial analysis methods)

#### 6.2.1 Five Attractiveness Calculation Methods

**1. Poisson Method (`--attractiveness poisson`)**

- **Algorithm**: Uses Poisson probability distribution to generate random attractiveness values
- **Process**: For each edge, draws random numbers from two separate Poisson distributions - one for departures (λ=3.5) and one for arrivals (λ=2.0). The Poisson distribution naturally produces positive integer values with realistic variation around the mean.
- **Mathematical Basis**: Poisson distributions model random events occurring at a constant average rate, making them suitable for modeling traffic generation where events (vehicle trips) happen independently
- **Characteristics**: Produces values clustered around the lambda parameters with occasional higher values, creating natural traffic variation without spatial bias
- **Use Case**: Baseline random traffic generation without spatial considerations
- **Independence**: No dependency on zones or network topology

**2. Land Use Method (`--attractiveness land_use`)**

- **Algorithm**: Calculates attractiveness based purely on the land use characteristics of zones adjacent to each edge
- **Process**: For each edge, identifies all zones within 10 meters using geometric intersection analysis. Calculates base attractiveness from the density and type of adjacent zones, with each zone contributing based on its attractiveness value and land use multipliers. When multiple zones are adjacent, computes weighted average where zones with higher attractiveness values have proportionally more influence. Final attractiveness values reflect pure spatial land use patterns without random components.
- **🔴 IMPLEMENTATION FIX NEEDED**: Current implementation incorrectly starts with random Poisson baseline values (`depart_base = np.random.poisson(lam=CONFIG.LAMBDA_DEPART)`) before applying land use multipliers. This introduces random variation unrelated to spatial land use patterns.
- **🔴 REQUIRED CHANGES**:
  - Remove Poisson baseline calculation from `calculate_attractiveness_land_use()` function
  - Calculate base attractiveness directly from zone density (e.g., sum of adjacent zone attractiveness values)
  - Apply land use multipliers to the zone-derived base values instead of random Poisson values
  - Ensure edges with no adjacent zones get minimal but non-zero attractiveness (e.g., default value of 1)
- **Land Use Logic**: Different land use types generate different traffic patterns - residential areas attract incoming traffic (people coming home) while employment areas generate outgoing traffic (people leaving for work). Mixed-use areas have balanced patterns.
- **Spatial Analysis**: Uses geometric intersection and distance calculations to determine edge-zone adjacency, ensuring edges near commercial districts get higher arrival attractiveness while edges near residential areas get higher departure attractiveness
- **Zone Types**: Residential, Employment, Mixed, Entertainment/Retail, Public Buildings, Public Open Space
- **Examples**:
  - Residential areas: High arrival (1.4x), moderate departure (0.8x) - people return home
  - Employment areas: High departure (1.3x), moderate arrival (0.9x) - people leave for work

**3. Gravity Method (`--attractiveness gravity`)**

- **Algorithm**: Models traffic attractiveness based on network accessibility and connectivity patterns
- **Process**: For each edge, calculates a "cluster size" by counting how many other edges connect to the same start and end nodes, representing the edge's position in the network hierarchy. Applies exponential decay with distance (using normalized distance of 1.0) and exponential growth with cluster connectivity. Multiplies by a random baseline factor to introduce variation.
- **Network Theory**: Based on gravity models from transportation planning where locations with better connectivity (more connections) attract more traffic, similar to how larger cities attract more travelers in regional models
- **Centrality Logic**: Edges connecting highly connected nodes (major intersections) get higher attractiveness than edges connecting peripheral nodes, reflecting real-world patterns where arterial roads carry more traffic than residential streets
- **Parameters**: `d_param = 0.95` (distance decay), `g_param = 1.02` (connectivity amplification)
- **Stochastic Element**: Includes random baseline factor to prevent purely deterministic results

**4. IAC Method (`--attractiveness iac`)**

- **Algorithm**: Integrated Attraction Coefficient that synthesizes multiple urban planning factors into a comprehensive attractiveness measure
- **Process**: Calculates separate gravity and land use components, then normalizes them against baseline values to create dimensionless factors. Introduces a random "mood" factor representing daily variations in travel behavior and a spatial preference factor. Multiplies all components together with a base attractiveness coefficient to produce the final IAC value.
- **Multi-Factor Integration**: Combines the network connectivity insights from gravity models with the land use patterns from zoning analysis, while accounting for behavioral unpredictability through stochastic elements
- **Research Foundation**: Based on established IAC methodology from urban traffic modeling literature that recognizes traffic generation as a function of both infrastructure (connectivity) and land use (activity patterns)
- **Normalization Strategy**: Converts gravity and land use results to relative factors (comparing against baseline Poisson values) so they can be meaningfully combined regardless of their original scales
- **Behavioral Elements**: Includes random mood factor and spatial preference to capture human decision-making variability in travel choices

**5. Hybrid Method (`--attractiveness hybrid`)**

- **Algorithm**: Combines multiple methodologies using a carefully balanced weighting scheme to capture benefits of each approach while mitigating individual weaknesses
- **Process**: Starts with pure Poisson values as the foundation, then calculates land use and gravity adjustments separately. Converts these adjustments to multiplicative factors relative to Poisson baselines, then reduces their impact (land use to 50%, gravity to 30%) to prevent any single method from dominating. Applies these dampened factors sequentially to the base Poisson values.
- **Weighting Philosophy**: Uses Poisson as the stable foundation (100% weight) because it provides consistent baseline variation. Land use gets moderate influence (50%) to incorporate spatial realism without over-constraining results. Gravity gets lighter influence (30%) to add network topology awareness without creating extreme centrality bias.
- **Robustness Strategy**: By combining methods with reduced individual impacts, the hybrid approach is less sensitive to problems in any single methodology (e.g., poor zone data or network topology issues) while still benefiting from their insights
- **Computational Balance**: Provides realistic spatial and topological variation while maintaining computational efficiency and avoiding the complexity of full IAC integration

#### 6.2.2 Temporal Variation System (4-Phase)

**Time-Dependent Mode (`--time_dependent` flag):**

**Phase Definition**:

- **Morning Peak (6:00-9:30)**: High outbound traffic (home→work), multipliers: depart 1.4x, arrive 0.7x
- **Midday Off-Peak (9:30-16:00)**: Balanced baseline traffic, multipliers: depart 1.0x, arrive 1.0x
- **Evening Peak (16:00-19:00)**: High inbound traffic (work→home), multipliers: depart 0.7x, arrive 1.5x
- **Night Low (19:00-6:00)**: Minimal activity, multipliers: depart 0.4x, arrive 0.4x

**Implementation**:

- **Base Calculation**: Generate base attractiveness using selected method without time dependency
- **Phase-Specific Profiles**: Create attractiveness values for each of 4 phases using multipliers
- **Active Phase**: Set current phase based on `--start_time_hour` parameter
- **Attribute Storage**: Store both base values and all phase-specific values in network file

#### 6.2.3 Edge Processing and Filtering

**Edge Selection**:

- **Include**: All regular network edges (roads, streets)
- **Exclude**: Internal edges (starting with ":") used for junction connections
- **Processing**: Iterate through all edges in rebuilt network file

**Attribute Assignment**:

- **Standard Mode**: `depart_attractiveness` and `arrive_attractiveness` attributes
- **Time-Dependent Mode**: Additional phase-specific attributes:
  - `morning_peak_depart_attractiveness`, `morning_peak_arrive_attractiveness`
  - `midday_offpeak_depart_attractiveness`, `midday_offpeak_arrive_attractiveness`
  - `evening_peak_depart_attractiveness`, `evening_peak_arrive_attractiveness`
  - `night_low_depart_attractiveness`, `night_low_arrive_attractiveness`

### 6.3 Spatial Analysis Integration

**Zone-Edge Adjacency Detection** (for land_use, iac, hybrid methods):

- **Edge Geometry**: Parse edge shape coordinates from network
- **Zone Polygon**: Load zone polygon coordinates from zones file
- **Spatial Query**: Determine if edge intersects or is within 10 meters of zone polygon
- **Multiple Zones**: Handle cases where edges are adjacent to multiple zones
- **Weighted Calculation**: Average attractiveness based on zone attractiveness values and types

### 6.4 Edge Attractiveness Validation

- **Step**: Verify attractiveness assignment was successful
- **Function**: `verify_assign_edge_attractiveness()` in `src/validate/validate_network.py`
- **Validation Checks**:
  - **Attribute Existence**: Confirm all edges have required attractiveness attributes
  - **Value Ranges**: Ensure attractiveness values are positive integers
  - **Distribution Check**: Verify reasonable distribution across network edges
  - **Time-Dependent Validation**: For temporal mode, check all phase-specific attributes exist
  - **Spatial Consistency**: Verify spatial methods produce location-appropriate values

### 6.5 Edge Attractiveness Assignment Completion

- **Step**: Confirm successful attractiveness assignment
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Assigned edge attractiveness successfully."
- **Output File**: Updated `workspace/grid.net.xml` with attractiveness attributes on all edges
- **Ready for Next Step**: Network edges now have traffic generation parameters for vehicle route generation

## 7. Vehicle Route Generation

### 7.1 Purpose and Process Overview

**Purpose of Vehicle Route Generation:**

- **Traffic Demand Realization**: Converts edge attractiveness values into actual vehicle trips with specific origins, destinations, and departure times
- **Realistic Vehicle Mix**: Creates diverse vehicle fleet reflecting real-world traffic composition
- **Routing Behavior**: Assigns different routing strategies to vehicles to simulate realistic navigation patterns
- **Temporal Distribution**: Distributes vehicle departures over time according to realistic daily patterns

### 7.2 Vehicle Route Generation Process

- **Step**: Generate complete vehicle route definitions for simulation
- **Function**: `generate_vehicle_routes()` in `src/traffic/builder.py`
- **Arguments Used**: `--num_vehicles`, `--vehicle_types`, `--departure_pattern`, `--routing_strategy`, `--seed`, `--end_time`
- **Input Files**:
  - `workspace/grid.net.xml` (network with attractiveness attributes)
  - Uses edge attractiveness values for origin/destination selection
  - Network topology for route calculation

#### 7.2.1 Vehicle Type System (3-Type Classification)

**Vehicle Type Distribution (`--vehicle_types` parameter):**

**1. Passenger Vehicles (Default: 60%)**

- **Vehicle Class**: Personal cars, sedans, SUVs
- **SUMO Definition**: `vClass="passenger"`
- **Physical Characteristics**:
  - Length: 5.0 meters
  - Max Speed: 13.9 m/s (50 km/h)
  - Acceleration: 2.6 m/s²
  - Deceleration: 4.5 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Most common vehicle type, represents private transportation

**2. Commercial Vehicles (Default: 30%)**

- **Vehicle Class**: Delivery trucks, freight vehicles, commercial vans
- **SUMO Definition**: `vClass="truck"`
- **Physical Characteristics**:
  - Length: 12.0 meters
  - Max Speed: 10.0 m/s (36 km/h)
  - Acceleration: 1.3 m/s²
  - Deceleration: 4.0 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Larger, slower vehicles representing freight and delivery traffic

**3. Public Transportation (Default: 10%)**

- **Vehicle Class**: Buses, public transit vehicles
- **SUMO Definition**: `vClass="bus"`
- **Physical Characteristics**:
  - Length: 10.0 meters
  - Max Speed: 11.1 m/s (40 km/h)
  - Acceleration: 1.2 m/s²
  - Deceleration: 4.0 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Public transit vehicles with specific operating characteristics

**Vehicle Type Validation:**

- **Percentage Sum**: Must total exactly 100%
- **Format**: "passenger 70 commercial 20 public 10"
- **Assignment**: Each generated vehicle randomly assigned type based on percentages

#### 7.2.2 Departure Pattern System

**Departure Pattern Distribution (`--departure_pattern` parameter):**

**1. Six Periods Pattern (Default: "six_periods")**

- **Research Basis**: Based on established 6-period daily traffic structure
- **Time Periods**:
  - Morning (6:00-7:30): 20% of daily traffic
  - Morning Rush (7:30-9:30): 30% of daily traffic
  - Noon (9:30-16:30): 25% of daily traffic
  - Evening Rush (16:30-18:30): 20% of daily traffic
  - Evening (18:30-22:00): 4% of daily traffic
  - Night (22:00-6:00): 1% of daily traffic
- **Distribution**: Vehicles assigned departure times within periods based on percentages

**2. Uniform Pattern ("uniform")**

- **Distribution**: Even distribution across entire simulation time
- **Calculation**: `departure_time = random_uniform(0, end_time)`
- **Use Case**: Baseline comparison without temporal bias

**3. Custom Rush Hours Pattern ("rush_hours:7-9:40,17-19:30,rest:10")**

- **Format**: Defines specific rush hour periods with percentages, remainder distributed to other times
- **Flexibility**: Allows custom peak periods for specific scenarios

**4. Granular Hourly Pattern ("hourly:7:25,8:35,rest:5")**

- **Format**: Assigns specific percentages to individual hours
- **Control**: Fine-grained temporal control for detailed analysis

#### 7.2.3 Routing Strategy System (4-Strategy Classification)

**Routing Strategy Assignment (`--routing_strategy` parameter):**

**1. Shortest Path Strategy ("shortest")**

- **Algorithm**: Static shortest path calculation at route generation time
- **Behavior**: Vehicles follow pre-calculated shortest routes without dynamic updates
- **Characteristics**: Fastest route calculation, no simulation-time overhead
- **Use Case**: Baseline routing without real-time adaptation

**2. Realtime Strategy ("realtime")**

- **Algorithm**: Simulates GPS navigation apps like Waze/Google Maps with dynamic rerouting every 30 seconds
- **Initial Route**: Uses fastest path algorithm, falls back to shortest path if fastest fails
- **Behavior**: Vehicles adapt routes based on current traffic conditions using frequent updates
- **Implementation**: TraCI-based route updates prioritizing responsiveness to traffic changes
- **Characteristics**: High update frequency (30s) for maximum traffic responsiveness

**3. Fastest Path Strategy ("fastest")**

- **Algorithm**: Pure travel-time optimization with dynamic rerouting every 45 seconds
- **Initial Route**: Uses fastest path algorithm exclusively (optimizes for travel time over distance)
- **Behavior**: Vehicles consistently seek minimum travel time routes regardless of distance
- **Implementation**: Less frequent but more computation-intensive rerouting focused on time efficiency
- **Characteristics**: Moderate update frequency (45s) with consistent time-optimization focus

**4. Attractiveness Strategy ("attractiveness")**

- **Algorithm**: Multi-criteria routing considering both travel time and edge attractiveness
- **Behavior**: Vehicles prefer routes through higher-attractiveness areas
- **Implementation**: Custom routing algorithm incorporating attractiveness weights
- **Characteristics**: Simulates preference for main roads or commercial areas

**Strategy Mixing:**

- **Format**: "shortest 70 realtime 30" assigns 70% shortest, 30% realtime
- **Validation**: Percentages must sum to 100%
- **Assignment**: Each vehicle randomly assigned strategy based on percentages

#### 7.2.4 Origin-Destination Selection

**Attractiveness-Based Selection:**

- **Departure Edges**: Selected based on `depart_attractiveness` values (weighted random selection)
- **Arrival Edges**: Selected based on `arrive_attractiveness` values (weighted random selection)
- **Spatial Distribution**: Higher attractiveness values increase selection probability
- **Route Feasibility**: Ensures valid routes exist between selected origin-destination pairs

### 7.3 Vehicle Route Generation Validation

- **Step**: Verify route generation was successful
- **Function**: `verify_generate_vehicle_routes()` in `src/validate/validate_traffic.py`
- **Validation Checks**:
  - **Vehicle Count**: Confirm total vehicles matches `--num_vehicles` parameter
  - **Route Structure**: Verify all routes have valid origin and destination edges
  - **Type Distribution**: Check vehicle type percentages match `--vehicle_types` specification
  - **Departure Timing**: Validate departure times follow `--departure_pattern` distribution
  - **Strategy Assignment**: Confirm routing strategies match `--routing_strategy` percentages
  - **XML Validity**: Ensure output file is valid SUMO route XML format

### 7.4 Vehicle Route Generation Completion

- **Step**: Confirm successful route generation
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Generated vehicle routes successfully."
- **Output File**: `workspace/vehicles.rou.xml` with complete route definitions for all vehicles
- **Ready for Next Step**: Vehicle routes are prepared for SUMO configuration generation

## 8. SUMO Configuration Generation

### 8.1 Purpose and Process Overview

**Purpose of SUMO Configuration Generation:**

- **Simulation Setup**: Creates the central configuration file that links all generated components for SUMO execution
- **File Coordination**: Ensures proper referencing of network, routes, and zones files
- **Parameter Integration**: Incorporates simulation parameters (step length, end time, GUI settings) into configuration
- **Simulation Preparation**: Final step before dynamic simulation execution

### 8.2 SUMO Configuration Generation Process

- **Step**: Generate SUMO configuration file linking all simulation components
- **Function**: `generate_sumo_conf_file()` in `src/sumo_integration/sumo_utils.py`
- **Arguments Used**: `--step_length`, `--end_time`, `--gui`
- **Input Files**:
  - `workspace/grid.net.xml` (complete network with attractiveness)
  - `workspace/vehicles.rou.xml` (vehicle routes and types)
  - `workspace/zones.poly.xml` (zones for visualization)

### 8.3 Configuration File Creation

**SUMO Configuration File:**

- **Output**: `workspace/grid.sumocfg` with references to all simulation components
- **Content**:
  - Network file reference: `workspace/grid.net.xml`
  - Route file reference: `workspace/vehicles.rou.xml`
  - Additional files: `workspace/zones.poly.xml`
  - Simulation parameters: step length, end time, GUI settings
- **Format**: XML configuration following SUMO standards

### 8.4 SUMO Configuration Generation Completion

- **Step**: Confirm successful configuration file generation
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Generated SUMO configuration successfully."
- **Output File**: `workspace/grid.sumocfg` ready for simulation execution
- **Ready for Next Step**: All components prepared for dynamic simulation

## 9. Dynamic Simulation with Traffic Control

### 9.1 Purpose and Process Overview

**Purpose of Dynamic Simulation:**

- **Traffic Flow Execution**: Runs the actual vehicle simulation using all generated network and traffic components
- **Real-Time Control**: Applies dynamic traffic control algorithms to optimize signal timing during simulation
- **Performance Measurement**: Collects traffic metrics (travel times, completion rates, throughput) for analysis
- **Research Platform**: Enables controlled comparison of different traffic control methods under identical conditions

### 9.2 Dynamic Simulation Process

- **Step**: Execute SUMO simulation with real-time traffic control integration
- **Function**: `SumoController.run()` in `src/sumo_integration/sumo_controller.py`
- **Arguments Used**: `--traffic_control`, `--gui`, `--step_length`, `--end_time`, `--time_dependent`, `--start_time_hour`, `--routing_strategy`
- **Input Files**:
  - `workspace/grid.sumocfg` (SUMO configuration file)
  - `workspace/grid.net.xml` (complete network with attractiveness)
  - `workspace/vehicles.rou.xml` (vehicle routes and types)
  - `workspace/zones.poly.xml` (zones for visualization)

#### 9.2.1 TraCI Controller Initialization

**Controller Setup:**

- **Purpose**: Establishes Python-SUMO communication bridge for real-time control
- **Implementation**: `SumoController` class with per-step callback system
- **Features**:
  - Step-by-step simulation control
  - Real-time traffic light manipulation
  - Dynamic vehicle rerouting for realtime/fastest strategies
  - Traffic metrics collection throughout simulation

#### 9.2.2 Traffic Control Method Integration

**Conditional Object Initialization:**

- **Performance Optimization**: Only loads traffic control objects needed for selected method
- **Method-Specific Setup**: Different initialization paths based on `--traffic_control` argument

### 8.3 Traffic Control Methods

#### 9.3.1 Tree Method (Decentralized Bottleneck Prioritization Algorithm)

**Algorithm Overview:**

- **Research Foundation**: Decentralized traffic control strategy based on congestion tree identification and cost calculation
- **Core Principle**: Addresses conflicting traffic flows that compete on opposing cycle times during specific phases at traffic intersections
- **Methodology**: Identifies and prioritizes congestion bottlenecks based on their global network influence rather than local impact
- **Decision Making**: Each intersection makes decentralized decisions using tree-shaped congestion analysis
- **Optimization Goal**: Minimize overall network congestion by prioritizing bottlenecks with highest global cost

**Theoretical Foundation:**

**Multi-Stage Tree Method Process:**

**Stage I - Network Representation and Pre-calculations:**

- **Network Transformation**: Divides street segments into body links (main road segments between intersections) and head links (final approach lanes leading into intersections)
- **Body Links**: Represent continuous traffic flow approaching junctions, capturing traffic buildup over time and critical for identifying congestion patterns
- **Head Links**: Shorter segments where vehicles queue before entering intersections, crucial for determining green time allocation
- **Network Graph**: Transforms urban area into network where nodes represent intersections/lane mergers and links represent street segments

**Stage II - Link Properties Pre-calculation:**

- **Traffic Parameters**: Establishes number of lanes and maximum travel speed (vf) for each body link
- **Flow Capacity**: Calculates maximum flow Qmax using fundamental traffic law (q(t) = v(t) \* k(t)) and May's formula
- **May's Equation**: v(t)/vf^(1-m) = 1 - (k(t)/kj)^(l-1) where kj = 150 veh/km, m = 0.8, l = 2.8
- **Maximum Flow Speed**: Derives Vqmax representing speed at which flow is maximized for each link

**Stage III - Real-Time Congestion Assessment:**

- **Cycle-Based Analysis**: Monitors speed on each body link during every traffic light cycle
- **Congestion Criteria**: Link classified as congested if observed speed < Vqmax, flowing if speed ≥ Vqmax
- **Dynamic Evaluation**: Real-time analysis conducted continuously throughout simulation

**Stage IV - Congestion Tree Formation:**

- **Trunk Identification**: Body links with head links leading to intersection identified as tree trunks (roots)
- **Recursive Construction**: Congestion trees built by adding adjacent congested body links feeding into trunk, continuing until all feeding links are non-congested
- **Tree Membership**: Body links may belong to multiple trees but serve as trunk for only one tree
- **Branch Classification**: All body links in tree classified as branches regardless of role

**Stage V - Tree Cost Calculation:**

- **Delay-Based Costing**: Cost represents additional time required to cross road compared to maximum flow conditions
- **Cost Formula**: C(t) = dij _ (1/v(t) - 1/vqmax) _ (q(t)*N*T)/60
  - dij: link length (km)
  - q(t): current flow on link
  - v(t): current speed on link
  - vqmax: speed under maximum flow conditions
  - N: number of lanes
  - T: traffic light cycle time (minutes)
- **Branch Cost Distribution**: Link cost divided by number of trees containing that link
- **Tree Total Cost**: Sum of all constituent branch costs, measured in vehicle hours (VH)

**Stage VI - Phase Cost Assignment:**

- **Phase-Link Mapping**: Each phase facilitates movement along specific body and head links
- **Cost Distribution**: Congested body link costs evenly distributed among all trees incorporating the link
- **Trunk Assignment**: Tree costs assigned to their respective trunks
- **Weight Calculation**: Head links receive weights (0-1) based on traffic volume handled during last cycle, normalized against other head links on same body link
- **Phase Cost**: Sum of corresponding body link costs multiplied by head link weights

**Stage VII - Dynamic Phase Duration Calculation:**

- **Competitive Allocation**: Fixed cycle length and phase order with dynamic duration distribution
- **Cost-Based Adjustment**: Phase durations for next cycle determined by current cycle performance
- **Proportional Distribution**: Available duration divided among phases according to their respective costs
- **Balancing Mechanism**: Higher cost phases receive proportionally longer durations to address congestion

**Implementation Process:**

- **Network JSON Generation**: Converts SUMO network to JSON format for algorithm processing
- **Tree Structure Loading**: Builds network tree and run configuration using `load_tree()`
- **Graph Construction**: Creates algorithm Graph object with network topology
- **Cycle Time Calculation**: Determines optimal signal cycle timing based on network characteristics

**Runtime Behavior:**

- **Step Frequency**: Updates traffic light states every simulation step
- **Algorithm Execution**: Calls `graph.update_traffic_lights()` with current time and cycle parameters
- **Phase Translation**: Converts algorithm decisions to SUMO traffic light color strings based on cost calculations
- **Signal Application**: Pushes new traffic light states to SUMO via TraCI

**Key Advantages:**

- **Global Optimization**: Prioritizes traffic flows based on global network cost rather than local impact
- **Real-Time Adaptability**: Analytical simplicity enables swift real-time adjustments in each cycle
- **Decentralized Operation**: Each intersection operates independently while considering network-wide effects
- **Bottleneck Focus**: Accurately identifies root causes of traffic congestion and upstream impacts
- **Fixed Cycle Benefits**: Maintains predictable phase ordering to avoid driver confusion while optimizing durations

**Setup Requirements:**

- **Files**: Network JSON, tree configuration, graph structure
- **Objects**: `Network`, `Graph`, cycle time calculation
- **Validation**: Runtime verification of algorithm behavior at configured frequency

#### 9.3.2 Actuated Control (SUMO Built-in)

**Algorithm Overview:**

- **SUMO Native**: Uses SUMO's built-in actuated traffic control system
- **Gap-Based Logic**: Extends green phases when vehicles are detected, switches when gaps occur
- **Sensor Simulation**: Simulates inductive loop detectors at intersection approaches
- **Adaptive Timing**: Adjusts signal timing based on real-time traffic detection

**Implementation Process:**

- **Minimal Setup**: No additional algorithm objects required
- **Automatic Operation**: SUMO handles all signal logic internally
- **Configuration**: Uses signal timings and detector positions from network generation

**Runtime Behavior:**

- **Autonomous Control**: No per-step intervention required from TraCI controller
- **Gap Detection**: Automatically detects vehicle presence and absence
- **Phase Extension**: Extends green phases when vehicles present, minimum/maximum timing constraints apply

#### 9.3.3 Fixed Timing Control (Static)

**Algorithm Overview:**

- **Static Timing**: Uses predefined, fixed-duration signal phases throughout simulation
- **Predictable Operation**: No adaptation to traffic conditions, consistent timing patterns
- **Baseline Comparison**: Serves as control group for comparing adaptive methods

**Implementation Process:**

- **No Setup**: Uses static timing from traffic light file generation
- **Grid Configuration**: Consistent timing across all intersections for fair comparison

**Runtime Behavior:**

- **No Intervention**: No per-step control required
- **Fixed Cycles**: Signal phases follow predetermined durations regardless of traffic
- **Predictable Patterns**: Enables controlled experimental conditions

### 8.4 Dynamic Routing Integration

**Routing Strategy Execution:**

- **Static Strategies**: Shortest and attractiveness routes remain unchanged during simulation
- **Dynamic Strategies**: Realtime (30s) and fastest (45s) strategies trigger route updates via TraCI
- **Update Mechanism**: Controller tracks vehicle strategies and applies rerouting at specified intervals
- **Route Optimization**: Uses current edge travel times for dynamic route calculation

### 8.5 Traffic Metrics Collection

**Performance Measurement:**

- **Real-Time Tracking**: Collects traffic statistics throughout simulation
- **Key Metrics**:
  - **Vehicle Arrivals**: Count of vehicles reaching destinations
  - **Vehicle Departures**: Count of vehicles entering simulation
  - **Completion Rate**: Percentage of vehicles successfully completing trips
  - **Average Travel Time**: Mean time from departure to arrival
  - **Throughput**: Vehicle flow rates through network

**Experimental Output:**

- **Metrics Display**: Prints experiment metrics at simulation completion
- **Research Format**: Structured output suitable for comparative analysis
- **Statistical Analysis**: Enables comparison between traffic control methods

### 8.6 Dynamic Simulation Validation

**Runtime Verification:**

- **Algorithm Behavior**: Validates traffic control decisions during simulation
- **Performance Monitoring**: Tracks simulation health and progress
- **Error Handling**: Graceful handling of simulation errors and edge cases

### 8.7 Dynamic Simulation Completion

- **Step**: Confirm successful simulation execution
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Simulation completed successfully."
- **Output**: Complete traffic simulation with performance metrics
- **Research Results**: Traffic control method performance data ready for analysis

#### 9.4.1 Tree Method (`--traffic_control tree_method`)

- **Algorithm**: Tree Method decentralized traffic control with bottleneck detection
- **OSM Adaptation**: Modified to handle missing connections in real street networks using `.get()` method
- **Objects**: Requires Network, Graph, and cycle time calculation
- **Real-time**: Updates traffic light phases based on current traffic conditions
- **Validation**: `verify_tree_method_integration_setup()` and `verify_algorithm_runtime_behavior()`

#### 9.4.2 Actuated Control (`--traffic_control actuated`)

- **Algorithm**: SUMO's gap-based actuated signal control
- **Process**: Let SUMO handle traffic lights automatically based on vehicle detection
- **Setup**: No additional objects required
- **Baseline**: Serves as primary comparison baseline for Tree Method performance

#### 9.4.3 Fixed Control (`--traffic_control fixed`)

- **Algorithm**: Static timing from configuration files
- **OSM Mode**: Uses original OSM signal timing when available
- **Non-OSM Mode**: Uses generated static timing from `--traffic_light_strategy`
- **Setup**: No additional objects required
- **Research**: Traditional baseline for traffic control comparison

### 8.4 Simulation Output (Both Modes)

- **Configuration**: `workspace/grid.sumocfg` with all simulation parameters
- **Metrics**: Travel times, completion rates, throughput, vehicle arrivals/departures
- **Validation**: `verify_generate_sumo_conf_file()` ensures configuration integrity
- **Experimental**: Designed for statistical comparison of traffic control methods

## 10. Validation

### 10.1 Validation Framework Overview

- **Purpose**: Comprehensive runtime validation system for ensuring pipeline integrity and correctness
- **Location**: `src/validate/` directory with 4 core modules
- **Usage**: Inline validation at each pipeline step with custom error handling

#### 9.1.1 errors.py

- **ValidationError**: Custom exception class for validation failures

#### 9.1.2 validate_network.py

**Network Generation and Processing Validation Functions:**

- **verify_generate_grid_network()**: Validates synthetic grid generation with junction removal
- **verify_insert_split_edges()**: Validates edge splitting into body + head segments
- **verify_split_edges_with_flow_based_lanes()**: Validates comprehensive edge splitting and lane assignment
- **verify_extract_zones_from_junctions()**: Validates zone extraction from junction topology
- **verify_rebuild_network()**: Validates network compilation from separate XML files
- **verify_set_lane_counts()**: Validates lane assignment algorithms and distribution
- **verify_assign_edge_attractiveness()**: Validates attractiveness calculation methods
- **verify_generate_sumo_conf_file()**: Validates SUMO configuration file generation

#### 9.1.3 validate_traffic.py

**Traffic Generation Validation Functions:**

- **verify_generate_vehicle_routes()**: Validates vehicle route generation with connectivity and statistics checks

#### 9.1.4 validate_simulation.py

**Simulation Runtime Validation Functions:**

- **verify_tree_method_integration_setup()**: Validates Tree Method traffic control algorithm initialization
- **verify_algorithm_runtime_behavior()**: Validates algorithm behavior during simulation runtime

#### 9.1.5 validate_split_edges_with_lanes.py

**Edge Splitting and Lane Assignment Validation Functions:**

- **verify_split_edges_with_flow_based_lanes()**: Comprehensive validation of edge splitting and lane assignment

##### Validation Implementation Details:

**Primary Validation Checks:**

1. **Tail Lane Consistency Validation**:
   - **Purpose**: Ensure tail segments retain original edge lane counts
   - **Method**: Compare tail segment lane count against original edge definition
   - **Validation**: `actual_tail_lanes == original_edge_lanes`
   - **Error Example**: "Edge A0B0: tail lanes (1) != original lanes (2)"

2. **Head Lane Accuracy Validation**:
   - **Purpose**: Confirm head segments have correct total movement lanes
   - **Method**: Count total lanes used by all outgoing movements
   - **Validation**: `actual_head_lanes == total_movement_lanes`
   - **Calculation**: `total_movement_lanes = sum(lanes_per_movement for all movements)`
   - **Error Example**: "Edge A0B0: head lanes (3) != total movement lanes (4)"

3. **Movement Integrity Validation**:
   - **Purpose**: Ensure each head lane serves exactly one outgoing movement
   - **Method**: Parse connection file to verify one-to-one movement mapping
   - **Validation**: Each head lane has exactly one outgoing connection
   - **Error Example**: "Edge A0B0: head lane 2 has 3 connections (expected 1)"

4. **Connectivity Validation**:
   - **Purpose**: Verify complete tail-to-head connectivity
   - **Method**: Check all head lanes have incoming connections from tail segments
   - **Validation**: Every head lane receives traffic from at least one tail lane
   - **Error Example**: "Edge A0B0: head lanes {1,3} have no incoming connections from tail"

5. **Complete Flow Validation**:
   - **Purpose**: Ensure all tail lanes connect to head segments
   - **Method**: Verify every tail lane has outgoing connections to head segment
   - **Validation**: No tail lanes are orphaned or disconnected
   - **Error Example**: "Edge A0B0: tail lanes {0,2} have no outgoing connections"

**Structural Validation Checks:**

- **Network Completeness**: Validate all expected tail and head segments exist in network
- **Lane Index Consistency**: Ensure proper sequential lane numbering (0,1,2...)
- **Connection Bounds**: Verify all connections reference valid lane indices within bounds
- **Edge Mapping**: Confirm proper edge-to-movement-to-lane assignments

**Validation Process Flow:**

1. **Network Parsing**: Extract lane information from rebuilt `.net.xml` file
2. **Connection Analysis**: Parse movement data from `.con.xml` file  
3. **Movement Calculation**: Analyze total movement lanes per edge
4. **Cross-Validation**: Compare network structure against connection requirements
5. **Error Reporting**: Generate detailed error messages for any validation failures
6. **Success Confirmation**: Display validation success with edge count

**Integration and Timing:**

- **Execution Point**: After network rebuild (Step 4) when all segments exist
- **CLI Integration**: Seamlessly integrated into pipeline with proper error handling
- **Error Handling**: Uses ValidationError for consistent error reporting
- **Performance**: Validates all edges efficiently in single pass

**Example Validation Output:**

```
Starting comprehensive split edges validation...
✅ VALIDATION PASSED: 80 edges validated successfully
```

**Error Example Output:**

```
❌ VALIDATION FAILED: 3 errors found:
  - Edge A0B0: tail lanes (1) != original lanes (2)
  - Edge C2D2: head lanes (3) != total movement lanes (4)  
  - Edge E1E2: head lane 1 has 0 connections (expected 1)
```

#### 9.1.6 validate_arguments.py

**CLI Argument Validation Functions:**

- **validate_arguments()**: Validates command-line arguments for consistency and format correctness

## 11. Software Testing Framework

### 11.1 Testing Framework Overview

**Purpose of Professional Test Suite:**

- **Code Quality**: Ensures implementation correctness and maintainability
- **Regression Prevention**: Catches issues introduced by code changes
- **Development Support**: Provides automated validation for complex pipeline operations
- **Performance Monitoring**: Tracks system performance and prevents degradation
- **Documentation**: Serves as executable examples of system usage

**Framework Architecture:**

- **Location**: `tests/` directory in project root
- **Framework**: pytest with professional configuration and plugins
- **Coverage**: pytest-cov for comprehensive code coverage analysis
- **Structure**: Three-tier architecture (unit/integration/system tests)
- **Scope**: Complete SUMO Traffic Generator pipeline validation

### 11.2 Test Categories

#### 11.2.1 Unit Tests (`tests/unit/`)

**Purpose**: Test isolated components and functions

- **Scope**: Parsers, validators, utilities, configuration classes
- **Framework**: pytest with mocking for external dependencies
- **Coverage**: Individual functions and classes
- **Mock Dependencies**: External services (SUMO, file I/O)

#### 11.2.2 Integration Tests (`tests/integration/`)

**Purpose**: Test individual pipeline steps in isolation

- **Scope**: Network generation, lane assignment, traffic generation, simulation execution
- **Framework**: pytest with actual workspace directory
- **Coverage**: Individual pipeline steps and their file outputs
- **Validation**: File generation, format correctness, parameter handling

**Key Integration Tests:**

- **Network Generation**: Grid creation and OSM import validation
- **Lane Assignment**: Edge splitting and lane configuration testing
- **Traffic Generation**: Route and vehicle file creation
- **Pipeline Steps**: Each of the 9 pipeline steps tested individually

#### 11.2.3 System Tests (`tests/system/`)

**Purpose**: Test complete end-to-end scenarios

- **Scope**: Complete SUMO Traffic Generator scenarios from CLAUDE.md
- **Framework**: pytest with performance regression testing
- **Coverage**: Full scenarios, method comparisons, reproducibility

**Scenario Testing (14 verified scenarios):**

- **Basic Scenarios**: Grid dimensions 3x3, 5x5, 7x7 with various vehicle counts
- **Traffic Control Comparisons**: Tree Method vs Actuated vs Fixed timing
- **OSM Network Testing**: Real-world street network validation
- **Tree Method Samples**: Research benchmark validation
- **Performance Testing**: Travel time and completion rate analysis

### 11.3 Test Infrastructure

#### 11.3.1 Coverage Reporting

**Configuration**: `.coveragerc` in project root with organized data storage

```ini
[run]
source = src
data_file = tests/coverage/.coverage
parallel = true

[html]
directory = tests/coverage/htmlcov
show_contexts = true
```

**Coverage Organization:**

- **Data Storage**: `tests/coverage/.coverage` - SQLite coverage database
- **HTML Reports**: `tests/coverage/htmlcov/` - Browser-viewable coverage reports
- **XML Reports**: `tests/coverage/coverage.xml` - CI/CD integration format

#### 11.3.2 Golden Master Testing

**Performance Baselines**: JSON files with expected performance metrics

- **`tests/system/fixtures/golden_master_3x3.json`**: 3x3 grid baseline (48 edges, 33 junctions)
- **`tests/system/fixtures/golden_master_5x5.json`**: 5x5 grid baseline (152 edges, 100 junctions)

**Regression Detection**: Automatically detects performance degradation

#### 11.3.3 Test Utilities

**Helper Functions** (`tests/utils/`):

- **`test_helpers.py`**: Workspace management, XML parsing, metric extraction
- **`assertions.py`**: Custom assertion classes for file validation
- **`conftest.py`**: pytest configuration and shared fixtures

**Workspace Management:**

- **Consistent Directory Handling**: All tests use actual `workspace/` directory
- **File Validation**: Automated checking of generated SUMO files
- **Cleanup**: Automatic workspace cleanup between tests

### 11.4 Running Tests

#### 11.4.1 Complete Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html

# View coverage report
open tests/coverage/htmlcov/index.html
```

#### 11.4.2 Category-Specific Testing

```bash
# Unit tests only
pytest tests/unit/

# Integration tests (pipeline steps)
pytest tests/integration/

# System tests (full scenarios)
pytest tests/system/

# Performance tests
pytest tests/system/test_performance.py
```

#### 11.4.3 Test Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run system tests with detailed output
pytest tests/system/ -v

# Run specific scenario tests
pytest tests/system/test_scenarios.py::test_basic_3x3_grid -v
```

### 11.5 Test Scenarios

#### 11.5.1 Synthetic Grid Scenarios

**Basic Grid Tests:**

- **3x3 Grid**: 150 vehicles, basic functionality validation
- **5x5 Grid**: 500 vehicles, moderate complexity testing
- **7x7 Grid**: 1000 vehicles, high complexity validation

**Traffic Control Method Comparisons:**

- **Tree Method**: Dynamic decentralized traffic control
- **SUMO Actuated**: Gap-based signal control
- **Fixed Timing**: Static signal phases

#### 11.5.2 OSM Network Scenarios

**Real-World Testing:**

- **Manhattan East Village**: 500 vehicles on real street network
- **Complex Intersections**: Dead-end streets and irregular topology
- **Signal Preservation**: Maintains original OSM traffic light timing

#### 11.5.3 Tree Method Sample Validation

**Research Benchmark Testing:**

- **Sample Networks**: Pre-built complex urban networks (946 vehicles)
- **Method Comparison**: Tree Method vs baseline methods on identical conditions
- **Performance Validation**: Confirms Tree Method implementation correctness

### 11.6 Performance Testing

#### 11.6.1 Travel Time Analysis

**Metrics Tracked:**

- **Average Travel Time**: Mean vehicle travel time across simulation
- **Completion Rates**: Percentage of vehicles reaching destination
- **Throughput**: Vehicle arrivals and departures per time unit

#### 11.6.2 Regression Detection

**Golden Master Comparison:**

- **Baseline Validation**: Compares current performance against established baselines
- **Tolerance Ranges**: Allows for acceptable variation while detecting significant changes
- **Automatic Alerts**: Test failures when performance degrades beyond thresholds

### 11.7 Framework Integration

**Clear Separation of Concerns:**

- **`tests/`**: Software engineering quality assurance
  - Code correctness and reliability
  - Regression prevention
  - Development workflow support

- **`evaluation/`**: Research validation and performance analysis
  - Statistical benchmarks and comparisons
  - Publication-ready experimental results
  - Dataset management for reproducible research

Both frameworks complement each other but serve different purposes in maintaining a high-quality research codebase.

### 11.8 Getting Started

#### 11.8.1 Installation

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-mock

# Verify test environment
pytest --version
```

#### 11.8.2 First Test Run

```bash
# Run complete test suite
pytest tests/

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View results
open tests/coverage/htmlcov/index.html
```

#### 11.8.3 Development Workflow

1. **Write Tests First**: Follow test-driven development practices
2. **Run Tests Frequently**: Execute relevant tests during development
3. **Check Coverage**: Ensure new code has appropriate test coverage
4. **Validate Performance**: Run performance tests for changes affecting algorithms
5. **Pre-Commit Testing**: Run full test suite before code commits

### 11.9 Test Maintenance

#### 11.9.1 Adding New Tests

**Guidelines:**

- **Follow Directory Structure**: Place tests in appropriate category (unit/integration/system)
- **Use Naming Conventions**: Test files start with `test_`, test functions start with `test_`
- **Include Documentation**: Add docstrings explaining test purpose and validation
- **Update Baselines**: Modify golden master files when intentional changes occur

#### 11.9.2 Continuous Integration

**CI/CD Integration:**

- **Automated Testing**: All tests run automatically on code changes
- **Coverage Requirements**: Maintain minimum coverage thresholds
- **Performance Monitoring**: Track performance trends over time
- **Failure Notifications**: Immediate alerts for test failures or performance regression

## 12. Scripts

### 12.1 Development and Testing Scripts

- **Purpose**: Provide auxiliary tools and utilities for development, testing, and data management
- **Location**: `tools/scripts/` directory in project root
- **Usage**: Support development workflow and provide verified test data for users

#### 12.1.1 OSM Sample Data Download Script

- **Script**: `tools/scripts/download_osm_samples.py`
- **Function**: Download verified working OSM areas for testing and demonstration
- **Usage**: `python tools/scripts/download_osm_samples.py`

##### Sample Areas Provided:

- **Manhattan Upper West**: Grid pattern, 300/300 vehicle success rate (40.7800, -73.9850, 40.7900, -73.9750)
- **San Francisco Downtown**: Strong grid layout, 298/300 vehicle success rate (37.7850, -122.4100, 37.7950, -122.4000)
- **Washington DC Downtown**: Planned grid system, 300/300 vehicle success rate (38.8950, -77.0350, 38.9050, -77.0250)

### 12.2 Testing Infrastructure Scripts

#### 12.2.1 Test Execution Scripts

**Automated Test Running:**

- **Full Test Suite**: `pytest tests/` - Runs all test categories
- **Coverage Generation**: `pytest tests/ --cov=src --cov-report=html` - Generates coverage reports
- **Category Testing**: Individual test category execution (unit/integration/system)

#### 12.2.2 Test Utilities

**Test Helper Functions** (`tests/utils/`):

- **Workspace Management**: Consistent directory handling across all test types
- **XML Validation**: SUMO file format validation and metric extraction
- **Custom Assertions**: Specialized assertion classes for traffic simulation validation
- **Golden Master Management**: Performance baseline comparison and updates

#### 12.2.3 Coverage Organization

**Coverage Configuration** (`.coveragerc`):

- **Source Tracking**: Monitors `src/` directory for code coverage
- **Data Organization**: Stores coverage data in `tests/coverage/` directory
- **Report Generation**: HTML and XML format reports for development and CI/CD

## Project Architecture Updates

This specification reflects the current modular architecture:

### Core Modules
- **`src/orchestration/`**: High-level simulation coordination and traffic controller abstractions
- **`src/sumo_integration/`**: SUMO/TraCI interface layer and utilities  
- **`src/pipeline/`**: Pipeline pattern implementation with factory and step classes
- **`src/args/`**: Dedicated argument parsing module

### Quality Assurance Framework
- **`tests/`**: Professional software testing framework with three-tier architecture
  - **`tests/unit/`**: Component isolation testing with mocking
  - **`tests/integration/`**: Pipeline step testing with actual workspace
  - **`tests/system/`**: End-to-end scenario testing with performance baselines
  - **`tests/utils/`**: Test infrastructure and helper functions
  - **`tests/coverage/`**: Coverage data and HTML/XML reports

### Research and Evaluation Framework
- **`evaluation/benchmarks/`**: Statistical comparison studies and performance analysis
- **`evaluation/datasets/`**: Research datasets including OSM files and Tree Method networks

## 13. Experimental Evaluation Framework

### 13.1 Overview

The experimental evaluation framework provides comprehensive statistical validation of the Tree Method traffic control algorithm against established baselines. The framework implements rigorous experimental methodology with multiple replications, statistical analysis, and reproducible benchmarks.

#### 13.1.1 Purpose and Scope

**Primary Objectives**:
- **Validate Tree Method Claims**: Test published performance improvements (20-45% vs fixed timing, 10-25% vs actuated)
- **Statistical Rigor**: 20-run experiments with confidence intervals and significance testing
- **Baseline Comparison**: Systematic comparison against SUMO Actuated, Fixed timing, and Random proxy methods
- **Scalability Analysis**: Performance evaluation across multiple network sizes and traffic conditions

**Experimental Methodology**:
- **Controlled Variables**: Fixed random seeds for reproducible results
- **Statistical Power**: Minimum 20 runs per method for robust confidence intervals
- **Method Isolation**: Identical network and traffic conditions across all control methods
- **Performance Metrics**: Travel times, completion rates, throughput, vehicle arrivals/departures

#### 13.1.2 Framework Architecture

**Experimental Structure**:
```
evaluation/benchmarks/
├── decentralized-traffic-bottlenecks/    # Original Tree Method validation
│   ├── Experiment1-realistic-high-load/  # 1200 vehicles, realistic traffic
│   ├── Experiment2-rand-high-load/       # 1200 vehicles, random patterns  
│   ├── Experiment3-realistic-moderate-load/  # 600 vehicles, realistic traffic
│   └── Experiment4-rand-moderate-load/   # 600 vehicles, random patterns
└── synthetic-grids/                      # Large-scale grid validation
    ├── grids-5x5/                       # 324 scenarios, 19,440 runs
    ├── grids-7x7/                       # 324 scenarios, 19,440 runs
    └── grids-9x9/                       # 324 scenarios, 19,440 runs
```

### 13.2 Decentralized Traffic Bottlenecks Experiments

#### 13.2.1 Experiment Design

**Four Major Experiments**:
1. **Experiment 1**: Realistic High Load (1200 vehicles, 2-hour simulation)
2. **Experiment 2**: Random High Load (1200 vehicles, random O-D patterns)
3. **Experiment 3**: Realistic Moderate Load (600 vehicles, 2-hour simulation)  
4. **Experiment 4**: Random Moderate Load (600 vehicles, random O-D patterns)

**Statistical Framework**:
- **Replications**: 20 runs per method (80 total per experiment)
- **Random Seeds**: 1-20 for reproducible statistical analysis
- **Traffic Control Methods**: Tree Method, SUMO Actuated, Fixed timing, Random proxy
- **Total Simulations**: 320 runs across all experiments

#### 13.2.2 Execution Methodology

**Automated Execution**:
```bash
# Run complete experiment (80 simulations)
cd evaluation/benchmarks/decentralized-traffic-bottlenecks/Experiment1-realistic-high-load
./run_experiment.sh

# Statistical analysis with confidence intervals
python analyze_results.py
```

**Performance Tracking**:
- **Real-time Progress**: Live tracking of completed runs and estimated time remaining
- **Error Handling**: Automatic retry mechanisms and failure logging
- **Resource Management**: Virtual environment activation and dependency verification

#### 13.2.3 Expected Outcomes

**Research Validation Targets**:
- **Tree Method vs Fixed**: 20-45% improvement in average travel times
- **Tree Method vs Actuated**: 10-25% improvement in average travel times
- **Completion Rates**: Higher vehicle completion rates under congestion
- **Statistical Significance**: p < 0.05 for performance differences

**Performance Metrics**:
- **Average Travel Time**: Primary outcome measure
- **Completion Rate**: Percentage of vehicles reaching destinations
- **Throughput**: Vehicles completed per unit time
- **Queue Statistics**: Maximum and average queue lengths

### 13.3 Synthetic Grid Experiments

#### 13.3.1 Large-Scale Validation Framework

**Grid Size Progression**:
- **5×5 Grid**: 25 intersections, moderate complexity
- **7×7 Grid**: 49 intersections, high complexity  
- **9×9 Grid**: 81 intersections, maximum complexity

**Experimental Matrix per Grid**:
- **Vehicle Count Levels**: 3 different traffic volumes
- **Vehicle Types**: 2 distribution patterns (passenger/commercial/public)
- **Routing Strategies**: 3 approaches (shortest/realtime/fastest)
- **Departure Patterns**: 3 temporal distributions
- **Simulation Durations**: 2 time horizons
- **Junction Removal**: 3 levels (network resilience testing)
- **Total Scenarios**: 324 unique parameter combinations per grid

#### 13.3.2 Statistical Scope

**Massive Scale Validation**:
- **Scenarios per Grid**: 324 unique parameter combinations
- **Methods per Scenario**: 3 traffic control approaches
- **Replications per Method**: 20 runs with different seeds
- **Total Runs per Grid**: 19,440 individual simulations
- **Grand Total**: 58,320 simulations across all grid sizes

**Centralized Configuration**:
```json
// experiment_config.json
{
  "grids": {
    "5x5": {"vehicle_counts": [600, 900, 1200]},
    "7x7": {"vehicle_counts": [1000, 1500, 2000]}, 
    "9x9": {"vehicle_counts": [1500, 2250, 3000]}
  },
  "shared_parameters": {
    "vehicle_types": ["passenger 60 commercial 30 public 10", "passenger 70 commercial 20 public 10"],
    "routing_strategies": ["shortest 100", "realtime 60 fastest 40", "shortest 50 realtime 50"],
    "departure_patterns": ["uniform", "six_periods", "rush_hours:7-9:40,17-19:40,rest:20"]
  }
}
```

#### 13.3.3 Execution Framework

**Hierarchical Execution**:
```bash
# Master execution across all grids
./run_all_experiments.sh

# Grid-specific execution  
cd grids-5x5 && ./run_all_runs.sh

# Statistical analysis and aggregation
python analyze_all_experiments.py
```

**Automated Analysis Pipeline**:
- **Grid-Level Analysis**: Performance statistics within each grid size
- **Cross-Grid Scalability**: Performance trends across network complexity
- **Method Comparison**: Statistical significance testing between control methods
- **Publication-Ready Results**: Confidence intervals, effect sizes, significance tests

### 13.4 Method Comparison Studies

#### 13.4.1 Traffic Control Methods

**Tree Method (Primary)**:
- **Algorithm**: Decentralized traffic bottleneck optimization
- **Implementation**: `src/orchestration/traffic_controller.py:TreeMethodController`
- **Characteristics**: Dynamic signal optimization based on real-time traffic bottlenecks
- **Expected Performance**: Superior under moderate to high congestion

**SUMO Actuated (Primary Baseline)**:
- **Algorithm**: Gap-based actuated signal control
- **Implementation**: Native SUMO actuated traffic light programs
- **Characteristics**: Extends green phases based on vehicle detection
- **Baseline Status**: Industry standard for adaptive signal control

**Fixed Timing (Static Baseline)**:
- **Algorithm**: Pre-timed signal plans with fixed phase durations
- **Implementation**: `src/orchestration/traffic_controller.py:FixedController`
- **Characteristics**: Deterministic 90-second cycles with consistent phase timing
- **Baseline Status**: Traditional traffic control method

**Random Proxy (Control)**:
- **Algorithm**: Mixed routing strategies simulating random driver behavior
- **Implementation**: Random mixture of shortest/realtime/fastest routing
- **Characteristics**: Tests impact of routing vs signal control
- **Control Status**: Isolates signal control effects from routing effects

#### 13.4.2 Performance Evaluation

**Statistical Analysis Framework**:
- **Primary Metrics**: Mean travel time, completion rate, throughput
- **Statistical Tests**: Two-sample t-tests with Bonferroni correction
- **Effect Size**: Cohen's d for practical significance assessment
- **Confidence Intervals**: 95% CI for all performance differences

**Reporting Standards**:
- **Significance Threshold**: p < 0.05 for statistical significance
- **Effect Size Interpretation**: Small (0.2), Medium (0.5), Large (0.8) effects
- **Practical Significance**: Minimum 5% improvement for practical relevance
- **Reproducibility**: All results include random seeds and configuration parameters

### 13.5 Validation and Baselines

#### 13.5.1 Research Validation

**Tree Method Original Claims**:
- **Source**: "Enhancing Traffic Flow Efficiency through an Innovative Decentralized Traffic Control"
- **Performance Claims**: 20-45% improvement vs fixed timing, 10-25% vs actuated control
- **Validation Approach**: Replication with identical experimental conditions
- **Success Criteria**: Performance within claimed ranges with statistical significance

**Baseline Establishment**:
- **SUMO Actuated**: Industry-standard adaptive control baseline
- **Fixed Timing**: Traditional pre-timed control baseline  
- **Random Control**: Methodological control for routing effects
- **Performance Regression**: Historical performance tracking for regression detection

#### 13.5.2 Quality Assurance

**Experimental Controls**:
- **Seed Management**: Controlled randomization for reproducible results
- **Parameter Validation**: Automatic validation of experimental parameters
- **Error Detection**: Comprehensive error logging and automatic retry mechanisms
- **Resource Monitoring**: Memory and CPU usage tracking during large experiments

**Result Validation**:
- **Sanity Checks**: Automatic detection of unrealistic results
- **Outlier Detection**: Statistical outlier identification and investigation
- **Consistency Verification**: Cross-method consistency checks
- **Manual Review**: Expert review of statistical results before publication

### 13.6 Experimental Infrastructure

#### 13.6.1 Automation Framework

**Experiment Orchestration**:
- **Master Scripts**: Top-level experiment coordination and progress tracking
- **Grid-Specific Scripts**: Parameterized execution for each grid configuration
- **Individual Experiments**: Single-run execution with error handling and logging
- **Analysis Pipeline**: Automated statistical analysis with publication-ready outputs

**Resource Management**:
- **Virtual Environment**: Automatic dependency management and environment activation
- **Parallel Execution**: Multi-core utilization for independent experiment runs
- **Storage Management**: Organized result storage with automatic cleanup
- **Progress Monitoring**: Real-time execution status and time estimation

#### 13.6.2 Analysis and Reporting

**Statistical Analysis Tools**:
- **Descriptive Statistics**: Mean, standard deviation, confidence intervals
- **Inferential Statistics**: t-tests, ANOVA, effect size calculation
- **Visualization**: Performance charts, confidence interval plots, comparison matrices
- **Export Formats**: CSV data, publication-ready tables, statistical summaries

**Report Generation**:
- **Executive Summary**: High-level performance comparison and conclusions
- **Detailed Analysis**: Method-by-method performance breakdown with statistics
- **Scalability Analysis**: Performance trends across network sizes and traffic levels
- **Research Validation**: Comparison with published Tree Method claims

#### 13.6.3 Usage Examples

**Running Complete Evaluation**:
```bash
# Activate environment
source .venv/bin/activate

# Run all decentralized traffic bottleneck experiments (320 runs)
cd evaluation/benchmarks/decentralized-traffic-bottlenecks
for experiment in Experiment*; do
    cd $experiment && ./run_experiment.sh && cd ..
done

# Run synthetic grid experiments (58,320 runs)
cd evaluation/benchmarks/synthetic-grids
./run_all_experiments.sh

# Generate comprehensive analysis report
python generate_evaluation_report.py
```

**Individual Experiment Execution**:
```bash
# Single experiment with statistical analysis
cd evaluation/benchmarks/decentralized-traffic-bottlenecks/Experiment1-realistic-high-load
./run_experiment.sh
python analyze_results.py

# Expected output: Statistical summary with confidence intervals
# Tree Method: 145.2 ± 8.3s (95% CI)
# Actuated: 162.7 ± 9.1s (95% CI) 
# Fixed: 198.4 ± 12.2s (95% CI)
# Statistical significance: p < 0.001
```

**Custom Experiment Configuration**:
```bash
# Quick validation run (reduced replications)
cd evaluation/benchmarks/synthetic-grids/grids-5x5
./run_all_runs.sh --max-runs 5  # 5 runs instead of 20

# Specific method testing
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 --num_vehicles 800 --end-time 3600 \
  --traffic_control tree_method --seed 42
```

This experimental framework represents the most comprehensive traffic control evaluation system available, providing statistically rigorous validation of the Tree Method algorithm across multiple network scales and traffic conditions.

## Advanced Features

### Custom Edge Lane Definition System

#### Overview
The system supports manual specification of lane configurations for specific edges in synthetic grid networks, overriding automatic lane assignment algorithms. This is implemented as a post-processing step after the standard edge splitting with flow-based lane assignment.

#### Feature Specification

##### CLI Arguments
- `--custom_lanes`: Direct specification of custom lane configurations
  - Format: `"EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=..."`
  - Supports flexible syntax: tail-only, head-only, full specification, dead-end creation
  - Multiple edge configurations separated by semicolons

- `--custom_lanes_file`: File-based configuration for complex scenarios
  - Same syntax as CLI argument, one configuration per line
  - Supports comments (lines starting with #) and empty lines
  - UTF-8 encoding required
  - Mutually exclusive with `--custom_lanes`

##### Supported Configuration Types
1. **Full Specification**: `A1B1=tail:2,head:B1B0:1,B1C1:2`
   - Custom tail lanes + explicit head movement assignments
   - Complete edge reconfiguration with bidirectional impact

2. **Tail-Only Customization**: `A1B1=tail:2`
   - Custom tail lanes, preserve all existing head movements
   - Minimal impact scope, only affects upstream junction

3. **Head Movements Only**: `A1B1=head:B1B0:1,B1C1:2`
   - Automatic tail lanes, custom movement assignments
   - Only affects downstream junction

4. **Dead-End Creation**: `A1B1=tail:2,head:`
   - Custom tail lanes, remove ALL head movements
   - Creates dead-end street (vehicles can enter but not exit)

##### Technical Implementation
- **Integration Point**: Step 3.5 in pipeline (after edge splitting, before traffic generation)
- **Shared Code Architecture**: Reuses existing spatial logic functions from `split_edges_with_lanes.py`
- **Complete Deletion/Regeneration Strategy**: Avoids partial update conflicts
- **Bidirectional Impact Management**: Updates both upstream and downstream junctions
- **Traffic Light System Regeneration**: Complete recalculation of state strings and linkIndex values

##### Constraints and Limitations
- **Synthetic Networks Only**: Not supported with OSM files (`--osm_file`)
- **Grid Network Requirement**: Edge IDs must follow pattern `A1B1`, `B2C2`, etc.
- **Lane Count Bounds**: Tail and movement lane counts must be 1-3
- **Movement Selection**: Only user-specified movements exist, others are deleted
- **Precedence**: Overrides `--lane_count` for specified edges

##### Validation Rules
- Edge ID format validation (must match grid pattern)
- Lane count bounds checking (1-3 range)
- Movement destination validation
- File existence and readability (for file-based config)
- Mutual exclusivity between CLI and file options
- Cross-argument compatibility checking

##### Error Handling
- Immediate exit on validation errors with descriptive messages
- Line-specific error reporting for file-based configurations
- Network validation using SUMO's built-in tools
- No automatic error recovery or rollback mechanisms

##### Use Cases
- **Research Applications**: Test specific lane configurations for algorithm validation
- **Scenario Testing**: Create controlled traffic conditions for Tree Method evaluation
- **Construction Simulation**: Block or restrict specific routes
- **Intersection Modeling**: Match real-world intersection configurations
- **Edge Case Testing**: Validate system behavior under unusual conditions

#### Integration Notes
This specification should be added to the "Advanced Features" section of SPECIFICATION.md, maintaining consistency with existing documentation style and cross-referencing related features like `--lane_count` and traffic control methods.

### Development Tools
- **`tools/scripts/`**: Development utilities and data management scripts
- **`workspace/`**: Generated simulation files (temporary, gitignored)

### Configuration Files
- **`.coveragerc`**: Coverage configuration for pytest-cov integration
- **`.gitignore`**: Excludes temporary files, coverage data, and workspace artifacts
- **`CLAUDE.md`**: Development instructions for Claude Code AI assistant
- **`SPECIFICATION.md`**: This comprehensive technical specification
