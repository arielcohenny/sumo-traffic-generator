# Initialization and Setup

## Data Directory Cleanup

- **Step**: Clean and prepare output directory
- **Function**: Directory cleanup via pipeline orchestration
- **Process**:
  - Remove existing `workspace/` directory if it exists (`shutil.rmtree(CONFIG.output_dir)`)
  - Create fresh `workspace/` directory (`CONFIG.output_dir.mkdir(exist_ok=True)`)
- **Purpose**: Ensures clean state for each simulation run

## Command-Line Argument Parsing

- **Step**: Parse and validate all CLI arguments
- **Function**: `src/args/parser.py` via CLI orchestration
- **Process**: Parse 20 available arguments with defaults and validation
- **Available Arguments**: See Available Arguments section below

## Seed Initialization

- **Step**: Initialize random seed for reproducible simulations
- **Function**: Seed generation via pipeline orchestration
- **Process**:
  - Use provided `--seed` value if specified
  - Generate random seed if not provided: `random.randint(0, 2**32 - 1)`
  - Print seed value for reproduction: `print(f"Using seed: {seed}")`

## Available Arguments

### `--grid_dimension` (float, default: 5)

Defines the grid's number of rows and columns for synthetic network generation. Not applicable when `--osm_file` is provided.

### `--block_size_m` (int, default: 200)

Sets block size in meters for grid network generation. Not applicable when `--osm_file` is provided.

### `--junctions_to_remove` (str, default: "0")

Number of junctions to remove or comma-separated list of specific junction IDs (e.g., "5" or "A0,B1,C2"). Not applicable when `--osm_file` is provided.

### `--lane_count` (str, default: "realistic")

Sets the lane count. 3 algorithms are available:

- `realistic`: Zone-based demand calculation
- `random`: Randomized within bounds (1-3 lanes)
- Integer value: Fixed count for all edges

### `--num_vehicles` (int, default: 300)

Total vehicles to generate.

### `--seed` (int, optional)

Controls randomization. If not provided, random seed is generated.

### `--step-length` (float, default: 1.0)

Simulation step length in seconds for TraCI control loop.

### `--end-time` (int, default: 7200)

Simulation duration in seconds.

### `--attractiveness` (str, default: "land_use")

Sets the departure and arrival attractiveness of each edge. Five methods available:

- `land_use`: Zone-based calculation (default)
- `poisson`: Random distribution
- `gravity`: Distance-based model
- `iac`: Intersection accessibility calculation
- `hybrid`: Combined approach

### `--time_dependent` (flag)

Applies 4-phase variations to synthetic zone attractiveness.

### `--start_time_hour` (float, default: 0.0)

Real-world hour when simulation starts (0-24) for temporal attractiveness. Used with `--time_dependent` for phase calculation.

### `--departure_pattern` (str, default: "uniform")

Vehicle departure timing. Four patterns available:

- `six_periods`: Research-based daily structure
- `uniform`: Even distribution
- `rush_hours:7-9:40,17-19:30,rest:10`: Custom rush hour definition
- `hourly:7:25,8:35,rest:5`: Granular hourly control

### `--routing_strategy` (str, default: "shortest 100")

Vehicle routing behavior. Four strategies with percentage mixing:

- `shortest`: Static shortest path
- `realtime`: 30-second dynamic rerouting
- `fastest`: 45-second fastest path rerouting
- `attractiveness`: Multi-criteria routing

### `--vehicle_types` (str, default: "passenger 60 commercial 30 public 10")

Vehicle type distribution. Three types with percentage assignment:

- `passenger`: Cars (5.0m length, 13.9 m/s max speed)
- `commercial`: Trucks (12.0m length, 10.0 m/s max speed)
- `public`: Buses (10.0m length, 11.1 m/s max speed)

### `--traffic_light_strategy` (str, default: "opposites")

Applied strategies for traffic lights. Two strategies available. Not applicable when `--osm_file` is provided.

- `opposites`: Opposing directions signal together
- `incoming`: Each edge gets separate phase

### `--traffic_control` (str, default: "tree_method")

Dynamic signal control. Four methods available:

- `tree_method`: Tree Method (Decentralized Bottleneck Prioritization Algorithm)
- `atlcs`: ATLCS (Adaptive Traffic Light Control System with Tree Method coordination)
- `actuated`: SUMO gap-based control
- `fixed`: Static timing from configuration

### `--tree-method-interval` (int, default: 90)

Tree Method calculation interval in seconds. Controls how often the Tree Method algorithm runs its optimization calculations.

**Performance Configuration:**

- **Lower values (30-60s)**: More responsive traffic control, higher CPU usage
- **Higher values (120-300s)**: More efficient computation, less responsive control
- **Default (90s)**: Balanced efficiency and responsiveness

**Valid Range:** 30-300 seconds

**Independence:** Tree Method timing is completely independent of traffic light cycle timing.

**Technical Implementation:** Overrides `TREE_METHOD_ITERATION_INTERVAL_SEC` constant in `src/config.py`.

**Examples:**

```bash
# Responsive control (every 60 seconds)
python -m src.cli --traffic_control tree_method --tree-method-interval 60

# Efficient control (every 2 minutes)
python -m src.cli --traffic_control tree_method --tree-method-interval 120

# High-performance scenarios (every 3 minutes)
python -m src.cli --traffic_control tree_method --tree-method-interval 180
```

### `--bottleneck-detection-interval` (int, default: 60)

Enhanced bottleneck detection interval in seconds for ATLCS. Controls how often the ATLCS enhanced bottleneck detector runs its analysis.

**Enhanced Bottleneck Detection Features:**

- **Advanced Metrics**: Uses density, speed, queue length, and waiting time (vs Tree Method's speed-only)
- **Predictive Analysis**: Identifies bottlenecks before they fully form
- **Multi-Criteria Assessment**: Combines multiple traffic indicators for robust detection
- **Real-Time Responsiveness**: More frequent updates than Tree Method's strategic intervals

**Performance Configuration:**

- **Lower values (30-45s)**: More responsive bottleneck detection, higher CPU usage
- **Higher values (90-120s)**: More efficient computation, less responsive detection
- **Default (60s)**: Balanced detection frequency and computational efficiency

**Valid Range:** 30-120 seconds

**Integration:** Works alongside Tree Method's 90-second strategic calculations to provide tactical bottleneck prevention.

### `--atlcs-interval` (int, default: 5)

ATLCS dynamic pricing update interval in seconds for ATLCS. Controls how often the ATLCS pricing engine calculates congestion-based pricing updates.

**ATLCS Dynamic Pricing Features:**

- **Congestion-Based Pricing**: Higher congestion severity receives higher priority scores
- **Signal Priority Calculation**: Converts pricing data to traffic light extension recommendations
- **Real-Time Adaptation**: Rapid response to changing traffic conditions
- **Bottleneck Prevention**: Extends green phases dynamically to prevent jam formation

**Performance Configuration:**

- **Lower values (1-3s)**: Maximum responsiveness, highest CPU usage
- **Higher values (10-15s)**: More efficient computation, reduced responsiveness
- **Default (5s)**: Optimal balance for real-time traffic light control

**Valid Range:** 1-15 seconds

**Technical Implementation:** Updates shared phase durations that Tree Method can access, enabling coordinated traffic control.

**Examples:**

```bash
# Responsive ATLCS configuration
python -m src.cli --traffic_control atlcs --bottleneck-detection-interval 45 --atlcs-interval 3

# Efficient ATLCS configuration
python -m src.cli --traffic_control atlcs --bottleneck-detection-interval 90 --atlcs-interval 10

# Balanced ATLCS with Tree Method coordination
python -m src.cli --traffic_control atlcs --tree-method-interval 90 --bottleneck-detection-interval 60 --atlcs-interval 5
```

### `--gui` (flag)

Launch SUMO GUI.

### `--osm_file` (str, optional)

Path to OSM file that replaces synthetic grid generation.

### `--land_use_block_size_m` (float, default: 25.0)

Zone cell size in meters for both OSM (intelligent zones) and non-OSM (traditional zones) mode.

**Default**: 25.0m for both network types (following research paper methodology from "A Simulation Model for Intra-Urban Movements")

**Purpose**: Controls the resolution of land use zone generation. Creates fine-grained cells for detailed spatial analysis independent of street block size.

### `--tree_method_sample` (str, optional)

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

### `--custom_lanes` (str, optional)

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

### `--custom_lanes_file` (str, optional)

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

## Argument Validation

- **Step**: Validate all CLI arguments before processing
- **Function**: Argument validation (to be implemented)
- **Process**: Comprehensive validation of all input parameters
- **Validations Required**:

### Routing Strategy Validation

- **Current**: Implemented in `parse_routing_strategy()` in `src/traffic/routing.py`
- **Checks**:
  - Format validation: Must be pairs of strategy name + percentage
  - Valid strategies: {"shortest", "realtime", "fastest", "attractiveness"}
  - Percentage range: 0-100 for each strategy
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

### Vehicle Types Validation

- **Current**: Implemented in `parse_vehicle_types()` in `src/traffic/vehicle_types.py`
- **Checks**:
  - Format validation: Must be pairs of vehicle type + percentage
  - Valid types: {"passenger", "commercial", "public"}
  - Percentage range: 0-100 for each type
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

### Departure Pattern Validation

- **Current**: Not implemented
- **Needed Checks**:
  - Valid pattern names: {"six_periods", "uniform"}
  - Format validation for "rush_hours:7-9:40,17-19:30,rest:10"
  - Format validation for "hourly:7:25,8:35,rest:5"
  - Hour range validation (0-24)
  - Percentage validation for custom patterns
  - Time range validation (start < end hours)

### OSM File Validation

- **Current**: Not implemented
- **Needed Checks**:
  - File existence verification
  - File format validation (XML structure)
  - OSM-specific validation (nodes, ways, bounds elements present)
  - File readability and permissions

### Numeric Range Validations

- **Current**: Basic type checking only
- **Needed Checks**:
  - `--grid_dimension`: > 0, reasonable upper bound (e.g., ≤ 20)
  - `--block_size_m`: > 0, reasonable range (50-1000m)
  - `--num_vehicles`: > 0, reasonable upper bound
  - `--step-length`: > 0, reasonable range (0.1-10.0 seconds)
  - `--end-time`: > 0
  - `--start_time_hour`: 0-24 range
  - `--land_use_block_size_m`: > 0, reasonable range (10-100m). **Default**: 25.0m for both network types (research paper methodology)

### Junctions to Remove Validation

- **Current**: String input only
- **Needed Checks**:
  - Format validation for comma-separated junction IDs
  - Numeric validation when integer count provided
  - Range validation (can't exceed grid capacity)
  - Junction ID format validation for specific removal

### Lane Count Validation

- **Current**: String acceptance only
- **Needed Checks**:
  - Valid algorithm names: {"realistic", "random"}
  - Integer validation for fixed count mode
  - Range validation for fixed counts (1-5 lanes)

### Cross-Argument Validation

- **Current**: Not implemented
- **Needed Checks**:
  - OSM file vs grid parameters (mutually exclusive usage)
  - Time-dependent features requiring appropriate end-time duration
  - Grid dimension vs junctions to remove capacity limits
  - Traffic light strategy compatibility with network type

### Choice Validations

- **Current**: Implemented via argparse choices
- **Existing Checks**:

  - `--attractiveness`: {"poisson", "land_use", "gravity", "iac", "hybrid"}
  - `--traffic_light_strategy`: {"opposites", "incoming"}
  - `--traffic_control`: {"tree_method", "actuated", "fixed"}

- **Error Handling**: Use existing `ValidationError` class for consistent error reporting
