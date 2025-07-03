# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based SUMO traffic generator that creates dynamic traffic simulations with intelligent signal control. It generates orthogonal grid networks, extracts zones, applies configurable lane assignments, and uses Nimrod's decentralized traffic control algorithm for dynamic signal optimization.

## Common Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Basic run
env PYTHONUNBUFFERED=1 python -m src.cli

# With GUI and custom parameters
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 7 --num_vehicles 500 --gui

# Full parameter example
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --junctions_to_remove 0 \
  --fixed_lane_count 2 \
  --num_vehicles 300 \
  --seed 42 \
  --step-length 1.0 \
  --end-time 3600 \
  --gui
```

### Testing
```bash
# Run tests (when implemented)
pytest

# Run validation functions (currently commented out in CLI)
# The validation runs automatically as part of the pipeline
```

### Code Quality (to be implemented)
```bash
# Linting and formatting (not yet configured)
# Recommended: black, flake8/ruff, mypy
```

## Architecture Overview

### Pipeline Architecture
The application follows a sequential 8-step pipeline:

1. **Network Generation** (`src/network/generate_grid.py`): Creates orthogonal grid using SUMO's netgenerate
2. **Edge Splitting** (`src/network/split_edges.py`): Splits edges for enhanced network complexity
3. **Zone Extraction** (`src/network/zones.py`): Extracts polygonal zones from junctions (currently disabled)
4. **Lane Configuration** (`src/network/lane_counts.py`): Applies configurable lane assignments
5. **Edge Attractiveness** (`src/network/edge_attrs.py`): Computes departure/arrival weights using Poisson distribution
6. **Traffic Light Injection** (`src/network/traffic_lights.py`): Adds default four-phase signal plans
7. **Route Generation** (`src/traffic/`): Generates vehicle routes using shortest-path computation
8. **Dynamic Simulation** (`src/sim/sumo_controller.py`): Runs SUMO with TraCI integration and Nimrod's algorithm

### Key Modules

**Core Orchestration**:
- `src/cli.py`: Main entry point and pipeline orchestration
- `src/config.py`: Central configuration using dataclasses

**Network Generation**:
- `src/network/`: All network manipulation and generation logic
- Uses SUMO's netgenerate, netconvert tools
- Validates network structure at each step

**Traffic Generation**:
- `src/traffic/builder.py`: Main traffic generation orchestrator
- `src/traffic/routing.py`: Shortest path routing using Dijkstra
- `src/traffic/xml_writer.py`: SUMO XML file generation

**Simulation Control**:
- `src/sim/sumo_controller.py`: TraCI wrapper with per-step callback API
- `src/sim/sumo_utils.py`: SUMO utility functions

**Traffic Control**:
- `src/traffic_control/decentralized_traffic_bottlenecks/`: Nimrod's Tree Method implementation
- `src/traffic_control/decentralized_traffic_bottlenecks/integration.py`: Bridges simulator with algorithm

**Validation**:
- `src/validate/`: Runtime validation functions for each pipeline step
- Uses custom validation errors and comprehensive checking

### Configuration System

Central configuration in `src/config.py` using dataclasses:
- `GridConfig`: Network generation parameters
- `TrafficConfig`: Vehicle and route parameters  
- `SimulationConfig`: SUMO simulation parameters
- `AlgorithmConfig`: Nimrod's algorithm parameters

### Generated Files Structure

All generated files are placed in `data/` directory:
- `grid.net.xml`: SUMO network file
- `grid.nod.xml`, `grid.edg.xml`, `grid.con.xml`, `grid.tll.xml`: Network components
- `vehicles.rou.xml`: Vehicle routes
- `zones.poly.xml`: Zone polygons (when enabled)
- `grid.sumocfg`: SUMO configuration file

## Development Notes

### Dependencies
- **SUMO**: Requires SUMO installation with netgenerate, netconvert, sumo, sumo-gui
- **Python Libraries**: numpy, shapely, geopandas, sumolib, traci, xmltodict, alive-progress
- **Testing**: pytest included but no tests currently implemented

### Current Implementation Status
- Several pipeline steps have commented-out code or early exit points
- Most validation functions are disabled in the main CLI
- Zone extraction (Step 2) is currently disabled
- No unit tests despite pytest dependency

### Key Patterns
- **Validation-First**: Each pipeline step has corresponding validation functions
- **Strategy Pattern**: Used in traffic generation (routing, sampling strategies)
- **Adapter Pattern**: TraCI integration through SumoController class
- **Pipeline Pattern**: Sequential processing with clear step boundaries

### External Tool Integration
- Uses SUMO's `randomTrips.py` for vehicle generation
- Integrates with TraCI for real-time simulation control
- Leverages SUMO's netgenerate and netconvert for network creation

### Development Workflow
1. Modify configuration in `src/config.py`
2. Run pipeline with `python -m src.cli`
3. Check generated files in `data/` directory
4. Use `--gui` flag for visual debugging
5. Validate results using the validation functions (when enabled)

## Detailed Module Documentation

### Network Generation (`src/network/`)

#### `generate_grid.py`
**Purpose**: Creates orthogonal grid networks using SUMO's netgenerate tool and manages junction removal.

**Key Functions**:
- `generate_full_grid_network(dimension, block_size_m, fixed_lane_count)`: Creates complete grid using netgenerate
- `pick_random_junction_ids(seed, num_junctions_to_remove, dimension)`: Randomly selects junctions for removal
- `wipe_crossing_from_*()`: Suite of functions to remove junctions from all network files (.nod, .edg, .con, .tll)
- `generate_grid_network()`: Main orchestrator that generates grid and optionally removes junctions

**Key Algorithms**:
- Uses regex `r"^([A-Z]+\d*)([A-Z]+\d*)$"` for edge ID matching
- Implements proper connection reindexing to maintain traffic light logic consistency
- Applies junction joining for cleaner network topology

**Design Decisions**:
- Separates full grid generation from junction removal for modularity
- Maintains data integrity across multiple XML files when removing junctions
- Uses SUMO's built-in tools rather than manual network construction

#### `split_edges.py`
**Purpose**: Implements edge splitting to create more complex network topology by dividing edges into body and head segments.

**Key Functions**:
- `parse_shape(shape_str)`: Converts SUMO shape format to coordinate tuples
- `split_shape(coords, split_dist)`: Splits polyline at specific distance from end using Shapely geometry
- `insert_split_edges()`: Main function performing 4-step process: parse network, compute splits, update nodes/edges, redirect connections

**Key Algorithms**:
- Uses Shapely LineString for robust geometric operations
- Calculates split position as (total_length - split_dist) from downstream end
- Handles both simple and complex multi-point geometries

**Design Decisions**:
- Fixed split distance from downstream end (CONFIG.HEAD_DISTANCE)
- Leverages Shapely for geometric operations rather than manual calculations
- Maintains consistency across multiple XML files

#### `lane_counts.py`
**Purpose**: Sophisticated lane count assignment system that randomizes lane counts while maintaining proper traffic flow and signal logic.

**Key Functions**:
- `set_lane_counts(seed, min_lanes, max_lanes)`: Main function implementing complex multi-step process
- `infer_direction(from_edge_id, to_edge_id)`: Determines turn direction using edge geometry and cross products
- `map_to_lanes(R, to_edge_id)`: Intelligent lane-to-lane mapping with overflow distribution
- `classify_connection_direction(conn)`: Classifies connections for traffic light phasing

**Key Algorithms**:
- Groups base edges with their _H variants for coordinated lane assignment
- Uses right-to-left mapping convention with overflow distribution
- Calculates vectors and cross products for turn direction classification
- Updates traffic light states to match new lane configurations

**Design Decisions**:
- Logical movement assignments (left lanes for left turns, etc.)
- Conflict prevention through proper lane assignment
- Bidirectional mapping handling both sparse and dense configurations

#### `edge_attrs.py`
**Purpose**: Assigns random attractiveness attributes to edges using Poisson distribution for realistic traffic modeling.

**Key Functions**:
- `assign_edge_attractiveness(seed, net_file_in, net_file_out, lambda_depart, lambda_arrive)`: Adds depart/arrive attractiveness using Poisson sampling

**Key Algorithms**:
- Uses NumPy's Poisson distribution for sampling
- Applies consistent random seed for reproducibility
- Supports in-place modification when no output file specified

**Design Decisions**:
- Poisson distribution chosen for realistic traffic flow modeling
- Separate lambda parameters for departure and arrival attractiveness

#### `traffic_lights.py`
**Purpose**: Injects static traffic light control logic with proper connection indexing and phase timing.

**Key Functions**:
- `inject_traffic_lights(net_file, program_id, green_duration, yellow_duration)`: Implements 5-step process for comprehensive traffic light control
- `is_ns(conn)`: Classifies connection as North-South movement

**Key Algorithms**:
- Four-phase control: NS-green, NS-yellow, EW-green, EW-yellow
- Uses netconvert for proper linkIndex assignment
- Maintains proper connection states while adding control logic

**Design Decisions**:
- Conflict-free phasing ensures NS and EW movements don't conflict
- Preserves original connection states while adding control logic
- Automatic junction detection for controllable intersections

#### `zones.py`
**Purpose**: Extracts zone polygons from junction grids and assigns land use classifications following the methodology from "A Simulation Model for Intra-Urban Movements" paper.

**Key Functions**:
- `extract_zones_from_junctions(cell_size, seed, fill_polygons, inset)`: Main function implementing cellular grid methodology
  - Parses junction coordinates from raw .nod.xml file
  - Creates (n-1)×(n-1) zones for n×n junctions
  - Assigns land use types and attractiveness values
  - Outputs both GeoJSON and SUMO polygon files
- `assign_land_use_to_zones(features, seed)`: Uses BFS clustering algorithm for realistic spatial land use distribution
  - Implements contiguous clustering based on CONFIG.land_uses percentages
  - Creates realistic spatial patterns rather than random assignment

**Key Algorithms**:
- **Cellular Grid Creation**: Creates zones as rectangles between adjacent junctions (not covering entire coordinate space)
- **BFS Clustering**: Uses breadth-first search to create contiguous land use clusters
  - Calculates target counts for each land use type based on percentages
  - Implements 4-connected neighbor finding (cardinal directions)
  - Ensures realistic spatial distribution with contiguous areas
- **Attractiveness Assignment**: Assigns random attractiveness values (θᵢ) following normal distribution
  - Uses mean=0.5, std=0.2 to keep values mostly in [0,1] range
  - Clips values to [0,1] bounds for consistency

**Zone Extraction Process**:
1. **Junction Parsing**: Extracts coordinates from .nod.xml, filtering out internal nodes and split edge nodes
2. **Grid Analysis**: Determines unique x,y coordinates and infers cell size if not provided
3. **Zone Creation**: Creates rectangular zones between adjacent junctions with optional inset
4. **Land Use Assignment**: Applies clustering algorithm to assign realistic land use patterns
5. **Attractiveness Values**: Assigns normally distributed attractiveness values for each zone
6. **File Output**: Generates both GeoJSON (for analysis) and SUMO .poly.xml (for simulation)

**Zone Properties**:
- `zone_id`: Unique identifier (format: "Z_i_j")
- `i`, `j`: Grid coordinates
- `cell_size`: Size of cell in meters
- `center_x`, `center_y`: Zone center coordinates
- `area`: Zone area (cell_size²)
- `land_use`: Assigned land use type from CONFIG.land_uses
- `color`: Visualization color for land use type
- `attractiveness`: Attractiveness value (θᵢ) for traffic generation

**Design Decisions**:
- **Academic Methodology**: Based on established urban simulation research
- **Zones as Intervals**: Defined as rectangular areas between junctions, not centered on them
- **Realistic Clustering**: Uses BFS to create contiguous land use areas rather than random assignment
- **Dual Output**: GeoJSON for analysis/visualization, SUMO polygons for simulation integration
- **Configurable Parameters**: Supports customizable cell size, fill options, and boundary insets
- **Extensible Design**: ThetaGenerator protocol allows custom attractiveness functions
- **Statistical Distributions**: Uses normal distribution for attractiveness values following academic standards

### Traffic Generation (`src/traffic/`)

#### `builder.py`
**Purpose**: Main orchestrator for vehicle creation and route generation, coordinating sampling, routing, and XML writing.

**Key Functions**:
- `generate_vehicle_routes(net_file, output_file, num_vehicles, seed)`: Main orchestration function with retry logic

**Key Algorithms**:
- Uses weighted random sampling for vehicle types based on CONFIG.vehicle_weights
- Implements retry logic (up to 20 attempts) for finding valid routes
- Assigns incremental departure times (vehicle ID as departure time)

**Design Decisions**:
- Strategy pattern with pluggable sampler and router
- Robust error handling with retry mechanism
- Filters out internal edges from network
- Skips vehicles that can't find valid routes rather than failing

#### `edge_sampler.py`
**Purpose**: Implements edge sampling strategies for selecting start/end edges using attractiveness-based weighted sampling.

**Key Functions**:
- `AttractivenessBasedEdgeSampler`: Concrete implementation using edge attributes
- `_weights(edges, attr)`: Extracts weight values with fallback to uniform distribution
- `_choose(edges, weights, k)`: Performs weighted random sampling

**Key Algorithms**:
- Uses `random.choices` for weighted sampling
- Graceful fallback to uniform weights when all weights are zero
- Separate sampling for start (`depart_attractiveness`) and end (`arrive_attractiveness`) edges

**Design Decisions**:
- Strategy pattern with abstract EdgeSampler interface
- Fallback pattern for graceful degradation
- Dependency injection of RNG for testability

#### `routing.py`
**Purpose**: Implements routing strategies for computing paths between edges using shortest path algorithms.

**Key Functions**:
- `ShortestPathRoutingStrategy`: Concrete implementation using SUMO's Dijkstra
- `compute_route(start_edge, end_edge)`: Computes shortest path between edges

**Key Algorithms**:
- Leverages SUMO's built-in `getShortestPath` method
- Handles edge ID to edge object conversion
- Returns empty list for invalid/unreachable routes

**Design Decisions**:
- Strategy pattern with abstract RoutingStrategy interface
- Adapter pattern wrapping SUMO's pathfinding API
- Null object pattern (empty list) instead of None for failed routes

#### `xml_writer.py`
**Purpose**: Handles XML serialization for SUMO route files with proper formatting.

**Key Functions**:
- `write_routes(outfile, vehicles, vehicle_types)`: Generates SUMO-compatible routes XML

**Key Algorithms**:
- Uses ElementTree for XML generation
- Serializes route edges as space-separated strings
- Converts all attribute values to strings for XML compatibility

**Design Decisions**:
- Builder pattern for incremental XML construction
- Separates vehicle types from vehicle instances
- UTF-8 encoding with XML declaration

### Validation (`src/validate/`)

#### `errors.py`
**Purpose**: Defines custom exception types for validation errors.

**Key Classes**:
- `ValidationError(RuntimeError)`: Custom exception for validation failures

**Design Decisions**:
- Inherits from RuntimeError for compatibility
- Provides specific exception type for validation failures

#### `validate_network.py`
**Purpose**: Contains runtime invariant validators for network generation steps.

**Key Functions**:
- `verify_generate_grid_network()`: Validates grid network structure, junction count, edge count, bounding box
- `verify_extract_zones_from_junctions()`: Cross-validates network structure against generated zones
- `verify_set_lane_counts()`: Validates lane assignments and connectivity with sophisticated tolerance handling
- `verify_assign_edge_attractiveness()`: Statistical validation of Poisson distribution parameters
- `verify_inject_traffic_lights()`: Validates traffic light consistency across junctions and connections

**Key Algorithms**:
- Uses `4 * grid_dim * (grid_dim - 1)` formula for theoretical maximum edges
- Complex connectivity analysis with border/interior lane tolerance
- Statistical validation with configurable tolerance for Poisson parameters
- Cross-references junctions, connections, and traffic light logic

**Design Decisions**:
- Fail-fast validation with immediate exception raising
- Comprehensive checking with multiple validation aspects per function
- Configurable tolerance for statistical validations

#### `validate_traffic.py`
**Purpose**: Contains inline validator for vehicle route generation.

**Key Functions**:
- `verify_generate_vehicle_routes()`: Validates routes XML against network topology

**Key Algorithms**:
- Cross-validates routes against network topology
- Statistical check for departure time distribution
- Allows configurable shortfall tolerance (default 2%)

**Design Decisions**:
- Tolerance-based validation for realistic shortfall handling
- Comprehensive route validation checking every edge
- Statistical validation with warnings for suspicious patterns

### Simulation Control (`src/sim/`)

#### `sumo_controller.py`
**Purpose**: TraCI wrapper providing simplified interface for SUMO simulation control.

**Key Functions**:
- `SumoController.__init__()`: Initializes controller with simulation parameters
- `start()`: Starts SUMO with TraCI connection
- `run(control_callback)`: Runs complete simulation loop with per-step callback
- `step()`: Advances simulation by one time step
- `close()`: Closes TraCI connection

**Key Algorithms**:
- Selects appropriate SUMO binary (sumo-gui vs sumo) based on GUI flag
- Implements complete simulation lifecycle management
- Uses callback pattern for extensible control logic

**Design Decisions**:
- Adapter pattern simplifying TraCI interface
- Callback pattern for pluggable control logic
- Supports both GUI and batch modes through single interface

#### `sumo_utils.py`
**Purpose**: Utility functions for SUMO operations including network rebuilding and configuration management.

**Key Functions**:
- `rebuild_network()`: Rebuilds SUMO network using netconvert with all components
- `build_sumo_command()`: Builds standardized SUMO command-line arguments
- `run_sumo()`: Runs single SUMO simulation using subprocess
- `generate_sumo_conf_file()`: Creates SUMO configuration XML file
- `start_sumo_gui()`: Starts SUMO GUI process

**Key Algorithms**:
- Uses subprocess for external SUMO tool integration
- Template-based XML configuration generation
- Comprehensive error handling with informative messages

**Design Decisions**:
- Utility/helper pattern for related functions
- Command builder pattern for centralized command construction
- Template pattern for XML configuration generation
- Extensive optional parameters for flexibility

## Key Architectural Patterns

### Overall System Architecture:
1. **Pipeline Pattern**: Sequential processing with clear step boundaries
2. **Strategy Pattern**: Pluggable algorithms for sampling, routing, and validation
3. **Adapter Pattern**: Wraps SUMO APIs with consistent interfaces
4. **Validation-First**: Each pipeline step has corresponding validation functions
5. **Configuration-Driven**: Central configuration system using dataclasses

### Design Principles:
- **Separation of Concerns**: Clear module boundaries and responsibilities
- **Extensibility**: Abstract interfaces allow for different implementations
- **Robustness**: Comprehensive error handling and validation
- **Reproducibility**: Seeded random number generation throughout
- **SUMO Integration**: Leverages SUMO's native tools and data structures