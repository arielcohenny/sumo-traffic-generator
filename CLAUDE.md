# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Instructions for Claude Code

**CRITICAL RULES FOR CLAUDE CODE ASSISTANCE:**

- **DO NOT** implement features, changes, or solutions that the user did not explicitly request
- **DO NOT** implement features, changes, or solutions that are not in the SPECIFICATION.md. If asked to do something not in the specification, do not do it and inform the user
- **DO NOT** add fallback solutions, error handling, or "helpful" additions without explicit user approval
- **DO NOT** change the architecture, pipeline steps, or execution order without user consent
- **DO NOT** write the name Nimrod explicitly but instead use 'Tree Method'
- **NEVER MODIFY SPECIFICATION.md WITHOUT EXPLICIT USER APPROVAL**
- **ALWAYS** ask for permission before making structural changes
- **STICK TO** exactly what the user asks for - nothing more, nothing less
- When the user asks a question, answer it directly without implementing unsolicited "improvements"

**COMMIT BEHAVIOR:**

- **DO NOT** include co-author lines in commit messages
- **DO NOT** include "Generated with Claude Code" references
- **DO NOT** include "Co-Authored-By: Claude <noreply@anthropic.com>" lines
- **KEEP COMMITS SIMPLE**: Only include the user as the commit author
- Commit messages should be clear and descriptive without AI attribution

## Project Overview

This is a Python-based SUMO traffic generator that creates dynamic traffic simulations with intelligent signal control. It supports synthetic orthogonal grid networks, applies configurable lane assignments, and uses Tree Method's decentralized traffic control algorithm for dynamic signal optimization.

## Common Commands

### Setup and Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and GUI support
pip install -r requirements.txt
pip install -e .  # Installs 'dbps' command for GUI

# Verify GUI installation
dbps --help
```

### GUI Interface

#### Web GUI (Recommended)

```bash
# Launch the visual web interface (recommended for most users)
dbps

# The GUI provides:
# - Interactive parameter configuration with real-time validation
# - Automatic CLI command generation for scripting
# - Real-time simulation monitoring with progress updates
# - Integrated results visualization and log display
# - Chrome app mode for desktop-like experience

# GUI workflow:
# 1. Run 'dbps' to launch web interface
# 2. Configure parameters using visual widgets
# 3. Optionally enable "SUMO GUI" checkbox for traffic visualization
# 4. Click "Run Simulation" to execute
# 5. Monitor progress and view results in real-time
```

### Running the Application

#### Synthetic Grid Networks

```bash
# Basic run
env PYTHONUNBUFFERED=1 python -m src.cli

# With GUI and custom parameters
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 7 --num_vehicles 500 --gui

# Full parameter example with vehicle types and routing strategies
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --junctions_to_remove 0 \
  --lane_count realistic \
  --num_vehicles 300 \
  --seed 42 \
  --step-length 1.0 \
  --end-time 7200 \
  --workspace custom_output \
  --attractiveness land_use \
  --time_dependent \
  --start_time_hour 7.0 \
  --routing_strategy 'shortest 70 realtime 30' \
  --vehicle_types 'passenger 90 public 10' \
  --departure_pattern six_periods \
  --traffic_control tree_method \
  --gui
```

### Workspace Configuration

The `--workspace` parameter specifies the **parent directory** where a `workspace/` folder will be created for simulation output files:

```bash
# Default behavior (creates './workspace/' in current directory)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100

# Custom parent directory (creates 'my_simulation/workspace/')
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --workspace my_simulation

# Absolute path (creates '/home/user/simulations/test1/workspace/')
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --workspace "/home/user/simulations/test1"

# Relative path from current directory (creates 'results/experiment_01/workspace/')
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --workspace "results/experiment_01"

# Safe to use current directory (creates './workspace/', same as default)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --workspace "./"
```

**Generated Files Structure:**
```
[workspace_directory]/
├── grid.net.xml          # SUMO network file
├── grid.nod.xml          # Network nodes
├── grid.edg.xml          # Network edges  
├── grid.con.xml          # Network connections
├── grid.tll.xml          # Traffic light definitions
├── vehicles.rou.xml      # Vehicle routes and types
├── zones.poly.xml        # Land use zones
├── grid.sumocfg          # SUMO configuration
├── sumo_statistics.xml   # Simulation results
└── sumo_errors.log       # Error logs (if any)
```

**Notes:**
- Default creates `"./workspace/"` in current working directory  
- A `workspace/` subdirectory is always created in the specified parent directory
- Only the `workspace/` folder contents are cleaned, never the parent directory
- **Safe**: You can use `--workspace ./` or any directory without risk of data loss
- Parent directory is automatically created if it doesn't exist
- GUI automatically uses the configured workspace for file display

### Tested Scenarios (5x5 Grid, 150m Blocks)

These 10 scenarios have been verified to work correctly:

```bash
# Scenario 1: Morning Rush Hour Peak Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 800 --step-length 1.0 --end-time 7200 --departure_pattern six_periods --start_time_hour 7.0 --gui

# Scenario 2: Light Evening Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 500 --step-length 1.0 --end-time 5400 --departure_pattern uniform --start_time_hour 20.0 --gui

# Scenario 3: All-Day Urban Simulation
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1200 --step-length 1.0 --end-time 28800 --departure_pattern six_periods --attractiveness land_use --workspace "all_day_simulation" --gui

# Scenario 4: Custom Rush Hour Pattern
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 750 --step-length 1.0 --end-time 10800 --departure_pattern 'rush_hours:7-9:50,17-19:40,rest:10' --routing_strategy 'shortest 70 realtime 30' --gui

# Scenario 5: Midnight to Dawn Low Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 600 --step-length 1.0 --end-time 21600 --departure_pattern six_periods --start_time_hour 0.0 --time_dependent --gui

# Scenario 6: High-Density Stress Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1500 --step-length 1.0 --end-time 14400 --departure_pattern uniform --routing_strategy 'shortest 50 realtime 30 fastest 20' --workspace "stress_test_results" --gui

# Scenario 7: Time-Dependent Attractiveness Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 900 --step-length 1.0 --end-time 12600 --departure_pattern six_periods --time_dependent --start_time_hour 8.0 --gui

# Scenario 8: Weekend Traffic Pattern
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 650 --step-length 1.0 --end-time 18000 --departure_pattern uniform --start_time_hour 10.0 --attractiveness land_use --gui

# Scenario 9: Infrastructure Disruption Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1000 --step-length 1.0 --end-time 9000 --departure_pattern six_periods --routing_strategy 'shortest 40 realtime 60' --seed 123 --gui

# Scenario 10: Multi-Modal Traffic Mix
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 850 --step-length 1.0 --end-time 16200 --departure_pattern six_periods --vehicle_types 'passenger 90 public 10' --attractiveness land_use --gui

# Quick Development Tests
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 500 --end-time 1800 --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 1000 --end-time 3600 --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 800 --routing_strategy 'realtime 100' --end-time 3600 --gui

# Traffic Control Method Comparison Tests
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 800 --end-time 3600 --traffic_control tree_method --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 800 --end-time 3600 --traffic_control actuated --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 800 --end-time 3600 --traffic_control fixed --gui

# Experimental comparison (identical conditions, different control methods)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control tree_method
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control actuated
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control fixed
```

### Tree Method Sample Testing

Use original test cases to validate Tree Method implementation:

```bash
# Basic Tree Method sample run with GUI
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --gui

# Compare different traffic control methods on same pre-built network
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --gui
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control actuated --gui
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control fixed --gui

# Performance testing without GUI
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --end-time 3600
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control actuated --end-time 3600

# Extended simulation (2 hours like original)
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --end-time 7300

# Quick validation runs
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --end-time 1800 --gui
```

**Tree Method Sample Features:**

- **Bypass Mode**: Skips Steps 1-8, goes directly to Step 9 (Dynamic Simulation)
- **Pre-built Networks**: Uses original complex urban networks (946 vehicles, 2-hour simulation)
- **Research Validation**: Test our Tree Method implementation against established benchmarks
- **Method Comparison**: Compare Tree Method vs Actuated vs Fixed on identical network conditions
- **File Management**: Automatically copies and adapts sample files to our pipeline naming convention

### Experimental Framework

The project includes a comprehensive experimental framework for comparing traffic control methods:

```bash
# Run moderate traffic experiment (600 vehicles, 2 hours)
cd evaluation/benchmarks/experiment-01-moderate-traffic
./run_experiment.sh

# Run high traffic experiment (1200 vehicles, 2 hours)
cd evaluation/benchmarks/experiment-02-high-traffic
./run_experiment.sh

# Analyze results after experiment completion
python analyze_results.py

# View experiment progress (during execution)
ls -la results/*/  # Check progress across all methods
tail -f results/tree_method/run_1.log  # Monitor specific run

# Quick single-method test (for debugging)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 200 --num_vehicles 600 --end-time 7200 --traffic_control tree_method --seed 1

# Compare methods with identical conditions
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --seed 42 --traffic_control tree_method
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --seed 42 --traffic_control actuated
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --seed 42 --traffic_control fixed
```

**Experimental Framework Features:**

- **Automated Execution**: Scripts run 80 simulations (20 per method) with different seeds
- **Methods Compared**: Tree Method, SUMO Actuated, Fixed timing, Random proxy
- **Metrics Tracked**: Travel times, completion rates, throughput, vehicle arrivals/departures
- **Statistical Analysis**: Confidence intervals, significance tests, performance comparisons
- **Virtual Environment**: Proper activation ensures all dependencies are available
- **Research Validation**: Framework designed to replicate Tree Method experimental methodology

**Expected Experiment Results:**

- **Tree Method vs Fixed**: 20-45% improvement in travel times
- **Tree Method vs Actuated**: 10-25% improvement
- **Higher Completion Rates**: Tree Method should achieve better completion rates under congestion
- **Statistical Significance**: 20 iterations provide robust confidence intervals

### Testing

```bash
# Run all software tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Pipeline integration tests
pytest tests/validation/     # Domain validation tests

# Run performance benchmarks
cd evaluation/benchmarks/experiment-01-moderate-traffic && ./run_experiment.sh
```

### Code Quality (to be implemented)

```bash
# Linting and formatting (not yet configured)
# Recommended: black, flake8/ruff, mypy
```

## Seed System (Multiple Seeds Support)

The system supports both single and multiple seed configurations for fine-grained experimental control:

### Single Seed (Backward Compatible)

```bash
# Traditional approach - sets all seeds to the same value
env PYTHONUNBUFFERED=1 python -m src.cli --seed 42 --num_vehicles 100 --gui
```

### Multiple Seeds (Advanced Control)

For research and experimentation, you can control different aspects independently:

```bash
# Network structure vs traffic separation
env PYTHONUNBUFFERED=1 python -m src.cli \
  --network-seed 42 \
  --private-traffic-seed 123 \
  --public-traffic-seed 456 \
  --num_vehicles 100 --gui

# Example: Same network, different private traffic patterns
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 200 --num_vehicles 100
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 101 --public-traffic-seed 200 --num_vehicles 100  # Different private traffic, same public

# Example: Same private traffic, different public transportation
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 200 --num_vehicles 100
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 201 --num_vehicles 100  # Same private traffic, different public
```

### Seed Types

- **`--network-seed`**: Controls network structure generation
  - Junction removal selection
  - Lane count assignment (when using "random" algorithm)  
  - Land use zone clustering and type assignment
  - Edge attractiveness value generation
  
- **`--private-traffic-seed`**: Controls passenger vehicle generation
  - Vehicle type assignment for private vehicles
  - Route generation (start/end edges, path computation)
  - Departure time generation for private vehicles
  
- **`--public-traffic-seed`**: Controls public transportation vehicle generation
  - Vehicle type assignment for public vehicles
  - Route generation for public vehicles
  - Departure time generation for public vehicles

### Use Cases

1. **Network Sensitivity Analysis**: Keep traffic patterns constant while testing different network topologies
2. **Traffic Pattern Studies**: Keep network constant while varying private vs public traffic scenarios
3. **Public Transport Planning**: Test different public transport schedules against consistent private traffic
4. **Reproducible Research**: Enable precise control over experimental variables for scientific studies

## Architecture Overview

### Pipeline Architecture

The application follows a sequential 7-step pipeline with support for both synthetic and real-world networks:

#### For Synthetic Grid Networks:

1. **Network Generation** (`src/network/generate_grid.py`): Creates orthogonal grid using SUMO's netgenerate (always 1-lane)
2. **Integrated Edge Splitting with Flow-Based Lane Assignment** (`src/network/split_edges_with_lanes.py`): Splits edges and assigns lanes based on traffic flow requirements with sophisticated movement distribution

#### Pipeline Steps (3-8):

3. **Zone Extraction** (`src/network/zones.py`): Extracts polygonal zones from junctions (currently disabled)
4. **Edge Attractiveness** (`src/network/edge_attrs.py`): Computes departure/arrival weights using multiple research-based methods with 4-phase temporal system
5. **Traffic Light Injection** (`src/network/traffic_lights.py`): Adds default four-phase signal plans
6. **Route Generation** (`src/traffic/`): Generates vehicle routes using 4-strategy routing system and 3-type vehicle assignment
7. **SUMO Configuration Generation**: Creates SUMO configuration files
8. **Dynamic Simulation** (`src/sumo_integration/sumo_controller.py`): Runs SUMO with TraCI integration, Tree Method algorithm, and real-time phase switching

### Key Modules

**Core Orchestration**:

- `src/cli.py`: Main entry point and pipeline orchestration
- `src/config.py`: Central configuration using dataclasses

**Network Generation**:

- `src/network/`: All network manipulation and generation logic
- `src/network/split_edges_with_lanes.py`: Edge splitting algorithm for synthetic networks
- Uses SUMO's netgenerate tools
- Validates network structure at each step

**Traffic Generation**:

- `src/traffic/builder.py`: Main traffic generation orchestrator
- `src/traffic/routing.py`: Four-strategy routing system with percentage mixing
- `src/traffic/vehicle_types.py`: Three-type vehicle system with validation
- `src/traffic/xml_writer.py`: SUMO XML file generation

**Simulation Control**:

- `src/sumo_integration/sumo_controller.py`: TraCI wrapper with per-step callback API
- `src/sumo_integration/sumo_utils.py`: SUMO utility functions
- `src/orchestration/simulator.py`: Main simulation orchestrator
- `src/orchestration/traffic_controller.py`: Traffic control abstractions and factory

**Traffic Control**:

- `src/traffic_control/decentralized_traffic_bottlenecks/`: Tree Method implementation
- `src/traffic_control/decentralized_traffic_bottlenecks/integration.py`: Bridges simulator with algorithm
- `src/traffic_control/decentralized_traffic_bottlenecks/classes/net_data_builder.py`: Network adapter with dead-end street handling for real topologies

**Validation**:

- `src/validate/`: Runtime validation functions for each pipeline step
- Uses custom validation errors and comprehensive checking

### Configuration System

Central configuration in `src/config.py` using dataclasses:

- `GridConfig`: Network generation parameters
- `TrafficConfig`: Vehicle and route parameters
- `SimulationConfig`: SUMO simulation parameters
- `AlgorithmConfig`: Tree Method algorithm parameters

**Key Configuration Constants**:

- `HEAD_DISTANCE = 50`: Distance from downstream end when splitting edges
- `MIN_LANES = 1`, `MAX_LANES = 3`: Lane count bounds for randomization
- `LAMBDA_DEPART = 3.5`, `LAMBDA_ARRIVE = 2.0`: Poisson distribution parameters for edge attractiveness
- `DEFAULT_JUNCTION_RADIUS = 10.0`: Junction radius in meters
- `DEFAULT_ROUTING_STRATEGY = "shortest 100"`: Default routing strategy
- `DEFAULT_VEHICLE_TYPES = "passenger 90 public 10"`: Default vehicle distribution

### Land Use Zone Generation System

**Zone-Based Traffic Generation**: The system creates land use zones for traffic generation:

- **Junction-Based Zones**: Creates zones based on network topology using proven cellular grid methodology
- **Research-Based Land Use**: Six land use types with clustering algorithm (Residential, Mixed, Employment, Public Buildings, Public Open Space, Entertainment/Retail)
- **Configurable Subdivision**: Zone size independent of junction spacing, controlled by `--land_use_block_size_m` (default 25.0m)
- **Variety & Realism**: Generates diverse zone types with appropriate colors and attractiveness values

### Generated Files Structure

All generated files are placed in `workspace/` directory:

- `grid.net.xml`: SUMO network file
- `grid.nod.xml`, `grid.edg.xml`, `grid.con.xml`, `grid.tll.xml`: Network components
- `vehicles.rou.xml`: Vehicle routes with types and routing strategies
- `zones.poly.xml`: Zone polygons with land use classification
- `grid.sumocfg`: SUMO configuration file

## Development Notes

### Deprecated Files

- **`src/network/split_edges.py`**: ❌ **DEPRECATED** - Replaced by integrated approach in `split_edges_with_lanes.py`
  - No longer imported or used anywhere in the codebase
  - Safe to delete to reduce codebase complexity
  - Functionality merged into unified edge splitting implementation

### Dependencies

- **SUMO**: Requires SUMO installation with netgenerate, netconvert, sumo, sumo-gui
- **Python Libraries**: numpy, shapely, geopandas, sumolib, traci, xmltodict, alive-progress
- **Testing**: pytest for software testing (tests/), separate evaluation framework for research (evaluation/)

### Current Implementation Status

- ✅ **Integrated Edge Splitting**: Fully implemented and operational
- ✅ **Error Resolution**: Fixed TraCI integration issues and XML parsing problems
- ✅ **Code Cleanup**: Removed unused functions and eliminated code duplication
- Several pipeline steps have commented-out code or early exit points
- Most validation functions are disabled in the main CLI
- Zone extraction (Step 2) is currently disabled
- Separate testing framework (tests/) for software quality and evaluation framework (evaluation/) for research

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

#### GUI-First Development (Recommended)
1. Launch web GUI with `dbps`
2. Configure parameters using visual widgets with real-time validation
3. Enable "SUMO GUI" checkbox for traffic visualization
4. Run simulation and monitor progress in web interface
5. Analyze results using integrated log display and file browser

#### CLI Development
1. Modify configuration in `src/config.py`
2. Run pipeline with `python -m src.cli`
3. Check generated files in `workspace/` directory
4. Use `--gui` flag for SUMO visual debugging
5. Validate results using the validation functions (when enabled)

#### Dual GUI Testing
```bash
# Launch web GUI first
dbps
# Configure parameters and enable SUMO GUI option
# Both interfaces will work together:
# - Web GUI: Parameter control, progress monitoring, log display
# - SUMO GUI: Real-time traffic visualization and analysis
```

## Memory

- This project is a sophisticated SUMO traffic simulation framework with intelligent grid network generation and dynamic traffic control
- **Dual GUI System**: Provides both web-based parameter configuration (`dbps` command) and SUMO traffic visualization (`--gui` flag) that work together
- Uses a comprehensive 8-step pipeline for network and traffic generation
- Implements advanced algorithms for zone extraction, land use assignment, and traffic routing
- Designed with multiple architectural patterns including Strategy, Adapter, and Pipeline patterns
- Supports reproducible simulations through seeded random generation
- **Integrated Edge Splitting with Flow-Based Lane Assignment**:
  - ✅ **COMPLETED & WORKING**: Unified edge splitting and lane assignment in single optimized process
  - Replaces separate edge splitting and lane configuration steps with `split_edges_with_lanes.py`
  - Always starts with 1-lane network from netgenerate for consistent behavior
  - Head lanes = max(number_of_movements, lane_count) to ensure sufficient capacity
  - Tail lanes = lane_count (based on realistic/random/fixed algorithms)
  - Sophisticated movement distribution algorithm that maximizes total movement assignments
  - Maintains spatial logic: right→right lanes, left→left lanes, straight→middle, u-turn→leftmost
  - Prevents crossing conflicts while ensuring all movements are preserved
  - Even distribution of tail lanes to head lanes for optimal traffic flow
  - **Recent Fixes**: Resolved TraCI integration errors and XML parsing issues (Phase handling, Vehicle ID extraction)
- **Lane Count Algorithms**: Three modes for lane assignment - `realistic` (zone-based demand calculation), `random` (randomized within bounds), and `fixed` (uniform count)
- **Edge Attractiveness Methods**: Three research-based methods (land_use, poisson, iac) with 4-phase temporal system
- **4-Phase Temporal System**:
  - Research-based bimodal traffic patterns with morning/evening peaks
  - Pre-calculated attractiveness profiles for efficient simulation
  - Real-time phase switching during simulation based on start time
  - Supports both full-day (24h) and rush hour analysis
  - 1:1 time mapping (1 sim second = 1 real second)
- **4-Strategy Routing System**:
  - Four routing strategies: shortest (static), realtime (30s rerouting), fastest (45s rerouting), attractiveness (multi-criteria)
  - Percentage-based vehicle assignment with validation (must sum to 100%)
  - Dynamic rerouting via TraCI for realtime and fastest strategies
  - Integration with existing temporal and attractiveness systems
  - Research-based implementation mimicking GPS navigation apps
- **2-Type Vehicle System**:
  - Two vehicle types: passenger (cars), public (buses)
  - Percentage-based vehicle assignment with CLI support (e.g., "passenger 90 public 10")
  - Validation ensures percentages sum to 100%
  - Default distribution: 90% passenger, 10% public
  - Each type has distinct characteristics: length, maxSpeed, acceleration, deceleration, sigma
  - Seamless integration with routing strategies and temporal systems
- **Traffic Light Strategies**:
  - Three phasing strategies:
    - `opposites` (default): Opposing directions signal together, 90s cycle (42s+3s green+yellow per direction)
    - `incoming`: Each edge gets independent phase, 132s cycle (30s+3s per edge)
    - `partial_opposites`: Separates straight+right from left+u-turn movements, 90s cycle (30s+3s straight+right, 9s+3s left+u-turn per direction)
  - **partial_opposites Requirements**:
    - Minimum 2 lanes per edge (enforced by validation)
    - Lane assignment: lane 0 for straight+right, lane 1+ for left+u-turn
    - Automatic validation of phase structure and cycle timing
    - Compatible with all lane count algorithms (realistic, random, fixed)
  - **Usage Examples**:
    ```bash
    # Default opposites strategy (1+ lanes)
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --gui

    # Incoming strategy (each edge independent)
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --traffic_light_strategy incoming --gui

    # Partial opposites strategy (requires 2+ lanes)
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --lane_count 2 --traffic_light_strategy partial_opposites --gui

    # Partial opposites with realistic lane assignment (automatically enforces 2+ lanes)
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 800 --lane_count realistic --traffic_light_strategy partial_opposites --end-time 3600 --gui

    # Compare strategies with identical network conditions
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --seed 42 --traffic_light_strategy opposites --gui
    env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --seed 42 --lane_count 2 --traffic_light_strategy partial_opposites --gui
    ```
- **Vehicle Departure Patterns**:
  - Replaced sequential departure (0, 1, 2, 3...) with realistic temporal distribution based on research papers
  - Default: six_periods system with research-based 6-period daily structure (Morning 20%, Morning Rush 30%, Noon 25%, Evening Rush 20%, Evening 4%, Night 1%)
  - Alternative patterns: uniform distribution, custom rush_hours
  - Automatically scales to simulation end_time (default 24 hours)
  - Compatible with all routing strategies and vehicle types
  - CLI support via --traffic_light_strategy parameter
  - Built on netgenerate's --tls.layout functionality for proven traffic signal logic
  - Compatible with any lane configuration (1+ lanes) and Tree Method optimization
  - Opposites strategy for efficient green time usage, incoming strategy for unbalanced traffic scenarios
- **Advanced Features**:
  - Zone-based traffic demand calculation using land use types and attractiveness values
  - Spatial analysis for edge-zone adjacency detection
  - Research-based land use multipliers and gravity model parameters
  - Modular architecture allowing combination of spatial, temporal, land use, and routing factors
- **Tree Method Traffic Control Algorithm**:
  - Uses Tree Method decentralized traffic control algorithm for dynamic signal optimization
  - Implements intelligent signal control through tree-based method
  - Dynamically adjusts traffic light phases based on real-time traffic bottlenecks
  - Aims to minimize overall traffic congestion by decentralized decision-making
  - Adapts signal timing based on local traffic conditions at each intersection
- **Traffic Control System**:
  - ✅ **COMPLETED & WORKING**: Implemented `--traffic_control` argument for switching between different traffic control methods
  - **Three Control Methods**: `tree_method` (default, Tree Method algorithm), `actuated` (SUMO gap-based), `fixed` (static timing)
  - **Conditional Object Initialization**: Only loads Tree Method objects when using tree_method for optimal performance
  - **Experimental Comparison**: Enables A/B testing between different traffic control approaches using identical network conditions
  - **Baseline Evaluation**: SUMO Actuated serves as primary baseline for comparing Tree Method performance
  - **Integration**: Seamlessly works with all existing features (routing strategies, vehicle types, temporal patterns)
- **Recent Updates (Latest Session)**:
  - ✅ **Fixed Integration Errors**: Resolved "string indices must be integers, not 'str'" errors in TraCI integration
  - ✅ **XML Phase Parsing**: Fixed traffic light phase parsing to handle both single and multiple phase cases
  - ✅ **Vehicle ID Handling**: Improved vehicle index extraction with proper error handling
  - ✅ **Route File Parsing**: Fixed Path object to string conversion in TraCI controller
  - ✅ **Code Cleanup**: Removed 427 lines of unused code from `lane_counts.py` (78% reduction)
  - ✅ **Eliminated Duplication**: Removed duplicate `load_zones_data` function, now imports from `edge_attrs.py`
  - ✅ **File Removal**: Can safely delete `split_edges.py` (no longer referenced)
  - ✅ **Working System**: All components now operational with successful test runs
  - ✅ **Traffic Control Implementation**: Successfully implemented traffic control method switching with tested examples
  - ✅ **partial_opposites Traffic Light Strategy**: Implemented third traffic light strategy that separates straight+right from left+u-turn movements
    - 90-second cycle time: 30s+3s for straight+right, 9s+3s for left+u-turn per direction
    - Enforces minimum 2 lanes per edge with comprehensive validation
    - Turn angle-based movement classification for accurate phase assignment
    - Post-processing approach consistent with incoming strategy
    - Full integration with lane count algorithms and validation pipeline
    - Documented in `docs/PARTIAL_OPPOSITES_STRATEGY.md` with implementation roadmap
- **Experimental Framework**:
  - ✅ **COMPLETED & WORKING**: Comprehensive experimental framework for traffic control method comparison
  - **Two Main Experiments**: moderate-traffic (600 vehicles) and high-traffic (1200 vehicles) over 2-hour simulations
  - **Statistical Rigor**: 20 iterations per method (80 total simulations per experiment) with different random seeds
  - **Four Methods Compared**: Tree Method, SUMO Actuated, Fixed timing, and Random proxy (mixed routing)
  - **Automated Execution**: `run_experiment.sh` scripts handle all 80 simulations with proper virtual environment activation
  - **Metrics Collection**: Real-time tracking of travel times, completion rates, throughput, and vehicle arrivals/departures
  - **Statistical Analysis**: `analyze_results.py` provides summary statistics, confidence intervals, and performance comparisons
  - **Research Replication**: Framework designed to validate Tree Method claims of 20-45% improvement vs fixed timing
  - **Experimental Structure**:
    ```
    evaluation/benchmarks/
    ├── experiment-01-moderate-traffic/  # 600 vehicles, 2-hour simulation
    │   ├── run_experiment.sh           # Automated experiment runner
    │   ├── analyze_results.py          # Statistical analysis script
    │   └── results/                    # Output directory (tree_method, actuated, fixed, random)
    └── experiment-02-high-traffic/     # 1200 vehicles, 2-hour simulation
        ├── run_experiment.sh           # Same structure as experiment-01
        ├── analyze_results.py
        └── results/
    ```
  - **Usage**: `cd evaluation/benchmarks/experiment-01-moderate-traffic && ./run_experiment.sh && python analyze_results.py`
  - **Virtual Environment Fix**: Resolved shapely import issues by adding proper virtual environment activation to experiment scripts
  - **Expected Results**: Tree Method should demonstrate 20-45% improvement in travel times vs fixed timing, 10-25% vs actuated
  - **Publication Ready**: Framework generates publication-quality statistical analysis with confidence intervals and significance tests
- **Land Use Zone Generation System**:
  - ✅ **COMPLETED & WORKING**: Zone system for synthetic grid networks
  - **Junction-Based Zones**: Creates zones based on network topology using proven cellular grid methodology
  - **Research-Based Land Use**: Six land use types with clustering algorithm (Residential, Mixed, Employment, Public Buildings, Public Open Space, Entertainment/Retail)
  - **Configurable Subdivision**: Zone size controlled by `--land_use_block_size_m` (default 25.0m)
  - **Variety & Realism**: Generates diverse zone types with appropriate colors and attractiveness values
  - **File**: `src/network/zones.py` - Updated `extract_zones_from_junctions` function with configurable cell size
  - **Updated Validation**: Fixed zone count validation in `src/validate/validate_network.py` to work with configurable zone sizes
  - **Integration**: Works in CLI pipeline Step 2
- **Reminder**: make sure to periodically update CLAUDE.md and README.md to reflect project developments and improvements
