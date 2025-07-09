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

# Full parameter example with vehicle types and routing strategies
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --junctions_to_remove 0 \
  --lane_count realistic \
  --num_vehicles 300 \
  --seed 42 \
  --step-length 1.0 \
  --end-time 86400 \
  --attractiveness poisson \
  --time_dependent \
  --start_time_hour 7.0 \
  --routing_strategy 'shortest 70 realtime 30' \
  --vehicle_types 'passenger 70 commercial 20 public 10' \
  --departure_pattern six_periods \
  --traffic_control tree_method \
  --gui
```

### Tested Scenarios (5x5 Grid, 150m Blocks)

These 10 scenarios have been verified to work correctly:

```bash
# Scenario 1: Morning Rush Hour Peak Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 800 --step-length 1.0 --end-time 7200 --departure_pattern six_periods --start_time_hour 7.0 --gui

# Scenario 2: Light Evening Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 500 --step-length 1.0 --end-time 5400 --departure_pattern uniform --start_time_hour 20.0 --gui

# Scenario 3: All-Day Urban Simulation
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1200 --step-length 1.0 --end-time 28800 --departure_pattern six_periods --attractiveness poisson --gui

# Scenario 4: Custom Rush Hour Pattern
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 750 --step-length 1.0 --end-time 10800 --departure_pattern 'rush_hours:7-9:50,17-19:40,rest:10' --routing_strategy 'shortest 70 realtime 30' --gui

# Scenario 5: Midnight to Dawn Low Traffic
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 600 --step-length 1.0 --end-time 21600 --departure_pattern six_periods --start_time_hour 0.0 --time_dependent --gui

# Scenario 6: High-Density Stress Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1500 --step-length 1.0 --end-time 14400 --departure_pattern uniform --routing_strategy 'shortest 50 realtime 30 fastest 20' --gui

# Scenario 7: Time-Dependent Attractiveness Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 900 --step-length 1.0 --end-time 12600 --departure_pattern six_periods --time_dependent --start_time_hour 8.0 --gui

# Scenario 8: Weekend Traffic Pattern
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 650 --step-length 1.0 --end-time 18000 --departure_pattern uniform --start_time_hour 10.0 --attractiveness poisson --gui

# Scenario 9: Infrastructure Disruption Test
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1000 --step-length 1.0 --end-time 9000 --departure_pattern six_periods --routing_strategy 'shortest 40 realtime 60' --seed 123 --gui

# Scenario 10: Multi-Modal Traffic Mix
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 850 --step-length 1.0 --end-time 16200 --departure_pattern six_periods --vehicle_types 'passenger 50 commercial 40 public 10' --attractiveness hybrid --gui

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

The application follows a sequential 7-step pipeline:

1. **Network Generation** (`src/network/generate_grid.py`): Creates orthogonal grid using SUMO's netgenerate (always 1-lane)
2. **Integrated Edge Splitting with Flow-Based Lane Assignment** (`src/network/split_edges_with_lanes.py`): Splits edges and assigns lanes based on traffic flow requirements with sophisticated movement distribution
3. **Zone Extraction** (`src/network/zones.py`): Extracts polygonal zones from junctions (currently disabled)
4. **Edge Attractiveness** (`src/network/edge_attrs.py`): Computes departure/arrival weights using multiple research-based methods with 4-phase temporal system
5. **Traffic Light Injection** (`src/network/traffic_lights.py`): Adds default four-phase signal plans
6. **Route Generation** (`src/traffic/`): Generates vehicle routes using 4-strategy routing system and 3-type vehicle assignment
7. **Dynamic Simulation** (`src/sim/sumo_controller.py`): Runs SUMO with TraCI integration, Nimrod's algorithm, and real-time phase switching

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
- `src/traffic/routing.py`: Four-strategy routing system with percentage mixing
- `src/traffic/vehicle_types.py`: Three-type vehicle system with validation
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

**Key Configuration Constants**:

- `HEAD_DISTANCE = 50`: Distance from downstream end when splitting edges
- `MIN_LANES = 1`, `MAX_LANES = 3`: Lane count bounds for randomization
- `LAMBDA_DEPART = 3.5`, `LAMBDA_ARRIVE = 2.0`: Poisson distribution parameters for edge attractiveness
- `DEFAULT_JUNCTION_RADIUS = 10.0`: Junction radius in meters
- `DEFAULT_ROUTING_STRATEGY = "shortest 100"`: Default routing strategy
- `DEFAULT_VEHICLE_TYPES = "passenger 60 commercial 30 public 10"`: Default vehicle distribution

### Generated Files Structure

All generated files are placed in `data/` directory:

- `grid.net.xml`: SUMO network file
- `grid.nod.xml`, `grid.edg.xml`, `grid.con.xml`, `grid.tll.xml`: Network components
- `vehicles.rou.xml`: Vehicle routes with types and routing strategies
- `zones.poly.xml`: Zone polygons (when enabled)
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
- **Testing**: pytest included but no tests currently implemented

### Current Implementation Status

- ✅ **Integrated Edge Splitting**: Fully implemented and operational
- ✅ **Error Resolution**: Fixed TraCI integration issues and XML parsing problems
- ✅ **Code Cleanup**: Removed unused functions and eliminated code duplication
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

## Memory

- This project is a sophisticated SUMO traffic simulation framework with intelligent grid network generation and dynamic traffic control
- Uses a comprehensive 7-step pipeline for network and traffic generation
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
- **Edge Attractiveness Methods**: Five research-based methods (poisson, land_use, gravity, iac, hybrid) with 4-phase temporal system
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
- **3-Type Vehicle System**:
  - Three vehicle types: passenger (cars), commercial (trucks), public (buses)
  - Percentage-based vehicle assignment with CLI support (e.g., "passenger 70 commercial 20 public 10")
  - Validation ensures percentages sum to 100%
  - Default distribution: 60% passenger, 30% commercial, 10% public
  - Each type has distinct characteristics: length, maxSpeed, acceleration, deceleration, sigma
  - Seamless integration with routing strategies and temporal systems
- **Traffic Light Strategies**:
  - Two phasing strategies: opposites (default, opposing directions together) and incoming (each edge separate)
- **Vehicle Departure Patterns**:
  - Replaced sequential departure (0, 1, 2, 3...) with realistic temporal distribution based on research papers
  - Default: six_periods system with research-based 6-period daily structure (Morning 20%, Morning Rush 30%, Noon 25%, Evening Rush 20%, Evening 4%, Night 1%)
  - Alternative patterns: uniform distribution, custom rush_hours, granular hourly control
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
- **Nimrod's Traffic Control Algorithm**: 
  - Uses Nimrod's decentralized traffic control algorithm for dynamic signal optimization
  - Implements intelligent signal control through tree-based method
  - Dynamically adjusts traffic light phases based on real-time traffic bottlenecks
  - Aims to minimize overall traffic congestion by decentralized decision-making
  - Adapts signal timing based on local traffic conditions at each intersection
- **Traffic Control System**: 
  - ✅ **COMPLETED & WORKING**: Implemented `--traffic_control` argument for switching between different traffic control methods
  - **Three Control Methods**: `tree_method` (default, Nimrod's algorithm), `actuated` (SUMO gap-based), `fixed` (static timing)
  - **Conditional Object Initialization**: Only loads Nimrod's objects when using tree_method for optimal performance
  - **Experimental Comparison**: Enables A/B testing between different traffic control approaches using identical network conditions
  - **Baseline Evaluation**: SUMO Actuated serves as primary baseline for comparing Nimrod's Tree Method performance
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
- **Reminder**: make sure to periodically update CLAUDE.md and README.md to reflect project developments and improvements