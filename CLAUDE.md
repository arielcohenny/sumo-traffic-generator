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
  --attractiveness hybrid \
  --time_dependent \
  --start_time_hour 7.0 \
  --routing_strategy "shortest 70 realtime 30" \
  --vehicle_types "passenger 70 commercial 20 public 10" \
  --traffic_light_strategy opposites \
  --gui

# Commercial traffic scenario with incoming strategy
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 7 \
  --num_vehicles 500 \
  --vehicle_types "passenger 40 commercial 55 public 5" \
  --routing_strategy "shortest 50 fastest 50" \
  --traffic_light_strategy incoming \
  --gui

# Public transport focused scenario
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --num_vehicles 300 \
  --vehicle_types "passenger 50 commercial 20 public 30" \
  --routing_strategy "shortest 80 realtime 20" \
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
4. **Lane Configuration** (`src/network/lane_counts.py`): Applies configurable lane assignments using realistic, random, or fixed algorithms
5. **Edge Attractiveness** (`src/network/edge_attrs.py`): Computes departure/arrival weights using multiple research-based methods with 4-phase temporal system
6. **Traffic Light Injection** (`src/network/traffic_lights.py`): Adds default four-phase signal plans
7. **Route Generation** (`src/traffic/`): Generates vehicle routes using 4-strategy routing system and 3-type vehicle assignment
8. **Dynamic Simulation** (`src/sim/sumo_controller.py`): Runs SUMO with TraCI integration, Nimrod's algorithm, and real-time phase switching

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

- `HEAD_DISTANCE = 30`: Distance from downstream end when splitting edges
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

## Memory

- This project is a sophisticated SUMO traffic simulation framework with intelligent grid network generation and dynamic traffic control
- Uses a comprehensive 8-step pipeline for network and traffic generation
- Implements advanced algorithms for zone extraction, land use assignment, and traffic routing
- Designed with multiple architectural patterns including Strategy, Adapter, and Pipeline patterns
- Supports reproducible simulations through seeded random generation
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
- **Reminder**: make sure to periodically update CLAUDE.md and README.md to reflect project developments and improvements