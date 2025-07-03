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