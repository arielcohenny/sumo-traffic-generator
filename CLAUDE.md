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

# Full parameter example with 4-phase temporal system
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --junctions_to_remove 0 \
  --lane_count realistic \
  --num_vehicles 300 \
  --seed 42 \
  --step-length 1.0 \
  --end-time 3600 \
  --attractiveness hybrid \
  --time_dependent \
  --start_time_hour 7.0 \
  --gui

# Rush hour analysis (morning peak)
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 7 \
  --num_vehicles 500 \
  --end-time 7200 \
  --time_dependent \
  --start_time_hour 7.5 \
  --gui

# Full day simulation (24 hours)
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --num_vehicles 1000 \
  --end-time 86400 \
  --time_dependent \
  --start_time_hour 0.0
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
7. **Route Generation** (`src/traffic/`): Generates vehicle routes using shortest-path computation
8. **Dynamic Simulation** (`src/sim/sumo_controller.py`): Runs SUMO with TraCI integration, Nimrod's algorithm, and real-time phase switching

### Key Modules

**Core Orchestration**:

- `src/cli.py`: Main entry point and pipeline orchestration
- `src/config.py`: Central configuration using dataclasses

**Network Generation**:

- `src/network/`: All network manipulation and generation logic
- `src/network/edge_attrs.py`: Advanced edge attractiveness with 4-phase temporal support
- Uses SUMO's netgenerate, netconvert tools
- Validates network structure at each step

**Traffic Generation**:

- `src/traffic/builder.py`: Main traffic generation orchestrator
- `src/traffic/routing.py`: Shortest path routing using Dijkstra
- `src/traffic/xml_writer.py`: SUMO XML file generation

**Simulation Control**:

- `src/sim/sumo_controller.py`: TraCI wrapper with per-step callback API and 4-phase temporal switching
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
- `MIN_LANES = 1`, `MAX_LANES = 3`: Lane count bounds for randomization and realistic algorithm
- `LAMBDA_DEPART = 3.5`, `LAMBDA_ARRIVE = 2.0`: Poisson distribution parameters for edge attractiveness
- `DEFAULT_JUNCTION_RADIUS = 10.0`: Junction radius in meters

**4-Phase Temporal System Configuration**:

- **Morning Peak** (6:00-9:30): Depart multiplier 1.4, Arrive multiplier 0.7
- **Midday Off-Peak** (9:30-16:00): Depart multiplier 1.0, Arrive multiplier 1.0
- **Evening Peak** (16:00-19:00): Depart multiplier 0.7, Arrive multiplier 1.5
- **Night Low** (19:00-6:00): Depart multiplier 0.4, Arrive multiplier 0.4

**Time Mapping**: 1:1 ratio (1 simulation second = 1 real-world second)

**Lane Count Algorithm Parameters**:
- Land use weights: Mixed (3.0), Employment (2.5), Entertainment/Retail (2.5), Public Buildings (2.0), Residential (1.5), Public Open Space (1.0)
- Lane assignment thresholds: ≤1.5 (1 lane), 1.5-2.5 (2 lanes), >2.5 (3 lanes)
- Perimeter modifier: +1 lane for boundary edges (max 3 total)

**Edge Attractiveness Parameters**:
- **Gravity model**: d_param=0.95, g_param=1.02, base_random~Normal(1.0,0.3)
- **4-Phase multipliers**: Morning peak (1.4/0.7), Midday off-peak (1.0/1.0), Evening peak (0.7/1.5), Night low (0.4/0.4)
- **Land use multipliers**: Residential (0.8/1.4), Employment (1.3/0.9), Mixed (1.1/1.1), etc.

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

## Lane Count Algorithms

The system supports three algorithms for determining the number of lanes on each edge. Users can select the algorithm via the `--lane_count` command line argument.

### Algorithm Types:

#### 1. **realistic** (Default)
- **Description**: Zone-based traffic demand calculation using land use types and attractiveness values
- **Method**: 
  1. Loads zone data from generated GeoJSON file
  2. Finds zones adjacent to each edge using spatial analysis
  3. Calculates weighted demand score based on zone types and attractiveness
  4. Assigns lane count (1-3) based on demand level with perimeter modifiers
- **Land Use Weights**:
  - **Mixed**: 3.0 (highest traffic generation - mixed commercial/residential)
  - **Employment**: 2.5 (high peak traffic - office/industrial areas)
  - **Entertainment/Retail**: 2.5 (high commercial traffic - shopping/entertainment)
  - **Public Buildings**: 2.0 (moderate institutional traffic)
  - **Residential**: 1.5 (moderate residential traffic)
  - **Public Open Space**: 1.0 (lower recreational traffic)
- **Usage**: `--lane_count realistic`
- **Lane Assignment Logic**:
  - Score ≤ 1.5: 1 lane
  - Score 1.5-2.5: 2 lanes  
  - Score > 2.5: 3 lanes
  - Perimeter edges get +1 lane (max 3) to handle boundary traffic
- **Characteristics**: Data-driven, realistic traffic patterns based on urban planning
- **Best for**: Realistic simulations, urban planning studies, scenarios with meaningful land use

#### 2. **random**
- **Description**: Random assignment within defined bounds (original algorithm)
- **Method**: Each edge gets a random lane count between MIN_LANES and MAX_LANES
- **Parameters**:
  - `MIN_LANES = 1`
  - `MAX_LANES = 3`
- **Usage**: `--lane_count random`
- **Characteristics**: Unpredictable variety, good for stress testing
- **Best for**: Testing network robustness, scenarios without specific land use requirements

#### 3. **fixed** (Integer value)
- **Description**: Uniform lane count across all edges
- **Method**: Sets every edge to the specified number of lanes
- **Usage**: `--lane_count <integer>` (e.g., `--lane_count 2`)
- **Valid range**: 1-3 lanes
- **Characteristics**: Uniform, predictable, simplified analysis
- **Best for**: Baseline comparisons, simplified scenarios, capacity studies

### Implementation Details:

- **Zone Integration**: Realistic algorithm uses zone data from `data/zones.geojson`
- **Spatial Analysis**: Uses Shapely geometry operations to find edge-zone adjacency
- **Validation**: All algorithms include validation to ensure proper lane assignment
- **Reproducibility**: Random algorithm respects the `--seed` parameter
- **Network Consistency**: Body and head edge segments maintain consistent lane counts

### Command Line Usage:

```bash
# Realistic algorithm (default)
env PYTHONUNBUFFERED=1 python -m src.cli --lane_count realistic

# Random algorithm
env PYTHONUNBUFFERED=1 python -m src.cli --lane_count random

# Fixed lane count
env PYTHONUNBUFFERED=1 python -m src.cli --lane_count 2
env PYTHONUNBUFFERED=1 python -m src.cli --lane_count 3
```

### Algorithm Selection Guidelines:

- **Use `realistic`** for: Studies requiring realistic traffic patterns, urban planning simulations, research with meaningful land use data
- **Use `random`** for: Network robustness testing, baseline scenarios, stress testing different lane configurations  
- **Use `fixed`** for: Simplified analysis, capacity studies, baseline comparisons, debugging network issues

## Edge Attractiveness Methods

The system supports multiple algorithms for calculating edge attractiveness (departure and arrival weights). Users can select the base method via the `--attractiveness` command line argument and optionally apply temporal modifiers.

### Base Methods:

#### 1. **poisson** (Default)
- **Description**: Uses Poisson distribution for random attractiveness values
- **Parameters**: 
  - `LAMBDA_DEPART = 3.5` (departure attractiveness)
  - `LAMBDA_ARRIVE = 2.0` (arrival attractiveness)
- **Usage**: `--attractiveness poisson`
- **Characteristics**: Simple, fast, purely random distribution
- **Best for**: Baseline simulations, testing, scenarios without specific land use patterns

#### 2. **land_use**
- **Description**: Adjusts attractiveness based on land use zone types using multipliers
- **Method**: Applies zone-specific multipliers to base Poisson values
- **Land Use Multipliers**:
  - **Residential**: Departure 0.8, Arrival 1.4 (people leave for work, return home)
  - **Employment**: Departure 1.3, Arrival 0.9 (work locations generate trips)
  - **Mixed**: Departure 1.1, Arrival 1.1 (balanced commercial/residential)
  - **Entertainment/Retail**: Departure 0.7, Arrival 1.3 (destinations for leisure)
  - **Public Buildings**: Departure 0.9, Arrival 1.0 (moderate institutional traffic)
  - **Public Open Space**: Departure 0.6, Arrival 0.8 (recreational destinations)
- **Usage**: `--attractiveness land_use`
- **Characteristics**: Realistic traffic patterns based on urban planning principles
- **Best for**: Realistic simulations, urban planning studies

#### 3. **gravity**
- **Description**: Implements gravity model based on distance and cluster size
- **Formula**: `attractiveness = (d_param^distance) × (g_param^cluster_size) × base_random`
- **Parameters**:
  - `d_param = 0.95` (distance decay parameter)
  - `g_param = 1.02` (cluster growth parameter)
  - `base_random`: Normal distribution (μ=1.0, σ=0.3)
- **Usage**: `--attractiveness gravity`
- **Characteristics**: Spatially-aware, considers network topology
- **Best for**: Studies focusing on spatial relationships and accessibility

#### 4. **iac** (Integrated Attraction Coefficient)
- **Description**: Sophisticated method from research literature combining multiple factors
- **Formula**: `IAC = d^(-δ) × g^(γ) × θ × m_rand × f_spatial`
- **Components**:
  - Distance factor: `d^(-δ)` where d=0.95
  - Cluster factor: `g^(γ)` where g=1.02
  - Base attractiveness: `θ` (normal distribution)
  - Random mood: `m_rand` (temporal variation)
  - Spatial preference: `f_spatial` (connectivity-based)
- **Usage**: `--attractiveness iac`
- **Characteristics**: Most sophisticated, research-based, multiple influencing factors
- **Best for**: Academic research, detailed behavioral studies

#### 5. **hybrid**
- **Description**: Combines Poisson base with spatial and land use adjustments
- **Method**: 
  1. Generate base Poisson values
  2. Apply land use multipliers (reduced impact vs. pure land_use)
  3. Apply spatial connectivity adjustment
- **Usage**: `--attractiveness hybrid`
- **Characteristics**: Balances randomness with realistic patterns
- **Best for**: General-purpose simulations requiring both randomness and realism

### Temporal Modifiers:

#### **--time_dependent** (4-Phase Temporal System)
- **Description**: Applies research-based 4-phase time-of-day variations to any base method
- **Can be combined with**: Any base method (poisson, land_use, gravity, iac, hybrid)
- **Research-Based 4-Phase System**:
  - **Morning Peak (6:00-9:30)**: Depart ×1.4, Arrive ×0.7 (High outbound: home→work)
  - **Midday Off-Peak (9:30-16:00)**: Depart ×1.0, Arrive ×1.0 (Balanced baseline)
  - **Evening Peak (16:00-19:00)**: Depart ×0.7, Arrive ×1.5 (High inbound: work→home)
  - **Night Low (19:00-6:00)**: Depart ×0.4, Arrive ×0.4 (Minimal activity)
- **Implementation**: 
  - Pre-calculates attractiveness profiles for all 4 phases
  - Real-time phase switching during simulation
  - 1:1 time mapping (1 sim second = 1 real-world second)
- **Additional Parameter**: `--start_time_hour` (0-24) to set simulation start time
- **Characteristics**: Bimodal traffic patterns with realistic rush hour behavior
- **Best for**: Full-day simulations, rush hour analysis, realistic traffic studies

### Implementation Details:

- **Zone Data Integration**: Methods `land_use` and `hybrid` use zone data from `zones.poly.xml`
- **Spatial Analysis**: Methods `gravity`, `iac`, and `hybrid` analyze network topology
- **Temporal Integration**: `--time_dependent` flag works with any base method
- **Reproducibility**: All methods respect the `--seed` parameter for consistent results
- **Validation**: Each method includes validation to ensure reasonable attractiveness ranges

### Command Line Usage:

```bash
# Base methods without time dependency
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness poisson
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness land_use
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness gravity
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness iac
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness hybrid

# Base methods WITH 4-phase temporal system
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness land_use --time_dependent --start_time_hour 7.0
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness hybrid --time_dependent --start_time_hour 16.5
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness gravity --time_dependent --start_time_hour 0.0
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness iac --time_dependent --start_time_hour 12.0

# Even simple methods benefit from 4-phase temporal system
env PYTHONUNBUFFERED=1 python -m src.cli --attractiveness poisson --time_dependent --start_time_hour 8.0
```

### Method Combinations:

The modular design allows for powerful combinations:

- **`--attractiveness land_use --time_dependent --start_time_hour 7.0`**: Realistic land use patterns with morning rush hour start
- **`--attractiveness hybrid --time_dependent --start_time_hour 0.0`**: Most comprehensive approach for full-day simulation
- **`--attractiveness gravity --time_dependent --start_time_hour 16.5`**: Spatial accessibility with evening rush hour analysis
- **`--attractiveness iac --time_dependent --start_time_hour 12.0`**: Research-based model with midday baseline

## Detailed Module Documentation

[... rest of the existing content remains the same ...]

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
- **Advanced Features**:
  - Zone-based traffic demand calculation using land use types and attractiveness values
  - Spatial analysis for edge-zone adjacency detection
  - Research-based land use multipliers and gravity model parameters
  - Modular architecture allowing combination of spatial, temporal, and land use factors
- **Nimrod's Traffic Control Algorithm**: 
  - Uses Nimrod's decentralized traffic control algorithm for dynamic signal optimization
  - Implements intelligent signal control through tree-based method
  - Dynamically adjusts traffic light phases based on real-time traffic bottlenecks
  - Aims to minimize overall traffic congestion by decentralized decision-making
  - Adapts signal timing based on local traffic conditions at each intersection