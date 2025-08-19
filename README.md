# SUMO Traffic Generator

A comprehensive Python-based SUMO traffic simulation framework with intelligent signal control, supporting synthetic grids and Tree Method research data sets.

## Key Features

- **Dual Network Support**: Synthetic grids and Tree Method research datasets
- **Intelligent Traffic Control**: Tree Method decentralized algorithm for dynamic signal optimization
- **Advanced Traffic Generation**: Multi-strategy routing, vehicle types, and temporal patterns
- **Configurable Lane Assignment**: Flow-based lane allocation with realistic traffic demand
- **Research-Grade Evaluation**: Statistical benchmarks and performance analysis framework

## Installation

```bash
# Clone repository
git clone https://github.com/arielcohenny/sumo-traffic-generator.git
cd sumo-traffic-generator

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and GUI support
pip install -r requirements.txt
pip install -e .  # Installs 'dbps' command for GUI

# Verify installation
dbps --help  # Should show GUI launch options
```

## Quick Start

### Web GUI Interface (Recommended)

```bash
# Launch the visual web interface
dbps

# The GUI provides:
# - Interactive parameter configuration
# - Real-time simulation monitoring
# - Automatic CLI command generation
# - Integrated results visualization
```

### Command Line Interface

#### Synthetic Grid Network

```bash
# Basic 5x5 grid with 500 vehicles
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 500 --gui

# Advanced configuration with Tree Method control
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 7 \
  --num_vehicles 800 \
  --traffic_control tree_method \
  --routing_strategy "shortest 70 realtime 30" \
  --gui
```

### Tree Method Research Datasets

```bash
# Validate against original Tree Method research networks
env PYTHONUNBUFFERED=1 python -m src.cli \
  --tree_method_sample evaluation/datasets/networks/ \
  --traffic_control tree_method \
  --gui
```

## GUI Interfaces

DBPS provides two complementary GUI interfaces for different use cases:

### 1. Web GUI Interface (`dbps` command)

**Purpose**: Interactive parameter configuration and simulation management  
**Technology**: Streamlit-based web interface with Chrome app mode

```bash
# Launch the web GUI (recommended for most users)
dbps
```

**Features**:

- **Visual Parameter Configuration**: Interactive widgets for all 20+ parameters
- **Real-time Validation**: Parameter errors highlighted immediately
- **Command Generation**: Automatic CLI command generation for scripting
- **Progress Monitoring**: Live simulation progress with step-by-step updates
- **Results Visualization**: Integrated display of logs, statistics, and outputs
- **Chrome App Mode**: Desktop-like application window experience

**Interface Sections**:

- **Network Generation**: Grid size, block dimensions, junction removal, lane configuration
- **Traffic Parameters**: Vehicle count, routing strategies, vehicle types, departure patterns
- **Simulation Control**: Duration, step length, random seed, GUI options
- **Zone & Attractiveness**: Land use modeling, temporal patterns, attractiveness methods
- **Traffic Control**: Algorithm selection (Tree Method, ATLCS, Actuated, Fixed) with parameters

### 2. SUMO GUI Integration (`--gui` flag)

**Purpose**: Real-time traffic visualization during simulation execution  
**Technology**: SUMO's built-in traffic visualization system

```bash
# Launch CLI with SUMO visualization
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 500 --gui

# Or enable via Web GUI checkbox
# Check "SUMO GUI" option in the web interface
```

**Features**:

- **Real-time Traffic Visualization**: Live display of vehicles, traffic flow, and congestion
- **Network Topology**: Visual representation of roads, intersections, and lane configurations
- **Signal State Monitoring**: Traffic light phases and timing with algorithm interventions
- **Performance Analysis**: Visual identification of bottlenecks and traffic patterns
- **Algorithm Visualization**: Tree Method and ATLCS decision displays

**Use Cases**:

- **Development**: Visual verification of algorithm behavior
- **Research**: Traffic pattern analysis and performance validation
- **Education**: Demonstration of traffic flow dynamics
- **Debugging**: Visual identification of routing or network issues

### Dual GUI Workflow (Recommended)

```bash
# 1. Launch web GUI for parameter configuration
dbps

# 2. Configure parameters using visual widgets
# 3. Enable "SUMO GUI" checkbox for visualization
# 4. Click "Run Simulation" button
# 5. Both GUIs work together:
#    - Web GUI: Parameter control and progress monitoring
#    - SUMO GUI: Real-time traffic visualization
```

**Benefits of Dual GUI**:

- **Easy Configuration**: Web GUI for intuitive parameter setup
- **Rich Visualization**: SUMO GUI for detailed traffic analysis
- **Progress Tracking**: Web GUI shows pipeline progress and logs
- **Visual Validation**: SUMO GUI confirms expected traffic behavior

## Project Structure

```
├── src/                    # Core application code
│   ├── network/           # Network generation and processing
│   ├── traffic/           # Vehicle routing and generation
│   ├── orchestration/     # High-level simulation coordination
│   └── sumo_integration/  # SUMO/TraCI interface layer
├── evaluation/            # Research validation framework
│   ├── benchmarks/        # Performance comparison studies
│   └── datasets/          # Research networks
├── tests/                 # Software testing framework
├── tools/                 # Development utilities
└── workspace/             # Generated simulation files (temporary)
```

## Documentation

- **Technical Specification**: See [SPECIFICATION.md](SPECIFICATION.md) for complete technical details, parameters, and implementation documentation
- **Research & Benchmarks**: See [evaluation/](evaluation/) for performance studies, datasets, and experimental framework

## License

This project is licensed under the MIT License.
