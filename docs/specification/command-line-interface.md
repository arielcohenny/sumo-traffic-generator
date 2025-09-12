# Command Line Interface

## Overview

The DBPS Command Line Interface (CLI) provides direct access to all simulation parameters through command-line arguments. The CLI is the primary execution interface for automated scripts, batch processing, and programmatic access to the SUMO Traffic Generator pipeline.

## Basic Usage

### Simple Execution

```bash
# Basic 5x5 grid with default parameters (300 vehicles, 2-hour simulation)
env PYTHONUNBUFFERED=1 python -m src.cli

# Basic execution with SUMO GUI
env PYTHONUNBUFFERED=1 python -m src.cli --gui
```

### Advanced Configuration

```bash
# Advanced synthetic network with Tree Method control
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 7 \
  --num_vehicles 800 \
  --traffic_control tree_method \
  --routing_strategy "shortest 70 realtime 30" \
  --vehicle_types "passenger 60 commercial 30 public 10" \
  --gui

# Tree Method research sample
env PYTHONUNBUFFERED=1 python -m src.cli \
  --tree_method_sample evaluation/datasets/networks/ \
  --traffic_control tree_method \
  --end-time 7200
```

## Command Line Arguments

### Network Generation Arguments

#### `--grid_dimension` (int, default: 5)

Defines the grid's number of rows and columns for synthetic network generation.

- **Range**: 1-20
- **Example**: `--grid_dimension 7` creates a 7×7 grid

#### `--block_size_m` (int, default: 200)

Sets block size in meters for grid network generation.

- **Range**: 50-500 meters
- **Step**: 25 meter increments
- **Example**: `--block_size_m 150` creates 150m×150m city blocks

#### `--junctions_to_remove` (str, default: "0")

Number of junctions to remove or comma-separated list of specific junction IDs.

- **Formats**:
  - Integer: `"5"` (remove 5 random junctions)
  - Specific IDs: `"A0,B1,C2"` (remove specific junctions)

#### `--lane_count` (str, default: "realistic")

Sets the lane assignment algorithm.

- **Options**:
  - `"realistic"`: Zone-based demand calculation
  - `"random"`: Randomized within bounds (1-3 lanes)
  - Integer value: Fixed count for all edges (1-5)
- **Example**: `--lane_count realistic`

### Traffic Generation Arguments

#### `--num_vehicles` (int, default: 300)

Total number of vehicles to generate for the simulation.

- **Range**: 1-10,000
- **Step**: 50 vehicle increments recommended
- **Example**: `--num_vehicles 800`

#### `--routing_strategy` (str, default: "shortest 100")

Vehicle routing behavior with percentage mixing of four strategies.

- **Strategies**:
  - `shortest`: Static shortest path calculation
  - `realtime`: 30-second dynamic rerouting
  - `fastest`: 45-second fastest path rerouting
  - `attractiveness`: Multi-criteria routing
- **Format**: Space-separated strategy-percentage pairs
- **Validation**: Percentages must sum to 100
- **Examples**:
  - `"shortest 100"` (all shortest path)
  - `"shortest 70 realtime 30"` (70% shortest, 30% realtime)
  - `"shortest 50 realtime 30 fastest 20"` (mixed strategies)

#### `--vehicle_types` (str, default: "passenger 60 commercial 30 public 10")

Vehicle type distribution with percentage assignment.

- **Types**:
  - `passenger`: Cars (5.0m length, 13.9 m/s max speed)
  - `commercial`: Trucks (12.0m length, 10.0 m/s max speed)
  - `public`: Buses (10.0m length, 11.1 m/s max speed)
- **Format**: Space-separated type-percentage pairs
- **Validation**: Percentages must sum to 100
- **Example**: `"passenger 70 commercial 20 public 10"`

#### `--departure_pattern` (str, default: "uniform")

Vehicle departure timing pattern.

- **Patterns**:
  - `uniform`: Even distribution across simulation time (default)
  - `six_periods`: Research-based daily structure (Morning 20%, Morning Rush 30%, Noon 25%, Evening Rush 20%, Evening 4%, Night 1%)
  - `rush_hours:7-9:40,17-19:30,rest:10`: Custom rush hour definition
- **Example**: `--departure_pattern uniform`

### Simulation Control Arguments

#### `--seed` (int, optional)

**Single seed mode (backward compatible)**: Sets all seeds (network, private-traffic, public-traffic) to the same value.

- **Range**: 1-999,999
- **Default**: Random seed generated if not provided
- **Example**: `--seed 42`
- **Note**: Cannot be used with individual seed parameters

#### `--network-seed` (int, optional)

**Network structure seed**: Controls network generation aspects (junction removal, lane assignment, land use, edge attractiveness).

- **Range**: 1-999,999
- **Default**: Uses `--seed` value or random seed if not provided
- **Example**: `--network-seed 42`

#### `--private-traffic-seed` (int, optional)

**Private traffic seed**: Controls passenger and commercial vehicle generation (routes, departure times, vehicle type assignment).

- **Range**: 1-999,999
- **Default**: Uses `--seed` value or random seed if not provided
- **Example**: `--private-traffic-seed 123`

#### `--public-traffic-seed` (int, optional)

**Public traffic seed**: Controls public transportation vehicle generation (routes, departure times, vehicle type assignment).

- **Range**: 1-999,999
- **Default**: Uses `--seed` value or random seed if not provided
- **Example**: `--public-traffic-seed 456`

#### `--step-length` (float, default: 1.0)

Simulation step length in seconds for TraCI control loop.

- **Range**: 1.0-10.0 seconds
- **Step**: 1.0 second increments
- **Example**: `--step-length 1.0`

#### `--end-time` (int, default: 7200)

Simulation duration in seconds.

- **Range**: 1-86,400 seconds (24 hours)
- **Default**: 7200 seconds (2 hours)
- **Step**: 3600 second (1 hour) increments recommended
- **Constraint**: Only configurable with `uniform` departure pattern; other patterns require 86,400s (24 hours)
- **Example**: `--end-time 3600` (1 hour simulation with uniform pattern)

#### `--gui` (flag)

Launch SUMO's built-in GUI for traffic visualization.

- **No value required**: Presence of flag enables GUI
- **Example**: `--gui`

#### `--workspace` (str, default: ".")

Parent directory where 'workspace' folder will be created for simulation output files.

- **Default**: "." (creates './workspace/' in current directory)
- **Behavior**: Always creates a 'workspace/' subdirectory in the specified parent directory
- **Safety**: Only the 'workspace/' folder contents are cleaned, never the parent directory
- **Examples**:
  - `--workspace .` creates './workspace/' (same as default)
  - `--workspace /home/user/experiments` creates '/home/user/experiments/workspace/'
  - `--workspace results` creates 'results/workspace/'
- **Generated Files**: All SUMO files (network, routes, config, statistics) are placed in the workspace folder

### Zone & Attractiveness Arguments

#### `--land_use_block_size_m` (float, default: 25.0)

Zone cell size in meters for land use generation.

- **Range**: 10.0-100.0 meters
- **Step**: 5.0 meter increments
- **Purpose**: Controls resolution of land use zone generation
- **Default**: 25.0m (research paper methodology)
- **Example**: `--land_use_block_size_m 30.0`

#### `--attractiveness` (str, default: "land_use")

Departure and arrival attractiveness calculation method.

- **Methods**:
  - `land_use`: Zone-based land use attractiveness (default)
  - `poisson`: Random Poisson distribution
  - `iac`: Intersection accessibility calculation
- **Example**: `--attractiveness land_use`

#### `--time_dependent` (flag)

Apply 4-phase temporal variations to attractiveness values.

- **Phases**: Morning peak, midday, evening peak, night
- **Requires**: `--start_time_hour` for proper phase calculation
- **Example**: `--time_dependent --start_time_hour 7.0`

#### `--start_time_hour` (float, default: 0.0)

Real-world hour when simulation starts (0-24) for temporal attractiveness.

- **Range**: 0.0-24.0 hours
- **Step**: 0.5 hour increments
- **Used with**: `--time_dependent` flag
- **Constraint**: Only configurable with `uniform` departure pattern; other patterns require 0.0 (midnight)
- **Example**: `--start_time_hour 7.5` (7:30 AM start with uniform pattern)

### Traffic Control Arguments

#### `--traffic_light_strategy` (str, default: "opposites")

Traffic signal phasing strategy for synthetic networks.

- **Strategies**:
  - `opposites`: Opposing directions signal together (more efficient)
  - `incoming`: Each edge gets separate phase (more flexible)
- **Example**: `--traffic_light_strategy opposites`

#### `--traffic_control` (str, default: "tree_method")

Dynamic signal control method.

- **Methods**:
  - `tree_method`: Tree Method (Decentralized Bottleneck Prioritization Algorithm)
  - `atlcs`: ATLCS (Adaptive Traffic Light Control System with Tree Method coordination)
  - `actuated`: SUMO gap-based control
  - `fixed`: Static timing from configuration
- **Example**: `--traffic_control tree_method`

#### `--tree-method-interval` (int, default: 90)

Tree Method calculation interval in seconds.

- **Range**: 30-300 seconds
- **Step**: 10 second increments
- **Performance**: Lower values = more responsive, higher CPU usage
- **Example**: `--tree-method-interval 60`

#### `--bottleneck-detection-interval` (int, default: 60)

ATLCS enhanced bottleneck detection interval in seconds.

- **Range**: 30-120 seconds
- **Used with**: `--traffic_control atlcs`
- **Features**: Advanced metrics, predictive analysis, multi-criteria assessment
- **Example**: `--bottleneck-detection-interval 45`

#### `--atlcs-interval` (int, default: 5)

ATLCS dynamic pricing update interval in seconds.

- **Range**: 1-15 seconds
- **Used with**: `--traffic_control atlcs`
- **Features**: Congestion-based pricing, signal priority calculation
- **Example**: `--atlcs-interval 3`

### Advanced Configuration Arguments

#### `--custom_lanes` (str, optional)

Custom lane definitions for specific edges in synthetic networks.

- **Format**: `"EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=..."`
- **Examples**:
  - Full: `"A1B1=tail:2,head:B1B0:1,B1C1:2"`
  - Tail-only: `"A1B1=tail:2"`
  - Dead-end: `"A1B1=tail:2,head:"`
- **Constraints**: Synthetic networks only, 1-3 lanes

#### `--custom_lanes_file` (str, optional)

File containing custom lane definitions.

- **Format**: Same as `--custom_lanes`, one per line
- **Features**: Comments supported (# prefix), UTF-8 encoding
- **Mutually exclusive**: With `--custom_lanes`

#### `--tree_method_sample` (str, optional)

Path to pre-built Tree Method sample files for validation.

- **Purpose**: Test against original research networks
- **Behavior**: Bypasses Steps 1-8, goes to simulation directly
- **Requirements**: Folder with `network.net.xml`, `vehicles.trips.xml`, `simulation.sumocfg.xml`
- **Example**: `--tree_method_sample evaluation/datasets/networks/`

## Usage Examples

### Development and Testing

```bash
# Quick development test (30 minutes)
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 200 --end-time 1800 --gui

# Stress test scenario
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 10 --num_vehicles 2000 --end-time 7200

# Custom lane configuration test
env PYTHONUNBUFFERED=1 python -m src.cli --custom_lanes "A1B1=tail:3,head:B1C1:2,B1B0:1" --gui
```

### Research and Validation

```bash
# Tree Method validation with original research network
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method

# Compare traffic control methods (identical conditions)
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 800 --seed 42 --traffic_control tree_method
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 800 --seed 42 --traffic_control actuated
env PYTHONUNBUFFERED=1 python -m src.cli --num_vehicles 800 --seed 42 --traffic_control fixed

# ATLCS performance testing
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control atlcs --bottleneck-detection-interval 30 --atlcs-interval 2
```

### Advanced Scenarios

```bash
# Large grid simulation
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 6 --num_vehicles 1000 --traffic_control tree_method

# Rush hour simulation (morning peak)
env PYTHONUNBUFFERED=1 python -m src.cli \
  --departure_pattern "rush_hours:7-9:60,17-19:30,rest:10" \
  --start_time_hour 7.0 \
  --time_dependent \
  --num_vehicles 1500
```

### Multiple Seed Examples (Advanced Control)

#### Network Sensitivity Analysis

```bash
# Test different network topologies with identical traffic
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 100 --private-traffic-seed 42 --public-traffic-seed 42 --num_vehicles 500
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 101 --private-traffic-seed 42 --public-traffic-seed 42 --num_vehicles 500
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 102 --private-traffic-seed 42 --public-traffic-seed 42 --num_vehicles 500
```

#### Private Traffic Pattern Studies

```bash
# Same network and public transport, different private traffic patterns
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 200 --num_vehicles 800
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 101 --public-traffic-seed 200 --num_vehicles 800
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 102 --public-traffic-seed 200 --num_vehicles 800
```

#### Public Transport Planning

```bash
# Same network and private traffic, different public transport scenarios
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 300 --num_vehicles 600
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 301 --num_vehicles 600
env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --private-traffic-seed 100 --public-traffic-seed 302 --num_vehicles 600
```

#### Experimental Research Setup

```bash
# Controlled experiment with all aspects independent
env PYTHONUNBUFFERED=1 python -m src.cli \
  --network-seed 1001 \
  --private-traffic-seed 2001 \
  --public-traffic-seed 3001 \
  --grid_dimension 5 \
  --num_vehicles 1000 \
  --traffic_control tree_method \
  --end-time 7200
```

## Integration with GUI

### Command Generation

The Web GUI automatically generates equivalent CLI commands for all parameter configurations, enabling:

- **Script Export**: Copy CLI commands from GUI for automation
- **Reproducibility**: Share exact parameter configurations
- **Batch Processing**: Convert GUI configurations to scripted workflows

### Parameter Validation

Both CLI and GUI use the same validation system:

- **Consistent Error Messages**: Same validation rules and error reporting
- **Range Enforcement**: Identical parameter bounds and constraints
- **Cross-Parameter Validation**: Same dependency checking logic

### Execution Pipeline

CLI and GUI use identical execution pipelines:

- **Same Steps**: Both interfaces execute the same 9-step pipeline
- **Same Outputs**: Identical workspace files and results
- **Same Performance**: No execution overhead difference

## Error Handling

### Validation Errors

```bash
# Example validation error
python -m src.cli --routing_strategy "shortest 60 realtime 30"
# Error: Routing strategy percentages must sum to 100, got 90

# Correct usage
python -m src.cli --routing_strategy "shortest 60 realtime 40"
```

### File Errors

```bash
# Invalid grid dimension
python -m src.cli --grid_dimension 0
# Error: grid_dimension must be between 1 and 20

# Custom lanes file error
python -m src.cli --custom_lanes_file invalid.txt
# Error: Custom lanes file syntax error on line 3: invalid format
```

### Range Errors

```bash
# Invalid parameter range
python -m src.cli --grid_dimension 25
# Error: grid_dimension must be between 1 and 20, got 25
```

## Performance Considerations

### Recommended Configurations

- **Small Networks**: `--grid_dimension 3-5`, `--num_vehicles 100-500`
- **Medium Networks**: `--grid_dimension 5-8`, `--num_vehicles 500-1500`
- **Large Networks**: `--grid_dimension 8-15`, `--num_vehicles 1500-5000`

### Memory Usage

- **Vehicle Count**: ~1MB per 1000 vehicles
- **Grid Size**: ~10MB per 100 edges
- **Tree Method Samples**: Varies by network complexity

### CPU Usage

- **Tree Method**: Higher CPU usage with lower intervals
- **ATLCS**: Most CPU-intensive with frequent updates
- **Actuated/Fixed**: Lowest CPU usage

## Troubleshooting

### Common Issues

```bash
# SUMO not found
export SUMO_HOME=/usr/local/opt/sumo/share/sumo
export PATH=$PATH:$SUMO_HOME/bin

# Permission errors
chmod +w workspace/

# Python path issues
export PYTHONPATH=$PYTHONPATH:/path/to/sumo-traffic-generator
```

### Debug Mode

```bash
# Enable debug logging
export DBPS_DEBUG=1
env PYTHONUNBUFFERED=1 python -m src.cli --gui
```
