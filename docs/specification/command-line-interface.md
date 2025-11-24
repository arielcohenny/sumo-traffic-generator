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
  --vehicle_types "passenger 90 public 10" \
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

#### `--vehicle_types` (str, default: "passenger 90 public 10")

Vehicle type distribution with percentage assignment.

- **Types**:
  - `passenger`: Cars (5.0m length, 13.9 m/s max speed)
  - `public`: Buses (10.0m length, 11.1 m/s max speed)
- **Format**: Space-separated type-percentage pairs
- **Validation**: Percentages must sum to 100
- **Example**: `"passenger 90 public 10"`

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

**Private traffic seed**: Controls passenger vehicle generation (routes, departure times, vehicle type assignment).

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

#### `--start_time_hour` (float, default: 0.0)

Real-world hour when simulation starts (0-24) for temporal attractiveness.

- **Range**: 0.0-24.0 hours
- **Step**: 0.5 hour increments
- **Constraint**: Only configurable with `uniform` departure pattern; other patterns require 0.0 (midnight)
- **Example**: `--start_time_hour 7.5` (7:30 AM start with uniform pattern)

### Traffic Control Arguments

#### `--traffic_light_strategy` (str, default: "partial_opposites")

Traffic signal phasing strategy for synthetic networks.

- **Strategies**:
  - `partial_opposites`: Straight+right and left+u-turn movements in separate phases (requires 2+ lanes, most realistic)
  - `opposites`: Opposing directions signal together (more efficient)
  - `incoming`: Each edge gets separate phase (more flexible)
- **Example**: `--traffic_light_strategy partial_opposites`

#### `--traffic_control` (str, default: "tree_method")

Dynamic signal control method for managing traffic lights.

- **Methods**:
  - **`tree_method`** (default): Tree Method - Decentralized Bottleneck Prioritization Algorithm
    - Real-time bottleneck detection and traffic signal optimization
    - Decentralized junction-level decision making
    - Research-based algorithm with proven 20-45% travel time improvements
    - See `--tree-method-interval`, `--tree-method-m`, `--tree-method-l` for configuration

  - **`atlcs`**: ATLCS - Adaptive Traffic Light Control System
    - Enhanced bottleneck detection with predictive analysis
    - Junction-level handoff coordination with Tree Method
    - Dynamic congestion-based pricing for signal priority
    - See `--bottleneck-detection-interval`, `--atlcs-interval` for configuration
    - Full documentation: `docs/ATLCS.md`

  - **`rl`**: Reinforcement Learning - Deep Q-Network (DQN) based control
    - ML-based adaptive signal control using trained neural networks
    - Supports both training mode (random exploration) and inference mode (using trained models)
    - Can leverage imitation learning from Tree Method demonstrations
    - See `--rl_model_path`, `--rl-cycle-lengths`, `--rl-cycle-strategy` for configuration
    - Training workflow: `scripts/train_rl_production.py`
    - Full documentation: `docs/RL_IMPLEMENTATION.md`, `docs/IMITATION_LEARNING_GUIDE.md`

  - **`actuated`**: SUMO Actuated - Gap-based vehicle detection control
    - Uses induction loop detectors to extend green phases when vehicles present
    - Industry-standard baseline for traffic signal control
    - Parameters: max-gap=3.0s, min/max duration bounds (10-70s)

  - **`fixed`**: Static Timing - Pre-configured phase durations
    - Uses fixed green/yellow phase durations from traffic light definitions
    - Simplest baseline for experimental comparison
    - Equal green time distribution across all phases

- **Example**: `--traffic_control tree_method`
- **Comparison**: For research comparing control methods, run identical simulations with different `--traffic_control` values
- **Baseline**: `actuated` serves as primary baseline, `fixed` as secondary baseline

#### `--rl_model_path` (str, optional)

Path to trained reinforcement learning model for inference mode.

- **Used with**: `--traffic_control rl`
- **Format**: Path to `.zip` checkpoint file (Stable-Baselines3 format)
- **Behavior**:
  - If provided: RL controller loads model and runs in inference mode
  - If not provided: RL controller runs in training mode with random actions
- **Example**: `--rl_model_path models/checkpoint/rl_traffic_model_410000_steps.zip`
- **Training**: See `scripts/train_rl_production.py` for model training workflow
- **Documentation**: Full details in `docs/RL_IMPLEMENTATION.md` and `docs/IMITATION_LEARNING_GUIDE.md`

#### `--rl-cycle-lengths` (int list, optional)

List of cycle lengths in seconds for reinforcement learning control.

- **Used with**: `--traffic_control rl`
- **Default**: [90] (single fixed cycle length)
- **Format**: Space-separated integers (e.g., `60 90 120`)
- **Behavior**: RL agent can select from these cycle lengths based on `--rl-cycle-strategy`
- **Examples**:
  - Fixed cycle: `--rl-cycle-lengths 90`
  - Variable cycles: `--rl-cycle-lengths 60 90 120`
  - Extended range: `--rl-cycle-lengths 60 75 90 105 120`
- **Note**: More cycle options increase action space complexity

#### `--rl-cycle-strategy` (str, optional)

Strategy for selecting cycle length from `--rl-cycle-lengths` options.

- **Used with**: `--traffic_control rl`
- **Default**: `fixed`
- **Choices**:
  - `fixed`: Always use first cycle length from list
  - `random`: Randomly select cycle length at each decision point
  - `sequential`: Cycle through lengths in order
- **Example**: `--rl-cycle-strategy random`
- **Note**: Only affects behavior when multiple cycle lengths are provided

#### `--tree-method-interval` (int, default: 90)

Tree Method calculation interval in seconds.

- **Range**: 30-300 seconds
- **Step**: 10 second increments
- **Performance**: Lower values = more responsive, higher CPU usage
- **Example**: `--tree-method-interval 60`

#### `--tree-method-m` (float, default: 0.8)

Tree Method M parameter for speed-density relationship in traffic flow calculations.

- **Range**: 0.1-2.0
- **Purpose**: Controls speed-density curve shape in fundamental diagram
- **Default**: 0.8 (from original Tree Method research)
- **Example**: `--tree-method-m 0.9`
- **Note**: Advanced parameter - modify only for traffic flow research

#### `--tree-method-l` (float, default: 2.8)

Tree Method L parameter for speed-density relationship in traffic flow calculations.

- **Range**: 1.0-5.0
- **Purpose**: Controls speed-density curve shape in fundamental diagram
- **Default**: 2.8 (from original Tree Method research)
- **Example**: `--tree-method-l 3.0`
- **Note**: Advanced parameter - modify only for traffic flow research

#### `--log_bottleneck_events` (flag, default: disabled)

Enable logging of vehicle counts per edge to `workspace/bottleneck_events.csv`.

- **No value required**: Presence of flag enables logging
- **Logging Interval**: Every 360 seconds (6 minutes)
- **CSV Format**: `step,link,num_vehicles` (one row per edge at each interval)
- **Purpose**: Track vehicle distribution across network over time for analysis
- **Performance**: Minimal overhead, logs only at 6-minute intervals
- **Example**: `--log_bottleneck_events`
- **Output File**: `workspace/bottleneck_events.csv`
- **Use Cases**:
  - Network congestion analysis
  - Vehicle distribution studies
  - Traffic flow pattern research
  - Bottleneck identification over time

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

# Tree Method with custom flow parameters (traffic flow research)
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control tree_method --tree-method-m 0.9 --tree-method-l 3.0 --seed 42
```

### Advanced Scenarios

```bash
# Large grid simulation
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 6 --num_vehicles 1000 --traffic_control tree_method

# Rush hour simulation (morning peak)
env PYTHONUNBUFFERED=1 python -m src.cli \
  --departure_pattern "rush_hours:7-9:60,17-19:30,rest:10" \
  --start_time_hour 7.0 \
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
