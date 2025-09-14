# SUMO GUI Integration

## Overview

DBPS integrates with SUMO's built-in GUI for real-time traffic visualization during simulation execution. The SUMO GUI provides comprehensive visual representation of traffic flow, signal states, and network topology, complementing DBPS's traffic control algorithms with detailed visualization capabilities.

## Purpose and Benefits

### Traffic Visualization

- **Real-time Vehicle Movement**: Live visualization of individual vehicles and traffic flow
- **Network Topology Display**: Visual representation of road network, intersections, and lane configurations
- **Signal State Monitoring**: Real-time display of traffic light phases and timing
- **Performance Analysis**: Visual identification of bottlenecks, congestion, and traffic patterns

### Research and Development

- **Algorithm Validation**: Visual verification of Tree Method and ATLCS performance
- **Parameter Tuning**: Immediate visual feedback when adjusting traffic control parameters
- **Debugging Support**: Visual identification of routing issues, vehicle conflicts, or network problems
- **Educational Tool**: Demonstration and understanding of traffic flow dynamics

## Activation Methods

### Command Line Interface

```bash
# Basic SUMO GUI activation
env PYTHONUNBUFFERED=1 python -m src.cli --gui

# Advanced configuration with GUI
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 7 \
  --num_vehicles 800 \
  --traffic_control tree_method \
  --gui

# Tree Method sample with GUI visualization
env PYTHONUNBUFFERED=1 python -m src.cli \
  --tree_method_sample evaluation/datasets/networks/ \
  --traffic_control tree_method \
  --gui
```

### Web GUI Interface

1. **Launch Web GUI**: Run `dbps` command to open Streamlit interface
2. **Configure Parameters**: Set simulation parameters using web widgets
3. **Enable SUMO GUI**: Check "SUMO GUI" checkbox in Simulation Control section
4. **Execute Simulation**: Click "Run Simulation" button
5. **SUMO Window Opens**: SUMO GUI window appears automatically when simulation starts

## SUMO GUI Features and Controls

### Main Window Components

#### **Network View**

- **Road Network**: Visual display of streets, intersections, and lane markings
- **Zoom and Pan**: Mouse controls for navigation (scroll wheel zoom, click-drag pan)
- **View Modes**: Different visualization modes for network elements
- **Grid Overlay**: Optional coordinate grid for spatial reference

#### **Vehicle Display**

- **Individual Vehicles**: Color-coded vehicles by type (passenger/commercial/public)
- **Vehicle Information**: Click vehicles to see ID, route, speed, and status
- **Traffic Density**: Visual representation of congestion levels
- **Route Visualization**: Display of vehicle paths and destinations

#### **Traffic Signal Display**

- **Signal States**: Real-time traffic light colors (red, yellow, green)
- **Phase Timing**: Current phase duration and remaining time
- **Signal Plans**: Display of signal timing configuration
- **Algorithm Control**: Visual indication of Tree Method/ATLCS interventions

### Control Panel Features

#### **Simulation Control**

- **Play/Pause**: Start, stop, or pause simulation execution
- **Speed Control**: Adjust simulation playback speed (0.1x to 100x real-time)
- **Step-by-Step**: Execute simulation one time step at a time
- **Time Display**: Current simulation time and elapsed duration

#### **View Options**

- **Vehicle Names**: Toggle display of vehicle IDs
- **Edge Names**: Toggle display of road/edge identifiers
- **Junction Names**: Toggle display of intersection identifiers
- **Traffic Light IDs**: Toggle display of signal identifiers

#### **Statistics Panel**

- **Vehicle Counts**: Running, waiting, arrived, departed vehicle statistics
- **Performance Metrics**: Average speed, travel time, throughput measures
- **Network Status**: Edge occupancy, junction throughput, signal efficiency

### Visualization Modes

#### **Standard View**

- **Default Visualization**: Standard road network with moving vehicles
- **Color Coding**: Vehicles colored by type, roads by speed limit or occupancy
- **Traffic Lights**: Standard traffic signal display with phase colors

#### **Traffic Density View**

- **Congestion Visualization**: Roads colored by traffic density (green=free, red=congested)
- **Bottleneck Identification**: Visual highlighting of congested areas
- **Flow Patterns**: Directional arrows showing traffic flow intensity

#### **Speed Visualization**

- **Speed Coloring**: Vehicles colored by current speed (red=slow, green=fast)
- **Average Speeds**: Road segments colored by average traffic speed
- **Speed Limits**: Reference display of maximum allowed speeds

#### **Algorithm Visualization**

- **Tree Method Indicators**: Visual highlighting of Tree Method interventions
- **ATLCS Displays**: Congestion pricing and bottleneck detection indicators
- **Control Actions**: Visual feedback when algorithms modify signal timing

## Integration Architecture

### TraCI Connection

- **Real-time Control**: SUMO GUI connects via TraCI (Traffic Control Interface)
- **Bidirectional Communication**: DBPS algorithms send commands, SUMO provides feedback
- **Synchronization**: GUI display synchronized with DBPS algorithm execution
- **Performance Impact**: Minimal overhead for visualization updates

### Data Exchange

- **Vehicle States**: Position, speed, route, type information
- **Network Status**: Edge occupancy, travel times, traffic densities
- **Signal States**: Current phases, remaining durations, algorithm modifications
- **Statistics**: Real-time performance metrics and counters

### File Integration

- **Network Files**: GUI loads same network.xml as DBPS pipeline
- **Route Files**: GUI displays same vehicle routes as generated by DBPS
- **Configuration**: GUI uses same SUMO configuration as DBPS execution

## Performance Considerations

### System Requirements

- **Graphics**: Dedicated graphics card recommended for large networks
- **Memory**: Additional ~200MB RAM for GUI rendering
- **CPU**: ~10-15% additional CPU usage for visualization updates
- **Display**: Minimum 1024×768 resolution, 1920×1080 recommended

### Performance Impact

- **Simulation Speed**: 10-30% slower execution when GUI enabled
- **Network Size**: Impact scales with number of vehicles and network complexity
- **Update Frequency**: GUI updates every simulation step (default 1 second intervals)
- **Memory Usage**: Scales with network size and vehicle count

### Optimization Settings

```bash
# Reduce GUI update frequency for better performance
env PYTHONUNBUFFERED=1 python -m src.cli --step-length 2.0 --gui

# Disable some GUI features for performance
# (configured within SUMO GUI settings menu)
```

## Algorithm Visualization

### Tree Method Integration

- **Bottleneck Detection**: Visual highlighting of detected bottleneck intersections
- **Phase Extensions**: Visual indication when Tree Method extends green phases
- **Priority Calculations**: Color-coded display of intersection priorities
- **Decision Tree**: Optional display of Tree Method decision logic

### ATLCS Integration

- **Congestion Pricing**: Color-coded display of dynamic pricing levels
- **Bottleneck Predictions**: Visual indicators of predicted bottleneck formation
- **Phase Adjustments**: Real-time display of ATLCS phase modifications
- **Coordination Display**: Visual representation of Tree Method-ATLCS coordination

### Actuated Control Visualization

- **Detector States**: Display of loop detector activations and measurements
- **Gap Measurements**: Visual representation of vehicle gap detection
- **Phase Extensions**: Display of actuated phase extension decisions
- **Queue Detection**: Visual indication of queue formation and clearance

## Usage Scenarios

### Development and Testing

```bash
# Quick visual verification during development
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --gui

# Algorithm comparison with visualization
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control tree_method --gui
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control actuated --gui
```

### Research and Analysis

```bash
# Large-scale performance analysis
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 10 --num_vehicles 2000 --gui

# Large grid network analysis
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 8 --num_vehicles 1000 --gui
```

### Demonstration and Education

```bash
# Rush hour traffic demonstration
env PYTHONUNBUFFERED=1 python -m src.cli \
  --departure_pattern "rush_hours:7-9:60,17-19:30,rest:10" \
  --gui

# Traffic control method comparison
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control tree_method --seed 42 --gui
env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control fixed --seed 42 --gui
```

## Advanced Features

### Custom Visualization

- **Configuration Files**: Modify SUMO GUI settings via XML configuration
- **Color Schemes**: Customize vehicle and network colors
- **Display Options**: Configure which elements are visible by default
- **View Presets**: Save and restore specific view configurations

### Data Export

- **Screenshots**: Export simulation snapshots at specific time points
- **Video Recording**: Record simulation execution as video files
- **Trajectory Data**: Export vehicle movement data for external analysis
- **Network Statistics**: Export real-time statistics data

### Integration with External Tools

- **Matplotlib Integration**: Export data for Python-based analysis
- **CSV Export**: Performance statistics in spreadsheet format
- **XML Output**: Detailed simulation results in structured format

## Troubleshooting

### Common Issues

#### SUMO GUI Won't Launch

```bash
# Check SUMO installation
sumo-gui --version

# Verify SUMO_HOME environment variable
echo $SUMO_HOME
export SUMO_HOME=/usr/local/opt/sumo/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

#### Graphics Performance Issues

- **Solution 1**: Reduce simulation speed in SUMO GUI
- **Solution 2**: Disable detailed vehicle visualization
- **Solution 3**: Use smaller network or fewer vehicles for testing
- **Solution 4**: Close other graphics-intensive applications

#### GUI Connection Errors

```bash
# Check TraCI connection
# Error: "Could not connect to SUMO via TraCI"
# Solution: Verify SUMO GUI process is running and accessible
```

#### Display Issues

- **Resolution Problems**: Increase screen resolution or adjust SUMO GUI window size
- **Color Issues**: Modify SUMO GUI color scheme in settings
- **Text Rendering**: Adjust font sizes in SUMO GUI preferences

### Debug Mode

```bash
# Enable SUMO GUI debug output
env SUMO_GUI_DEBUG=1 python -m src.cli --gui

# Verbose TraCI communication logging
env TRACI_DEBUG=1 python -m src.cli --gui
```

### Performance Monitoring

```bash
# Monitor system resources during GUI execution
top -p $(pgrep sumo-gui)

# Check memory usage
ps -o pid,ppid,cmd,%mem,vsz --sort=-%mem | grep sumo
```

## Integration with Web GUI

### Dual GUI Mode

- **Web GUI + SUMO GUI**: Both interfaces can run simultaneously
- **Parameter Configuration**: Use Web GUI for parameter setup
- **Visualization**: Use SUMO GUI for traffic visualization
- **Coordination**: Both GUIs reflect same simulation state

### Workflow Integration

1. **Setup**: Configure parameters in Web GUI (`dbps` command)
2. **Enable**: Check "SUMO GUI" option in Web GUI
3. **Execute**: Start simulation from Web GUI
4. **Visualize**: SUMO GUI opens automatically showing traffic
5. **Monitor**: Use Web GUI for progress, SUMO GUI for visualization

### Information Flow

- **Web GUI → SUMO GUI**: Parameter configuration and execution control
- **SUMO GUI → Web GUI**: Simulation progress and completion status
- **Shared State**: Both GUIs access same simulation files and status

## Technical Specifications

### SUMO Version Compatibility

- **Minimum Version**: SUMO 1.16.0
- **Recommended Version**: SUMO 1.19.0 or later
- **GUI Components**: sumo-gui, TraCI, netconvert, netgenerate

### File Format Support

- **Network Files**: .net.xml (SUMO network format)
- **Route Files**: .rou.xml (SUMO route format)
- **Configuration**: .sumocfg (SUMO configuration format)
- **Additional Files**: Traffic light definitions, vehicle types, etc.

### Communication Protocols

- **TraCI**: Traffic Control Interface for real-time communication
- **XML**: Configuration and data file format
- **TCP/IP**: Network communication between DBPS and SUMO GUI

### Platform Support

- **Windows**: Full support with SUMO Windows installation
- **macOS**: Full support with SUMO macOS installation
- **Linux**: Full support with SUMO Linux installation
- **Docker**: Containerized support with X11 forwarding for GUI display
