# Web GUI Interface

## Overview

The DBPS Web GUI is a Streamlit-based graphical interface that provides an intuitive, web-based interface for configuring and running traffic simulations. The GUI offers:

- **Visual Parameter Configuration**: Interactive widgets for all CLI parameters
- **Real-time Simulation Monitoring**: Progress tracking with live updates
- **Command Generation**: Automatic CLI command generation from GUI parameters
- **Output Visualization**: Integrated display of simulation results and logs
- **Chrome App Mode**: Standalone desktop application experience
- **Parameter Validation**: Real-time validation with error highlighting

## Installation and Setup

### Prerequisites

- **Python 3.8+** with virtual environment support
- **SUMO Traffic Simulator** with GUI support
- **Google Chrome Browser** (recommended for app mode)
- **All DBPS Dependencies** from `requirements.txt`

### Installation Process

```bash
# Install DBPS package with GUI dependencies
pip install -e .

# Verify installation
dbps --help
# Should show: "Launch the DBPS GUI as a desktop application"

# Check dependencies
python -c "import streamlit; print('Streamlit OK')"
python -c "import src.ui.streamlit_app; print('DBPS GUI OK')"
```

### System Requirements

- **RAM**: 2GB minimum, 4GB recommended for large simulations
- **Storage**: 500MB for DBPS + 1GB temporary workspace
- **Network**: Localhost access on port 8501

## Launching the GUI

### Basic Launch

```bash
# Simple launch (recommended)
dbps

# The GUI will:
# 1. Start Streamlit server on localhost:8501
# 2. Launch Chrome in app mode for desktop experience
# 3. Show startup progress and readiness status
# 4. Provide desktop-like window with native controls
```

### Launch Process Details

1. **Server Startup**: Streamlit server initializes on port 8501
2. **Readiness Check**: System waits up to 30 seconds for server readiness
3. **Chrome Launch**: Opens Chrome in `--app` mode for standalone experience
4. **User Interface**: Web-based interface loads with parameter widgets

### Troubleshooting Launch

```bash
# Manual server start (debug mode)
streamlit run src/ui/streamlit_app.py --server.port=8501 --browser.gatherUsageStats=false

# Check port availability
lsof -i :8501

# Browser fallback (if Chrome app mode fails)
open http://localhost:8501
```

## User Interface Components

### Main Layout Structure

#### **Header Section**

- **Application Title**: "DBPS - Decentralised Bottleneck Prioritization Simulation"
- **Status Indicators**: Server connection, parameter validation status
- **Progress Bar**: Real-time simulation progress (when running)

#### **Parameter Configuration Sidebar**

- **Collapsible Sections**: Organized parameter groups with expand/collapse controls
- **Real-time Validation**: Parameter errors highlighted immediately
- **Reset Controls**: Individual section and global parameter reset options

#### **Main Content Area**

- **Command Preview**: Generated CLI command updates in real-time
- **Execution Controls**: Run simulation button with validation status
- **Output Display**: Logs, progress, and results during/after execution

#### **Footer Section**

- **File Management**: Links to generated workspace files
- **Documentation**: Quick links to specification and help
- **Version Information**: DBPS version and build information

### Parameter Configuration Sections

#### 1. **Network Generation Section**

**Purpose**: Configure synthetic network topology and structure

**Grid Dimension**

- **Widget**: Slider with numeric input
- **Range**: 1-20 (default: 5)
- **Display**: Real-time grid size calculation (e.g., "5×5 = 25 intersections")
- **Validation**: Must be positive integer within range

**Block Size**

- **Widget**: Slider with meter unit display
- **Range**: 50-500 meters (default: 200m, step: 25m)
- **Display**: Real-time distance visualization
- **Validation**: Reasonable urban block sizes

**Junctions to Remove**

- **Widget**: Text input with format examples
- **Format**: Integer count or comma-separated junction IDs
- **Examples**: "3" (random removal) or "A1,B2,C3" (specific removal)
- **Validation**: Format checking and junction ID validation

**Lane Count Algorithm**

- **Widget**: Select box with description tooltips
- **Options**:
  - "Realistic" (zone-based demand calculation)
  - "Random" (1-3 lanes randomly assigned)
  - "Fixed" (user-specified count 1-5)
- **Conditional Input**: Fixed count number input appears when "Fixed" selected

**Custom Lane Configuration**

- **Widget**: Expandable text area with syntax highlighting
- **Format**: `EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N`
- **Features**:
  - Live syntax validation
  - Format examples and help text
  - Error highlighting with line numbers
- **Alternative**: File upload option for complex configurations

#### 2. **Traffic Generation Section**

**Purpose**: Configure vehicle behavior and routing strategies

**Vehicle Count**

- **Widget**: Slider with vehicle count display
- **Range**: 1-10,000 (default: 300, step: 50)
- **Display**: Real-time density calculation (vehicles per intersection)
- **Performance Warning**: Shows estimated memory/CPU usage for large counts

**Routing Strategy Configuration**

- **Widget**: Four-component percentage sliders with real-time total
- **Components**:
  - **Shortest**: Static shortest path (slider 0-100%)
  - **Realtime**: 30-second rerouting (slider 0-100%)
  - **Fastest**: 45-second fastest path (slider 0-100%)
  - **Attractiveness**: Multi-criteria routing (slider 0-100%)
- **Validation**: Real-time sum validation (must equal 100%)
- **Display**: Pie chart visualization of strategy distribution

**Vehicle Type Distribution**

- **Widget**: Three-component percentage sliders with vehicle icons
- **Components**:
  - **Passenger**: Cars 🚗 (slider 0-100%)
  - **Commercial**: Trucks 🚛 (slider 0-100%)
  - **Public**: Buses 🚌 (slider 0-100%)
- **Validation**: Real-time sum validation (must equal 100%)
- **Display**: Visual icons and percentage breakdown

**Departure Pattern**

- **Widget**: Select box with custom configuration options
- **Options**:
  - "Six Periods" (research-based daily pattern)
  - "Uniform" (even distribution)
  - "Custom Rush Hours" (opens time configuration)
- **Conditional Inputs**: Additional time configuration widgets based on selection

#### 3. **Simulation Control Section**

**Purpose**: Configure simulation execution parameters

**Random Seed**

- **Widget**: Number input with random generation button
- **Range**: 1-999,999 (default: 42)
- **Features**:
  - "Generate Random" button for new seeds
  - Reproducibility explanation tooltip
- **Validation**: Positive integer within range

**Step Length**

- **Widget**: Slider with time unit display
- **Range**: 1.0-10.0 seconds (default: 1.0s, step: 1.0s)
- **Display**: Real-time frequency calculation (steps per simulation minute)
- **Performance**: Warning for very small step lengths

**Simulation Duration**

- **Widget**: Slider with time unit conversion (conditionally enabled)
- **Range**: 1-86,400 seconds (default: 7200s/2 hours)
- **Display**: Multiple time formats (seconds, minutes, hours)
- **Presets**: Quick selection buttons (30min, 1hr, 2hr, 4hr, 8hr)
- **Constraint**: Only configurable with `uniform` departure pattern; fixed at 86,400s (24h) for other patterns

**Simulation Start Time**

- **Widget**: Time picker with 24-hour format (conditionally enabled)
- **Range**: 0.0-24.0 hours (step: 0.5 hours)
- **Display**: Both 24-hour (14:30) and 12-hour (2:30 PM) formats
- **Integration**: Works with time dependency and departure patterns
- **Constraint**: Only configurable with `uniform` departure pattern; fixed at 0.0 (midnight) for other patterns

**SUMO GUI Option**

- **Widget**: Checkbox with preview thumbnail
- **Description**: "Launch SUMO visualization during simulation"
- **Impact**: Note about performance impact and window management

#### 4. **Zone & Attractiveness Section**

**Purpose**: Configure land use and traffic attraction modeling

**Land Use Block Size**

- **Widget**: Slider with spatial resolution display
- **Range**: 10.0-100.0 meters (default: 25.0m, step: 5.0m)
- **Display**: Grid cell count calculation for current network
- **Performance**: Impact warning for very small block sizes

**Attractiveness Method**

- **Widget**: Select box with method descriptions
- **Options**:
  - "Poisson" (random Poisson distribution)
  - "Land Use" (zone-based calculation)
  - "IAC" (intersection accessibility calculation)
- **Help**: Detailed tooltips explaining each method

**Time Dependency**

- **Widget**: Checkbox with temporal pattern preview
- **Description**: "Apply 4-phase temporal variations (morning peak, midday, evening peak, night)"

#### 5. **Traffic Control Section**

**Purpose**: Configure dynamic signal control algorithms

**Control Method**

- **Widget**: Select box with algorithm descriptions and performance characteristics
- **Options**:
  - "Tree Method" (decentralized bottleneck prioritization)
  - "ATLCS" (adaptive traffic light control with Tree Method coordination)
  - "Actuated" (SUMO gap-based control)
  - "Fixed" (static timing)
- **Performance**: CPU usage and responsiveness indicators for each method

**Tree Method Configuration**

- **Widget**: Slider with performance impact indicator
- **Range**: 30-300 seconds (default: 90s, step: 10s)
- **Display**: Real-time responsiveness vs. efficiency trade-off visualization
- **Recommendations**: Performance guidance based on network size

**ATLCS Configuration** (shown when ATLCS selected)

- **Bottleneck Detection Interval**:
  - **Range**: 30-120 seconds (default: 60s)
  - **Description**: "Enhanced multi-criteria bottleneck detection frequency"
- **Pricing Update Interval**:
  - **Range**: 1-15 seconds (default: 5s)
  - **Description**: "Dynamic congestion pricing calculation frequency"

### Advanced Features

#### **Real-time Parameter Validation**

- **Immediate Feedback**: Parameters validated as user types/selects
- **Error Highlighting**: Invalid parameters shown with red borders
- **Error Messages**: Specific, actionable error descriptions
- **Dependency Checking**: Related parameters validated together (e.g., routing strategy percentages)
- **Range Enforcement**: Sliders and inputs respect min/max bounds automatically

#### **Command Line Preview**

- **Live Generation**: CLI command updates in real-time as parameters change
- **Syntax Highlighting**: Color-coded command structure for readability
- **Copy Button**: One-click copying of generated command
- **Format Options**: Pretty-printed multi-line or single-line formats
- **Parameter Explanation**: Hover tooltips explaining each CLI argument

#### **Parameter Persistence**

- **Session Storage**: Parameters saved automatically during session
- **Configuration Export**: Save parameter sets as JSON files
- **Configuration Import**: Load previously saved parameter configurations
- **URL Parameters**: Shareable URLs with embedded parameter configurations

#### **Progress Monitoring**

- **Pipeline Steps**: Visual progress through 9-step pipeline execution
- **Real-time Updates**: Live progress bar with step descriptions
- **Time Estimates**: Remaining time estimation based on network complexity
- **Cancellation**: Ability to stop execution mid-process

## Output Display System

### Execution Logs

- **Real-time Streaming**: Live log output during simulation execution
- **Log Levels**: Color-coded by severity (INFO, WARNING, ERROR)
- **Filtering**: Filter logs by level, module, or keyword
- **Search**: Text search within log history
- **Download**: Export logs as text files

### Error Handling

- **Error Highlighting**: Critical errors prominently displayed
- **Troubleshooting Hints**: Contextual suggestions for common issues
- **Stack Traces**: Detailed error information for debugging (collapsible)
- **Recovery Options**: Suggested parameter adjustments for error resolution

### Results Summary

- **Simulation Statistics**: Vehicle counts, completion rates, travel times
- **Performance Metrics**: Execution time, memory usage, file sizes
- **Network Analysis**: Junction counts, edge statistics, connectivity metrics
- **Traffic Control Performance**: Algorithm-specific performance indicators

### File Management

- **Workspace Browser**: Visual browser for generated files
- **File Downloads**: Direct download links for SUMO files (network.xml, routes.xml, etc.)
- **File Previews**: In-browser preview for XML and configuration files
- **Cleanup Options**: Workspace cleanup with selective file removal

## Technical Architecture

### Technology Stack

- **Frontend Framework**: Streamlit 1.28.0+ (web-based UI framework)
- **Backend Integration**: Direct Python integration with DBPS pipeline
- **Browser Integration**: Chrome app mode for desktop experience
- **State Management**: Streamlit session state for parameter persistence
- **Real-time Updates**: Streamlit's reactive update system

### File Organization

```
src/ui/
├── streamlit_app.py      # Main Streamlit application entry point
├── parameter_widgets.py  # Parameter widget definitions and validation
├── output_display.py     # Output display and log management
└── __init__.py          # UI module initialization

src/gui.py               # CLI entry point and Chrome launcher
```

### Process Flow

1. **Initialization**: `dbps` command starts Streamlit server
2. **UI Loading**: Web interface loads with default parameters
3. **Parameter Configuration**: User modifies parameters via widgets
4. **Validation**: Real-time parameter validation and error checking
5. **Execution**: Pipeline execution with progress monitoring
6. **Results**: Output display with logs, statistics, and file access

### Integration with CLI Pipeline

- **Shared Validation**: Same validation functions as CLI
- **Identical Execution**: Same pipeline steps and algorithms
- **Parameter Mapping**: Direct mapping between GUI widgets and CLI arguments
- **Configuration Export**: GUI parameters convertible to CLI commands

## Browser Compatibility

### Recommended Browsers

- **Chrome/Chromium**: Full support, app mode available
- **Firefox**: Full support, standard browser window
- **Safari**: Full support, standard browser window
- **Edge**: Full support, standard browser window

### App Mode Features (Chrome)

- **Standalone Window**: Appears as desktop application
- **Native Controls**: Standard window controls (minimize, maximize, close)
- **No Browser UI**: No address bar or browser buttons
- **Desktop Integration**: Appears in dock/taskbar as DBPS application

### Responsive Design

- **Desktop Optimized**: Primary focus on desktop/laptop usage
- **Minimum Resolution**: 1024×768 recommended
- **Sidebar Responsive**: Parameter sidebar adapts to window width
- **Mobile Friendly**: Basic functionality on tablet/mobile (not primary use case)

## Performance Considerations

### Client-Side Performance

- **Browser Requirements**: Modern browser with JavaScript enabled
- **Memory Usage**: ~50MB browser memory for interface
- **CPU Usage**: Minimal (interface rendering only)
- **Network Usage**: Minimal (local server communication)

### Server-Side Performance

- **Memory Usage**: Streamlit ~100MB + DBPS pipeline memory requirements
- **CPU Usage**: GUI overhead ~5% of total simulation CPU usage
- **I/O Usage**: File watching for real-time updates
- **Port Usage**: Default port 8501 (configurable)

### Large Network Handling

- **Parameter Limits**: Same as CLI (up to 10,000 vehicles)
- **Progress Updates**: Efficient progress reporting without performance impact
- **Log Streaming**: Circular buffer prevents memory overflow for long simulations
- **File Management**: Lazy loading for large output files

## Troubleshooting

### Launch Issues

#### Chrome Not Found

```bash
# Error: Could not launch Chrome app mode
# Solution: Verify Chrome installation
which "google-chrome" || which "chrome" || which "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

#### Port Already in Use

```bash
# Error: Address already in use
# Solution: Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9
dbps  # Retry launch
```

#### Streamlit Import Error

```bash
# Error: ModuleNotFoundError: No module named 'streamlit'
# Solution: Install GUI dependencies
pip install streamlit>=1.28.0
```

### Runtime Issues

#### Parameter Validation Errors

- **Cause**: Invalid parameter combinations or ranges
- **Solution**: Check error messages and adjust highlighted parameters
- **Example**: Routing strategy percentages not summing to 100%

#### Simulation Execution Errors

- **Cause**: SUMO not found, workspace permissions, or invalid network configurations
- **Solution**: Check error logs, verify SUMO installation, ensure workspace is writable
- **Debug**: Enable debug logging with `DBPS_DEBUG=1`

#### Browser Connectivity Issues

- **Cause**: Firewall blocking localhost connections
- **Solution**: Allow localhost connections on port 8501
- **Alternative**: Use standard browser window instead of app mode

### Debug Mode

```bash
# Enable comprehensive debug logging
DBPS_DEBUG=1 dbps

# Manual Streamlit start with debug
DBPS_DEBUG=1 streamlit run src/ui/streamlit_app.py --logger.level=debug
```

### Log Analysis

- **GUI Logs**: Streamlit server logs (stdout/stderr)
- **Pipeline Logs**: DBPS execution logs (within GUI output display)
- **Browser Console**: Client-side JavaScript errors (F12 developer tools)
- **SUMO Logs**: SUMO execution logs (displayed in GUI output section)

## Configuration Options

### Environment Variables

```bash
# Server configuration
export DBPS_GUI_PORT=8502        # Change default port
export DBPS_GUI_HOST=0.0.0.0     # Allow external connections (security risk)
export DBPS_DEBUG=1              # Enable debug logging

# Browser configuration
export DBPS_BROWSER=firefox      # Use alternative browser
export DBPS_NO_APP_MODE=1        # Disable Chrome app mode
```

### Advanced Settings

- **Custom CSS**: Modify `src/ui/styles.css` for appearance customization
- **Widget Configuration**: Modify `src/ui/parameter_widgets.py` for widget behavior
- **Progress Intervals**: Adjust progress update frequency in configuration

## Security Considerations

### Local Server Security

- **Localhost Only**: Default binding to localhost:8501 (not accessible externally)
- **No Authentication**: Designed for single-user local use
- **File Access**: Limited to DBPS workspace directory
- **Process Isolation**: Runs with user permissions only

### Network Security

- **No External Dependencies**: All communication with local Streamlit server
- **No Data Transmission**: No external network requests during normal operation
- **Configuration Privacy**: Parameter configurations stored locally only

### Best Practices

- **Firewall Configuration**: Ensure port 8501 not exposed externally
- **User Permissions**: Run with standard user account (not administrator)
- **Workspace Isolation**: Keep workspace directory separate from sensitive files
