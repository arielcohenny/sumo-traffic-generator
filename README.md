SUMO traffic simulation framework with real-world OpenStreetMap (OSM) support

This project is a comprehensive Python-based SUMO traffic generator that creates dynamic traffic simulations with intelligent signal control. It supports both synthetic orthogonal grid networks and real-world OpenStreetMap (OSM) data, with configurable lane assignments and Tree Method decentralized traffic control algorithm for dynamic signal optimization.

## Key Features

### Network Support
- **Synthetic Grid Networks**: Orthogonal n×n grids with configurable topology
- **OpenStreetMap Integration**: Real-world street networks from OSM data
- **Universal Edge Processing**: Same algorithms work for both synthetic and real networks

### Core Pipeline

1. **Network Generation/Import**: 
   - Synthetic: Creates orthogonal grid using SUMO's netgenerate 
   - OSM: Imports real street networks with comprehensive netconvert parameters
   - Supports both 1-lane baseline generation and complex real-world topologies

2. **Configurable Land Use Zone Generation**: Dual-mode zone system optimized for both network types:

   **For OpenStreetMap Networks**: Intelligent zone generation with advanced analysis methods
   - **Real OSM Data**: Extracts actual land use tags from OpenStreetMap (residential, commercial, industrial, education, healthcare)
   - **Building Analysis**: Converts OSM building polygons into traffic generation zones with capacity estimation
   - **POI Integration**: Incorporates points of interest (shops, restaurants, schools, hospitals) as zone enhancers
   - **Intelligent Inference**: When OSM data insufficient (<15% coverage), combines network topology analysis, accessibility metrics, and infrastructure analysis
   - **Lane Count Analysis**: Uses lane counts to distinguish commercial/arterial roads from residential streets
   - **Score-Based Classification**: Multi-factor scoring system determines optimal zone type for each grid cell

   **For Synthetic Networks**: Traditional zone extraction with configurable resolution
   - **Junction-Based Zones**: Creates zones based on network topology using proven cellular grid methodology
   - **Research-Based Land Use**: Six land use types with clustering algorithm (Residential, Mixed, Employment, Public Buildings, Public Open Space, Entertainment/Retail)
   - **Variety & Realism**: Generates diverse zone types with appropriate colors and attractiveness values
   
   **Unified Configuration**: `--land_use_block_size_m` parameter controls grid resolution for both network types (default 200m)

3. **Integrated Edge Splitting with Flow-Based Lane Assignment**: Unified edge splitting and lane configuration into a single optimized process:

   - **Flow-Based Splitting**: Edges split at dynamic head distance (min(50m, edge_length/3)) for both synthetic and real networks
   - **Movement Preservation**: All traffic movements (left, right, straight, u-turn) preserved through sophisticated lane mapping
   - **Spatial Logic**: Lane assignments follow real-world patterns (right→right lanes, left→left lanes, straight→middle)
   - **Three Lane Algorithms**: Realistic (zone-based demand), Random (within bounds), Fixed (uniform count)
   - **Real-World Compatibility**: Handles dead-end streets, irregular intersections, and complex urban topologies

4. **Edge Attractiveness Modeling**: Multiple research-based methods for computing departure/arrival weights:

   - **Poisson**: Original distribution approach (λ_depart=3.5, λ_arrive=2.0)
   - **Land Use**: Zone-type multipliers (Residential, Employment, Mixed, etc.)
   - **Gravity**: Network centrality and spatial distance factors
   - **IAC**: Integrated Attraction Coefficient combining multiple factors
   - **Hybrid**: Weighted combination of spatial and land use approaches
   - **4-Phase Temporal**: Research-based time-of-day variations with bimodal traffic patterns

5. Route Generation: Built a vehicle‑route prototype that leverages edge attractiveness and shortest‑path computation; compatible with SUMO’s randomTrips.py for scalable trip creation. All vehicles starting at time 0.

6. SUMO Configuration Authoring – Automatically generates a .sumocfg that wires together the network, route, additional files, and simulation parameters in a single ready‑to‑run configuration.

7. TraCI Runtime Integration – Introduced sim/sumo_controller.py, a thin wrapper around TraCI that launches SUMO (GUI or headless), advances the simulation, and exposes a per‑step callback API for custom control logic.

8. Tree Method’s Tree‑Method Control – Integrated the decentralized‑traffic‑bottlenecks library: the pipeline now converts the network to a JSON tree, builds Tree Method’s Graph, computes an optimal phase map each step, and applies it via TraCI—enabling fully dynamic, decentralized signal control during the simulation.

### OpenStreetMap (OSM) Integration

- **Real-World Networks**: Import street networks directly from OpenStreetMap data
- **Comprehensive Import**: Uses 14 specialized netconvert parameters for urban networks
- **Signal Preservation**: Maintains original OSM traffic light IDs and timing
- **Algorithm Compatibility**: All existing features work seamlessly with real street topology
- **Manhattan Testing**: Successfully validated with 4.6MB Manhattan East Village data
- **Performance Metrics**: 96% departure rate, 63% completion rate on real Manhattan streets

## Installation

````bash
# 1. Clone this repo
git clone https://github.com/arielcohenny/sumo-traffic-generator.git
cd sumo-traffic-generator

# 2. Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt


## Usage & Parameters

### Synthetic Grid Networks
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension <int>        # Number of rows/columns (default: 5)
  --block_size_m <float>        # Block length in meters (default: 200)
  --junctions_to_remove <int|string>  # Internal junctions to delete: integer count or comma-separated IDs (default: 0)
  --lane_count <str|int>        # Lane assignment algorithm: 'realistic' (default, zone-based), 'random', or integer (fixed count for all edges)
  --num_vehicles <int>          # Total trips to generate (default: 300)
  --seed <int>                  # RNG seed (optional)
  --step-length <float>         # Simulation step length in seconds (default: 1.0)
  --end-time <int>              # Total simulation duration in seconds (default: 86400 - 24 hours/full day)
  --attractiveness <str>        # Edge attractiveness method: 'poisson' (default), 'land_use', 'gravity', 'iac', or 'hybrid'
  --time_dependent              # Apply 4-phase time-of-day variations to the selected attractiveness method
  --start_time_hour <float>     # Real-world hour when simulation starts (0-24, default: 0.0 for midnight)
  --routing_strategy <str>      # Routing strategy with percentages (default: 'shortest 100')
  --vehicle_types <str>         # Vehicle types with percentages (default: 'passenger 60 commercial 30 public 10')
  --land_use_block_size_m <float> # Land use zone grid resolution in meters (default: 200.0)
  --traffic_light_strategy <str> # Traffic light phasing strategy: 'opposites' (default) or 'incoming'
  --traffic_control <str>       # Traffic control method: 'tree_method' (default), 'actuated', or 'fixed'
  --gui                         # Launch SUMO in GUI mode (sumo-gui) instead of headless sumo
````

### OpenStreetMap (OSM) Networks
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --osm_file <path>             # Path to OSM file (replaces grid generation)
  --num_vehicles <int>          # Total trips to generate (default: 300)
  --seed <int>                  # RNG seed (optional)
  --step-length <float>         # Simulation step length in seconds (default: 1.0)
  --end-time <int>              # Total simulation duration in seconds (default: 86400 - 24 hours/full day)
  --attractiveness <str>        # Edge attractiveness method: 'poisson' (default), 'land_use', 'gravity', 'iac', or 'hybrid'
  --time_dependent              # Apply 4-phase time-of-day variations to the selected attractiveness method
  --start_time_hour <float>     # Real-world hour when simulation starts (0-24, default: 0.0 for midnight)
  --routing_strategy <str>      # Routing strategy with percentages (default: 'shortest 100')
  --vehicle_types <str>         # Vehicle types with percentages (default: 'passenger 60 commercial 30 public 10')
  --land_use_block_size_m <float> # Land use zone grid resolution in meters (default: 200.0)
  --traffic_control <str>       # Traffic control method: 'tree_method' (default), 'actuated', or 'fixed'
  --gui                         # Launch SUMO in GUI mode (sumo-gui) instead of headless sumo

# Examples with verified working OSM samples:
# Manhattan Upper West Side (300/300 vehicles successful)
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/manhattan_upper_west.osm --num_vehicles 300 --end-time 3600 --gui

# San Francisco Downtown (298/300 vehicles successful)
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/sf_downtown.osm --num_vehicles 300 --traffic_control tree_method --gui

# Washington DC Downtown (300/300 vehicles successful)
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/dc_downtown.osm --num_vehicles 300 --traffic_control tree_method --gui

# Traffic control comparison on verified OSM data
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/manhattan_upper_west.osm --num_vehicles 300 --traffic_control tree_method --seed 42
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/manhattan_upper_west.osm --num_vehicles 300 --traffic_control actuated --seed 42
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/manhattan_upper_west.osm --num_vehicles 300 --traffic_control fixed --seed 42

# Intelligent zone generation with different resolutions
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --land_use_block_size_m 100 --gui   # Fine-grained zones
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file src/osm/samples/sf_downtown.osm --num_vehicles 300 --land_use_block_size_m 150 --gui  # OSM with custom resolution
````

## Parameter Reference

### Core Network Parameters

#### `--osm_file <path>` (OSM networks only)

**Purpose:** Import real-world street network from OpenStreetMap data instead of generating synthetic grid.

- **Format:** Path to .osm file (e.g., `src/osm/export.osm`)
- **Features:** Automatically processes complex urban topologies including dead-end streets, irregular intersections, and existing traffic signals
- **Compatibility:** Works with all traffic generation and control features
- **Verified Working Samples:**
  - `--osm_file src/osm/samples/manhattan_upper_west.osm` → Manhattan Upper West Side (300/300 vehicles)
  - `--osm_file src/osm/samples/sf_downtown.osm` → San Francisco Downtown (298/300 vehicles)
  - `--osm_file src/osm/samples/dc_downtown.osm` → Washington DC Downtown (300/300 vehicles)

### Core Network Parameters (Synthetic Grids)

#### `--grid_dimension <int>` (default: 5)

**Purpose:** Defines the size of the orthogonal grid network.

- **Format:** Single integer (e.g., `5` creates a 5×5 grid)
- **Range:** Minimum 3 for functional networks
- **Impact:** Larger grids create more complex networks but increase computation time
- **Examples:**
  - `--grid_dimension 3` → 3×3 grid (9 junctions)
  - `--grid_dimension 7` → 7×7 grid (49 junctions)

#### `--block_size_m <float>` (default: 200)

**Purpose:** Sets the distance between adjacent junctions in meters.

- **Format:** Floating-point number in meters
- **Range:** 50-1000m typical for urban scenarios
- **Impact:** Affects vehicle travel times and network density
- **Examples:**
  - `--block_size_m 100` → Compact city blocks (dense urban)
  - `--block_size_m 300` → Large suburban blocks

#### `--junctions_to_remove <int|string>` (default: 0)

**Purpose:** Creates network disruptions by removing internal junctions.

- **Format Options:**
  - **Integer:** Random removal count (e.g., `1`, `3`)
  - **List:** Specific junction IDs (e.g., `"A1,B2,C3"`)
- **Impact:** Tests network resilience and routing adaptability
- **Examples:**
  - `--junctions_to_remove 0` → Perfect grid (no disruptions)
  - `--junctions_to_remove 2` → Remove 2 random junctions
  - `--junctions_to_remove "A1,C3"` → Remove specific junctions

#### `--land_use_block_size_m <float>` (default: 200.0)

**Purpose:** Controls the resolution of zone generation for both network types.

- **Format:** Floating-point number in meters
- **Range:** 50-500m typical for different analysis scales
- **Dual-Mode Behavior:**
  - **OSM Networks:** Grid resolution for intelligent zone analysis (topology + accessibility + infrastructure)
  - **Synthetic Networks:** Subdivision size for traditional zone extraction (independent of junction spacing)
- **Zone System Features:**
  - **For OSM:** Uses actual land use tags when available, intelligent inference when data insufficient
  - **For Synthetic:** Creates diverse zone types (Residential, Mixed, Employment, Public Buildings, etc.)
  - **Lane Count Analysis:** Higher lane counts indicate commercial areas, lower counts suggest residential
  - **Research-Based:** Six land use types with clustering algorithm and appropriate attractiveness values
- **Examples:**
  - `--land_use_block_size_m 75` → Fine-grained zones (75m×75m cells)
  - `--land_use_block_size_m 150` → Medium resolution zones 
  - `--land_use_block_size_m 300` → Coarse-grained zones (300m×300m cells)

### Traffic Generation Parameters

#### `--num_vehicles <int>` (default: 300)

**Purpose:** Total number of vehicles to generate for the simulation.

- **Range:** 50-2000+ depending on network size and duration
- **Impact:** Higher counts create more realistic traffic but slower simulation
- **Guidelines:**
  - **Light traffic:** 200-500 vehicles
  - **Medium traffic:** 500-1000 vehicles
  - **Heavy traffic:** 1000-1500+ vehicles

#### `--vehicle_types <str>` (default: "passenger 60 commercial 30 public 10")

**Purpose:** Defines the mix of different vehicle types with percentages.

- **Format:** `"type1 percentage1 type2 percentage2 type3 percentage3"`
- **Vehicle Types:**
  - **`passenger`:** Standard cars (length: 5m, max speed: 50km/h)
  - **`commercial`:** Trucks/delivery (length: 12m, max speed: 40km/h)
  - **`public`:** Buses (length: 10m, max speed: 35km/h)
- **Constraint:** Percentages must sum to 100
- **Examples:**
  - `"passenger 90 commercial 8 public 2"` → Car-heavy scenario
  - `"passenger 40 commercial 55 public 5"` → Industrial area

#### `--departure_pattern <str>` (default: "six_periods")

**Purpose:** Controls when vehicles enter the simulation throughout the day.

- **Options:**
  - **`six_periods`:** Research-based 6-period temporal system
  - **`uniform`:** Even distribution across simulation time
  - **`rush_hours:7-9:40,17-19:30,rest:10`:** Custom rush hour definition
  - **`hourly:7:25,8:35,rest:5`:** Hour-by-hour weight specification
- **Impact:** Creates realistic traffic flow patterns vs artificial uniform loading

### Simulation Control Parameters

#### `--step-length <float>` (default: 1.0)

**Purpose:** Simulation timestep in seconds for TraCI control loop.

- **Range:** 0.1-5.0 seconds typical
- **Impact:** Smaller values = more precise control, slower simulation
- **Recommendations:**
  - **Precise control:** 0.5-1.0 seconds
  - **Fast simulation:** 2.0-5.0 seconds

#### `--end-time <int>` (default: 86400)

**Purpose:** Total simulation duration in seconds.

- **Common Values:**
  - **1800:** 30 minutes (quick tests)
  - **3600:** 1 hour (standard tests)
  - **7200:** 2 hours (rush hour analysis)
  - **86400:** 24 hours (full day simulation)

#### `--start_time_hour <float>` (default: 0.0)

**Purpose:** Real-world hour when simulation begins (0-24).

- **Impact:** Affects time-dependent attractiveness and departure patterns
- **Examples:**
  - `--start_time_hour 7.0` → Start at 7am (morning rush)
  - `--start_time_hour 17.0` → Start at 5pm (evening rush)

#### `--seed <int>` (optional)

**Purpose:** Random number generator seed for reproducible results.

- **Usage:** Omit for random behavior, specify for consistent results
- **Example:** `--seed 42` → Always generates identical scenarios

#### `--gui`

**Purpose:** Launch SUMO with graphical interface instead of headless mode.

- **Impact:** Enables visual monitoring but slows simulation
- **Recommended:** For development, debugging, and demonstrations

### Advanced Traffic Parameters

#### `--routing_strategy <str>` (default: "shortest 100")

**Purpose:** Defines how vehicles choose routes with percentage-based mixing.

- **Strategies:**
  - **`shortest`:** Static shortest path (distance-based)
  - **`realtime`:** Dynamic GPS-style routing (reroutes every 30s)
  - **`fastest`:** Time-based routing (reroutes every 45s)
  - **`attractiveness`:** Multi-criteria routing considering destinations
- **Format:** `"strategy1 percentage1 strategy2 percentage2"`
- **Constraint:** Percentages must sum to 100
- **Examples:**
  - `"shortest 100"` → All vehicles use static shortest paths
  - `"shortest 70 realtime 30"` → Mixed static/dynamic routing
  - `"shortest 25 realtime 25 fastest 25 attractiveness 25"` → All strategies

#### `--attractiveness <str>` (default: "poisson")

**Purpose:** Method for calculating edge departure/arrival attractiveness weights.

- **Methods:**
  - **`poisson`:** Statistical distribution (λ_depart=3.5, λ_arrive=2.0)
  - **`land_use`:** Zone-type based multipliers (residential, commercial, etc.)
  - **`gravity`:** Network centrality and spatial distance factors
  - **`iac`:** Integrated Attraction Coefficient (combines multiple factors)
  - **`hybrid`:** Weighted combination of land use + spatial + Poisson
- **Impact:** Affects where vehicles start/end trips, creating realistic flow patterns

#### `--time_dependent`

**Purpose:** Apply 4-phase time-of-day variations to attractiveness patterns.

- **Phases:**
  - **Morning Peak (6:00-9:30):** High outbound traffic (home→work)
  - **Midday Off-Peak (9:30-16:00):** Balanced baseline traffic
  - **Evening Peak (16:00-19:00):** High inbound traffic (work→home)
  - **Night Low (19:00-6:00):** Minimal activity
- **Impact:** Creates realistic daily traffic rhythm vs static patterns

### Network Configuration Parameters

#### `--lane_count <str|int>` (default: "realistic")

**Purpose:** Algorithm for assigning lane counts to network edges.

- **Options:**
  - **`realistic`:** Zone-based demand calculation (1-3 lanes)
  - **`random`:** Random assignment between min/max bounds
  - **`<integer>`:** Fixed lane count for all edges (e.g., `2`)
- **Impact:** Affects network capacity and traffic flow characteristics

#### `--traffic_light_strategy <str>` (default: "opposites")

**Purpose:** Traffic signal phasing strategy at intersections.

- **Strategies:**
  - **`opposites`:** Opposing directions move together (North-South, then East-West)
  - **`incoming`:** Each incoming edge gets individual phase
- **Impact:** `opposites` = efficient green time, `incoming` = conflict-free but more phases

### Lane Count Algorithms

- **`realistic`** (default): Uses land use zones to calculate traffic demand scores and assign 1-3 lanes based on zone types and attractiveness values
- **`random`**: Original random assignment between MIN_LANES (1) and MAX_LANES (3)
- **`<integer>`**: Fixed lane count for all edges (e.g., `--lane_count 2`)

### Attractiveness Methods

- **`poisson`** (default): Poisson distribution with λ_depart=3.5, λ_arrive=2.0
- **`land_use`**: Land use type multipliers (Residential: depart 0.8/arrive 1.4, Employment: 1.3/0.9, Mixed: 1.1/1.1, etc.)
- **`gravity`**: Network centrality-based using gravity model with distance and cluster size factors
- **`iac`**: Integrated Attraction Coefficient combining gravity, land use, and spatial preference factors
- **`hybrid`**: Weighted combination (50% land use + 30% spatial + 20% base Poisson)

### 4-Phase Temporal System

When `--time_dependent` is used, applies research-based 4-phase time-of-day multipliers to any base method:

- **Morning Peak** (6:00-9:30): Depart ×1.4, Arrive ×0.7 (High outbound: home→work)
- **Midday Off-Peak** (9:30-16:00): Depart ×1.0, Arrive ×1.0 (Balanced baseline)
- **Evening Peak** (16:00-19:00): Depart ×0.7, Arrive ×1.5 (High inbound: work→home)
- **Night Low** (19:00-6:00): Depart ×0.4, Arrive ×0.4 (Minimal activity)

The system generates pre-calculated attractiveness profiles for all 4 phases and switches between them in real-time during simulation based on the `--start_time_hour` parameter. This enables both full-day simulations (24 hours) and rush hour analysis with 1:1 time mapping (1 simulation second = 1 real-world second).

### 4-Strategy Routing System

The system supports 4 routing strategies with percentage-based mixing:

- **`shortest`**: Static shortest path routing (default)
- **`realtime`**: Dynamic Waze/Google Maps-style navigation (reroutes every 30 seconds)
- **`fastest`**: Dynamic fastest route based on current travel times (reroutes every 45 seconds)
- **`attractiveness`**: Multi-criteria routing considering destination attractiveness

**Usage Examples:**

```bash
# Default (100% shortest path)
--routing_strategy "shortest 100"

# Mixed strategies
--routing_strategy "shortest 70 realtime 30"

# All 4 strategies
--routing_strategy "shortest 25 realtime 25 fastest 25 attractiveness 25"

# Heavy dynamic routing
--routing_strategy "realtime 80 fastest 20"
```

**Key Features:**

- Percentages must sum to 100 (validated automatically)
- Dynamic strategies use TraCI for real-time rerouting
- Strategies are assigned per-vehicle during route generation
- Integration with existing 4-phase temporal system

### 3-Type Vehicle System

The system supports 3 vehicle types with percentage-based mixing:

- **`passenger`**: Standard passenger cars (default characteristics)
- **`commercial`**: Commercial trucks and delivery vehicles (longer, slower acceleration)
- **`public`**: Public transit buses (largest, specific route behavior)

**Usage Examples:**

```bash
# Default distribution
--vehicle_types "passenger 60 commercial 30 public 10"

# Car-heavy scenario
--vehicle_types "passenger 90 commercial 8 public 2"

# Commercial-heavy scenario (industrial area)
--vehicle_types "passenger 40 commercial 55 public 5"

# Transit-focused scenario
--vehicle_types "passenger 50 commercial 20 public 30"
```

**Key Features:**

- Percentages must sum to 100 (validated automatically)
- Each vehicle type has distinct physical characteristics:
  - **Length**: passenger (5m), commercial (8m), public (12m)
  - **Max Speed**: passenger (50 m/s), commercial (40 m/s), public (35 m/s)
  - **Acceleration**: passenger (2.6 m/s²), commercial (1.8 m/s²), public (1.2 m/s²)
- Vehicle types are assigned per-vehicle during route generation
- Integration with routing strategies and temporal systems

### Traffic Light Strategies

The system supports two traffic light phasing strategies:

- **`opposites`** (default): Opposing directions move together (North-South concurrent, then East-West concurrent)

  - More efficient green time usage for balanced traffic
  - Standard 2-phase operation: concurrent movements reduce conflicts
  - Best for grid networks with similar traffic volumes in opposing directions

- **`incoming`**: Each incoming edge gets its own phase sequence
  - Each approach direction gets individual green time
  - More phases but eliminates all conflicts between directions
  - Better for unbalanced traffic or safety-critical intersections

**Usage Examples:**

```bash
# Default opposites strategy (North-South together, then East-West together)
--traffic_light_strategy opposites

# Individual edge strategy (each direction separate)
--traffic_light_strategy incoming
```

Both strategies work with any lane configuration and are compatible with the Tree Method optimization algorithm.

### Traffic Control Methods

#### `--traffic_control <str>` (default: "tree_method")

**Purpose:** Selects the traffic control algorithm for signal optimization.

- **Options:**
  - **`tree_method`** (default): Uses Tree Method's Tree Method for decentralized traffic optimization
  - **`actuated`**: Uses SUMO's built-in actuated control (gap-based vehicle detection)
  - **`fixed`**: Uses fixed-time control with static timings from traffic light configuration

**Traffic Control Comparison:**

- **`tree_method`**: 
  - Dynamic, decentralized signal optimization
  - Adapts to real-time traffic conditions
  - Uses Tree Method's Tree Method algorithm for bottleneck prioritization
  - Best for complex traffic scenarios with varying demand patterns

- **`actuated`**: 
  - Vehicle-responsive signal control
  - Extends green phases when vehicles are detected
  - Uses SUMO's gap-based detection system
  - Good baseline for comparison with dynamic methods

- **`fixed`**: 
  - Static signal timing from pre-configured plans
  - Uses timing from `grid.tll.xml` file
  - Predictable but not adaptive to traffic conditions
  - Traditional signal control approach

**Usage Examples:**

```bash
# Default Tree Method (dynamic optimization)
--traffic_control tree_method

# SUMO Actuated control (vehicle-responsive)
--traffic_control actuated

# Fixed timing (static control)
--traffic_control fixed
```

**Experimental Analysis:**

This parameter enables comparative studies between different traffic control approaches:
- Use `tree_method` for advanced optimization performance
- Use `actuated` as primary baseline for comparison
- Use `fixed` for traditional signal control analysis
- Run multiple experiments with identical parameters except traffic control method
- Compare metrics like average travel time, throughput, and congestion levels

### Vehicle Departure Patterns

The system supports multiple temporal distribution patterns for vehicle departure times, replacing the original sequential departure (0, 1, 2, 3...) with realistic temporal patterns based on research:

- **`six_periods`** (default): Research-based 6-period system from mobility papers

  - **Morning** (6am-12pm): 20% of vehicles - Gradual traffic buildup
  - **Morning Rush** (7:30am-9:30am): 30% of vehicles - Peak commuter traffic
  - **Noon** (12pm-5pm): 25% of vehicles - Steady daytime activity
  - **Evening Rush** (5pm-7pm): 20% of vehicles - Evening commute home
  - **Evening** (7pm-10pm): 4% of vehicles - Social/entertainment trips
  - **Night** (10pm-6am): 1% of vehicles - Minimal overnight activity

- **`uniform`**: Even distribution across simulation time with small buffer at end

- **`rush_hours:7-9:40,17-19:30,rest:10`**: Custom rush hour definition

  - Define specific rush periods with intensity weights
  - `rest` parameter controls non-rush hour baseline traffic

- **`hourly:7:25,8:35,9:20,17:30,18:25,19:15,rest:5`**: Hour-by-hour control
  - Specify weight for each hour (0-23)
  - Very granular control over temporal patterns

**Usage Examples:**

```bash
# Default research-based 6-period system
--departure_pattern six_periods

# Simple uniform distribution
--departure_pattern uniform

# Custom rush hours (7-9am: 40%, 5-7pm: 30%, rest: 10%)
--departure_pattern "rush_hours:7-9:40,17-19:30,rest:10"

# Detailed hourly control
--departure_pattern "hourly:7:25,8:35,9:20,17:30,18:25,19:15,rest:5"
```

**Key Features:**

- Scales automatically to simulation `--end-time` (default 24 hours)
- Rush hour peaks align with real-world commuter patterns
- Compatible with all routing strategies and vehicle types
- Generates realistic traffic density variations throughout simulation

Omit --seed to use a random value each run.

File Structure & Descriptions
src/
├── cli.py # Orchestrates workflow
├── config.py # Configuration definitions for default parameters and settings
├── traffic/ # Traffic generation modules
│ ├── builder.py # Constructs the grid network and writes SUMO XML files
│ ├── edge_sampler.py # Implements random lane assignment and edge attractiveness
│ ├── routing.py # Shortest-path routing strategy using SUMO-lib’s Dijkstra
│ ├── xml_writer.py # Generates route and configuration XML for SUMO
│ └── **init**.py
├── network/ # Network utilities and zone extraction
│ ├── generate_grid.py # Creates orthogonal grid using SUMO's netgenerate
│ ├── split_edges_with_lanes.py # Integrated edge splitting with flow-based lane assignment
│ ├── zones.py # Polygon extraction and zone generation from junctions
│ ├── lane_counts.py # Core lane calculation algorithms (realistic, random, fixed)
│ ├── edge_attrs.py # Multiple edge attractiveness methods with time dependency
│ ├── traffic_lights.py # Traffic light injection and signal plans
│ └── **init**.py
├── sim/ # Simulation utilities
│ ├── sumo_controller.py # TraCI wrapper with per‑step callback and dynamic rerouting support
│ ├── sumo_utils.py # Wrapper functions for invoking SUMO and randomTrips.py
│ └── **init**.py
├── traffic_control/ # Signal‑control logic (third‑party & glue code)
│ └── decentralized_traffic_bottlenecks/
│ ├── integration.py # Bridges our simulator with Tree Method’s algorithm
│ ├── config.py, # Centralises algorithm defaults & hyper‑parameters (cycle time, max queue, …)
│ ├── enums.py, # Enumerations capturing cost types, algorithm modes, TLS states, etc.
│ ├── utils.py, # Shared helpers: JSON I/O, matrix ops, and miscellaneous maths
│ ├── classes/ # Core algorithm data‑structures (Graph, Network, …)
│ └── **init**.py
│ └── **init**.py
├── requirements.txt # Project dependencies
└── README.md # Short description of the project

````

## Example Scenarios

Here are 10 comprehensive scenarios for testing the SUMO traffic generator with a 5×5 grid, 150m blocks, varying traffic conditions, and GUI visualization:

### **Scenario 1: Morning Rush Hour Peak Traffic**
```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 800 --step-length 1.0 --end-time 7200 --departure_pattern six_periods --start_time_hour 7.0 --gui
````

**Description:** Simulates 2-hour morning rush period starting at 7am with heavy traffic (800 vehicles) using research-based departure patterns. Perfect for studying congestion during peak commuting hours.

### **Scenario 2: Light Evening Traffic**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 500 --step-length 1.0 --end-time 5400 --departure_pattern uniform --start_time_hour 20.0 --gui
```

**Description:** 1.5-hour evening simulation with moderate traffic (500 vehicles) uniformly distributed. Tests system performance during lighter traffic periods with no junction disruptions.

### **Scenario 3: All-Day Urban Simulation**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1200 --step-length 1.0 --end-time 28800 --departure_pattern six_periods --attractiveness poisson --gui
```

**Description:** Full 8-hour simulation (8am-4pm) with high vehicle density (1200 vehicles) and one random junction removed. Uses realistic traffic patterns to study long-term traffic flow dynamics.

### **Scenario 4: Custom Rush Hour Pattern**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 750 --step-length 1.0 --end-time 10800 --departure_pattern 'rush_hours:7-9:50,17-19:40,rest:10' --routing_strategy 'shortest 70 realtime 30' --gui
```

**Description:** 3-hour simulation with custom rush hour patterns (strong morning and evening peaks) and mixed routing strategies. Tests dynamic rerouting under realistic traffic conditions.

### **Scenario 5: Midnight to Dawn Low Traffic**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 600 --step-length 1.0 --end-time 21600 --departure_pattern six_periods --start_time_hour 0.0 --time_dependent --gui
```

**Description:** 6-hour overnight simulation (midnight-6am) with time-dependent attractiveness patterns. Studies traffic behavior during low-demand periods with dynamic edge attractiveness changes.

### **Scenario 6: High-Density Stress Test**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1500 --step-length 1.0 --end-time 14400 --departure_pattern uniform --routing_strategy 'shortest 50 realtime 30 fastest 20' --gui
```

**Description:** 4-hour stress test with very high vehicle density (1500 vehicles) and diverse routing strategies. Tests system stability and Tree Method's algorithm under extreme congestion conditions.

### **Scenario 7: Time-Dependent Attractiveness Test**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 900 --step-length 1.0 --end-time 12600 --departure_pattern six_periods --time_dependent --start_time_hour 8.0 --gui
```

**Description:** 3.5-hour simulation with time-dependent edge attractiveness that changes dynamically during the simulation. Tests how traffic patterns adapt to changing attractiveness profiles during morning hours.

### **Scenario 8: Weekend Traffic Pattern**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 0 --num_vehicles 650 --step-length 1.0 --end-time 18000 --departure_pattern uniform --start_time_hour 10.0 --attractiveness poisson --gui
```

**Description:** 5-hour weekend simulation (10am-3pm) with moderate, uniformly distributed traffic using default Poisson attractiveness. Simulates recreational travel patterns without peak hour constraints.

### **Scenario 9: Infrastructure Disruption Test**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 1000 --step-length 1.0 --end-time 9000 --departure_pattern six_periods --routing_strategy 'shortest 40 realtime 60' --seed 123 --gui
```

**Description:** 2.5-hour simulation testing network resilience with one junction removed and heavy real-time rerouting (60%). Uses fixed seed for reproducible infrastructure disruption analysis.

### **Scenario 10: Multi-Modal Traffic Mix**

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 850 --step-length 1.0 --end-time 16200 --departure_pattern six_periods --vehicle_types 'passenger 50 commercial 40 public 10' --attractiveness hybrid --gui
```

**Description:** 4.5-hour simulation with heavy commercial vehicle presence (40%) and hybrid attractiveness model. Studies mixed traffic flow with significant freight movement and public transport interaction.

### **Quick Testing Commands**

For rapid testing and development:

```bash
# Basic functionality test (30 minutes)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 500 --end-time 1800 --gui

# Performance test (1 hour, high density)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 1000 --end-time 3600 --gui

# Algorithm test (disrupted network with dynamic routing)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --junctions_to_remove 1 --num_vehicles 800 --routing_strategy 'realtime 100' --end-time 3600 --gui
```

### **Traffic Control Comparison Tests**

For comparing different traffic control methods:

```bash
# Test 1: Tree Method (Tree Method's algorithm)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --traffic_control tree_method --gui

# Test 2: SUMO Actuated (baseline comparison)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --traffic_control actuated --gui

# Test 3: Fixed timing (traditional control)
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 800 --end-time 3600 --traffic_control fixed --gui

# Experimental comparison with identical conditions
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control tree_method
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control actuated
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 1000 --end-time 3600 --seed 42 --traffic_control fixed
```

Each scenario tests different aspects of the traffic simulation system including temporal patterns, routing strategies, vehicle compositions, and network disruptions while maintaining realistic parameters for meaningful analysis.

## Experimental Framework

The system includes a comprehensive experimental framework for comparing different traffic control methods, replicating Tree Method's research methodology for traffic control evaluation.

### Experiment Structure

The framework is located in the `experiments/` directory with two main experiments:

```
experiments/
├── experiment-01-moderate-traffic/  # 600 vehicles, 2-hour simulation
│   ├── run_experiment.sh           # Automated experiment runner
│   ├── analyze_results.py          # Statistical analysis script
│   └── results/                    # Output directory
│       ├── tree_method/           # Tree Method's Tree Method results
│       ├── actuated/              # SUMO Actuated control results
│       ├── fixed/                 # Fixed-time control results
│       └── random/                # Mixed routing as random proxy
└── experiment-02-high-traffic/     # 1200 vehicles, 2-hour simulation
    ├── run_experiment.sh           # Same structure as experiment-01
    ├── analyze_results.py
    └── results/
```

### Traffic Control Methods Compared

1. **Tree Method** (`tree_method`): Tree Method's decentralized traffic optimization algorithm
2. **SUMO Actuated** (`actuated`): Vehicle-responsive signal control with gap-based detection
3. **Fixed Timing** (`fixed`): Static signal timing from pre-configured plans
4. **Random Proxy** (`random`): Fixed timing with mixed routing strategies (25% each strategy)

### Running Experiments

Each experiment runs 20 iterations per traffic control method (80 total simulations) with different random seeds to ensure statistical significance:

```bash
# Run moderate traffic experiment (600 vehicles)
cd experiments/experiment-01-moderate-traffic
./run_experiment.sh

# Run high traffic experiment (1200 vehicles)
cd experiments/experiment-02-high-traffic
./run_experiment.sh

# Analyze results after completion
python analyze_results.py
```

### Experimental Parameters

Both experiments use consistent parameters except for vehicle count:

- **Grid**: 5×5 network with 200m blocks
- **Duration**: 2 hours (7200 seconds)
- **Step Length**: 1.0 seconds
- **Departure Pattern**: Uniform distribution
- **Iterations**: 20 per method for statistical significance
- **Seeds**: 1-20 for reproducible results

**Moderate Traffic (Experiment 01):**
- 600 vehicles total
- Tests normal traffic conditions
- Expected completion rates: 70-90%

**High Traffic (Experiment 02):**
- 1200 vehicles total  
- Tests congested traffic conditions
- Expected completion rates: 40-70%

### Metrics Collected

Each simulation tracks and outputs:

- **Total Vehicles**: Number of vehicles in simulation
- **Vehicles Reached Destination**: Successful trip completions
- **Completion Rate**: Percentage of vehicles reaching destination
- **Average Travel Time**: Mean journey duration for completed trips
- **Simulation Time**: Total simulation duration

### Statistical Analysis

The `analyze_results.py` scripts provide:

- **Summary Statistics**: Mean, median, standard deviation for each method
- **Comparative Analysis**: Performance differences between methods
- **Confidence Intervals**: 95% confidence intervals for statistical significance
- **Improvement Calculations**: Percentage improvements of Tree Method vs baselines
- **Visualization**: Box plots and bar charts comparing methods

**Sample Analysis Output:**

```
=== EXPERIMENT RESULTS SUMMARY ===

Method: tree_method
  Average Travel Time: 1456.32 ± 45.67 seconds
  Completion Rate: 0.847 ± 0.023 (84.7%)
  
Method: actuated  
  Average Travel Time: 1632.45 ± 52.34 seconds
  Completion Rate: 0.793 ± 0.031 (79.3%)

Tree Method Improvements:
  vs Actuated: 10.8% faster travel time, 6.8% higher completion rate
  vs Fixed: 15.2% faster travel time, 12.4% higher completion rate
```

### Expected Results

Based on Tree Method's research, the Tree Method should demonstrate:

- **20-45% improvement** in travel times vs fixed timing
- **10-25% improvement** vs actuated control  
- **Higher completion rates** under congested conditions
- **More consistent performance** across different traffic loads

### Reproducing Research Results

The experimental framework enables:

1. **Validation** of Tree Method's Tree Method claims
2. **Baseline Comparison** against standard SUMO control methods
3. **Scalability Testing** across different traffic densities
4. **Statistical Significance** through multiple iterations
5. **Publication-Ready Results** with proper confidence intervals

### Running Custom Experiments

Create new experiments by copying the structure:

```bash
# Create new experiment directory
mkdir experiments/experiment-03-custom
cp experiments/experiment-01-moderate-traffic/* experiments/experiment-03-custom/

# Modify parameters in run_experiment.sh
# Update vehicle count, grid size, or other parameters
# Run experiment and analysis
```

The framework provides a rigorous foundation for traffic control research and algorithm validation.
