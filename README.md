Project Progress: Key Milestones

1. Grid Generation: Created an n×n junction network (grid.net.xml), enabling flexible removal of internal nodes - supports both random selection by count and explicit junction ID specification for dynamic topology.

2. Zone Extraction: Derived polygonal zones from adjacent junctions per Table 1 in A Simulation Model for Intra‑Urban Movements.

3. Lane Configuration: Applied configurable lane assignment to each edge with three algorithms:
   - **Realistic**: Zone-based traffic demand calculation using land use types
   - **Random**: Randomized assignment within defined bounds (1-3 lanes)
   - **Fixed**: Uniform lane count across all edges

4. Edge Attractiveness Modeling: Multiple research-based methods for computing departure/arrival weights:
   - **Poisson**: Original distribution approach (λ_depart=3.5, λ_arrive=2.0)
   - **Land Use**: Zone-type multipliers (Residential, Employment, Mixed, etc.)
   - **Gravity**: Network centrality and spatial distance factors
   - **IAC**: Integrated Attraction Coefficient combining multiple factors
   - **Hybrid**: Weighted combination of spatial and land use approaches
   - **4-Phase Temporal**: Research-based time-of-day variations with bimodal traffic patterns

5. Route Generation: Built a vehicle‑route prototype that leverages edge attractiveness and shortest‑path computation; compatible with SUMO’s randomTrips.py for scalable trip creation. All vehicles starting at time 0.

6. SUMO Configuration Authoring – Automatically generates a .sumocfg that wires together the network, route, additional files, and simulation parameters in a single ready‑to‑run configuration.

7. TraCI Runtime Integration – Introduced sim/sumo_controller.py, a thin wrapper around TraCI that launches SUMO (GUI or headless), advances the simulation, and exposes a per‑step callback API for custom control logic.

8. Nimrod’s Tree‑Method Control – Integrated the decentralized‑traffic‑bottlenecks library: the pipeline now converts the network to a JSON tree, builds Nimrod’s Graph, computes an optimal phase map each step, and applies it via TraCI—enabling fully dynamic, decentralized signal control during the simulation.

Installation

```bash
# 1. Clone this repo
git clone https://github.com/arielcohenny/sumo-traffic-generator.git
cd sumo-traffic-generator

# 2. Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt


Usage & Parameters
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
  --gui                         # Launch SUMO in GUI mode (sumo-gui) instead of headless sumo
```

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
│ ├── split_edges.py # Splits edges for enhanced network complexity  
│ ├── zones.py # Polygon extraction and zone generation from junctions
│ ├── lane_counts.py # Lane assignment algorithms (realistic, random, fixed)
│ ├── edge_attrs.py # Multiple edge attractiveness methods with time dependency
│ ├── traffic_lights.py # Traffic light injection and signal plans
│ └── **init**.py
├── sim/ # Simulation utilities
│ ├── sumo_controller.py # TraCI wrapper with per‑step callback and dynamic rerouting support
│ ├── sumo_utils.py # Wrapper functions for invoking SUMO and randomTrips.py
│ └── **init**.py
├── traffic_control/ # Signal‑control logic (third‑party & glue code)
│ └── decentralized_traffic_bottlenecks/
│     ├── integration.py  # Bridges our simulator with Nimrod’s algorithm
│     ├── config.py, # Centralises algorithm defaults & hyper‑parameters (cycle time, max queue, …)
│     ├── enums.py, # Enumerations capturing cost types, algorithm modes, TLS states, etc.
│     ├── utils.py, # Shared helpers: JSON I/O, matrix ops, and miscellaneous maths
│     ├── classes/   # Core algorithm data‑structures (Graph, Network, …)
│     └── **init**.py
│ └── **init**.py
├── requirements.txt # Project dependencies
└── README.md # Short description of the project
```
