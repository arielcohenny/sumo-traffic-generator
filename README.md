Project Progress: Key Milestones

1. Grid Generation: Created an n×n junction network (grid.net.xml), enabling random removal of specified internal nodes for dynamic topology.

2. Zone Extraction: Derived polygonal zones from adjacent junctions per Table 1 in A Simulation Model for Intra‑Urban Movements.

3. Lane Configuration: Applied configurable lane assignment to each edge (randomized within defined bounds).

4. Edge Attractiveness Modeling: Computed departure/arrival weights per edge using a Poisson distribution, guiding vehicle origin and destination selection.

5. Route Generation: Built a vehicle‑route prototype that leverages edge attractiveness and shortest‑path computation; compatible with SUMO’s randomTrips.py for scalable trip creation. All vehicles starting at time 0.

6. Static Traffic‑Light Injection – Added a first‑pass signal plan (inject_traffic_lights) that inserts default four‑phase logic for every controlled junction, ensuring valid TLS state strings for subsequent TraCI control.

7. SUMO Configuration Authoring – Automatically generates a .sumocfg that wires together the network, route, additional files, and simulation parameters in a single ready‑to‑run configuration.

8. TraCI Runtime Integration – Introduced sim/sumo_controller.py, a thin wrapper around TraCI that launches SUMO (GUI or headless), advances the simulation, and exposes a per‑step callback API for custom control logic.

9. Nimrod’s Tree‑Method Control – Integrated the decentralized‑traffic‑bottlenecks library: the pipeline now converts the network to a JSON tree, builds Nimrod’s Graph, computes an optimal phase map each step, and applies it via TraCI—enabling fully dynamic, decentralized signal control during the simulation.

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
env PYTHONUNBUFFERED=1 python -m src.cli \
--grid_dimension <int> # Number of rows/columns (default: 5)
--block_size_m <float> # Block length in meters (default: 200)
--blocks_to_remove <int> # Internal junctions to delete (default: 0)
--num_vehicles <int> # Total trips to generate (default: 300)
--seed <int> # RNG seed (optional)
--step-length #Simulation step length in seconds (for TraCI loop) default=1.0
--end-time #Total simulation duration in seconds. (default 3600)
--gui #Launch SUMO in GUI mode (sumo-gui) instead of headless sumo.

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
│ ├── generator.py # Parses junctions, removes internal nodes, and builds zones
│ ├── lanes.py # Lane count configuration logic
│ ├── zones.py # Polygon extraction and buffering for zones
│ ├── edge_attrs.py # Computes edge attractiveness (λ_depart / λ_arrive)
│ └── **init**.py
├── sim/ # Simulation utilities
│ ├── sumo_controller.py # Thin TraCI wrapper with per‑step callback support
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
