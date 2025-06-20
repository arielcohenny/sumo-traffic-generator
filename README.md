Project Progress: Key Milestones

1. Grid Generation: Created an n×n junction network (grid.net.xml), enabling random removal of specified internal nodes for dynamic topology.

2. Zone Extraction: Derived polygonal zones from adjacent junctions per Table 1 in A Simulation Model for Intra‑Urban Movements.

3. Lane Configuration: Applied configurable lane assignment to each edge (randomized within defined bounds).

4. Edge Attractiveness Modeling: Computed departure/arrival weights per edge using a Poisson distribution, guiding vehicle origin and destination selection.

5. Route Generation: Built a vehicle‑route prototype that leverages edge attractiveness and shortest‑path computation; compatible with SUMO’s randomTrips.py for scalable trip creation. All vehicles starting at time 0.

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
├── cli.py # Command-line interface: parses arguments and orchestrates workflow
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
│ └── **init**.py
├── sim/ # Simulation utilities
│ ├── sumo_utils.py # Wrapper functions for invoking SUMO and randomTrips.py
│ └── **init**.py
├── requirements.txt # Project dependencies
└── README.md # Short description of the project

cli.py: Entry point; loads args, calls modules in sequence (grid → zones → lanes → attractiveness → routes).
config.py: Central location for defaults (dimensions, vehicle counts, seeds).
builder.py: Grid creation logic and output of grid.net.xml.
edge_sampler.py: Assigns lanes and computes edge weights.
routing.py: Calculates routes between selected edges using SUMO-lib’s getShortestPath.
xml_writer.py: Formats and writes vehicle route and SUMO config files.
generator.py: Identifies and removes internal junctions, gathers X/Y for zones.
lanes.py: Applies random lane counts per edge within configured bounds.
zones.py: Builds zone polygons and applies optional inset buffering.
sumo_utils.py: Interfaces with SUMO binaries and randomTrips.py for trip file generation.
```
