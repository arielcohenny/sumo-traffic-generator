# SUMO Traffic Generator

A comprehensive Python-based SUMO traffic simulation framework with intelligent signal control, supporting synthetic grids, real-world OpenStreetMap data, and Tree Method research datasets.

## Key Features

- **Triple Network Support**: Synthetic grids, real-world OSM data, and Tree Method research datasets
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

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Synthetic Grid Network
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

### Real-World OSM Network
```bash
# Manhattan street network with Tree Method optimization
env PYTHONUNBUFFERED=1 python -m src.cli \
  --osm_file evaluation/datasets/osm/manhattan_upper_west.osm \
  --num_vehicles 500 \
  --traffic_control tree_method \
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

## Project Structure

```
├── src/                    # Core application code
│   ├── network/           # Network generation and processing
│   ├── traffic/           # Vehicle routing and generation  
│   ├── orchestration/     # High-level simulation coordination
│   └── sumo_integration/  # SUMO/TraCI interface layer
├── evaluation/            # Research validation framework
│   ├── benchmarks/        # Performance comparison studies
│   └── datasets/          # OSM files and research networks
├── tests/                 # Software testing framework
├── tools/                 # Development utilities
└── workspace/             # Generated simulation files (temporary)
```

## Documentation

- **Technical Specification**: See [SPECIFICATION.md](SPECIFICATION.md) for complete technical details, parameters, and implementation documentation
- **Research & Benchmarks**: See [evaluation/](evaluation/) for performance studies, datasets, and experimental framework

## License

This project is licensed under the MIT License.