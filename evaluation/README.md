# Evaluation Framework

This directory contains all evaluation components for the SUMO Traffic Generator project, including statistical benchmarks and testing datasets.

## Directory Structure

```
evaluation/
├── benchmarks/          # Statistical comparison studies
│   ├── experiment-01-moderate-traffic/
│   └── experiment-02-high-traffic/
├── datasets/           # Sample data for testing and validation
│   ├── osm/           # Real-world OpenStreetMap files
│   └── networks/      # Pre-built Tree Method research networks
└── README.md          # This file
```

## Benchmarks

### Statistical Comparison Studies

The `benchmarks/` directory contains automated experimental frameworks for comparing traffic control methods:

#### Experiment 01: Moderate Traffic
- **Location**: `benchmarks/experiment-01-moderate-traffic/`
- **Scenario**: 600 vehicles, 2-hour simulation
- **Purpose**: Standard traffic conditions comparison

#### Experiment 02: High Traffic  
- **Location**: `benchmarks/experiment-02-high-traffic/`
- **Scenario**: 1200 vehicles, 2-hour simulation
- **Purpose**: Stress testing under congestion

### Running Benchmarks

```bash
# Run moderate traffic experiment
cd evaluation/benchmarks/experiment-01-moderate-traffic
./run_experiment.sh

# Run high traffic experiment
cd evaluation/benchmarks/experiment-02-high-traffic
./run_experiment.sh

# Analyze results (after completion)
python analyze_results.py
```

### Traffic Control Methods Compared

1. **Tree Method**: Decentralized traffic control algorithm
2. **SUMO Actuated**: Gap-based signal control (baseline)
3. **Fixed Timing**: Static signal timing
4. **Random**: Mixed routing strategies (control)

### Expected Results

- **Tree Method vs Fixed**: 20-45% improvement in travel times
- **Tree Method vs Actuated**: 10-25% improvement
- **Statistical Significance**: 20 iterations provide robust confidence intervals

## Datasets

### OSM Data (`datasets/osm/`)

Real-world OpenStreetMap files for testing on actual street networks:

- **Purpose**: Validate algorithms on real urban topologies
- **Usage**: Use with `--osm_file evaluation/datasets/osm/[filename].osm`
- **Example**: Manhattan East Village street network

### Research Networks (`datasets/networks/`)

Pre-built Tree Method research networks for validation:

- **Purpose**: Bypass network generation, test directly on established benchmarks
- **Usage**: Use with `--tree_method_sample evaluation/datasets/networks/`
- **Features**: Complex urban networks (946 vehicles, 2-hour simulation)

### Using Datasets

```bash
# OSM real-world networks
env PYTHONUNBUFFERED=1 python -m src.cli --osm_file evaluation/datasets/osm/export.osm --num_vehicles 500 --gui

# Pre-built research networks
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --gui

# Compare methods on identical network
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control actuated
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control fixed
```

## Research Applications

### Tree Method Validation

This evaluation framework enables:

1. **Algorithm Validation**: Test Tree Method implementation against established benchmarks
2. **Performance Comparison**: Statistical analysis of different traffic control methods
3. **Real-World Testing**: Validate on actual street topologies from OSM data
4. **Reproducible Research**: Consistent experimental methodology with proper random seeding

### Publication-Ready Analysis

- **Statistical Rigor**: Multiple iterations with confidence intervals
- **Method Comparison**: Head-to-head comparison of traffic control algorithms
- **Performance Metrics**: Travel times, completion rates, throughput analysis
- **Research Replication**: Framework designed to validate published Tree Method claims

## File Management

### Dataset Integration

- **OSM Files**: Automatically processed through SUMO netconvert pipeline
- **Network Files**: Direct integration with simulation bypass mode
- **Result Storage**: Organized by method and iteration for analysis

### Quality Assurance

- **Network Validation**: Minimum topology requirements for viable simulations
- **Data Integrity**: XML validation and structure verification
- **Performance Monitoring**: Resource usage tracking during experiments

## Getting Started

1. **Choose Evaluation Type**:
   - Benchmarks: For statistical comparison studies
   - Datasets: For testing specific scenarios

2. **Select Appropriate Data**:
   - OSM: Real-world street networks
   - Networks: Research-validated topologies

3. **Run Experiments**:
   - Use provided scripts for automated execution
   - Follow experimental protocols for reproducibility

4. **Analyze Results**:
   - Use included analysis tools for statistical evaluation
   - Generate publication-ready reports and visualizations

## Research Focus

This evaluation framework focuses specifically on research validation and performance analysis:

1. **Performance Benchmarks**: Statistical comparison of traffic control methods
2. **Dataset Management**: Organized sample data for reproducible research
3. **Publication-Ready Analysis**: Rigorous experimental methodology with confidence intervals

For software testing (unit tests, integration tests, domain validation), see the `tests/` directory.

For detailed usage instructions, see the main project documentation in `CLAUDE.md`.