# Traffic Control Experiments

This directory contains experiments designed to replicate and extend Tree Method traffic control research comparing different traffic signal control methods.

## Experiments Overview

### Experiment 01: Moderate Traffic Load
- **Location**: `experiment-01-moderate-traffic/`
- **Purpose**: Compare traffic control methods under moderate congestion
- **Parameters**: 600 vehicles, 5×5 grid, 2-hour simulation
- **Methods**: Tree Method, SUMO Actuated, Fixed, Random

### Experiment 02: High Traffic Load  
- **Location**: `experiment-02-high-traffic/`
- **Purpose**: Compare traffic control methods under high congestion
- **Parameters**: 1200 vehicles, 5×5 grid, 2-hour simulation
- **Methods**: Tree Method, SUMO Actuated, Fixed, Random

## Traffic Control Methods

1. **Tree Method**: Decentralized bottleneck prioritization algorithm
2. **SUMO Actuated**: Gap-based vehicle detection (baseline comparison)
3. **Fixed**: Static signal timing from pre-configured plans
4. **Random**: Mixed routing strategies simulating unpredictable behavior

## Running Experiments

### Prerequisites
- Python virtual environment activated
- SUMO installed and accessible
- All project dependencies installed

### Execute Experiments

```bash
# Run moderate traffic experiment
cd experiment-01-moderate-traffic
chmod +x run_experiment.sh
./run_experiment.sh

# Run high traffic experiment  
cd experiment-02-high-traffic
chmod +x run_experiment.sh
./run_experiment.sh
```

### Analyze Results

```bash
# Analyze moderate traffic results
cd experiment-01-moderate-traffic
python analyze_results.py

# Analyze high traffic results
cd experiment-02-high-traffic
python analyze_results.py
```

## Expected Outputs

### Experiment Logs
- `results/[method]/run_[1-20].log`: Individual simulation logs
- 20 runs per method for statistical significance

### Analysis Results
- `results/summary_statistics.json`: Statistical summary
- `results/[experiment]_traffic_comparison.png`: Comparison plots
- Console output with performance metrics

## Key Metrics

Based on Tree Method research methodology:

1. **Average Travel Time**: Mean time for vehicles to reach destinations
2. **Throughput**: Number of vehicles successfully completing trips
3. **Completion Rate**: Percentage of vehicles reaching destinations
4. **Performance Improvements**: Tree Method vs baseline comparisons

## Expected Results

According to Tree Method research findings:
- Tree Method should show 20-45% travel time reduction vs SUMO Actuated
- Tree Method should show 7-24% throughput improvement vs SUMO Actuated
- Benefits more pronounced under high traffic load conditions
- Tree Method should outperform all other methods consistently

## File Structure

```
experiments/
├── README.md
├── experiment-01-moderate-traffic/
│   ├── run_experiment.sh
│   ├── analyze_results.py
│   └── results/
│       ├── tree_method/
│       ├── actuated/
│       ├── fixed/
│       └── random/
└── experiment-02-high-traffic/
    ├── run_experiment.sh
    ├── analyze_results.py
    └── results/
        ├── tree_method/
        ├── actuated/
        ├── fixed/
        └── random/
```

## Notes

- Each experiment runs 20 iterations per method (80 total runs)
- Results are seeded for reproducibility
- Random method uses mixed routing strategies as proxy for unpredictable behavior
- Analysis scripts require matplotlib, seaborn, pandas, and numpy
- Experiments take approximately 2-4 hours to complete depending on hardware