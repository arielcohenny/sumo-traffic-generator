# Synthetic Grid Benchmark Suite

A comprehensive, centralized framework for systematic Tree Method validation across synthetic grid networks of varying complexity.

## Overview

The Synthetic Grid Benchmark Suite provides a systematic approach to testing the Tree Method traffic control algorithm against baseline methods (SUMO Actuated and Fixed Timing) across different network scales and traffic scenarios. The framework is designed for scientific rigor with centralized configuration management and comprehensive statistical analysis.

## Key Features

- **Centralized Configuration**: Single JSON file controls all parameters across experiments
- **Multi-Scale Testing**: 5x5, 7x7, and 9x9 grid networks with scaled vehicle counts
- **Comprehensive Parameter Matrix**: 324 unique scenarios per grid size
- **Statistical Rigor**: 20 runs per scenario for robust confidence intervals
- **Automated Analysis**: Hierarchical analysis from individual runs to master statistics
- **Publication Ready**: Generates research-quality statistical comparisons

## Framework Architecture

```
synthetic-grids/
├── experiment_config.json           # Central configuration
├── run_all_experiments.sh          # Master execution script
├── analyze_all_experiments.py      # Master analysis script
├── grids-5x5/                      # 5x5 grid experiments
│   ├── run_all_runs.sh
│   ├── analyze_experiment.py
│   └── [1/, 2/, 3/, ... experiment folders]
├── grids-7x7/                      # 7x7 grid experiments
│   ├── run_all_runs.sh
│   ├── analyze_experiment.py
│   └── [1/, 2/, 3/, ... experiment folders]
└── grids-9x9/                      # 9x9 grid experiments
    ├── run_all_runs.sh
    ├── analyze_experiment.py
    └── [1/, 2/, 3/, ... experiment folders]
```

## Experiment Matrix

### Grid Configurations

- **5x5 Grid**: Light (400), Moderate (800), Heavy (1600) vehicles
- **7x7 Grid**: Light (800), Moderate (1600), Heavy (3200) vehicles
- **9x9 Grid**: Light (1400), Moderate (2800), Heavy (5600) vehicles

### Parameter Variations

- **Vehicle Types**: 2 realistic variations (Mixed Urban, Transit-Heavy)
- **Routing Strategies**: 3 GPS usage patterns (Conservative, Balanced, Aggressive)
- **Departure Patterns**: 3 temporal distributions (Rush Hour, Uniform, Peak-Focused)
- **Simulation Durations**: 2 time scales (2+ hours, 24 hours)
- **Network Disruption**: 3 levels (0, 1, 2 junctions removed)

### Totals

- **324 unique scenarios per grid**
- **972 total experiments across all grids**
- **2,916 total simulations (×3 traffic control methods)**
- **58,320 total runs (×20 iterations for statistical validity)**

## Quick Start

### 1. Configuration

Edit the central configuration file to customize experiments:

```bash
vim experiment_config.json
```

### 2. Run Experiments

#### Single Grid Size

```bash
# Run all experiments for 5x5 grid
cd grids-5x5 && ./run_all_runs.sh

# Run all experiments for 7x7 grid
cd grids-7x7 && ./run_all_runs.sh

# Run all experiments for 9x9 grid
cd grids-9x9 && ./run_all_runs.sh
```

#### All Grid Sizes

```bash
# Run all experiments across all grid sizes
./run_all_experiments.sh
```

### 3. Analyze Results

#### Individual Grid Analysis

```bash
# Analyze specific grid
cd grids-5x5 && python analyze_experiment.py
```

#### Master Analysis

```bash
# Comprehensive analysis across all grids
python analyze_all_experiments.py
```

## Configuration Management

### Central Configuration File: `experiment_config.json`

The framework uses a single configuration file that controls all experimental parameters:

```json
{
  "grid_configurations": {
    "5x5": {
      "dimension": 5,
      "vehicle_counts": { "light": 400, "moderate": 800, "heavy": 1600 }
    },
    "7x7": {
      "dimension": 7,
      "vehicle_counts": { "light": 800, "moderate": 1600, "heavy": 3200 }
    },
    "9x9": {
      "dimension": 9,
      "vehicle_counts": { "light": 1400, "moderate": 2800, "heavy": 5600 }
    }
  },
  "shared_parameters": {
    "vehicle_types": [
      "passenger 90 public 10",
      "passenger 80 public 20"
    ],
    "routing_strategies": [
      "shortest 80 realtime 20",
      "shortest 50 realtime 40 fastest 10",
      "shortest 30 realtime 50 fastest 20"
    ],
    "departure_patterns": [
      "six_periods",
      "uniform",
      "rush_hours:7-9:40,17-19:30,rest:20"
    ],
    "simulation_durations": [7300, 86400],
    "junctions_removed": [0, 1, 2]
  }
}
```

### Making Changes

To modify experimental parameters:

1. **Edit Configuration**: Update `experiment_config.json`
2. **Automatic Propagation**: All scripts read the config dynamically
3. **Run Experiments**: Changes take effect immediately

Example - Adding a new vehicle type:

```bash
# Edit the config file
vim experiment_config.json
# Add "passenger 95 public 5" to vehicle_types array

# Run experiments - they automatically use the new configuration
./run_all_experiments.sh
```

## Execution Flow

### Hierarchical Execution

```
run_all_experiments.sh
└── grids-5x5/run_all_runs.sh → individual run_experiment.sh scripts
└── grids-7x7/run_all_runs.sh → individual run_experiment.sh scripts
└── grids-9x9/run_all_runs.sh → individual run_experiment.sh scripts
```

### Hierarchical Analysis

```
analyze_all_experiments.py
└── grids-5x5/analyze_experiment.py → individual analyze_results.py scripts
└── grids-7x7/analyze_experiment.py → individual analyze_results.py scripts
└── grids-9x9/analyze_experiment.py → individual analyze_results.py scripts
```

## Built-in Safety Features

### Validation and Confirmation

- **JSON Validation**: Automatic syntax checking before execution
- **Experiment Counting**: Shows total runs and estimated time before starting
- **User Confirmation**: Requires explicit approval for expensive computations
- **Timeout Protection**: Prevents runaway simulations

### Error Handling

- **Graceful Failures**: Individual experiment failures don't stop the entire suite
- **Progress Tracking**: Real-time status updates during execution
- **Comprehensive Logging**: Detailed logs for debugging failed experiments
- **Automatic Recovery**: Analysis scripts regenerate missing JSON files

## Performance Characteristics

### Execution Time Estimates

- **Per Run**: ~45 minutes average (varies by grid size and traffic load)
- **Per Grid**: ~250 hours (324 experiments × 3 methods × 20 runs × 45min)
- **Full Suite**: ~750 hours across all 3 grids

### Storage Requirements

- **Per Run**: ~0.5 MB (simulation logs)
- **Full Suite**: ~30 GB total storage

### Computational Requirements

- **CPU**: Multi-core recommended for reasonable execution times
- **Memory**: 4-8 GB RAM depending on grid size
- **Disk I/O**: Moderate (primarily log file writing)

## Analysis Capabilities

### Individual Run Analysis

- Vehicle metrics (entered, arrived, completion rate)
- Travel time statistics (mean, std, range)
- Traffic control method comparison
- Configuration tracking

### Grid-Level Analysis

- Aggregated statistics across all experiments in a grid
- Method performance comparisons
- Tree Method validation metrics
- Scalability insights

## Research Applications

### Tree Method Validation

- **Baseline Comparison**: Systematic comparison vs SUMO Actuated and Fixed Timing
- **Scalability Analysis**: Performance across different network complexities
- **Parameter Sensitivity**: Impact of vehicle mix, routing, temporal patterns
- **Network Resilience**: Performance under junction removal/disruption

### Expected Results

Based on research literature, Tree Method should demonstrate:

- **20-45% improvement** in travel times vs fixed timing
- **10-25% improvement** vs SUMO actuated timing
- **Better completion rates** under high congestion scenarios
- **Consistent performance** across different grid sizes

### Statistical Validity

- **20 runs per scenario** provide robust confidence intervals
- **Controlled variables** ensure fair comparison between methods
- **Realistic parameters** based on real-world traffic patterns
- **Comprehensive coverage** of urban traffic scenarios

## Troubleshooting

### Common Issues

1. **Configuration Errors**

   ```bash
   # Validate JSON syntax
   python -c "import json; json.load(open('experiment_config.json'))"
   ```

2. **Missing Results**

   ```bash
   # Check experiment execution logs
   grep -r "ERROR\|Failed" grids-*/*/results/
   ```

3. **Analysis Failures**
   ```bash
   # Regenerate analysis files
   cd grids-5x5 && python analyze_experiment.py
   ```

### Performance Optimization

1. **Parallel Execution**: Run different grid sizes simultaneously
2. **Selective Testing**: Modify config to test specific scenarios first
3. **Storage Management**: Archive old results to free disk space

## Integration with Main Framework

This benchmark suite integrates seamlessly with the main SUMO traffic generator:

- **Uses existing CLI**: All experiments use `python -m src.cli`
- **Standard parameters**: Compatible with all main framework features
- **Consistent output**: Generates standard SUMO simulation logs
- **Analysis compatibility**: Works with existing analysis tools

## Version Control and Reproducibility

- **Configuration Versioning**: Track parameter changes via git
- **Seed Management**: Controlled random seeds for reproducible results
- **Environment Documentation**: Clear dependency and setup requirements
- **Result Archival**: JSON files suitable for long-term storage

## Contributing

To extend the framework:

1. **Add Parameters**: Update `experiment_config.json` with new parameter arrays
2. **Modify Scripts**: Grid-specific scripts automatically adapt to config changes
3. **Test Changes**: Use small-scale tests before full execution
4. **Document Updates**: Update this README with new capabilities

## License and Citation

This framework is part of the Tree Method traffic control research project. When using this framework in research, please cite:

> "Decentralised Bottleneck Prioritization Strategy for Traffic Flow Improvement" - Transportation Research Interdisciplinary Perspectives

---

**Framework Version**: 1.0  
**Last Updated**: 2025-07-26  
**Maintainer**: Tree Method Research Team
