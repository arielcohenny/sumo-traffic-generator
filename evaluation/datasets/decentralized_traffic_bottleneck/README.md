# Decentralized Traffic Bottleneck Test Cases

This directory contains all test cases from the original research repository:
https://github.com/nimrodSerokTAU/decentralized-traffic-bottlenecks

## Structure

```
decentralized_traffic_bottleneck/
├── Experiment1-realistic-high-load/     # 20 test cases
├── Experiment2-rand-high-load/          # 20 test cases  
├── Experiment3-realistic-moderate-load/ # 20 test cases
├── Experiment4-and-moderate-load/       # 20 test cases
└── README.md                            # This file
```

## Test Case Contents

Each numbered directory (1-20) within each experiment contains:

- **network.net.xml**: SUMO network file
- **vehicles.trips.xml**: Vehicle trip definitions  
- **simulation.sumocfg.xml**: SUMO simulation configuration
- **configurations.txt**: Algorithm configurations for Tree Method comparison

## Total Dataset

- **4 experiments** with different traffic patterns and loads
- **20 test cases** per experiment
- **80 total test cases** for comprehensive validation
- **320 files** downloaded (4 files × 80 test cases)

## Usage

Use with `--tree_method_sample` CLI argument:

```bash
# Run specific test case
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1

# Future: Run all cases in experiment (planned enhancement)
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load --run_all
```

## Research Validation Purpose

These test cases enable:

1. **Algorithm Validation**: Test Tree Method implementation against original research data
2. **Performance Comparison**: Compare results with original CurrentTreeDvd, SUMOActuated, Random, and Uniform algorithms
3. **Statistical Robustness**: 80 test cases provide strong statistical validation
4. **Reproducibility**: Verify implementation produces equivalent results to original research

## Citation

Original repository: nimrodSerokTAU/decentralized-traffic-bottlenecks
Paper: [Citation needed for the decentralized traffic bottlenecks research]