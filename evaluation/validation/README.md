# Tree Method Implementation Validation Framework

This validation framework enables systematic comparison of our Tree Method implementation against the original research results from the decentralized-traffic-bottlenecks repository.

## Framework Structure

```
evaluation/validation/
├── README.md                           # This file
├── run_single_validation.py            # Single test case validation
├── download_all_original_results.py    # Download original research results
├── statistical_analysis.py             # Statistical analysis and reporting
├── run_complete_validation.py          # Complete validation pipeline
├── baselines/                          # Original research results
│   ├── original_results.xlsx          # Main results spreadsheet
│   └── Experiment*_*.txt              # Aggregate results per experiment
└── results/                            # Our validation results
    ├── *.json                          # Individual validation results
    ├── validation_report.json          # Statistical analysis report
    └── visualizations/                 # Generated plots and charts
        ├── travel_time_differences.png
        ├── travel_time_correlation.png
        └── execution_time_distribution.png
```

## Validation Process

### Phase 1: Single Test Case Validation

Validate a single test case to establish baseline accuracy:

```bash
# Basic validation
python evaluation/validation/run_single_validation.py \
  evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1 \
  --traffic_control tree_method \
  --output validation_result.json

# Compare different traffic control methods
python evaluation/validation/run_single_validation.py \
  evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1 \
  --traffic_control actuated \
  --output actuated_result.json
```

### Phase 2: Systematic Validation

Run validation across multiple test cases:

```bash
# Quick validation (30 minutes simulation time)
python evaluation/validation/run_complete_validation.py \
  --test-cases evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1 \
               evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/2 \
  --traffic-control tree_method actuated \
  --end-time 1800

# Full validation (2 hours simulation time like original research)
python evaluation/validation/run_complete_validation.py \
  --test-cases evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/* \
  --traffic-control tree_method \
  --end-time 7300 \
  --download-results
```

### Phase 3: Statistical Analysis

Generate comprehensive statistical report:

```bash
# Run analysis on existing validation results
python evaluation/validation/statistical_analysis.py
```

## Validation Metrics

### Primary Metrics
- **Average Travel Time**: Mean vehicle travel time across simulation
- **Completion Rate**: Percentage of vehicles reaching destination
- **Total Vehicles**: Number of vehicles in simulation
- **Execution Time**: Time taken to run simulation

### Comparison Analysis
- **Travel Time Difference**: Percentage difference from original results
- **Completion Rate Difference**: Absolute difference in vehicles completed
- **Statistical Correlation**: Pearson correlation with original results
- **Method Performance**: ANOVA comparison across traffic control methods

### Tolerance Thresholds
- **Travel Time**: ±10% acceptable difference
- **Completion Rate**: ±50 vehicles acceptable difference
- **High Correlation**: r > 0.8 expected for travel times

## Expected Results

Since we're using the original Tree Method classes, we expect:

1. **High Fidelity**: Very close agreement with original CurrentTreeDvd results
2. **Travel Time Correlation**: r > 0.95 correlation coefficient
3. **Completion Rate Agreement**: Within ±25 vehicles for most test cases
4. **Method Rankings**: Tree Method > Actuated > Fixed (relative performance)

## Usage Examples

### Quick Validation Test
```bash
# Test one case with Tree Method (5 minutes)
source .venv/bin/activate
python evaluation/validation/run_complete_validation.py \
  --test-cases evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1 \
  --traffic-control tree_method \
  --end-time 300
```

### Research Validation
```bash
# Validate 5 test cases with multiple methods (full research protocol)
python evaluation/validation/run_complete_validation.py \
  --test-cases evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/{1,2,3,4,5} \
  --traffic-control tree_method actuated fixed \
  --end-time 7300 \
  --download-results
```

### Publication-Ready Analysis
```bash
# Full experiment validation (20 test cases, all methods)
for experiment in Experiment1-realistic-high-load Experiment2-rand-high-load; do
  python evaluation/validation/run_complete_validation.py \
    --test-cases evaluation/datasets/decentralized_traffic_bottleneck/$experiment/* \
    --traffic-control tree_method actuated fixed \
    --end-time 7300
done

# Generate comprehensive report
python evaluation/validation/statistical_analysis.py
```

## Results Interpretation

### Validation Success Criteria
- **Travel Time**: Mean difference < 5%, 90% of cases within ±10%
- **Completion Rate**: Mean difference < 25 vehicles, 80% within ±50 vehicles  
- **Correlation**: r > 0.8 for travel times, r > 0.7 for completion rates
- **Method Comparison**: Tree Method performs better than baselines

### Common Issues
- **File Download Errors**: Some original result files may return 404
- **Simulation Timeouts**: Complex networks may need longer timeout
- **Missing Dependencies**: Ensure matplotlib, seaborn, scipy installed

## Integration with Testing Framework

This validation framework complements the software testing framework:

- **`tests/`**: Code correctness, regression prevention, development workflow
- **`evaluation/validation/`**: Research fidelity, algorithm accuracy, publication validation
- **`evaluation/benchmarks/`**: Performance comparison, statistical experiments

## Next Steps

1. **Single Case Validation**: Establish baseline with one test case
2. **Method Comparison**: Compare Tree Method vs Actuated vs Fixed
3. **Statistical Validation**: Run comprehensive analysis across experiments
4. **Publication Report**: Generate publication-ready validation results