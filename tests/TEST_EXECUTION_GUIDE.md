# SUMO Traffic Generator Test Suite

Professional test suite for the SUMO Traffic Generator with comprehensive coverage of synthetic grid networks and Tree Method algorithm validation.

## 🚀 Quick Start

### Prerequisites

Ensure you have:

- Python 3.9+ with virtual environment activated
- SUMO installed and accessible (`sumo`, `netgenerate`, `netconvert` in PATH)
- All project dependencies installed (`pip install -r requirements.txt`)
- pytest and testing dependencies (`pip install pytest pytest-cov pytest-timeout pytest-xdist`)

### Basic Test Execution

```bash
# Run all fast tests (< 1 minute each)
pytest tests/ -m smoke -v

# Run scenario tests (2-5 minutes each)
pytest tests/ -m scenario -v

# Run integration tests (CLI arguments)
pytest tests/ -m integration -v

# Run complete test suite with coverage
pytest tests/ --cov=src --cov-report=html -v
```

## 📋 Test Categories

### 🔥 Smoke Tests (Ultra-Fast: < 30 seconds each)

Quick validation tests for immediate feedback:

```bash
# All smoke tests
pytest tests/ -m smoke -v

# Specific smoke test categories
pytest tests/system/test_smoke.py::TestQuickValidation::test_cli_help -v
pytest tests/system/test_smoke.py::TestQuickValidation::test_minimal_synthetic_network -v
pytest tests/system/test_smoke.py::TestQuickValidation::test_tree_method_sample_smoke -v
```

### 🎯 Scenario Tests (Medium: 2-5 minutes each)

Full pipeline validation based on proven CLAUDE.md scenarios:

```bash
# All scenario tests
pytest tests/ -m scenario -v

# Synthetic grid scenarios
pytest tests/system/test_scenarios.py::TestSyntheticGridScenarios -v

# Tree Method sample scenarios
pytest tests/system/test_scenarios.py::TestTreeMethodSample -v

# Traffic control method comparison
pytest tests/system/test_scenarios.py::TestTrafficControlComparison -v
```

### 🔧 Integration Tests (Medium: 1-3 minutes each)

Pipeline step validation and component interactions:

```bash
# All integration tests
pytest tests/ -m integration -v

# Individual pipeline steps
pytest tests/integration/test_pipeline_steps.py::TestNetworkGeneration -v
pytest tests/integration/test_pipeline_steps.py::TestEdgeSplittingWithLanes -v
pytest tests/integration/test_pipeline_steps.py::TestTrafficGeneration -v

# Complete pipeline sequence
pytest tests/integration/test_pipeline_steps.py::TestPipelineSequence -v

# CLI Argument Tests (82 parametrized tests)
pytest tests/integration/test_cli_arguments.py -v

# Specific argument categories
pytest tests/integration/test_cli_arguments.py::TestNetworkArguments -v
pytest tests/integration/test_cli_arguments.py::TestTrafficArguments -v
pytest tests/integration/test_cli_arguments.py::TestSimulationArguments -v
pytest tests/integration/test_cli_arguments.py::TestZoneArguments -v
pytest tests/integration/test_cli_arguments.py::TestTrafficControlArguments -v
```

**Note:** CLI argument tests use `--file-generation-only` flag to run steps 1-7 without SUMO simulation, enabling fast validation.

### ⚡ Performance Tests (Extended: 2-10 minutes each)

Performance baselines and regression detection:

```bash
# Performance baseline tests
pytest tests/system/test_performance.py::TestPerformanceBaselines -v

# Traffic control performance comparison
pytest tests/system/test_performance.py::TestTrafficControlPerformance -v

# Scalability tests (slower)
pytest tests/ -m slow -v
```

### 🧪 Unit Tests (Fast: < 5 seconds each)

Individual function and class testing:

```bash
# All unit tests
pytest tests/ -m unit -v

# Configuration system tests
pytest tests/unit/test_config.py -v
```

## 🎛️ Test Execution Options

### By Test Type

```bash
# Fast feedback loop (development)
pytest tests/ -m "smoke or unit" -v --tb=short

# Medium validation (pre-commit)
pytest tests/ -m "smoke or scenario" -v --timeout=300

# Full validation (CI/CD)
pytest tests/ -v --cov=src --cov-report=html --timeout=600

# Performance validation
pytest tests/ -m "scenario and not slow" -v --timeout=900
```

### By Network Type

```bash
# Synthetic grid networks only
pytest tests/ -k "synthetic or grid" -v

# Tree Method sample networks only
pytest tests/ -m tree_method_sample -v

# Traffic control method comparison
pytest tests/ -k "traffic_control" -v
```

### By Component

```bash
# Network generation pipeline
pytest tests/ -k "network" -v

# Traffic generation pipeline
pytest tests/ -k "traffic" -v

# Simulation execution
pytest tests/ -k "simulation" -v

# Configuration system
pytest tests/ -k "config" -v
```

## 🔍 Debugging and Analysis

### Verbose Output

```bash
# Maximum verbosity with timing
pytest tests/ -v -s --durations=10

# Show all print statements
pytest tests/ -v -s --capture=no

# Detailed failure information
pytest tests/ -v --tb=long
```

### Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Coverage with branch analysis
pytest tests/ --cov=src --cov-branch --cov-report=html
```

### Performance Profiling

```bash
# Test execution timing
pytest tests/ --durations=0

# Memory usage monitoring
pytest tests/system/test_performance.py::TestMemoryPerformance -v -s

# Performance comparison analysis
pytest tests/system/test_performance.py::TestTrafficControlPerformance -v -s
```

## 🏗️ Development Workflow

### Pre-Commit Testing

```bash
# Quick validation before commit
pytest tests/ -m smoke --timeout=60 -v
```

### Feature Development

```bash
# Test specific feature during development
pytest tests/ -k "feature_name" -v -s

# Test with specific configuration
pytest tests/system/test_scenarios.py::TestSyntheticGridScenarios::test_minimal_grid_smoke -v -s
```

### Performance Regression Testing

```bash
# Run baseline performance tests
pytest tests/system/test_performance.py::TestPerformanceBaselines -v

# Compare traffic control methods
pytest tests/system/test_performance.py::TestTrafficControlPerformance -v
```

## 📊 Continuous Integration

### GitHub Actions Workflow

The project includes automated testing with three tiers:

1. **Fast Tests** (2-3 minutes): Smoke tests + unit tests
2. **Medium Tests** (10-15 minutes): Scenario tests
3. **Performance Tests** (30+ minutes): Full benchmark subset

### Local CI Simulation

```bash
# Simulate CI fast tier
pytest tests/ -m "smoke or unit" --timeout=180 -v

# Simulate CI medium tier
pytest tests/ -m "scenario and not slow" --timeout=900 -v

# Simulate CI performance tier
pytest tests/ -m "scenario or integration" --timeout=1800 -v
```

## 📁 Test Structure

```
tests/
├── conftest.py                     # Pytest configuration & fixtures
├── system/                         # End-to-end system tests
│   ├── test_scenarios.py           # Main scenario validation
│   ├── test_smoke.py               # Quick validation tests
│   ├── test_performance.py         # Performance & regression tests
│   └── fixtures/                   # Test data & golden masters
├── integration/                    # Pipeline integration tests
│   ├── test_pipeline_steps.py      # Individual step validation
│   └── test_cli_arguments.py       # CLI argument validation (82 tests)
├── unit/                           # Unit tests
│   └── test_config.py              # Configuration system tests
├── utils/                          # Test utilities
│   ├── test_helpers.py             # Common helper functions
│   └── assertions.py               # Custom assertion classes
└── validation/                     # Domain validation tests (existing)
```

## 🎯 Test Focus Areas

This test suite focuses on:

✅ **Synthetic Grid Networks**: 3x3 to 7x7 grids with comprehensive parameter testing
✅ **Tree Method Algorithm**: Sample data validation and performance comparison  
✅ **Traffic Control Methods**: tree_method, actuated, fixed comparison
✅ **Pipeline Integration**: 7-step pipeline validation
✅ **Performance Regression**: Baseline tracking and improvement validation
✅ **Configuration Validation**: Parameter parsing and validation
✅ **Reproducibility**: Fixed-seed consistency testing

## 🔧 Troubleshooting

### Common Issues

**SUMO Not Found**:

```bash
# Check SUMO installation
which sumo
echo $SUMO_HOME

# Install SUMO (Ubuntu/Debian)
sudo apt-get install sumo sumo-tools sumo-doc
```

**Import Errors**:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Install test dependencies
pip install pytest pytest-cov pytest-timeout pytest-xdist
```

**Timeout Issues**:

```bash
# Increase timeout for slow tests
pytest tests/ --timeout=1200 -v

# Run without timeout
pytest tests/ --timeout=0 -v
```

**Tree Method Sample Missing**:

```bash
# Ensure sample data exists
ls evaluation/datasets/networks/

# Skip sample tests if needed
pytest tests/ -m "not tree_method_sample" -v
```

### Test Environment Validation

```bash
# Validate test environment
pytest tests/system/test_smoke.py::TestQuickValidation::test_cli_help -v

# Check configuration loading
pytest tests/unit/test_config.py::TestGridConfig::test_valid_grid_config -v

# Verify SUMO integration
pytest tests/system/test_smoke.py::TestQuickValidation::test_minimal_synthetic_network -v
```

## 📈 Expected Results

### Performance Baselines

- **3x3 Grid (50 vehicles, 60s)**: < 30 seconds execution
- **5x5 Grid (200 vehicles, 300s)**: < 120 seconds execution
- **Tree Method vs Fixed**: 20-45% travel time improvement
- **Tree Method vs Actuated**: 10-25% travel time improvement

### Success Metrics

- **Smoke Tests**: 100% pass rate, < 60 seconds total
- **Scenario Tests**: 100% pass rate, < 30 minutes total
- **Integration Tests**: 100% pass rate, individual step validation
- **Performance Tests**: Within baseline tolerance (15% regression threshold)

## 🎉 Getting Started Checklist

1. ✅ **Environment Setup**: Virtual environment, SUMO, dependencies
2. ✅ **Quick Validation**: `pytest tests/ -m smoke -v`
3. ✅ **Scenario Testing**: `pytest tests/ -m scenario -v`
4. ✅ **Full Test Suite**: `pytest tests/ --cov=src -v`
5. ✅ **Performance Check**: `pytest tests/system/test_performance.py -v`

---

**Ready to test?** Start with: `pytest tests/ -m smoke -v`
