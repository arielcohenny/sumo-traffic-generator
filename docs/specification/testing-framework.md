# Software Testing Framework

## Testing Framework Overview

**Purpose of Professional Test Suite:**

- **Code Quality**: Ensures implementation correctness and maintainability
- **Regression Prevention**: Catches issues introduced by code changes
- **Development Support**: Provides automated validation for complex pipeline operations
- **Performance Monitoring**: Tracks system performance and prevents degradation
- **Documentation**: Serves as executable examples of system usage

**Framework Architecture:**

- **Location**: `tests/` directory in project root
- **Framework**: pytest with professional configuration and plugins
- **Coverage**: pytest-cov for comprehensive code coverage analysis
- **Structure**: Three-tier architecture (unit/integration/system tests)
- **Scope**: Complete SUMO Traffic Generator pipeline validation

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test isolated components and functions

- **Scope**: Parsers, validators, utilities, configuration classes
- **Framework**: pytest with mocking for external dependencies
- **Coverage**: Individual functions and classes
- **Mock Dependencies**: External services (SUMO, file I/O)

### Integration Tests (`tests/integration/`)

**Purpose**: Test individual pipeline steps in isolation

- **Scope**: Network generation, lane assignment, traffic generation, simulation execution
- **Framework**: pytest with actual workspace directory
- **Coverage**: Individual pipeline steps and their file outputs
- **Validation**: File generation, format correctness, parameter handling

**Key Integration Tests:**

- **Network Generation**: Grid creation and OSM import validation
- **Lane Assignment**: Edge splitting and lane configuration testing
- **Traffic Generation**: Route and vehicle file creation
- **Pipeline Steps**: Each of the 9 pipeline steps tested individually

### System Tests (`tests/system/`)

**Purpose**: Test complete end-to-end scenarios

- **Scope**: Complete SUMO Traffic Generator scenarios from CLAUDE.md
- **Framework**: pytest with performance regression testing
- **Coverage**: Full scenarios, method comparisons, reproducibility

**Scenario Testing (14 verified scenarios):**

- **Basic Scenarios**: Grid dimensions 3x3, 5x5, 7x7 with various vehicle counts
- **Traffic Control Comparisons**: Tree Method vs Actuated vs Fixed timing
- **OSM Network Testing**: Real-world street network validation
- **Tree Method Samples**: Research benchmark validation
- **Performance Testing**: Travel time and completion rate analysis

## Test Infrastructure

### Coverage Reporting

**Configuration**: `.coveragerc` in project root with organized data storage

```ini
[run]
source = src
data_file = tests/coverage/.coverage
parallel = true

[html]
directory = tests/coverage/htmlcov
show_contexts = true
```

**Coverage Organization:**

- **Data Storage**: `tests/coverage/.coverage` - SQLite coverage database
- **HTML Reports**: `tests/coverage/htmlcov/` - Browser-viewable coverage reports
- **XML Reports**: `tests/coverage/coverage.xml` - CI/CD integration format

### Golden Master Testing

**Performance Baselines**: JSON files with expected performance metrics

- **`tests/system/fixtures/golden_master_3x3.json`**: 3x3 grid baseline (48 edges, 33 junctions)
- **`tests/system/fixtures/golden_master_5x5.json`**: 5x5 grid baseline (152 edges, 100 junctions)

**Regression Detection**: Automatically detects performance degradation

### Test Utilities

**Helper Functions** (`tests/utils/`):

- **`test_helpers.py`**: Workspace management, XML parsing, metric extraction
- **`assertions.py`**: Custom assertion classes for file validation
- **`conftest.py`**: pytest configuration and shared fixtures

**Workspace Management:**

- **Consistent Directory Handling**: All tests use actual `workspace/` directory
- **File Validation**: Automated checking of generated SUMO files
- **Cleanup**: Automatic workspace cleanup between tests

## Running Tests

### Complete Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html

# View coverage report
open tests/coverage/htmlcov/index.html
```

### Category-Specific Testing

```bash
# Unit tests only
pytest tests/unit/

# Integration tests (pipeline steps)
pytest tests/integration/

# System tests (full scenarios)
pytest tests/system/

# Performance tests
pytest tests/system/test_performance.py
```

### Test Markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run system tests with detailed output
pytest tests/system/ -v

# Run specific scenario tests
pytest tests/system/test_scenarios.py::test_basic_3x3_grid -v
```

## Test Scenarios

### Synthetic Grid Scenarios

**Basic Grid Tests:**

- **3x3 Grid**: 150 vehicles, basic functionality validation
- **5x5 Grid**: 500 vehicles, moderate complexity testing
- **7x7 Grid**: 1000 vehicles, high complexity validation

**Traffic Control Method Comparisons:**

- **Tree Method**: Dynamic decentralized traffic control
- **SUMO Actuated**: Gap-based signal control
- **Fixed Timing**: Static signal phases

### OSM Network Scenarios

**Real-World Testing:**

- **Manhattan East Village**: 500 vehicles on real street network
- **Complex Intersections**: Dead-end streets and irregular topology
- **Signal Preservation**: Maintains original OSM traffic light timing

### Tree Method Sample Validation

**Research Benchmark Testing:**

- **Sample Networks**: Pre-built complex urban networks (946 vehicles)
- **Method Comparison**: Tree Method vs baseline methods on identical conditions
- **Performance Validation**: Confirms Tree Method implementation correctness

## Performance Testing

### Travel Time Analysis

**Metrics Tracked:**

- **Average Travel Time**: Mean vehicle travel time across simulation
- **Completion Rates**: Percentage of vehicles reaching destination
- **Throughput**: Vehicle arrivals and departures per time unit

### Regression Detection

**Golden Master Comparison:**

- **Baseline Validation**: Compares current performance against established baselines
- **Tolerance Ranges**: Allows for acceptable variation while detecting significant changes
- **Automatic Alerts**: Test failures when performance degrades beyond thresholds

## Framework Integration

**Clear Separation of Concerns:**

- **`tests/`**: Software engineering quality assurance
  - Code correctness and reliability
  - Regression prevention
  - Development workflow support

- **`evaluation/`**: Research validation and performance analysis
  - Statistical benchmarks and comparisons
  - Publication-ready experimental results
  - Dataset management for reproducible research

Both frameworks complement each other but serve different purposes in maintaining a high-quality research codebase.

## Getting Started

### Installation

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-mock

# Verify test environment
pytest --version
```

### First Test Run

```bash
# Run complete test suite
pytest tests/

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View results
open tests/coverage/htmlcov/index.html
```

### Development Workflow

1. **Write Tests First**: Follow test-driven development practices
2. **Run Tests Frequently**: Execute relevant tests during development
3. **Check Coverage**: Ensure new code has appropriate test coverage
4. **Validate Performance**: Run performance tests for changes affecting algorithms
5. **Pre-Commit Testing**: Run full test suite before code commits

## Test Maintenance

### Adding New Tests

**Guidelines:**

- **Follow Directory Structure**: Place tests in appropriate category (unit/integration/system)
- **Use Naming Conventions**: Test files start with `test_`, test functions start with `test_`
- **Include Documentation**: Add docstrings explaining test purpose and validation
- **Update Baselines**: Modify golden master files when intentional changes occur

### Continuous Integration

**CI/CD Integration:**

- **Automated Testing**: All tests run automatically on code changes
- **Coverage Requirements**: Maintain minimum coverage thresholds
- **Performance Monitoring**: Track performance trends over time
- **Failure Notifications**: Immediate alerts for test failures or performance regression