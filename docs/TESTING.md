# SUMO Traffic Generator Testing Plan

This document provides a comprehensive overview of the current testing infrastructure, identifies existing issues, and outlines a plan for improving test coverage and reliability.

## Current Testing Overview

### Test Structure
The project currently has **6 test files** organized in a professional pytest-based testing framework:

```
tests/
├── conftest.py                     # Pytest configuration & fixtures
├── unit/
│   └── test_config.py              # Configuration system unit tests
├── integration/
│   └── test_pipeline_steps.py      # Pipeline step integration tests
├── system/
│   ├── test_smoke.py               # Quick validation (< 30s each)
│   ├── test_scenarios.py           # Full scenarios (2-5 min each)
│   └── test_performance.py         # Performance & regression tests
└── utils/
    ├── test_helpers.py             # Test utility functions
    └── assertions.py               # Custom assertion classes
```

### Test Categories & Execution

The testing framework uses **6 pytest markers** for test organization:

```bash
# Quick validation (< 1 minute total)
pytest tests/ -m smoke -v

# Full scenario tests (2-5 minutes each)
pytest tests/ -m scenario -v

# Pipeline integration tests
pytest tests/ -m integration -v

# Unit tests for individual functions
pytest tests/ -m unit -v

# Tree Method sample data tests
pytest tests/ -m tree_method_sample -v

# Long-running performance tests
pytest tests/ -m slow -v
```

### Current Test Coverage

#### ✅ What's Currently Tested:
- **Configuration System**: Unit tests for CONFIG, OSMConfig, NetworkConfig classes
- **Basic Pipeline Execution**: Smoke tests for 3x3 grids with minimal parameters
- **Network Generation**: Synthetic grid creation (netgenerate integration)
- **Edge Splitting with Lanes**: Integrated lane assignment validation
- **File Generation**: XML file creation and structure validation
- **Tree Method Sample Import**: Sample data processing and validation
- **Traffic Control Methods**: Basic validation of tree_method, actuated, fixed
- **CLI Argument Parsing**: Parameter validation and error handling

#### ⚠️ What's Missing or Incomplete:
- **OSM Import Functionality**: No tests for real-world street network import
- **Full Long-Running Scenarios**: Most tests use shortened 30-60 second simulations
- **Comprehensive Performance Baselines**: Performance tests exist but baselines may be outdated
- **Complete Pipeline Sequence**: Integration tests cover individual steps but not full pipeline
- **Traffic Generation Details**: Limited validation of routing strategies and vehicle types
- **Edge Cases**: Error handling, invalid inputs, resource constraints

### Test Infrastructure

#### Fixtures (conftest.py)
Professional fixture system providing:
- **`temp_workspace()`**: Isolated temporary workspace with automatic cleanup
- **`minimal_test_args()`**: Standard CLI arguments for fast tests
- **`config_instance()`**: Access to global CONFIG instance
- **`sample_networks_path()`**: Path to Tree Method sample data
- **`expected_files_list()`**: Expected output files from pipeline

#### Helper Utilities (test_helpers.py)
Comprehensive helper functions:
- **`run_cli_command()`**: Execute CLI with timeout and error handling
- **`validate_output_files()`**: Validate generated files exist and are valid XML
- **`get_simulation_metrics()`**: Extract metrics from simulation output
- **`validate_xml_structure()`**: XML file structure validation
- **`compare_metrics()`**: Compare metrics with tolerance for regression testing

#### Assertion Classes (assertions.py)
High-level assertion methods:
- **`SystemTestAssertions`** class with methods like:
  - `assert_simulation_completed_successfully()`
  - `assert_all_files_generated()`
  - `assert_network_properties()`
  - `assert_vehicle_metrics_within_bounds()`

### How Tests Are Currently Executed

#### Working Commands:
```bash
# Fast feedback loop (development)
pytest tests/ -m "smoke or unit" -v --tb=short

# Pre-commit validation 
pytest tests/ -m "smoke or scenario" -v --timeout=300

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=html -v

# Performance validation
pytest tests/system/test_performance.py -v

# Debug with full output
pytest tests/ -v -s --capture=no --tb=long
```

#### Coverage Reporting:
- **HTML Reports**: `tests/coverage/htmlcov/index.html`
- **Terminal Reports**: Coverage with missing lines
- **70% Minimum**: Currently configured requirement

## Detailed Test Breakdown

### Unit Tests (tests/unit/)

#### test_config.py (4 test classes, 12 test methods)
**Purpose**: Validate configuration system classes and default values

- **TestGlobalConfig** (4 methods):
  - `test_config_instance_exists()`: Verifies CONFIG instance exists with required attributes
  - `test_vehicle_types_configuration()`: Validates vehicle type definitions (passenger, commercial, public)
  - `test_default_values()`: Checks default constants (MIN_LANES=1, MAX_LANES=3, HEAD_DISTANCE=50)
  - `test_file_paths()`: Validates configured file paths (workspace/, grid.net.xml, etc.)

- **TestOSMConfig** (3 methods):
  - `test_osm_config_creation()`: Tests OSMConfig class instantiation with defaults
  - `test_osm_config_with_values()`: Tests custom OSM configuration values
  - `test_highway_types_filter()`: Validates highway type filtering (primary, secondary, residential)

- **TestNetworkConfig** (2 methods):
  - `test_network_config_grid()`: Tests NetworkConfig for synthetic grid networks
  - `test_network_config_osm()`: Tests NetworkConfig for OSM-based networks

- **TestConfigIntegration** (3 methods):
  - `test_global_config_with_network_config()`: Tests CONFIG integration with different network types
  - `test_vehicle_type_distribution()`: Validates default vehicle distribution sums to 100%
  - `test_land_use_configuration()`: Checks land use types and percentages

### System Tests (tests/system/)

#### test_smoke.py (3 test classes, 9 test methods)
**Purpose**: Quick validation tests (< 30 seconds each) for immediate feedback

- **TestQuickValidation** (4 methods):
  - `test_cli_help()`: Validates CLI help command works
  - `test_minimal_synthetic_network()`: Tests 3x3 grid, 10 vehicles, 30s simulation - validates essential files exist
  - `test_tree_method_sample_smoke()`: Quick Tree Method sample validation (30s) - validates network XML structure
  - `test_configuration_validation()`: Tests invalid parameter rejection (grid_dimension=0)

- **TestPipelineSteps** (2 methods):
  - `test_network_generation_step()`: Tests 3x3 grid network file generation
  - `test_traffic_generation_step()`: Tests vehicle route generation with 5 vehicles

- **TestErrorHandling** (3 methods):
  - `test_missing_sample_directory()`: Tests graceful handling of missing Tree Method sample
  - `test_invalid_parameter_combinations()`: Tests vehicle type percentage validation
  - `test_workspace_permissions()`: Validates workspace access and file permissions

#### test_scenarios.py (4 test classes, 11 test methods)
**Purpose**: Full pipeline scenarios based on CLAUDE.md (shortened for testing)

- **TestSyntheticGridScenarios** (3 methods):
  - `test_minimal_grid_smoke()`: 3x3 grid, 50 vehicles, 1 minute - basic pipeline validation
  - `test_morning_rush_scenario()`: 5x5 grid, 200 vehicles, 5 minutes - six_periods departure pattern
  - `test_evening_light_traffic()`: 5x5 grid, 150 vehicles, 5 minutes - uniform departure, evening start
  - `test_multi_modal_traffic_mix()`: Vehicle types (50% passenger, 40% commercial, 10% public), routing strategies

- **TestTreeMethodSample** (2 methods):
  - `test_tree_method_sample_basic()`: Pre-built network validation, 3 minutes simulation
  - `test_tree_method_sample_comparison()`: Parametrized test comparing tree_method, actuated, fixed on same network

- **TestTrafficControlComparison** (2 methods):
  - `test_traffic_control_methods()`: Parametrized test (fixed, actuated, tree_method) with identical conditions
  - `test_tree_method_integration()`: Specific Tree Method integration test with realtime routing

- **TestRegressionScenarios** (2 methods):
  - `test_performance_bounds_regression()`: 5x5 grid, 300 vehicles - checks travel time < 400s, completion rate > 70%
  - `test_reproducibility()`: Runs same scenario twice with fixed seed, compares metrics

#### test_performance.py (4 test classes, 12 test methods)
**Purpose**: Performance validation and regression detection

- **TestPerformanceBaselines** (2 methods):
  - `test_3x3_grid_performance_baseline()`: 50 vehicles, 60s - execution time < 30s, compares vs golden master
  - `test_5x5_grid_performance_baseline()`: 200 vehicles, 300s - execution time < 120s, 15% tolerance

- **TestTrafficControlPerformance** (2 methods):
  - `test_traffic_control_performance()`: Parametrized performance test for each control method
  - `test_tree_method_vs_baseline_performance()`: Direct comparison with identical conditions, measures improvement percentages

- **TestScalabilityPerformance** (2 methods, marked @slow):
  - `test_network_size_scaling()`: Tests 3x3 (50v), 4x4 (100v), 5x5 (200v) grids - execution time bounds
  - `test_vehicle_count_scaling()`: Tests 100, 200, 400 vehicles on 5x5 grid - 0.3s per vehicle max

- **TestMemoryPerformance** (2 methods):
  - `test_memory_efficiency()`: Monitors memory usage, < 200MB increase limit
  - `test_tree_method_sample_performance()`: Sample network performance, < 90s execution time

### Integration Tests (tests/integration/)

#### test_pipeline_steps.py (7 test classes, 13 test methods)
**Purpose**: Individual pipeline step validation

- **TestNetworkGeneration** (1 method):
  - `test_synthetic_grid_generation()`: Step 1 validation - 4x4 grid, checks network files exist and structure

- **TestEdgeSplittingWithLanes** (2 methods):
  - `test_edge_splitting_realistic_lanes()`: Step 2 validation with realistic lane assignment, > 20 lanes expected
  - `test_lane_assignment_modes()`: Parametrized test for "2", "random", "realistic" modes

- **TestTrafficGeneration** (2 methods):
  - `test_vehicle_route_generation()`: Steps 4-6 validation - 20 vehicles, passenger/commercial types
  - `test_departure_patterns()`: Parametrized test for "uniform", "six_periods" patterns

- **TestTrafficLightInjection** (1 method):
  - `test_traffic_light_strategies()`: Step 5 validation - "opposites", "incoming" strategies

- **TestSimulationExecution** (1 method):
  - `test_simulation_with_traffic_control()`: Step 7 validation - parametrized for all control methods

- **TestTreeMethodSampleIntegration** (1 method):
  - `test_sample_network_processing()`: Tree Method sample processing validation

- **TestPipelineSequence** (2 methods):
  - `test_complete_pipeline_synthetic()`: Full 7-step pipeline with all features enabled
  - `test_pipeline_reproducibility()`: Identical parameters test, compares metrics between runs

### Key Validation Points

#### File Generation Validation:
- **Essential Files**: grid.net.xml, vehicles.rou.xml, grid.sumocfg
- **Network Files**: grid.nod.xml, grid.edg.xml, grid.con.xml, grid.tll.xml
- **Optional Files**: zones.poly.xml (when zones enabled)

#### Performance Bounds:
- **3x3 Grid**: < 30s execution, 50 vehicles
- **5x5 Grid**: < 120s execution, 200+ vehicles  
- **Tree Method vs Fixed**: 20-45% improvement expected
- **Memory Usage**: < 200MB increase limit

#### Network Validation:
- **Edge Count**: Minimum thresholds based on grid size
- **Lane Assignment**: Multiple lanes present (> 10-20 depending on test)
- **XML Structure**: Valid XML with required elements

#### Traffic Validation:
- **Vehicle Types**: passenger, commercial, public distribution
- **Departure Patterns**: uniform, six_periods timing
- **Routing Strategies**: shortest, realtime combinations

## Issues Identified

### 1. Test Execution Problems
Based on your feedback about "not knowing what's going on":

#### Potential Issues:
- **Silent Failures**: Tests may pass but not actually validate expected behavior
- **Unclear Error Messages**: When tests fail, the cause may not be obvious
- **Missing Documentation**: Test purposes and expected outcomes not well documented
- **Timeout Issues**: Long-running tests may timeout without clear indication

#### Git Integration Issues:
- **Pre-commit Hooks**: No automated test execution before commits
- **CI/CD Integration**: No continuous integration setup visible
- **Test Selection**: Unclear which tests should run for different changes

### 2. Test Coverage Gaps

#### Missing Core Functionality:
- **OSM Import Pipeline**: Real-world street network testing
- **Full Scenario Validation**: Most CLAUDE.md scenarios not fully tested
- **Tree Method Algorithm**: Limited validation of actual algorithm behavior
- **Performance Regression**: Baselines may be stale or missing

#### Insufficient Edge Case Testing:
- **Invalid Parameters**: Limited testing of error conditions
- **Resource Constraints**: No testing of memory/disk limitations
- **Concurrent Execution**: No testing of multiple simultaneous runs

### 3. Test Reliability Issues

#### Potential Reliability Problems:
- **Non-Deterministic Results**: Random elements may cause test flakiness
- **Environment Dependencies**: Tests may depend on specific SUMO versions or system configuration
- **Workspace Conflicts**: Tests may interfere with each other or main workspace
- **Sample Data Dependencies**: Tree Method tests depend on external data that may be missing

## Improvement Plan

### Phase 1: Immediate Fixes (Week 1)
**Goal**: Make existing tests reliable and understandable

1. **Test Documentation Enhancement**
   - Add clear docstrings to all test methods explaining purpose and expected outcomes
   - Document what each test validates and why it matters
   - Add inline comments for complex test logic

2. **Error Message Improvement**
   - Enhance assertion messages to clearly indicate what failed and why
   - Add debugging output for test failures
   - Improve timeout handling with better error messages

3. **Test Reliability Fixes**
   - Ensure all tests use fixed seeds for deterministic results
   - Fix workspace isolation issues
   - Add better cleanup and setup procedures

### Phase 2: Coverage Expansion (Week 2-3)
**Goal**: Fill critical testing gaps

1. **OSM Import Testing**
   - Create OSM test data files
   - Add OSM import pipeline tests
   - Validate OSM-specific functionality

2. **Full Scenario Testing**
   - Implement all 14 verified scenarios from CLAUDE.md
   - Add longer-running scenario tests with proper timeouts
   - Validate end-to-end pipeline behavior

3. **Tree Method Algorithm Testing**
   - Add specific Tree Method algorithm validation
   - Compare Tree Method vs baseline methods
   - Validate performance improvements

### Phase 3: Performance & Regression (Week 4)
**Goal**: Establish reliable performance baselines

1. **Performance Baseline Creation**
   - Run comprehensive performance tests to establish current baselines
   - Document expected performance characteristics
   - Set up regression detection thresholds

2. **Golden Master Testing**
   - Create golden master files for known-good outputs
   - Add regression testing against golden masters
   - Validate output consistency across runs

### Phase 4: CI/CD Integration (Week 5)
**Goal**: Integrate testing into development workflow

1. **Pre-commit Integration**
   - Set up pre-commit hooks to run smoke tests
   - Add automated test selection based on changed files
   - Prevent commits that break basic functionality

2. **Continuous Integration**
   - Set up GitHub Actions or similar CI system
   - Define test tiers (smoke, full, performance)
   - Add automated test reporting

## Recommended Test Execution Strategy

### For Development:
```bash
# Before making changes
pytest tests/ -m smoke -v

# After making changes
pytest tests/ -m "smoke or unit" -v

# Before committing
pytest tests/ -m "smoke or scenario" -v --timeout=600
```

### For Releases:
```bash
# Full validation
pytest tests/ --cov=src --cov-report=html -v

# Performance validation
pytest tests/ -m "scenario or integration" -v

# Generate coverage report
open tests/coverage/htmlcov/index.html
```

### For Debugging:
```bash
# Single test with full output
pytest tests/system/test_smoke.py::TestQuickValidation::test_minimal_synthetic_network -v -s

# All output with timing
pytest tests/ -v -s --durations=10 --tb=long
```

## Success Metrics

### Short Term (1-2 weeks):
- [ ] All existing tests run reliably without false positives/negatives
- [ ] Clear error messages when tests fail
- [ ] Documentation explains what each test validates
- [ ] Pre-commit hooks prevent broken commits

### Medium Term (3-4 weeks):
- [ ] OSM import functionality fully tested
- [ ] All CLAUDE.md scenarios have corresponding tests
- [ ] Performance baselines established and monitored
- [ ] 80%+ code coverage with meaningful tests

### Long Term (1-2 months):
- [ ] Continuous integration running all test tiers
- [ ] Automated performance regression detection
- [ ] New features automatically include comprehensive tests
- [ ] Test suite provides confidence for releases

## Next Steps

1. **Review Current Test Failures**: Run the full test suite and document any failures
2. **Prioritize Critical Issues**: Focus on tests that should work but don't
3. **Enhance Documentation**: Make it clear what each test does and why
4. **Fix Reliability Issues**: Ensure tests produce consistent results
5. **Expand Coverage**: Add missing functionality testing

This plan provides a roadmap for transforming the test suite from its current state into a reliable, comprehensive testing framework that supports confident development and releases.