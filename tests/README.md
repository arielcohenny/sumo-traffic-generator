# Software Testing Framework

This directory contains all software testing components for the SUMO Traffic Generator project, focused on code quality, correctness, and reliability.

## Directory Structure

```
tests/
├── unit/               # Unit tests for isolated components
├── integration/        # Pipeline step testing
├── system/             # End-to-end scenario testing
├── coverage/           # Coverage data and reports
│   ├── .coverage      # Coverage database
│   ├── htmlcov/       # HTML coverage reports
│   └── README.md      # Coverage documentation
├── utils/              # Test utilities and helpers
├── conftest.py         # Pytest configuration
└── README.md          # This file
```

## Testing Categories

### Unit Tests (`unit/`)

Test isolated components and functions:

- **Purpose**: Verify individual modules work correctly in isolation
- **Scope**: Parsers, validators, utilities, configuration classes
- **Framework**: pytest
- **Coverage**: Individual functions and classes
- **Mock Dependencies**: External services (SUMO, file I/O)

### Integration Tests (`integration/`)

Test individual pipeline steps:

- **Purpose**: Verify each pipeline step works correctly in isolation
- **Scope**: Network generation, lane assignment, traffic generation, simulation execution
- **Framework**: pytest with actual workspace directory
- **Coverage**: Individual pipeline steps and their file outputs

### System Tests (`system/`)

Test complete end-to-end scenarios:

- **Purpose**: Verify full pipeline workflows and realistic scenarios
- **Scope**: Complete SUMO Traffic Generator scenarios from CLAUDE.md
- **Framework**: pytest with performance regression testing
- **Coverage**: Full scenarios, method comparisons, reproducibility

### Coverage Reports (`coverage/`)

Test coverage analysis and reporting:

- **Purpose**: Track code coverage and identify untested areas
- **Location**: `tests/coverage/htmlcov/index.html` - View in browser
- **Configuration**: `.coveragerc` in project root
- **Generated by**: `pytest tests/ --cov=src --cov-report=html`

## Running Tests

### All Tests
```bash
# Run complete test suite
pytest tests/

# Run with coverage reporting
pytest tests/ --cov=src --cov-report=html
```

### Specific Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests (pipeline steps)
pytest tests/integration/

# System tests (full scenarios)
pytest tests/system/

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
open tests/coverage/htmlcov/index.html
```

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

1. **Install Testing Dependencies**:
   ```bash
   pip install pytest pytest-cov pytest-mock
   ```

2. **Run All Tests**:
   ```bash
   pytest tests/
   ```

3. **Generate Coverage Report**:
   ```bash
   pytest tests/ --cov=src --cov-report=html
   open tests/coverage/htmlcov/index.html
   ```

4. **Add New Tests**: Follow the directory structure and naming conventions

5. **Integrate with Development**: Run tests before commits and during code review

For research evaluation and performance benchmarks, see the `evaluation/` directory documentation.