# Software Testing Framework

This directory contains all software testing components for the SUMO Traffic Generator project, focused on code quality, correctness, and reliability.

## Directory Structure

```
tests/
├── unit/               # Unit tests for isolated components
├── integration/        # End-to-end pipeline testing
├── validation/         # Domain-specific validation testing
│   └── zones/         # Zone generation and land use validation
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

Test complete pipeline workflows:

- **Purpose**: Verify end-to-end system functionality
- **Scope**: Multi-step workflows, file I/O, SUMO integration
- **Framework**: pytest with temporary directories
- **Coverage**: Pipeline steps, file generation, external tool integration

### Domain Validation Tests (`validation/`)

Test traffic simulation domain logic:

- **Purpose**: Verify simulation realism and domain correctness
- **Scope**: Traffic engineering principles, network topology validation
- **Framework**: pytest with domain-specific assertions

#### Zone Validation (`validation/zones/`)

Existing zone generation and land use testing:

- `test_cli_intelligent_zones.py` - CLI intelligent zone generation tests
- `test_intelligent_zones.py` - Core intelligent zone algorithm tests
- `test_quick.py` - Quick zone generation tests
- `test_coordinate_transform.py` - Coordinate transformation tests
- `test_osm_zones.py` - OSM zone generation tests

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

# Integration tests only  
pytest tests/integration/

# Domain validation tests only
pytest tests/validation/

# Zone validation specifically
pytest tests/validation/zones/
```

### Legacy Zone Test Commands
```bash
# Quick zone tests (existing)
cd tests/validation/zones
python test_quick.py

# All zone tests (existing)
pytest tests/validation/zones/
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

2. **Run Existing Tests**:
   ```bash
   pytest tests/validation/zones/
   ```

3. **Add New Tests**: Follow the directory structure and naming conventions

4. **Integrate with Development**: Run tests before commits and during code review

For research evaluation and performance benchmarks, see the `evaluation/` directory documentation.