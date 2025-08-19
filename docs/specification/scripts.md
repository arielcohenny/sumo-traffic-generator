# Scripts

## Development and Testing Scripts

- **Purpose**: Provide auxiliary tools and utilities for development, testing, and data management
- **Location**: `tools/scripts/` directory in project root
- **Usage**: Support development workflow and provide verified test data for users


## Testing Infrastructure Scripts

### Test Execution Scripts

**Automated Test Running:**

- **Full Test Suite**: `pytest tests/` - Runs all test categories
- **Coverage Generation**: `pytest tests/ --cov=src --cov-report=html` - Generates coverage reports
- **Category Testing**: Individual test category execution (unit/integration/system)

### Test Utilities

**Test Helper Functions** (`tests/utils/`):

- **Workspace Management**: Consistent directory handling across all test types
- **XML Validation**: SUMO file format validation and metric extraction
- **Custom Assertions**: Specialized assertion classes for traffic simulation validation
- **Golden Master Management**: Performance baseline comparison and updates

### Coverage Organization

**Coverage Configuration** (`.coveragerc`):

- **Source Tracking**: Monitors `src/` directory for code coverage
- **Data Organization**: Stores coverage data in `tests/coverage/` directory
- **Report Generation**: HTML and XML format reports for development and CI/CD