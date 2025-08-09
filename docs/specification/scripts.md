# Scripts

## Development and Testing Scripts

- **Purpose**: Provide auxiliary tools and utilities for development, testing, and data management
- **Location**: `tools/scripts/` directory in project root
- **Usage**: Support development workflow and provide verified test data for users

### OSM Sample Data Download Script

- **Script**: `tools/scripts/download_osm_samples.py`
- **Function**: Download verified working OSM areas for testing and demonstration
- **Usage**: `python tools/scripts/download_osm_samples.py`

#### Sample Areas Provided:

- **Manhattan Upper West**: Grid pattern, 300/300 vehicle success rate (40.7800, -73.9850, 40.7900, -73.9750)
- **San Francisco Downtown**: Strong grid layout, 298/300 vehicle success rate (37.7850, -122.4100, 37.7950, -122.4000)
- **Washington DC Downtown**: Planned grid system, 300/300 vehicle success rate (38.8950, -77.0350, 38.9050, -77.0250)

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