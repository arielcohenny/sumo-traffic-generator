# Test Organization

## Directory Structure

- `zones/` - Zone generation and land use testing
  - Tests for both intelligent zones (OSM) and traditional zones (synthetic)
  - Coordinate system validation
  - Zone type distribution verification

## Test Files

All zone-related test files are organized in `tests/zones/`:

- `test_cli_intelligent_zones.py` - CLI intelligent zone generation tests
- `test_intelligent_zones.py` - Core intelligent zone algorithm tests
- `test_quick.py` - Quick zone generation tests
- `test_zone_fix.py` - Zone fixing and validation tests
- `debug_import.py` - Import debugging utilities
- `test_coordinate_transform.py` - Coordinate transformation tests
- `test_osm_zones.py` - OSM zone generation tests  
- `test_zone_creation.py` - Manual zone creation tests

## Running Tests

```bash
# Run zone tests
cd tests/zones
python test_quick.py

# Run all zone tests
pytest tests/zones/
```