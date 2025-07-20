# Test Fixtures

This directory contains test data and expected results for system testing.

## Files

### `sample_osm.xml`
Minimal OSM file for testing OSM import functionality. Contains:
- 3x3 street grid (9 nodes, 6 ways)
- Mix of primary and secondary roads
- Traffic signal at central intersection
- Sample land use areas for zone testing

### `expected_metrics.json`
Expected performance metrics and bounds for different test scenarios:
- Network topology bounds (edge/junction counts)
- Vehicle performance bounds (departure/completion rates)
- Travel time expectations
- Performance baselines for regression testing

## Usage

These fixtures are automatically loaded by pytest fixtures in `conftest.py`:

```python
# Use sample OSM file
def test_osm_import(test_osm_file):
    # test_osm_file fixture provides path to sample_osm.xml
    pass

# Use expected metrics
def test_performance(expected_files_list):
    # expected_files_list fixture provides file expectations
    pass
```

## Adding New Fixtures

To add new test data:

1. **OSM Files**: Add `.osm` files for different network topologies
2. **Expected Results**: Add `.json` files with metric bounds
3. **Golden Master**: Add reference output files for regression testing
4. **Update Fixtures**: Add corresponding pytest fixtures in `conftest.py`

## Maintenance

- Update expected metrics when baseline performance changes
- Validate OSM files using JOSM or similar tools
- Keep fixtures minimal to ensure fast test execution