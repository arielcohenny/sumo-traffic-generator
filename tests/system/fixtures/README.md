# Test Fixtures

This directory contains test data and expected results for system testing.

## Files

### `expected_metrics.json`
Expected performance metrics and bounds for different test scenarios:
- Network topology bounds (edge/junction counts)
- Vehicle performance bounds (departure/completion rates)
- Travel time expectations
- Performance baselines for regression testing

## Usage

These fixtures are automatically loaded by pytest fixtures in `conftest.py`:

```python
# Use expected metrics
def test_performance(expected_metrics):
    # expected_metrics fixture provides performance bounds
    pass

# Use expected metrics
def test_performance(expected_files_list):
    # expected_files_list fixture provides file expectations
    pass
```

## Adding New Fixtures

To add new test data:

1. **Network Files**: Add sample network files for different test scenarios
2. **Expected Results**: Add `.json` files with metric bounds
3. **Golden Master**: Add reference output files for regression testing
4. **Update Fixtures**: Add corresponding pytest fixtures in `conftest.py`

## Maintenance

- Update expected metrics when baseline performance changes
- Validate network files using SUMO tools
- Keep fixtures minimal to ensure fast test execution