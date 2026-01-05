# Custom Departure Pattern Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `rush_hours` pattern with a flexible `custom` pattern that uses absolute clock times and percentage allocations.

**Architecture:** Modify the existing departure pattern system to support `custom:HH:MM-HH:MM,percent;...` syntax. Times are absolute clock times within the simulation window. Unspecified percentage is distributed proportionally across remaining time gaps.

**Tech Stack:** Python, argparse, pytest

---

## Task 1: Add Constants

**Files:**
- Modify: `src/constants.py:436-451`

**Step 1: Update constants**

Replace the rush_hours constants with custom pattern constants:

```python
# Find these lines (around 436-451):
DEPARTURE_PATTERN_UNIFORM = "uniform"
DEPARTURE_PATTERN_SIX_PERIODS = "six_periods"
DEPARTURE_PATTERN_RUSH_HOURS = "rush_hours"
# ...
RUSH_HOURS_PREFIX = "rush_hours:"
RUSH_HOURS_REST = "rest"
RUSH_HOURS_SEPARATOR = ":"

# Replace with:
DEPARTURE_PATTERN_UNIFORM = "uniform"
DEPARTURE_PATTERN_SIX_PERIODS = "six_periods"
DEPARTURE_PATTERN_CUSTOM = "custom"

CUSTOM_PATTERN_PREFIX = "custom:"
CUSTOM_WINDOW_SEPARATOR = ";"
CUSTOM_TIME_PERCENT_SEPARATOR = ","
CUSTOM_TIME_RANGE_SEPARATOR = "-"
```

**Step 2: Commit**

```bash
git add src/constants.py
git commit -m "refactor: replace rush_hours constants with custom pattern constants"
```

---

## Task 2: Write Validation Tests

**Files:**
- Create: `tests/unit/test_custom_departure_pattern.py`

**Step 1: Create test file with validation tests**

```python
"""Tests for custom departure pattern validation and calculation."""

import pytest
from unittest.mock import MagicMock
from src.validate.validate_arguments import (
    _validate_departure_pattern,
    _validate_custom_pattern,
    _parse_custom_pattern,
)
from src.validate.errors import ValidationError


class TestCustomPatternValidation:
    """Tests for custom departure pattern validation."""

    def test_valid_single_window(self):
        """Single window with valid format."""
        _validate_custom_pattern("custom:9:00-9:30,40", start_hour=8.0, end_time=18000)

    def test_valid_multiple_windows(self):
        """Multiple windows with valid format."""
        _validate_custom_pattern(
            "custom:9:00-9:30,40;10:00-10:45,30",
            start_hour=8.0,
            end_time=18000
        )

    def test_valid_trailing_semicolon(self):
        """Trailing semicolon should be allowed."""
        _validate_custom_pattern(
            "custom:9:00-9:30,40;10:00-10:45,30;",
            start_hour=8.0,
            end_time=18000
        )

    def test_error_percentage_exceeds_100(self):
        """Error when percentages sum > 100."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:00-9:30,60;10:00-10:45,50",
                start_hour=8.0,
                end_time=18000
            )
        assert "sum to 110%" in str(exc_info.value)

    def test_error_window_before_start(self):
        """Error when window starts before simulation."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:7:00-8:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "outside simulation range" in str(exc_info.value)

    def test_error_window_after_end(self):
        """Error when window extends past simulation end."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:12:00-14:00,40",
                start_hour=8.0,
                end_time=18000  # 5 hours = ends at 13:00
            )
        assert "outside simulation range" in str(exc_info.value)

    def test_error_overlapping_windows(self):
        """Error when windows overlap."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:00-10:00,40;9:30-10:30,30",
                start_hour=8.0,
                end_time=18000
            )
        assert "overlap" in str(exc_info.value)

    def test_error_start_after_end(self):
        """Error when window start is after end."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:30-9:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "start time must be before end time" in str(exc_info.value)

    def test_error_invalid_time_format(self):
        """Error on invalid time format."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:75-10:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "minutes must be 0-59" in str(exc_info.value)

    def test_error_negative_percentage(self):
        """Error on negative percentage."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:00-9:30,-10",
                start_hour=8.0,
                end_time=18000
            )
        assert "positive" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


class TestCustomPatternParsing:
    """Tests for parsing custom pattern syntax."""

    def test_parse_single_window(self):
        """Parse single window correctly."""
        windows = _parse_custom_pattern("custom:9:00-9:30,40")
        assert len(windows) == 1
        assert windows[0] == {"start": (9, 0), "end": (9, 30), "percent": 40}

    def test_parse_multiple_windows(self):
        """Parse multiple windows correctly."""
        windows = _parse_custom_pattern("custom:9:00-9:30,40;10:00-10:45,30")
        assert len(windows) == 2
        assert windows[0] == {"start": (9, 0), "end": (9, 30), "percent": 40}
        assert windows[1] == {"start": (10, 0), "end": (10, 45), "percent": 30}

    def test_parse_ignores_trailing_semicolon(self):
        """Trailing semicolon produces same result."""
        windows1 = _parse_custom_pattern("custom:9:00-9:30,40")
        windows2 = _parse_custom_pattern("custom:9:00-9:30,40;")
        assert windows1 == windows2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_custom_departure_pattern.py -v
```

Expected: FAIL with ImportError (functions don't exist yet)

**Step 3: Commit test file**

```bash
git add tests/unit/test_custom_departure_pattern.py
git commit -m "test: add validation tests for custom departure pattern"
```

---

## Task 3: Implement Pattern Parsing

**Files:**
- Modify: `src/validate/validate_arguments.py`

**Step 1: Add imports at top of file**

```python
# Add to existing imports from src.constants:
from src.constants import (
    # ... existing imports ...
    CUSTOM_PATTERN_PREFIX,
    CUSTOM_WINDOW_SEPARATOR,
    CUSTOM_TIME_PERCENT_SEPARATOR,
    CUSTOM_TIME_RANGE_SEPARATOR,
)
```

**Step 2: Add parsing function**

Add after the existing `_validate_rush_hours_pattern` function (around line 320):

```python
def _parse_custom_pattern(pattern: str) -> list:
    """
    Parse custom departure pattern into list of time windows.

    Args:
        pattern: Pattern string like "custom:9:00-9:30,40;10:00-10:45,30"

    Returns:
        List of dicts: [{"start": (h, m), "end": (h, m), "percent": int}, ...]
    """
    if not pattern.startswith(CUSTOM_PATTERN_PREFIX):
        raise ValidationError(f"Custom pattern must start with '{CUSTOM_PATTERN_PREFIX}'")

    pattern_body = pattern[len(CUSTOM_PATTERN_PREFIX):]
    windows = []

    # Split by semicolon, filter empty parts (handles trailing semicolon)
    parts = [p.strip() for p in pattern_body.split(CUSTOM_WINDOW_SEPARATOR) if p.strip()]

    for part in parts:
        # Split time range from percentage: "9:00-9:30,40"
        if CUSTOM_TIME_PERCENT_SEPARATOR not in part:
            raise ValidationError(
                f"Invalid window format '{part}': expected 'HH:MM-HH:MM,percent'"
            )

        time_range, percent_str = part.rsplit(CUSTOM_TIME_PERCENT_SEPARATOR, 1)

        # Parse percentage
        try:
            percent = int(percent_str)
        except ValueError:
            raise ValidationError(f"Invalid percentage '{percent_str}': must be an integer")

        if percent <= 0:
            raise ValidationError(f"Percentage must be positive, got {percent}")

        # Split time range: "9:00-9:30"
        if CUSTOM_TIME_RANGE_SEPARATOR not in time_range:
            raise ValidationError(
                f"Invalid time range '{time_range}': expected 'HH:MM-HH:MM'"
            )

        start_str, end_str = time_range.split(CUSTOM_TIME_RANGE_SEPARATOR, 1)

        # Parse times
        start_time = _parse_time_hhmm(start_str)
        end_time = _parse_time_hhmm(end_str)

        windows.append({
            "start": start_time,
            "end": end_time,
            "percent": percent
        })

    return windows


def _parse_time_hhmm(time_str: str) -> tuple:
    """
    Parse HH:MM time string to (hour, minute) tuple.

    Args:
        time_str: Time string like "9:00" or "14:30"

    Returns:
        Tuple of (hour, minute) as integers
    """
    time_str = time_str.strip()

    if ":" not in time_str:
        raise ValidationError(f"Invalid time format '{time_str}': expected HH:MM")

    parts = time_str.split(":")
    if len(parts) != 2:
        raise ValidationError(f"Invalid time format '{time_str}': expected HH:MM")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        raise ValidationError(f"Invalid time format '{time_str}': hour and minute must be integers")

    if hour < 0 or hour > 23:
        raise ValidationError(f"Invalid time format '{time_str}': hour must be 0-23")

    if minute < 0 or minute > 59:
        raise ValidationError(f"Invalid time format '{time_str}': minutes must be 0-59")

    return (hour, minute)
```

**Step 3: Run parsing tests**

```bash
pytest tests/unit/test_custom_departure_pattern.py::TestCustomPatternParsing -v
```

Expected: PASS for parsing tests

**Step 4: Commit**

```bash
git add src/validate/validate_arguments.py
git commit -m "feat: add custom departure pattern parsing"
```

---

## Task 4: Implement Pattern Validation

**Files:**
- Modify: `src/validate/validate_arguments.py`

**Step 1: Add validation function**

Add after `_parse_time_hhmm`:

```python
def _validate_custom_pattern(pattern: str, start_hour: float, end_time: int) -> None:
    """
    Validate custom departure pattern against simulation bounds.

    Args:
        pattern: Pattern string like "custom:9:00-9:30,40;10:00-10:45,30"
        start_hour: Simulation start time in hours (e.g., 8.0 for 8:00 AM)
        end_time: Simulation duration in seconds

    Raises:
        ValidationError: If pattern is invalid
    """
    windows = _parse_custom_pattern(pattern)

    if not windows:
        raise ValidationError("Custom pattern must have at least one time window")

    # Calculate simulation bounds
    sim_start_hour = start_hour
    sim_start_min = int((start_hour % 1) * 60)
    sim_end_seconds = end_time
    sim_end_hour = start_hour + (end_time / 3600)

    # Validate each window
    total_percent = 0
    for window in windows:
        start_h, start_m = window["start"]
        end_h, end_m = window["end"]
        percent = window["percent"]

        total_percent += percent

        # Convert to decimal hours for comparison
        window_start_decimal = start_h + start_m / 60
        window_end_decimal = end_h + end_m / 60

        # Check start before end
        if window_start_decimal >= window_end_decimal:
            raise ValidationError(
                f"Invalid window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}: "
                f"start time must be before end time"
            )

        # Check within simulation bounds
        if window_start_decimal < sim_start_hour:
            raise ValidationError(
                f"Window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} is outside "
                f"simulation range {int(sim_start_hour):02d}:{int(sim_start_min):02d}-"
                f"{int(sim_end_hour):02d}:{int((sim_end_hour % 1) * 60):02d}"
            )

        if window_end_decimal > sim_end_hour:
            raise ValidationError(
                f"Window {start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d} is outside "
                f"simulation range {int(sim_start_hour):02d}:{int(sim_start_min):02d}-"
                f"{int(sim_end_hour):02d}:{int((sim_end_hour % 1) * 60):02d}"
            )

    # Check total percentage
    if total_percent > 100:
        raise ValidationError(
            f"Specified percentages sum to {total_percent}%, must be <= 100%"
        )

    # Check for overlaps
    _check_window_overlaps(windows)


def _check_window_overlaps(windows: list) -> None:
    """
    Check that no windows overlap.

    Args:
        windows: List of window dicts with start/end tuples

    Raises:
        ValidationError: If any windows overlap
    """
    # Convert to decimal hours for easier comparison
    ranges = []
    for w in windows:
        start = w["start"][0] + w["start"][1] / 60
        end = w["end"][0] + w["end"][1] / 60
        ranges.append((start, end, w))

    # Sort by start time
    ranges.sort(key=lambda x: x[0])

    # Check adjacent pairs for overlap
    for i in range(len(ranges) - 1):
        _, end1, w1 = ranges[i]
        start2, _, w2 = ranges[i + 1]

        if end1 > start2:
            s1, e1 = w1["start"], w1["end"]
            s2, e2 = w2["start"], w2["end"]
            raise ValidationError(
                f"Windows {s1[0]:02d}:{s1[1]:02d}-{e1[0]:02d}:{e1[1]:02d} and "
                f"{s2[0]:02d}:{s2[1]:02d}-{e2[0]:02d}:{e2[1]:02d} overlap"
            )
```

**Step 2: Update `_validate_departure_pattern` to call new validation**

Replace the existing `_validate_departure_pattern` function:

```python
def _validate_departure_pattern(departure_pattern: str, start_hour: float = None, end_time: int = None) -> None:
    """Validate departure pattern format."""
    from src.constants import DEPARTURE_PATTERN_SIX_PERIODS, DEPARTURE_PATTERN_UNIFORM

    # Check for basic patterns
    if departure_pattern in [DEPARTURE_PATTERN_SIX_PERIODS, DEPARTURE_PATTERN_UNIFORM]:
        return

    # Check for custom pattern
    if departure_pattern.startswith(CUSTOM_PATTERN_PREFIX):
        if start_hour is None or end_time is None:
            # Basic syntax check only (used during initial arg parsing)
            _parse_custom_pattern(departure_pattern)
        else:
            # Full validation with simulation bounds
            _validate_custom_pattern(departure_pattern, start_hour, end_time)
        return

    raise ValidationError(
        f"Invalid departure pattern: {departure_pattern}. "
        f"Valid patterns: 'six_periods', 'uniform', 'custom:HH:MM-HH:MM,percent;...'"
    )
```

**Step 3: Run all validation tests**

```bash
pytest tests/unit/test_custom_departure_pattern.py -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/validate/validate_arguments.py
git commit -m "feat: add custom departure pattern validation"
```

---

## Task 5: Write Departure Time Calculation Tests

**Files:**
- Modify: `tests/unit/test_custom_departure_pattern.py`

**Step 1: Add calculation tests to test file**

```python
# Add to test file after existing tests

from src.traffic.builder import _calculate_custom_deterministic


class TestCustomDepartureTimeCalculation:
    """Tests for custom departure time calculation."""

    def test_single_window_100_percent(self):
        """Single window with 100% gets all vehicles."""
        # start=8:00, duration=5h (18000s), 100 vehicles
        # Window: 9:00-10:00 with 100%
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:9:00-10:00,100",
            start_hour=8.0,
            end_time=18000
        )

        assert len(times) == 100
        # All times should be between 3600s (9:00-8:00=1h) and 7200s (10:00-8:00=2h)
        assert all(3600 <= t <= 7200 for t in times)

    def test_single_window_with_rest(self):
        """Single window with rest distributed to gaps."""
        # 40% in window, 60% in rest
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:9:00-9:30,40",
            start_hour=8.0,
            end_time=18000  # ends at 13:00
        )

        assert len(times) == 100

        # Count vehicles in window (3600s to 5400s = 9:00 to 9:30)
        window_vehicles = sum(1 for t in times if 3600 <= t < 5400)
        assert window_vehicles == 40

    def test_multiple_windows_with_rest(self):
        """Multiple windows with proportional rest distribution."""
        # 40% + 30% = 70% in windows, 30% in rest
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:9:00-9:30,40;10:00-10:45,30",
            start_hour=8.0,
            end_time=18000
        )

        assert len(times) == 100

        # Window 1: 9:00-9:30 (3600-5400s) = 40 vehicles
        w1_count = sum(1 for t in times if 3600 <= t < 5400)
        assert w1_count == 40

        # Window 2: 10:00-10:45 (7200-9900s) = 30 vehicles
        w2_count = sum(1 for t in times if 7200 <= t < 9900)
        assert w2_count == 30

    def test_rest_proportional_to_duration(self):
        """Rest vehicles distributed proportionally by gap duration."""
        # start=8:00, end=13:00 (5 hours)
        # Window: 10:00-11:00 (50%)
        # Rest windows: 8:00-10:00 (2h), 11:00-13:00 (2h) - equal duration
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:10:00-11:00,50",
            start_hour=8.0,
            end_time=18000
        )

        assert len(times) == 100

        # 50% rest = 50 vehicles split equally between two 2-hour gaps
        rest_before = sum(1 for t in times if 0 <= t < 7200)  # 8:00-10:00
        rest_after = sum(1 for t in times if 10800 <= t < 18000)  # 11:00-13:00

        # Should be roughly equal (25 each)
        assert rest_before == 25
        assert rest_after == 25

    def test_uniform_distribution_within_window(self):
        """Vehicles uniformly distributed within each window."""
        times = _calculate_custom_deterministic(
            num_vehicles=10,
            pattern="custom:9:00-10:00,100",
            start_hour=8.0,
            end_time=18000
        )

        # Check uniform spacing (3600s window / 10 vehicles = 360s apart)
        times_sorted = sorted(times)
        intervals = [times_sorted[i+1] - times_sorted[i] for i in range(len(times_sorted)-1)]

        # All intervals should be roughly equal
        avg_interval = sum(intervals) / len(intervals)
        assert all(abs(i - avg_interval) < 1 for i in intervals)  # Within 1 second

    def test_output_sorted_chronologically(self):
        """Output times are sorted."""
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:9:00-9:30,40;10:00-10:45,30",
            start_hour=8.0,
            end_time=18000
        )

        assert times == sorted(times)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_custom_departure_pattern.py::TestCustomDepartureTimeCalculation -v
```

Expected: FAIL with ImportError

**Step 3: Commit**

```bash
git add tests/unit/test_custom_departure_pattern.py
git commit -m "test: add departure time calculation tests for custom pattern"
```

---

## Task 6: Implement Departure Time Calculation

**Files:**
- Modify: `src/traffic/builder.py`

**Step 1: Add imports**

```python
# Add to imports at top of file
from src.constants import (
    # ... existing imports ...
    CUSTOM_PATTERN_PREFIX,
    CUSTOM_WINDOW_SEPARATOR,
    CUSTOM_TIME_PERCENT_SEPARATOR,
    CUSTOM_TIME_RANGE_SEPARATOR,
    DEPARTURE_PATTERN_CUSTOM,
)
from src.validate.validate_arguments import _parse_custom_pattern
```

**Step 2: Add calculation function**

Add after `_calculate_rush_hours_deterministic` function:

```python
def _calculate_custom_deterministic(
    num_vehicles: int,
    pattern: str,
    start_hour: float,
    end_time: int
) -> List[int]:
    """
    Calculate deterministic departure times for custom pattern.

    Args:
        num_vehicles: Total number of vehicles
        pattern: Custom pattern string like "custom:9:00-9:30,40;10:00-10:45,30"
        start_hour: Simulation start time in hours (e.g., 8.0)
        end_time: Simulation duration in seconds

    Returns:
        List of departure times in seconds from simulation start, sorted
    """
    departure_times = []

    # Parse pattern
    windows = _parse_custom_pattern(pattern)

    # Calculate simulation bounds in seconds from midnight
    sim_start_seconds = int(start_hour * 3600)
    sim_end_seconds = sim_start_seconds + end_time

    # Convert windows to seconds and calculate totals
    window_specs = []
    total_specified_percent = 0

    for w in windows:
        start_h, start_m = w["start"]
        end_h, end_m = w["end"]

        window_start = start_h * 3600 + start_m * 60
        window_end = end_h * 3600 + end_m * 60

        window_specs.append({
            "start": window_start,
            "end": window_end,
            "percent": w["percent"],
            "duration": window_end - window_start
        })
        total_specified_percent += w["percent"]

    rest_percent = 100 - total_specified_percent

    # Calculate rest windows (gaps between specified windows and simulation bounds)
    rest_windows = _compute_rest_windows_custom(window_specs, sim_start_seconds, sim_end_seconds)
    rest_total_duration = sum(w["duration"] for w in rest_windows)

    # Allocate vehicles to specified windows
    vehicles_assigned = 0
    for spec in window_specs:
        window_vehicles = int((spec["percent"] / 100) * num_vehicles)
        vehicles_assigned += window_vehicles

        if window_vehicles > 0:
            # Generate uniformly spaced departure times
            interval = spec["duration"] / window_vehicles
            for i in range(window_vehicles):
                # Convert from absolute time to simulation time (offset from start)
                abs_time = spec["start"] + i * interval
                sim_time = abs_time - sim_start_seconds
                departure_times.append(int(sim_time))

    # Allocate vehicles to rest windows proportionally
    rest_vehicles = num_vehicles - vehicles_assigned

    if rest_vehicles > 0 and rest_total_duration > 0:
        for rest_window in rest_windows:
            # Proportional allocation by duration
            window_vehicles = int((rest_window["duration"] / rest_total_duration) * rest_vehicles)

            if window_vehicles > 0:
                interval = rest_window["duration"] / window_vehicles
                for i in range(window_vehicles):
                    abs_time = rest_window["start"] + i * interval
                    sim_time = abs_time - sim_start_seconds
                    departure_times.append(int(sim_time))

    # Handle rounding: add any missing vehicles to largest window
    while len(departure_times) < num_vehicles:
        # Add to middle of simulation
        departure_times.append(end_time // 2)

    return sorted(departure_times)


def _compute_rest_windows_custom(
    specified_windows: List[dict],
    sim_start: int,
    sim_end: int
) -> List[dict]:
    """
    Compute rest windows (gaps) between specified windows.

    Args:
        specified_windows: List of window specs with start/end in seconds
        sim_start: Simulation start in seconds from midnight
        sim_end: Simulation end in seconds from midnight

    Returns:
        List of rest window specs
    """
    if not specified_windows:
        return [{"start": sim_start, "end": sim_end, "duration": sim_end - sim_start}]

    # Sort windows by start time
    sorted_windows = sorted(specified_windows, key=lambda w: w["start"])

    rest_windows = []

    # Gap before first window
    if sorted_windows[0]["start"] > sim_start:
        rest_windows.append({
            "start": sim_start,
            "end": sorted_windows[0]["start"],
            "duration": sorted_windows[0]["start"] - sim_start
        })

    # Gaps between windows
    for i in range(len(sorted_windows) - 1):
        gap_start = sorted_windows[i]["end"]
        gap_end = sorted_windows[i + 1]["start"]
        if gap_end > gap_start:
            rest_windows.append({
                "start": gap_start,
                "end": gap_end,
                "duration": gap_end - gap_start
            })

    # Gap after last window
    if sorted_windows[-1]["end"] < sim_end:
        rest_windows.append({
            "start": sorted_windows[-1]["end"],
            "end": sim_end,
            "duration": sim_end - sorted_windows[-1]["end"]
        })

    return rest_windows
```

**Step 3: Update `calculate_temporal_departure_times` to use custom pattern**

Find the function (around line 957) and update the pattern matching:

```python
def calculate_temporal_departure_times(num_vehicles: int, departure_pattern: str, start_time: float, end_time: int) -> List[int]:
    """
    Calculate deterministic departure times for all vehicles based on departure pattern.
    ...existing docstring...
    """
    departure_times = []

    if departure_pattern == DEPARTURE_PATTERN_UNIFORM:
        # ... existing uniform code ...
        if num_vehicles <= 0:
            return departure_times

        interval = end_time / num_vehicles
        for i in range(num_vehicles):
            departure_time = i * interval
            departure_times.append(int(departure_time))

    elif departure_pattern == DEPARTURE_PATTERN_SIX_PERIODS:
        departure_times = _calculate_six_periods_deterministic(
            num_vehicles, end_time)

    elif departure_pattern.startswith(CUSTOM_PATTERN_PREFIX):
        # New custom pattern handling
        departure_times = _calculate_custom_deterministic(
            num_vehicles, departure_pattern, start_time, end_time)

    else:
        # Default to six_periods for unknown patterns
        departure_times = _calculate_six_periods_deterministic(
            num_vehicles, end_time)

    return sorted(departure_times)
```

**Step 4: Run calculation tests**

```bash
pytest tests/unit/test_custom_departure_pattern.py::TestCustomDepartureTimeCalculation -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/traffic/builder.py
git commit -m "feat: implement custom departure time calculation"
```

---

## Task 7: Remove Old Rush Hours Code

**Files:**
- Modify: `src/traffic/builder.py`
- Modify: `src/validate/validate_arguments.py`
- Modify: `src/constants.py`

**Step 1: Remove `_calculate_rush_hours_deterministic` from builder.py**

Delete the function `_calculate_rush_hours_deterministic` and `_compute_rest_windows` (the old one).

**Step 2: Remove `_validate_rush_hours_pattern` from validate_arguments.py**

Delete the `_validate_rush_hours_pattern` function.

**Step 3: Remove old constants from constants.py**

Remove:
```python
DEPARTURE_PATTERN_RUSH_HOURS = "rush_hours"
RUSH_HOURS_PREFIX = "rush_hours:"
RUSH_HOURS_REST = "rest"
RUSH_HOURS_SEPARATOR = ":"
```

**Step 4: Run all tests**

```bash
pytest tests/ -v -k "departure"
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/traffic/builder.py src/validate/validate_arguments.py src/constants.py
git commit -m "refactor: remove deprecated rush_hours pattern code"
```

---

## Task 8: Update Argument Parser Help Text

**Files:**
- Modify: `src/args/parser.py:114-119`

**Step 1: Update help text**

Find and update the `--departure_pattern` argument:

```python
parser.add_argument(
    "--departure_pattern",
    type=str,
    default=DEFAULT_DEPARTURE_PATTERN,
    help=(
        f"Vehicle departure pattern: '{DEFAULT_DEPARTURE_PATTERN}' (default, even distribution), "
        f"'six_periods' (research-based), or 'custom:HH:MM-HH:MM,percent;...' "
        f"(e.g., 'custom:9:00-9:30,40;10:00-10:45,30' for 40%% at 9:00-9:30, 30%% at 10:00-10:45, "
        f"remaining 30%% distributed to other times)"
    )
)
```

**Step 2: Commit**

```bash
git add src/args/parser.py
git commit -m "docs: update departure_pattern help text for custom pattern"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/specification/command-line-interface.md:112-121`

**Step 1: Update documentation**

Find and replace the `--departure_pattern` section:

```markdown
#### `--departure_pattern` (str, default: "uniform")

Vehicle departure timing pattern.

- **Patterns**:
  - `uniform`: Even distribution across simulation time (default)
  - `six_periods`: Research-based daily structure (Morning 20%, Morning Rush 30%, Noon 25%, Evening Rush 20%, Evening 4%, Night 1%)
  - `custom:HH:MM-HH:MM,percent;...`: Precise time window control with percentages
- **Custom Pattern Format**:
  - Syntax: `custom:start-end,percent;start-end,percent;...`
  - Times are absolute clock times (HH:MM format)
  - Windows must fall within simulation range [start_time, start_time + duration]
  - Percentages need not sum to 100%; remainder distributed uniformly to gaps
  - Example: `custom:9:00-9:30,40;10:00-10:45,30` = 40% at 9-9:30, 30% at 10-10:45, 30% elsewhere
- **Validation**:
  - Percentages must sum to <= 100%
  - Windows cannot overlap
  - Window start must be before end
- **Example**: `--departure_pattern "custom:8:00-9:00,50;17:00-18:00,30"`
```

**Step 2: Commit**

```bash
git add docs/specification/command-line-interface.md
git commit -m "docs: update command-line-interface.md for custom departure pattern"
```

---

## Task 10: Integration Test

**Files:**
- Modify: `tests/integration/test_cli_arguments.py`

**Step 1: Add integration test**

Find the `test_departure_pattern_values` test and add custom pattern:

```python
@pytest.mark.integration
@pytest.mark.parametrize("pattern", [
    "uniform",
    "six_periods",
    "custom:9:00-10:00,50;11:00-12:00,30",
])
def test_departure_pattern_values(self, temp_workspace, pattern):
    """Test different departure patterns."""
    result = run_file_generation([
        "--grid_dimension", "3",
        "--departure_pattern", pattern,
        "--num_vehicles", "20",
        "--seed", "42",
        "--start_time_hour", "8.0",
        "--end-time", "18000",
        "--workspace", str(temp_workspace),
    ])
    assert result.returncode == 0, f"Failed with departure_pattern={pattern}: {result.stderr}"
    validate_output_files(get_workspace_dir(temp_workspace))
```

**Step 2: Run integration tests**

```bash
pytest tests/integration/test_cli_arguments.py::TestDeparturePattern -v
```

Expected: All PASS

**Step 3: Commit**

```bash
git add tests/integration/test_cli_arguments.py
git commit -m "test: add integration test for custom departure pattern"
```

---

## Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the departure patterns section**

Find the "Vehicle Departure Patterns" section and update:

```markdown
- **Vehicle Departure Patterns**:
  - Replaced sequential departure (0, 1, 2, 3...) with realistic temporal distribution based on research papers
  - Default: uniform distribution across simulation time
  - Alternative: six_periods system with research-based 6-period daily structure
  - Custom: `custom:HH:MM-HH:MM,percent;...` for precise time window control
    - Example: `custom:9:00-9:30,40;10:00-10:45,30` = 40% at 9-9:30, 30% at 10-10:45, 30% elsewhere
    - Times are absolute clock times within simulation range
    - Unspecified percentage distributed proportionally to time gaps
  - Automatically works with simulation start_time and end_time
  - Compatible with all routing strategies and vehicle types
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with custom departure pattern info"
```

---

## Task 12: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All PASS

**Step 2: Manual test**

```bash
source .venv/bin/activate
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --start_time_hour 8.0 \
  --end-time 18000 \
  --departure_pattern "custom:9:00-9:30,40;10:00-10:45,30" \
  --seed 42
```

Expected: Simulation runs successfully

**Step 3: Verify vehicle departure times in output**

```bash
grep 'depart=' workspace/vehicles.rou.xml | head -20
```

Check that departure times cluster around the specified windows.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete custom departure pattern implementation"
```

---

## Summary

This implementation:
1. Replaces `rush_hours` with `custom` pattern
2. Uses absolute clock times (HH:MM format)
3. Distributes unspecified percentage proportionally to time gaps
4. Validates against simulation bounds, overlaps, and percentage limits
5. Maintains backward compatibility with `uniform` and `six_periods` patterns
