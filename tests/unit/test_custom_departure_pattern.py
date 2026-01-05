"""Tests for custom departure pattern validation and calculation."""

import pytest
from src.validate.validate_arguments import (
    _validate_departure_pattern,
    _validate_custom_pattern,
    _parse_custom_pattern,
)
from src.validate.errors import ValidationError


class TestCustomPatternValidation:
    """Tests for custom departure pattern validation."""

    @pytest.mark.unit
    def test_valid_single_window(self):
        """Single window with valid format."""
        _validate_custom_pattern("custom:9:00-9:30,40", start_hour=8.0, end_time=18000)

    @pytest.mark.unit
    def test_valid_multiple_windows(self):
        """Multiple windows with valid format."""
        _validate_custom_pattern(
            "custom:9:00-9:30,40;10:00-10:45,30",
            start_hour=8.0,
            end_time=18000
        )

    @pytest.mark.unit
    def test_valid_trailing_semicolon(self):
        """Trailing semicolon should be allowed."""
        _validate_custom_pattern(
            "custom:9:00-9:30,40;10:00-10:45,30;",
            start_hour=8.0,
            end_time=18000
        )

    @pytest.mark.unit
    def test_error_percentage_exceeds_100(self):
        """Error when percentages sum > 100."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:00-9:30,60;10:00-10:45,50",
                start_hour=8.0,
                end_time=18000
            )
        assert "sum to 110%" in str(exc_info.value)

    @pytest.mark.unit
    def test_error_window_before_start(self):
        """Error when window starts before simulation."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:7:00-8:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "outside simulation range" in str(exc_info.value)

    @pytest.mark.unit
    def test_error_window_after_end(self):
        """Error when window extends past simulation end."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:12:00-14:00,40",
                start_hour=8.0,
                end_time=18000  # 5 hours = ends at 13:00
            )
        assert "outside simulation range" in str(exc_info.value)

    @pytest.mark.unit
    def test_error_overlapping_windows(self):
        """Error when windows overlap."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:00-10:00,40;9:30-10:30,30",
                start_hour=8.0,
                end_time=18000
            )
        assert "overlap" in str(exc_info.value)

    @pytest.mark.unit
    def test_error_start_after_end(self):
        """Error when window start is after end."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:30-9:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "start time must be before end time" in str(exc_info.value)

    @pytest.mark.unit
    def test_error_invalid_time_format(self):
        """Error on invalid time format."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_custom_pattern(
                "custom:9:75-10:00,40",
                start_hour=8.0,
                end_time=18000
            )
        assert "minutes must be 0-59" in str(exc_info.value)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_parse_single_window(self):
        """Parse single window correctly."""
        windows = _parse_custom_pattern("custom:9:00-9:30,40")
        assert len(windows) == 1
        assert windows[0] == {"start": (9, 0), "end": (9, 30), "percent": 40}

    @pytest.mark.unit
    def test_parse_multiple_windows(self):
        """Parse multiple windows correctly."""
        windows = _parse_custom_pattern("custom:9:00-9:30,40;10:00-10:45,30")
        assert len(windows) == 2
        assert windows[0] == {"start": (9, 0), "end": (9, 30), "percent": 40}
        assert windows[1] == {"start": (10, 0), "end": (10, 45), "percent": 30}

    @pytest.mark.unit
    def test_parse_ignores_trailing_semicolon(self):
        """Trailing semicolon produces same result."""
        windows1 = _parse_custom_pattern("custom:9:00-9:30,40")
        windows2 = _parse_custom_pattern("custom:9:00-9:30,40;")
        assert windows1 == windows2


from src.traffic.builder import _calculate_custom_deterministic


class TestCustomDepartureTimeCalculation:
    """Tests for custom departure time calculation."""

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_output_sorted_chronologically(self):
        """Output times are sorted."""
        times = _calculate_custom_deterministic(
            num_vehicles=100,
            pattern="custom:9:00-9:30,40;10:00-10:45,30",
            start_hour=8.0,
            end_time=18000
        )

        assert times == sorted(times)
