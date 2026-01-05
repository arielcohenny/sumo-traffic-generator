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
