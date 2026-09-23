# src/validate/errors.py
"""Common exception types for inline validators."""


class ValidationError(RuntimeError):
    """Raised when an inline runtime validator finds a violation."""
