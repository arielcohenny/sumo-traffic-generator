"""
Base class for pipeline steps.

This module provides the abstract base class for all pipeline steps,
ensuring consistent interface and error handling.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from src.validate.errors import ValidationError


class BaseStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self, args: Any):
        """Initialize step with command line arguments.

        Args:
            args: Parsed command line arguments from argparse
        """
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self) -> None:
        """Execute the pipeline step.

        This method must be implemented by concrete step classes.
        Raises:
            ValidationError: If step execution fails validation
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the step execution results.

        This method must be implemented by concrete step classes.
        Raises:
            ValidationError: If validation fails
        """
        pass

    def run(self) -> None:
        """Run the step with error handling and validation.

        This is the main entry point that handles the complete step execution.
        """
        try:
            # self.logger.info(f"Starting {self.__class__.__name__}")
            self.execute()
            self.validate()
            # self.logger.info(f"Completed {self.__class__.__name__} successfully")
        except ValidationError as ve:
            self.logger.error(
                f"{self.__class__.__name__} validation failed: {ve}")
            raise
        except Exception as e:
            self.logger.error(
                f"{self.__class__.__name__} execution failed: {e}")
            raise
