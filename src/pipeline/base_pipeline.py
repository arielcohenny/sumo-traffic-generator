"""
Base pipeline class for SUMO traffic generation and simulation.

This module provides the abstract base class that defines the pipeline interface
and common functionality for all pipeline implementations.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from src.config import CONFIG


class BasePipeline(ABC):
    """Abstract base class for traffic generation and simulation pipelines."""
    
    def __init__(self, args: Any):
        """Initialize pipeline with command line arguments.
        
        Args:
            args: Parsed command line arguments from argparse
        """
        self.args = args
        
        # Update workspace configuration before any file operations
        if hasattr(args, 'workspace') and args.workspace:
            CONFIG.update_workspace(args.workspace)
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @abstractmethod
    def execute(self) -> None:
        """Execute the complete pipeline.
        
        This method must be implemented by concrete pipeline classes.
        """
        pass
    
    def _log_step(self, step_number: int, step_name: str) -> None:
        """Log the start of a pipeline step.
        
        Args:
            step_number: The step number (1-9)
            step_name: Human-readable name of the step
        """
        self.logger.info(f"--- Step {step_number}: {step_name} ---")
    
    def _validate_output_directory(self) -> None:
        """Ensure output directory exists and is clean."""
        if CONFIG.output_dir.exists():
            import shutil
            shutil.rmtree(CONFIG.output_dir)
        CONFIG.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Prepared output directory: {CONFIG.output_dir}")