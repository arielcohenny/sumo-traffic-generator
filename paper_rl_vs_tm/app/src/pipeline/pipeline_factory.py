"""
Pipeline factory for creating appropriate pipeline instances.

This module provides a factory for creating the correct pipeline type
based on command line arguments.
"""

from typing import Any
from .base_pipeline import BasePipeline
from .standard_pipeline import StandardPipeline
from .sample_pipeline import SamplePipeline


class PipelineFactory:
    """Factory for creating pipeline instances."""
    
    @staticmethod
    def create_pipeline(args: Any) -> BasePipeline:
        """Create appropriate pipeline based on arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            Pipeline instance (StandardPipeline or SamplePipeline)
        """
        if args.tree_method_sample:
            return SamplePipeline(args)
        else:
            return StandardPipeline(args)