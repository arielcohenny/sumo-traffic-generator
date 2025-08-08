"""
Seed utility functions for the SUMO traffic generator.

Provides centralized seed management to ensure reproducible simulations
across all pipeline steps.
"""

import random
from typing import Any


def get_cached_seed(args: Any) -> int:
    """Get the cached random seed, generating one if not provided.
    
    This function ensures that all pipeline steps use the same seed for
    reproducible simulations. The seed is cached on the args object to
    avoid generating different seeds across steps.
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        int: The random seed to use for this simulation
    """
    if hasattr(args, '_seed'):
        return args._seed
    
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args._seed = seed
    return seed