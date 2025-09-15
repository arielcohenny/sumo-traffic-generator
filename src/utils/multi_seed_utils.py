"""
Multi-seed utility functions for the SUMO traffic generator.

Provides separate seed management for network generation, private traffic, and public traffic
to enable fine-grained experimental control while maintaining backward compatibility.
"""

import random
from typing import Any


def get_network_seed(args: Any) -> int:
    """Get the network seed for network structure generation.

    Controls: junction removal, lane assignment, land use generation, edge attractiveness.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        int: The random seed to use for network generation
    """
    if hasattr(args, '_network_seed'):
        return args._network_seed

    # Seed resolution logic
    if args.seed is not None:
        # Backward compatibility: --seed sets all seeds
        seed = args.seed
    elif hasattr(args, 'network_seed') and args.network_seed is not None:
        # Use explicit --network-seed
        seed = args.network_seed
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    args._network_seed = seed
    return seed


def get_private_traffic_seed(args: Any) -> int:
    """Get the private traffic seed for passenger vehicle generation.

    Controls: private vehicle type assignment, route generation, departure times.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        int: The random seed to use for private traffic generation
    """
    if hasattr(args, '_private_traffic_seed'):
        return args._private_traffic_seed

    # Seed resolution logic
    if args.seed is not None:
        # Backward compatibility: --seed sets all seeds
        seed = args.seed
    elif hasattr(args, 'private_traffic_seed') and args.private_traffic_seed is not None:
        # Use explicit --private-traffic-seed
        seed = args.private_traffic_seed
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    args._private_traffic_seed = seed
    return seed


def get_public_traffic_seed(args: Any) -> int:
    """Get the public traffic seed for public vehicle generation.

    Controls: public vehicle type assignment, route generation, departure times.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        int: The random seed to use for public traffic generation
    """
    if hasattr(args, '_public_traffic_seed'):
        return args._public_traffic_seed

    # Seed resolution logic
    if args.seed is not None:
        # Backward compatibility: --seed sets all seeds
        seed = args.seed
    elif hasattr(args, 'public_traffic_seed') and args.public_traffic_seed is not None:
        # Use explicit --public-traffic-seed
        seed = args.public_traffic_seed
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    args._public_traffic_seed = seed
    return seed


def get_cached_seed(args: Any) -> int:
    """Get the cached random seed for backward compatibility.

    This function maintains compatibility with existing code that uses a single seed.
    It returns the network seed by default, but this should be phased out in favor
    of the specific seed functions above.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        int: The random seed to use (network seed for compatibility)
    """
    # For backward compatibility, return the network seed
    return get_network_seed(args)
