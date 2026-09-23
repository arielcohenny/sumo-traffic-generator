"""
Multi-seed utility functions for the SUMO traffic generator.

Provides separate seed management for network generation, private traffic, and public traffic
to enable fine-grained experimental control while maintaining backward compatibility.
"""

import random
from typing import Any


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute value from either dict or object.

    Args:
        obj: Object to get attribute from (dict or namespace)
        key: Attribute name
        default: Default value if not found

    Returns:
        The attribute value or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return getattr(obj, key, default)


def _set_attr(obj: Any, key: str, value: Any) -> None:
    """Set attribute value on either dict or object.

    Args:
        obj: Object to set attribute on (dict or namespace)
        key: Attribute name
        value: Value to set
    """
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def get_network_seed(args: Any) -> int:
    """Get the network seed for network structure generation.

    Controls: junction removal, lane assignment, land use generation, edge attractiveness.

    Args:
        args: Parsed command line arguments from argparse or dict

    Returns:
        int: The random seed to use for network generation
    """
    if _get_attr(args, '_network_seed') is not None:
        return _get_attr(args, '_network_seed')

    # Seed resolution logic
    seed_val = _get_attr(args, 'seed')
    network_seed_val = _get_attr(args, 'network_seed')

    if seed_val is not None:
        # Backward compatibility: --seed sets all seeds
        seed = seed_val
    elif network_seed_val is not None:
        # Use explicit --network-seed
        seed = network_seed_val
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    _set_attr(args, '_network_seed', seed)
    return seed


def get_private_traffic_seed(args: Any) -> int:
    """Get the private traffic seed for passenger vehicle generation.

    Controls: private vehicle type assignment, route generation, departure times.

    Args:
        args: Parsed command line arguments from argparse or dict

    Returns:
        int: The random seed to use for private traffic generation
    """
    if _get_attr(args, '_private_traffic_seed') is not None:
        return _get_attr(args, '_private_traffic_seed')

    # Seed resolution logic
    seed_val = _get_attr(args, 'seed')
    private_traffic_seed_val = _get_attr(args, 'private_traffic_seed')

    if seed_val is not None:
        # Backward compatibility: --seed sets all seeds
        seed = seed_val
    elif private_traffic_seed_val is not None:
        # Use explicit --private-traffic-seed
        seed = private_traffic_seed_val
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    _set_attr(args, '_private_traffic_seed', seed)
    return seed


def get_public_traffic_seed(args: Any) -> int:
    """Get the public traffic seed for public vehicle generation.

    Controls: public vehicle type assignment, route generation, departure times.

    Args:
        args: Parsed command line arguments from argparse or dict

    Returns:
        int: The random seed to use for public traffic generation
    """
    if _get_attr(args, '_public_traffic_seed') is not None:
        return _get_attr(args, '_public_traffic_seed')

    # Seed resolution logic
    seed_val = _get_attr(args, 'seed')
    public_traffic_seed_val = _get_attr(args, 'public_traffic_seed')

    if seed_val is not None:
        # Backward compatibility: --seed sets all seeds
        seed = seed_val
    elif public_traffic_seed_val is not None:
        # Use explicit --public-traffic-seed
        seed = public_traffic_seed_val
    else:
        # Generate random seed
        seed = random.randint(0, 2**32 - 1)

    _set_attr(args, '_public_traffic_seed', seed)
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
