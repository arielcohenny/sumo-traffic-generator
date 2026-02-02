"""Shared utilities for RL traffic signal control.

Contains functions used by both environment.py and controller.py
to avoid code duplication.
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax.

    Args:
        x: Array of raw values

    Returns:
        Array of probabilities summing to 1.0
    """
    if len(x) == 0:
        raise ValueError("Cannot apply softmax to empty array.")
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def proportions_to_durations(proportions, cycle_length, min_phase_time):
    """Convert proportions to integer durations with constraints.

    Args:
        proportions: Array of proportions summing to 1.0 (e.g., [0.25, 0.15, 0.40, 0.20])
        cycle_length: Total cycle time in seconds (e.g., 90)
        min_phase_time: Minimum duration per phase in seconds (e.g., 10)

    Returns:
        List of integer durations summing exactly to cycle_length
    """
    num_phases = len(proportions)
    available_time = cycle_length - (num_phases * min_phase_time)

    # Calculate durations
    durations = [min_phase_time + (p * available_time) for p in proportions]

    # Round to integers
    durations = [int(round(d)) for d in durations]

    # Ensure exact sum (adjust largest phase if needed)
    diff = cycle_length - sum(durations)
    if diff != 0:
        max_idx = np.argmax(durations)
        durations[max_idx] += diff

    return durations
