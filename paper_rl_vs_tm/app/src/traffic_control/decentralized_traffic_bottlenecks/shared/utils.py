
def is_calculation_time(step, tree_method_interval):
    """
    Check if it's time to run Tree Method calculations.

    Args:
        step: Current simulation step
        tree_method_interval: Tree Method calculation interval in seconds

    Returns:
        bool: True if Tree Method calculations should run at this step
    """
    return step > 0 and step % tree_method_interval == 0


def calc_iteration_from_step(step, tree_method_interval):
    """
    Calculate Tree Method iteration number from simulation step.

    Args:
        step: Current simulation step
        tree_method_interval: Tree Method calculation interval in seconds

    Returns:
        int: Tree Method iteration number
    """
    return (step + 1) // tree_method_interval


def validate_vehicle_id(vehicle_id):
    """
    Validate vehicle ID format and fail fast on unexpected formats.

    Args:
        vehicle_id: Vehicle ID string from SUMO

    Returns:
        str: The validated vehicle ID

    Raises:
        SystemExit: If vehicle ID format is invalid
    """
    if not isinstance(vehicle_id, str):
        print(
            f"ERROR: Vehicle ID must be string, got {type(vehicle_id)}: {vehicle_id}")
        exit(1)

    if not vehicle_id.startswith('veh'):
        print(f"ERROR: Vehicle ID must start with 'veh', got: {vehicle_id}")
        exit(1)

    try:
        # Verify the suffix is a valid integer
        int(vehicle_id[3:])
        return vehicle_id
    except ValueError:
        print(f"ERROR: Vehicle ID suffix must be integer, got: {vehicle_id}")
        exit(1)
