
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


def get_vehicle_inx(vehicle_id):
    """Extract vehicle index from vehicle ID format 'veh{index}'"""
    if isinstance(vehicle_id, str) and vehicle_id.startswith('veh'):
        try:
            return int(vehicle_id[3:])
        except ValueError:
            # Handle cases where the suffix after 'veh' is not a number
            return 0
    else:
        # Handle unexpected vehicle ID format
        return 0
