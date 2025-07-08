
def is_calculation_time(step, seconds_in_cycle):
    return step > 0 and step % seconds_in_cycle == 0


def calc_iteration_from_step(step, seconds_in_cycle):
    return (step + 1) // seconds_in_cycle


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
