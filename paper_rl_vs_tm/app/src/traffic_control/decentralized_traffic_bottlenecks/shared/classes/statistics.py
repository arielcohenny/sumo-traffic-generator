
class IterationStats:
    def __init__(self, iterations_count, output_path):
        self.iteration_entered_count = []
        self.iteration_arrived_count = []
        self.vehicles_inside_dict = {}
        self.prev_sec_list = []
        self.total_driving_time = 0
        self.total_duration_plus_before_insert = 0
        self.iterations_count = iterations_count + 1
        self.node_phases = []
        self.output_path = output_path


class VehicleStats:
    def __init__(self, vehicle_id, sec_since_start, iteration):
        self.vehicle_id = vehicle_id
        self.sec_since_start: int = sec_since_start
        self.iteration = iteration
        self.time_inside = 0

    def calc_time_inside(self, current_sec_since_start):
        return current_sec_since_start - self.sec_since_start
