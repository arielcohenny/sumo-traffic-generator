class Vehicle:
    def __init__(self, vehicle_id, step):
        self.vehicle_id: str = vehicle_id
        self.start_step: int = step
        self.end_step = None
        self.time = 0

    def end_drive(self, step: int):
        self.end_step = step
        self.time = self.end_step - self.start_step
        return self.time
