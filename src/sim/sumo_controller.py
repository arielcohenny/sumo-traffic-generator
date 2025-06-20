import traci


class SumoController:
    # def __init__(self, sumo_cfg: str, step_length: float, end_time: float):
    #     self.sumo_cfg = sumo_cfg
    #     self.step_length = step_length
    #     self.end_time = end_time

    def __init__(self,
                 sumo_cfg: str,
                 step_length: float,
                 end_time: float,
                 gui: bool = False):
        """
        :param gui: if True, launch sumo-gui; otherwise, batch sumo.
        """
        self.sumo_cfg = sumo_cfg
        self.step_length = step_length
        self.end_time = end_time
        self.gui = gui

    # def start(self):
    #     """
    #     Start SUMO with TraCI.
    #     """
    #     sumo_cmd = ["sumo", "-c", self.sumo_cfg,
    #                 "--step-length", str(self.step_length)]
    #     traci.start(sumo_cmd)

    def start(self):
        """
        Start SUMO (GUI or batch) with TraCI.
        """
        print("Starting SUMO with TraCI...")
        binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            # "--aggregate-warnings=1", # aggregates warnings of the same type whenever more than 1 occur
            # "--no-warnings",  # carry over “no warnings” preference
            # "--start"  # Start stepping immediately
        ]
        traci.start(sumo_cmd)
        print("SUMO TraCI started")

    def step(self):
        """
        Advance the simulation by one step.
        """
        traci.simulationStep()

    def close(self):
        """
        Close the TraCI connection.
        """
        traci.close()

    def run(self, control_callback):
        """
        Run the simulation loop.

        :param control_callback: A function that takes the current simulation time (int)
                                 and applies control actions.
        """
        print("Running SUMO simulation...")
        self.start()
        current_time = 0
        while current_time < self.end_time:
            self.step()
            control_callback(current_time)
            current_time += self.step_length
        self.close()
