import traci
import xml.etree.ElementTree as ET
from src.config import CONFIG


class SumoController:
    # def __init__(self, sumo_cfg: str, step_length: float, end_time: float):
    #     self.sumo_cfg = sumo_cfg
    #     self.step_length = step_length
    #     self.end_time = end_time

    def __init__(self,
                 sumo_cfg: str,
                 step_length: float,
                 end_time: float,
                 gui: bool = False,
                 time_dependent: bool = False,
                 start_time_hour: float = 0.0):
        """
        :param gui: if True, launch sumo-gui; otherwise, batch sumo.
        :param time_dependent: if True, enables 4-phase attractiveness switching
        :param start_time_hour: real-world hour when simulation starts (0-24)
        """
        self.sumo_cfg = sumo_cfg
        self.step_length = step_length
        self.end_time = end_time
        self.gui = gui
        self.time_dependent = time_dependent
        self.start_time_hour = start_time_hour
        self.current_phase = None
        self.phase_profiles = {}

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

    def get_current_phase(self, current_hour: float) -> str:
        """Get current traffic phase based on hour of day (0-24)"""
        if 6.0 <= current_hour < 9.5:
            return "morning_peak"
        elif 9.5 <= current_hour < 16.0:  
            return "midday_offpeak"
        elif 16.0 <= current_hour < 19.0:
            return "evening_peak"
        else:
            return "night_low"

    def load_phase_profiles(self):
        """Load all 4 phase profiles from the network file"""
        if not self.time_dependent:
            return
        
        net_file = CONFIG.network_file
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        phases = ["morning_peak", "midday_offpeak", "evening_peak", "night_low"]
        
        for phase in phases:
            self.phase_profiles[phase] = {}
            for edge in root.findall("edge"):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(":"):
                    depart_attr = edge.get(f"{phase}_depart_attractiveness")
                    arrive_attr = edge.get(f"{phase}_arrive_attractiveness")
                    if depart_attr and arrive_attr:
                        self.phase_profiles[phase][edge_id] = {
                            'depart': int(depart_attr),
                            'arrive': int(arrive_attr)
                        }

    def update_edge_attractiveness(self, new_phase: str):
        """Update edge attractiveness values via TraCI for the new phase"""
        if not self.time_dependent or new_phase not in self.phase_profiles:
            return
        
        print(f"Switching to phase: {new_phase}")
        
        # Update attractiveness for all edges
        for edge_id, attractiveness in self.phase_profiles[new_phase].items():
            try:
                # Note: SUMO doesn't have direct TraCI commands to change edge attractiveness
                # This would require modifying the route generation logic or using additional files
                # For now, we'll track the phase change for logging/debugging
                pass
            except:
                # Edge might not exist in current simulation
                continue
        
        self.current_phase = new_phase

    def check_phase_transition(self, current_time_seconds: float):
        """Check if we need to transition to a new phase"""
        if not self.time_dependent:
            return
        
        # Convert simulation time to real-world hours
        hours_elapsed = current_time_seconds / 3600.0  # Convert seconds to hours
        current_hour = (self.start_time_hour + hours_elapsed) % 24.0
        
        new_phase = self.get_current_phase(current_hour)
        
        if self.current_phase != new_phase:
            self.update_edge_attractiveness(new_phase)

    def run(self, control_callback):
        """
        Run the simulation loop with optional phase switching.

        :param control_callback: A function that takes the current simulation time (int)
                                 and applies control actions.
        """
        print("Running SUMO simulation...")
        self.start()
        
        # Load phase profiles if time-dependent
        if self.time_dependent:
            self.load_phase_profiles()
            # Set initial phase
            initial_phase = self.get_current_phase(self.start_time_hour)
            self.current_phase = initial_phase
            print(f"Starting simulation at {self.start_time_hour:.1f}h in phase: {initial_phase}")
        
        current_time = 0
        while current_time < self.end_time:
            self.step()
            
            # Check for phase transitions
            if self.time_dependent:
                self.check_phase_transition(current_time)
            
            # Apply control callback (Nimrod's algorithm, etc.)
            control_callback(current_time)
            current_time += self.step_length
            
        self.close()
