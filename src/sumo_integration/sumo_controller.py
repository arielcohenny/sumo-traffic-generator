import traci
from src.config import CONFIG
import xml.etree.ElementTree as ET


# Constants for traffic control
DEFAULT_REALTIME_REROUTING_INTERVAL = 30  # seconds
DEFAULT_FASTEST_REROUTING_INTERVAL = 45   # seconds

# Traffic phase time boundaries (hours)
MORNING_PEAK_START = 6.0
MORNING_PEAK_END = 9.5
MIDDAY_OFFPEAK_START = 9.5
MIDDAY_OFFPEAK_END = 16.0
EVENING_PEAK_START = 16.0
EVENING_PEAK_END = 19.0

# Conversion constants
SECONDS_PER_HOUR = 3600


class SumoController:
    def __init__(self,
                 sumo_cfg: str,
                 step_length: float,
                 end_time: float,
                 gui: bool = False,
                 time_dependent: bool = False,
                 start_time_hour: float = 0.0,
                 routing_strategy: str = "shortest 100"):
        """
        :param gui: if True, launch sumo-gui; otherwise, batch sumo.
        :param time_dependent: if True, enables 4-phase attractiveness switching
        :param start_time_hour: real-world hour when simulation starts (0-24)
        :param routing_strategy: routing strategy specification for dynamic rerouting
        """
        self.sumo_cfg = sumo_cfg
        self.step_length = step_length
        self.end_time = end_time
        self.gui = gui
        self.time_dependent = time_dependent
        self.start_time_hour = start_time_hour
        self.current_phase = None
        self.phase_profiles = {}

        # Dynamic routing support
        self.routing_strategy = routing_strategy
        self.vehicle_strategies = {}  # vehicle_id -> strategy_name
        self.vehicle_rerouting_times = {}  # vehicle_id -> next_rerouting_time
        self.strategy_intervals = {
            'realtime': DEFAULT_REALTIME_REROUTING_INTERVAL,
            'fastest': DEFAULT_FASTEST_REROUTING_INTERVAL
        }
        
        # Metrics tracking
        self.total_arrived = 0
        self.total_departed = 0
        self.vehicle_travel_times = {}  # Track when each vehicle started its journey
        self.vehicle_departures = {}    # Track departure times
        self.vehicle_arrivals = {}      # Track arrival times

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
            "--statistic-output", "workspace/sumo_statistics.xml",
            "--tripinfo-output.write-unfinished", "true",
            "--error-log", "workspace/sumo_errors.log",
        ]
        traci.start(sumo_cmd)
        print("SUMO TraCI started")

    def step(self):
        """
        Advance the simulation by one step.
        """
        traci.simulationStep()

    def track_vehicle_metrics(self, current_time):
        """Track vehicle departures and arrivals during simulation."""
        try:
            # Get newly departed vehicles
            departed_vehicles = traci.simulation.getDepartedIDList()
            for veh_id in departed_vehicles:
                if veh_id not in self.vehicle_departures:
                    self.vehicle_departures[veh_id] = current_time
                    self.total_departed += 1
            
            # Get newly arrived vehicles
            arrived_vehicles = traci.simulation.getArrivedIDList()
            for veh_id in arrived_vehicles:
                if veh_id not in self.vehicle_arrivals:
                    self.vehicle_arrivals[veh_id] = current_time
                    self.total_arrived += 1
                    
                    # Calculate travel time if we have departure time
                    if veh_id in self.vehicle_departures:
                        travel_time = current_time - self.vehicle_departures[veh_id]
                        self.vehicle_travel_times[veh_id] = travel_time
                        
        except Exception as e:
            # Don't let metrics tracking break the simulation
            pass

    def collect_final_metrics(self):
        """Collect final simulation metrics using tracked data."""
        try:
            simulation_time = traci.simulation.getTime()
            
            # Use our tracked metrics
            arrived_vehicles = self.total_arrived
            departed_vehicles = self.total_departed
            
            # Calculate average travel time from tracked data
            mean_travel_time = 0.0
            if self.vehicle_travel_times:
                total_travel_time = sum(self.vehicle_travel_times.values())
                mean_travel_time = total_travel_time / len(self.vehicle_travel_times)
            
            metrics = {
                'arrived_vehicles': arrived_vehicles,
                'departed_vehicles': departed_vehicles,
                'loaded_vehicles': traci.simulation.getLoadedNumber(),
                'simulation_time': simulation_time,
                'mean_travel_time': mean_travel_time,
                'mean_waiting_time': 0.0  # Would need additional tracking
            }
            
            # Calculate completion rate
            if departed_vehicles > 0:
                metrics['completion_rate'] = arrived_vehicles / departed_vehicles
            else:
                metrics['completion_rate'] = 0.0
                
            return metrics
        except Exception as e:
            print(f"Error collecting final metrics: {e}")
            return {
                'arrived_vehicles': 0,
                'departed_vehicles': 0,
                'loaded_vehicles': 0,
                'simulation_time': 0,
                'completion_rate': 0.0,
                'mean_travel_time': 0.0,
                'mean_waiting_time': 0.0
            }

    def close(self):
        """
        Close the TraCI connection.
        """
        traci.close()

    def get_current_phase(self, current_hour: float) -> str:
        """Get current traffic phase based on hour of day (0-24)"""
        if MORNING_PEAK_START <= current_hour < MORNING_PEAK_END:
            return "morning_peak"
        elif MIDDAY_OFFPEAK_START <= current_hour < MIDDAY_OFFPEAK_END:
            return "midday_offpeak"
        elif EVENING_PEAK_START <= current_hour < EVENING_PEAK_END:
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

        phases = ["morning_peak", "midday_offpeak",
                  "evening_peak", "night_low"]

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
            except Exception:
                # Edge might not exist in current simulation
                continue

        self.current_phase = new_phase

    def check_phase_transition(self, current_time_seconds: float):
        """Check if we need to transition to a new phase"""
        if not self.time_dependent:
            return

        # Convert simulation time to real-world hours
        hours_elapsed = current_time_seconds / SECONDS_PER_HOUR
        current_hour = (self.start_time_hour + hours_elapsed) % 24.0

        new_phase = self.get_current_phase(current_hour)

        if self.current_phase != new_phase:
            self.update_edge_attractiveness(new_phase)

    def load_vehicle_strategies(self):
        """Load vehicle routing strategies from route file."""
        try:
            tree = ET.parse(self.sumo_cfg)
            root = tree.getroot()

            # Find route file in config
            route_file = None
            for route_files in root.findall(".//route-files"):
                route_file = route_files.get('value')
                break

            if not route_file:
                return

            # Handle relative path - route file is relative to config file directory
            from pathlib import Path
            config_dir = Path(self.sumo_cfg).parent
            route_path = config_dir / route_file

            # Parse route file to get vehicle strategies
            route_tree = ET.parse(str(route_path))
            route_root = route_tree.getroot()

            for vehicle in route_root.findall('vehicle'):
                veh_id = vehicle.get('id')
                strategy = vehicle.get('routing_strategy', 'shortest')
                self.vehicle_strategies[veh_id] = strategy

                # Set initial rerouting time for dynamic strategies
                if strategy in self.strategy_intervals:
                    interval = self.strategy_intervals[strategy]
                    self.vehicle_rerouting_times[veh_id] = interval

        except Exception as e:
            print(f"Warning: Could not load vehicle strategies: {e}")

    def handle_dynamic_rerouting(self, current_time: float):
        """Handle dynamic rerouting for vehicles with real-time strategies."""
        try:
            # Get all vehicles currently in simulation
            vehicle_ids = traci.vehicle.getIDList()

            for veh_id in vehicle_ids:
                strategy = self.vehicle_strategies.get(veh_id, 'shortest')

                # Only reroute vehicles with dynamic strategies
                if strategy not in self.strategy_intervals:
                    continue

                # Check if it's time to reroute this vehicle
                next_reroute_time = self.vehicle_rerouting_times.get(veh_id, 0)
                if current_time >= next_reroute_time:
                    try:
                        # Get current route and destination
                        current_route = traci.vehicle.getRoute(veh_id)
                        if not current_route:
                            continue

                        current_edge = traci.vehicle.getRoadID(veh_id)
                        destination = current_route[-1]

                        # Skip if vehicle is on internal edge or already at destination
                        if current_edge.startswith(':') or current_edge == destination:
                            continue

                        # Compute new route based on strategy
                        if strategy == 'realtime':
                            # Use fastest path based on current conditions
                            new_route = traci.simulation.findRoute(
                                current_edge, destination)
                            if new_route and new_route.edges:
                                traci.vehicle.setRoute(veh_id, new_route.edges)

                        elif strategy == 'fastest':
                            # Use fastest path based on current travel times
                            new_route = traci.simulation.findRoute(
                                current_edge, destination)
                            if new_route and new_route.edges:
                                traci.vehicle.setRoute(veh_id, new_route.edges)

                        # Schedule next rerouting
                        interval = self.strategy_intervals[strategy]
                        self.vehicle_rerouting_times[veh_id] = current_time + interval

                    except Exception as e:
                        # Don't let rerouting errors stop the simulation
                        continue

        except Exception as e:
            # Don't let rerouting errors stop the simulation
            pass

    def run(self, control_callback):
        """
        Run the simulation loop with optional phase switching and dynamic rerouting.

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
            print(
                f"Starting simulation at {self.start_time_hour:.1f}h in phase: {initial_phase}")

        # Load vehicle routing strategies for dynamic rerouting
        self.load_vehicle_strategies()
        dynamic_strategies = [
            s for s in self.vehicle_strategies.values() if s in self.strategy_intervals]
        if dynamic_strategies:
            print(
                f"Dynamic rerouting enabled for strategies: {set(dynamic_strategies)}")

        current_time = 0
        while current_time < self.end_time:
            self.step()

            # Track vehicle metrics at each step
            self.track_vehicle_metrics(current_time)

            # Check for phase transitions
            if self.time_dependent:
                self.check_phase_transition(current_time)

            # Handle dynamic rerouting
            self.handle_dynamic_rerouting(current_time)

            # Apply control callback (Tree Method algorithm, etc.)
            control_callback(current_time)
            current_time += self.step_length

        # Collect final metrics before closing
        self.final_metrics = self.collect_final_metrics()
        
        # Print final status
        print(f"Simulation ended at time {current_time}")
        self.close()
