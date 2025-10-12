import traci
import sys
from src.config import CONFIG
import xml.etree.ElementTree as ET
from typing import NoReturn, List

from src.constants import (
    SUMO_ROUTING_MODE_DEFAULT,
    SUMO_ROUTING_MODE_AGGREGATED,
    REALTIME_REROUTING_INTERVAL_SECONDS,
    FASTEST_REROUTING_INTERVAL_SECONDS,
    ATTRACTIVENESS_REROUTING_INTERVAL_SECONDS,
    REALTIME_ROUTE_IMPROVEMENT_THRESHOLD_PCT,
    FASTEST_ROUTE_IMPROVEMENT_THRESHOLD_PCT,
    ATTRACTIVENESS_ROUTE_IMPROVEMENT_THRESHOLD_PCT,
    ROUTING_ERROR_REALTIME_FAILED,
    ROUTING_ERROR_FASTEST_FAILED,
    ROUTING_ERROR_ATTRACTIVENESS_FAILED,
    ROUTING_ERROR_TRACI_FAILURE,
    ROUTING_ERROR_INTERSECTION_RESTRICTION,
    ROUTING_ERROR_XML_PARSING,
    ROUTING_ERROR_MSG_TEMPLATE,
    TRACI_ERROR_MSG,
    SUMO_CONSTRAINT_ERROR_MSG,
    XML_PARSING_ERROR_MSG,
    ROUTING_SHORTEST,
    ROUTING_REALTIME,
    ROUTING_FASTEST,
    ROUTING_ATTRACTIVENESS
)


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
                 start_time_hour: float = 0.0,
                 routing_strategy: str = "shortest 100"):
        """
        :param gui: if True, launch sumo-gui; otherwise, batch sumo.
        :param start_time_hour: real-world hour when simulation starts (0-24)
        :param routing_strategy: routing strategy specification for dynamic rerouting
        """
        self.sumo_cfg = sumo_cfg
        self.step_length = step_length
        self.end_time = end_time
        self.gui = gui
        self.start_time_hour = start_time_hour
        self.current_phase = None
        self.phase_profiles = {}

        # Dynamic routing support
        self.routing_strategy = routing_strategy
        self.vehicle_strategies = {}  # vehicle_id -> strategy_name
        self.vehicle_rerouting_times = {}  # vehicle_id -> next_rerouting_time
        self.strategy_intervals = {
            ROUTING_REALTIME: REALTIME_REROUTING_INTERVAL_SECONDS,
            ROUTING_FASTEST: FASTEST_REROUTING_INTERVAL_SECONDS,
            ROUTING_ATTRACTIVENESS: ATTRACTIVENESS_REROUTING_INTERVAL_SECONDS
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
        if new_phase not in self.phase_profiles:
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

        # Trigger attractiveness rerouting on phase change
        self.trigger_attractiveness_rerouting_on_phase_change(new_phase)

    def check_phase_transition(self, current_time_seconds: float):
        """Check if we need to transition to a new phase"""
        # Convert simulation time to real-world hours
        hours_elapsed = current_time_seconds / SECONDS_PER_HOUR
        current_hour = (self.start_time_hour + hours_elapsed) % 24.0

        new_phase = self.get_current_phase(current_hour)

        if self.current_phase != new_phase:
            self.update_edge_attractiveness(new_phase)

    def trigger_attractiveness_rerouting_on_phase_change(self, new_phase: str) -> None:
        """Trigger rerouting for attractiveness vehicles on phase change - TERMINATES ON ERROR."""
        try:
            # Get all vehicles currently in simulation
            vehicle_ids = traci.vehicle.getIDList()

            # Filter for attractiveness vehicles
            attractiveness_vehicles = [
                veh_id for veh_id in vehicle_ids
                if self.vehicle_strategies.get(veh_id) == ROUTING_ATTRACTIVENESS
            ]

            # Force immediate rerouting for all attractiveness vehicles
            for veh_id in attractiveness_vehicles:
                try:
                    current_time = traci.simulation.getTime()
                    self.handle_attractiveness_rerouting(veh_id, current_time, new_phase)

                    # Reset their rerouting timer to prevent immediate rerouting again
                    interval = self.strategy_intervals[ROUTING_ATTRACTIVENESS]
                    self.vehicle_rerouting_times[veh_id] = current_time + interval

                except Exception as e:
                    # If individual vehicle rerouting fails, log and terminate
                    error_msg = TRACI_ERROR_MSG.format(
                        code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                        vehicle_id=veh_id,
                        command="phase_change_rerouting",
                        reason=str(e)
                    )
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_TRACI_FAILURE,
                vehicle_id="BATCH",
                command="phase_change_attractiveness_rerouting",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def load_vehicle_strategies(self):
        """Load vehicle routing strategies from route file - TERMINATES ON ERROR."""
        try:
            # Parse SUMO config file
            tree = ET.parse(self.sumo_cfg)
            root = tree.getroot()

            # Find route file in config
            route_file = None
            for route_files in root.findall(".//route-files"):
                route_file = route_files.get('value')
                break

            if not route_file:
                error_msg = XML_PARSING_ERROR_MSG.format(
                    code=ROUTING_ERROR_XML_PARSING,
                    details=f"No route-files element found in SUMO config: {self.sumo_cfg}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Handle relative path - route file is relative to config file directory
            from pathlib import Path
            config_dir = Path(self.sumo_cfg).parent
            route_path = config_dir / route_file

            # Validate route file exists
            if not route_path.exists():
                error_msg = XML_PARSING_ERROR_MSG.format(
                    code=ROUTING_ERROR_XML_PARSING,
                    details=f"Route file does not exist: {route_path}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Parse route file to get vehicle strategies
            route_tree = ET.parse(str(route_path))
            route_root = route_tree.getroot()

            valid_strategies = {ROUTING_SHORTEST, ROUTING_REALTIME, ROUTING_FASTEST, ROUTING_ATTRACTIVENESS}
            vehicle_ids_seen = set()

            for vehicle in route_root.findall('vehicle'):
                veh_id = vehicle.get('id')
                if not veh_id:
                    error_msg = XML_PARSING_ERROR_MSG.format(
                        code=ROUTING_ERROR_XML_PARSING,
                        details="Vehicle element missing required 'id' attribute"
                    )
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)

                # Check for duplicate vehicle IDs
                if veh_id in vehicle_ids_seen:
                    error_msg = XML_PARSING_ERROR_MSG.format(
                        code=ROUTING_ERROR_XML_PARSING,
                        details=f"Duplicate vehicle ID found: {veh_id}"
                    )
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)
                vehicle_ids_seen.add(veh_id)

                # Get routing strategy (required attribute - no default)
                strategy = vehicle.get('routing_strategy')
                if not strategy:
                    error_msg = XML_PARSING_ERROR_MSG.format(
                        code=ROUTING_ERROR_XML_PARSING,
                        details=f"Vehicle {veh_id} missing required 'routing_strategy' attribute"
                    )
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)

                # Validate strategy is known
                if strategy not in valid_strategies:
                    error_msg = XML_PARSING_ERROR_MSG.format(
                        code=ROUTING_ERROR_XML_PARSING,
                        details=f"Vehicle {veh_id} has unknown routing strategy: {strategy}. Valid strategies: {valid_strategies}"
                    )
                    print(error_msg, file=sys.stderr)
                    sys.exit(1)

                self.vehicle_strategies[veh_id] = strategy

                # Set initial rerouting time for dynamic strategies
                if strategy in self.strategy_intervals:
                    interval = self.strategy_intervals[strategy]
                    self.vehicle_rerouting_times[veh_id] = interval

        except ET.ParseError as e:
            error_msg = XML_PARSING_ERROR_MSG.format(
                code=ROUTING_ERROR_XML_PARSING,
                details=f"XML parsing error: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            error_msg = XML_PARSING_ERROR_MSG.format(
                code=ROUTING_ERROR_XML_PARSING,
                details=f"Vehicle strategy loading failed: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def handle_realtime_rerouting(self, vehicle_id: str, current_time: float) -> None:
        """Handle realtime strategy rerouting with SUMO aggregated mode - TERMINATES ON ERROR."""
        try:
            # Get current route for improvement validation
            current_route = traci.vehicle.getRoute(vehicle_id)
            if not current_route:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_REALTIME_FAILED,
                    strategy=ROUTING_REALTIME,
                    vehicle_id=vehicle_id,
                    reason="Vehicle has no current route"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Get current position for route calculation
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            destination = current_route[-1]

            # Skip if vehicle is on internal edge or already at destination
            if current_edge.startswith(':') or current_edge == destination:
                return

            # Calculate potential new route using aggregated mode
            traci.vehicle.setRoutingMode(vehicle_id, SUMO_ROUTING_MODE_AGGREGATED)
            potential_route = traci.simulation.findRoute(current_edge, destination)

            if not potential_route or not potential_route.edges:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_REALTIME_FAILED,
                    strategy=ROUTING_REALTIME,
                    vehicle_id=vehicle_id,
                    reason=f"No route found from {current_edge} to {destination}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Get route from current position to destination for comparison
            try:
                current_edge_index = current_route.index(current_edge)
                remaining_current_route = current_route[current_edge_index:]
            except ValueError:
                # Current edge not in route (vehicle deviated), use full current route for comparison
                remaining_current_route = current_route

            # Calculate route improvement
            improvement_pct = self.calculate_route_improvement(
                vehicle_id, remaining_current_route, potential_route.edges
            )

            # Only apply route if improvement exceeds threshold
            if improvement_pct >= REALTIME_ROUTE_IMPROVEMENT_THRESHOLD_PCT:
                traci.vehicle.setRoute(vehicle_id, potential_route.edges)

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_REALTIME_FAILED,
                vehicle_id=vehicle_id,
                command="rerouteTraveltime",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def handle_fastest_rerouting(self, vehicle_id: str, current_time: float) -> None:
        """Handle fastest strategy rerouting with effort-based routing - TERMINATES ON ERROR."""
        try:
            # Get current route for improvement validation
            current_route = traci.vehicle.getRoute(vehicle_id)
            if not current_route:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_FASTEST_FAILED,
                    strategy=ROUTING_FASTEST,
                    vehicle_id=vehicle_id,
                    reason="Vehicle has no current route"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Get current position for route calculation
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            destination = current_route[-1]

            # Skip if vehicle is on internal edge or already at destination
            if current_edge.startswith(':') or current_edge == destination:
                return

            # Calculate potential new route using default mode for fastest path
            traci.vehicle.setRoutingMode(vehicle_id, SUMO_ROUTING_MODE_DEFAULT)
            potential_route = traci.simulation.findRoute(current_edge, destination)

            if not potential_route or not potential_route.edges:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_FASTEST_FAILED,
                    strategy=ROUTING_FASTEST,
                    vehicle_id=vehicle_id,
                    reason=f"No route found from {current_edge} to {destination}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Get route from current position to destination for comparison
            try:
                current_edge_index = current_route.index(current_edge)
                remaining_current_route = current_route[current_edge_index:]
            except ValueError:
                # Current edge not in route (vehicle deviated), use full current route for comparison
                remaining_current_route = current_route

            # Calculate route improvement
            improvement_pct = self.calculate_route_improvement(
                vehicle_id, remaining_current_route, potential_route.edges
            )

            # Only apply route if improvement exceeds threshold
            if improvement_pct >= FASTEST_ROUTE_IMPROVEMENT_THRESHOLD_PCT:
                traci.vehicle.setRoute(vehicle_id, potential_route.edges)

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_FASTEST_FAILED,
                vehicle_id=vehicle_id,
                command="rerouteEffort",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def handle_attractiveness_rerouting(self, vehicle_id: str, current_time: float, current_phase: str) -> None:
        """Handle attractiveness strategy with multi-criteria scoring - TERMINATES ON ERROR."""
        try:
            # Get current route and destination
            current_route = traci.vehicle.getRoute(vehicle_id)
            if not current_route:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                    strategy=ROUTING_ATTRACTIVENESS,
                    vehicle_id=vehicle_id,
                    reason="Vehicle has no current route"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            destination = current_route[-1]

            # Skip if vehicle is on internal edge or already at destination
            if current_edge.startswith(':') or current_edge == destination:
                return

            # For attractiveness routing, use findRoute with custom effort evaluation
            # This is a simplified implementation - could be enhanced with multiple route comparison
            potential_route = traci.simulation.findRoute(current_edge, destination)

            if not potential_route or not potential_route.edges:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                    strategy=ROUTING_ATTRACTIVENESS,
                    vehicle_id=vehicle_id,
                    reason=f"No route found from {current_edge} to {destination}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Get route from current position to destination for comparison
            try:
                current_edge_index = current_route.index(current_edge)
                remaining_current_route = current_route[current_edge_index:]
            except ValueError:
                # Current edge not in route (vehicle deviated), use full current route for comparison
                remaining_current_route = current_route

            # Calculate route improvement
            improvement_pct = self.calculate_route_improvement(
                vehicle_id, remaining_current_route, potential_route.edges
            )

            # Only apply route if improvement exceeds threshold
            if improvement_pct >= ATTRACTIVENESS_ROUTE_IMPROVEMENT_THRESHOLD_PCT:
                traci.vehicle.setRoute(vehicle_id, potential_route.edges)

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                vehicle_id=vehicle_id,
                command="attractiveness_rerouting",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def calculate_route_improvement(self, vehicle_id: str, current_route: List[str], new_route: List[str]) -> float:
        """
        Calculate improvement percentage between current and new routes.

        Args:
            vehicle_id: SUMO vehicle ID
            current_route: Current route edges
            new_route: Proposed new route edges

        Returns:
            Improvement percentage (positive = new route is better, negative = worse)

        Raises:
            Terminates program on TraCI errors
        """
        try:
            # Calculate travel times for both routes
            current_travel_time = 0.0
            new_travel_time = 0.0

            # Calculate current route travel time
            for edge_id in current_route:
                if not edge_id.startswith(':'):  # Skip internal edges
                    current_travel_time += traci.edge.getTraveltime(edge_id)

            # Calculate new route travel time
            for edge_id in new_route:
                if not edge_id.startswith(':'):  # Skip internal edges
                    new_travel_time += traci.edge.getTraveltime(edge_id)

            # Avoid division by zero
            if current_travel_time <= 0:
                return 0.0

            # Calculate improvement percentage
            # Positive = new route is better (faster)
            # Negative = new route is worse (slower)
            improvement_pct = ((current_travel_time - new_travel_time) / current_travel_time) * 100.0

            return improvement_pct

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_TRACI_FAILURE,
                vehicle_id=vehicle_id,
                command="calculate_route_improvement",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def validate_route_before_application(self, vehicle_id: str, new_route: List[str]) -> None:
        """Validate route meets SUMO constraints before application - TERMINATES ON INVALID ROUTE."""
        try:
            # Check if vehicle is at intersection (SUMO constraint)
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge.startswith(':'):
                error_msg = SUMO_CONSTRAINT_ERROR_MSG.format(
                    code=ROUTING_ERROR_INTERSECTION_RESTRICTION,
                    vehicle_id=vehicle_id,
                    constraint="Cannot change route while vehicle is at intersection"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Validate route is not empty
            if not new_route:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                    strategy="validation",
                    vehicle_id=vehicle_id,
                    reason="Route is empty"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Validate first edge matches current location (SUMO requirement)
            if new_route[0] != current_edge:
                error_msg = SUMO_CONSTRAINT_ERROR_MSG.format(
                    code=ROUTING_ERROR_INTERSECTION_RESTRICTION,
                    vehicle_id=vehicle_id,
                    constraint=f"First edge in route ({new_route[0]}) must match current location ({current_edge})"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_TRACI_FAILURE,
                vehicle_id=vehicle_id,
                command="route_validation",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def handle_dynamic_rerouting(self, current_time: float):
        """Handle dynamic rerouting for vehicles with real-time strategies - TERMINATES ON ERROR."""
        try:
            # Get all vehicles currently in simulation
            vehicle_ids = traci.vehicle.getIDList()

            for veh_id in vehicle_ids:
                strategy = self.vehicle_strategies.get(veh_id, ROUTING_SHORTEST)

                # Only reroute vehicles with dynamic strategies
                if strategy not in self.strategy_intervals:
                    continue

                # Check if it's time to reroute this vehicle
                next_reroute_time = self.vehicle_rerouting_times.get(veh_id, 0)
                if current_time >= next_reroute_time:
                    # Route based on strategy
                    if strategy == ROUTING_REALTIME:
                        self.handle_realtime_rerouting(veh_id, current_time)
                    elif strategy == ROUTING_FASTEST:
                        self.handle_fastest_rerouting(veh_id, current_time)
                    elif strategy == ROUTING_ATTRACTIVENESS:
                        current_phase = self.get_current_phase_for_time(current_time)
                        self.handle_attractiveness_rerouting(veh_id, current_time, current_phase)

                    # Schedule next rerouting
                    interval = self.strategy_intervals[strategy]
                    self.vehicle_rerouting_times[veh_id] = current_time + interval

        except Exception as e:
            error_msg = TRACI_ERROR_MSG.format(
                code=ROUTING_ERROR_TRACI_FAILURE,
                vehicle_id="UNKNOWN",
                command="dynamic_rerouting",
                reason=str(e)
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def get_current_phase_for_time(self, current_time: float) -> str:
        """Get current phase based on simulation time."""
        # Convert simulation time to real-world hours
        hours_elapsed = current_time / SECONDS_PER_HOUR
        current_hour = (self.start_time_hour + hours_elapsed) % 24.0
        return self.get_current_phase(current_hour)

    def run(self, control_callback):
        """
        Run the simulation loop with optional phase switching and dynamic rerouting.

        :param control_callback: A function that takes the current simulation time (int)
                                 and applies control actions.
        """
        print("Running SUMO simulation...")
        self.start()

        # Load phase profiles if time-dependent
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
