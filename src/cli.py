import argparse
import random
import shutil
from pathlib import Path
import traci
import xml.etree.ElementTree as ET
import sumolib


from src.sim.sumo_utils import generate_sumo_conf_file, rebuild_network

from src.network.generate_grid import generate_grid_network
from src.network.split_edges_with_lanes import split_edges_with_flow_based_lanes
from src.network.edge_attrs import assign_edge_attractiveness
from src.network.zones import extract_zones_from_junctions
from src.network.intelligent_zones import convert_zones_to_projected_coordinates, extract_zones_from_osm
from src.network.import_osm import generate_network_from_osm

from src.traffic_control.decentralized_traffic_bottlenecks.integration import load_tree
from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
from src.traffic_control.decentralized_traffic_bottlenecks.classes.network import Network
from src.traffic_control.decentralized_traffic_bottlenecks.classes.net_data_builder import build_network_json
from src.traffic_control.decentralized_traffic_bottlenecks.utils import is_calculation_time, calc_iteration_from_step
from src.traffic_control.decentralized_traffic_bottlenecks.enums import AlgoType, CostType
from src.traffic_control.decentralized_traffic_bottlenecks.classes.iterations_trees import IterationTrees

from src.validate.errors import ValidationError
from src.validate.validate_network import verify_generate_grid_network, verify_extract_zones_from_junctions, verify_rebuild_network, verify_assign_edge_attractiveness, verify_generate_sumo_conf_file
from src.validate.validate_traffic import verify_generate_vehicle_routes
from src.validate.validate_simulation import verify_tree_method_integration_setup, verify_algorithm_runtime_behavior
from src.validate.validate_arguments import validate_arguments
from src.validate.validate_intelligent_zones import verify_convert_zones_to_projected_coordinates
from src.validate.validate_split_edges_with_lanes import verify_split_edges_with_flow_based_lanes

from src.traffic.builder import generate_vehicle_routes
from src.config import CONFIG


def validate_tree_method_sample_args(args):
    """Validate arguments when using --tree_method_sample"""
    if args.tree_method_sample:
        # Incompatible arguments (generate network-related)
        incompatible = []
        if args.osm_file:
            incompatible.append('--osm_file')
        if args.grid_dimension != 5:  # non-default
            incompatible.append('--grid_dimension')
        if args.block_size_m != 200:  # non-default
            incompatible.append('--block_size_m')
        if args.junctions_to_remove != "0":
            incompatible.append('--junctions_to_remove')
        if args.lane_count != "realistic":
            incompatible.append('--lane_count')
            
        if incompatible:
            raise ValidationError(f"--tree_method_sample incompatible with: {', '.join(incompatible)}")


def update_sumo_config_paths():
    """Update SUMO config file to reference our file naming convention"""
    tree = ET.parse(CONFIG.config_file)
    root = tree.getroot()
    
    # Update file paths to match our naming
    for input_elem in root.findall('.//input'):
        net_file = input_elem.find('net-file')
        if net_file is not None:
            net_file.set('value', 'grid.net.xml')
            
        route_files = input_elem.find('route-files')
        if route_files is not None:
            route_files.set('value', 'vehicles.rou.xml')
    
    # Save updated config
    tree.write(CONFIG.config_file, encoding='utf-8', xml_declaration=True)
    print("Updated SUMO config file paths")


def setup_tree_method_sample(sample_folder: str):
    """Copy and adapt Tree Method sample files for our pipeline"""
    try:
        # Validate sample folder exists
        sample_path = Path(sample_folder)
        if not sample_path.exists():
            raise ValueError(f"Sample folder not found: {sample_folder}")
        
        # Required files in sample folder
        required_files = {
            'network.net.xml': CONFIG.network_file,           # -> data/grid.net.xml
            'vehicles.trips.xml': CONFIG.routes_file,         # -> data/vehicles.rou.xml  
            'simulation.sumocfg.xml': CONFIG.config_file      # -> data/grid.sumocfg
        }
        
        # Copy and rename files to our convention
        for source_name, target_path in required_files.items():
            source_file = sample_path / source_name
            if not source_file.exists():
                raise ValueError(f"Required file missing: {source_file}")
            
            shutil.copy2(source_file, target_path)
            print(f"Copied {source_name} -> {target_path}")
        
        # Update SUMO config file to use our file naming convention
        update_sumo_config_paths()
        
    except FileNotFoundError as e:
        print(f"Error: Sample file not found - {e}")
        exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied - {e}")
        exit(1)
    except Exception as e:
        print(f"Error setting up Tree Method sample: {e}")
        exit(1)


def parse_routing_strategy(routing_strategy):
    """Parse routing strategy string to extract individual strategies and percentages"""
    strategies = {}
    parts = routing_strategy.split()
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            strategy = parts[i]
            percentage = float(parts[i + 1])
            strategies[strategy] = percentage
    return strategies


def should_reroute_vehicles(step, strategy, last_reroute, interval):
    """Check if vehicles should be rerouted based on strategy and interval"""
    return step - last_reroute >= interval


def reroute_vehicles_by_strategy(strategy_type):
    """Reroute vehicles based on strategy type"""
    vehicle_ids = traci.vehicle.getIDList()
    for vehicle_id in vehicle_ids:
        try:
            if strategy_type == "realtime":
                traci.vehicle.rerouteEffort(vehicle_id)
            elif strategy_type == "fastest":
                traci.vehicle.reroute(vehicle_id)
        except traci.exceptions.TraCIException:
            # Vehicle might have left the simulation
            continue


def run_tree_method_simulation(args):
    """Universal simulation using Tree Method approach for all traffic control methods"""
    print("--- Step 9: Dynamic Simulation (Tree Method Universal Approach) ---")
    
    # Initialize variables for all traffic control methods
    tree_data = run_config = network_data = graph = seconds_in_cycle = None
    iteration_trees = []  # List to store iteration trees
    
    # Traffic control method-specific initialization
    if args.traffic_control == 'tree_method':
        print("Initializing Tree Method objects...")
        
        # Build network JSON for Tree Method
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        build_network_json(CONFIG.network_file, json_file)
        print(f"Built network JSON file: {json_file}")
        
        # Load Tree Method objects
        try:
            tree_data, run_config = load_tree(
                net_file=CONFIG.network_file,
                sumo_cfg=CONFIG.config_file
            )
            print("Loaded network tree and run configuration successfully.")
            
            network_data = Network(json_file)
            print("Loaded network data from JSON.")
            graph = Graph(args.end_time)
            print("Initialized Tree Method Graph object.")
            graph.build(network_data.edges_list, network_data.junctions_dict)
            print("Built Tree Method Graph from network data.")
            seconds_in_cycle = network_data.calc_cycle_time()
            print("Built network graph and calculated cycle time.")
            
            # Initialize iteration trees (will be created as needed during simulation)
            
            # Verify Tree Method integration setup
            verify_tree_method_integration_setup(
                tree_data, run_config, network_data, graph, seconds_in_cycle)
            print("Tree Method integration setup verified successfully.")
            
        except ValidationError as ve:
            print(f"Tree Method integration setup validation failed: {ve}")
            exit(1)
    
    elif args.traffic_control == "actuated":
        print("Using SUMO Actuated traffic control - no additional setup needed.")
    
    elif args.traffic_control == "fixed":
        print("Using Fixed-time traffic control - no additional setup needed.")
    
    # Parse routing strategies
    routing_strategies = {}
    if hasattr(args, 'routing_strategy') and args.routing_strategy:
        routing_strategies = parse_routing_strategy(args.routing_strategy)
    
    # Choose SUMO binary based on GUI flag
    if args.gui:
        sumo_binary = sumolib.checkBinary('sumo-gui')
    else:
        sumo_binary = sumolib.checkBinary('sumo')
    
    # Start TraCI (Tree Method way)
    print("Starting SUMO simulation with TraCI...")
    traci.start([sumo_binary, '-c', str(CONFIG.config_file)])
    
    # Initialize simulation variables
    step = 0
    iteration = 0
    last_realtime_reroute = 0
    last_fastest_reroute = 0
    
    # Simulation metrics
    total_vehicles = 0
    completed_vehicles = 0
    simulation_start_time = step
    
    try:
        # Main simulation loop (Tree Method pattern)
        while traci.simulation.getMinExpectedNumber() > 0 and step < args.end_time:
            
            # Tree Method: Calculation time check
            if args.traffic_control == "tree_method" and is_calculation_time(step, seconds_in_cycle):
                iteration = calc_iteration_from_step(step, seconds_in_cycle)
                if iteration > 0:  # Skip first iteration
                    print(f"Tree Method calculation at step {step}, iteration {iteration}")
                    try:
                        # Perform Tree Method calculations
                        ended_iteration = iteration - 1
                        this_iter_trees_costs = graph.calculate_iteration(
                            ended_iteration, 
                            iteration_trees, 
                            step, 
                            seconds_in_cycle,
                            run_config.cost_type if run_config else CostType.TREE_CURRENT, 
                            run_config.algo_type if run_config else AlgoType.BABY_STEPS
                        )
                        graph.calc_nodes_statistics(ended_iteration, seconds_in_cycle)
                    except Exception as e:
                        print(f"Warning: Tree Method calculation failed at step {step}: {e}")
            
            # Traffic control method-specific updates (BEFORE simulation step)
            if args.traffic_control == "tree_method" and graph:
                try:
                    # Tree Method traffic light updates
                    graph.update_traffic_lights(step, seconds_in_cycle, 
                                               run_config.algo_type if run_config else AlgoType.BABY_STEPS)
                except Exception as e:
                    print(f"Warning: Tree Method traffic light update failed at step {step}: {e}")
            
            # Dynamic rerouting logic (our addition)
            if routing_strategies:
                if "realtime" in routing_strategies and should_reroute_vehicles(step, "realtime", last_realtime_reroute, 30):
                    reroute_vehicles_by_strategy("realtime")
                    last_realtime_reroute = step
                    
                if "fastest" in routing_strategies and should_reroute_vehicles(step, "fastest", last_fastest_reroute, 45):
                    reroute_vehicles_by_strategy("fastest")
                    last_fastest_reroute = step
            
            # Core simulation step (Tree Method pattern)
            traci.simulationStep()
            
            # Post-step processing (metrics collection)
            if step % 100 == 0:  # Log every 100 steps
                current_vehicles = len(traci.vehicle.getIDList())
                if step == simulation_start_time:
                    total_vehicles = max(total_vehicles, current_vehicles)
                print(f"Step {step}: {current_vehicles} vehicles active")
            
            # Tree Method: Post-step data collection
            if args.traffic_control == "tree_method" and graph:
                try:
                    # Skip fill_link_in_step if there are None edges - data quality issue in Tree Method samples
                    if hasattr(graph, 'all_links'):
                        valid_links = [link for link in graph.all_links if link.edge_name != "None" and link.edge_name is not None]
                        if len(valid_links) == len(graph.all_links):
                            graph.fill_link_in_step()
                        # Skip this step if there are None edges to avoid TraCI errors
                    
                    graph.add_vehicles_to_step()
                    graph.close_prev_vehicle_step(step)
                    current_iteration = calc_iteration_from_step(step, seconds_in_cycle)
                    graph.fill_head_iteration()
                except Exception as e:
                    # Only print every 100 steps to avoid spam
                    if step % 100 == 0:
                        print(f"Warning: Tree Method post-processing failed at step {step}: {e}")
            
            # Runtime validation (every 30 steps by default)
            if args.traffic_control == "tree_method" and step % CONFIG.SIMULATION_VERIFICATION_FREQUENCY == 0:
                try:
                    # Get current phase map for validation
                    phase_map = graph.get_traffic_lights_phases(step) if graph else {}
                    verify_algorithm_runtime_behavior(
                        step, phase_map, graph, CONFIG.SIMULATION_VERIFICATION_FREQUENCY
                    )
                except ValidationError as ve:
                    print(f"Algorithm runtime validation failed at step {step}: {ve}")
            
            step += 1
        
        # Simulation completed
        print(f"Simulation completed at step {step}")
        
        # Collect final metrics
        final_vehicles = len(traci.vehicle.getIDList())
        completed_vehicles = total_vehicles - final_vehicles
        
        print("\n=== SIMULATION METRICS ===")
        print(f"Total simulation steps: {step}")
        print(f"Total vehicles: {total_vehicles}")
        print(f"Completed vehicles: {completed_vehicles}")
        if total_vehicles > 0:
            completion_rate = (completed_vehicles / total_vehicles) * 100
            print(f"Completion rate: {completion_rate:.1f}%")
        print(f"Traffic control method: {args.traffic_control}")
        
    except Exception as e:
        print(f"Simulation error: {e}")
        raise
    finally:
        # Clean shutdown
        try:
            traci.close()
        except:
            pass
    
    print("Tree Method simulation completed successfully!")




def main():
    # --- Clean the data directory ---
    if CONFIG.output_dir.exists():
        shutil.rmtree(CONFIG.output_dir)
    CONFIG.output_dir.mkdir(exist_ok=True)

    # --- Command-line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Generate and simulate a SUMO orthogonal grid network with dynamic traffic-light control."
    )
    parser.add_argument(
        "--grid_dimension",
        type=float,
        default=5,
        help="The grid's number of rows and columns. Default is 5 (5x5 grid)."
    )
    parser.add_argument(
        "--block_size_m",
        type=int,
        default=200,
        help="Block size in meters. Default is 200m."
    )
    parser.add_argument(
        "--junctions_to_remove",
        type=str,
        default="0",
        help="Number of junctions to remove from the grid (e.g., '5') or comma-separated list of specific junction IDs (e.g., 'A0,B1,C2'). Default is 0."
    )
    parser.add_argument(
        "--lane_count",
        type=str,
        default="realistic",
        help="Lane count algorithm: 'realistic' (default, zone-based), 'random', or integer (fixed count for all edges)."
    )
    parser.add_argument(
        "--num_vehicles",
        type=int,
        default=CONFIG.DEFAULT_NUM_VEHICLES,
        help=f"Number of vehicles to generate. Default is {CONFIG.DEFAULT_NUM_VEHICLES}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for generating randomness. If not provided, a random seed will be used."
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=1.0,
        help="Simulation step length in seconds (for TraCI loop)."
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=86400,
        help="Total simulation duration in seconds. Default is 86400 (24 hours/full day)."
    )
    parser.add_argument(
        "--attractiveness",
        type=str,
        default="poisson",
        choices=["poisson", "land_use", "gravity", "iac", "hybrid"],
        help="Edge attractiveness method: 'poisson' (default), 'land_use', 'gravity', 'iac', or 'hybrid'."
    )
    parser.add_argument(
        "--time_dependent",
        action="store_true",
        help="Apply 4-phase time-of-day variations to the selected attractiveness method"
    )
    parser.add_argument(
        "--start_time_hour",
        type=float,
        default=0.0,
        help="Real-world hour when simulation starts (0-24, default: 0.0 for midnight)"
    )
    parser.add_argument(
        "--departure_pattern",
        type=str,
        default="six_periods",
        help="Vehicle departure pattern: 'six_periods' (default, research-based), 'uniform', 'rush_hours:7-9:40,17-19:30,rest:10', or 'hourly:7:25,8:35,rest:5'"
    )
    parser.add_argument(
        "--routing_strategy",
        type=str,
        default="shortest 100",
        help="Routing strategy with percentages (e.g., 'shortest 70 realtime 30' or 'shortest 20 realtime 30 fastest 45 attractiveness 5'). Default: 'shortest 100'"
    )
    parser.add_argument(
        "--vehicle_types",
        type=str,
        default=CONFIG.DEFAULT_VEHICLE_TYPES,
        help="Vehicle types with percentages (e.g., 'passenger 70 commercial 20 public 10'). Default: 'passenger 60 commercial 30 public 10'"
    )
    parser.add_argument(
        "--traffic_light_strategy",
        type=str,
        default="opposites",
        choices=["opposites", "incoming"],
        help="Traffic light phasing strategy: 'opposites' (default, opposing directions together) or 'incoming' (each edge gets own phase)"
    )
    parser.add_argument(
        "--traffic_control",
        type=str,
        default="tree_method",
        choices=["tree_method", "actuated", "fixed"],
        help="Traffic control method: 'tree_method' (default, Tree Method algorithm), 'actuated' (SUMO gap-based), or 'fixed' (static timing)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO in GUI mode (sumo-gui) instead of headless sumo"
    )
    parser.add_argument(
        "--osm_file",
        type=str,
        help="Path to OSM file to use instead of generating synthetic grid network"
    )
    parser.add_argument(
        "--land_use_block_size_m",
        type=float,
        default=25.0,
        help="Size of land use zone grid blocks in meters. Default: 25.0m (following research paper methodology). Controls resolution of zone generation."
    )
    parser.add_argument(
        "--tree_method_sample",
        type=str,
        metavar="FOLDER_PATH",
        help="Use pre-built Tree Method sample from specified folder (skips steps 1-8, goes directly to simulation)"
    )
    args = parser.parse_args()

    # --- Validate arguments ---
    try:
        validate_arguments(args)
        validate_tree_method_sample_args(args)
    except ValidationError as e:
        print(f"Error: {e}")
        exit(1)

    try:
        # --- Initialize seed ---
        seed = args.seed if args.seed is not None else random.randint(
            0, 2**32 - 1)
        print(f"Using seed: {seed}")

        # --- Check for Tree Method Sample Bypass Mode ---
        if args.tree_method_sample:
            print(f"Tree Method Sample Mode: Using pre-built network from {args.tree_method_sample}")
            print("Skipping Steps 1-8, going directly to Step 9 (Dynamic Simulation)")
            setup_tree_method_sample(args.tree_method_sample)
            run_tree_method_simulation(args)
            return

        # --- Step 1: Network Generation ---
        if args.osm_file:
            print("Begin simulating SUMO network from OSM data...")
            generate_network_from_osm(args.osm_file)
            print("Successfully generated SUMO network from OSM data.")
        else:
            print("Begin simulating SUMO orthogonal grid network...")
            generate_grid_network(
                seed,
                int(args.grid_dimension),
                int(args.block_size_m),
                args.junctions_to_remove,
                args.lane_count,
                args.traffic_light_strategy
            )
            try:
                verify_generate_grid_network(
                    seed,
                    int(args.grid_dimension),
                    int(args.block_size_m),
                    args.junctions_to_remove,
                    args.lane_count,
                    args.traffic_light_strategy
                )
            except ValidationError as ve:
                print(f"Failed generating network file: {ve}")
                exit(1)
            print(f"Generated grid successfully.")

        # --- Step 2: Zone Generation ---

        if args.osm_file:
            print("Generating OSM-based intelligent zones...")
            try:
                num_zones = extract_zones_from_osm(
                    osm_file_path=args.osm_file,
                    land_use_block_size_m=args.land_use_block_size_m,
                    zones_file=CONFIG.zones_file
                )
                print(
                    f"Generated and saved {num_zones} intelligent zones to {CONFIG.zones_file}")
            except Exception as e:
                print(f"Failed to generate OSM zones: {e}")
                exit(1)
        else:
            print("Generating synthetic network zones using traditional method...")
            # For synthetic networks, use the original zone extraction method with configurable block size
            extract_zones_from_junctions(
                args.land_use_block_size_m,
                seed=seed,
                fill_polygons=True,
                inset=0.0
            )
            try:
                verify_extract_zones_from_junctions(
                    args.land_use_block_size_m,
                    seed=seed,
                    fill_polygons=True,
                    inset=0.0
                )
            except ValidationError as ve:
                print(f"Failed to extract land use zones: {ve}")
                exit(1)
            print(
                f"Extracted land use zones successfully using traditional method with {args.land_use_block_size_m}m blocks.")

        # --- Step 3: Integrated Edge Splitting with Lane Assignment ---
        if args.lane_count != "0" and not (args.lane_count.isdigit() and args.lane_count == "0"):
            split_edges_with_flow_based_lanes(
                seed=seed,
                min_lanes=CONFIG.MIN_LANES,
                max_lanes=CONFIG.MAX_LANES,
                algorithm=args.lane_count,
                block_size_m=args.block_size_m
            )
            print(
                "Successfully completed integrated edge splitting with lane assignment.")

            # Validate the split edges with flow-based lanes (immediately after the operation)
            try:
                verify_split_edges_with_flow_based_lanes(
                    connections_file=str(CONFIG.network_con_file),
                    edges_file=str(CONFIG.network_edg_file),
                    nodes_file=str(CONFIG.network_nod_file)
                )
                print("Split edges validation passed successfully.")
            except ValidationError as ve:
                print(f"Split edges validation failed: {ve}")
                exit(1)
            except ValueError as ve:
                print(f"Split edges validation failed: {ve}")
                exit(1)
        else:
            print("Skipping lane assignment (lane_count is 0).")

        # --- Step 4: Network Rebuild ---
        rebuild_network()
        try:
            verify_rebuild_network()
        except ValidationError as ve:
            print(f"Failed to rebuild the network: {ve}")
            exit(1)
        print("Rebuilt the network successfully.")

        # --- Step 5: Zone Coordinate Conversion (OSM Mode Only) ---
        if args.osm_file and Path(CONFIG.zones_file).exists():
            print("Converting OSM zone coordinates from geographic to projected...")
            try:
                # Convert zones from geographic to projected coordinates
                convert_zones_to_projected_coordinates(
                    CONFIG.zones_file, CONFIG.network_file)
                print("Successfully converted zone coordinates to projected system.")
            except Exception as e:
                print(f"Failed to convert zone coordinates: {e}")
                print("Zones will remain in geographic coordinates.")

            print("Validating zone coverage against network bounds...")
            try:
                verify_convert_zones_to_projected_coordinates(
                    CONFIG.zones_file, CONFIG.network_file)
                print("Zone coverage validation passed.")
            except ValidationError as ve:
                print(f"Zone coverage validation failed: {ve}")
                exit(1)
            except Exception as e:
                print(f"Zone coverage validation failed: {e}")
                exit(1)

        # --- Step 6: Edge Attractiveness Assignment ---
        assign_edge_attractiveness(
            seed, args.attractiveness, args.time_dependent, args.start_time_hour)
        try:
            verify_assign_edge_attractiveness(
                seed, args.attractiveness, args.time_dependent)
        except ValidationError as ve:
            print(f"Failed to assign edge attractiveness: {ve}")
            exit(1)
        print("Assigned edge attractiveness successfully.")

        # --- Step 7: Vehicle Route Generation ---
        generate_vehicle_routes(
            net_file=CONFIG.network_file,
            output_file=CONFIG.routes_file,
            num_vehicles=args.num_vehicles,
            seed=seed,
            routing_strategy=args.routing_strategy,
            vehicle_types=args.vehicle_types,
            end_time=args.end_time,
            departure_pattern=args.departure_pattern
        )
        try:
            verify_generate_vehicle_routes(
                net_file=CONFIG.network_file,
                output_file=CONFIG.routes_file,
                num_vehicles=args.num_vehicles,
                seed=seed,
            )
        except ValidationError as ve:
            print(f"Failed to generate vehicle routes: {ve}")
            exit(1)
        print("Generated vehicle routes successfully.")

        # --- Step 8: SUMO Configuration Generation ---
        sumo_cfg_path = generate_sumo_conf_file(
            CONFIG.config_file,
            CONFIG.network_file,
            route_file=CONFIG.routes_file,
            zones_file=CONFIG.zones_file,
        )
        try:
            verify_generate_sumo_conf_file()
        except ValidationError as ve:
            print(f"SUMO configuration validation failed: {ve}")
            exit(1)
        print(f"Generated SUMO configuration file successfully.")

        # --- Step 9: Dynamic Simulation (Tree Method Universal Approach) ---
        run_tree_method_simulation(args)

    except Exception as e:
        import traceback
        print(f"An error occurred. Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
