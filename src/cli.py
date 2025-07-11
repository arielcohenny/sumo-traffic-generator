import argparse
import random
import shutil
from pathlib import Path
import traci


from src.sim.sumo_controller import SumoController
from src.sim.sumo_utils import generate_sumo_conf_file, rebuild_network

from src.network.generate_grid import generate_grid_network
from src.network.split_edges_with_lanes import split_edges_with_flow_based_lanes
from src.network.edge_attrs import assign_edge_attractiveness
from src.network.zones import extract_zones_from_junctions
from src.network.import_osm import import_osm_network

from src.traffic_control.decentralized_traffic_bottlenecks.integration import load_tree
from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
from src.traffic_control.decentralized_traffic_bottlenecks.classes.network import Network
from src.traffic_control.decentralized_traffic_bottlenecks.classes.net_data_builder import build_network_json

from src.validate.errors import ValidationError
from src.validate.validate_network import verify_generate_grid_network, verify_extract_zones_from_junctions, verify_rebuild_network, verify_assign_edge_attractiveness, verify_generate_sumo_conf_file
from src.validate.validate_traffic import verify_generate_vehicle_routes
from src.validate.validate_simulation import verify_nimrod_integration_setup, verify_algorithm_runtime_behavior

from src.traffic.builder import generate_vehicle_routes
from src.config import CONFIG


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
        help="Traffic control method: 'tree_method' (default, Nimrod's algorithm), 'actuated' (SUMO gap-based), or 'fixed' (static timing)."
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
    args = parser.parse_args()

    try:
        # --- Initialize seed ---
        seed = args.seed if args.seed is not None else random.randint(
            0, 2**32 - 1)
        print(f"Using seed: {seed}")

        # --- Step 1: Generate Network (Grid or OSM) ---
        if args.osm_file:
            print("Begin simulating SUMO network from OSM data...")
            import_osm_network(args.osm_file, "data/grid")
            print("Successfully imported OSM network.")
            
            # Move OSM files to expected locations in data/ directory
            grid_dir = Path("data/grid")
            if grid_dir.exists():
                # Move files from data/grid/osm_network.* to data/grid.*
                for file_pattern in ["*.nod.xml", "*.edg.xml", "*.con.xml", "*.tll.xml"]:
                    for src_file in grid_dir.glob(file_pattern):
                        # Extract the file extension part (e.g., "nod.xml" from "osm_network.nod.xml")
                        suffix = src_file.name.split(".", 1)[1] if "." in src_file.name else src_file.suffix
                        dst_file = Path("data") / f"grid.{suffix}"
                        shutil.move(str(src_file), str(dst_file))
                        print(f"Moved {src_file} to {dst_file}")
                # Clean up empty grid directory
                if not list(grid_dir.iterdir()):
                    grid_dir.rmdir()
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

        # --- Step 2: Extract Zones ---
        extract_zones_from_junctions(
            args.block_size_m,
            seed=seed,
            fill_polygons=True,
            inset=0.0
        )
        try:
            verify_extract_zones_from_junctions(
                args.block_size_m,
                seed=seed,
                fill_polygons=True,
                inset=0.0
            )
        except ValidationError as ve:
            print(f"Failed to extract land use zones: {ve}")
            exit(1)
        print("Extracted land use zones successfully.")

        # --- Step 3: Integrated Edge Splitting with Flow-Based Lane Assignment ---
        if args.lane_count != "0" and not (args.lane_count.isdigit() and args.lane_count == "0"):
            split_edges_with_flow_based_lanes(
                seed=seed,
                min_lanes=CONFIG.MIN_LANES,
                max_lanes=CONFIG.MAX_LANES,
                algorithm=args.lane_count
            )
            print(
                "Successfully completed integrated edge splitting with flow-based lane assignment.")
        else:
            print("Skipping lane assignment (lane_count is 0).")

        # --- Step 4: Rebuild Network ---
        rebuild_network()
        try:
            verify_rebuild_network()
        except ValidationError as ve:
            print(f"Failed to rebuild the network: {ve}")
            exit(1)
        print("Rebuilt the network successfully.")

        # --- Step 5: Assign Edge Attractiveness ---
        assign_edge_attractiveness(
            seed, args.attractiveness, args.time_dependent, args.start_time_hour)
        try:
            verify_assign_edge_attractiveness(
                seed, args.attractiveness, args.time_dependent)
        except ValidationError as ve:
            print(f"Failed to assign edge attractiveness: {ve}")
            exit(1)
        print("Assigned edge attractiveness successfully.")

        # --- Step 6: Generate Vehicle Routes ---
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

        # --- Step 7: Generate SUMO Configuration File ---
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

        # --- Step 8: Dynamic Simulation via TraCI & Nimrod’s Tree Method ---
        # Initialize traffic control method-specific objects
        if args.traffic_control == "tree_method":
            # Load network tree structure and runner configuration
            json_file = Path(CONFIG.network_file).with_suffix(".json")
            if json_file.exists():
                json_file.unlink()
            tree_data, run_config = load_tree(
                net_file=CONFIG.network_file,
                sumo_cfg=sumo_cfg_path
            )
            print("Loaded network tree and run configuration successfully.")

            # Build Nimrod's runtime objects once (outside the callback)
            json_file = Path(CONFIG.network_file).with_suffix(".json")
            build_network_json(CONFIG.network_file, json_file)
            print(f"Built network JSON file: {json_file}")

            network_data = Network(json_file)
            print("Loaded network data from JSON.")
            graph = Graph(args.end_time)
            print("Initialized Nimrod's Graph object.")
            graph.build(network_data.edges_list, network_data.junctions_dict)
            print("Built Nimrod's Graph from network data.")
            seconds_in_cycle = network_data.calc_cycle_time()
            print("Built network graph and calculated cycle time.")

            # Verify Nimrod integration setup
            try:
                verify_nimrod_integration_setup(
                    tree_data, run_config, network_data, graph, seconds_in_cycle)
            except ValidationError as ve:
                print(f"Nimrod integration setup validation failed: {ve}")
                exit(1)
            print("Nimrod integration setup verified successfully.")
        
        elif args.traffic_control == "actuated":
            print("Using SUMO Actuated traffic control - no additional setup needed.")
            # Set variables to None for actuated control
            tree_data = run_config = network_data = graph = seconds_in_cycle = None
        
        elif args.traffic_control == "fixed":
            print("Using Fixed-time traffic control - no additional setup needed.")
            # Set variables to None for fixed control
            tree_data = run_config = network_data = graph = seconds_in_cycle = None

        # Initialize the TraCI controller
        controller = SumoController(
            sumo_cfg=sumo_cfg_path,
            step_length=args.step_length,
            end_time=args.end_time,
            gui=args.gui,
            time_dependent=args.time_dependent,
            start_time_hour=args.start_time_hour,
            routing_strategy=args.routing_strategy
        )
        print("Initialized TraCI controller successfully.")


        # Per‑step TraCI controller  ✅
        def control_callback(current_time: int):
            """Apply selected traffic control method each simulation step."""
            
            if args.traffic_control == "tree_method":
                # Nimrod's Tree Method
                # 1) Update domain model and compute new Boolean decisions
                graph.update_traffic_lights(current_time,
                                            seconds_in_cycle,
                                            run_config.algo_type)

                # 2) Translate to SUMO colour strings (Graph does this for us)
                # Ariel: adding a guard here to ensure we have valid phase_map
                phase_map = graph.get_traffic_lights_phases(int(current_time))
                if not phase_map:   # guard for None / empty dict
                    return

                # Runtime verification of algorithm behavior
                try:
                    verify_algorithm_runtime_behavior(
                        current_time,
                        phase_map,
                        graph,
                        CONFIG.SIMULATION_VERIFICATION_FREQUENCY
                    )
                except ValidationError as ve:
                    print(
                        f"Algorithm runtime validation failed at step {current_time}: {ve}")
                    traci.close()
                    exit(1)

                # 3) Push every TLS state to TraCI
                for tls_id, state in phase_map.items():
                    traci.trafficlight.setRedYellowGreenState(tls_id, state)
            
            elif args.traffic_control == "actuated":
                # SUMO Actuated control - let SUMO handle traffic lights automatically
                # No explicit control needed, SUMO's actuated logic will manage phases
                pass
            
            elif args.traffic_control == "fixed":
                # Fixed-time control - let SUMO use the static timings from grid.tll.xml
                # No explicit control needed, SUMO will use the predefined phase durations
                pass

        # Run the simulation
        controller.run(control_callback)
        print("Simulation completed successfully.")
        
        # Print metrics for experiment analysis
        print("\n=== EXPERIMENT METRICS ===")
        print(f"Total vehicles: {args.num_vehicles}")
        print(f"Simulation time: {args.end_time}")
        
        # Get traffic statistics from controller
        if hasattr(controller, 'final_metrics'):
            metrics = controller.final_metrics
            print(f"Vehicles reached destination: {metrics['arrived_vehicles']}")
            print(f"Vehicles departed: {metrics['departed_vehicles']}")
            print(f"Completion rate: {metrics['completion_rate']:.3f}")
            print(f"Average travel time: {metrics['mean_travel_time']:.2f}")
        else:
            print("No metrics available")
        
        print("=== END METRICS ===\n")

    except Exception as e:
        import traceback
        print(f"An error occurred. Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
