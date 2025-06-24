import os
import argparse
import random
import shutil
from alive_progress import *
import traci

from src.sim.sumo_controller import SumoController
from src.sim.sumo_utils import generate_sumo_conf_file

from src.network.generator import *
from src.network.lanes import set_lane_counts
from src.network.edge_attrs import assign_edge_attractiveness
from src.network.zones import extract_zones_from_junctions
from src.traffic.builder import generate_vehicle_routes
from src.config import CONFIG

from src.traffic_control.decentralized_traffic_bottlenecks.integration import load_tree
from src.traffic_control.decentralized_traffic_bottlenecks.classes.graph import Graph
from src.traffic_control.decentralized_traffic_bottlenecks.classes.network import Network
from src.traffic_control.decentralized_traffic_bottlenecks.classes.net_data_builder import build_network_json

from src.validate.errors import ValidationError
from src.validate.validate_network import verify_generate_grid_network, verify_extract_zones_from_junctions, verify_set_lane_counts, verify_assign_edge_attractiveness, verify_inject_traffic_lights
from src.validate.validate_traffic import verify_generate_vehicle_routes


def main():
    # --- Global Variables and Constants ---
    if os.path.exists(CONFIG.output_dir):
        shutil.rmtree(CONFIG.output_dir)
    os.makedirs(CONFIG.output_dir, exist_ok=True)

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
        "--blocks_to_remove",
        type=int,
        default=0,
        help="Number of blocks to remove from the grid. Default is 0."
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
        default=3600,
        help="Total simulation duration in seconds."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO in GUI mode (sumo-gui) instead of headless sumo"
    )
    args = parser.parse_args()

    try:
        print("Begin simulating SUMO orthogonal grid network...")

        # --- Initialize seed ---
        seed = args.seed if args.seed is not None else random.randint(
            0, 2**32 - 1)
        print(f"Using seed: {seed}")

        # --- Step 1: Generate the Orthogonal Grid Network ---
        generate_grid_network(
            seed,
            int(args.grid_dimension),
            int(args.block_size_m),
            CONFIG.network_file,
            int(args.blocks_to_remove)
        )
        try:
            verify_generate_grid_network(
                seed,
                int(args.grid_dimension),
                int(args.block_size_m),
                CONFIG.network_file,
                int(args.blocks_to_remove),
            )
        except ValidationError as ve:
            print(f"Failed generating network file: {ve}")
            exit(1)
        print(f"Generated network file: {CONFIG.network_file} successfully.")

        # --- Step 2: Extract Zones - --
        extract_zones_from_junctions(
            CONFIG.network_file,
            args.block_size_m,
            CONFIG.output_dir,
            seed=seed,
            fill_polygons=True,
            inset=0.0
        )
        try:
            verify_extract_zones_from_junctions(
                CONFIG.network_file,
                CONFIG.output_dir,
            )
        except ValidationError as ve:
            print(f"Zone extraction failed: {ve}")
            exit(1)
        print("Extracted zones from junctions successfully.")

        # --- Step 3: Set Lane Counts ---
        set_lane_counts(
            net_file_in=CONFIG.network_file,
            net_file_out=CONFIG.network_file,
            seed=seed,
            min_lanes=CONFIG.MIN_LANES,
            max_lanes=CONFIG.MAX_LANES
        )
        try:
            verify_set_lane_counts(
                CONFIG.network_file,
                min_lanes=CONFIG.MIN_LANES,
                max_lanes=CONFIG.MAX_LANES,
            )
        except ValidationError as ve:
            print(f"Lane-count validation failed: {ve}")
            exit(1)
        print("Successfully set lane counts for edges.")

        # --- Step 4: Assign Edge Attractiveness ---
        assign_edge_attractiveness(
            seed,
            CONFIG.network_file,
            lambda_depart=CONFIG.LAMBDA_DEPART,
            lambda_arrive=CONFIG.LAMBDA_ARRIVE,
        )
        try:
            verify_assign_edge_attractiveness(
                seed,
                CONFIG.network_file,
                lambda_depart=CONFIG.LAMBDA_DEPART,
                lambda_arrive=CONFIG.LAMBDA_ARRIVE,
            )
        except ValidationError as ve:
            print(f"Edge-attractiveness validation failed: {ve}")
            exit(1)
        print("Successfully assigned edge attractiveness")

        # --- Step 5: Inject Static Traffic Lights ---
        inject_traffic_lights(CONFIG.network_file)
        try:
            verify_inject_traffic_lights(CONFIG.network_file)
        except ValidationError as ve:
            print(f"Traffic-light validation failed: {ve}")
            exit(1)
        print("Successfully injected static traffic lights into network")

        # --- Step 6: Generate Vehicle Routes ---
        generate_vehicle_routes(
            net_file=CONFIG.network_file,
            output_file=CONFIG.routes_file,
            num_vehicles=args.num_vehicles,
            seed=seed
        )
        try:
            verify_generate_vehicle_routes(
                net_file=CONFIG.network_file,
                output_file=CONFIG.routes_file,
                num_vehicles=args.num_vehicles,
                seed=seed,
            )
        except ValidationError as ve:
            print(f"Route validation failed: {ve}")
            exit(1)
        print("Generated vehicle routes.")

        # --- Step 7: Generate SUMO Configuration File ---
        sumo_cfg_path = generate_sumo_conf_file(
            CONFIG.config_file,
            CONFIG.network_file,
            route_file=CONFIG.routes_file,
            zones_file=CONFIG.zones_file,
        )
        print(f"Generated SUMO configuration file: {sumo_cfg_path}")

        # --- Step 8: Dynamic Simulation via TraCI & Nimrod’s Tree Method ---
        # Load network tree structure and runner configuration
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        if json_file.exists():
            json_file.unlink()
        tree_data, run_config = load_tree(
            net_file=CONFIG.network_file,
            sumo_cfg=sumo_cfg_path
        )
        print("Loaded network tree and run configuration.")

        # Initialize the TraCI controller
        controller = SumoController(
            sumo_cfg=sumo_cfg_path,
            step_length=args.step_length,
            end_time=args.end_time,
            gui=args.gui
        )
        print("Initialized TraCI controller.")

        # Build Nimrod’s runtime objects once (outside the callback)
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

        # Per‑step TraCI controller  ✅
        def control_callback(current_time: int):
            """Apply Nimrod's Tree‑Method phase decisions each simulation step."""

            # 1) Update domain model and compute new Boolean decisions
            graph.update_traffic_lights(current_time,
                                        seconds_in_cycle,
                                        run_config.algo_type)

            # 2) Translate to SUMO colour strings (Graph does this for us)
            # Ariel: adding a guard here to ensure we have valid phase_map
            phase_map = graph.get_traffic_lights_phases(int(current_time))
            if not phase_map:   # guard for None / empty dict
                return

            # 3) Push every TLS state to TraCI
            for tls_id, state in phase_map.items():
                traci.trafficlight.setRedYellowGreenState(tls_id, state)

        # Run the simulation
        controller.run(control_callback)
        print("Simulation completed successfully.")

    except Exception as e:
        print(f"An error occurred. Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
