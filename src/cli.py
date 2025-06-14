import os
import argparse
import random
import shutil
from alive_progress import *

from src.network.generator import generate_grid_network
from src.network.lanes import set_lane_counts
from src.network.edge_attrs import assign_edge_attractiveness
from src.network.zones import extract_zones_from_junctions
from src.sim.sumo_utils import generate_sumo_conf_file, run_sumo
from src.traffic.builder import generate_vehicle_routes
from src.config import CONFIG

# --- Global Variables and Constants ---
# Clear the output directory if it exists
if os.path.exists(CONFIG.output_dir):
    shutil.rmtree(CONFIG.output_dir)
os.makedirs(CONFIG.output_dir, exist_ok=True)

# --- Command-line Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate and simulate a SUMO orthogonal grid network.")
parser.add_argument("--grid_dimension", type=float, default=5, help="The grid's number of rows and columns. Default is 5 (5x5 grid).")
parser.add_argument("--block_size_m", type=int, default=200, help="Block size in meters. Default is 200m.")
parser.add_argument("--blocks_to_remove", type=int, default=0, help="Number of blocks to remove from the grid. Default is 0")
parser.add_argument("--num_vehicles", type=int, default=CONFIG.DEFAULT_NUM_VEHICLES, help=f"Number of vehicles to generate. Default is {CONFIG.DEFAULT_NUM_VEHICLES}.")
parser.add_argument("--seed", type=int, help="Seed for generating randomness. If not provided, a random seed will be used.")
args = parser.parse_args()

try:
    print("Begin simulating SUMO orthogonal grid network...")

    # --- Global Variables ---
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")

    # --- Step 1: Generate the Orthogonal Grid Network ---
    generate_grid_network(seed, int(args.grid_dimension), int(args.block_size_m), CONFIG.network_file, int(args.blocks_to_remove))
    print(f"Generated network file: {CONFIG.network_file}")

    # --- Step 2: Extract Zones ---
    extract_zones_from_junctions(CONFIG.network_file, args.block_size_m, CONFIG.output_dir, seed, fill_polygons=True, inset=0.0)
    print(f"Extracted zones")

    # --- Step 2: Set Lane Counts ---
    set_lane_counts(net_file_in=CONFIG.network_file, net_file_out=CONFIG.network_file, seed=seed)
    print(f"Set lane counts for edges")

    # --- Step 3: Set depart/arrive attractiveness for each edge ---
    assign_edge_attractiveness(seed, CONFIG.network_file, lambda_depart=3.5, lambda_arrive=2.0)
    print(f"Assigned edge attractiveness")

    # ---- Step 4: Generate vehicle demand (.rou.xml) ----
    generate_vehicle_routes(net_file=CONFIG.network_file, output_file=CONFIG.routes_file, num_vehicles=args.num_vehicles, seed=seed)
    print("Generated vehicle routes.")

    # --- Step : Load the Network and Start SUMO ---
    generate_sumo_conf_file(CONFIG.config_file, CONFIG.network_file, route_file=CONFIG.routes_file)
    run_sumo(CONFIG.config_file, CONFIG.zones_file)

except Exception as e:
    print(f"An error occurred. Error: {e}")
    exit(1)
