"""
Collect Tree Method Demonstrations for Imitation Learning.

This script runs Tree Method simulations with parameter variation and collects
(state, action) pairs for behavioral cloning pre-training. It operates in read-only
mode and does not modify Tree Method's code or behavior.

Uses a JSON configuration file to specify:
- Fixed parameters (same across all scenarios)
- Varying parameters (randomly sampled per scenario for diversity)

Usage:
    # Using default configuration
    python scripts/collect_tree_method_demonstrations.py --scenarios 500 --base-seed 42

    # Using custom configuration
    python scripts/collect_tree_method_demonstrations.py --scenarios 500 --config configs/my_config.json

    # Custom output path
    python scripts/collect_tree_method_demonstrations.py --scenarios 100 --output models/my_demos.npz
"""

# IMPORTANT: Do NOT import TrafficControllerFactory or TreeMethodDemonstrationAdapter at module level!
# They must be imported INSIDE _collect_single_scenario so they get fresh instances after module reload
from src.rl.environment import TrafficControlEnv
from src.rl.constants import (
    DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS,
    DEMONSTRATION_DEFAULT_CONFIG_FILE,
    DEMONSTRATION_DECISION_INTERVAL_SECONDS,
    DEFAULT_CYCLE_LENGTH
)
import os
import sys
import argparse
import logging
import numpy as np
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_variation_config(config_path: str) -> Dict:
    """
    Load variation configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dict with 'fixed_params' and 'varying_params' keys
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading variation config from: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate config structure
    if 'fixed_params' not in config:
        raise ValueError("Config must contain 'fixed_params' key")
    if 'varying_params' not in config:
        raise ValueError("Config must contain 'varying_params' key")

    logger.info(f"Fixed parameters: {list(config['fixed_params'].keys())}")
    logger.info(f"Varying parameters: {list(config['varying_params'].keys())}")

    return config


def sample_varying_params(varying_params: Dict) -> Dict:
    """
    Randomly sample one value from each varying parameter list.

    Args:
        varying_params: Dict mapping parameter name to list of possible values

    Returns:
        Dict mapping parameter name to sampled value
    """
    return {key: random.choice(values) for key, values in varying_params.items()}


def collect_demonstrations(
    num_scenarios: int,
    base_seed: int,
    config: Dict,
    output_file: str,
    cycle_lengths: List[int],
    cycle_strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations by running Tree Method simulations with parameter variation.

    Args:
        num_scenarios: Number of scenarios to collect (different seeds and parameter combinations)
        base_seed: Base random seed
        config: Configuration dict with 'fixed_params' and 'varying_params'
        output_file: Path to save demonstrations
        cycle_lengths: List of cycle lengths (e.g., [90] for fixed, [60, 90, 120] for variable)
        cycle_strategy: Strategy for cycle selection ('fixed', 'random', 'sequential')

    Returns:
        Tuple of (states, actions) numpy arrays
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TREE METHOD DEMONSTRATION COLLECTION")
    logger.info("=" * 80)
    logger.info(f"Scenarios to collect: {num_scenarios}")
    logger.info(f"Base seed: {base_seed}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Cycle lengths: {cycle_lengths}")
    logger.info(f"Cycle strategy: {cycle_strategy}")
    logger.info(f"Fixed parameters: {config['fixed_params']}")
    logger.info(
        f"Varying parameter options: {list(config['varying_params'].keys())}")
    logger.info("=" * 80)

    all_states = []
    all_actions = []
    per_scenario_params = []  # Track parameters used for each scenario

    # Extract fixed and varying params
    fixed_params = config['fixed_params']
    varying_params = config['varying_params']

    # Estimate time (use average end-time from varying params if available)
    avg_end_time = sum(varying_params.get(
        'end-time', [3600])) / len(varying_params.get('end-time', [3600]))
    estimated_time_per_scenario = avg_end_time * \
        1.5  # Rough estimate (simulation + overhead)
    estimated_total_hours = (
        num_scenarios * estimated_time_per_scenario) / 3600
    logger.info(
        f"Estimated collection time: {estimated_total_hours:.1f} hours")
    logger.info("")

    start_time = datetime.now()

    # Clean up any previous demonstration_temp folder at START
    demo_temp_dir = "demonstration_temp"
    if os.path.exists(demo_temp_dir):
        try:
            shutil.rmtree(demo_temp_dir)
            logger.info(f"✓ Cleaned up previous demonstration_temp/")
        except Exception as e:
            logger.warning(
                f"Could not clean up previous {demo_temp_dir}/: {e}")

    for scenario_idx in range(num_scenarios):
        scenario_seed = base_seed + scenario_idx

        # Sample varying parameters for this scenario
        sampled_varying_params = sample_varying_params(varying_params)

        # Select cycle length for this scenario based on strategy
        if cycle_strategy == 'random':
            selected_cycle_length = random.choice(cycle_lengths)
        elif cycle_strategy == 'sequential':
            selected_cycle_length = cycle_lengths[scenario_idx % len(
                cycle_lengths)]
        else:  # 'fixed'
            selected_cycle_length = cycle_lengths[0]

        logger.info(
            f"--- Scenario {scenario_idx + 1}/{num_scenarios} (seed={scenario_seed}) ---")
        logger.info(f"Sampled parameters: {sampled_varying_params}")
        logger.info(
            f"Cycle length for this scenario: {selected_cycle_length}s")

        try:
            # Collect single scenario
            states, actions = _collect_single_scenario(
                scenario_idx=scenario_idx,
                base_seed=scenario_seed,
                fixed_params=fixed_params,
                varying_params=sampled_varying_params,
                cycle_length=selected_cycle_length,
                cycle_lengths=cycle_lengths,
                cycle_strategy=cycle_strategy
            )

            if states is not None and actions is not None:
                all_states.append(states)
                all_actions.append(actions)
                per_scenario_params.append({
                    'scenario_idx': scenario_idx,
                    'seed': scenario_seed,
                    'sampled_params': sampled_varying_params
                })
                logger.info(f"✓ Collected {len(states)} state-action pairs")
            else:
                logger.warning(
                    f"✗ Scenario {scenario_idx + 1} failed to collect data")

            # Progress reporting
            if (scenario_idx + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                avg_time_per_scenario = elapsed / (scenario_idx + 1)
                remaining_scenarios = num_scenarios - (scenario_idx + 1)
                eta_hours = remaining_scenarios * avg_time_per_scenario

                logger.info("")
                logger.info(
                    f"Progress: {scenario_idx + 1}/{num_scenarios} ({(scenario_idx + 1)/num_scenarios*100:.1f}%)")
                logger.info(f"Elapsed: {elapsed:.2f}h, ETA: {eta_hours:.2f}h")
                logger.info(
                    f"Total demonstrations so far: {sum(len(s) for s in all_states)}")
                logger.info("")

        except Exception as e:
            logger.error(f"✗ Error in scenario {scenario_idx + 1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Concatenate all demonstrations
    if all_states:
        final_states = np.concatenate(all_states, axis=0)
        final_actions = np.concatenate(all_actions, axis=0)

        logger.info("=" * 80)
        logger.info("COLLECTION COMPLETE")
        logger.info(
            f"Total scenarios collected: {len(all_states)}/{num_scenarios}")
        logger.info(f"Total demonstrations: {len(final_states)}")
        logger.info(f"State shape: {final_states.shape}")
        logger.info(f"Action shape: {final_actions.shape}")

        # Save demonstrations
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(
            output_file,
            states=final_states,
            actions=final_actions,
            metadata={
                'num_scenarios': len(all_states),
                'base_seed': base_seed,
                'fixed_params': fixed_params,
                'varying_params_options': varying_params,
                'per_scenario_params': per_scenario_params,
                'collection_date': datetime.now().isoformat()
            }
        )
        logger.info(f"✓ Saved to: {output_file}")

        # Clean up temporary demonstration workspace
        # DISABLED FOR DEBUGGING - Check if route files differ
        demo_temp_dir = "demonstration_temp"
        if os.path.exists(demo_temp_dir):
            logger.info(
                f"⚠️  KEEPING demonstration_temp for debugging - check route files!")
            # try:
            #     shutil.rmtree(demo_temp_dir)
            #     logger.info(
            #         f"✓ Cleaned up temporary workspace: {demo_temp_dir}/")
            # except Exception as e:
            #     logger.warning(f"Could not clean up {demo_temp_dir}/: {e}")

        # Move CSV reward analysis files to output directory
        output_dir = os.path.dirname(output_file)
        csv_pattern = "reward_analysis_episode_*.csv"
        import glob
        csv_files = glob.glob(csv_pattern)
        if csv_files:
            moved_count = 0
            for csv_file in csv_files:
                try:
                    dest_path = os.path.join(
                        output_dir, os.path.basename(csv_file))
                    shutil.move(csv_file, dest_path)
                    moved_count += 1
                except Exception as e:
                    logger.warning(f"Could not move {csv_file}: {e}")
            logger.info(f"✓ Moved {moved_count} CSV files to: {output_dir}/")

        logger.info("=" * 80)

        return final_states, final_actions
    else:
        logger.error("No demonstrations were collected!")
        return None, None


def _collect_single_scenario(
    scenario_idx: int,
    base_seed: int,
    fixed_params: Dict,
    varying_params: Dict,
    cycle_length: int,
    cycle_lengths: List[int],
    cycle_strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations from a single Tree Method simulation.

    Args:
        scenario_idx: Scenario index (for logging)
        base_seed: Base random seed for this scenario
        fixed_params: Dict of fixed parameters (same across all scenarios)
        varying_params: Dict of sampled varying parameters (specific to this scenario)
        cycle_length: Cycle length to use for this scenario's tree-method-interval
        cycle_lengths: List of all cycle lengths (passed to RL environment)
        cycle_strategy: Cycle selection strategy (passed to RL environment)

    Returns:
        Tuple of (states, actions) for this scenario
    """
    logger = logging.getLogger(__name__)

    # Build environment parameters by merging fixed and varying params
    env_params_parts = []

    # Add fixed parameters
    for key, value in fixed_params.items():
        env_params_parts.append(f"--{key} {value}")

    # Add varying parameters (sampled for this scenario)
    for key, value in varying_params.items():
        # Handle values that may contain spaces (wrap in quotes if needed)
        if isinstance(value, str) and ' ' in value:
            env_params_parts.append(f'--{key} "{value}"')
        else:
            env_params_parts.append(f"--{key} {value}")

    # Add tree-method-interval for this scenario (controls Tree Method's decision interval)
    env_params_parts.append(f"--tree-method-interval {cycle_length}")

    # Add traffic seeds (fixed starting values, incremented per scenario)
    env_params_parts.append(
        f"--private-traffic-seed {72632 + scenario_idx * 4535}")
    env_params_parts.append(
        f"--public-traffic-seed {27031 + scenario_idx * 4535}")

    # Add workspace
    env_params_parts.append(
        f"--workspace demonstration_temp/scenario_{scenario_idx}")

    # Join all parts
    env_params = " ".join(env_params_parts)

    # Create RL environment (for state observation)
    # Pass cycle parameters to RL environment
    rl_env = TrafficControlEnv(
        env_params_string=env_params,
        episode_number=scenario_idx,
        cycle_lengths=cycle_lengths,
        cycle_strategy=cycle_strategy
    )

    states = []
    actions = []

    try:
        # Reset environment (this starts SUMO and populates junction_ids)
        # CRITICAL: Pass scenario seed to prevent all episodes from using same random seed
        initial_state, _ = rl_env.reset(seed=base_seed)

        # NOW we can get junction IDs (after SUMO has started)
        junction_ids = rl_env.junction_ids

        # Create Tree Method controller (this is the expert we're learning from)
        # Note: We create it separately to have direct access for demonstration collection

        # FIX: Update CONFIG with the correct workspace path BEFORE importing Tree Method
        # The RL environment generates files in demonstration_temp/scenario_X/workspace/
        # but Tree Method will look in the default workspace/ unless we update CONFIG
        # CRITICAL: Must update CONFIG before importing Tree Method controller module
        from src.config import CONFIG
        CONFIG.update_workspace(f"demonstration_temp/scenario_{scenario_idx}")

        # CRITICAL: Import AFTER module deletion in previous episode's cleanup
        # This ensures we get fresh class definitions, not cached ones
        from src.orchestration.traffic_controller import TrafficControllerFactory
        from src.rl.demonstration_collector import TreeMethodDemonstrationAdapter
        from src.args.parser import create_argument_parser
        import shlex

        # Parse env_params string into args object for Tree Method controller
        parser = create_argument_parser()
        args_list = shlex.split(env_params)
        args = parser.parse_args(args_list)

        # Set traffic control to tree_method
        args.traffic_control = 'tree_method'

        # Create Tree Method controller using factory
        tree_method_controller = TrafficControllerFactory.create(
            'tree_method', args)

        # Initialize Tree Method (loads network, builds graph, etc.)
        tree_method_controller.initialize()

        # Create demonstration adapter (read-only observer)
        demo_adapter = TreeMethodDemonstrationAdapter(
            tree_method_controller, junction_ids)

        step_count = 0
        done = False
        import traci

        # DEBUG: Log first few vehicle IDs to verify traffic variation
        logged_vehicles = False

        while not done:
            # Step SUMO once per iteration (like normal CLI simulation)
            traci.simulationStep()

            # DEBUG: Log vehicle IDs at step 100 to verify traffic variation
            if rl_env.current_step == 100 and not logged_vehicles:
                vehicle_ids = traci.vehicle.getIDList()
                logger.info(
                    f"🚗 VEHICLE DEBUG Episode {scenario_idx} Step 100: {len(vehicle_ids)} vehicles")
                logger.info(f"   First 5 IDs: {list(vehicle_ids)[:5]}")
                logged_vehicles = True

            # Call Tree Method at EVERY step (it internally decides when to calculate)
            # This allows Tree Method to collect vehicle count data continuously
            tree_method_controller.update(rl_env.current_step)

            # Extract action only when Tree Method makes a decision (every cycle_length)
            # This matches when Tree Method actually runs its calculations
            if rl_env.current_step > 0 and rl_env.current_step % cycle_length == 0:
                tree_method_action = demo_adapter.extract_rl_action()

                if tree_method_action is not None:
                    # Get current RL state AFTER Tree Method's decision
                    current_state = rl_env._get_observation()

                    # Store demonstration pair (state, Tree Method's action)
                    states.append(current_state)
                    actions.append(tree_method_action)

            # Increment step counter
            rl_env.current_step += 1
            step_count += 1

            # Log reward data every 100 steps (for CSV analysis)
            if rl_env.current_step % 100 == 0 and hasattr(rl_env, '_compute_reward'):
                try:
                    # Compute reward to populate CSV (don't actually use it)
                    _ = rl_env._compute_reward()
                except Exception as e:
                    logger.warning(
                        f"Failed to log reward data at step {rl_env.current_step}: {e}")

            # Check if simulation ended
            min_expected_vehicles = traci.simulation.getMinExpectedNumber()
            done = min_expected_vehicles == 0 or rl_env.current_step >= rl_env.end_time

            # Periodic logging
            if step_count % 100 == 0:
                logger.debug(
                    f"  Step {step_count}, collected {len(states)} pairs")

    except Exception as e:
        logger.error(f"Error during scenario collection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

    finally:
        # Cleanup RL environment
        rl_env.close()

        # Explicitly cleanup Tree Method controller to prevent state leakage
        if 'tree_method_controller' in locals():
            try:
                # Close any TraCI connections
                if hasattr(tree_method_controller, 'graph'):
                    del tree_method_controller.graph
                if hasattr(tree_method_controller, 'network_data'):
                    del tree_method_controller.network_data
                if hasattr(tree_method_controller, 'tree_data'):
                    del tree_method_controller.tree_data
                if hasattr(tree_method_controller, 'run_config'):
                    del tree_method_controller.run_config
                del tree_method_controller
            except:
                pass

        # CRITICAL FIX: Delete the JSON file to force clean reload next episode
        # The Network class caches data from JSON, and Phase objects modify it in-place
        # Deleting and regenerating ensures complete independence between episodes
        from src.config import CONFIG
        from pathlib import Path
        import os
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        if os.path.exists(json_file):
            try:
                os.remove(json_file)
                logger.debug(
                    f"🗑️  Deleted JSON file for clean reload: {json_file}")
            except Exception as e:
                logger.warning(f"Could not delete JSON file {json_file}: {e}")

        # Clear any module-level caches
        # CRITICAL: Must reload ALL Tree Method modules including controller and integration
        import sys
        modules_to_reload = [
            # Orchestration layer
            'src.orchestration.traffic_controller',
            # Tree Method controller and integration
            'src.traffic_control.decentralized_traffic_bottlenecks.tree_method.controller',
            'src.traffic_control.decentralized_traffic_bottlenecks.tree_method.integration',
            # Shared classes (data structures)
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.graph',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.node',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.phase',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.network',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.link',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.net_data_builder',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.iterations_trees',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.current_load_tree',
            # Shared utilities and config
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.utils',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.config',
            'src.traffic_control.decentralized_traffic_bottlenecks.shared.enums',
        ]
        for mod in modules_to_reload:
            if mod in sys.modules:
                del sys.modules[mod]

    if states:
        return np.array(states), np.array(actions)
    else:
        return None, None


def main():
    """Main entry point for demonstration collection."""
    parser = argparse.ArgumentParser(
        description="Collect Tree Method demonstrations with parameter variation for imitation learning"
    )
    parser.add_argument(
        '--scenarios',
        type=int,
        default=DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS,
        help=f'Number of scenarios to collect (default: {DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS})'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=f'Path to variation config JSON file (default: {DEMONSTRATION_DEFAULT_CONFIG_FILE})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: models/demonstrations/rl_demonstration_TIMESTAMP/demo_TIMESTAMP.npz)'
    )
    parser.add_argument(
        '--cycle-lengths',
        type=int,
        nargs='+',
        default=[DEFAULT_CYCLE_LENGTH],
        help=f'List of cycle lengths in seconds (default: [{DEFAULT_CYCLE_LENGTH}]). Example: --cycle-lengths 60 90 120'
    )
    parser.add_argument(
        '--cycle-strategy',
        type=str,
        choices=['fixed', 'random', 'sequential'],
        default='fixed',
        help='Strategy for selecting cycle length per scenario (default: fixed)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config_path = args.config if args.config else DEMONSTRATION_DEFAULT_CONFIG_FILE
    config = load_variation_config(config_path)

    # Generate timestamped output path in organized subdirectory structure
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create subdirectory under models/demonstrations/ for this demonstration run
        output_dir = os.path.join(
            "models/demonstrations", f"rl_demonstration_{timestamp}")
        output_file = os.path.join(output_dir, f"demo_{timestamp}.npz")
    else:
        output_file = args.output
        output_dir = os.path.dirname(output_file)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect demonstrations
    print("DEBUG : Starting demonstration collection...")
    collect_demonstrations(
        num_scenarios=args.scenarios,
        base_seed=args.base_seed,
        config=config,
        output_file=output_file,
        cycle_lengths=args.cycle_lengths,
        cycle_strategy=args.cycle_strategy
    )


if __name__ == "__main__":
    main()
