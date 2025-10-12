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

from src.orchestration.traffic_controller import TrafficControllerFactory
from src.rl.demonstration_collector import TreeMethodDemonstrationAdapter
from src.rl.environment import TrafficControlEnv
from src.rl.constants import (
    DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS,
    DEMONSTRATION_DEFAULT_CONFIG_FILE,
    DEMONSTRATION_DECISION_INTERVAL_SECONDS
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
    output_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations by running Tree Method simulations with parameter variation.

    Args:
        num_scenarios: Number of scenarios to collect (different seeds and parameter combinations)
        base_seed: Base random seed
        config: Configuration dict with 'fixed_params' and 'varying_params'
        output_file: Path to save demonstrations

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

    for scenario_idx in range(num_scenarios):
        scenario_seed = base_seed + scenario_idx

        # Sample varying parameters for this scenario
        sampled_varying_params = sample_varying_params(varying_params)

        logger.info(
            f"--- Scenario {scenario_idx + 1}/{num_scenarios} (seed={scenario_seed}) ---")
        logger.info(f"Sampled parameters: {sampled_varying_params}")

        try:
            # Collect single scenario
            states, actions = _collect_single_scenario(
                scenario_idx=scenario_idx,
                base_seed=scenario_seed,
                fixed_params=fixed_params,
                varying_params=sampled_varying_params
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
        demo_temp_dir = "demonstration_temp"
        if os.path.exists(demo_temp_dir):
            try:
                shutil.rmtree(demo_temp_dir)
                logger.info(
                    f"✓ Cleaned up temporary workspace: {demo_temp_dir}/")
            except Exception as e:
                logger.warning(f"Could not clean up {demo_temp_dir}/: {e}")

        logger.info("=" * 80)

        return final_states, final_actions
    else:
        logger.error("No demonstrations were collected!")
        return None, None


def _collect_single_scenario(
    scenario_idx: int,
    base_seed: int,
    fixed_params: Dict,
    varying_params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations from a single Tree Method simulation.

    Args:
        scenario_idx: Scenario index (for logging)
        base_seed: Base random seed for this scenario
        fixed_params: Dict of fixed parameters (same across all scenarios)
        varying_params: Dict of sampled varying parameters (specific to this scenario)

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

    # Add traffic seeds (derived from base_seed + scenario_idx)
    env_params_parts.append(
        f"--private-traffic-seed {base_seed + scenario_idx + 10000}")
    env_params_parts.append(
        f"--public-traffic-seed {base_seed + scenario_idx + 20000}")

    # Add workspace
    env_params_parts.append(
        f"--workspace demonstration_temp/scenario_{scenario_idx}")

    # Join all parts
    env_params = " ".join(env_params_parts)

    # Create RL environment (for state observation)
    rl_env = TrafficControlEnv(env_params)

    states = []
    actions = []

    try:
        # Reset environment (this starts SUMO and populates junction_ids)
        initial_state, _ = rl_env.reset()

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

        from src.orchestration.traffic_controller import TrafficControllerFactory
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

        while not done:
            # Let Tree Method make decision and apply it via TraCI
            # Tree Method directly changes traffic lights - we don't apply through RL env
            tree_method_controller.update(rl_env.current_step)

            # Extract Tree Method's action (read-only observation of what it did)
            tree_method_action = demo_adapter.extract_rl_action()

            if tree_method_action is not None:
                # Get current RL state BEFORE advancing simulation
                current_state = rl_env._get_observation()

                # Store demonstration pair (state, Tree Method's action)
                states.append(current_state)
                actions.append(tree_method_action)

            # Advance simulation WITHOUT applying actions (Tree Method already did)
            # Manually step SUMO for DECISION_INTERVAL_SECONDS
            import traci
            from src.rl.constants import DECISION_INTERVAL_SECONDS
            for _ in range(DECISION_INTERVAL_SECONDS):
                traci.simulationStep()
                rl_env.current_step += 1

            # Check if simulation ended
            min_expected_vehicles = traci.simulation.getMinExpectedNumber()
            done = min_expected_vehicles == 0 or rl_env.current_step >= rl_env.end_time

            step_count += 1

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
        # Cleanup
        rl_env.close()

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
        help='Output file path (default: models/tree_method_demonstration/demo_TIMESTAMP.npz)'
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

    # Generate timestamped output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("models/tree_method_demonstration", f"demo_{timestamp}.npz")
    else:
        output_file = args.output

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Collect demonstrations
    collect_demonstrations(
        num_scenarios=args.scenarios,
        base_seed=args.base_seed,
        config=config,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
