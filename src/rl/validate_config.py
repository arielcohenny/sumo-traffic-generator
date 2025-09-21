"""
Configuration Validation Script for RL Training.

This script validates the RL training configuration and ensures it's
compatible with the existing SUMO pipeline infrastructure.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rl.config import get_rl_config, validate_rl_environment_compatibility
from src.constants import (
    DEFAULT_GRID_DIMENSION, DEFAULT_BLOCK_SIZE_M, DEFAULT_NUM_VEHICLES,
    DEFAULT_END_TIME, MIN_GRID_DIMENSION, MAX_GRID_DIMENSION
)


def validate_rl_configuration():
    """
    Validate RL training configuration and check compatibility.

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("=== RL CONFIGURATION VALIDATION ===")

    try:
        # Get configuration
        config = get_rl_config()
        logger.info(f"Configuration loaded successfully")

        # Display configuration summary
        logger.info("\n" + config.get_summary())

        # Validate individual parameters
        logger.info("\n=== PARAMETER VALIDATION ===")

        # Network validation
        logger.info(f"Grid dimension: {config.grid_dimension} (valid range: {MIN_GRID_DIMENSION}-{MAX_GRID_DIMENSION})")
        if not (MIN_GRID_DIMENSION <= config.grid_dimension <= MAX_GRID_DIMENSION):
            logger.error(f"Grid dimension outside valid range!")
            return False

        # Traffic validation
        logger.info(f"Vehicle count: {config.num_vehicles}")
        logger.info(f"Vehicle density: {config.vehicle_density_per_intersection:.1f} vehicles/intersection")

        if config.vehicle_density_per_intersection < 5:
            logger.warning("Low vehicle density - may not create meaningful congestion")
        elif config.vehicle_density_per_intersection > 30:
            logger.warning("High vehicle density - may cause gridlock")

        # Episode validation
        logger.info(f"Episode duration: {config.episode_duration_minutes:.1f} minutes")
        logger.info(f"Decisions per episode: {config.decisions_per_episode}")

        if config.decisions_per_episode < 10:
            logger.warning("Few decisions per episode - limited learning opportunities")
        elif config.decisions_per_episode > 1000:
            logger.warning("Many decisions per episode - may slow training")

        # Dimension validation
        logger.info(f"\n=== DIMENSION VALIDATION ===")
        logger.info(f"Estimated state vector size: {config.state_vector_size_estimate}")
        logger.info(f"Action vector size: {config.action_vector_size}")

        # Test dimension validation function
        try:
            validate_rl_environment_compatibility(
                config.state_vector_size_estimate,
                config.action_vector_size
            )
            logger.info("Dimension validation passed")
        except ValueError as e:
            logger.error(f"Dimension validation failed: {e}")
            return False

        # CLI compatibility check
        logger.info(f"\n=== CLI COMPATIBILITY CHECK ===")
        cli_args = config.get_cli_args()
        logger.info(f"CLI arguments generated: {len(cli_args)} parameters")

        # Check for required CLI arguments
        required_args = ['grid_dimension', 'num_vehicles', 'end_time', 'traffic_control']
        for arg in required_args:
            if arg not in cli_args:
                logger.error(f"Missing required CLI argument: {arg}")
                return False
            logger.info(f"  {arg}: {cli_args[arg]}")

        # Parallel execution validation
        logger.info(f"\n=== PARALLEL EXECUTION VALIDATION ===")
        logger.info(f"Parallel environments: {config.n_parallel_envs}")

        # Test workspace generation for each environment
        try:
            for env_idx in range(config.n_parallel_envs):
                env_cli_args = config.get_cli_args_for_env(env_idx, "test_workspace")
                expected_workspace = f"test_workspace/env_{env_idx:03d}"
                if env_cli_args['workspace'] != expected_workspace:
                    logger.error(f"Workspace generation failed for env {env_idx}: expected {expected_workspace}, got {env_cli_args['workspace']}")
                    return False
                logger.info(f"  Environment {env_idx}: workspace={env_cli_args['workspace']}, seed={env_cli_args['seed']}")
            logger.info("Parallel workspace generation validated")
        except Exception as e:
            logger.error(f"Parallel execution validation failed: {e}")
            return False

        # Comparison with defaults
        logger.info(f"\n=== COMPARISON WITH SYSTEM DEFAULTS ===")
        logger.info(f"Grid dimension: RL={config.grid_dimension}, System={DEFAULT_GRID_DIMENSION}")
        logger.info(f"Block size: RL={config.block_size_m}, System={DEFAULT_BLOCK_SIZE_M}")
        logger.info(f"Vehicle count: RL={config.num_vehicles}, System={DEFAULT_NUM_VEHICLES}")
        logger.info(f"End time: RL={config.end_time}, System={DEFAULT_END_TIME}")

        logger.info(f"\n✅ RL CONFIGURATION VALIDATION PASSED")
        logger.info(f"Configuration is ready for Phase 2 implementation")

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def display_configuration_details():
    """Display detailed configuration information."""
    config = get_rl_config()

    print("\n" + "="*60)
    print("RL TRAINING CONFIGURATION DETAILS")
    print("="*60)

    print(f"\nNETWORK TOPOLOGY:")
    print(f"  Grid Layout: {config.grid_dimension}×{config.grid_dimension}")
    print(f"  Total Intersections: {config.num_intersections}")
    print(f"  Block Size: {config.block_size_m} meters")
    print(f"  Junctions Removed: {config.junctions_to_remove}")

    print(f"\nTRAFFIC CONFIGURATION:")
    print(f"  Total Vehicles: {config.num_vehicles}")
    print(f"  Vehicle Types: {config.vehicle_types}")
    print(f"  Vehicle Density: {config.vehicle_density_per_intersection:.1f} per intersection")

    print(f"\nEPISODE STRUCTURE:")
    print(f"  Duration: {config.end_time} seconds ({config.episode_duration_minutes:.1f} minutes)")
    print(f"  Decision Interval: {config.decision_interval_seconds} seconds")
    print(f"  Decisions per Episode: {config.decisions_per_episode}")
    print(f"  Measurement Interval: {config.measurement_interval_steps} steps")

    print(f"\nPARALLEL EXECUTION:")
    print(f"  Parallel Environments: {config.n_parallel_envs}")
    print(f"  Workspace Isolation: Enabled (unique workspace per environment)")
    print(f"  Seed Variation: Base seed + environment index for diversity")

    print(f"\nRL DIMENSIONS:")
    print(f"  Estimated State Vector: {config.state_vector_size_estimate} features")
    print(f"  Action Vector: {config.action_vector_size} choices")
    print(f"  Estimated Edges: {config.estimated_num_edges}")

    print(f"\nFIXED VALUES WARNING:")
    print(f"  ⚠️  These values are LOCKED for network-specific training")
    print(f"  ⚠️  Changing them after Phase 2 requires significant code changes")
    print(f"  ⚠️  State/action dimensions determine neural network architecture")

    print("="*60)


if __name__ == "__main__":
    """Run configuration validation."""
    print("RL Training Configuration Validator")

    # Run validation
    success = validate_rl_configuration()

    if success:
        # Display detailed configuration
        display_configuration_details()
        print(f"\n🎉 Phase 1.5 COMPLETED SUCCESSFULLY")
        print(f"Ready to proceed with Phase 2: Core RL Environment")
    else:
        print(f"\n❌ Phase 1.5 VALIDATION FAILED")
        print(f"Fix configuration issues before proceeding to Phase 2")
        sys.exit(1)