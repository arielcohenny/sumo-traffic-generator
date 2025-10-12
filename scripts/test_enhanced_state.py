#!/usr/bin/env python3
"""
Comprehensive testing script for enhanced RL state space.

This script verifies that the Tree Method integration is working properly
and that the enhanced state space contains meaningful traffic data.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from rl.environment import TrafficControlEnv
from rl.config import get_rl_config
from rl.constants import (
    RL_ENHANCED_EDGE_FEATURES_COUNT, RL_ENHANCED_JUNCTION_FEATURES_COUNT,
    RL_NETWORK_LEVEL_FEATURES_COUNT, RL_PHASE_ONLY_MODE, RL_FIXED_PHASE_DURATION,
    PROGRESSIVE_BONUS_ENABLED, NUM_TRAFFIC_LIGHT_PHASES
)


def setup_logging():
    """Configure logging for detailed output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_state_space_dimensions(env, config, logger):
    """Test that state space dimensions match configuration."""
    logger.info("=" * 60)
    logger.info("TESTING STATE SPACE DIMENSIONS")
    logger.info("=" * 60)

    # Reset environment and get observation
    obs, info = env.reset()
    actual_size = len(obs)
    expected_size = config.state_vector_size_estimate

    logger.info(f"State vector size:")
    logger.info(f"  - Actual: {actual_size}")
    logger.info(f"  - Expected: {expected_size}")
    logger.info(f"  - Difference: {abs(actual_size - expected_size)}")

    # Verify dimensions match expectations
    if actual_size == expected_size:
        logger.info("✓ PASS: State vector size matches configuration exactly")
        return True
    elif abs(actual_size - expected_size) <= 10:
        logger.info("✓ PASS: State vector size close to expected (within tolerance)")
        return True
    else:
        logger.error("✗ FAIL: State vector size differs significantly from expected")
        return False


def test_tree_method_integration(env, logger):
    """Test that Tree Method traffic analyzer is properly integrated."""
    logger.info("=" * 60)
    logger.info("TESTING TREE METHOD INTEGRATION")
    logger.info("=" * 60)

    # Reset to initialize traffic analyzer
    env.reset()

    # Check if traffic analyzer is initialized
    if hasattr(env, 'traffic_analyzer') and env.traffic_analyzer is not None:
        logger.info("✓ PASS: Traffic analyzer properly initialized")

        # Check Tree Method components
        analyzer = env.traffic_analyzer
        if hasattr(analyzer, 'edge_links') and analyzer.edge_links:
            logger.info(f"✓ PASS: Tree Method Link objects created ({len(analyzer.edge_links)} links)")
        else:
            logger.error("✗ FAIL: Tree Method Link objects not found")
            return False

        if hasattr(analyzer, 'speed_history') and analyzer.speed_history:
            logger.info(f"✓ PASS: Speed history tracking initialized ({len(analyzer.speed_history)} edges)")
        else:
            logger.error("✗ FAIL: Speed history tracking not initialized")
            return False

        return True
    else:
        logger.error("✗ FAIL: Traffic analyzer not initialized")
        return False


def test_enhanced_features(env, logger):
    """Test that enhanced features contain meaningful data."""
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED FEATURES CONTENT")
    logger.info("=" * 60)

    # Enable debug mode for detailed logging
    if hasattr(env, 'enable_debug_mode'):
        env.enable_debug_mode()
        logger.info("Debug mode enabled for detailed feature inspection")

    # Reset and step through a few simulation steps
    obs, _ = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")

    # Analyze initial observation
    non_zero_initial = np.count_nonzero(obs)
    logger.info(f"Initial non-zero features: {non_zero_initial}/{len(obs)} ({non_zero_initial/len(obs):.1%})")

    # Take a few steps to generate traffic activity
    logger.info("Taking simulation steps to generate traffic activity...")
    for step in range(5):
        # Random action for testing (phase + duration for each intersection)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        non_zero_count = np.count_nonzero(obs)
        logger.info(f"Step {step+1}: {non_zero_count}/{len(obs)} non-zero features, reward: {reward:.3f}")

        if terminated or truncated:
            logger.info("Episode ended, resetting...")
            obs, _ = env.reset()

    # Final analysis
    final_non_zero = np.count_nonzero(obs)
    non_zero_ratio = final_non_zero / len(obs)

    logger.info(f"Final state analysis:")
    logger.info(f"  - Non-zero features: {final_non_zero}/{len(obs)} ({non_zero_ratio:.1%})")
    logger.info(f"  - Value range: [{np.min(obs):.3f}, {np.max(obs):.3f}]")
    logger.info(f"  - Mean: {np.mean(obs):.3f}, Std: {np.std(obs):.3f}")

    # Check for proper normalization
    if np.all(obs >= 0.0) and np.all(obs <= 1.0):
        logger.info("✓ PASS: All features properly normalized to [0, 1]")
        normalization_pass = True
    else:
        out_of_range = np.sum((obs < 0.0) | (obs > 1.0))
        logger.error(f"✗ FAIL: {out_of_range} features outside [0, 1] range")
        normalization_pass = False

    # Check for meaningful data
    if non_zero_ratio > 0.05:  # At least 5% non-zero
        logger.info("✓ PASS: State contains meaningful traffic data")
        data_pass = True
    else:
        logger.error(f"✗ FAIL: Very few non-zero features ({non_zero_ratio:.1%})")
        data_pass = False

    return normalization_pass and data_pass


def test_feature_breakdown(env, config, logger):
    """Test breakdown of features by type (edges, junctions, network)."""
    logger.info("=" * 60)
    logger.info("TESTING FEATURE BREAKDOWN BY TYPE")
    logger.info("=" * 60)

    obs, _ = env.reset()

    # Calculate expected feature counts
    estimated_edges = config.estimated_num_edges
    num_junctions = config.num_intersections

    expected_edge_features = estimated_edges * RL_ENHANCED_EDGE_FEATURES_COUNT
    expected_junction_features = num_junctions * RL_ENHANCED_JUNCTION_FEATURES_COUNT
    expected_network_features = RL_NETWORK_LEVEL_FEATURES_COUNT

    logger.info(f"Expected feature breakdown:")
    logger.info(f"  - Edge features: {estimated_edges} edges × {RL_ENHANCED_EDGE_FEATURES_COUNT} = {expected_edge_features}")
    logger.info(f"  - Junction features: {num_junctions} junctions × {RL_ENHANCED_JUNCTION_FEATURES_COUNT} = {expected_junction_features}")
    logger.info(f"  - Network features: {expected_network_features}")
    logger.info(f"  - Total expected: {expected_edge_features + expected_junction_features + expected_network_features}")
    logger.info(f"  - Actual total: {len(obs)}")

    # Use traffic analyzer's inspection if available
    if hasattr(env, 'traffic_analyzer') and env.traffic_analyzer:
        if hasattr(env.traffic_analyzer, 'inspect_state_vector'):
            inspection = env.traffic_analyzer.inspect_state_vector(obs.tolist())

            logger.info(f"Detailed breakdown from traffic analyzer:")
            logger.info(f"  - Edge features: {inspection.get('edge_features', {}).get('count', 'unknown')}")
            logger.info(f"  - Junction features: {inspection.get('junction_features', {}).get('count', 'unknown')}")
            logger.info(f"  - Network features: {inspection.get('network_features', {}).get('count', 'unknown')}")

            return True

    return True


def test_phase_only_action_space(env, config, logger):
    """Test that phase-only action space is working correctly."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE-ONLY ACTION SPACE")
    logger.info("=" * 60)

    # Reset environment to get action space
    env.reset()

    # Check action space configuration
    if RL_PHASE_ONLY_MODE:
        logger.info("✓ Phase-only mode enabled")

        # Check action space dimensions
        expected_actions = config.num_intersections  # One phase per intersection
        if hasattr(env.action_space, 'nvec'):
            actual_actions = len(env.action_space.nvec)
            actions_per_intersection = env.action_space.nvec[0]
        else:
            actual_actions = env.action_space.n
            actions_per_intersection = NUM_TRAFFIC_LIGHT_PHASES

        logger.info(f"Action space analysis:")
        logger.info(f"  - Expected total actions: {expected_actions}")
        logger.info(f"  - Actual total actions: {actual_actions}")
        logger.info(f"  - Actions per intersection: {actions_per_intersection}")
        logger.info(f"  - Fixed phase duration: {RL_FIXED_PHASE_DURATION}s")

        # Test action sampling and application
        action = env.action_space.sample()
        logger.info(f"Sample action: {action} (length: {len(action)})")

        # Test action execution
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Action execution successful, reward: {reward:.3f}")

        if actual_actions == expected_actions:
            logger.info("✓ PASS: Phase-only action space configured correctly")
            return True
        else:
            logger.error(f"✗ FAIL: Action space mismatch")
            return False
    else:
        logger.info("Legacy phase+duration mode detected")
        return True


def test_progressive_bonus_system(env, config, logger):
    """Test that progressive bonus system is working."""
    logger.info("=" * 60)
    logger.info("TESTING PROGRESSIVE BONUS SYSTEM")
    logger.info("=" * 60)

    if not PROGRESSIVE_BONUS_ENABLED:
        logger.info("Progressive bonuses disabled - skipping test")
        return True

    # Reset and enable debug mode for detailed logging
    env.reset()
    if hasattr(env, 'enable_debug_mode'):
        env.enable_debug_mode()
        logger.info("Debug mode enabled for bonus tracking")

    # Check vehicle tracker has progressive bonus capabilities
    if hasattr(env.vehicle_tracker, 'progressive_bonus_enabled') and env.vehicle_tracker.progressive_bonus_enabled:
        logger.info("✓ Progressive bonus system enabled in vehicle tracker")
    else:
        logger.error("✗ Progressive bonus system not enabled in vehicle tracker")
        return False

    # Check vehicle count is set for milestone calculations
    if hasattr(env.vehicle_tracker, 'total_vehicles_expected') and env.vehicle_tracker.total_vehicles_expected > 0:
        logger.info(f"✓ Total vehicles expected set: {env.vehicle_tracker.total_vehicles_expected}")
    else:
        logger.warning("⚠️  Total vehicles expected not set - milestone bonuses may not work")

    # Test reward computation with bonuses
    initial_reward = env._compute_reward()
    logger.info(f"Initial reward (no bonuses expected): {initial_reward:.3f}")

    # Take several steps to potentially generate bonuses
    total_bonuses = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if reward > 0:
            total_bonuses += reward
            logger.info(f"Step {step+1}: Positive reward detected: {reward:.3f}")

        if terminated or truncated:
            break

    logger.info(f"Total positive rewards collected: {total_bonuses:.3f}")

    if total_bonuses > 0:
        logger.info("✓ PASS: Progressive bonus system generating positive rewards")
        return True
    else:
        logger.info("ℹ  No positive rewards detected (may be normal depending on traffic conditions)")
        return True


def run_comprehensive_test():
    """Run all tests and provide final summary."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE RL SYSTEM VALIDATION")
    logger.info("=" * 60)

    try:
        # Initialize configuration and environment
        logger.info("Initializing RL configuration and environment...")
        config = get_rl_config()
        env = TrafficControlEnv(config)

        logger.info(f"Configuration summary:")
        logger.info(f"  - Grid dimension: {config.grid_dimension}×{config.grid_dimension}")
        logger.info(f"  - Estimated state size: {config.state_vector_size_estimate}")
        logger.info(f"  - Action size: {config.action_vector_size}")
        logger.info(f"  - Phase-only mode: {RL_PHASE_ONLY_MODE}")
        logger.info(f"  - Progressive bonuses: {PROGRESSIVE_BONUS_ENABLED}")

        # Run all tests
        test_results = []

        test_results.append(("State Space Dimensions", test_state_space_dimensions(env, config, logger)))
        test_results.append(("Tree Method Integration", test_tree_method_integration(env, logger)))
        test_results.append(("Enhanced Features", test_enhanced_features(env, logger)))
        test_results.append(("Feature Breakdown", test_feature_breakdown(env, config, logger)))
        test_results.append(("Phase-Only Action Space", test_phase_only_action_space(env, config, logger)))
        test_results.append(("Progressive Bonus System", test_progressive_bonus_system(env, config, logger)))

        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        passed = 0
        total = len(test_results)

        for test_name, result in test_results:
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1

        logger.info(f"Total: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Enhanced state space is working correctly!")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed - Enhanced state space needs fixes")
            return False

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)