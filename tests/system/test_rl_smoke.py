"""
RL Training Smoke Tests.

Verifies that the core RL training pipeline works end-to-end:
environment creation, observation/action spaces, reward computation,
PPO training, and model save/load.

Uses the smallest viable configuration (3x3 grid, 30 vehicles, 120s)
to keep execution fast.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Skip entire module if SUMO is not available
try:
    import sumolib
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not available"),
]

# Minimal simulation parameters: smallest grid, fewest vehicles, shortest time
MINIMAL_ENV_PARAMS = (
    "--grid_dimension 3 "
    "--block_size_m 100 "
    "--num_vehicles 30 "
    "--end-time 120 "
    "--seed 42 "
    "--lane_count 2 "
    "--routing_strategy 'shortest 100' "
    "--vehicle_types 'passenger 100' "
    "--departure_pattern uniform "
    "--step-length 1.0"
)


@pytest.fixture
def rl_workspace():
    """Isolated temp workspace for RL tests with automatic cleanup."""
    temp_dir = tempfile.mkdtemp(prefix="rl_smoke_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def env(rl_workspace):
    """Create a minimal TrafficControlEnv and clean up after test."""
    from src.rl.environment import TrafficControlEnv

    params = MINIMAL_ENV_PARAMS + f" --workspace {rl_workspace}"
    environment = TrafficControlEnv(
        env_params_string=params,
        cycle_lengths=[90],
        cycle_strategy='fixed',
    )
    yield environment
    environment.close()


class TestEnvironmentLifecycle:
    """Test that the RL environment can be created, reset, stepped, and closed."""

    def test_reset_returns_observation(self, env):
        """Environment reset produces a valid initial observation."""
        obs, info = env.reset()

        assert obs is not None, "reset() returned None observation"
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
        assert obs.shape == env.observation_space.shape, (
            f"Observation shape {obs.shape} != space shape {env.observation_space.shape}"
        )
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"

    def test_step_returns_valid_tuple(self, env):
        """A single step returns (obs, reward, terminated, truncated, info)."""
        env.reset()

        # Take a random action
        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5, f"step() returned {len(result)} values, expected 5"
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray), f"obs type: {type(obs)}"
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float)), f"reward type: {type(reward)}"
        assert np.isfinite(reward), f"Reward is not finite: {reward}"
        assert isinstance(terminated, bool), f"terminated type: {type(terminated)}"
        assert isinstance(truncated, bool), f"truncated type: {type(truncated)}"
        assert isinstance(info, dict), f"info type: {type(info)}"

    def test_multiple_steps(self, env):
        """Environment survives multiple consecutive steps."""
        env.reset()
        rewards = []

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

            assert np.all(np.isfinite(obs)), "Observation became NaN/Inf during episode"

        assert len(rewards) > 0, "No steps completed"

    def test_reset_after_episode(self, env):
        """Environment can reset and start a new episode."""
        # First episode
        obs1, _ = env.reset()

        action = env.action_space.sample()
        env.step(action)

        # Second episode
        obs2, _ = env.reset()

        assert obs2 is not None, "Second reset returned None"
        assert obs2.shape == obs1.shape, "Observation shape changed between episodes"


class TestObservationSpace:
    """Test observation space properties."""

    def test_observation_within_bounds(self, env):
        """Observations stay within the declared space bounds."""
        obs, _ = env.reset()

        low = env.observation_space.low
        high = env.observation_space.high

        violations_low = np.sum(obs < low)
        violations_high = np.sum(obs > high)

        assert violations_low == 0, (
            f"{violations_low} values below lower bound. "
            f"Min obs: {obs.min()}, bound: {low.min()}"
        )
        assert violations_high == 0, (
            f"{violations_high} values above upper bound. "
            f"Max obs: {obs.max()}, bound: {high.max()}"
        )

    def test_observation_not_all_zeros(self, env):
        """After a step, observation should contain non-zero values."""
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        non_zero = np.count_nonzero(obs)
        assert non_zero > 0, "Observation is all zeros after a step"


class TestRewardFunction:
    """Test reward computation."""

    def test_reward_is_finite_across_steps(self, env):
        """Reward remains finite throughout an episode."""
        env.reset()

        for step in range(5):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)

            assert np.isfinite(reward), f"Reward became non-finite at step {step}: {reward}"

            if terminated or truncated:
                break


class TestPPOIntegration:
    """Test that PPO can initialize, train, save, and load with this environment."""

    def test_ppo_training_and_save_load(self, rl_workspace):
        """Full PPO smoke test: init → train → save → load → predict."""
        from stable_baselines3 import PPO
        from src.rl.environment import TrafficControlEnv

        params = MINIMAL_ENV_PARAMS + f" --workspace {rl_workspace}"
        env = TrafficControlEnv(
            env_params_string=params,
            cycle_lengths=[90],
            cycle_strategy='fixed',
        )

        try:
            # Initialize PPO with small buffers for fast test
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=64,
                batch_size=32,
                n_epochs=2,
                learning_rate=3e-4,
                verbose=0,
            )

            # Train for minimal steps (just enough for 1 update: n_steps=64)
            model.learn(total_timesteps=64)

            # Save model
            model_path = Path(rl_workspace) / "test_model"
            model.save(str(model_path))
            assert model_path.with_suffix(".zip").exists(), "Model file was not saved"

            # Close training env before loading model for prediction
            env.close()

            # Load model and predict using observation space directly
            loaded_model = PPO.load(str(model_path))

            # Create a dummy observation matching the saved model's space
            obs = np.zeros(loaded_model.observation_space.shape, dtype=np.float32)
            action, _ = loaded_model.predict(obs, deterministic=True)

            assert action is not None, "Loaded model returned None action"
            assert action.shape == loaded_model.action_space.shape, (
                f"Action shape {action.shape} != space shape {loaded_model.action_space.shape}"
            )
        finally:
            try:
                env.close()
            except Exception:
                pass  # May already be closed
