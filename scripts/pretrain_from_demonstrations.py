"""
Behavioral Cloning Pre-training from Tree Method Demonstrations.

This script trains an RL policy to imitate Tree Method using supervised learning
on collected (state, action) demonstrations. The pre-trained model can then be
fine-tuned with RL for further improvement.

Usage:
    python scripts/pretrain_from_demonstrations.py --input data/demonstrations/tree_method_demos.npz
    python scripts/pretrain_from_demonstrations.py --input demos.npz --output custom_pretrained.zip
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.data import Dataset, DataLoader, random_split

from src.rl.constants import (
    PRETRAINING_LEARNING_RATE,
    PRETRAINING_BATCH_SIZE,
    PRETRAINING_EPOCHS,
    PRETRAINING_VALIDATION_SPLIT,
    PRETRAINING_VERBOSE,
    RL_USE_CONTINUOUS_ACTIONS,
    STATE_NORMALIZATION_MIN,
    STATE_NORMALIZATION_MAX
)


class DemonstrationDataset(Dataset):
    """PyTorch dataset for Tree Method demonstrations."""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        """
        Initialize demonstration dataset.

        Args:
            states: State observations (N, state_dim)
            actions: Expert actions
                - If RL_USE_CONTINUOUS_ACTIONS: (N, num_junctions * 4) float logits
                - Otherwise: (N, num_junctions) int phase indices
        """
        self.states = torch.FloatTensor(states)
        # Use FloatTensor for continuous actions, LongTensor for discrete
        if RL_USE_CONTINUOUS_ACTIONS:
            self.actions = torch.FloatTensor(actions)
        else:
            self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def load_demonstrations(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load demonstrations from file.

    Args:
        file_path: Path to demonstrations .npz file

    Returns:
        Tuple of (states, actions, metadata)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading demonstrations from: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    states = data['states']
    actions = data['actions']
    metadata = data['metadata'].item() if 'metadata' in data else {}

    logger.info(f"Loaded {len(states)} demonstrations")
    logger.info(f"State shape: {states.shape}")
    logger.info(f"Action shape: {actions.shape}")
    if metadata:
        logger.info(f"Metadata: {metadata}")

    return states, actions, metadata


def pretrain_policy(
    demonstrations_file: str,
    output_model_path: str,
    learning_rate: float = PRETRAINING_LEARNING_RATE,
    batch_size: int = PRETRAINING_BATCH_SIZE,
    epochs: int = PRETRAINING_EPOCHS,
    validation_split: float = PRETRAINING_VALIDATION_SPLIT
) -> str:
    """
    Pre-train PPO policy using behavioral cloning on Tree Method demonstrations.

    Args:
        demonstrations_file: Path to demonstrations .npz file
        output_model_path: Path to save pre-trained model
        learning_rate: Learning rate for supervised learning
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data for validation

    Returns:
        Path to saved pre-trained model
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("BEHAVIORAL CLONING PRE-TRAINING")
    logger.info("=" * 80)

    # Load demonstrations
    states, actions, metadata = load_demonstrations(demonstrations_file)

    # Create dataset
    full_dataset = DemonstrationDataset(states, actions)

    # Split into train/validation
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Training set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create a dummy PPO model to get policy architecture
    # We'll extract and train just the policy network
    logger.info("Initializing PPO policy architecture...")

    # Determine state and action dimensions from data
    state_dim = states.shape[1]

    if RL_USE_CONTINUOUS_ACTIONS:
        # Continuous actions: shape is (N, num_junctions * 4)
        # where 4 is the number of phases
        action_dim = actions.shape[1]
        num_junctions = action_dim // 4  # Assuming 4 phases per junction
        actions_per_junction = 4
        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Total action dimension: {action_dim}")
        logger.info(f"Number of junctions: {num_junctions}")
        logger.info(f"Actions per junction (phases): {actions_per_junction}")
        logger.info("Using CONTINUOUS action space for duration-based control")
    else:
        # Discrete actions: shape is (N, num_junctions)
        num_junctions = actions.shape[1]
        actions_per_junction = int(np.max(actions) + 1)  # Assume actions are 0-indexed
        logger.info(f"State dimension: {state_dim}")
        logger.info(f"Number of junctions: {num_junctions}")
        logger.info(f"Actions per junction: {actions_per_junction}")
        logger.info("Using DISCRETE action space for phase-only control")

    # Create dummy environment for PPO initialization
    # We need this to get the correct policy architecture
    from gymnasium import spaces
    import gymnasium as gym

    class DummyEnv(gym.Env):
        """Minimal dummy environment for PPO initialization."""
        def __init__(self, state_dim, action_space_config):
            super().__init__()
            # Match RL environment observation space exactly (including floating point tolerance)
            self.observation_space = spaces.Box(
                low=STATE_NORMALIZATION_MIN,
                high=STATE_NORMALIZATION_MAX + 0.01,  # Add tolerance for floating point precision
                shape=(state_dim,),
                dtype=np.float32
            )
            self.action_space = action_space_config

        def reset(self, seed=None, options=None):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, False, False, {}
            )

    # Create appropriate action space
    if RL_USE_CONTINUOUS_ACTIONS:
        # Use finite bounds for PPO compatibility
        # Actions are logits that get normalized via softmax
        # Range of -10 to +10 covers practical spectrum (exp(-10) ≈ 0%, exp(+10) ≈ 100%)
        action_space_config = spaces.Box(
            low=-10.0, high=10.0,
            shape=(num_junctions * actions_per_junction,),
            dtype=np.float32
        )
    else:
        action_space_config = spaces.MultiDiscrete([actions_per_junction] * num_junctions)

    dummy_env = DummyEnv(state_dim, action_space_config)
    # CRITICAL: Use [256, 256] architecture to match RL training (src/rl/constants.py TRAINING_NETWORK_ARCHITECTURE)
    # This ensures pretrained weights can properly transfer to RL training
    model = PPO("MlpPolicy", dummy_env, policy_kwargs={'net_arch': [256, 256]}, verbose=0)

    # Extract policy network
    policy = model.policy
    policy.train()

    # Setup optimizer
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Loss function based on action type
    if RL_USE_CONTINUOUS_ACTIONS:
        criterion = nn.MSELoss()  # Regression for continuous actions
        logger.info("Using MSE loss for continuous action prediction")
    else:
        criterion = nn.CrossEntropyLoss()  # Classification for discrete actions
        logger.info("Using CrossEntropy loss for discrete action prediction")

    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(epochs):
        # Training phase
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()

            # Get policy predictions
            # PPO policy outputs action distribution
            features = policy.extract_features(batch_states)
            action_logits = policy.action_net(policy.mlp_extractor.forward_actor(features))

            if RL_USE_CONTINUOUS_ACTIONS:
                # Continuous actions: direct MSE loss on all outputs
                total_loss = criterion(action_logits, batch_actions)

                # For continuous actions, "accuracy" is measured as percentage within threshold
                with torch.no_grad():
                    threshold = 0.5  # Consider prediction correct if within 0.5 of target
                    correct_mask = torch.abs(action_logits - batch_actions) < threshold
                    train_correct += correct_mask.sum().item()
                    train_total += batch_actions.numel()
            else:
                # Discrete actions: compute loss for each junction
                total_loss = 0.0
                for junction_idx in range(num_junctions):
                    junction_logits = action_logits[:, junction_idx * actions_per_junction:(junction_idx + 1) * actions_per_junction]
                    junction_targets = batch_actions[:, junction_idx]
                    total_loss += criterion(junction_logits, junction_targets)

                    # Accuracy
                    predictions = torch.argmax(junction_logits, dim=1)
                    train_correct += (predictions == junction_targets).sum().item()
                    train_total += len(predictions)

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                features = policy.extract_features(batch_states)
                action_logits = policy.action_net(policy.mlp_extractor.forward_actor(features))

                if RL_USE_CONTINUOUS_ACTIONS:
                    # Continuous actions: direct MSE loss
                    total_loss = criterion(action_logits, batch_actions)

                    # Accuracy threshold metric
                    threshold = 0.5
                    correct_mask = torch.abs(action_logits - batch_actions) < threshold
                    val_correct += correct_mask.sum().item()
                    val_total += batch_actions.numel()
                else:
                    # Discrete actions: per-junction loss
                    total_loss = 0.0
                    for junction_idx in range(num_junctions):
                        junction_logits = action_logits[:, junction_idx * actions_per_junction:(junction_idx + 1) * actions_per_junction]
                        junction_targets = batch_actions[:, junction_idx]
                        total_loss += criterion(junction_logits, junction_targets)

                        predictions = torch.argmax(junction_logits, dim=1)
                        val_correct += (predictions == junction_targets).sum().item()
                        val_total += len(predictions)

                val_loss += total_loss.item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

        # Logging
        if PRETRAINING_VERBOSE:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    logger.info("=" * 80)

    # Save pre-trained model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save(output_model_path)
    logger.info(f"✓ Saved pre-trained model to: {output_model_path}")

    # Save training history
    history_path = output_model_path.replace('.zip', '_history.npz')
    np.savez(
        history_path,
        history=training_history,
        metadata={
            'demonstrations_file': demonstrations_file,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'training_date': datetime.now().isoformat()
        }
    )
    logger.info(f"✓ Saved training history to: {history_path}")

    return output_model_path


def main():
    """Main entry point for behavioral cloning pre-training."""
    parser = argparse.ArgumentParser(
        description="Pre-train RL policy from Tree Method demonstrations"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input demonstrations file (.npz format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pre-trained model path (.zip format). If not specified, creates models/pretrained/rl_pretrained_TIMESTAMP/'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=PRETRAINING_LEARNING_RATE,
        help=f'Learning rate (default: {PRETRAINING_LEARNING_RATE})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=PRETRAINING_BATCH_SIZE,
        help=f'Batch size (default: {PRETRAINING_BATCH_SIZE})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=PRETRAINING_EPOCHS,
        help=f'Number of epochs (default: {PRETRAINING_EPOCHS})'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get paths from arguments
    input_file = args.input

    # Auto-generate output path in new timestamped subdirectory under models/pretrained/
    if args.output is None:
        from datetime import datetime
        # Create new timestamped subdirectory under models/pretrained/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("models/pretrained", f"rl_pretrained_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"rl_pretrained_{timestamp}.zip")
        print(f"Auto-generated output directory: {output_dir}")
        print(f"Output model path: {output_file}")
    else:
        output_file = args.output

    # Pre-train
    pretrain_policy(
        demonstrations_file=input_file,
        output_model_path=output_file,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
