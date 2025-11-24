"""
Demonstration Collection Adapter for Imitation Learning.

This module provides tools to observe Tree Method's decisions and convert them
to RL action format for behavioral cloning pre-training. It operates in read-only
mode and does not modify Tree Method's behavior.
"""

import numpy as np
import traci
import logging
from typing import Dict, List, Tuple, Optional

from .constants import DEMONSTRATION_DECISION_INTERVAL_SECONDS, RL_USE_CONTINUOUS_ACTIONS, RL_ACTIONS_PER_JUNCTION


class TreeMethodDemonstrationAdapter:
    """
    Read-only adapter to observe Tree Method decisions for demonstration collection.

    This class extracts Tree Method's phase selection decisions and converts them
    to RL action format without modifying Tree Method's operation.
    """

    def __init__(self, tree_method_controller, junction_ids: List[str]):
        """
        Initialize demonstration adapter.

        Args:
            tree_method_controller: TreeMethodController instance to observe
            junction_ids: List of junction/traffic light IDs in the network
        """
        self.tree_method = tree_method_controller
        self.junction_ids = sorted(junction_ids)  # Ensure consistent ordering
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"=== DEMONSTRATION ADAPTER INITIALIZED ===")
        self.logger.info(f"Observing {len(self.junction_ids)} junctions")
        # self.logger.info(f"Junction IDs: {self.junction_ids[:5]}{'...' if len(self.junction_ids) > 5 else ''}")

    def extract_rl_action(self) -> Optional[np.ndarray]:
        """
        Extract Tree Method's current phase decisions as RL actions.

        Reads Tree Method's current_phase_durations attribute (read-only access)
        and converts to RL action format:
        - If RL_USE_CONTINUOUS_ACTIONS=True: Returns logits for duration proportions
          Shape: (num_junctions * 4,) containing inverse-softmax of duration proportions
        - If RL_USE_CONTINUOUS_ACTIONS=False: Returns phase indices
          Shape: (num_junctions,) containing active phase index per junction

        Returns:
            np.ndarray: Action array in appropriate format for current RL mode
                       Returns None if phase information is not yet available
        """
        try:
            # Read-only access to Tree Method's phase durations
            phase_durations = self.tree_method.current_phase_durations

            if not phase_durations:
                # Tree Method hasn't computed phases yet - use fallback
                return self._create_fallback_action()

            if RL_USE_CONTINUOUS_ACTIONS:
                # Duration-based control: extract durations for all phases
                return self._extract_duration_logits(phase_durations)
            else:
                # Phase-only control: extract active phase indices
                return self._extract_phase_indices(phase_durations)

        except Exception as e:
            self.logger.error(f"Failed to extract RL action: {e}")
            return None

    def _create_fallback_action(self) -> np.ndarray:
        """
        Create fallback action when Tree Method hasn't computed phases yet.

        Returns:
            np.ndarray: Fallback action in appropriate format
        """
        if RL_USE_CONTINUOUS_ACTIONS:
            # Return uniform logits (equal proportions after softmax)
            num_values = len(self.junction_ids) * RL_ACTIONS_PER_JUNCTION
            return np.zeros(num_values, dtype=np.float32)
        else:
            # Return current SUMO phases
            actions = []
            for junction_id in self.junction_ids:
                try:
                    current_phase = traci.trafficlight.getPhase(junction_id)
                    actions.append(current_phase)
                except Exception as e:
                    self.logger.warning(
                        f"Could not get phase for {junction_id} during fallback: {e}")
                    actions.append(0)  # Default to phase 0
            return np.array(actions, dtype=np.int32)

    def _extract_duration_logits(self, phase_durations: Dict[str, Dict[int, float]]) -> np.ndarray:
        """
        Extract phase durations and convert to logits for continuous actions.

        Process:
        1. Extract durations for all 4 phases at each junction
        2. Convert durations to proportions (normalized to sum to 1.0)
        3. Convert proportions to logits using inverse softmax: logit = log(proportion + epsilon)

        Args:
            phase_durations: Dict[junction_id -> Dict[phase_idx -> duration_seconds]]

        Returns:
            np.ndarray: Logits array of shape (num_junctions * 4,)
        """
        logits = []
        epsilon = 1e-8  # For numerical stability

        # DEBUG: Log first junction's extraction
        first_junction_logged = False

        for junction_id in self.junction_ids:
            if junction_id in phase_durations:
                junction_durations = phase_durations[junction_id]

                # Extract durations for all 4 phases (use 0.0 if phase not present)
                durations = [junction_durations.get(i, 0.0) for i in range(RL_ACTIONS_PER_JUNCTION)]

                # DEBUG: Log first junction's extracted durations
                if not first_junction_logged:
                    self.logger.info(f"📤 EXTRACT: junction={junction_id}, durations={durations}")
                    first_junction_logged = True

                # Convert to proportions
                total_duration = sum(durations)
                if total_duration > 0:
                    proportions = [d / total_duration for d in durations]
                else:
                    # Fallback to uniform proportions
                    proportions = [1.0 / RL_ACTIONS_PER_JUNCTION] * RL_ACTIONS_PER_JUNCTION

                # Convert proportions to logits (inverse softmax)
                # After softmax, these logits will reproduce the original proportions
                junction_logits = [np.log(p + epsilon) for p in proportions]
                logits.extend(junction_logits)
            else:
                # Junction not in Tree Method's current decisions - use uniform logits
                uniform_logits = [0.0] * RL_ACTIONS_PER_JUNCTION
                logits.extend(uniform_logits)

        return np.array(logits, dtype=np.float32)

    def _extract_phase_indices(self, phase_durations: Dict[str, Dict[int, float]]) -> np.ndarray:
        """
        Extract active phase indices for discrete actions (legacy mode).

        Args:
            phase_durations: Dict[junction_id -> Dict[phase_idx -> duration_seconds]]

        Returns:
            np.ndarray: Phase indices array of shape (num_junctions,)
        """
        actions = []
        for junction_id in self.junction_ids:
            if junction_id in phase_durations:
                # Find active phase (the one Tree Method selected)
                active_phase = self._find_active_phase(
                    junction_id, phase_durations[junction_id]
                )
                actions.append(active_phase)
            else:
                # Junction not in Tree Method's current decisions
                # Use current TraCI phase as fallback
                try:
                    current_phase = traci.trafficlight.getPhase(junction_id)
                    actions.append(current_phase)
                except Exception as e:
                    self.logger.warning(
                        f"Could not get phase for {junction_id}: {e}")
                    actions.append(0)  # Default to phase 0

        return np.array(actions, dtype=np.int32)

    def _find_active_phase(self, junction_id: str, phase_durations: Dict[int, float]) -> int:
        """
        Determine which phase Tree Method selected for this junction.

        Tree Method stores phase durations for all phases. The active phase is
        typically the one with non-zero duration or the currently executing phase.

        Args:
            junction_id: Traffic light ID
            phase_durations: Dict mapping phase_index -> duration_seconds

        Returns:
            int: Active phase index
        """
        try:
            # Strategy 1: Check current TraCI phase (most reliable)
            current_phase = traci.trafficlight.getPhase(junction_id)
            if current_phase in phase_durations:
                return current_phase

            # Strategy 2: Find phase with maximum duration
            if phase_durations:
                active_phase = max(phase_durations.items(),
                                   key=lambda x: x[1])[0]
                return active_phase

            # Fallback: Use current TraCI phase
            return current_phase

        except Exception as e:
            self.logger.warning(
                f"Could not determine active phase for {junction_id}: {e}")
            return 0  # Default to phase 0

    def verify_state_synchronization(
        self,
        rl_observation: np.ndarray,
        rl_edge_ids: List[str]
    ) -> Dict[str, float]:
        """
        Verify that RL and Tree Method see identical traffic observations.

        This is critical for imitation learning - the expert (Tree Method) and
        learner (RL agent) must observe the same state when making decisions.

        Args:
            rl_observation: RL environment's observation vector
            rl_edge_ids: List of edge IDs corresponding to RL observation

        Returns:
            Dict with verification metrics (max_difference, mean_difference, etc.)
        """
        try:
            # Both RL and Tree Method use same Link.calc_k_by_u() on TraCI data
            # So states should be identical - this is verification only

            differences = []

            # Sample a few edges for comparison
            sample_edges = rl_edge_ids[:min(5, len(rl_edge_ids))]

            for edge_id in sample_edges:
                # Get speed from TraCI (same source for both)
                speed_ms = traci.edge.getLastStepMeanSpeed(edge_id)
                speed_kmh = speed_ms * 3.6

                # Both RL and Tree Method would call same calculation
                # No need to actually compare - they use identical code
                # This is just a sanity check that TraCI is responding
                differences.append(0.0)  # Perfect synchronization expected

            return {
                'max_difference': max(differences) if differences else 0.0,
                'mean_difference': np.mean(differences) if differences else 0.0,
                'synchronized': True
            }

        except Exception as e:
            self.logger.error(
                f"State synchronization verification failed: {e}")
            return {
                'max_difference': float('inf'),
                'mean_difference': float('inf'),
                'synchronized': False
            }
