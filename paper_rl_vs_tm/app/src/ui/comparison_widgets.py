"""
Streamlit widgets for comparison mode in DBPS GUI.

Provides UI components for configuring and running multi-run comparisons
with network reuse.
"""

import random
import streamlit as st
from pathlib import Path
from typing import List, Optional, Tuple

from src.constants import (
    MAX_SEED,
    DEFAULT_COMPARISON_NUM_RUNS,
    MIN_COMPARISON_RUNS,
    MAX_COMPARISON_RUNS,
    DEFAULT_BASE_PRIVATE_SEED,
    DEFAULT_BASE_PUBLIC_SEED,
    DEFAULT_COMPARISON_METHODS,
    VALID_TRAFFIC_CONTROLS,
)
from src.orchestration.run_spec import RunSpec


def _generate_run_specs(
    num_runs: int,
    base_private: int,
    base_public: int,
    methods: List[str]
) -> List[RunSpec]:
    """Generate run specifications with random seeds from base.

    Uses base seeds as RNG seeds to generate reproducible random seed sequences.

    Args:
        num_runs: Number of seed variations to generate
        base_private: Base seed for generating private traffic seeds
        base_public: Base seed for generating public traffic seeds
        methods: List of traffic control methods to compare

    Returns:
        List of RunSpec instances (num_runs * len(methods) total)
    """
    # Generate private seeds using base_private as RNG seed
    private_rng = random.Random(base_private)
    private_seeds = [private_rng.randint(0, MAX_SEED) for _ in range(num_runs)]

    # Generate public seeds using base_public as RNG seed
    public_rng = random.Random(base_public)
    public_seeds = [public_rng.randint(0, MAX_SEED) for _ in range(num_runs)]

    # Create run specs for each seed × method combination
    run_specs = []
    for i in range(num_runs):
        for method in methods:
            run_specs.append(RunSpec(
                traffic_control=method,
                private_traffic_seed=private_seeds[i],
                public_traffic_seed=public_seeds[i],
                name=f"{method}_seed{i}"
            ))

    return run_specs


class ComparisonWidgets:
    """Collection of widgets for comparison mode configuration."""

    @staticmethod
    def render_comparison_mode() -> Tuple[bool, Optional[List[RunSpec]], Optional[Path]]:
        """Render the comparison mode UI section.

        Returns:
            Tuple of (enabled, run_specs, network_path):
            - enabled: Whether comparison mode is active
            - run_specs: List of RunSpec if comparison mode enabled, None otherwise
            - network_path: Path to existing network if loading, None otherwise
        """
        enabled = st.checkbox(
            "Enable Comparison Mode",
            value=st.session_state.get("comparison_mode_enabled", False),
            key="comparison_mode_enabled",
            help="Run multiple simulations with different seeds and control methods, then compare results"
        )

        if not enabled:
            return False, None, None

        # Network source section
        st.markdown("#### Network Source")

        network_col1, network_col2 = st.columns(2)

        with network_col1:
            network_action = st.radio(
                "Network Source",
                options=["Generate New", "Load Existing"],
                key="comparison_network_action",
                help="Generate a new network or use an existing one"
            )

        network_path = None
        if network_action == "Load Existing":
            with network_col2:
                network_path_str = st.text_input(
                    "Network Path",
                    value=st.session_state.get("comparison_network_path", ""),
                    key="comparison_network_path",
                    help="Path to existing network/ folder"
                )
                if network_path_str:
                    network_path = Path(network_path_str)
                    if network_path.exists() and (network_path / "grid.net.xml").exists():
                        st.success("Network found")
                    else:
                        st.warning("Network not found at this path")
                        network_path = None

        # Comparison configuration section
        st.markdown("#### Comparison Configuration")

        # Number of seed variations
        num_runs = st.number_input(
            "Number of Seed Variations",
            min_value=MIN_COMPARISON_RUNS,
            max_value=MAX_COMPARISON_RUNS,
            value=st.session_state.get("comparison_num_runs", DEFAULT_COMPARISON_NUM_RUNS),
            key="comparison_num_runs",
            help="Number of different seed combinations to test"
        )

        # Base seeds (separate fields)
        seed_col1, seed_col2 = st.columns(2)

        with seed_col1:
            base_private_seed = st.number_input(
                "Base Private Seed",
                min_value=0,
                max_value=MAX_SEED,
                value=st.session_state.get("comparison_base_private", DEFAULT_BASE_PRIVATE_SEED),
                key="comparison_base_private",
                help="Base seed for generating private traffic seeds (reproducible random sequence)"
            )

        with seed_col2:
            base_public_seed = st.number_input(
                "Base Public Seed",
                min_value=0,
                max_value=MAX_SEED,
                value=st.session_state.get("comparison_base_public", DEFAULT_BASE_PUBLIC_SEED),
                key="comparison_base_public",
                help="Base seed for generating public traffic seeds (reproducible random sequence)"
            )

        # Methods multi-select
        methods = st.multiselect(
            "Traffic Control Methods",
            options=list(VALID_TRAFFIC_CONTROLS),
            default=st.session_state.get("comparison_methods", DEFAULT_COMPARISON_METHODS),
            key="comparison_methods",
            help="Select which traffic control methods to compare"
        )

        # Validate methods selection
        if not methods:
            st.warning("Please select at least one traffic control method")
            return True, None, network_path

        # Generate run specs
        run_specs = _generate_run_specs(
            num_runs, base_private_seed, base_public_seed, methods
        )

        # Summary
        total_runs = len(run_specs)
        st.info(f"**{total_runs} total runs** = {num_runs} seeds x {len(methods)} methods")

        # Show generated seeds preview (collapsed)
        with st.expander("Preview Generated Seeds", expanded=False):
            # Show first few seeds
            preview_count = min(5, num_runs)
            st.markdown("**First few seed pairs:**")

            # Generate preview of seeds
            private_rng = random.Random(base_private_seed)
            public_rng = random.Random(base_public_seed)

            preview_data = []
            for i in range(preview_count):
                p_seed = private_rng.randint(0, 999999)
                pub_seed = public_rng.randint(0, 999999)
                preview_data.append(f"Seed {i}: private={p_seed}, public={pub_seed}")

            st.text("\n".join(preview_data))

            if num_runs > preview_count:
                st.text(f"... and {num_runs - preview_count} more")

        return True, run_specs, network_path

    @staticmethod
    def render_generate_network_only() -> bool:
        """Render a button to generate network only.

        Returns:
            True if the button was clicked
        """
        return st.button(
            "Generate Network Only",
            key="generate_network_only_btn",
            help="Generate network files (steps 1-5) without running simulations. "
                 "Useful for preparing a network to reuse across multiple comparison runs."
        )
