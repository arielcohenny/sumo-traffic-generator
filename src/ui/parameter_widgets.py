"""
Streamlit parameter widgets for DBPS GUI.

This module defines all the parameter input widgets corresponding to CLI arguments.
"""

import random
import streamlit as st
from typing import Dict, Any

# Import all constants from centralized module
from src.constants import (
    # Default Values
    DEFAULT_GRID_DIMENSION, DEFAULT_BLOCK_SIZE_M, DEFAULT_FIXED_LANE_COUNT,
    DEFAULT_NUM_VEHICLES, DEFAULT_SHORTEST_ROUTING_PCT, DEFAULT_REALTIME_ROUTING_PCT,
    DEFAULT_FASTEST_ROUTING_PCT, DEFAULT_ATTRACTIVENESS_ROUTING_PCT,
    DEFAULT_PASSENGER_VEHICLE_PCT, DEFAULT_PUBLIC_VEHICLE_PCT, DEFAULT_PASSENGER_ROUTES,
    DEFAULT_PUBLIC_ROUTES, DEFAULT_SEED, DEFAULT_STEP_LENGTH, DEFAULT_END_TIME,
    DEFAULT_LAND_USE_BLOCK_SIZE_M, DEFAULT_START_TIME_HOUR, DEFAULT_BOTTLENECK_DETECTION_INTERVAL,
    DEFAULT_ATLCS_INTERVAL, DEFAULT_TREE_METHOD_INTERVAL, DEFAULT_ATTRACTIVENESS, DEFAULT_LANE_COUNT,
    DEFAULT_DEPARTURE_PATTERN, DEFAULT_TRAFFIC_LIGHT_STRATEGY, DEFAULT_TRAFFIC_CONTROL, DEPARTURE_PATTERN_SIX_PERIODS, DEPARTURE_PATTERN_UNIFORM,

    # Min/Max Values
    MIN_GRID_DIMENSION, MAX_GRID_DIMENSION, MIN_BLOCK_SIZE_M, MAX_BLOCK_SIZE_M,
    MIN_LANE_COUNT, MAX_LANE_COUNT, MIN_NUM_VEHICLES, MAX_NUM_VEHICLES,
    MIN_PERCENTAGE, MAX_PERCENTAGE, MIN_SEED, MAX_SEED, MIN_STEP_LENGTH, MAX_STEP_LENGTH,
    MIN_END_TIME, MAX_END_TIME, MIN_LAND_USE_BLOCK_SIZE_M, MAX_LAND_USE_BLOCK_SIZE_M,
    MIN_START_TIME_HOUR, MAX_START_TIME_HOUR, MIN_BOTTLENECK_DETECTION_INTERVAL,
    MAX_BOTTLENECK_DETECTION_INTERVAL, MIN_ATLCS_INTERVAL, MAX_ATLCS_INTERVAL,
    MIN_TREE_METHOD_INTERVAL, MAX_TREE_METHOD_INTERVAL,

    # Step Values
    STEP_BLOCK_SIZE_M, STEP_NUM_VEHICLES, STEP_LENGTH_STEP, STEP_END_TIME,
    STEP_LAND_USE_BLOCK_SIZE_M, STEP_START_TIME_HOUR, STEP_TREE_METHOD_INTERVAL,

    # Other Constants
    TEMP_FILE_PREFIX, DEFAULT_WORKSPACE_DIR,

    # Departure Pattern Constraints
    UNIFORM_DEPARTURE_PATTERN, FIXED_START_TIME_HOUR
)

# Import Tree Method parameters
from src.traffic_control.decentralized_traffic_bottlenecks.shared.config import M, L


class ParameterWidgets:
    """Collection of parameter input widgets for DBPS GUI."""

    @staticmethod
    def network_section() -> Dict[str, Any]:
        """Network configuration parameters."""
        params = {}

        st.subheader("🏗️ Network Configuration")

        params["grid_dimension"] = st.number_input(
            "Grid Dimension",
            min_value=MIN_GRID_DIMENSION,
            max_value=MAX_GRID_DIMENSION,
            value=DEFAULT_GRID_DIMENSION,
            help="Number of rows and columns in the grid (e.g., 5 = 5x5 grid)"
        )

        params["block_size_m"] = st.slider(
            "Block Size (meters)",
            min_value=MIN_BLOCK_SIZE_M,
            max_value=MAX_BLOCK_SIZE_M,
            value=DEFAULT_BLOCK_SIZE_M,
            step=STEP_BLOCK_SIZE_M,
            help="Distance between intersections in meters"
        )

        # Visual junction selection
        params["junctions_to_remove"] = ParameterWidgets._junction_selection_widget(
            params["grid_dimension"])

        # Lane configuration
        lane_options = ["realistic", "random", "fixed"]
        lane_type = st.selectbox(
            "Lane Count Algorithm",
            lane_options,
            index=lane_options.index(DEFAULT_LANE_COUNT),
            help="How to assign lane counts to edges"
        )

        if lane_type == "fixed":
            lane_count = st.number_input(
                "Fixed Lane Count",
                min_value=MIN_LANE_COUNT,
                max_value=MAX_LANE_COUNT,
                value=DEFAULT_FIXED_LANE_COUNT,
                help="Number of lanes for all edges"
            )
            params["lane_count"] = str(lane_count)
        else:
            params["lane_count"] = lane_type

        # Advanced lane configuration
        with st.expander("Advanced Lane Configuration"):
            params["custom_lanes"] = st.text_area(
                "Custom Lane Definitions",
                help="Format: EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=...",
                placeholder="Example: EdgeA=tail:2,head:EdgeB:1,EdgeC:1"
            )

            custom_lanes_file = st.file_uploader(
                "Custom Lanes File",
                type=['txt'],
                help="Upload file with custom lane definitions"
            )

            if custom_lanes_file is not None:
                temp_path = f"{TEMP_FILE_PREFIX}{custom_lanes_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(custom_lanes_file.getbuffer())
                params["custom_lanes_file"] = temp_path
            else:
                params["custom_lanes_file"] = None

        return params

    @staticmethod
    def traffic_section() -> Dict[str, Any]:
        """Traffic generation parameters."""
        params = {}

        st.subheader("🚗 Traffic Generation")

        params["num_vehicles"] = st.number_input(
            "Number of Vehicles",
            min_value=MIN_NUM_VEHICLES,
            max_value=MAX_NUM_VEHICLES,
            value=DEFAULT_NUM_VEHICLES,
            step=STEP_NUM_VEHICLES,
            help="Total number of vehicles to generate for the simulation"
        )

        # Routing strategy
        st.write("**Routing Strategy**")
        routing_col1, routing_col2, routing_col3, routing_col4 = st.columns(4)

        with routing_col1:
            shortest_pct = st.number_input(
                "Shortest %",
                min_value=MIN_PERCENTAGE,
                max_value=MAX_PERCENTAGE,
                value=DEFAULT_SHORTEST_ROUTING_PCT,
                key="shortest",
                help="Static shortest path by distance using Dijkstra algorithm. Routes computed once, never change during simulation. Minimizes total distance traveled."
            )
        with routing_col2:
            realtime_pct = st.number_input(
                "Realtime %",
                min_value=MIN_PERCENTAGE,
                max_value=MAX_PERCENTAGE,
                value=DEFAULT_REALTIME_ROUTING_PCT,
                key="realtime",
                help="Dynamic routing that responds to current traffic conditions. 30-second rerouting intervals. Uses real-time traffic data to avoid congestion, like GPS apps that reroute around traffic jams. → 'What's fastest RIGHT NOW?' (reacts to live traffic)"
            )
        with routing_col3:
            fastest_pct = st.number_input(
                "Fastest %",
                min_value=MIN_PERCENTAGE,
                max_value=MAX_PERCENTAGE,
                value=DEFAULT_FASTEST_ROUTING_PCT,
                key="fastest",
                help="Dynamic routing optimized for minimum travel time. 45-second rerouting intervals. Focuses on speed/time rather than distance - may choose longer routes if they're faster. → 'What's typically the fastest route?' (optimizes for speed patterns)"
            )
        with routing_col4:
            attractiveness_pct = st.number_input(
                "Attractiveness %",
                min_value=MIN_PERCENTAGE,
                max_value=MAX_PERCENTAGE,
                value=DEFAULT_ATTRACTIVENESS_ROUTING_PCT,
                key="attractiveness",
                help="Multi-criteria routing combining efficiency + destination appeal. Balances shortest/fastest path with attractiveness of areas. Simulates drivers choosing scenic or interesting routes."
            )

        total_routing = shortest_pct + realtime_pct + fastest_pct + attractiveness_pct
        if total_routing != MAX_PERCENTAGE:
            st.error(
                f"Routing percentages must sum to {MAX_PERCENTAGE}% (current: {total_routing}%)")

        routing_parts = []
        if shortest_pct > 0:
            routing_parts.append(f"shortest {shortest_pct}")
        if realtime_pct > 0:
            routing_parts.append(f"realtime {realtime_pct}")
        if fastest_pct > 0:
            routing_parts.append(f"fastest {fastest_pct}")
        if attractiveness_pct > 0:
            routing_parts.append(f"attractiveness {attractiveness_pct}")

        params["routing_strategy"] = " ".join(
            routing_parts) if routing_parts else f"shortest {DEFAULT_SHORTEST_ROUTING_PCT}"

        # Vehicle types
        st.write("**Vehicle Types**")
        vehicle_col1, vehicle_col2 = st.columns(2)

        with vehicle_col1:
            passenger_pct = st.number_input(
                "Passenger %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_PASSENGER_VEHICLE_PCT, key="passenger")
        with vehicle_col2:
            public_pct = st.number_input(
                "Public %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_PUBLIC_VEHICLE_PCT, key="public")

        total_vehicles = passenger_pct + public_pct
        if total_vehicles != MAX_PERCENTAGE:
            st.error(
                f"Vehicle type percentages must sum to {MAX_PERCENTAGE}% (current: {total_vehicles}%)")

        params["vehicle_types"] = f"passenger {passenger_pct} public {public_pct}"

        # Route Pattern Configuration
        st.write("**🚗 Route Pattern Configuration**")
        st.write("Configure spatial route patterns for different vehicle types")

        # Passenger Route Patterns (show only if passenger percentage > 0)
        if passenger_pct > 0:
            st.write("*Passenger Vehicle Route Patterns*")
            pass_col1, pass_col2, pass_col3, pass_col4 = st.columns(4)

            with pass_col1:
                passenger_in_pct = st.number_input(
                    "In-bound %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=30, key="passenger_in",
                    help="Start at network boundary, end inside network (external arrivals)")
            with pass_col2:
                passenger_out_pct = st.number_input(
                    "Out-bound %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=30, key="passenger_out",
                    help="Start inside network, end at boundary (departures to external)")
            with pass_col3:
                passenger_inner_pct = st.number_input(
                    "Inner %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=25, key="passenger_inner",
                    help="Both start and end inside the network (local trips)")
            with pass_col4:
                passenger_pass_pct = st.number_input(
                    "Pass-through %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=15, key="passenger_pass",
                    help="Start at boundary, end at different boundary (transit through area)")

            total_passenger_routes = passenger_in_pct + \
                passenger_out_pct + passenger_inner_pct + passenger_pass_pct
            if total_passenger_routes != MAX_PERCENTAGE:
                st.error(
                    f"Passenger route percentages must sum to {MAX_PERCENTAGE}% (current: {total_passenger_routes}%)")

            params["passenger_routes"] = f"in {passenger_in_pct} out {passenger_out_pct} inner {passenger_inner_pct} pass {passenger_pass_pct}"
        else:
            params["passenger_routes"] = DEFAULT_PASSENGER_ROUTES

        # Public Route Patterns (show only if public percentage > 0)
        if public_pct > 0:
            st.write("*Public Vehicle Route Patterns*")
            pub_col1, pub_col2, pub_col3, pub_col4 = st.columns(4)

            with pub_col1:
                public_in_pct = st.number_input(
                    "In-bound %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=25, key="public_in",
                    help="Transit routes bringing people into the urban area")
            with pub_col2:
                public_out_pct = st.number_input(
                    "Out-bound %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=25, key="public_out",
                    help="Transit routes taking people out of the urban area")
            with pub_col3:
                public_inner_pct = st.number_input(
                    "Inner %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=35, key="public_inner",
                    help="Internal transit routes within the urban area")
            with pub_col4:
                public_pass_pct = st.number_input(
                    "Pass-through %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE,
                    value=15, key="public_pass",
                    help="Transit routes passing through the area")

            total_public_routes = public_in_pct + \
                public_out_pct + public_inner_pct + public_pass_pct
            if total_public_routes != MAX_PERCENTAGE:
                st.error(
                    f"Public route percentages must sum to {MAX_PERCENTAGE}% (current: {total_public_routes}%)")

            params["public_routes"] = f"in {public_in_pct} out {public_out_pct} inner {public_inner_pct} pass {public_pass_pct}"
        else:
            params["public_routes"] = DEFAULT_PUBLIC_ROUTES

        # Departure pattern with session state for cross-section coordination
        departure_options = [DEPARTURE_PATTERN_SIX_PERIODS,
                             DEPARTURE_PATTERN_UNIFORM]
        departure_pattern = st.selectbox(
            "Departure Pattern",
            departure_options,
            index=departure_options.index(DEFAULT_DEPARTURE_PATTERN),
            help="How vehicles are distributed over time"
        )

        # Store in session state for other sections to access
        st.session_state.departure_pattern = departure_pattern

        # Show constraint info for non-uniform patterns
        if departure_pattern != UNIFORM_DEPARTURE_PATTERN:
            st.info("**Time constraints**: Non-uniform patterns require midnight start (0.0h) and 24-hour duration (86400s) for realistic daily cycles.")

        params["departure_pattern"] = departure_pattern

        return params

    @staticmethod
    def simulation_section() -> Dict[str, Any]:
        """Simulation control parameters."""
        params = {}

        st.subheader("⚙️ Simulation Control")

        # Seed configuration with multiple seed support
        seed_mode = st.radio(
            "Seed Configuration",
            options=["Single Seed (Simple)", "Multiple Seeds (Advanced)"],
            help="Single seed sets all seeds to the same value (backward compatible). "
                 "Multiple seeds allow fine-grained control over network vs traffic generation."
        )

        if seed_mode == "Single Seed (Simple)":
            # Original single seed behavior for backward compatibility
            use_random_seed = st.checkbox("Use Random Seed", value=True)
            if use_random_seed:
                if st.button("Generate New Seed"):
                    st.session_state.generated_seed = random.randint(
                        MIN_SEED, MAX_SEED)

                if 'generated_seed' not in st.session_state:
                    st.session_state.generated_seed = DEFAULT_SEED

                params["seed"] = st.session_state.generated_seed
                st.info(f"Using seed: {params['seed']}")
            else:
                params["seed"] = st.number_input(
                    "Custom Seed",
                    min_value=MIN_SEED,
                    max_value=MAX_SEED,
                    value=DEFAULT_SEED,
                    help="Reproducible random seed (sets all seeds to same value)"
                )
        else:
            # Multiple seeds mode for advanced control
            st.markdown(
                "**Configure seeds for different simulation aspects:**")

            # Initialize session state for seeds with independent random values
            if 'multi_network_seed' not in st.session_state:
                st.session_state.multi_network_seed = random.randint(
                    MIN_SEED, MAX_SEED)
            if 'multi_private_seed' not in st.session_state:
                st.session_state.multi_private_seed = random.randint(
                    MIN_SEED, MAX_SEED)
            if 'multi_public_seed' not in st.session_state:
                st.session_state.multi_public_seed = random.randint(
                    MIN_SEED, MAX_SEED)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Network Structure**")
                st.caption(
                    "Junction removal, lanes, land use, edge attractiveness")
                params["network-seed"] = st.number_input(
                    "Network Seed",
                    min_value=MIN_SEED,
                    max_value=MAX_SEED,
                    value=st.session_state.multi_network_seed,
                    help="Seed for network structure generation"
                )
                # Update session state when user changes value manually
                st.session_state.multi_network_seed = params["network-seed"]

            with col2:
                st.markdown("**Private Traffic**")
                st.caption("Passenger vehicles only")
                params["private-traffic-seed"] = st.number_input(
                    "Private Traffic Seed",
                    min_value=MIN_SEED,
                    max_value=MAX_SEED,
                    value=st.session_state.multi_private_seed,
                    help="Seed for private vehicle generation"
                )
                # Update session state when user changes value manually
                st.session_state.multi_private_seed = params["private-traffic-seed"]

            with col3:
                st.markdown("**Public Traffic**")
                st.caption("Public transportation vehicles")
                params["public-traffic-seed"] = st.number_input(
                    "Public Traffic Seed",
                    min_value=MIN_SEED,
                    max_value=MAX_SEED,
                    value=st.session_state.multi_public_seed,
                    help="Seed for public vehicle generation"
                )
                # Update session state when user changes value manually
                st.session_state.multi_public_seed = params["public-traffic-seed"]

            # Generate completely independent seeds
            if st.button("Generate New Seeds"):
                st.session_state.multi_network_seed = random.randint(
                    MIN_SEED, MAX_SEED)
                st.session_state.multi_private_seed = random.randint(
                    MIN_SEED, MAX_SEED)
                st.session_state.multi_public_seed = random.randint(
                    MIN_SEED, MAX_SEED)
                st.rerun()

        params["step_length"] = st.number_input(
            "Step Length (seconds)",
            min_value=MIN_STEP_LENGTH,
            max_value=MAX_STEP_LENGTH,
            value=DEFAULT_STEP_LENGTH,
            step=STEP_LENGTH_STEP,
            help="Simulation time step granularity in complete seconds"
        )

        # Time input with departure pattern constraints
        departure_pattern = st.session_state.get(
            'departure_pattern', DEFAULT_DEPARTURE_PATTERN)
        is_uniform_pattern = departure_pattern == UNIFORM_DEPARTURE_PATTERN

        if is_uniform_pattern:
            # Show normal end_time control for uniform pattern
            params["end_time"] = st.number_input(
                "Simulation Duration (seconds)",
                min_value=MIN_END_TIME,
                max_value=MAX_END_TIME,
                value=DEFAULT_END_TIME,
                step=STEP_END_TIME,
                help="How long to run the simulation in seconds"
            )
        else:
            # Fixed duration for non-uniform patterns
            params["end_time"] = MAX_END_TIME
            st.info(
                f"🕐 **Fixed Duration**: {MAX_END_TIME:,} seconds (24 hours) - required for '{departure_pattern}' pattern")

        # Start time with departure pattern constraints
        if is_uniform_pattern:
            # Show normal start_time_hour control for uniform pattern
            params["start_time_hour"] = st.slider(
                "Simulation Start Time (hour)",
                min_value=MIN_START_TIME_HOUR,
                max_value=MAX_START_TIME_HOUR,
                value=DEFAULT_START_TIME_HOUR,
                step=STEP_START_TIME_HOUR,
                help="Real-world hour when simulation begins (0 = midnight)"
            )
        else:
            # Fixed start time for non-uniform patterns
            params["start_time_hour"] = FIXED_START_TIME_HOUR
            st.info(
                f"🌙 **Fixed Start Time**: {FIXED_START_TIME_HOUR} hours (midnight) - required for '{departure_pattern}' pattern")

        params["gui"] = st.checkbox(
            "Launch SUMO GUI",
            value=False,
            help="Open visual simulation interface (slower but interactive)"
        )

        # Only show hide_zones option when GUI is enabled
        if params["gui"]:
            params["hide_zones"] = st.checkbox(
                "Hide Zones",
                value=False,
                help="Hide zone polygons from SUMO GUI (zones still used for traffic generation)"
            )
        else:
            params["hide_zones"] = False

        params["workspace"] = st.text_input(
            "Workspace Directory",
            value=DEFAULT_WORKSPACE_DIR,
            help="Parent directory where 'workspace' folder will be created for simulation output files"
        )

        params["log_bottleneck_events"] = st.checkbox(
            "Enable Bottleneck Event Logging",
            value=False,
            help="Log vehicle counts per edge to workspace/bottleneck_events.csv every 360 seconds (6 minutes). Useful for debugging and research."
        )

        return params

    @staticmethod
    def zone_attractiveness_section() -> Dict[str, Any]:
        """Zone and attractiveness parameters."""
        params = {}

        st.subheader("🗺️ Zone & Attractiveness Configuration")

        params["land_use_block_size_m"] = st.slider(
            "Land Use Block Size (meters)",
            min_value=MIN_LAND_USE_BLOCK_SIZE_M,
            max_value=MAX_LAND_USE_BLOCK_SIZE_M,
            value=DEFAULT_LAND_USE_BLOCK_SIZE_M,
            step=STEP_LAND_USE_BLOCK_SIZE_M,
            help="Resolution of zone generation grid"
        )

        attractiveness_options = ["land_use", "poisson", "iac"]
        params["attractiveness"] = st.selectbox(
            "Attractiveness Method",
            attractiveness_options,
            index=attractiveness_options.index(DEFAULT_ATTRACTIVENESS),
            help="Algorithm for calculating edge attractiveness"
        )

        return params

    @staticmethod
    def traffic_control_section() -> Dict[str, Any]:
        """Traffic control parameters."""
        params = {}

        st.subheader("🚦 Traffic Control Configuration")

        strategy_options = ["partial_opposites", "opposites", "incoming"]
        params["traffic_light_strategy"] = st.radio(
            "Traffic Light Strategy",
            strategy_options,
            index=strategy_options.index(DEFAULT_TRAFFIC_LIGHT_STRATEGY),
            help="How traffic light phases are organized. partial_opposites requires 2+ lanes per edge."
        )

        control_options = ["tree_method", "atlcs", "actuated", "fixed"]
        params["traffic_control"] = st.selectbox(
            "Traffic Control Method",
            control_options,
            index=control_options.index(DEFAULT_TRAFFIC_CONTROL),
            help="Algorithm for traffic light optimization"
        )

        if params["traffic_control"] == "atlcs":
            params["bottleneck_detection_interval"] = st.number_input(
                "Bottleneck Detection Interval (seconds)",
                min_value=MIN_BOTTLENECK_DETECTION_INTERVAL,
                max_value=MAX_BOTTLENECK_DETECTION_INTERVAL,
                value=DEFAULT_BOTTLENECK_DETECTION_INTERVAL,
                help="How often to detect bottlenecks"
            )

            params["atlcs_interval"] = st.number_input(
                "ATLCS Update Interval (seconds)",
                min_value=MIN_ATLCS_INTERVAL,
                max_value=MAX_ATLCS_INTERVAL,
                value=DEFAULT_ATLCS_INTERVAL,
                help="ATLCS pricing update frequency"
            )
        else:
            params["bottleneck_detection_interval"] = DEFAULT_BOTTLENECK_DETECTION_INTERVAL
            params["atlcs_interval"] = DEFAULT_ATLCS_INTERVAL

        if params["traffic_control"] == "tree_method":
            params["tree_method_interval"] = st.slider(
                "Tree Method Interval (seconds)",
                min_value=MIN_TREE_METHOD_INTERVAL,
                max_value=MAX_TREE_METHOD_INTERVAL,
                value=DEFAULT_TREE_METHOD_INTERVAL,
                step=STEP_TREE_METHOD_INTERVAL,
                help="How often Tree Method runs optimization"
            )

            # Advanced Tree Method Parameters
            with st.expander("⚙️ Advanced Tree Method Parameters"):
                st.caption("Advanced parameters for traffic flow calculations. Modify only for research purposes.")

                params["tree_method_m"] = st.number_input(
                    "Tree Method M Parameter",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(M),
                    step=0.1,
                    format="%.1f",
                    help="M parameter for speed-density relationship in traffic flow calculations"
                )

                params["tree_method_l"] = st.number_input(
                    "Tree Method L Parameter",
                    min_value=1.0,
                    max_value=5.0,
                    value=float(L),
                    step=0.1,
                    format="%.1f",
                    help="L parameter for speed-density relationship in traffic flow calculations"
                )
        else:
            params["tree_method_interval"] = DEFAULT_TREE_METHOD_INTERVAL
            params["tree_method_m"] = M
            params["tree_method_l"] = L

        return params

    @staticmethod
    def sample_testing_section() -> Dict[str, Any]:
        """Sample testing parameters."""
        params = {}

        st.subheader("🧪 Sample Testing")

        sample_folder = st.text_input(
            "Tree Method Sample Folder",
            help="Path to pre-built Tree Method sample (optional)"
        )
        params["tree_method_sample"] = sample_folder if sample_folder else None

        return params

    @staticmethod
    def _junction_selection_widget(grid_dimension: int) -> str:
        """Visual junction selection widget."""
        st.write("**Junctions to Remove**")

        # Selection mode
        selection_mode = st.radio(
            "Selection Mode",
            ["Random", "Manual Selection"],
            help="Choose random removal or manually select specific junctions"
        )

        if selection_mode == "Random":
            num_random = st.number_input(
                "Number of Random Junctions to Remove",
                min_value=0,
                max_value=max(0, grid_dimension * grid_dimension - 1),
                value=0,
                help="How many junctions to remove randomly"
            )
            return str(num_random)

        else:  # Manual Selection
            st.write("Click to select junctions to remove:")

            # Initialize session state for junction selection
            if 'selected_junctions' not in st.session_state:
                st.session_state.selected_junctions = set()

            # Initialize reset counter for forcing checkbox refresh
            if 'junction_reset_counter' not in st.session_state:
                st.session_state.junction_reset_counter = 0

            # Reset button
            if st.button("🔄 Clear All Selections"):
                st.session_state.selected_junctions = set()
                # Increment counter to force new keys for checkboxes
                st.session_state.junction_reset_counter += 1
                st.rerun()

            # Create grid layout
            for row in range(grid_dimension):
                cols = st.columns(grid_dimension)
                row_letter = chr(ord('A') + row)

                for col in range(grid_dimension):
                    junction_id = f"{row_letter}{col}"

                    with cols[col]:
                        # Use unique key that changes when reset counter changes
                        checkbox_key = f"junction_{junction_id}_{st.session_state.junction_reset_counter}"

                        # Use checkbox for each junction
                        is_selected = st.checkbox(
                            junction_id,
                            key=checkbox_key,
                            value=junction_id in st.session_state.selected_junctions
                        )

                        if is_selected:
                            st.session_state.selected_junctions.add(
                                junction_id)
                        else:
                            st.session_state.selected_junctions.discard(
                                junction_id)

            # Display selected junctions
            if st.session_state.selected_junctions:
                selected_list = sorted(
                    list(st.session_state.selected_junctions))
                st.info(f"Selected junctions: {', '.join(selected_list)}")
                return ','.join(selected_list)
            else:
                st.info("No junctions selected")
                return "0"

    @staticmethod
    def collect_all_parameters() -> Dict[str, Any]:
        """Collect all parameters from all sections."""
        all_params = {}

        with st.expander("🏗️ Network Configuration", expanded=True):
            all_params.update(ParameterWidgets.network_section())

        with st.expander("🚗 Traffic Generation", expanded=True):
            all_params.update(ParameterWidgets.traffic_section())

        with st.expander("⚙️ Simulation Control", expanded=True):
            all_params.update(ParameterWidgets.simulation_section())

        with st.expander("🗺️ Zone & Attractiveness Configuration", expanded=True):
            all_params.update(ParameterWidgets.zone_attractiveness_section())

        with st.expander("🚦 Traffic Control Configuration", expanded=True):
            all_params.update(ParameterWidgets.traffic_control_section())

        with st.expander("🧪 Sample Testing"):
            all_params.update(ParameterWidgets.sample_testing_section())

        return all_params
