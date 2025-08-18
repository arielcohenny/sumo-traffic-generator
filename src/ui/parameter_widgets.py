"""
Streamlit parameter widgets for DBPS GUI.

This module defines all the parameter input widgets corresponding to CLI arguments.
"""

import streamlit as st
from typing import Dict, Any

# Import all constants from centralized module
from src.constants import (
    # Default Values
    DEFAULT_GRID_DIMENSION, DEFAULT_BLOCK_SIZE_M, DEFAULT_FIXED_LANE_COUNT,
    DEFAULT_NUM_VEHICLES, DEFAULT_SHORTEST_ROUTING_PCT, DEFAULT_REALTIME_ROUTING_PCT,
    DEFAULT_FASTEST_ROUTING_PCT, DEFAULT_ATTRACTIVENESS_ROUTING_PCT,
    DEFAULT_PASSENGER_VEHICLE_PCT, DEFAULT_COMMERCIAL_VEHICLE_PCT, DEFAULT_PUBLIC_VEHICLE_PCT,
    DEFAULT_SEED, DEFAULT_STEP_LENGTH, DEFAULT_END_TIME, DEFAULT_LAND_USE_BLOCK_SIZE_M,
    DEFAULT_START_TIME_HOUR, DEFAULT_BOTTLENECK_DETECTION_INTERVAL, DEFAULT_ATLCS_INTERVAL,
    DEFAULT_TREE_METHOD_INTERVAL,
    
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
    
    # Rush Hours Values
    DEFAULT_MORNING_START, DEFAULT_MORNING_END, DEFAULT_MORNING_PCT,
    DEFAULT_EVENING_START, DEFAULT_EVENING_END, DEFAULT_EVENING_PCT, DEFAULT_REST_PCT,
    
    # Other Constants
    TEMP_FILE_PREFIX
)


class ParameterWidgets:
    """Collection of parameter input widgets for DBPS GUI."""

    @staticmethod
    def network_section() -> Dict[str, Any]:
        """Network configuration parameters."""
        params = {}

        st.subheader("🏗️ Network Configuration")

        # Remove OSM support - only synthetic grid
        params["osm_file"] = None

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
        lane_type = st.selectbox(
            "Lane Count Algorithm",
            ["realistic", "random", "fixed"],
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
                "Shortest %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_SHORTEST_ROUTING_PCT, key="shortest")
        with routing_col2:
            realtime_pct = st.number_input(
                "Realtime %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_REALTIME_ROUTING_PCT, key="realtime")
        with routing_col3:
            fastest_pct = st.number_input(
                "Fastest %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_FASTEST_ROUTING_PCT, key="fastest")
        with routing_col4:
            attractiveness_pct = st.number_input(
                "Attractiveness %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_ATTRACTIVENESS_ROUTING_PCT, key="attractiveness")

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
        vehicle_col1, vehicle_col2, vehicle_col3 = st.columns(3)

        with vehicle_col1:
            passenger_pct = st.number_input(
                "Passenger %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_PASSENGER_VEHICLE_PCT, key="passenger")
        with vehicle_col2:
            commercial_pct = st.number_input(
                "Commercial %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_COMMERCIAL_VEHICLE_PCT, key="commercial")
        with vehicle_col3:
            public_pct = st.number_input(
                "Public %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_PUBLIC_VEHICLE_PCT, key="public")

        total_vehicles = passenger_pct + commercial_pct + public_pct
        if total_vehicles != MAX_PERCENTAGE:
            st.error(
                f"Vehicle type percentages must sum to {MAX_PERCENTAGE}% (current: {total_vehicles}%)")

        params["vehicle_types"] = f"passenger {passenger_pct} commercial {commercial_pct} public {public_pct}"

        # Departure pattern
        departure_pattern = st.selectbox(
            "Departure Pattern",
            ["six_periods", "uniform", "rush_hours", "hourly"],
            help="How vehicles are distributed over time"
        )

        if departure_pattern == "rush_hours":
            st.write("Custom Rush Hours Pattern:")
            morning_start = st.number_input(
                "Morning Start", min_value=MIN_PERCENTAGE, max_value=23, value=DEFAULT_MORNING_START, key="morning_start")
            morning_end = st.number_input(
                "Morning End", min_value=MIN_PERCENTAGE, max_value=23, value=DEFAULT_MORNING_END, key="morning_end")
            morning_pct = st.number_input(
                "Morning %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_MORNING_PCT, key="morning_pct")

            evening_start = st.number_input(
                "Evening Start", min_value=MIN_PERCENTAGE, max_value=23, value=DEFAULT_EVENING_START, key="evening_start")
            evening_end = st.number_input(
                "Evening End", min_value=MIN_PERCENTAGE, max_value=23, value=DEFAULT_EVENING_END, key="evening_end")
            evening_pct = st.number_input(
                "Evening %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_EVENING_PCT, key="evening_pct")

            rest_pct = st.number_input(
                "Rest of Day %", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE, value=DEFAULT_REST_PCT, key="rest_pct")

            params["departure_pattern"] = f"rush_hours:{morning_start}-{morning_end}:{morning_pct},{evening_start}-{evening_end}:{evening_pct},rest:{rest_pct}"
        else:
            params["departure_pattern"] = departure_pattern

        return params

    @staticmethod
    def simulation_section() -> Dict[str, Any]:
        """Simulation control parameters."""
        params = {}

        st.subheader("⚙️ Simulation Control")

        # Seed configuration
        use_random_seed = st.checkbox("Use Random Seed", value=True)
        if use_random_seed:
            if st.button("Generate New Seed"):
                import random
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
                help="Reproducible random seed"
            )

        params["step_length"] = st.number_input(
            "Step Length (seconds)",
            min_value=MIN_STEP_LENGTH,
            max_value=MAX_STEP_LENGTH,
            value=DEFAULT_STEP_LENGTH,
            step=STEP_LENGTH_STEP,
            help="Simulation time step granularity in complete seconds"
        )

        # Time input in seconds (no conversion)
        params["end_time"] = st.number_input(
            "Simulation Duration (seconds)",
            min_value=MIN_END_TIME,
            max_value=MAX_END_TIME,  # 48 hours in seconds
            value=DEFAULT_END_TIME,  # 24 hours in seconds
            step=STEP_END_TIME,  # 1 hour increments
            help="How long to run the simulation in seconds"
        )

        params["gui"] = st.checkbox(
            "Launch SUMO GUI",
            value=False,
            help="Open visual simulation interface (slower but interactive)"
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

        params["attractiveness"] = st.selectbox(
            "Attractiveness Method",
            ["poisson", "land_use", "gravity", "iac", "hybrid"],
            help="Algorithm for calculating edge attractiveness"
        )

        params["time_dependent"] = st.checkbox(
            "Time-Dependent Attractiveness",
            help="Apply 4-phase time-of-day variations"
        )

        params["start_time_hour"] = st.slider(
            "Simulation Start Time (hour)",
            min_value=MIN_START_TIME_HOUR,
            max_value=MAX_START_TIME_HOUR,
            value=DEFAULT_START_TIME_HOUR,
            step=STEP_START_TIME_HOUR,
            help="Real-world hour when simulation begins (0 = midnight)"
        )

        return params

    @staticmethod
    def traffic_control_section() -> Dict[str, Any]:
        """Traffic control parameters."""
        params = {}

        st.subheader("🚦 Traffic Control Configuration")

        params["traffic_light_strategy"] = st.radio(
            "Traffic Light Strategy",
            ["opposites", "incoming"],
            help="How traffic light phases are organized"
        )

        params["traffic_control"] = st.selectbox(
            "Traffic Control Method",
            ["tree_method", "atlcs", "actuated", "fixed"],
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
        else:
            params["tree_method_interval"] = DEFAULT_TREE_METHOD_INTERVAL

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
