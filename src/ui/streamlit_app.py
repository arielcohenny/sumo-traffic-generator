"""
DBPS Streamlit GUI Application.

Main Streamlit interface for the Decentralised Bottleneck Prioritization Simulation.
"""

from src.constants import (
    # Default Parameter Values
    DEFAULT_GRID_DIMENSION, DEFAULT_BLOCK_SIZE_M, DEFAULT_JUNCTIONS_TO_REMOVE,
    DEFAULT_LANE_COUNT, DEFAULT_NUM_VEHICLES, DEFAULT_ROUTING_STRATEGY,
    DEFAULT_VEHICLE_TYPES, DEFAULT_PASSENGER_ROUTES, DEFAULT_PUBLIC_ROUTES,
    DEFAULT_DEPARTURE_PATTERN, DEFAULT_STEP_LENGTH,
    DEFAULT_END_TIME, DEFAULT_LAND_USE_BLOCK_SIZE_M, DEFAULT_ATTRACTIVENESS,
    DEFAULT_START_TIME_HOUR, DEFAULT_TRAFFIC_LIGHT_STRATEGY, DEFAULT_TRAFFIC_CONTROL,
    DEFAULT_BOTTLENECK_DETECTION_INTERVAL, DEFAULT_ATLCS_INTERVAL, DEFAULT_TREE_METHOD_INTERVAL,
    DEFAULT_WORKSPACE_DIR,

    # Progress Bar Values
    PROGRESS_START, PROGRESS_PIPELINE_CREATED, PROGRESS_EXECUTION_STARTED,
    PROGRESS_EXECUTION_RUNNING, PROGRESS_COMPLETED
)
from src.utils.logging import setup_logging, get_logger
from src.pipeline.pipeline_factory import PipelineFactory
from src.validate.errors import ValidationError
from src.validate.validate_arguments import validate_arguments
from src.args.parser import create_argument_parser
from src.ui.output_display import OutputDisplay
from src.ui.parameter_widgets import ParameterWidgets
import streamlit as st
import sys
from pathlib import Path
import argparse
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import constants from centralized module


def generate_command_line(params: Dict[str, Any]) -> str:
    """Generate the equivalent command line for the given parameters."""
    cmd_parts = ["env PYTHONUNBUFFERED=1 python -m src.cli"]

    # Add parameters in logical order
    # Network parameters
    cmd_parts.append(
        f"--grid_dimension {params.get('grid_dimension', DEFAULT_GRID_DIMENSION)}")
    cmd_parts.append(
        f"--block_size_m {params.get('block_size_m', DEFAULT_BLOCK_SIZE_M)}")

    junctions = params.get('junctions_to_remove', DEFAULT_JUNCTIONS_TO_REMOVE)
    if junctions and junctions != DEFAULT_JUNCTIONS_TO_REMOVE:
        cmd_parts.append(f"--junctions_to_remove \"{junctions}\"")

    cmd_parts.append(
        f"--lane_count {params.get('lane_count', DEFAULT_LANE_COUNT)}")

    if params.get('custom_lanes'):
        cmd_parts.append(f"--custom_lanes \"{params['custom_lanes']}\"")

    if params.get('custom_lanes_file'):
        cmd_parts.append(
            f"--custom_lanes_file \"{params['custom_lanes_file']}\"")

    # Traffic parameters
    cmd_parts.append(
        f"--num_vehicles {params.get('num_vehicles', DEFAULT_NUM_VEHICLES)}")
    cmd_parts.append(
        f"--routing_strategy \"{params.get('routing_strategy', DEFAULT_ROUTING_STRATEGY)}\"")
    cmd_parts.append(
        f"--vehicle_types \"{params.get('vehicle_types', DEFAULT_VEHICLE_TYPES)}\"")
    
    # Route patterns (only add if not default values)
    passenger_routes = params.get('passenger_routes', DEFAULT_PASSENGER_ROUTES)
    if passenger_routes != DEFAULT_PASSENGER_ROUTES:
        cmd_parts.append(f"--passenger-routes \"{passenger_routes}\"")
    
    public_routes = params.get('public_routes', DEFAULT_PUBLIC_ROUTES)
    if public_routes != DEFAULT_PUBLIC_ROUTES:
        cmd_parts.append(f"--public-routes \"{public_routes}\"")
    
    cmd_parts.append(
        f"--departure_pattern {params.get('departure_pattern', DEFAULT_DEPARTURE_PATTERN)}")

    # Simulation parameters - handle multiple seed types
    if params.get('seed'):
        cmd_parts.append(f"--seed {params['seed']}")
    else:
        # Add individual seeds if specified
        if params.get('network-seed'):
            cmd_parts.append(f"--network-seed {params['network-seed']}")
        if params.get('private-traffic-seed'):
            cmd_parts.append(f"--private-traffic-seed {params['private-traffic-seed']}")
        if params.get('public-traffic-seed'):
            cmd_parts.append(f"--public-traffic-seed {params['public-traffic-seed']}")

    cmd_parts.append(
        f"--step-length {params.get('step_length', DEFAULT_STEP_LENGTH)}")
    cmd_parts.append(f"--end-time {params.get('end_time', DEFAULT_END_TIME)}")

    if params.get('gui', False):
        cmd_parts.append("--gui")
    
    # Add workspace if different from default
    workspace = params.get('workspace', DEFAULT_WORKSPACE_DIR)
    if workspace != DEFAULT_WORKSPACE_DIR:
        cmd_parts.append(f"--workspace \"{workspace}\"")

    # Zone & attractiveness parameters
    cmd_parts.append(
        f"--land_use_block_size_m {params.get('land_use_block_size_m', DEFAULT_LAND_USE_BLOCK_SIZE_M)}")
    cmd_parts.append(
        f"--attractiveness {params.get('attractiveness', DEFAULT_ATTRACTIVENESS)}")


    cmd_parts.append(
        f"--start_time_hour {params.get('start_time_hour', DEFAULT_START_TIME_HOUR)}")

    # Traffic control parameters
    cmd_parts.append(
        f"--traffic_light_strategy {params.get('traffic_light_strategy', DEFAULT_TRAFFIC_LIGHT_STRATEGY)}")
    cmd_parts.append(
        f"--traffic_control {params.get('traffic_control', DEFAULT_TRAFFIC_CONTROL)}")
    cmd_parts.append(
        f"--bottleneck-detection-interval {params.get('bottleneck_detection_interval', DEFAULT_BOTTLENECK_DETECTION_INTERVAL)}")
    cmd_parts.append(
        f"--atlcs-interval {params.get('atlcs_interval', DEFAULT_ATLCS_INTERVAL)}")
    cmd_parts.append(
        f"--tree-method-interval {params.get('tree_method_interval', DEFAULT_TREE_METHOD_INTERVAL)}")

    if params.get('tree_method_sample'):
        cmd_parts.append(
            f"--tree_method_sample \"{params['tree_method_sample']}\"")

    # Format as single line for easy copying
    formatted_cmd = " ".join(cmd_parts)
    return formatted_cmd


def convert_params_to_args(params: Dict[str, Any]) -> argparse.Namespace:
    """Convert GUI parameters to argparse Namespace for validation and execution."""
    # Create a mock argument list
    args_list = []

    # Network parameters
    args_list.extend(
        ["--grid_dimension", str(params.get("grid_dimension", DEFAULT_GRID_DIMENSION))])
    args_list.extend(
        ["--block_size_m", str(params.get("block_size_m", DEFAULT_BLOCK_SIZE_M))])
    args_list.extend(
        ["--junctions_to_remove", str(params.get("junctions_to_remove", DEFAULT_JUNCTIONS_TO_REMOVE))])

    args_list.extend(
        ["--lane_count", str(params.get("lane_count", DEFAULT_LANE_COUNT))])

    if params.get("custom_lanes"):
        args_list.extend(["--custom_lanes", params["custom_lanes"]])

    if params.get("custom_lanes_file"):
        args_list.extend(["--custom_lanes_file", params["custom_lanes_file"]])

    # Traffic parameters
    args_list.extend(
        ["--num_vehicles", str(params.get("num_vehicles", DEFAULT_NUM_VEHICLES))])
    args_list.extend(
        ["--routing_strategy", params.get("routing_strategy", DEFAULT_ROUTING_STRATEGY)])
    args_list.extend(
        ["--vehicle_types", params.get("vehicle_types", DEFAULT_VEHICLE_TYPES)])
    args_list.extend(
        ["--departure_pattern", params.get("departure_pattern", DEFAULT_DEPARTURE_PATTERN)])

    # Simulation parameters - handle multiple seed types
    if params.get("seed"):
        args_list.extend(["--seed", str(params["seed"])])
    else:
        # Add individual seeds if specified
        if params.get("network-seed"):
            args_list.extend(["--network-seed", str(params["network-seed"])])
        if params.get("private-traffic-seed"):
            args_list.extend(["--private-traffic-seed", str(params["private-traffic-seed"])])
        if params.get("public-traffic-seed"):
            args_list.extend(["--public-traffic-seed", str(params["public-traffic-seed"])])

    args_list.extend(
        ["--step-length", str(params.get("step_length", DEFAULT_STEP_LENGTH))])
    args_list.extend(
        ["--end-time", str(params.get("end_time", DEFAULT_END_TIME))])

    if params.get("gui", False):
        args_list.append("--gui")

    # Add workspace if different from default
    workspace = params.get('workspace', DEFAULT_WORKSPACE_DIR)
    if workspace != DEFAULT_WORKSPACE_DIR:
        args_list.extend(["--workspace", workspace])

    # Advanced parameters
    args_list.extend(["--land_use_block_size_m",
                     str(params.get("land_use_block_size_m", DEFAULT_LAND_USE_BLOCK_SIZE_M))])
    args_list.extend(
        ["--attractiveness", params.get("attractiveness", DEFAULT_ATTRACTIVENESS)])


    args_list.extend(
        ["--start_time_hour", str(params.get("start_time_hour", DEFAULT_START_TIME_HOUR))])
    args_list.extend(["--traffic_light_strategy",
                     params.get("traffic_light_strategy", DEFAULT_TRAFFIC_LIGHT_STRATEGY)])
    args_list.extend(
        ["--traffic_control", params.get("traffic_control", DEFAULT_TRAFFIC_CONTROL)])
    args_list.extend(["--bottleneck-detection-interval",
                     str(params.get("bottleneck_detection_interval", DEFAULT_BOTTLENECK_DETECTION_INTERVAL))])
    args_list.extend(
        ["--atlcs-interval", str(params.get("atlcs_interval", DEFAULT_ATLCS_INTERVAL))])
    args_list.extend(
        ["--tree-method-interval", str(params.get("tree_method_interval", DEFAULT_TREE_METHOD_INTERVAL))])

    if params.get("tree_method_sample"):
        args_list.extend(
            ["--tree_method_sample", params["tree_method_sample"]])

    # Parse using existing parser
    parser = create_argument_parser()
    return parser.parse_args(args_list)


def save_configuration(params: Dict[str, Any], name: str):
    """Save current configuration as a preset."""
    if 'saved_configs' not in st.session_state:
        st.session_state.saved_configs = {}

    st.session_state.saved_configs[name] = params.copy()
    st.success(f"Configuration '{name}' saved!")


def load_configuration(name: str) -> Dict[str, Any]:
    """Load a saved configuration preset."""
    if 'saved_configs' not in st.session_state:
        st.session_state.saved_configs = {}

    return st.session_state.saved_configs.get(name, {})


def run_simulation(params: Dict[str, Any]):
    """Execute the simulation with given parameters."""
    try:
        # Convert parameters to args namespace
        args = convert_params_to_args(params)

        # Validate arguments
        validate_arguments(args)

        # Show validation success
        st.success("✅ Parameters validated successfully!")

        # Create progress indicators and stop button
        progress_bar = st.progress(PROGRESS_START)
        status_text = st.empty()

        # Stop button container
        stop_container = st.empty()

        # Initialize simulation state
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            st.session_state.simulation_stopped = False

        # Set simulation as running
        st.session_state.simulation_running = True
        st.session_state.simulation_stopped = False

        # Setup logging for GUI
        setup_logging()
        logger = get_logger(__name__)

        # Show stop button
        with stop_container:
            if st.button("🛑 Stop Simulation", type="secondary"):
                st.session_state.simulation_stopped = True
                st.warning("⚠️ Stopping simulation...")
                return

        status_text.text("🔄 Creating pipeline...")
        progress_bar.progress(PROGRESS_PIPELINE_CREATED)

        # Check for stop signal
        if st.session_state.simulation_stopped:
            st.warning("⚠️ Simulation stopped by user")
            return

        # Create and execute pipeline
        pipeline = PipelineFactory.create_pipeline(args)

        status_text.text("🚀 Executing simulation...")
        progress_bar.progress(PROGRESS_EXECUTION_STARTED)

        # Check for stop signal
        if st.session_state.simulation_stopped:
            st.warning("⚠️ Simulation stopped by user")
            return

        # Execute pipeline
        status_text.text("🚀 Executing simulation...")
        progress_bar.progress(PROGRESS_EXECUTION_RUNNING)

        # Execute pipeline (this will take time)
        pipeline.execute()

        status_text.text("✅ Simulation completed!")
        progress_bar.progress(PROGRESS_COMPLETED)

        # Check if simulation was stopped
        if st.session_state.simulation_stopped:
            st.warning("⚠️ Simulation stopped by user")
            return

        progress_bar.progress(PROGRESS_COMPLETED)
        status_text.text("✅ Simulation completed successfully!")

        # Clear stop button
        stop_container.empty()

        # Reset simulation state
        st.session_state.simulation_running = False

        # Show results
        st.success("🎉 Simulation completed!")

        # Display output files
        OutputDisplay.show_results()

    except ValidationError as e:
        st.session_state.simulation_running = False
        st.error(f"❌ Validation Error: {e}")
    except Exception as e:
        st.session_state.simulation_running = False
        st.error(f"❌ Execution Error: {e}")
        st.exception(e)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DBPS - Decentralised Bottleneck Prioritization Simulation",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={}
    )

    # Main title
    st.title("🚦 DBPS - Decentralised Bottleneck Prioritization Simulation")
    st.markdown(
        "*Intelligent traffic simulation with dynamic signal optimization*")

    # Configuration Management Header
    with st.expander("⚙️ Configuration Management", expanded=False):
        col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

        with col1:
            config_name = st.text_input(
                "Configuration Name", placeholder="e.g., High Traffic Test", key="config_name_input")

        with col2:
            save_config = st.button("💾 Save", key="save_config_btn")

        with col3:
            if 'saved_configs' in st.session_state and st.session_state.saved_configs:
                selected_config = st.selectbox(
                    "Saved Configurations",
                    [""] + list(st.session_state.saved_configs.keys()),
                    key="config_selector"
                )
            else:
                selected_config = st.selectbox("Saved Configurations", [
                                               "No saved configurations"], disabled=True)

        with col4:
            load_config = st.button("📁 Load", key="load_config_btn", disabled=not (
                selected_config and selected_config != "No saved configurations"))

    # Main parameter collection
    st.header("📋 Simulation Parameters")

    # Collect all parameters
    params = ParameterWidgets.collect_all_parameters()

    # Handle configuration save/load
    if save_config and config_name:
        save_configuration(params, config_name)
        st.rerun()

    if load_config and selected_config and selected_config != "":
        loaded_params = load_configuration(selected_config)
        if loaded_params:
            # Store loaded params in session state to update widgets
            for key, value in loaded_params.items():
                if f"param_{key}" in st.session_state:
                    st.session_state[f"param_{key}"] = value
            st.success(f"✅ Configuration '{selected_config}' loaded!")
            st.rerun()

    # Parameter summary
    with st.expander("📊 Parameter Summary"):
        st.json(params)

    # Command line preview
    st.header("💻 Command Line Preview")
    command_line = generate_command_line(params)
    st.code(command_line, language="bash")

    # # Copy button for command line
    # if st.button("📋 Copy Command to Clipboard"):
    #     st.write(
    #         "Copy the command above and paste it in your terminal to run the simulation directly.")

    # Execution section
    st.header("🚀 Execution")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Preparing simulation..."):
                run_simulation(params)

    with col2:
        if st.button("🔍 Validate Only", use_container_width=True):
            try:
                args = convert_params_to_args(params)
                validate_arguments(args)
                st.success("✅ Validation passed!")
            except ValidationError as e:
                st.error(f"❌ Validation failed: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col3:
        if st.button("📋 Export Config", use_container_width=True):
            config_json = json.dumps(params, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=config_json,
                file_name="dbps_config.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
