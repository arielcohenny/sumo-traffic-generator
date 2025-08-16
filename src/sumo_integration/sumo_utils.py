import subprocess
from pathlib import Path
from typing import List, Optional
from src.config import CONFIG

# Convert the generated network files to the final .net.xml format
# Rebuild with fresh internals + connections


def rebuild_network() -> None:
    # Convert network with connections and traffic light logic files to preserve manual lane assignments
    # This ensures netconvert uses our geometric angle-based lane assignments instead of generating its own
    basic_cmd = [
        "netconvert",
        "--node-files",       str(CONFIG.network_nod_file),
        "--edge-files",       str(CONFIG.network_edg_file),
        "--connection-files", str(CONFIG.network_con_file),
        "--tllogic-files",    str(CONFIG.network_tll_file),
        # "--junctions.join=true", # Joins junctions that are close to each other (recommended for OSM import); default: false
        "--output-file",      str(CONFIG.network_file)
    ]

    try:
        result = subprocess.run(basic_cmd, check=True,
                                capture_output=True, text=True)
        print("Network conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Network conversion failed: {e.stderr}")
        raise


def build_sumo_command(
    config_file: str,
    step_length: float,
    additional_args: List[str] = None,
    sumo_binary: str = "sumo-gui"
) -> List[str]:
    """
    Build the SUMO command-line arguments for TraCI or batch runs.
    """
    cmd = [sumo_binary, "-c", config_file, "--step-length", str(step_length)]
    if additional_args:
        cmd.extend(additional_args)
    return cmd


def generate_sumo_conf_file(
    config_file,
    network_file,
    route_file: Optional[str] = None,
    zones_file: Optional[str] = None,
) -> str:
    """
    Create a SUMO configuration file (.sumocfg).

    :param config_file: File path to write the configuration
    :param network_file: Path to the network .net.xml file
    :param route_file: Optional path to the route .xml file
    """
    print("Creating SUMO configuration file.")
    try:
        net_name = Path(network_file).name
        route_name = Path(route_file).name if route_file else None
        zones_name = Path(zones_file).name if zones_file else None

        # Build the configuration content step by step to avoid f-string backslash issues
        config_content = f"<configuration>\n    <input>\n        <net-file value=\"{net_name}\"/>\n"

        if route_name:
            config_content += f"        <route-files value=\"{route_name}\"/>\n"

        if zones_name:
            config_content += f"        <additional-files value=\"{zones_name}\"/>\n"

        config_content += "    </input>\n    <time>\n        <begin value=\"0\"/>\n        <end value=\"3600\"/>\n    </time>\n</configuration>"

        with open(config_file, "w") as f:
            f.write(config_content)
        print("SUMO configuration file created successfully.")
        return str(config_file)
    except Exception as e:
        print(f"Error creating SUMO configuration file: {e}")
        exit(1)


def execute_network_rebuild(args) -> None:
    """Execute network rebuild."""
    import logging
    from src.validate.validate_network import verify_rebuild_network
    from src.validate.errors import ValidationError

    logger = logging.getLogger(__name__)

    rebuild_network()
    try:
        verify_rebuild_network()
    except ValidationError as ve:
        logger.error(f"Failed to rebuild the network: {ve}")
        raise


def execute_config_generation(args) -> None:
    """Execute SUMO configuration generation."""
    import logging
    from src.validate.validate_network import verify_generate_sumo_conf_file
    from src.validate.errors import ValidationError

    logger = logging.getLogger(__name__)

    sumo_cfg_path = generate_sumo_conf_file(
        CONFIG.config_file,
        CONFIG.network_file,
        route_file=CONFIG.routes_file,
        zones_file=CONFIG.zones_file,
    )
    try:
        verify_generate_sumo_conf_file()
    except ValidationError as ve:
        logger.error(f"SUMO configuration validation failed: {ve}")
        raise


def update_sumo_config_paths() -> None:
    """Update SUMO config file to reference our file naming convention."""
    import logging
    import xml.etree.ElementTree as ET

    logger = logging.getLogger(__name__)

    tree = ET.parse(CONFIG.config_file)
    root = tree.getroot()

    # Update file paths to match our naming
    for input_elem in root.findall('.//input'):
        net_file = input_elem.find('net-file')
        if net_file is not None:
            net_file.set('value', 'grid.net.xml')

        route_files = input_elem.find('route-files')
        if route_files is not None:
            route_files.set('value', 'vehicles.rou.xml')

    # Save updated config
    tree.write(CONFIG.config_file, encoding='utf-8', xml_declaration=True)
    logger.info("Updated SUMO config file paths")


def override_end_time_from_config(args) -> None:
    """Extract end time from SUMO config and override CLI argument."""
    import logging
    import xml.etree.ElementTree as ET

    logger = logging.getLogger(__name__)

    try:
        tree = ET.parse(CONFIG.config_file)
        root = tree.getroot()

        # Find the end time in the config
        for time_elem in root.findall('.//time'):
            end_elem = time_elem.find('end')
            if end_elem is not None:
                config_end_time = int(end_elem.get('value'))

                logger.info(
                    f"Found SUMO config end time: {config_end_time} seconds")
                logger.info(f"CLI end time was: {args.end_time} seconds")

                # Override the CLI argument with the config value
                args.end_time = config_end_time

                logger.info(
                    f"Overriding end time to match SUMO config: {config_end_time} seconds")
                return

        # If no end time found in config, warn but continue with CLI value
        logger.warning(
            f"No end time found in SUMO config, using CLI value: {args.end_time}")

    except (ET.ParseError, ValueError, AttributeError) as e:
        logger.warning(f"Error parsing end time from SUMO config: {e}")
        logger.warning(f"Continuing with CLI end time: {args.end_time}")
