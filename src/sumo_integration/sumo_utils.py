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


def run_sumo(
    config_file: str,
    step_length: float,
    zones_file: str,
    sumo_binary: str = "sumo-gui"
) -> None:
    """
    Run a single SUMO simulation using subprocess.

    :param config_file: Path to the SUMO .sumocfg file
    :param step_length: Simulation step length in seconds
    :param zones_file: Path to additional files (e.g., zones) to include
    :param sumo_binary: SUMO executable ("sumo" or "sumo-gui")
    """
    # Build the command using the shared helper
    cmd = build_sumo_command(
        config_file,
        step_length,
        additional_args=["--additional-files", zones_file],
        sumo_binary=sumo_binary
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUMO simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running SUMO: {e.stderr}")
        exit(1)


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
