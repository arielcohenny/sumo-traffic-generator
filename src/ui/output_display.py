"""
Output display components for DBPS GUI.

This module handles the display of simulation results and generated files.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from src.utils.statistics import parse_sumo_statistics_file


class OutputDisplay:
    """Handles display of simulation outputs and results."""

    @staticmethod
    def show_results():
        """Display simulation results and generated files."""
        # Calculate absolute path to workspace directory
        project_root = Path(__file__).parent.parent.parent
        workspace_path = project_root / "workspace"

        if not workspace_path.exists():
            st.warning("No workspace directory found. Run a simulation first.")
            return

        st.header("📊 Simulation Results")

        # Results summary (before Generated Files)
        OutputDisplay._show_results_summary(workspace_path)

        # File overview
        OutputDisplay._show_file_overview(workspace_path)

        # Network information
        OutputDisplay._show_network_info(workspace_path)

        # Vehicle and route information
        OutputDisplay._show_vehicle_info(workspace_path)

        # Zone information
        OutputDisplay._show_zone_info(workspace_path)

        # SUMO configuration
        OutputDisplay._show_sumo_config(workspace_path)

    @staticmethod
    def _show_file_overview(workspace_path: Path):
        """Show overview of generated files."""
        with st.expander("📁 Generated Files", expanded=True):
            files = list(workspace_path.glob("*"))

            if not files:
                st.info("No files generated yet.")
                return

            # Create file summary table
            file_data = []
            for file_path in files:
                if file_path.is_file():
                    stat = file_path.stat()
                    file_data.append({
                        "File": file_path.name,
                        "Size (KB)": round(stat.st_size / 1024, 2),
                        "Type": file_path.suffix,
                        "Modified": pd.Timestamp.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })

            if file_data:
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)

                # Download buttons for key files
                col1, col2, col3 = st.columns(3, gap="large")

                with col1:
                    net_file = workspace_path / "grid.net.xml"
                    if net_file.exists():
                        with open(net_file, 'r') as f:
                            st.download_button(
                                "📥 Network File",
                                data=f.read(),
                                file_name="network.net.xml",
                                mime="application/xml"
                            )

                with col2:
                    routes_file = workspace_path / "vehicles.rou.xml"
                    if routes_file.exists():
                        with open(routes_file, 'r') as f:
                            st.download_button(
                                "📥 Routes File",
                                data=f.read(),
                                file_name="vehicles.rou.xml",
                                mime="application/xml"
                            )

                with col3:
                    config_file = workspace_path / "grid.sumocfg"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            st.download_button(
                                "📥 SUMO Config",
                                data=f.read(),
                                file_name="simulation.sumocfg",
                                mime="application/xml"
                            )

    @staticmethod
    def _show_network_info(workspace_path: Path):
        """Display network information."""
        with st.expander("🏗️ Network Information"):
            net_file = workspace_path / "grid.net.xml"

            if not net_file.exists():
                st.info("Network file not found.")
                return

            try:
                # Parse network XML
                tree = ET.parse(net_file)
                root = tree.getroot()

                # Count network elements
                edges = root.findall('.//edge')
                junctions = root.findall('.//junction')
                connections = root.findall('.//connection')
                traffic_lights = root.findall('.//tlLogic')

                # Display metrics with wider columns to prevent truncation
                col1, col2, col3, col4 = st.columns(
                    [1, 1, 1.2, 1.3], gap="large")

                with col1:
                    st.metric("Edges", len(edges))

                with col2:
                    st.metric("Junctions", len(junctions))

                with col3:
                    st.metric("Connections", len(connections))

                with col4:
                    st.metric("Traffic Lights", len(traffic_lights))

                # Network bounds
                if edges:
                    st.subheader("Network Bounds")
                    x_coords, y_coords = [], []

                    for edge in edges:
                        for lane in edge.findall('.//lane'):
                            shape = lane.get('shape', '')
                            if shape:
                                coords = shape.split()
                                for coord in coords:
                                    if ',' in coord:
                                        x, y = coord.split(',')
                                        x_coords.append(float(x))
                                        y_coords.append(float(y))

                    if x_coords and y_coords:
                        bounds_data = {
                            "Dimension": ["X (West-East)", "Y (South-North)"],
                            "Minimum": [min(x_coords), min(y_coords)],
                            "Maximum": [max(x_coords), max(y_coords)],
                            "Range": [max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
                        }

                        bounds_df = pd.DataFrame(bounds_data)
                        st.dataframe(bounds_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error parsing network file: {e}")

    @staticmethod
    def _show_vehicle_info(workspace_path: Path):
        """Display vehicle and route information."""
        with st.expander("🚗 Vehicle & Route Information"):
            routes_file = workspace_path / "vehicles.rou.xml"

            if not routes_file.exists():
                st.info("Routes file not found.")
                return

            try:
                # Parse routes XML
                tree = ET.parse(routes_file)
                root = tree.getroot()

                # Count vehicle types and vehicles
                vehicle_types = root.findall('.//vType')
                vehicles = root.findall('.//vehicle')
                routes = root.findall('.//route')

                # Display metrics
                col1, col2, col3 = st.columns(3, gap="large")

                with col1:
                    st.metric("Vehicle Types", len(vehicle_types))

                with col2:
                    st.metric("Total Vehicles", len(vehicles))

                with col3:
                    st.metric("Routes", len(routes))

                # Vehicle type breakdown
                if vehicle_types:
                    st.subheader("Vehicle Types")
                    vtype_data = []
                    for vtype in vehicle_types:
                        vtype_data.append({
                            "Type": vtype.get('id', 'Unknown'),
                            "Length (m)": vtype.get('length', 'N/A'),
                            "Max Speed (m/s)": vtype.get('maxSpeed', 'N/A'),
                            "Acceleration (m/s²)": vtype.get('accel', 'N/A'),
                            "Deceleration (m/s²)": vtype.get('decel', 'N/A')
                        })

                    if vtype_data:
                        vtype_df = pd.DataFrame(vtype_data)
                        st.dataframe(vtype_df, use_container_width=True)

                # Vehicle distribution by type
                if vehicles:
                    st.subheader("Vehicle Distribution")
                    type_counts = {}
                    departure_times = []

                    for vehicle in vehicles:
                        vtype = vehicle.get('type', 'Unknown')
                        type_counts[vtype] = type_counts.get(vtype, 0) + 1

                        depart = vehicle.get('depart', '0')
                        try:
                            departure_times.append(float(depart))
                        except ValueError:
                            pass

                    # Type distribution chart with graceful error handling
                    if type_counts:
                        type_df = pd.DataFrame(list(type_counts.items()), columns=[
                                               'Vehicle Type', 'Count'])

                        try:
                            # Try to use plotly for colored charts
                            import plotly.express as px

                            # Define colors for vehicle types
                            vehicle_colors = {
                                'passenger': '#28a745',    # Green
                                'commercial': '#fd7e14',   # Orange
                                'public': '#007bff'        # Blue
                            }

                            # Create plotly chart
                            fig = px.bar(type_df,
                                         x='Vehicle Type',
                                         y='Count',
                                         color='Vehicle Type',
                                         color_discrete_map=vehicle_colors,
                                         title="Vehicle Type Distribution")

                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        except ImportError:
                            # Fallback to basic Streamlit chart if plotly not available
                            st.bar_chart(type_df.set_index('Vehicle Type'))
                        except Exception:
                            # Fallback for any other chart rendering issues
                            st.warning(
                                "Chart rendering temporarily unavailable, showing basic chart")
                            st.bar_chart(type_df.set_index('Vehicle Type'))

                    # Departure time distribution
                    if departure_times:
                        st.subheader("Departure Time Distribution")
                        departure_df = pd.DataFrame(
                            {'Departure Time (seconds)': departure_times})
                        st.histogram = st.line_chart(
                            departure_df['Departure Time (seconds)'].value_counts().sort_index())

            except Exception as e:
                st.error(f"Error parsing routes file: {e}")

    @staticmethod
    def _show_zone_info(workspace_path: Path):
        """Display zone information."""
        with st.expander("🗺️ Zone Information"):
            zones_file = workspace_path / "zones.poly.xml"

            if not zones_file.exists():
                st.info("Zones file not found.")
                return

            # Define system colors for zone types
            zone_colors = {
                "Residential": "#FFA500",           # Orange
                "Employment": "#8B0000",           # Dark red
                "Public Buildings": "#000080",     # Dark blue
                "Mixed": "#FFFF00",                # Yellow
                "Entertainment/Retail": "#006400",  # Dark green
                "Public Open Space": "#90EE90"     # Light green
            }

            try:
                # Parse zones XML
                tree = ET.parse(zones_file)
                root = tree.getroot()

                polygons = root.findall('.//poly')

                if polygons:
                    st.metric("Total Zones", len(polygons))

                    # Zone type breakdown with colors
                    zone_types = {}
                    zone_colors_actual = {}

                    for poly in polygons:
                        zone_type = poly.get('type', 'Unknown')
                        zone_color = poly.get(
                            'color', '#808080')  # Default gray
                        zone_types[zone_type] = zone_types.get(
                            zone_type, 0) + 1
                        zone_colors_actual[zone_type] = zone_color

                    if zone_types:
                        st.subheader("Zone Type Distribution")

                        # Create clean zone data
                        total_zones = sum(zone_types.values())
                        zone_data = []
                        for zone_type, count in zone_types.items():
                            percentage = (count / total_zones *
                                          100) if total_zones > 0 else 0
                            zone_data.append({
                                'Zone Type': zone_type,
                                'Count': count,
                                'Percentage': f"{percentage:.1f}%"
                            })

                        # Display clean table
                        zone_df = pd.DataFrame(zone_data)
                        st.dataframe(
                            zone_df, use_container_width=True, hide_index=True)

                        # Show colored bar chart with graceful error handling
                        try:
                            # Try to use plotly for colored charts
                            import plotly.express as px

                            # Use zone colors for chart
                            zone_color_map = {}
                            for zone_type in zone_types.keys():
                                zone_color_map[zone_type] = zone_colors.get(
                                    zone_type, '#808080')

                            fig = px.bar(zone_df,
                                         x='Zone Type',
                                         y='Count',
                                         color='Zone Type',
                                         color_discrete_map=zone_color_map,
                                         title="Zone Distribution")

                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        except ImportError:
                            # Fallback to basic Streamlit chart if plotly not available
                            st.bar_chart(zone_df.set_index(
                                'Zone Type')['Count'])
                        except Exception:
                            # Fallback for any other chart rendering issues
                            st.warning(
                                "Chart rendering temporarily unavailable, showing basic chart")
                            st.bar_chart(zone_df.set_index(
                                'Zone Type')['Count'])

                else:
                    st.info("No zones found in file.")

            except Exception as e:
                st.error(f"Error parsing zones file: {e}")

    @staticmethod
    def _show_sumo_config(workspace_path: Path):
        """Display SUMO configuration information."""
        with st.expander("⚙️ SUMO Configuration"):
            config_file = workspace_path / "grid.sumocfg"

            if not config_file.exists():
                st.info("SUMO configuration file not found.")
                return

            try:
                # Parse SUMO config XML
                tree = ET.parse(config_file)
                root = tree.getroot()

                # Display configuration sections
                for section in root:
                    if section.tag in ['input', 'time', 'processing', 'output']:
                        st.subheader(f"{section.tag.title()} Configuration")

                        config_data = []
                        for option in section:
                            config_data.append({
                                "Option": option.tag.replace('-', '_'),
                                "Value": option.get('value', 'N/A')
                            })

                        if config_data:
                            config_df = pd.DataFrame(config_data)
                            st.dataframe(config_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error parsing SUMO config file: {e}")

    @staticmethod
    def _show_results_summary(workspace_path: Path):
        """Display simulation results summary from SUMO statistics output."""
        with st.expander("📈 Results Summary", expanded=True):
            # Parse SUMO statistics file using shared parser
            statistics_file = workspace_path / "sumo_statistics.xml"
            stats = parse_sumo_statistics_file(str(statistics_file))

            if not stats:
                st.warning(
                    "Statistics file not found or could not be parsed. Run a simulation first.")
                return

            # Display metrics in columns using parsed statistics
            col1, col2, col3 = st.columns(3, gap="large")

            with col1:
                st.metric("Vehicles Loaded", stats['loaded'])
                st.metric("Vehicles Inserted", stats['inserted'])

            with col2:
                st.metric("Completion Rate",
                          f"{stats['completion_rate']:.1f}%")
                st.metric("Vehicles Arrived", stats['arrived'])

            with col3:
                st.metric("Vehicles Running", stats['running'])
                st.metric("Vehicles Waiting", stats['waiting'])

            # Additional performance metrics
            st.subheader("Performance Metrics")
            perf_col1, perf_col2, perf_col3 = st.columns(3, gap="large")

            with perf_col1:
                st.metric("Average Duration", f"{stats['avg_duration']:.1f}s",
                          help="Average trip duration for completed vehicles only. Excludes vehicles still running or waiting.")

            with perf_col2:
                st.metric("Average Waiting Time", f"{stats['avg_waiting_time']:.1f}s",
                          help="Average time spent waiting (at traffic lights, in queues) for completed vehicles only.")

            with perf_col3:
                st.metric("Time Loss", f"{stats['avg_time_loss']:.1f}s",
                          help="Average extra time compared to free-flow travel for completed vehicles. Shows traffic delay impact.")

            # Efficiency metrics
            efficiency_col1, efficiency_col2, efficiency_col3 = st.columns(
                3, gap="large")

            with efficiency_col1:
                st.metric("Insertion Rate", f"{stats['insertion_rate']:.1f}%")

            with efficiency_col2:
                st.metric("Throughput", f"{stats['throughput']:.0f} veh/h",
                          help="Vehicles that completed their trips per hour.")

            with efficiency_col3:
                st.metric("Trip Statistics Count", stats['trip_count'],
                          help="Number of completed trip records in SUMO statistics. Should match 'Vehicles Arrived' but comes from different XML element.")

            st.success("✅ Statistics loaded successfully!")

            # Show error log if present and non-empty
            error_log_file = workspace_path / "sumo_errors.log"
            if error_log_file.exists() and error_log_file.stat().st_size > 0:
                with st.expander("⚠️ Errors and Warnings", expanded=False):
                    try:
                        with open(error_log_file, 'r') as f:
                            error_content = f.read()
                            # Truncate if too long
                            if len(error_content) > 2000:
                                error_content = error_content[:2000] + \
                                    "\n\n... (truncated)"
                            st.code(error_content, language="text")
                    except Exception as e:
                        st.error(f"Could not read error log: {e}")

    @staticmethod
    def show_live_simulation_status():
        """Display live simulation status (for future implementation)."""
        st.info("Live simulation monitoring will be implemented in Phase 2")

    @staticmethod
    def show_performance_metrics():
        """Display performance metrics (for future implementation)."""
        st.info("Performance metrics visualization will be implemented in Phase 3")
