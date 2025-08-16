#!/usr/bin/env python3
"""
OSM Import Module

This module handles importing OpenStreetMap data and converting it to SUMO network files,
replacing the synthetic grid generation functionality while maintaining compatibility
with the existing pipeline and Tree Method algorithm.
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import shutil

from ..config import OSMConfig

logger = logging.getLogger(__name__)


def import_osm_network(osm_file_path: str, output_prefix: str = "workspace/grid") -> None:
    """
    Simple function to import OSM network and convert to SUMO files.

    Args:
        osm_file_path: Path to OSM file
        output_prefix: Prefix for output files (default: "workspace/grid")
    """
    try:
        # Use netconvert to convert OSM directly to the desired prefix
        # Add additional options for better OSM processing
        cmd = [
            "netconvert",
            "--osm-files", osm_file_path,
            "--plain-output-prefix", output_prefix,
            "--geometry.remove",  # Remove unnecessary geometry points
            "--roundabouts.guess",  # Detect roundabouts
            "--junctions.join",  # Join nearby junctions
            "--tls.guess-signals",  # Guess traffic signals
            "--tls.discard-simple",  # Remove simple traffic lights
            "--ramps.guess",  # Detect highway ramps
            "--junctions.corner-detail", "5",  # Junction corner detail
            "--output.street-names",  # Preserve street names
            "--output.original-names",  # Keep original OSM names
            "--keep-edges.by-vclass", "passenger",  # Filter for passenger vehicles
            "--remove-edges.by-vclass", "pedestrian"  # Remove pedestrian-only edges
        ]

        print(f"Converting OSM file to SUMO network: {osm_file_path}")
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        print(f"Netconvert stdout: {result.stdout}")
        print(f"Netconvert stderr: {result.stderr}")

        if result.returncode == 0:
            print(
                f"Successfully converted OSM to SUMO network files with prefix: {output_prefix}")
        else:
            raise RuntimeError(f"netconvert failed: {result.stderr}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert OSM file: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "netconvert not found. Please ensure SUMO is installed and in your PATH.")


def generate_network_from_osm(osm_file_path: str) -> None:
    """
    Generate SUMO network from OSM file with proper file management

    This function combines OSM import with the file management logic needed
    to integrate with the existing pipeline that expects files in workspace/ directory.

    Args:
        osm_file_path: Path to OSM file
    """
    logger.info(f"Generating SUMO network from OSM file: {osm_file_path}")

    # Call existing import_osm_network function
    import_osm_network(osm_file_path, "workspace/grid")
    logger.info("Successfully imported OSM network.")

    # Move OSM files to expected locations in workspace/ directory
    grid_dir = Path("workspace/grid")
    if grid_dir.exists():
        logger.info("Moving OSM files to expected locations...")

        # Move files from workspace/grid/osm_network.* to workspace/grid.*
        files_moved = 0
        for file_pattern in ["*.nod.xml", "*.edg.xml", "*.con.xml", "*.tll.xml"]:
            for src_file in grid_dir.glob(file_pattern):
                # Extract the file extension part (e.g., "nod.xml" from "osm_network.nod.xml")
                suffix = src_file.name.split(
                    ".", 1)[1] if "." in src_file.name else src_file.suffix
                dst_file = Path("workspace") / f"grid.{suffix}"

                shutil.move(str(src_file), str(dst_file))
                logger.info(f"Moved {src_file} to {dst_file}")
                files_moved += 1

        # Clean up empty grid directory
        if not list(grid_dir.iterdir()):
            grid_dir.rmdir()
            logger.info("Cleaned up empty grid directory")

        logger.info(
            f"Successfully moved {files_moved} network files to workspace/ directory")
    else:
        logger.warning(
            "Grid directory not found - files may not have been generated correctly")


class OSMImporter:
    """
    Handles importing OSM data and converting to SUMO network files
    """

    def __init__(self, osm_file_path: str, output_dir: str = "data", config: Optional[OSMConfig] = None):
        """
        Initialize OSM importer

        Args:
            osm_file_path: Path to OSM file
            output_dir: Directory for output files
            config: OSM configuration object
        """
        self.osm_file_path = Path(osm_file_path)
        self.output_dir = Path(output_dir)
        self.config = config or OSMConfig()

        # Validate inputs
        if not self.osm_file_path.exists():
            raise FileNotFoundError(f"OSM file not found: {osm_file_path}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output file paths
        self.output_prefix = self.output_dir / "osm_network"

    def validate_osm_coverage(self) -> Dict[str, any]:
        """
        Validate OSM file has required elements for traffic simulation

        Returns:
            dict: Validation results with statistics and warnings
        """
        logger.info(f"Validating OSM file: {self.osm_file_path}")

        try:
            # Parse OSM file
            tree = ET.parse(self.osm_file_path)
            root = tree.getroot()

            # Count elements
            nodes = root.findall('node')
            ways = root.findall('way')

            # Count highway elements
            highway_ways = []
            traffic_signals = []

            for way in ways:
                highway_tag = way.find("tag[@k='highway']")
                if highway_tag is not None:
                    highway_ways.append(highway_tag.get('v'))

            for node in nodes:
                highway_tag = node.find("tag[@k='highway']")
                if highway_tag is not None and highway_tag.get('v') == 'traffic_signals':
                    traffic_signals.append(node.get('id'))

            # Extract bounding box
            bounds = root.find('bounds')
            if bounds is not None:
                bbox = {
                    'minlat': float(bounds.get('minlat')),
                    'minlon': float(bounds.get('minlon')),
                    'maxlat': float(bounds.get('maxlat')),
                    'maxlon': float(bounds.get('maxlon'))
                }
            else:
                # Calculate from nodes
                lats = [float(node.get('lat'))
                        for node in nodes if node.get('lat')]
                lons = [float(node.get('lon'))
                        for node in nodes if node.get('lon')]
                bbox = {
                    'minlat': min(lats),
                    'minlon': min(lons),
                    'maxlat': max(lats),
                    'maxlon': max(lons)
                }

            # Count highway types
            highway_counts = {}
            for hw_type in highway_ways:
                highway_counts[hw_type] = highway_counts.get(hw_type, 0) + 1

            # Calculate area (approximate)
            lat_dist = (bbox['maxlat'] - bbox['minlat']) * 111000  # meters
            lon_dist = (bbox['maxlon'] - bbox['minlon']) * 111000 * \
                abs(bbox['minlat'] + bbox['maxlat']) / \
                2 * 0.017453  # account for latitude
            area_km2 = (lat_dist * lon_dist) / 1_000_000

            validation_results = {
                'valid': True,
                'total_nodes': len(nodes),
                'total_ways': len(ways),
                'highway_ways': len(highway_ways),
                'traffic_signals': len(traffic_signals),
                'highway_types': highway_counts,
                'bounding_box': bbox,
                'area_km2': area_km2,
                'warnings': []
            }

            # Add warnings for potential issues
            if len(highway_ways) < 10:
                validation_results['warnings'].append(
                    "Very few highway ways found - network may be incomplete")

            if len(traffic_signals) == 0:
                validation_results['warnings'].append(
                    "No traffic signals found - signals will be generated")

            if area_km2 > 5:
                validation_results['warnings'].append(
                    f"Large area ({area_km2:.2f} km²) - simulation may be slow")

            # Check for required highway types
            required_types = set(self.config.filter_highway_types)
            found_types = set(highway_counts.keys())
            missing_types = required_types - found_types

            if missing_types:
                validation_results['warnings'].append(
                    f"Missing highway types: {missing_types}")

            logger.info(
                f"OSM validation complete: {len(highway_ways)} highway ways, {len(traffic_signals)} signals")
            return validation_results

        except Exception as e:
            logger.error(f"OSM validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'warnings': [f"Failed to parse OSM file: {e}"]
            }

    def import_to_sumo_files(self, output_prefix: Optional[str] = None) -> Dict[str, Path]:
        """
        Convert OSM file to raw SUMO network files using netconvert

        Args:
            output_prefix: Custom output prefix (optional)

        Returns:
            dict: Paths to generated SUMO files
        """
        if output_prefix:
            prefix = Path(output_prefix)
        else:
            prefix = self.output_prefix

        logger.info(
            f"Converting OSM to SUMO files: {self.osm_file_path} -> {prefix}")

        # Build netconvert command
        cmd = [
            'netconvert',
            '--osm-files', str(self.osm_file_path),
            '--plain-output-prefix', str(prefix),
            '--output-file', str(prefix) + '.net.xml'
        ]

        # Add OSM-specific options
        if self.config.filter_highway_types:
            # Create highway type filter
            highway_filter = ','.join(self.config.filter_highway_types)
            cmd.extend(['--keep-edges.by-vclass', 'passenger'])
            cmd.extend(['--remove-edges.by-vclass', 'pedestrian'])

        # Add additional netconvert options for better results
        cmd.extend([
            '--geometry.remove',  # Remove unnecessary geometry points
            '--roundabouts.guess',  # Detect roundabouts
            '--junctions.join',  # Join nearby junctions
            '--tls.guess-signals',  # Guess traffic signals
            '--tls.discard-simple',  # Remove simple traffic lights
            '--ramps.guess',  # Detect highway ramps
            '--junctions.corner-detail', '5',  # Junction corner detail
            '--output.street-names',  # Preserve street names
            '--output.original-names'  # Keep original OSM names
        ])

        try:
            # Run netconvert
            logger.info(f"Running netconvert command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                logger.info(f"netconvert output: {result.stdout}")
            if result.stderr:
                logger.warning(f"netconvert warnings: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"netconvert failed: {e}")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Failed to convert OSM to SUMO: {e}")

        # Check that files were created
        expected_files = {
            'nod': prefix.with_suffix('.nod.xml'),
            'edg': prefix.with_suffix('.edg.xml'),
            'con': prefix.with_suffix('.con.xml'),
            'tll': prefix.with_suffix('.tll.xml'),
            'net': prefix.with_suffix('.net.xml')
        }

        created_files = {}
        for file_type, file_path in expected_files.items():
            if file_path.exists():
                created_files[file_type] = file_path
                logger.info(f"Created {file_type} file: {file_path}")
            else:
                logger.warning(f"Expected file not created: {file_path}")

        if len(created_files) < 3:  # At least nod, edg, con should exist
            raise RuntimeError(
                f"Insufficient files created. Expected at least nod, edg, con files.")

        logger.info(
            f"Successfully converted OSM to SUMO files: {len(created_files)} files created")
        return created_files

    def get_network_statistics(self, sumo_files: Dict[str, Path]) -> Dict[str, any]:
        """
        Extract statistics from generated SUMO network files

        Args:
            sumo_files: Dictionary of SUMO file paths

        Returns:
            dict: Network statistics
        """
        stats = {
            'nodes': 0,
            'edges': 0,
            'connections': 0,
            'traffic_lights': 0,
            'edge_lengths': [],
            'junction_types': {}
        }

        try:
            # Parse node file
            if 'nod' in sumo_files:
                nod_tree = ET.parse(sumo_files['nod'])
                nodes = nod_tree.findall('.//node')
                stats['nodes'] = len(nodes)

                # Count junction types
                for node in nodes:
                    junction_type = node.get('type', 'unknown')
                    stats['junction_types'][junction_type] = \
                        stats['junction_types'].get(junction_type, 0) + 1

            # Parse edge file
            if 'edg' in sumo_files:
                edg_tree = ET.parse(sumo_files['edg'])
                edges = edg_tree.findall('.//edge')
                stats['edges'] = len(edges)

                # Extract edge lengths
                for edge in edges:
                    try:
                        # Calculate length from shape or use numLanes as proxy
                        shape = edge.get('shape', '')
                        if shape:
                            # Simple length estimation from shape points
                            # This is approximate - real calculation would use coordinate geometry
                            coords = shape.split()
                            if len(coords) >= 2:
                                # Estimate length (very rough approximation)
                                stats['edge_lengths'].append(
                                    len(coords) * 50)  # ~50m per segment
                    except:
                        pass

            # Parse connection file
            if 'con' in sumo_files:
                con_tree = ET.parse(sumo_files['con'])
                connections = con_tree.findall('.//connection')
                stats['connections'] = len(connections)

            # Parse traffic light file
            if 'tll' in sumo_files:
                tll_tree = ET.parse(sumo_files['tll'])
                traffic_lights = tll_tree.findall('.//tlLogic')
                stats['traffic_lights'] = len(traffic_lights)

        except Exception as e:
            logger.warning(f"Failed to extract network statistics: {e}")

        # Calculate summary statistics
        if stats['edge_lengths']:
            stats['avg_edge_length'] = sum(
                stats['edge_lengths']) / len(stats['edge_lengths'])
            stats['min_edge_length'] = min(stats['edge_lengths'])
            stats['max_edge_length'] = max(stats['edge_lengths'])

        logger.info(f"Network statistics: {stats['nodes']} nodes, {stats['edges']} edges, "
                    f"{stats['connections']} connections, {stats['traffic_lights']} traffic lights")

        return stats


def import_osm_network(osm_file_path: str, output_dir: str = "data",
                       config: Optional[OSMConfig] = None) -> Dict[str, Path]:
    """
    Convenience function to import OSM network

    Args:
        osm_file_path: Path to OSM file
        output_dir: Output directory
        config: OSM configuration

    Returns:
        dict: Paths to generated SUMO files
    """
    importer = OSMImporter(osm_file_path, output_dir, config)

    # Validate OSM file
    validation = importer.validate_osm_coverage()
    if not validation['valid']:
        raise ValueError(
            f"Invalid OSM file: {validation.get('error', 'Unknown error')}")

    # Print warnings
    for warning in validation['warnings']:
        logger.warning(warning)

    # Convert to SUMO files
    sumo_files = importer.import_to_sumo_files()

    # Get statistics
    stats = importer.get_network_statistics(sumo_files)
    logger.info(f"OSM import complete: {stats}")

    return sumo_files


if __name__ == "__main__":
    # Test with the existing OSM file
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))

    logging.basicConfig(level=logging.INFO)

    osm_file = "src/osm/export.osm"
    if os.path.exists(osm_file):
        try:
            files = import_osm_network(osm_file)
            print(f"Successfully imported OSM network: {files}")
        except Exception as e:
            print(f"Import failed: {e}")
    else:
        print(f"OSM file not found: {osm_file}")
