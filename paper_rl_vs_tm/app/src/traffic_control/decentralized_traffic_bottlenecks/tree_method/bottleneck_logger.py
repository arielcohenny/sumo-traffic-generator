"""
Bottleneck CSV Logger for Tree Method Traffic Control

This module provides CSV logging functionality to track vehicle counts per edge
at regular intervals (every 6 minutes).
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Optional


class BottleneckCSVLogger:
    """
    Logs vehicle counts per edge to CSV file for analysis.

    CSV Format:
    - step: Simulation step number (in seconds)
    - link: Edge name/ID
    - num_vehicles: Number of vehicles on that edge
    """

    def __init__(self, output_dir: str, enabled: bool = True):
        """
        Initialize the bottleneck logger.

        Args:
            output_dir: Directory where CSV file will be written
            enabled: Whether logging is enabled (for performance)
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.csv_file_path = None
        self.csv_file = None
        self.csv_writer = None

        if self.enabled:
            self._initialize_csv()

    def _initialize_csv(self):
        """Initialize the CSV file and writer."""
        # Resolve to absolute path
        self.output_dir = self.output_dir.resolve()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV file path
        self.csv_file_path = self.output_dir / "bottleneck_events.csv"

        # Open file and create CSV writer
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        self.csv_writer.writerow(['step', 'link', 'num_vehicles'])

        # Flush to ensure header is written
        self.csv_file.flush()

    def log_vehicle_counts(self, step: int, graph):
        """
        Log vehicle counts for all edges in the network.

        Args:
            step: Current simulation step (in seconds)
            graph: Graph object containing all network links
        """
        if not self.enabled:
            return

        # Iterate through all links and log their vehicle counts
        for link in graph.all_links:
            vehicle_count = link.current_vehicle_count

            # Write one row per edge
            self.csv_writer.writerow([
                step,
                link.edge_name,
                vehicle_count
            ])

        # Flush to ensure data is written
        self.csv_file.flush()

    def close(self):
        """Close the CSV file."""
        if self.enabled and self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
