"""
Bottleneck CSV Logger for Tree Method Traffic Control

This module provides CSV logging functionality to track the development of bottlenecks
and phase changes in the Tree Method algorithm over time.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Optional


class BottleneckCSVLogger:
    """
    Logs bottleneck information and phase changes to CSV file for analysis.

    CSV Format:
    - step: Simulation step number
    - junction_id: Traffic light/junction ID
    - phase_index: Current phase index
    - duration_sec: Phase duration in seconds
    - phase_cost: Cost associated with this phase
    - bottleneck_links: Colon-separated list of bottleneck link IDs
    - bottleneck_scores: Colon-separated list of bottleneck scores (speeds)
    - vehicle_counts: Colon-separated list of vehicle counts for each bottleneck
    - active_links_in_phase: Colon-separated list of links active in current phase
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
        self.csv_writer.writerow([
            'step',
            'junction_id',
            'phase_index',
            'duration_sec',
            'phase_cost',
            'bottleneck_links',
            'bottleneck_scores',
            'vehicle_counts',
            'active_links_in_phase'
        ])

        # Flush to ensure header is written
        self.csv_file.flush()

    def log_phase_change(
        self,
        step: int,
        junction_id: str,
        phase_index: int,
        duration_sec: int,
        phase_cost: float,
        bottleneck_links: List[str],
        bottleneck_scores: List[float],
        vehicle_counts: List[int],
        active_links_in_phase: List[str]
    ):
        """
        Log a phase change event with bottleneck information.

        Args:
            step: Current simulation step
            junction_id: Traffic light ID
            phase_index: Phase index being activated
            duration_sec: Duration of the phase in seconds
            phase_cost: Cost metric for this phase
            bottleneck_links: List of link IDs that are bottlenecks
            bottleneck_scores: List of bottleneck scores (speeds in km/h)
            vehicle_counts: List of vehicle counts for each bottleneck link
            active_links_in_phase: List of link IDs active in this phase
        """
        if not self.enabled:
            return

        # Format lists as colon-separated strings
        bottleneck_links_str = ':'.join(bottleneck_links) if bottleneck_links else ''
        bottleneck_scores_str = ':'.join(f'{score:.2f}' for score in bottleneck_scores) if bottleneck_scores else ''
        vehicle_counts_str = ':'.join(str(count) for count in vehicle_counts) if vehicle_counts else ''
        active_links_str = ':'.join(active_links_in_phase) if active_links_in_phase else ''

        # Write row
        self.csv_writer.writerow([
            step,
            junction_id,
            phase_index,
            duration_sec,
            f'{phase_cost:.2f}',
            bottleneck_links_str,
            bottleneck_scores_str,
            vehicle_counts_str,
            active_links_str
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
