"""
Shared statistics parsing and formatting utilities.

This module provides centralized functions for parsing SUMO statistics XML
and formatting statistics output for both CLI and GUI interfaces, ensuring
consistency across both interfaces.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def parse_sumo_statistics_file(statistics_file_path: str) -> Optional[Dict]:
    """
    Parse SUMO statistics XML file and return structured data.
    
    This function contains the exact working XML parsing logic from the GUI.
    
    Args:
        statistics_file_path: Path to the sumo_statistics.xml file
        
    Returns:
        Dictionary containing parsed statistics, or None if parsing fails
        {
            'loaded': int,
            'inserted': int, 
            'running': int,
            'waiting': int,
            'arrived': int,
            'completion_rate': float,
            'insertion_rate': float,
            'trip_count': int,
            'avg_duration': float,
            'avg_waiting_time': float,
            'avg_time_loss': float,
            'throughput': float,
            'sim_hours': float
        }
    """
    statistics_file = Path(statistics_file_path)
    
    if not statistics_file.exists():
        return None

    try:
        tree = ET.parse(statistics_file)
        root = tree.getroot()

        # Extract vehicle counts
        vehicles_elem = root.find('.//vehicles')
        vehicle_stats = root.find('.//vehicleTripStatistics')
        
        if vehicles_elem is not None and vehicle_stats is not None:
            # Vehicle counts from <vehicles> element
            loaded = int(vehicles_elem.get('loaded', 0))
            inserted = int(vehicles_elem.get('inserted', 0))
            running = int(vehicles_elem.get('running', 0))
            waiting = int(vehicles_elem.get('waiting', 0))
            
            # Calculate arrived (completed trips)
            arrived = loaded - running - waiting
            
            # Calculate completion rate
            completion_rate = (arrived / loaded * 100) if loaded > 0 else 0
            
            # Extract performance metrics from <vehicleTripStatistics>
            trip_count = int(vehicle_stats.get('count', 0))
            avg_duration = float(vehicle_stats.get('duration', 0))
            avg_waiting_time = float(vehicle_stats.get('waitingTime', 0))
            avg_time_loss = float(vehicle_stats.get('timeLoss', 0))
            
            # Calculate efficiency metrics
            insertion_rate = (inserted / loaded * 100) if loaded > 0 else 0
            
            # Throughput: vehicles that completed per hour
            sim_hours = 1200 / 3600  # 1200 seconds = 0.33 hours
            throughput = arrived / sim_hours if sim_hours > 0 else 0
            
            return {
                'loaded': loaded,
                'inserted': inserted,
                'running': running,
                'waiting': waiting,
                'arrived': arrived,
                'completion_rate': completion_rate,
                'insertion_rate': insertion_rate,
                'trip_count': trip_count,
                'avg_duration': avg_duration,
                'avg_waiting_time': avg_waiting_time,
                'avg_time_loss': avg_time_loss,
                'throughput': throughput,
                'sim_hours': sim_hours
            }
        else:
            return None
            
    except Exception:
        return None


def format_cli_statistics_output(stats: Dict, traffic_control_method: str, total_simulation_steps: int = None) -> List[str]:
    """
    Format statistics for CLI logging output.
    
    Args:
        stats: Dictionary containing parsed statistics
        traffic_control_method: Traffic control method used
        total_simulation_steps: Total simulation steps (optional)
        
    Returns:
        List of formatted log messages
    """
    if not stats:
        return [
            "=== SIMULATION COMPLETED ===",
            "Statistics file not found or could not be parsed.",
            f"Traffic control method: {traffic_control_method}"
        ]
    
    messages = [
        "=== SIMULATION COMPLETED ==="
    ]
    
    # Add total simulation steps if provided
    if total_simulation_steps is not None:
        messages.append(f"Total simulation steps: {total_simulation_steps}")
    
    # Core vehicle statistics
    messages.extend([
        f"Vehicles loaded: {stats['loaded']}",
        f"Vehicles inserted: {stats['inserted']}",
        f"Vehicles still running: {stats['running']}",
        f"Vehicles waiting: {stats['waiting']}",
        f"Vehicles arrived: {stats['arrived']}",
        f"Completion rate: {stats['completion_rate']:.1f}%",
        f"Insertion rate: {stats['insertion_rate']:.1f}%"
    ])
    
    # Performance metrics
    messages.extend([
        f"Average duration: {stats['avg_duration']:.1f}s",
        f"Average waiting time: {stats['avg_waiting_time']:.1f}s", 
        f"Average time loss: {stats['avg_time_loss']:.1f}s",
        f"Throughput: {stats['throughput']:.0f} veh/h",
        f"Trip statistics count: {stats['trip_count']}"
    ])
    
    # Traffic control method
    messages.append(f"Traffic control method: {traffic_control_method}")
    
    return messages