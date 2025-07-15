# src/validate/validate_simulation.py
"""
Runtime validation for Tree Method decentralized traffic control algorithm integration.

This module provides inline verification during simulation to ensure:
1. Tree Method objects are properly initialized and integrated
2. The algorithm executes correctly during runtime
3. Traffic light decisions are reasonable and responsive
4. No crashes or infinite loops in the algorithm
"""

from __future__ import annotations
from typing import Dict, Any
import traci

try:
    from .errors import ValidationError
except ImportError:
    class ValidationError(RuntimeError):
        pass

__all__ = [
    "verify_nimrod_integration_setup",
    "verify_algorithm_runtime_behavior",
]


# ---------------------------------------------------------------------------
#  Tree Method Integration Setup Verification
# ---------------------------------------------------------------------------

def verify_nimrod_integration_setup(
    tree_data: Any,
    run_config: Any, 
    network_data: Any,
    graph: Any,
    seconds_in_cycle: float
) -> None:
    """Verify Tree Method objects initialized correctly after setup phase.
    
    This function validates the Tree Method integration setup by checking:
    1. Tree data structure is valid
    2. Run configuration has expected parameters
    3. Network data loaded correctly from JSON
    4. Graph building completed successfully
    5. Cycle time calculation is reasonable
    
    Called once after Tree Method setup, before simulation starts.
    """
    
    # 1 ── validate tree_data structure ───────────────────────────────────────
    if tree_data is None:
        raise ValidationError("Tree data is None - load_tree() may have failed")
    
    # Check if tree_data has expected structure (adapt based on actual structure)
    if hasattr(tree_data, '__len__') and len(tree_data) == 0:
        raise ValidationError("Tree data is empty - no network topology loaded")
    
    # 2 ── validate run_config parameters ─────────────────────────────────────
    if run_config is None:
        raise ValidationError("Run config is None - load_tree() may have failed")
    
    # Check for expected algorithm configuration
    if not hasattr(run_config, 'algo_type'):
        raise ValidationError("Run config missing algo_type - Tree Method algorithm type not specified")
    
    # 3 ── validate network_data object ───────────────────────────────────────
    if network_data is None:
        raise ValidationError("Network data is None - Network object creation failed")
    
    # Check network has edges and junctions
    if not hasattr(network_data, 'edges_list'):
        raise ValidationError("Network data missing edges_list - JSON format may be incorrect")
    
    if not hasattr(network_data, 'junctions_dict'):
        raise ValidationError("Network data missing junctions_dict - JSON format may be incorrect")
    
    # Validate edges_list is populated
    if not network_data.edges_list or len(network_data.edges_list) == 0:
        raise ValidationError("Network edges_list is empty - no edges loaded from JSON")
    
    # Validate junctions_dict is populated  
    if not network_data.junctions_dict or len(network_data.junctions_dict) == 0:
        raise ValidationError("Network junctions_dict is empty - no junctions loaded from JSON")
    
    # 4 ── validate graph object building ─────────────────────────────────────
    if graph is None:
        raise ValidationError("Graph object is None - Graph creation failed")
    
    # Check that graph.build() was called and completed
    # (Adapt these checks based on actual Graph object structure)
    if hasattr(graph, 'junctions_dict') and not graph.junctions_dict:
        raise ValidationError("Graph junctions_dict is empty - graph.build() may have failed")
    
    # 5 ── validate cycle time calculation ────────────────────────────────────
    if seconds_in_cycle <= 0:
        raise ValidationError(f"Invalid cycle time: {seconds_in_cycle}s - must be positive")
    
    if seconds_in_cycle > 300:  # More than 5 minutes seems unreasonable
        raise ValidationError(f"Cycle time {seconds_in_cycle}s seems too long - check calculation")
    
    if seconds_in_cycle < 30:  # Less than 30 seconds seems too short
        raise ValidationError(f"Cycle time {seconds_in_cycle}s seems too short - check calculation")
    
    # 6 ── validate data consistency ──────────────────────────────────────────
    # Check that network data and graph have consistent junction counts
    network_junction_count = len(network_data.junctions_dict)
    
    # Try to get graph junction count (adapt based on actual Graph structure)
    if hasattr(graph, 'junctions_dict'):
        graph_junction_count = len(graph.junctions_dict)
        if network_junction_count != graph_junction_count:
            raise ValidationError(
                f"Junction count mismatch: Network has {network_junction_count}, "
                f"Graph has {graph_junction_count}")
    
    return


# ---------------------------------------------------------------------------
#  Runtime Algorithm Verification
# ---------------------------------------------------------------------------

def verify_algorithm_runtime_behavior(
    current_time: int,
    phase_map: Dict[str, str],
    graph: Any,
    verification_frequency: int
) -> None:
    """Inline verification during simulation runtime.
    
    This function validates the algorithm behavior during simulation by checking:
    1. Traffic light states are being generated
    2. Phase decisions are reasonable
    3. Algorithm is responsive to traffic conditions
    4. No crashes or infinite loops in algorithm execution
    
    Called periodically during simulation based on verification_frequency.
    """
    
    # Only verify at specified frequency
    if current_time % verification_frequency != 0:
        return
    
    # 1 ── validate phase_map generation ──────────────────────────────────────
    if phase_map is None:
        raise ValidationError(f"Phase map is None at time {current_time} - algorithm may have crashed")
    
    if not isinstance(phase_map, dict):
        raise ValidationError(f"Phase map is not a dictionary at time {current_time} - unexpected format")
    
    if len(phase_map) == 0:
        raise ValidationError(f"Phase map is empty at time {current_time} - no traffic light states generated")
    
    # 2 ── validate traffic light state format ────────────────────────────────
    for tls_id, state in phase_map.items():
        if not isinstance(state, str):
            raise ValidationError(
                f"Traffic light {tls_id} state is not a string at time {current_time}: {type(state)}")
        
        if not state:
            raise ValidationError(f"Traffic light {tls_id} has empty state at time {current_time}")
        
        # Check state contains only valid SUMO traffic light characters
        valid_chars = set('rygGY')  # red, yellow, green (lowercase/uppercase)
        if not all(c in valid_chars for c in state):
            raise ValidationError(
                f"Traffic light {tls_id} has invalid state '{state}' at time {current_time}")
    
    # 3 ── validate algorithm responsiveness ──────────────────────────────────
    # Check that traffic lights are actually changing over time
    # We'll track this by checking if all lights have the same state (suspicious)
    all_states = list(phase_map.values())
    if len(set(all_states)) == 1 and len(all_states) > 1:
        # All traffic lights have identical states - might indicate algorithm stuck
        if current_time > verification_frequency * 3:  # Give algorithm time to start
            raise ValidationError(
                f"All traffic lights have identical state '{all_states[0]}' at time {current_time} - "
                f"algorithm may be stuck")
    
    # 4 ── validate traffic conditions response ───────────────────────────────
    # Use TraCI to get basic traffic data and check algorithm responsiveness
    try:
        # Get list of all traffic lights from SUMO
        sumo_tls_ids = traci.trafficlight.getIDList()
        
        # Check that our phase_map covers all SUMO traffic lights
        missing_tls = set(sumo_tls_ids) - set(phase_map.keys())
        if missing_tls:
            raise ValidationError(
                f"Algorithm missing traffic light states for: {missing_tls} at time {current_time}")
        
        extra_tls = set(phase_map.keys()) - set(sumo_tls_ids)
        if extra_tls:
            raise ValidationError(
                f"Algorithm has extra traffic light states for: {extra_tls} at time {current_time}")
        
        # Check for basic traffic responsiveness
        # If there are vehicles waiting and lights haven't changed recently, that might be suspicious
        total_waiting_vehicles = 0
        for tls_id in sumo_tls_ids:
            # Get controlled lanes for this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                try:
                    waiting_count = traci.lane.getLastStepHaltingNumber(lane)
                    total_waiting_vehicles += waiting_count
                except traci.TraCIException:
                    # Lane might not exist or be accessible, skip
                    continue
        
        # If there are many waiting vehicles but we're early in simulation, 
        # check that algorithm is making decisions (not stuck in one state)
        if total_waiting_vehicles > len(sumo_tls_ids) * 2 and current_time > 60:
            # Significant waiting - check if algorithm is trying to respond
            # This is a basic heuristic; could be enhanced with more sophisticated checks
            pass  # For now, just ensuring no crashes - could add more logic here
            
    except traci.TraCIException as e:
        raise ValidationError(f"TraCI error during validation at time {current_time}: {e}")
    
    # 5 ── validate graph object consistency ───────────────────────────────────
    if graph is None:
        raise ValidationError(f"Graph object is None at time {current_time} - may have been corrupted")
    
    # Check that graph object still has expected structure
    if hasattr(graph, 'junctions_dict') and not graph.junctions_dict:
        raise ValidationError(f"Graph junctions_dict is empty at time {current_time} - object corrupted")
    
    return