"""
ATLCS (Adaptive Traffic Light Control System) implementation.
Implements T6/T7 research objectives by extending Tree Method with dynamic pricing.
"""

from typing import Any

import traci
from ..tree_method.controller import TreeMethodController
from .config import CONFIG as ATLCS_CONFIG


class ATLCSController(TreeMethodController):
    """
    Adaptive Traffic Light Control System implementing T6/T7 research objectives.
    Extends TreeMethodController to add dynamic pricing and enhanced bottleneck detection.
    """

    def __init__(self, args: Any):
        super().__init__(args)  # Initialize Tree Method foundation

        # ATLCS-specific components (initialized once, used throughout simulation)
        self.detector = None      # → Used in update() every 60s
        self.pricing = None       # → Used in update() every 30s
        self.coordinator = None      # → Used in update() every step

        # Configuration from CLI arguments
        # T6 calculations interval from CLI argument (in seconds)
        self.t6_interval = args.t6_interval
        # T7 pricing updates interval from CLI argument (in seconds)
        self.t7_interval = args.t7_interval

        # Control state tracking - junctions controlled by Tree Method
        self.tree_method_controlled_junctions = set()

        # Configuration intervals set

        # Track extensions per step to avoid multiple extensions of same phase
        self.extensions_this_step = set()  # {(tls_id, phase)} per step

    def initialize(self) -> None:
        """Initialize ATLCS components after Tree Method foundation."""
        # Initialize Tree Method foundation (REUSED)
        super().initialize()  # Existing Graph, Network, etc.

        # Add ATLCS-specific components
        from .enhancements.detector import BottleneckDetector
        from .pricing.engine import PricingEngine
        from .pricing.demand_supply_coordinator import DemandSupplyCoordinator

        # Initialize components that will be used in update()
        self.detector = BottleneckDetector(self.graph, self.network_data)
        self.pricing = PricingEngine()
        self.coordinator = DemandSupplyCoordinator(
            self.detector, self.pricing)

        # ATLCS initialization complete

    def update(self, step: int) -> None:
        """ATLCS simulation step - extends Tree Method with T6/T7 capabilities."""

        # Clear extensions tracker for this step
        self.extensions_this_step.clear()

        # PHASE 1: Tree Method foundation (REUSED - no duplication)
        super().update(step)  # Existing Tree Method calculations, traffic light control

        # PHASE 2: T6 Enhanced bottleneck detection (NEW)
        if self._is_calculation_time(step):
            bottleneck_data = self.detector.detect_and_prioritize(
                step, self.graph, self.iteration_trees)
            self.current_bottleneck_data = bottleneck_data  # Store for T7 usage

        # PHASE 3: T7 Dynamic pricing for traffic light control (NEW)
        pricing_updates = None
        if self._is_pricing_update_time(step):
            if hasattr(self, 'current_bottleneck_data') and self.current_bottleneck_data:
                pricing_updates = self.pricing.calculate_dynamic_prices(
                    self.current_bottleneck_data, step
                )
                self._apply_pricing_to_traffic_lights(pricing_updates, step)

        # PHASE 4: Demand-Supply coordination (NEW)
        if hasattr(self, 'current_bottleneck_data') and self.current_bottleneck_data:
            self.coordinator.coordinate(
                step,
                self.current_bottleneck_data,
                pricing_updates
            )

    def _is_calculation_time(self, step: int) -> bool:
        """Check if it's time for T6 bottleneck detection."""
        return step % self.t6_interval == 0

    def _is_pricing_update_time(self, step: int) -> bool:
        """Check if it's time for T7 pricing updates."""
        return step % self.t7_interval == 0

    def _apply_pricing_to_traffic_lights(self, pricing_updates, step: int) -> int:
        """Apply dynamic pricing to traffic light control via TraCI."""
        signal_adjustments = 0

        try:
            # Apply pricing-based priority to traffic lights
            for edge_id, price in pricing_updates.edge_prices.items():
                try:
                    # CRITICAL FIX: Map Tree Method edge index to actual SUMO edge name
                    actual_edge_name = self._map_tree_method_edge_to_sumo_edge(
                        edge_id)
                    if not actual_edge_name:
                        continue

                    # Find traffic light controlling this edge
                    tls_id = self._find_traffic_light_for_edge(
                        actual_edge_name)
                    if tls_id:
                        # High price = high congestion = needs longer green time
                        if price > ATLCS_CONFIG.HIGH_PRIORITY_THRESHOLD:  # High priority threshold
                            extension_seconds = ATLCS_CONFIG.HIGH_PRIORITY_EXTENSION_SEC  # Significant extension
                            self._extend_green_time_for_edge(
                                tls_id, actual_edge_name, extension_seconds, step)
                            signal_adjustments += 1
                        elif price > ATLCS_CONFIG.MEDIUM_PRIORITY_THRESHOLD:  # Medium priority threshold
                            extension_seconds = ATLCS_CONFIG.MEDIUM_PRIORITY_EXTENSION_SEC   # Moderate extension
                            self._extend_green_time_for_edge(
                                tls_id, actual_edge_name, extension_seconds, step)
                            signal_adjustments += 1

                except traci.TraCIException:
                    continue  # Skip traffic lights that can't be modified
                except Exception:
                    continue

        except Exception as e:
            self.logger.warning(f"ATLCS signal control failed: {e}")

        return signal_adjustments

    def _find_traffic_light_for_edge(self, edge_id) -> str:
        """Find traffic light that controls the given edge."""
        try:
            # Convert edge_id to string if it's not already
            edge_id_str = str(edge_id)

            # Get all traffic lights
            tls_ids = traci.trafficlight.getIDList()

            # For each traffic light, check if it controls lanes from this edge
            for tls_id in tls_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(
                    tls_id)
                for lane in controlled_lanes:
                    if lane.startswith(edge_id_str):
                        return tls_id

            # If no exact match, try more flexible matching (edge_id anywhere in lane name)
            for tls_id in tls_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(
                    tls_id)
                for lane in controlled_lanes:
                    if edge_id_str in lane:
                        return tls_id

        except Exception:
            pass

        return None

    def _map_tree_method_edge_to_sumo_edge(self, tree_method_edge_id):
        """Map Tree Method's internal edge ID to actual SUMO edge name."""
        try:
            # Tree Method edge_id is likely an index into self.graph.all_links
            edge_index = int(tree_method_edge_id)

            if hasattr(self, 'graph') and self.graph and hasattr(self.graph, 'all_links'):
                if 0 <= edge_index < len(self.graph.all_links):
                    link = self.graph.all_links[edge_index]
                    # Use the link's edge_name or link_id as the actual SUMO edge name
                    actual_edge_name = getattr(
                        link, 'edge_name', None) or getattr(link, 'link_id', None)
                    return actual_edge_name

        except (ValueError, AttributeError):
            pass

        return None

    def can_modify_junction(self, tls_id: str) -> bool:
        """Check if ATLCS is allowed to modify this junction's traffic lights."""
        # Check if Tree Method controls this junction (using parent class controlled_junctions)
        if hasattr(self, 'controlled_junctions'):
            if tls_id in self.controlled_junctions:
                return False

        # Also check the local tracking set for safety
        if hasattr(self, 'tree_method_controlled_junctions'):
            if tls_id in self.tree_method_controlled_junctions:
                return False

        # Default: ATLCS can modify (no Tree Method control detected)
        return True

    def _extend_green_time_for_edge(self, tls_id: str, edge_id: str, extension_seconds: int, step: int) -> None:
        """Extend green time for the traffic light phase that serves the given edge."""
        try:
            # Check if ATLCS can modify this junction
            if not self.can_modify_junction(tls_id):
                return
            # Get current traffic light state
            current_phase = traci.trafficlight.getPhase(tls_id)
            current_duration = traci.trafficlight.getNextSwitch(
                tls_id) - traci.simulation.getTime()

            # Check if current phase serves the edge
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            edge_lanes = [
                lane for lane in controlled_lanes if lane.startswith(edge_id)]

            if edge_lanes:
                # Extend current phase if it serves our edge
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                current_state = program.phases[current_phase].state

                # Check if any of our edge's lanes have green light (state 'G' or 'g')
                edge_has_green = any(current_state[controlled_lanes.index(lane)] in ['G', 'g']
                                     for lane in edge_lanes if lane in controlled_lanes)

                if edge_has_green and current_duration > 0:
                    # Check if we already extended this phase in this step
                    phase_key = (tls_id, current_phase)
                    if phase_key in self.extensions_this_step:
                        return  # Already extended this phase in this step

                    # Extend current phase duration
                    new_duration = current_duration + extension_seconds
                    traci.trafficlight.setPhaseDuration(tls_id, new_duration)

                    # Mark as extended in this step
                    self.extensions_this_step.add(phase_key)

                    # Update shared variable so Tree Method uses extended duration
                    if hasattr(self, 'current_phase_durations') and tls_id in self.current_phase_durations:
                        self.current_phase_durations[tls_id][current_phase] = new_duration

        except Exception:
            # Silently fail if traffic light manipulation fails
            pass
