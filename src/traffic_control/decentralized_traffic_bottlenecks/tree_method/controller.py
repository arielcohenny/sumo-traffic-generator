"""
Tree Method traffic controller implementation.
"""

from typing import Any
from pathlib import Path

import traci
from src.config import CONFIG
from src.orchestration.traffic_controller import TrafficController


class TreeMethodController(TrafficController):
    """Traffic controller implementing Tree Method algorithm."""

    def __init__(self, args: Any):
        super().__init__(args)
        self.tree_data = None
        self.run_config = None
        self.network_data = None
        self.graph = None
        self.seconds_in_cycle = None
        self.iteration_trees = []

        # Shared variable for traffic light durations (Tree Method calculates, ATLCS can modify)
        self.current_phase_durations = {}  # {tl_id: {phase_index: duration_seconds}}

        # Junction control state - junctions with multi-edge trees that Tree Method controls
        self.controlled_junctions = set()

    def initialize(self) -> None:
        """Initialize Tree Method objects and data structures."""
        from .integration import load_tree
        from ..shared.classes.graph import Graph
        from ..shared.classes.network import Network
        from ..shared.classes.net_data_builder import build_network_json
        from src.validate.validate_simulation import verify_tree_method_integration_setup

        # Tree Method initialization started

        # Build network JSON for Tree Method
        json_file = Path(CONFIG.network_file).with_suffix(".json")
        build_network_json(CONFIG.network_file, json_file)

        # Load Tree Method objects with original configuration parameters
        from ..shared.enums import CostType, AlgoType
        self.tree_data, self.run_config = load_tree(
            net_file=CONFIG.network_file,
            # Match original: TREE_CURRENT_DIVIDED
            cost_type=CostType.TREE_CURRENT_DIVIDED,
            algo_type=AlgoType.BABY_STEPS,           # Match original: BABY_STEPS
            sumo_cfg=CONFIG.config_file
        )

        self.network_data = Network(json_file)

        self.graph = Graph(self.args.end_time)

        self.graph.build(self.network_data.edges_list,
                         self.network_data.junctions_dict)

        # Use explicit Tree Method interval (independent of traffic light cycles)
        self.tree_method_interval = getattr(
            self.args, 'tree_method_interval', CONFIG.TREE_METHOD_ITERATION_INTERVAL_SEC)

        # Tree Method timing configuration set

        # Verify Tree Method integration setup
        verify_tree_method_integration_setup(
            self.tree_data, self.run_config, self.network_data, self.graph, self.tree_method_interval)

        # Initialize T6 bottleneck detector for junction control decisions
        from ..atlcs.enhancements.detector import BottleneckDetector
        self.t6_detector = BottleneckDetector(self.graph, self.network_data)

    def update(self, step: int) -> None:
        """Update Tree Method traffic control at given step."""
        from ..shared.utils import is_calculation_time, calc_iteration_from_step
        from ..shared.enums import AlgoType, CostType
        from src.validate.validate_simulation import verify_algorithm_runtime_behavior

        # Tree Method: Calculation time check
        is_calc_time = is_calculation_time(step, self.tree_method_interval)

        # Tree Method calculation time check

        if is_calc_time:
            iteration = calc_iteration_from_step(
                step, self.tree_method_interval)

            if iteration > 0:  # Skip first iteration
                try:
                    # Perform Tree Method calculations
                    ended_iteration = iteration - 1

                    self.graph.calculate_iteration(
                        ended_iteration,
                        self.iteration_trees,
                        step,
                        self.tree_method_interval,
                        self.run_config.cost_type if self.run_config else CostType.TREE_CURRENT,
                        self.run_config.algo_type if self.run_config else AlgoType.BABY_STEPS
                    )

                    self.graph.calc_nodes_statistics(
                        ended_iteration, self.tree_method_interval)

                    # Populate shared variable with calculated phase durations
                    self._populate_shared_phase_durations()

                    # Update junction control state based on tree complexity
                    self._update_junction_control_state(step)

                except ZeroDivisionError as zde:
                    import traceback
                    self.logger.error(
                        f"Tree Method division by zero at step {step}: {zde}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                except Exception as e:
                    self.logger.error(
                        f"Tree Method calculation failed at step {step}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Tree Method traffic light updates using shared durations
        # Note: Tree Method baseline operation is ALWAYS allowed - coordination only affects ATLCS
        try:
            self._update_traffic_lights_with_shared_durations(
                step, self.tree_method_interval,
                self.run_config.algo_type if self.run_config else AlgoType.BABY_STEPS
            )
        except Exception as e:
            self.logger.warning(
                f"Tree Method traffic light update failed at step {step}: {e}")

        # Post-step data collection
        try:
            # Skip fill_link_in_step if there are None edges - data quality issue in Tree Method samples
            if hasattr(self.graph, 'all_links'):
                valid_links = [link for link in self.graph.all_links if link.edge_name !=
                               "None" and link.edge_name is not None]
                if len(valid_links) == len(self.graph.all_links):
                    self.graph.fill_link_in_step()

            self.graph.add_vehicles_to_step()
            self.graph.close_prev_vehicle_step(step)
            current_iteration = calc_iteration_from_step(
                step, self.tree_method_interval)
            self.graph.fill_head_iteration()
        except Exception as e:
            # Only print every 100 steps to avoid spam
            if step % 100 == 0:
                self.logger.warning(
                    f"Tree Method post-processing failed at step {step}: {e}")

        # Runtime validation (every 30 steps by default)
        if step % CONFIG.SIMULATION_VERIFICATION_FREQUENCY == 0:
            try:
                # Save phase data - get_traffic_lights_phases doesn't return data, just saves it
                if self.graph:
                    self.graph.get_traffic_lights_phases(step)

                # Get actual traffic light states from TraCI
                phase_map = {}
                try:
                    tls_ids = traci.trafficlight.getIDList()
                    for tls_id in tls_ids:
                        current_state = traci.trafficlight.getRedYellowGreenState(
                            tls_id)
                        phase_map[tls_id] = current_state
                except traci.TraCIException:
                    phase_map = {}  # Fallback to empty if TraCI fails

                verify_algorithm_runtime_behavior(
                    step, phase_map, self.graph, CONFIG.SIMULATION_VERIFICATION_FREQUENCY
                )
            except Exception as ve:
                self.logger.warning(
                    f"Algorithm runtime validation failed at step {step}: {ve}")

    def _populate_shared_phase_durations(self) -> None:
        """Populate shared variable with Tree Method calculated phase durations."""
        if not hasattr(self, 'graph') or not self.graph:
            return

        try:
            # Clear existing durations
            self.current_phase_durations.clear()

            # Populate from Tree Method calculated phases
            for node_id in self.graph.tl_node_ids:
                node = self.graph.all_nodes[node_id]
                if node and hasattr(node, 'phases') and hasattr(node, 'tl'):
                    tl_id = node.tl
                    if tl_id:
                        # Store each phase duration
                        phase_durations = {}
                        for phase_index, phase in enumerate(node.phases):
                            if hasattr(phase, 'duration'):
                                phase_durations[phase_index] = phase.duration

                        if phase_durations:
                            self.current_phase_durations[tl_id] = phase_durations

        except Exception as e:
            self.logger.error(
                f"Failed to populate shared phase durations: {e}")

    def _update_junction_control_state(self, step: int) -> None:
        """Update junction control state based on bottleneck tree detection."""
        try:
            # Clear current control state - recalculate based on bottlenecks
            self.controlled_junctions.clear()

            # Use existing T6 bottleneck detection to identify bottleneck trees
            bottleneck_data = self.t6_detector.detect_and_prioritize(
                step, self.graph, self.iteration_trees)

            # Find trees that contain bottleneck edges
            bottleneck_trees = set()
            if bottleneck_data.prioritized_bottlenecks:
                for i, bottleneck in enumerate(bottleneck_data.prioritized_bottlenecks):
                    # Map bottleneck edge to trees that contain it
                    bottleneck_edge_id = getattr(bottleneck, 'edge_id', None)

                    if bottleneck_edge_id is not None:
                        # Find which trees contain this bottleneck edge
                        for j, iter_tree in enumerate(self.iteration_trees):
                            # Access the actual CurrentLoadTree objects from all_trees_per_iteration
                            if hasattr(iter_tree, 'all_trees_per_iteration') and iter_tree.all_trees_per_iteration:
                                for current_tree in iter_tree.all_trees_per_iteration.items():
                                    if self._current_tree_contains_edge(current_tree, bottleneck_edge_id):
                                        bottleneck_trees.add(current_tree)

            # Lock all junctions from bottleneck trees
            total_junctions = 0
            for current_tree in enumerate(bottleneck_trees):
                affected_junctions = self._find_junctions_in_current_tree(
                    current_tree)
                self.controlled_junctions.update(affected_junctions)
                total_junctions += len(affected_junctions)

        except Exception as e:
            self.logger.error(f"Failed to update junction control state: {e}")

    def _find_junctions_in_tree(self, tree) -> set:
        """Find junctions affected by a tree structure."""
        affected_junctions = set()

        try:
            # Method 1: Extract from links
            if hasattr(tree, 'links') and tree.links:
                for link in tree.links:
                    if hasattr(link, 'to_junction'):
                        affected_junctions.add(link.to_junction)
                    if hasattr(link, 'from_junction'):
                        affected_junctions.add(link.from_junction)

            # Method 2: Extract from edges
            elif hasattr(tree, 'edges') and tree.edges:
                for edge in tree.edges:
                    if hasattr(edge, 'to_junction'):
                        affected_junctions.add(edge.to_junction)
                    if hasattr(edge, 'from_junction'):
                        affected_junctions.add(edge.from_junction)

            # Method 3: Extract from nodes
            elif hasattr(tree, 'nodes') and tree.nodes:
                for node in tree.nodes:
                    if hasattr(node, 'tl') and node.tl:
                        affected_junctions.add(node.tl)

        except Exception as e:
            self.logger.warning(f"Error extracting junctions from tree: {e}")

        return affected_junctions

    def _tree_contains_edge(self, tree, edge_id) -> bool:
        """Check if a tree contains a specific edge ID."""
        try:

            # Check trunk link
            if hasattr(tree, 'trunk_link_id') and tree.trunk_link_id == edge_id:
                return True

            # Check in tree branches/links
            if hasattr(tree, 'all_my_branches') and tree.all_my_branches:
                if edge_id in tree.all_my_branches:
                    return True

            # Check in tree links
            if hasattr(tree, 'links') and tree.links:
                for link in tree.links:
                    link_id = getattr(link, 'id', None)
                    link_link_id = getattr(link, 'link_id', None)
                    if link_id == edge_id:
                        return True
                    if link_link_id == edge_id:
                        return True
            return False

        except Exception as e:
            self.logger.warning(
                f"Error checking if tree contains edge {edge_id}: {e}")

        return False

    def _current_tree_contains_edge(self, current_tree, edge_id) -> bool:
        """Check if a CurrentLoadTree contains a specific edge ID."""
        try:

            # Check trunk link
            trunk_id = getattr(current_tree, 'trunk_link_id', None)
            if trunk_id == edge_id:
                return True

            # Check all branches
            branches = getattr(current_tree, 'all_my_branches', {})
            if branches:
                if edge_id in branches:
                    return True
            return False

        except Exception as e:
            self.logger.warning(
                f"Error checking if CurrentLoadTree contains edge {edge_id}: {e}")

        return False

    def _find_junctions_in_current_tree(self, current_tree) -> set:
        """Find junctions affected by a CurrentLoadTree structure."""
        affected_junctions = set()

        try:
            # For CurrentLoadTree, we need to map edges to junctions using the graph
            trunk_id = getattr(current_tree, 'trunk_link_id', None)
            branches = getattr(current_tree, 'all_my_branches', {})

            all_edge_ids = []
            if trunk_id is not None:
                all_edge_ids.append(trunk_id)
            if branches:
                all_edge_ids.extend(branches.keys())

            # Map edge IDs to junctions using different strategies
            if hasattr(self, 'graph') and self.graph and hasattr(self.graph, 'all_links'):
                for edge_id in all_edge_ids:
                    # Strategy 1: Direct index lookup
                    try:
                        if edge_id < len(self.graph.all_links):
                            link = self.graph.all_links[edge_id]

                            # Extract junction information from the link
                            to_node_name = getattr(link, 'to_node_name', None)
                            from_node_name = getattr(
                                link, 'from_node_name', None)

                            if to_node_name:
                                affected_junctions.add(to_node_name)
                            if from_node_name:
                                affected_junctions.add(from_node_name)
                    except Exception as e:
                        self.logger.warning(
                            f"Direct index mapping failed for edge {edge_id}: {e}")

                    # Strategy 2: Search by link_id attribute
                    try:
                        for link in enumerate(self.graph.all_links):
                            if hasattr(link, 'link_id') and link.link_id == edge_id:
                                to_node_name = getattr(
                                    link, 'to_node_name', None)
                                from_node_name = getattr(
                                    link, 'from_node_name', None)

                                if to_node_name:
                                    affected_junctions.add(to_node_name)
                                if from_node_name:
                                    affected_junctions.add(from_node_name)
                                break
                    except Exception as e:
                        self.logger.warning(
                            f"Link_id search failed for edge {edge_id}: {e}")

        except Exception as e:
            self.logger.warning(
                f"Error extracting junctions from CurrentLoadTree: {e}")

        return affected_junctions

    def _update_traffic_lights_with_shared_durations(self, step: int, seconds_in_cycle: int, algo_type) -> None:
        """Update traffic lights using shared durations (allows ATLCS modifications to persist)."""
        if not hasattr(self, 'graph') or not self.graph:
            return

        try:
            from ..shared.classes.node import define_tl_program
            from ..shared.enums import AlgoType
            import random

            for node_id in self.graph.tl_node_ids:
                node = self.graph.all_nodes[node_id]
                if not node or not hasattr(node, 'tl') or not node.tl:
                    continue

                inner_sec = step % seconds_in_cycle
                tl_id = node.tl

                # Tree Method always provides baseline traffic light optimization
                # Coordination control (self.controlled_junctions) is separate and only affects ATLCS blocking
                # Tree Method can ALWAYS set phases - coordination is only for preventing ATLCS interference

                # Handle RANDOM algorithm (from original logic)
                if algo_type == AlgoType.RANDOM.name:
                    from ..shared.config import MIN_PHASE_TIME
                    if inner_sec % MIN_PHASE_TIME == 0:
                        phase_inx = random.randint(0, len(node.phases) - 1)
                        define_tl_program(tl_id, phase_inx, MIN_PHASE_TIME)
                    continue

                # Use shared durations if available, otherwise fall back to node.phases
                if tl_id in self.current_phase_durations:
                    phase_durations = self.current_phase_durations[tl_id]

                    # Find which phase should be active at this time
                    secs_sum = 0
                    for phase_index in range(len(node.phases)):
                        if inner_sec == secs_sum:
                            # Use shared duration (may be modified by ATLCS)
                            shared_duration = phase_durations.get(phase_index)
                            fallback_duration = node.phases[phase_index].duration
                            duration = shared_duration if shared_duration is not None else fallback_duration

                            define_tl_program(
                                tl_id, phase_index, duration)
                            break
                        secs_sum += phase_durations.get(
                            phase_index, node.phases[phase_index].duration)
                else:
                    # Fallback to original logic if shared durations not available
                    secs_sum = 0
                    for phase_index, phase in enumerate(node.phases):
                        if inner_sec == secs_sum:
                            define_tl_program(
                                tl_id, phase_index, phase.duration)
                            break
                        secs_sum += phase.duration

        except Exception as e:
            self.logger.error(
                f"Failed to update traffic lights with shared durations: {e}")
            # Fallback to original method
            if hasattr(self.graph, 'update_traffic_lights'):
                self.graph.update_traffic_lights(
                    step, seconds_in_cycle, algo_type)

    def cleanup(self) -> None:
        """Clean up Tree Method resources and report Tree Method statistics."""
        try:
            if hasattr(self, 'graph') and self.graph:
                # Report Tree Method's own duration statistics
                if hasattr(self.graph, 'ended_vehicles_count') and self.graph.ended_vehicles_count > 0:
                    tree_avg_duration = self.graph.vehicle_total_time / self.graph.ended_vehicles_count
                    self.logger.info("=== TREE METHOD STATISTICS ===")
                    self.logger.info(
                        f"Tree Method - Vehicles completed: {self.graph.ended_vehicles_count}")
                    self.logger.info(
                        f"Tree Method - Total driving time: {self.graph.vehicle_total_time}")
                    self.logger.info(
                        f"Tree Method - Average duration: {tree_avg_duration:.2f} steps")

        except Exception as e:
            self.logger.error(f"Error in Tree Method cleanup: {e}")

        # Tree Method objects are cleaned up by garbage collection
        pass
