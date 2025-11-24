"""
Traffic analysis utilities for RL using Tree Method functions.
Provides sophisticated traffic engineering metrics without code duplication.
"""

from typing import Dict, List, Tuple
import traci
import logging
from collections import deque

# Import Tree Method traffic flow functions (no duplication)
from src.traffic_control.decentralized_traffic_bottlenecks.shared.config import (
    MAX_DENSITY, MIN_VELOCITY, ITERATION_TIME_MINUTES
)
from src.traffic_control.decentralized_traffic_bottlenecks.shared.classes.link import (
    Link, QmaxProperties
)

from .constants import (
    TREE_METHOD_MAX_DENSITY, TREE_METHOD_MIN_VELOCITY,
    MAX_TIME_LOSS_MINUTES, MAX_COST_PER_EDGE, MAX_FLOW_PER_LANE_PER_HOUR,
    MAX_VEHICLE_COUNT_PER_EDGE, MOVING_AVERAGE_WINDOW_SIZE,
    RL_DYNAMIC_EDGE_FEATURES_COUNT, RL_DYNAMIC_JUNCTION_FEATURES_COUNT,
    RL_DYNAMIC_NETWORK_FEATURES_COUNT,
    # Individual feature toggles
    ENABLE_EDGE_SPEED_RATIO, ENABLE_EDGE_DENSITY_SIMPLE, ENABLE_EDGE_FLOW_SIMPLE,
    ENABLE_EDGE_CONGESTION_FLAG, ENABLE_EDGE_NORMALIZED_DENSITY, ENABLE_EDGE_NORMALIZED_FLOW,
    ENABLE_EDGE_IS_BOTTLENECK, ENABLE_EDGE_NORMALIZED_TIME_LOSS, ENABLE_EDGE_NORMALIZED_COST,
    ENABLE_EDGE_SPEED_TREND,
    ENABLE_JUNCTION_PHASE_NORMALIZED, ENABLE_JUNCTION_DURATION_NORMALIZED,
    ENABLE_JUNCTION_INCOMING_FLOW, ENABLE_JUNCTION_OUTGOING_FLOW,
    ENABLE_JUNCTION_UPSTREAM_BOTTLENECKS, ENABLE_JUNCTION_DOWNSTREAM_BOTTLENECKS,
    ENABLE_NETWORK_BOTTLENECK_RATIO, ENABLE_NETWORK_COST_NORMALIZED,
    ENABLE_NETWORK_VEHICLES_NORMALIZED, ENABLE_NETWORK_AVG_SPEED_NORMALIZED,
    ENABLE_NETWORK_CONGESTION_RATIO
)


class RLTrafficAnalyzer:
    """Provides Tree Method traffic analysis for RL state collection."""

    # Enable debug by default
    def __init__(self, edge_ids: List[str], m: float, l: float, debug: bool = False):
        """Initialize with network topology."""
        self.edge_ids = edge_ids
        self.edge_links: Dict[str, Link] = {}
        self.speed_history: Dict[str, deque] = {}
        self.debug = False  # debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.observation_count = 0
        self.m = m
        self.l = l

        # self.logger.info(f"=== RL TRAFFIC ANALYZER INITIALIZATION ===")
        # self.logger.info(f"Total edge IDs provided: {len(edge_ids)}")
        # self.logger.info(
        #     f"Edge IDs: {edge_ids[:10]}{'...' if len(edge_ids) > 10 else ''}")
        # self.logger.info(f"Debug mode: {self.debug}")

        self._initialize_tree_method_links()

    def _initialize_tree_method_links(self):
        """Create Tree Method Link objects for each edge."""
        initialized_count = 0
        failed_count = 0

        for edge_id in self.edge_ids:
            if ':' in edge_id:  # Skip internal edges
                continue

            # Get edge properties from SUMO
            try:
                # Use lane 0 for edge length
                length = traci.lane.getLength(f"{edge_id}_0")
                lanes = traci.edge.getLaneNumber(edge_id)
                max_speed = traci.lane.getMaxSpeed(
                    f"{edge_id}_0")  # Use lane 0 for edge max speed

                # Create Tree Method Link object (reuses existing code)
                link = Link(
                    link_id=hash(edge_id) % 10000,  # Simple ID mapping
                    edge_name=edge_id,
                    from_node_name="",  # Not needed for RL
                    to_node_name="",    # Not needed for RL
                    distance=length,    # This becomes distance_meters
                    lanes=lanes,
                    # This becomes free_flow_v_km_h (after conversion)
                    free_flow_v=max_speed,
                    head_names=[],      # Not needed for RL
                    m=self.m,
                    l=self.l
                )

                # Calculate optimal flow properties using Tree Method algorithms
                link.calc_max_properties()

                self.edge_links[edge_id] = link
                self.speed_history[edge_id] = deque(
                    maxlen=MOVING_AVERAGE_WINDOW_SIZE)
                initialized_count += 1

                if self.debug and initialized_count <= 3:  # Log first few for debugging
                    self.logger.info(
                        f"✓ Initialized Tree Method link: {edge_id}")
                    self.logger.info(
                        f"  - Length: {length:.1f}m, Lanes: {lanes}, Max speed: {max_speed:.1f}m/s")
                    self.logger.info(
                        f"  - Optimal speed: {link.q_max_properties.q_max_u:.1f}km/h")
                    self.logger.info(
                        f"  - Max flow: {link.q_max_properties.q_max:.1f}veh/h")

            except Exception as e:
                failed_count += 1
                # Always log the first few failures to understand the issue
                if failed_count <= 3:
                    self.logger.error(
                        f"✗ Failed to initialize link {edge_id}: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                elif self.debug:
                    self.logger.warning(
                        f"✗ Failed to initialize link {edge_id}: {e}")

        # self.logger.info(
        #     f"Tree Method links initialized: {initialized_count} success, {failed_count} failed")
        # self.logger.info(
        #     f"Successfully initialized edges: {list(self.edge_links.keys())}")

        if initialized_count == 0:
            self.logger.error(
                "=== CRITICAL: No Tree Method links were successfully initialized! ===")
            self.logger.error(
                "This will cause ALL OBSERVATION FEATURES TO BE ZERO!")
        elif self.debug:
            self.logger.info(
                f"Tree Method analyzer ready with {len(self.edge_links)} links")

    def get_enhanced_edge_features(self, edge_id: str) -> List[float]:
        """Get Tree Method traffic analysis features for an edge."""
        if edge_id not in self.edge_links:
            self.logger.error(
                f"=== ZERO FEATURES: Edge {edge_id} not in initialized links! ===")
            self.logger.error(
                f"Available initialized edges: {list(self.edge_links.keys())[:5]}{'...' if len(self.edge_links) > 5 else ''}")
            self.logger.error(
                f"This will contribute {RL_DYNAMIC_EDGE_FEATURES_COUNT} ZERO features to observation!")
            return [0.0] * RL_DYNAMIC_EDGE_FEATURES_COUNT

        link = self.edge_links[edge_id]
        # self.logger.info(f"DEBUG: Processing edge {edge_id} with link {link}")

        try:
            # Basic SUMO data
            current_speed_ms = traci.edge.getLastStepMeanSpeed(edge_id)
            current_speed_kmh = current_speed_ms * 3.6
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            waiting_time = traci.edge.getWaitingTime(edge_id)
            edge_length = link.distance_meters  # Use pre-stored value from initialization
            max_speed = link.free_flow_v_km_h / 3.6  # Convert from km/h to m/s

            # Debug logging for first few calls to understand data
            # if edge_id == self.edge_ids[0]:  # Log for first edge only
            #     self.logger.info(
            #         f"DEBUG: Edge {edge_id} - Speed: {current_speed_ms:.2f}m/s, Vehicles: {vehicle_count}, Waiting: {waiting_time:.1f}s")

            # Update speed history for trend calculation
            self.speed_history[edge_id].append(current_speed_kmh)

            # ORIGINAL RL FEATURES (4 features - keep existing)
            speed_ratio = current_speed_ms / max_speed if max_speed > 0 else 0.0
            density_simple = vehicle_count / edge_length if edge_length > 0 else 0.0
            flow_simple = vehicle_count / 1.0  # per second estimate
            congestion_flag = 1.0 if waiting_time > 30.0 else 0.0

            # TREE METHOD ENHANCED FEATURES (6 new features)

            # 1. Traffic flow theory density (using Tree Method calc_k_by_u)
            current_speed_safe = max(
                current_speed_kmh, TREE_METHOD_MIN_VELOCITY)
            traffic_density = link.calc_k_by_u(current_speed_safe)
            normalized_density = traffic_density / TREE_METHOD_MAX_DENSITY

            # 2. Traffic flow theory flow (speed × density × lanes)
            flow_per_lane_per_hour = current_speed_safe * traffic_density
            total_flow = flow_per_lane_per_hour * link.lanes
            normalized_flow = min(total_flow / MAX_FLOW_PER_LANE_PER_HOUR, 1.0)

            # 3. Bottleneck detection (Tree Method is_loaded logic)
            is_bottleneck = 1.0 if current_speed_safe < link.q_max_properties.q_max_u else 0.0

            # 4. Time loss calculation (Tree Method time_loss_m logic)
            if link.q_max_properties and current_speed_safe < link.q_max_properties.q_max_u:
                optimal_time = edge_length / \
                    (link.q_max_properties.q_max_u * 1000/3600)  # seconds
                actual_time = edge_length / \
                    (current_speed_ms + 0.001)  # avoid division by zero
                time_loss = max(0, actual_time - optimal_time) / 60  # minutes
            else:
                time_loss = 0.0
            normalized_time_loss = min(time_loss / MAX_TIME_LOSS_MINUTES, 1.0)

            # 5. Cost metric (Tree Method cost calculation: flow × time_loss)
            cost = total_flow * time_loss
            normalized_cost = min(cost / MAX_COST_PER_EDGE, 1.0)

            # 6. Speed trend (moving average vs current)
            if len(self.speed_history[edge_id]) >= 2:
                avg_speed = sum(
                    self.speed_history[edge_id]) / len(self.speed_history[edge_id])
                speed_trend = (current_speed_kmh - avg_speed) / \
                    max(avg_speed, 1.0)
                speed_trend = max(-1.0, min(1.0, speed_trend)
                                  )  # clip to [-1, 1]
            else:
                speed_trend = 0.0

            # Build features list based on enabled toggles
            features = []

            # Add features conditionally based on toggles
            if ENABLE_EDGE_SPEED_RATIO:
                features.append(speed_ratio)
            if ENABLE_EDGE_DENSITY_SIMPLE:
                features.append(density_simple)
            if ENABLE_EDGE_FLOW_SIMPLE:
                features.append(flow_simple)
            if ENABLE_EDGE_CONGESTION_FLAG:
                features.append(congestion_flag)
            if ENABLE_EDGE_NORMALIZED_DENSITY:
                features.append(normalized_density)
            if ENABLE_EDGE_NORMALIZED_FLOW:
                features.append(normalized_flow)
            if ENABLE_EDGE_IS_BOTTLENECK:
                features.append(is_bottleneck)
            if ENABLE_EDGE_NORMALIZED_TIME_LOSS:
                features.append(normalized_time_loss)
            if ENABLE_EDGE_NORMALIZED_COST:
                features.append(normalized_cost)
            if ENABLE_EDGE_SPEED_TREND:
                # Transform speed_trend from [-1,1] to [0,1] for observation space compatibility
                speed_trend_normalized = (speed_trend + 1.0) / 2.0
                features.append(speed_trend_normalized)

            # Debug logging for first few observations or interesting cases
            if self.debug and (self.observation_count < 3 or is_bottleneck > 0.5 or time_loss > 0.1):
                self.logger.info(f"Enhanced features for {edge_id}:")
                self.logger.info(
                    f"  - Speed: {current_speed_kmh:.1f}km/h (ratio={speed_ratio:.3f})")
                self.logger.info(
                    f"  - Vehicles: {vehicle_count}, Wait: {waiting_time:.1f}s")
                self.logger.info(
                    f"  - Tree Method density: {traffic_density:.1f} (norm={normalized_density:.3f})")
                self.logger.info(
                    f"  - Tree Method flow: {total_flow:.1f} (norm={normalized_flow:.3f})")
                self.logger.info(
                    f"  - Bottleneck: {is_bottleneck:.1f} (speed < {link.q_max_properties.q_max_u:.1f})")
                self.logger.info(
                    f"  - Time loss: {time_loss:.3f}min (norm={normalized_time_loss:.3f})")
                self.logger.info(
                    f"  - Cost: {cost:.1f} (norm={normalized_cost:.3f})")
                self.logger.info(f"  - Speed trend: {speed_trend:.3f}")

            return features

        except Exception as e:
            self.logger.error(f"ERROR calculating features for {edge_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return [0.0] * RL_DYNAMIC_EDGE_FEATURES_COUNT

    def get_network_level_features(self) -> List[float]:
        """Get network-wide traffic metrics using Tree Method concepts."""
        try:
            total_edges = len([eid for eid in self.edge_ids if ':' not in eid])
            loaded_count = 0
            total_cost = 0.0
            total_vehicles = 0
            speed_sum = 0.0
            congestion_chains = 0

            for edge_id in self.edge_ids:
                if ':' in edge_id or edge_id not in self.edge_links:
                    continue

                link = self.edge_links[edge_id]
                current_speed = traci.edge.getLastStepMeanSpeed(edge_id) * 3.6
                vehicles = traci.edge.getLastStepVehicleNumber(edge_id)

                # Count bottlenecks
                if current_speed < link.q_max_properties.q_max_u:
                    loaded_count += 1

                # Accumulate metrics
                total_vehicles += vehicles
                speed_sum += current_speed

                # Estimate cost (simplified)
                if current_speed < link.q_max_properties.q_max_u:
                    time_loss = 1.0  # simplified
                    flow = current_speed * link.lanes
                    total_cost += flow * time_loss

            # Normalize features
            bottleneck_ratio = loaded_count / max(total_edges, 1)
            cost_normalized = min(
                total_cost / (MAX_COST_PER_EDGE * total_edges), 1.0)
            vehicles_normalized = min(
                total_vehicles / (MAX_VEHICLE_COUNT_PER_EDGE * total_edges), 1.0)
            # normalize by 50 km/h
            avg_speed_normalized = (speed_sum / max(total_edges, 1)) / 50.0
            congestion_ratio = min(congestion_chains /
                                   max(total_edges, 1), 1.0)

            # Build network features list based on enabled toggles
            features = []
            if ENABLE_NETWORK_BOTTLENECK_RATIO:
                features.append(bottleneck_ratio)
            if ENABLE_NETWORK_COST_NORMALIZED:
                features.append(cost_normalized)
            if ENABLE_NETWORK_VEHICLES_NORMALIZED:
                features.append(vehicles_normalized)
            if ENABLE_NETWORK_AVG_SPEED_NORMALIZED:
                features.append(avg_speed_normalized)
            if ENABLE_NETWORK_CONGESTION_RATIO:
                features.append(congestion_ratio)

            return features

        except Exception as e:
            return [0.0] * RL_DYNAMIC_NETWORK_FEATURES_COUNT

    def get_enhanced_junction_features(self, junction_id: str) -> List[float]:
        """Get enhanced junction features using topology awareness."""
        try:
            # Original features (2)
            current_phase = traci.trafficlight.getPhase(junction_id)
            total_phases = len(
                traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
            phase_normalized = current_phase / total_phases if total_phases > 0 else 0.0

            remaining_duration = traci.trafficlight.getNextSwitch(
                junction_id) - traci.simulation.getTime()
            duration_normalized = remaining_duration / 120.0  # normalize by max duration

            # Enhanced features (4) - require network topology which we'll add
            # For now, placeholder values that can be enhanced later
            incoming_flow = 0.0
            outgoing_flow = 0.0
            upstream_bottlenecks = 0.0
            downstream_bottlenecks = 0.0

            # Build junction features list based on enabled toggles
            features = []
            if ENABLE_JUNCTION_PHASE_NORMALIZED:
                features.append(phase_normalized)
            if ENABLE_JUNCTION_DURATION_NORMALIZED:
                features.append(duration_normalized)
            if ENABLE_JUNCTION_INCOMING_FLOW:
                features.append(incoming_flow)
            if ENABLE_JUNCTION_OUTGOING_FLOW:
                features.append(outgoing_flow)
            if ENABLE_JUNCTION_UPSTREAM_BOTTLENECKS:
                features.append(upstream_bottlenecks)
            if ENABLE_JUNCTION_DOWNSTREAM_BOTTLENECKS:
                features.append(downstream_bottlenecks)

            return features

        except Exception as e:
            return [0.0] * RL_DYNAMIC_JUNCTION_FEATURES_COUNT

    def inspect_state_vector(self, observation: List[float], edge_count: int = None, junction_count: int = None) -> Dict:
        """Provide detailed inspection of the state vector for debugging."""
        self.observation_count += 1

        # Use provided counts or estimate from edge_ids
        if edge_count is None:
            edge_count = len([e for e in self.edge_ids if ':' not in e])
        if junction_count is None:
            junction_count = 9  # Default estimate

        expected_features = (edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT +
                             junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT +
                             RL_DYNAMIC_NETWORK_FEATURES_COUNT)

        inspection = {
            'total_features': len(observation),
            'expected_features': expected_features,
            'non_zero_features': sum(1 for x in observation if abs(x) > 1e-6),
            'feature_range': [min(observation), max(observation)],
            'observation_count': self.observation_count,
            'edge_count': edge_count,
            'junction_count': junction_count
        }

        # Break down by feature type
        feature_idx = 0

        # Edge features
        edge_features_count = edge_count * RL_DYNAMIC_EDGE_FEATURES_COUNT
        edge_features = observation[feature_idx:feature_idx +
                                    edge_features_count]
        inspection['edge_features'] = {
            'count': len(edge_features),
            'non_zero': sum(1 for x in edge_features if abs(x) > 1e-6),
            'range': [min(edge_features) if edge_features else 0, max(edge_features) if edge_features else 0]
        }
        feature_idx += edge_features_count

        # Junction features
        junction_features_count = junction_count * RL_DYNAMIC_JUNCTION_FEATURES_COUNT
        junction_features = observation[feature_idx:feature_idx +
                                        junction_features_count]
        inspection['junction_features'] = {
            'count': len(junction_features),
            'non_zero': sum(1 for x in junction_features if abs(x) > 1e-6),
            'range': [min(junction_features) if junction_features else 0, max(junction_features) if junction_features else 0]
        }
        feature_idx += junction_features_count

        # Network features
        network_features = observation[feature_idx:feature_idx +
                                       RL_DYNAMIC_NETWORK_FEATURES_COUNT]
        inspection['network_features'] = {
            'count': len(network_features),
            'non_zero': sum(1 for x in network_features if abs(x) > 1e-6),
            'range': [min(network_features) if network_features else 0, max(network_features) if network_features else 0]
        }

        return inspection

    def log_detailed_inspection(self, observation: List[float]) -> None:
        """Log detailed state vector inspection."""
        inspection = self.inspect_state_vector(observation)
        self.log_detailed_inspection_from_dict(inspection)

    def log_detailed_inspection_from_dict(self, inspection: Dict) -> None:
        """Log detailed state vector inspection from inspection dictionary."""

        if not self.debug:
            return  # Skip if not in debug mode to avoid log clutter

        self.logger.info(
            f"=== State Vector Inspection (Observation #{inspection['observation_count']}) ===")
        self.logger.info(
            f"Total features: {inspection['total_features']} (expected: {inspection['expected_features']})")
        self.logger.info(
            f"Non-zero features: {inspection['non_zero_features']}/{inspection['total_features']}")
        self.logger.info(
            f"Value range: [{inspection['feature_range'][0]:.3f}, {inspection['feature_range'][1]:.3f}]")

        self.logger.info(
            f"Edge features: {inspection['edge_features']['non_zero']}/{inspection['edge_features']['count']} non-zero")
        self.logger.info(
            f"Junction features: {inspection['junction_features']['non_zero']}/{inspection['junction_features']['count']} non-zero")
        self.logger.info(
            f"Network features: {inspection['network_features']['non_zero']}/{inspection['network_features']['count']} non-zero")

        if inspection['non_zero_features'] == 0:
            self.logger.warning(
                "⚠️  All features are zero - this may indicate an issue with data collection!")
        elif inspection['non_zero_features'] < inspection['total_features'] * 0.1:
            self.logger.warning(
                f"⚠️  Very few non-zero features ({inspection['non_zero_features']}) - check if simulation has enough activity")
        else:
            self.logger.info("✓ State vector contains meaningful data")
