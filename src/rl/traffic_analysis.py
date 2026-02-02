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

# =============================================================================
# FEATURE REGISTRIES
# =============================================================================
# Maps feature names (used in experiment YAML) to RLTrafficAnalyzer method names.
# These registries are the source of truth for valid feature names.

EDGE_FEATURE_REGISTRY = {
    "speed_ratio": "_compute_edge_speed_ratio",
    "density_simple": "_compute_edge_density_simple",
    "flow_simple": "_compute_edge_flow_simple",
    "congestion_flag": "_compute_edge_congestion_flag",
    "normalized_density": "_compute_edge_normalized_density",
    "normalized_flow": "_compute_edge_normalized_flow",
    "is_bottleneck": "_compute_edge_is_bottleneck",
    "normalized_time_loss": "_compute_edge_normalized_time_loss",
    "normalized_cost": "_compute_edge_normalized_cost",
    "speed_trend": "_compute_edge_speed_trend",
}

JUNCTION_FEATURE_REGISTRY = {
    "phase_normalized": "_compute_junction_phase_normalized",
    "duration_normalized": "_compute_junction_duration_normalized",
    "incoming_flow": "_compute_junction_incoming_flow",           # placeholder
    "outgoing_flow": "_compute_junction_outgoing_flow",           # placeholder
    "upstream_bottlenecks": "_compute_junction_upstream_bottlenecks",   # placeholder
    "downstream_bottlenecks": "_compute_junction_downstream_bottlenecks",  # placeholder
}

NETWORK_FEATURE_REGISTRY = {
    "bottleneck_ratio": "_compute_network_bottleneck_ratio",
    "cost_normalized": "_compute_network_cost_normalized",
    "vehicles_normalized": "_compute_network_vehicles_normalized",
    "avg_speed_normalized": "_compute_network_avg_speed_normalized",
    "congestion_ratio": "_compute_network_congestion_ratio",
}


class RLTrafficAnalyzer:
    """Provides Tree Method traffic analysis for RL state collection."""

    # Enable debug by default
    def __init__(self, edge_ids: List[str], m: float, l: float, debug: bool = False,
                 experiment_config=None):
        """Initialize with network topology.

        Args:
            edge_ids: List of SUMO edge IDs
            m: Tree Method m parameter
            l: Tree Method l parameter
            debug: Enable debug logging
            experiment_config: ExperimentConfig for config-driven features (None = use toggle constants)
        """
        self.edge_ids = edge_ids
        self.edge_links: Dict[str, Link] = {}
        self.speed_history: Dict[str, deque] = {}
        self.debug = False  # debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.observation_count = 0
        self.m = m
        self.l = l
        self.experiment_config = experiment_config

        # Pre-resolve feature method lists from config or use toggle-based defaults
        if experiment_config is not None:
            self._edge_feature_methods = [
                EDGE_FEATURE_REGISTRY[name] for name in experiment_config.edge_features
            ]
            self._junction_feature_methods = [
                JUNCTION_FEATURE_REGISTRY[name] for name in experiment_config.junction_features
            ]
            self._network_feature_methods = [
                NETWORK_FEATURE_REGISTRY[name] for name in experiment_config.network_features
            ]
            self._edge_feature_count = len(experiment_config.edge_features)
            self._junction_feature_count = len(experiment_config.junction_features)
            self._network_feature_count = len(experiment_config.network_features)
        else:
            # Legacy: use constants-based toggles (None signals toggle-based path)
            self._edge_feature_methods = None
            self._junction_feature_methods = None
            self._network_feature_methods = None
            self._edge_feature_count = RL_DYNAMIC_EDGE_FEATURES_COUNT
            self._junction_feature_count = RL_DYNAMIC_JUNCTION_FEATURES_COUNT
            self._network_feature_count = RL_DYNAMIC_NETWORK_FEATURES_COUNT

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

        if initialized_count == 0:
            self.logger.error(
                "=== CRITICAL: No Tree Method links were successfully initialized! ===")
            self.logger.error(
                "This will cause ALL OBSERVATION FEATURES TO BE ZERO!")
        elif self.debug:
            self.logger.info(
                f"Tree Method analyzer ready with {len(self.edge_links)} links")

    def _compute_edge_data(self, edge_id: str) -> Dict:
        """Compute all raw edge metrics once, used by individual feature methods.

        Returns dict with all computed values for the edge, or None on failure.
        """
        link = self.edge_links[edge_id]

        # Basic SUMO data
        current_speed_ms = traci.edge.getLastStepMeanSpeed(edge_id)
        current_speed_kmh = current_speed_ms * 3.6
        vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
        waiting_time = traci.edge.getWaitingTime(edge_id)
        edge_length = link.distance_meters
        max_speed = link.free_flow_v_km_h / 3.6  # m/s

        # Update speed history
        self.speed_history[edge_id].append(current_speed_kmh)

        # Derived values
        speed_ratio = current_speed_ms / max_speed if max_speed > 0 else 0.0
        density_simple = vehicle_count / edge_length if edge_length > 0 else 0.0
        flow_simple = vehicle_count / 1.0

        current_speed_safe = max(current_speed_kmh, TREE_METHOD_MIN_VELOCITY)
        traffic_density = link.calc_k_by_u(current_speed_safe)
        flow_per_lane_per_hour = current_speed_safe * traffic_density
        total_flow = flow_per_lane_per_hour * link.lanes
        is_bottleneck_val = current_speed_safe < link.q_max_properties.q_max_u

        # Time loss
        if link.q_max_properties and is_bottleneck_val:
            optimal_time = edge_length / (link.q_max_properties.q_max_u * 1000 / 3600)
            actual_time = edge_length / (current_speed_ms + 0.001)
            time_loss = max(0, actual_time - optimal_time) / 60
        else:
            time_loss = 0.0

        cost = total_flow * time_loss

        # Speed trend
        if len(self.speed_history[edge_id]) >= 2:
            avg_speed = sum(self.speed_history[edge_id]) / len(self.speed_history[edge_id])
            speed_trend = (current_speed_kmh - avg_speed) / max(avg_speed, 1.0)
            speed_trend = max(-1.0, min(1.0, speed_trend))
        else:
            speed_trend = 0.0

        return {
            'speed_ratio': speed_ratio,
            'density_simple': density_simple,
            'flow_simple': flow_simple,
            'congestion_flag': 1.0 if waiting_time > 30.0 else 0.0,
            'normalized_density': traffic_density / TREE_METHOD_MAX_DENSITY,
            'normalized_flow': min(total_flow / MAX_FLOW_PER_LANE_PER_HOUR, 1.0),
            'is_bottleneck': 1.0 if is_bottleneck_val else 0.0,
            'normalized_time_loss': min(time_loss / MAX_TIME_LOSS_MINUTES, 1.0),
            'normalized_cost': min(cost / MAX_COST_PER_EDGE, 1.0),
            'speed_trend': (speed_trend + 1.0) / 2.0,  # transform [-1,1] to [0,1]
            # Raw values for debug logging
            '_current_speed_kmh': current_speed_kmh,
            '_vehicle_count': vehicle_count,
            '_waiting_time': waiting_time,
            '_traffic_density': traffic_density,
            '_total_flow': total_flow,
            '_time_loss': time_loss,
            '_cost': cost,
            '_speed_trend_raw': speed_trend,
            '_q_max_u': link.q_max_properties.q_max_u,
        }

    # Individual feature methods for config-driven path
    def _compute_edge_speed_ratio(self, data: Dict) -> float:
        return data['speed_ratio']

    def _compute_edge_density_simple(self, data: Dict) -> float:
        return data['density_simple']

    def _compute_edge_flow_simple(self, data: Dict) -> float:
        return data['flow_simple']

    def _compute_edge_congestion_flag(self, data: Dict) -> float:
        return data['congestion_flag']

    def _compute_edge_normalized_density(self, data: Dict) -> float:
        return data['normalized_density']

    def _compute_edge_normalized_flow(self, data: Dict) -> float:
        return data['normalized_flow']

    def _compute_edge_is_bottleneck(self, data: Dict) -> float:
        return data['is_bottleneck']

    def _compute_edge_normalized_time_loss(self, data: Dict) -> float:
        return data['normalized_time_loss']

    def _compute_edge_normalized_cost(self, data: Dict) -> float:
        return data['normalized_cost']

    def _compute_edge_speed_trend(self, data: Dict) -> float:
        return data['speed_trend']

    def get_enhanced_edge_features(self, edge_id: str) -> List[float]:
        """Get Tree Method traffic analysis features for an edge."""
        if edge_id not in self.edge_links:
            self.logger.error(
                f"=== ZERO FEATURES: Edge {edge_id} not in initialized links! ===")
            self.logger.error(
                f"Available initialized edges: {list(self.edge_links.keys())[:5]}{'...' if len(self.edge_links) > 5 else ''}")
            self.logger.error(
                f"This will contribute {self._edge_feature_count} ZERO features to observation!")
            return [0.0] * self._edge_feature_count

        try:
            data = self._compute_edge_data(edge_id)

            # Config-driven path: use pre-resolved method list
            if self._edge_feature_methods is not None:
                features = [getattr(self, method)(data) for method in self._edge_feature_methods]
            else:
                # Legacy toggle-based path
                features = []
                if ENABLE_EDGE_SPEED_RATIO:
                    features.append(data['speed_ratio'])
                if ENABLE_EDGE_DENSITY_SIMPLE:
                    features.append(data['density_simple'])
                if ENABLE_EDGE_FLOW_SIMPLE:
                    features.append(data['flow_simple'])
                if ENABLE_EDGE_CONGESTION_FLAG:
                    features.append(data['congestion_flag'])
                if ENABLE_EDGE_NORMALIZED_DENSITY:
                    features.append(data['normalized_density'])
                if ENABLE_EDGE_NORMALIZED_FLOW:
                    features.append(data['normalized_flow'])
                if ENABLE_EDGE_IS_BOTTLENECK:
                    features.append(data['is_bottleneck'])
                if ENABLE_EDGE_NORMALIZED_TIME_LOSS:
                    features.append(data['normalized_time_loss'])
                if ENABLE_EDGE_NORMALIZED_COST:
                    features.append(data['normalized_cost'])
                if ENABLE_EDGE_SPEED_TREND:
                    features.append(data['speed_trend'])

            # Debug logging
            if self.debug and (self.observation_count < 3 or data['is_bottleneck'] > 0.5 or data['_time_loss'] > 0.1):
                self.logger.info(f"Enhanced features for {edge_id}:")
                self.logger.info(
                    f"  - Speed: {data['_current_speed_kmh']:.1f}km/h (ratio={data['speed_ratio']:.3f})")
                self.logger.info(
                    f"  - Vehicles: {data['_vehicle_count']}, Wait: {data['_waiting_time']:.1f}s")
                self.logger.info(
                    f"  - Tree Method density: {data['_traffic_density']:.1f} (norm={data['normalized_density']:.3f})")
                self.logger.info(
                    f"  - Tree Method flow: {data['_total_flow']:.1f} (norm={data['normalized_flow']:.3f})")
                self.logger.info(
                    f"  - Bottleneck: {data['is_bottleneck']:.1f} (speed < {data['_q_max_u']:.1f})")
                self.logger.info(
                    f"  - Time loss: {data['_time_loss']:.3f}min (norm={data['normalized_time_loss']:.3f})")
                self.logger.info(
                    f"  - Cost: {data['_cost']:.1f} (norm={data['normalized_cost']:.3f})")
                self.logger.info(f"  - Speed trend: {data['_speed_trend_raw']:.3f}")

            return features

        except Exception as e:
            self.logger.error(f"ERROR calculating features for {edge_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return [0.0] * self._edge_feature_count

    def _compute_network_data(self) -> Dict:
        """Compute all raw network-level metrics once."""
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

            if current_speed < link.q_max_properties.q_max_u:
                loaded_count += 1

            total_vehicles += vehicles
            speed_sum += current_speed

            if current_speed < link.q_max_properties.q_max_u:
                time_loss = 1.0
                flow = current_speed * link.lanes
                total_cost += flow * time_loss

        return {
            'bottleneck_ratio': loaded_count / max(total_edges, 1),
            'cost_normalized': min(total_cost / (MAX_COST_PER_EDGE * total_edges), 1.0),
            'vehicles_normalized': min(total_vehicles / (MAX_VEHICLE_COUNT_PER_EDGE * total_edges), 1.0),
            'avg_speed_normalized': (speed_sum / max(total_edges, 1)) / 50.0,
            'congestion_ratio': min(congestion_chains / max(total_edges, 1), 1.0),
        }

    # Individual network feature methods for config-driven path
    def _compute_network_bottleneck_ratio(self, data: Dict) -> float:
        return data['bottleneck_ratio']

    def _compute_network_cost_normalized(self, data: Dict) -> float:
        return data['cost_normalized']

    def _compute_network_vehicles_normalized(self, data: Dict) -> float:
        return data['vehicles_normalized']

    def _compute_network_avg_speed_normalized(self, data: Dict) -> float:
        return data['avg_speed_normalized']

    def _compute_network_congestion_ratio(self, data: Dict) -> float:
        return data['congestion_ratio']

    def get_network_level_features(self) -> List[float]:
        """Get network-wide traffic metrics using Tree Method concepts."""
        try:
            data = self._compute_network_data()

            # Config-driven path
            if self._network_feature_methods is not None:
                return [getattr(self, method)(data) for method in self._network_feature_methods]

            # Legacy toggle-based path
            features = []
            if ENABLE_NETWORK_BOTTLENECK_RATIO:
                features.append(data['bottleneck_ratio'])
            if ENABLE_NETWORK_COST_NORMALIZED:
                features.append(data['cost_normalized'])
            if ENABLE_NETWORK_VEHICLES_NORMALIZED:
                features.append(data['vehicles_normalized'])
            if ENABLE_NETWORK_AVG_SPEED_NORMALIZED:
                features.append(data['avg_speed_normalized'])
            if ENABLE_NETWORK_CONGESTION_RATIO:
                features.append(data['congestion_ratio'])

            return features

        except Exception as e:
            return [0.0] * self._network_feature_count

    def _compute_junction_data(self, junction_id: str) -> Dict:
        """Compute all raw junction metrics once."""
        current_phase = traci.trafficlight.getPhase(junction_id)
        total_phases = len(
            traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)

        remaining_duration = traci.trafficlight.getNextSwitch(
            junction_id) - traci.simulation.getTime()

        return {
            'phase_normalized': current_phase / total_phases if total_phases > 0 else 0.0,
            'duration_normalized': remaining_duration / 120.0,
            'incoming_flow': 0.0,  # placeholder
            'outgoing_flow': 0.0,  # placeholder
            'upstream_bottlenecks': 0.0,  # placeholder
            'downstream_bottlenecks': 0.0,  # placeholder
        }

    # Individual junction feature methods for config-driven path
    def _compute_junction_phase_normalized(self, data: Dict) -> float:
        return data['phase_normalized']

    def _compute_junction_duration_normalized(self, data: Dict) -> float:
        return data['duration_normalized']

    def _compute_junction_incoming_flow(self, data: Dict) -> float:
        return data['incoming_flow']

    def _compute_junction_outgoing_flow(self, data: Dict) -> float:
        return data['outgoing_flow']

    def _compute_junction_upstream_bottlenecks(self, data: Dict) -> float:
        return data['upstream_bottlenecks']

    def _compute_junction_downstream_bottlenecks(self, data: Dict) -> float:
        return data['downstream_bottlenecks']

    def get_enhanced_junction_features(self, junction_id: str) -> List[float]:
        """Get enhanced junction features using topology awareness."""
        try:
            data = self._compute_junction_data(junction_id)

            # Config-driven path
            if self._junction_feature_methods is not None:
                return [getattr(self, method)(data) for method in self._junction_feature_methods]

            # Legacy toggle-based path
            features = []
            if ENABLE_JUNCTION_PHASE_NORMALIZED:
                features.append(data['phase_normalized'])
            if ENABLE_JUNCTION_DURATION_NORMALIZED:
                features.append(data['duration_normalized'])
            if ENABLE_JUNCTION_INCOMING_FLOW:
                features.append(data['incoming_flow'])
            if ENABLE_JUNCTION_OUTGOING_FLOW:
                features.append(data['outgoing_flow'])
            if ENABLE_JUNCTION_UPSTREAM_BOTTLENECKS:
                features.append(data['upstream_bottlenecks'])
            if ENABLE_JUNCTION_DOWNSTREAM_BOTTLENECKS:
                features.append(data['downstream_bottlenecks'])

            return features

        except Exception as e:
            return [0.0] * self._junction_feature_count

    def inspect_state_vector(self, observation: List[float], edge_count: int = None, junction_count: int = None) -> Dict:
        """Provide detailed inspection of the state vector for debugging."""
        self.observation_count += 1

        # Use provided counts or estimate from edge_ids
        if edge_count is None:
            edge_count = len([e for e in self.edge_ids if ':' not in e])
        if junction_count is None:
            junction_count = 9  # Default estimate

        expected_features = (edge_count * self._edge_feature_count +
                             junction_count * self._junction_feature_count +
                             self._network_feature_count)

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
        edge_features_count = edge_count * self._edge_feature_count
        edge_features = observation[feature_idx:feature_idx +
                                    edge_features_count]
        inspection['edge_features'] = {
            'count': len(edge_features),
            'non_zero': sum(1 for x in edge_features if abs(x) > 1e-6),
            'range': [min(edge_features) if edge_features else 0, max(edge_features) if edge_features else 0]
        }
        feature_idx += edge_features_count

        # Junction features
        junction_features_count = junction_count * self._junction_feature_count
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
                                       self._network_feature_count]
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
