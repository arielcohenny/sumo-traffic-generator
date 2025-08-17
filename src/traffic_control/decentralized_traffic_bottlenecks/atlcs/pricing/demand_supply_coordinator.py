"""
Coordinates enhanced bottleneck detection with ATLCS pricing for traffic light control optimization.
"""

import traci


class DemandSupplyCoordinator:
    """Coordinates enhanced bottleneck detection with ATLCS pricing for traffic light control optimization."""

    def __init__(self, bottleneck_detector, atlcs_pricing):
        self.bottleneck_detector = bottleneck_detector    # Enhanced bottleneck detection integration
        self.atlcs_pricing = atlcs_pricing      # ATLCS integration

    def coordinate(self, step: int, bottleneck_data, pricing_updates) -> None:
        """MAIN COORDINATION METHOD - Called by ATLCSController.update() every step."""

        if bottleneck_data and pricing_updates:
            # SUPPLY SIDE: Enhance traffic signal control for bottlenecks
            self._enhance_signal_control_for_bottlenecks(bottleneck_data)

            # PRICING ANALYSIS: Monitor pricing effectiveness for signal decisions
            self._analyze_pricing_effectiveness(pricing_updates)

            # COORDINATION: Integrate bottleneck detection with pricing-based signal control
            self._coordinate_signal_optimization(
                bottleneck_data, pricing_updates)

    def _enhance_signal_control_for_bottlenecks(self, bottleneck_data) -> None:
        """USED BY: coordinate() - Supply side enhancement"""

        import logging
        logger = logging.getLogger(self.__class__.__name__)

        # Extend green time for bottleneck directions (ENHANCED thresholds and impact)
        extensions_applied = 0
        for bottleneck in bottleneck_data.prioritized_bottlenecks:
            if bottleneck.severity > 0.3:  # LOWERED from 0.7 to 0.3 for broader intervention
                # Find associated traffic light and extend green time
                tls_id = self._find_tls_for_edge(bottleneck.edge_id)
                if tls_id:
                    # INCREASED extension based on severity
                    if bottleneck.severity > 0.8:
                        extension = 25  # High severity: +25 seconds
                    elif bottleneck.severity > 0.5:
                        extension = 15  # Medium severity: +15 seconds
                    else:
                        extension = 10  # Low severity: +10 seconds

                    self._extend_green_time(
                        tls_id, bottleneck.direction, extension)
                    extensions_applied += 1

    def _analyze_pricing_effectiveness(self, pricing_updates) -> None:
        """USED BY: coordinate() - Analyze pricing data for signal control decisions"""

        import logging
        logger = logging.getLogger(self.__class__.__name__)

        # Analyze pricing data to inform traffic light control decisions
        if pricing_updates and hasattr(pricing_updates, 'edge_prices'):
            priced_edges = len(pricing_updates.edge_prices)
            if priced_edges > 0:
                avg_price = sum(
                    pricing_updates.edge_prices.values()) / priced_edges
                max_price = max(pricing_updates.edge_prices.values())
                high_priority_edges = [
                    edge for edge, price in pricing_updates.edge_prices.items() if price > avg_price * 1.5]

                # Store pricing analysis for signal control coordination
                self._pricing_analysis = {
                    'priced_edges_count': priced_edges,
                    'average_price': avg_price,
                    'max_price': max_price,
                    'high_priority_edges': high_priority_edges
                }

    def _coordinate_signal_optimization(self, bottleneck_data, pricing_updates) -> None:
        """USED BY: coordinate() - Coordinate bottleneck detection with pricing-based signal control"""

        import logging
        logger = logging.getLogger(self.__class__.__name__)

        # Coordinate enhanced bottleneck detection with ATLCS pricing for unified signal control
        bottleneck_count = len(
            bottleneck_data.prioritized_bottlenecks) if bottleneck_data.prioritized_bottlenecks else 0
        pricing_count = len(pricing_updates.edge_prices) if pricing_updates and hasattr(
            pricing_updates, 'edge_prices') else 0

        if bottleneck_count > 0 or pricing_count > 0:
            # Find edges that appear in both bottleneck detection and pricing
            bottleneck_edges = set(
                b.edge_id for b in bottleneck_data.prioritized_bottlenecks) if bottleneck_data.prioritized_bottlenecks else set()
            priced_edges = set(pricing_updates.edge_prices.keys(
            )) if pricing_updates and hasattr(pricing_updates, 'edge_prices') else set()

            overlapping_edges = bottleneck_edges.intersection(priced_edges)

            # Store coordination metrics for signal control optimization
            self._coordination_metrics = {
                'bottleneck_interventions': bottleneck_count,
                'pricing_interventions': pricing_count,
                'overlapping_edges': len(overlapping_edges),
                'coordination_ratio': len(overlapping_edges) / max(bottleneck_count, 1) if bottleneck_count > 0 else 0
            }

    def _find_tls_for_edge(self, edge_id: str) -> str:
        """Find traffic light controlling the given edge."""
        try:
            # Get all traffic lights
            tls_ids = traci.trafficlight.getIDList()

            # For each traffic light, check if it controls the edge
            for tls_id in tls_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(
                    tls_id)
                for lane in controlled_lanes:
                    if lane.startswith(edge_id):
                        return tls_id
        except Exception:
            pass

        return None

    def _extend_green_time(self, tls_id: str, direction: str, extension_seconds: int) -> None:
        """Extend green time for specific direction at traffic light."""
        try:
            current_duration = traci.trafficlight.getNextSwitch(
                tls_id) - traci.simulation.getTime()

            # Extend current phase duration
            new_duration = current_duration + extension_seconds
            traci.trafficlight.setPhaseDuration(tls_id, new_duration)

        except Exception:
            # Silently fail if traffic light manipulation fails
            pass
