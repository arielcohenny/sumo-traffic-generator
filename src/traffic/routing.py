# src/traffic/routing.py
from __future__ import annotations
from typing import List, Dict, Optional, NoReturn
from abc import ABC, abstractmethod
import random
import sys

from src.constants import (
    ATTR_CURRENT_PHASE,
    ROUTING_ERROR_REALTIME_FAILED,
    ROUTING_ERROR_FASTEST_FAILED,
    ROUTING_ERROR_ATTRACTIVENESS_FAILED,
    ROUTING_ERROR_INVALID_ROUTE,
    ROUTING_ERROR_MSG_TEMPLATE,
    ROUTING_SHORTEST,
    ROUTING_REALTIME,
    ROUTING_FASTEST,
    ROUTING_ATTRACTIVENESS
)


def convert_tail_to_head_edge(edge_id: str, net) -> str:
    """Convert tail edge to head edge for routing destinations.

    In multi-head edge architecture:
    - Tail edges (e.g., "A1B1") connect FROM junctions TO intermediate nodes
    - Head edges (e.g., "A1B1_H_straight") connect FROM intermediate nodes TO junctions

    For routing, destinations must be edges that END at junctions (head edges).
    We try head edges in order of preference: straight, right, left, uturn.

    Args:
        edge_id: Tail edge ID (e.g., "A1B1")
        net: SUMO network object to check edge existence

    Returns:
        Head edge ID that exists in the network
    """
    if '_H_' in edge_id:
        # Already a head edge, return as-is
        return edge_id

    # Try head edges in order of preference
    for suffix in ['straight', 'right', 'left', 'uturn']:
        candidate = f"{edge_id}_H_{suffix}"
        try:
            edge = net.getEdge(candidate)
            if edge is not None:
                # print(f"DEBUG: Converting tail edge '{edge_id}' → head edge '{candidate}'", file=sys.stderr)
                return candidate
        except (KeyError, RuntimeError):
            # Edge doesn't exist, try next suffix
            continue

    # If no head edge found, return the original (will likely fail routing, but with clear error)
    print(
        f"DEBUG: No head edge found for tail edge '{edge_id}', returning original", file=sys.stderr)
    return edge_id


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def compute_route(self, start_edge: str, end_edge: str) -> List[str]:
        """Compute route from start_edge to end_edge."""
        pass


class ShortestPathRoutingStrategy(RoutingStrategy):
    """
    Static shortest path routing using SUMO's built-in Dijkstra algorithm.
    Routes are computed once and never change during simulation.
    """

    def __init__(self, net):
        self.net = net  # sumolib.net.Net

    def compute_route(self, start_edge: str, end_edge: str) -> List[str]:
        """Compute shortest path by distance - TERMINATES ON ERROR."""
        try:
            # Convert end edge from tail to head edge for routing destination
            end_edge_for_routing = convert_tail_to_head_edge(
                end_edge, self.net)

            result = self.net.getShortestPath(
                self.net.getEdge(start_edge),
                self.net.getEdge(end_edge_for_routing)
            )
            if not result or result[0] is None:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_INVALID_ROUTE,
                    strategy=ROUTING_SHORTEST,
                    vehicle_id="N/A",
                    reason=f"No path found from {start_edge} to {end_edge}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            edge_objs, _ = result
            return [e.getID() for e in edge_objs]
        except Exception as e:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_INVALID_ROUTE,
                strategy=ROUTING_SHORTEST,
                vehicle_id="N/A",
                reason=f"Route computation failed: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)


class RealtimeRoutingStrategy(RoutingStrategy):
    """
    Real-time navigation routing that mimics GPS apps like Waze/Google Maps.
    Routes are updated dynamically during simulation based on current traffic conditions.
    """

    def __init__(self, net, rerouting_interval: int = 30):
        self.net = net  # sumolib.net.Net
        self.rerouting_interval = rerouting_interval  # seconds between rerouting checks

    def compute_route(self, start_edge: str, end_edge: str) -> List[str]:
        """Compute initial route - will be updated dynamically via TraCI - TERMINATES ON ERROR."""
        try:
            # Convert end edge from tail to head edge for routing destination
            end_edge_for_routing = convert_tail_to_head_edge(
                end_edge, self.net)

            # Use fastest path for realtime strategy initial route
            result = self.net.getFastestPath(
                self.net.getEdge(start_edge),
                self.net.getEdge(end_edge_for_routing)
            )
            if not result or result[0] is None:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_REALTIME_FAILED,
                    strategy=ROUTING_REALTIME,
                    vehicle_id="N/A",
                    reason=f"No fastest path found from {start_edge} to {end_edge}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            edge_objs, _ = result
            return [e.getID() for e in edge_objs]
        except Exception as e:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_REALTIME_FAILED,
                strategy=ROUTING_REALTIME,
                vehicle_id="N/A",
                reason=f"Route computation failed: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)


class FastestRoutingStrategy(RoutingStrategy):
    """
    Dynamic fastest route strategy based on current travel times.
    Routes are computed to minimize travel time rather than distance.
    """

    def __init__(self, net, rerouting_interval: int = 45):
        self.net = net  # sumolib.net.Net
        self.rerouting_interval = rerouting_interval  # seconds between rerouting checks

    def compute_route(self, start_edge: str, end_edge: str) -> List[str]:
        """Compute fastest path by travel time - TERMINATES ON ERROR."""
        try:
            # Convert end edge from tail to head edge for routing destination
            end_edge_for_routing = convert_tail_to_head_edge(
                end_edge, self.net)

            # Use SUMO's fastest path algorithm
            result = self.net.getFastestPath(
                self.net.getEdge(start_edge),
                self.net.getEdge(end_edge_for_routing)
            )
            if not result or result[0] is None:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_FASTEST_FAILED,
                    strategy=ROUTING_FASTEST,
                    vehicle_id="N/A",
                    reason=f"No fastest path found from {start_edge} to {end_edge}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            edge_objs, _ = result
            return [e.getID() for e in edge_objs]
        except Exception as e:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_FASTEST_FAILED,
                strategy=ROUTING_FASTEST,
                vehicle_id="N/A",
                reason=f"Route computation failed: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)


class AttractivenessRoutingStrategy(RoutingStrategy):
    """
    Multi-criteria routing that considers both path efficiency and destination attractiveness.
    Balances shortest/fastest path with attractiveness of intermediate destinations.
    """

    def __init__(self, net, attractiveness_weight: float = 0.3):
        self.net = net  # sumolib.net.Net
        # 0.0 = pure shortest, 1.0 = pure attractiveness
        self.attractiveness_weight = attractiveness_weight

    def compute_route(self, start_edge: str, end_edge: str) -> List[str]:
        """Compute route considering both efficiency and attractiveness."""
        try:
            # Convert end edge from tail to head edge for routing destination
            end_edge_for_routing = convert_tail_to_head_edge(
                end_edge, self.net)

            # Get multiple potential routes
            shortest_result = self.net.getShortestPath(
                self.net.getEdge(start_edge),
                self.net.getEdge(end_edge_for_routing)
            )

            if not shortest_result or shortest_result[0] is None:
                error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                    code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                    strategy=ROUTING_ATTRACTIVENESS,
                    vehicle_id="N/A",
                    reason=f"No shortest path found from {start_edge} to {end_edge}"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            shortest_edges, shortest_cost = shortest_result

            # Try to get alternative routes by varying the path slightly
            # This is a simplified approach - in a full implementation,
            # we could use k-shortest paths algorithms
            routes_with_scores = []

            # Score the shortest route
            shortest_route = [e.getID() for e in shortest_edges]
            shortest_score = self._score_route(shortest_route, shortest_cost)
            routes_with_scores.append((shortest_route, shortest_score))

            # Try fastest route if different from shortest
            try:
                fastest_result = self.net.getFastestPath(
                    self.net.getEdge(start_edge),
                    self.net.getEdge(end_edge)
                )
                if fastest_result and fastest_result[0] is not None:
                    fastest_edges, fastest_cost = fastest_result
                    fastest_route = [e.getID() for e in fastest_edges]
                    if fastest_route != shortest_route:
                        fastest_score = self._score_route(
                            fastest_route, fastest_cost)
                        routes_with_scores.append(
                            (fastest_route, fastest_score))
            except Exception:
                pass

            # Select route with best score
            best_route = max(routes_with_scores, key=lambda x: x[1])[0]
            return best_route

        except Exception as e:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_ATTRACTIVENESS_FAILED,
                strategy=ROUTING_ATTRACTIVENESS,
                vehicle_id="N/A",
                reason=f"Route computation failed: {str(e)}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def _score_route(self, route_edges: List[str], base_cost: float) -> float:
        """Score a route based on efficiency and attractiveness."""
        if not route_edges:
            return 0.0

        # Efficiency score (inverse of cost, normalized)
        efficiency_score = 1.0 / (1.0 + base_cost / 1000.0)  # normalize cost

        # Attractiveness score based on edges in the route
        attractiveness_score = 0.0
        valid_edges = 0

        for edge_id in route_edges:
            try:
                edge = self.net.getEdge(edge_id)
                # Get attractiveness from phase-specific attributes if available
                current_phase = getattr(
                    edge, ATTR_CURRENT_PHASE, 'morning_peak')
                arrive_attr_name = f"{current_phase}_arrive_attractiveness"
                arrive_attr = getattr(edge, arrive_attr_name, None)
                if arrive_attr is not None:
                    attractiveness_score += float(arrive_attr)
                    valid_edges += 1
                else:
                    # Fallback: use edge length as proxy for attractiveness
                    attractiveness_score += edge.getLength() / 100.0
                    valid_edges += 1
            except Exception:
                continue

        if valid_edges > 0:
            attractiveness_score /= valid_edges  # average attractiveness
            attractiveness_score /= 10.0  # normalize

        # Combine scores
        final_score = ((1.0 - self.attractiveness_weight) * efficiency_score +
                       self.attractiveness_weight * attractiveness_score)

        return final_score


class RoutingMixStrategy:
    """
    Manages multiple routing strategies with percentage-based assignment.
    Each vehicle is assigned a strategy based on the specified percentages.
    """

    def __init__(self, net, strategy_percentages: Dict[str, float]):
        self.net = net
        self.strategy_percentages = strategy_percentages
        self.strategies = {}
        self.rerouting_intervals = {}

        # Validate percentages sum to 100
        total = sum(strategy_percentages.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Routing strategy percentages must sum to 100, got {total}")

        # Initialize strategies based on percentages
        if strategy_percentages.get('shortest', 0) > 0:
            self.strategies['shortest'] = ShortestPathRoutingStrategy(net)

        if strategy_percentages.get('realtime', 0) > 0:
            self.strategies['realtime'] = RealtimeRoutingStrategy(
                net, rerouting_interval=30)
            self.rerouting_intervals['realtime'] = 30

        if strategy_percentages.get('fastest', 0) > 0:
            self.strategies['fastest'] = FastestRoutingStrategy(
                net, rerouting_interval=45)
            self.rerouting_intervals['fastest'] = 45

        if strategy_percentages.get('attractiveness', 0) > 0:
            self.strategies['attractiveness'] = AttractivenessRoutingStrategy(
                net)

    def assign_strategy_to_vehicle(self, vehicle_id: str, rng: random.Random) -> str:
        """Assign a routing strategy to a vehicle based on percentages."""
        strategy_names = list(self.strategy_percentages.keys())
        weights = list(self.strategy_percentages.values())

        chosen_strategy = rng.choices(strategy_names, weights=weights, k=1)[0]
        return chosen_strategy

    def compute_route(self, strategy_name: str, start_edge: str, end_edge: str) -> List[str]:
        """Compute route using the specified strategy - TERMINATES ON ERROR."""
        if strategy_name not in self.strategies:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_INVALID_ROUTE,
                strategy=strategy_name,
                vehicle_id="N/A",
                reason=f"Unknown routing strategy: {strategy_name}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        return self.strategies[strategy_name].compute_route(start_edge, end_edge)


def parse_routing_strategy(routing_arg: str) -> Dict[str, float]:
    """
    Parse routing strategy argument into percentages dictionary.

    Examples:
    - "shortest 100" -> {"shortest": 100.0}
    - "shortest 70 realtime 30" -> {"shortest": 70.0, "realtime": 30.0}
    - "shortest 20 realtime 30 fastest 45 attractiveness 5" -> all 4 strategies

    Args:
        routing_arg: Space-separated strategy names and percentages

    Returns:
        Dictionary mapping strategy names to percentages

    Raises:
        ValueError: If percentages don't sum to 100 or invalid format
    """
    if not routing_arg.strip():
        return {"shortest": 100.0}  # default

    tokens = routing_arg.strip().split()
    if len(tokens) % 2 != 0:
        error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
            code=ROUTING_ERROR_INVALID_ROUTE,
            strategy="parsing",
            vehicle_id="N/A",
            reason="Invalid format. Expected: 'strategy1 percentage1 strategy2 percentage2 ...'"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    valid_strategies = {ROUTING_SHORTEST, ROUTING_REALTIME,
                        ROUTING_FASTEST, ROUTING_ATTRACTIVENESS}
    percentages = {}

    for i in range(0, len(tokens), 2):
        strategy = tokens[i]
        try:
            percentage = float(tokens[i + 1])
        except ValueError:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_INVALID_ROUTE,
                strategy="parsing",
                vehicle_id="N/A",
                reason=f"Invalid percentage value: {tokens[i + 1]}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        if strategy not in valid_strategies:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_INVALID_ROUTE,
                strategy=strategy,
                vehicle_id="N/A",
                reason=f"Unknown routing strategy. Valid options: {valid_strategies}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        if percentage < 0 or percentage > 100:
            error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
                code=ROUTING_ERROR_INVALID_ROUTE,
                strategy=strategy,
                vehicle_id="N/A",
                reason=f"Percentage must be between 0 and 100, got {percentage}"
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        percentages[strategy] = percentage

    # Validate sum
    total = sum(percentages.values())
    if abs(total - 100.0) > 0.01:
        error_msg = ROUTING_ERROR_MSG_TEMPLATE.format(
            code=ROUTING_ERROR_INVALID_ROUTE,
            strategy="parsing",
            vehicle_id="N/A",
            reason=f"Routing strategy percentages must sum to 100, got {total}"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    return percentages
