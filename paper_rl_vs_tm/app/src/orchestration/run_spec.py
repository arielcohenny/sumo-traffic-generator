"""
Data classes for comparison run specifications and metrics.

Defines the structure for specifying comparison runs and storing their results.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from src.constants import VALID_TRAFFIC_CONTROLS


@dataclass
class RunSpec:
    """Specification for a single comparison run.

    Attributes:
        traffic_control: Traffic control method (tree_method, actuated, fixed, atlcs)
        private_traffic_seed: Seed for passenger vehicle generation
        public_traffic_seed: Seed for public vehicle generation
        name: Unique identifier for this run (e.g., "tree_method_seed100")
    """
    traffic_control: str
    private_traffic_seed: int
    public_traffic_seed: int
    name: str

    def __post_init__(self):
        """Validate run spec after initialization."""
        if self.traffic_control not in VALID_TRAFFIC_CONTROLS:
            raise ValueError(
                f"Invalid traffic_control '{self.traffic_control}'. "
                f"Must be one of: {VALID_TRAFFIC_CONTROLS}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "RunSpec":
        """Create RunSpec from dictionary.

        Args:
            data: Dictionary with keys: traffic_control, private_seed, public_seed, name

        Returns:
            RunSpec instance
        """
        return cls(
            traffic_control=data["traffic_control"],
            private_traffic_seed=data.get("private_seed", data.get("private_traffic_seed")),
            public_traffic_seed=data.get("public_seed", data.get("public_traffic_seed")),
            name=data["name"]
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "traffic_control": self.traffic_control,
            "private_seed": self.private_traffic_seed,
            "public_seed": self.public_traffic_seed,
            "name": self.name
        }


@dataclass
class RunMetrics:
    """Metrics collected from a single simulation run.

    Attributes:
        name: Run identifier (matches RunSpec.name)
        traffic_control: Traffic control method used
        private_traffic_seed: Seed used for passenger vehicles
        public_traffic_seed: Seed used for public vehicles
        avg_travel_time: Average vehicle travel time (seconds)
        std_travel_time: Standard deviation of travel times
        min_travel_time: Minimum travel time observed
        max_travel_time: Maximum travel time observed
        avg_waiting_time: Average time spent waiting (seconds)
        std_waiting_time: Standard deviation of waiting times
        completion_rate: Fraction of vehicles that completed their trip
        vehicles_departed: Total vehicles that departed
        vehicles_arrived: Total vehicles that arrived at destination
        throughput: Vehicles per hour that completed trips
        avg_queue_length: Average queue length at intersections
        max_queue_length: Maximum queue length observed
        simulation_time: Total simulation time (seconds)
    """
    name: str
    traffic_control: str
    private_traffic_seed: int
    public_traffic_seed: int

    # Travel time metrics
    avg_travel_time: float = 0.0
    std_travel_time: float = 0.0
    min_travel_time: float = 0.0
    max_travel_time: float = 0.0

    # Waiting time metrics
    avg_waiting_time: float = 0.0
    std_waiting_time: float = 0.0

    # Completion metrics
    completion_rate: float = 0.0
    vehicles_departed: int = 0
    vehicles_arrived: int = 0
    throughput: float = 0.0

    # Queue metrics
    avg_queue_length: float = 0.0
    max_queue_length: float = 0.0

    # Simulation info
    simulation_time: float = 0.0

    # Raw data for distribution charts
    travel_times: List[float] = field(default_factory=list)
    waiting_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "traffic_control": self.traffic_control,
            "private_seed": self.private_traffic_seed,
            "public_seed": self.public_traffic_seed,
            "avg_travel_time": self.avg_travel_time,
            "std_travel_time": self.std_travel_time,
            "min_travel_time": self.min_travel_time,
            "max_travel_time": self.max_travel_time,
            "avg_waiting_time": self.avg_waiting_time,
            "std_waiting_time": self.std_waiting_time,
            "completion_rate": self.completion_rate,
            "vehicles_departed": self.vehicles_departed,
            "vehicles_arrived": self.vehicles_arrived,
            "throughput": self.throughput,
            "avg_queue_length": self.avg_queue_length,
            "max_queue_length": self.max_queue_length,
            "simulation_time": self.simulation_time,
            # Don't include raw data in serialization to keep file size small
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetrics":
        """Create RunMetrics from dictionary."""
        return cls(
            name=data["name"],
            traffic_control=data["traffic_control"],
            private_traffic_seed=data.get("private_seed", 0),
            public_traffic_seed=data.get("public_seed", 0),
            avg_travel_time=data.get("avg_travel_time", 0.0),
            std_travel_time=data.get("std_travel_time", 0.0),
            min_travel_time=data.get("min_travel_time", 0.0),
            max_travel_time=data.get("max_travel_time", 0.0),
            avg_waiting_time=data.get("avg_waiting_time", 0.0),
            std_waiting_time=data.get("std_waiting_time", 0.0),
            completion_rate=data.get("completion_rate", 0.0),
            vehicles_departed=data.get("vehicles_departed", 0),
            vehicles_arrived=data.get("vehicles_arrived", 0),
            throughput=data.get("throughput", 0.0),
            avg_queue_length=data.get("avg_queue_length", 0.0),
            max_queue_length=data.get("max_queue_length", 0.0),
            simulation_time=data.get("simulation_time", 0.0),
        )
