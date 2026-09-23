"""
Comparison results container and serialization.

Stores aggregated results from multiple comparison runs and provides
conversion to DataFrame for visualization.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.orchestration.run_spec import RunMetrics


def _compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, min, and max statistics
    """
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class ComparisonResults:
    """Aggregated results from all comparison runs.

    Attributes:
        runs: List of metrics from each run
        network_config: Network configuration parameters used
        created_at: Timestamp when comparison was created
        comparison_name: Optional name for this comparison
    """
    runs: List[RunMetrics] = field(default_factory=list)
    network_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    comparison_name: Optional[str] = None

    def add_run(self, metrics: RunMetrics):
        """Add metrics from a completed run."""
        self.runs.append(metrics)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert results to pandas DataFrame for easy visualization.

        Returns:
            DataFrame with one row per run and columns for all metrics

        Raises:
            ImportError: If pandas is not installed
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for run in self.runs:
            rows.append({
                "Run": run.name,
                "Method": run.traffic_control,
                "Private Seed": run.private_traffic_seed,
                "Public Seed": run.public_traffic_seed,
                "Avg Travel Time (s)": round(run.avg_travel_time, 1),
                "Std Travel Time (s)": round(run.std_travel_time, 1),
                "Avg Waiting Time (s)": round(run.avg_waiting_time, 1),
                "Completion Rate (%)": round(run.completion_rate * 100, 1),
                "Throughput (veh/hr)": round(run.throughput, 1),
                "Vehicles Arrived": run.vehicles_arrived,
                "Vehicles Departed": run.vehicles_departed,
                "Avg Queue Length": round(run.avg_queue_length, 2),
                "Max Queue Length": round(run.max_queue_length, 1),
            })

        return pd.DataFrame(rows)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics across all runs.

        Returns:
            Dictionary with summary statistics grouped by traffic control method
        """
        if not self.runs:
            return {}

        # Group by traffic control method
        by_method: Dict[str, List[RunMetrics]] = {}
        for run in self.runs:
            if run.traffic_control not in by_method:
                by_method[run.traffic_control] = []
            by_method[run.traffic_control].append(run)

        summary = {}
        for method, runs in by_method.items():
            summary[method] = {
                "num_runs": len(runs),
                "avg_travel_time": _compute_statistics([r.avg_travel_time for r in runs]),
                "avg_waiting_time": _compute_statistics([r.avg_waiting_time for r in runs]),
                "completion_rate": _compute_statistics([r.completion_rate for r in runs]),
                "throughput": _compute_statistics([r.throughput for r in runs]),
            }

        return summary

    def to_json(self, path: Path):
        """Save results to JSON file.

        Args:
            path: Path to save JSON file
        """
        data = {
            "comparison_name": self.comparison_name,
            "created_at": self.created_at,
            "network_config": self.network_config,
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.to_summary_dict(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "ComparisonResults":
        """Load results from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ComparisonResults instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        results = cls(
            network_config=data.get("network_config", {}),
            created_at=data.get("created_at", ""),
            comparison_name=data.get("comparison_name"),
        )

        for run_data in data.get("runs", []):
            results.runs.append(RunMetrics.from_dict(run_data))

        return results

    def get_best_run(self, metric: str = "avg_travel_time", lower_is_better: bool = True) -> Optional[RunMetrics]:
        """Get the run with the best value for a given metric.

        Args:
            metric: Attribute name in RunMetrics to compare
            lower_is_better: If True, lower values are better (e.g., travel time)

        Returns:
            RunMetrics with the best value, or None if no runs
        """
        if not self.runs:
            return None

        def get_value(run: RunMetrics) -> float:
            return getattr(run, metric, float('inf') if lower_is_better else float('-inf'))

        if lower_is_better:
            return min(self.runs, key=get_value)
        else:
            return max(self.runs, key=get_value)

    def compare_methods(self) -> Dict[str, Dict[str, float]]:
        """Compare average metrics across different traffic control methods.

        Returns:
            Dictionary mapping method names to their average metrics
        """
        summary = self.to_summary_dict()

        comparison = {}
        for method, stats in summary.items():
            comparison[method] = {
                "avg_travel_time": stats["avg_travel_time"]["mean"],
                "avg_waiting_time": stats["avg_waiting_time"]["mean"],
                "completion_rate": stats["completion_rate"]["mean"],
                "throughput": stats["throughput"]["mean"],
            }

        return comparison
