"""
Streamlit display components for comparison results in DBPS GUI.

Provides visualization of multi-run comparison results including
tables and charts showing averages by method.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, List, Callable
import json

from src.constants import COMPARISON_CHART_HEIGHT, PERCENTAGE_CHART_MAX
from src.orchestration.comparison_results import ComparisonResults

# Try to import plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _aggregate_by_method(results: ComparisonResults) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics by traffic control method.

    Args:
        results: ComparisonResults instance

    Returns:
        Dictionary mapping method name to aggregated metrics
    """
    from statistics import mean, stdev

    by_method: Dict[str, List] = {}

    for run in results.runs:
        method = run.traffic_control
        if method not in by_method:
            by_method[method] = {
                "travel_times": [],
                "waiting_times": [],
                "completion_rates": [],
                "throughputs": [],
            }
        by_method[method]["travel_times"].append(run.avg_travel_time)
        by_method[method]["waiting_times"].append(run.avg_waiting_time)
        by_method[method]["completion_rates"].append(run.completion_rate)
        by_method[method]["throughputs"].append(run.throughput)

    aggregated = {}
    for method, data in by_method.items():
        n = len(data["travel_times"])
        aggregated[method] = {
            "avg_travel_time": mean(data["travel_times"]),
            "std_travel_time": stdev(data["travel_times"]) if n > 1 else 0,
            "avg_waiting_time": mean(data["waiting_times"]),
            "std_waiting_time": stdev(data["waiting_times"]) if n > 1 else 0,
            "avg_completion_rate": mean(data["completion_rates"]),
            "avg_throughput": mean(data["throughputs"]),
            "num_runs": n,
        }

    return aggregated


class ComparisonDisplay:
    """Display components for comparison results."""

    @staticmethod
    def show_results(results: ComparisonResults, key_prefix: str = "main"):
        """Display comprehensive comparison results.

        Args:
            results: ComparisonResults instance with all run metrics
            key_prefix: Unique prefix for Streamlit element keys to avoid duplicates
        """
        if not results.runs:
            st.warning("No results to display.")
            return

        st.header("Comparison Results")

        # Method comparison summary (moved to top)
        ComparisonDisplay._show_method_summary(results)

        # Charts section - showing averages by method
        if HAS_PLOTLY and HAS_PANDAS:
            st.subheader("Visualizations (Averages by Method)")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                ComparisonDisplay._show_travel_time_chart(results, key_prefix)

            with chart_cols[1]:
                ComparisonDisplay._show_throughput_chart(results, key_prefix)

            # Additional charts
            chart_cols2 = st.columns(2)

            with chart_cols2[0]:
                ComparisonDisplay._show_completion_rate_chart(results, key_prefix)

            with chart_cols2[1]:
                ComparisonDisplay._show_waiting_time_chart(results, key_prefix)
        else:
            st.info("Install plotly and pandas for charts: pip install plotly pandas")

        # Detailed runs table (collapsed by default)
        with st.expander("Detailed Results (All Runs)", expanded=False):
            ComparisonDisplay._show_summary_table(results)

        # Export section
        ComparisonDisplay._show_export_options(results, key_prefix)

    @staticmethod
    def _show_summary_table(results: ComparisonResults):
        """Display summary table of all runs."""
        if HAS_PANDAS:
            df = results.to_dataframe()
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            # Fallback: display as simple table
            headers = ["Run", "Method", "Seeds", "Avg Travel (s)", "Avg Wait (s)",
                       "Completion (%)", "Throughput (veh/hr)"]
            rows = []
            for run in results.runs:
                rows.append([
                    run.name,
                    run.traffic_control,
                    f"{run.private_traffic_seed}/{run.public_traffic_seed}",
                    f"{run.avg_travel_time:.1f}",
                    f"{run.avg_waiting_time:.1f}",
                    f"{run.completion_rate * 100:.1f}",
                    f"{run.throughput:.1f}"
                ])

            # Create markdown table
            table_md = "| " + " | ".join(headers) + " |\n"
            table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

            st.markdown(table_md)

    @staticmethod
    def _show_metric_bar_chart(
        results: ComparisonResults,
        key_prefix: str,
        chart_key: str,
        y_label: str,
        title: str,
        value_getter: Callable[[Dict], float],
        std_getter: Optional[Callable[[Dict], float]] = None,
        y_range: Optional[List[float]] = None
    ):
        """Display a bar chart for a metric aggregated by method.

        Args:
            results: ComparisonResults instance
            key_prefix: Unique prefix for Streamlit keys
            chart_key: Chart identifier for the key suffix
            y_label: Label for the y-axis
            title: Chart title
            value_getter: Function to extract value from aggregated dict
            std_getter: Optional function to extract std dev for error bars
            y_range: Optional y-axis range [min, max]
        """
        aggregated = _aggregate_by_method(results)

        data = {
            "Method": list(aggregated.keys()),
            y_label: [value_getter(v) for v in aggregated.values()],
        }
        if std_getter:
            data["Std Dev"] = [std_getter(v) for v in aggregated.values()]

        df = pd.DataFrame(data)

        fig = px.bar(
            df,
            x="Method",
            y=y_label,
            error_y="Std Dev" if std_getter else None,
            title=title,
            color="Method",
        )

        layout_kwargs = {"height": COMPARISON_CHART_HEIGHT, "showlegend": False}
        if y_range:
            layout_kwargs["yaxis_range"] = y_range
        fig.update_layout(**layout_kwargs)

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{chart_key}")

    @staticmethod
    def _show_travel_time_chart(results: ComparisonResults, key_prefix: str):
        """Display average travel time by method."""
        ComparisonDisplay._show_metric_bar_chart(
            results, key_prefix, "travel_time_chart",
            y_label="Avg Travel Time (s)",
            title="Average Travel Time by Method",
            value_getter=lambda v: v["avg_travel_time"],
            std_getter=lambda v: v["std_travel_time"]
        )

    @staticmethod
    def _show_throughput_chart(results: ComparisonResults, key_prefix: str):
        """Display average throughput by method."""
        ComparisonDisplay._show_metric_bar_chart(
            results, key_prefix, "throughput_chart",
            y_label="Avg Throughput (veh/hr)",
            title="Average Throughput by Method",
            value_getter=lambda v: v["avg_throughput"]
        )

    @staticmethod
    def _show_completion_rate_chart(results: ComparisonResults, key_prefix: str):
        """Display average completion rate by method."""
        ComparisonDisplay._show_metric_bar_chart(
            results, key_prefix, "completion_rate_chart",
            y_label="Avg Completion Rate (%)",
            title="Average Completion Rate by Method",
            value_getter=lambda v: v["avg_completion_rate"] * 100,
            y_range=[0, PERCENTAGE_CHART_MAX]
        )

    @staticmethod
    def _show_waiting_time_chart(results: ComparisonResults, key_prefix: str):
        """Display average waiting time by method."""
        ComparisonDisplay._show_metric_bar_chart(
            results, key_prefix, "waiting_time_chart",
            y_label="Avg Waiting Time (s)",
            title="Average Waiting Time by Method",
            value_getter=lambda v: v["avg_waiting_time"],
            std_getter=lambda v: v["std_waiting_time"]
        )

    @staticmethod
    def _show_method_summary(results: ComparisonResults):
        """Display summary statistics grouped by method."""
        aggregated = _aggregate_by_method(results)

        if not aggregated:
            return

        st.subheader("Summary by Method")

        cols = st.columns(len(aggregated))

        for i, (method, stats) in enumerate(aggregated.items()):
            with cols[i]:
                st.markdown(f"**{method}**")
                st.metric(
                    "Avg Travel Time",
                    f"{stats['avg_travel_time']:.1f}s",
                    help=f"Std Dev: {stats['std_travel_time']:.1f}s"
                )
                st.metric(
                    "Avg Completion Rate",
                    f"{stats['avg_completion_rate'] * 100:.1f}%",
                )
                st.metric(
                    "Avg Throughput",
                    f"{stats['avg_throughput']:.0f} veh/hr",
                )
                st.caption(f"Based on {stats['num_runs']} runs")

    @staticmethod
    def _show_export_options(results: ComparisonResults, key_prefix: str):
        """Display export options for results."""
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export as JSON
            results_dict = {
                "comparison_name": results.comparison_name,
                "created_at": results.created_at,
                "network_config": results.network_config,
                "runs": [run.to_dict() for run in results.runs],
                "summary": results.to_summary_dict()
            }
            json_str = json.dumps(results_dict, indent=2)

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="comparison_results.json",
                mime="application/json",
                key=f"{key_prefix}_download_json_btn"
            )

        with col2:
            # Export as CSV (if pandas available)
            if HAS_PANDAS:
                df = results.to_dataframe()
                csv_str = df.to_csv(index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name="comparison_results.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_download_csv_btn"
                )

    @staticmethod
    def load_and_show_results(results_path: Path):
        """Load results from file and display them.

        Args:
            results_path: Path to comparison_results.json file
        """
        try:
            results = ComparisonResults.from_json(results_path)
            ComparisonDisplay.show_results(results)
        except FileNotFoundError:
            st.error(f"Results file not found: {results_path}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in results file: {e}")
        except Exception as e:
            st.error(f"Error loading results: {e}")
