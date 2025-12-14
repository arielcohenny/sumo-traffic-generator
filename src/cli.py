"""
CLI interface for SUMO traffic generator.

"""

import json
import sys
from pathlib import Path
from typing import List

from src.args.parser import create_argument_parser
from src.validate.validate_arguments import validate_arguments
from src.validate.errors import ValidationError
from src.pipeline.pipeline_factory import PipelineFactory
from src.utils.logging import setup_logging, get_logger


def _parse_comparison_runs(args) -> List["RunSpec"]:
    """Parse comparison run specifications from CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of RunSpec instances
    """
    from src.orchestration.run_spec import RunSpec

    run_data = []

    if getattr(args, 'comparison_runs', None):
        run_data = json.loads(args.comparison_runs)
    elif getattr(args, 'comparison_runs_file', None):
        with open(args.comparison_runs_file, 'r') as f:
            run_data = json.load(f)

    return [RunSpec.from_dict(spec) for spec in run_data]


def _print_comparison_table(results: "ComparisonResults") -> None:
    """Print comparison results as a formatted table.

    Args:
        results: ComparisonResults instance
    """
    if not results.runs:
        print("No results to display.")
        return

    # Print header
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    # Print table header
    print(f"{'Run':<25} {'Method':<12} {'Seeds':<12} {'Avg Travel':<12} "
          f"{'Avg Wait':<10} {'Completion':<12} {'Throughput':<12}")
    print("-" * 100)

    # Print each run
    for run in results.runs:
        seeds = f"{run.private_traffic_seed}/{run.public_traffic_seed}"
        print(f"{run.name:<25} {run.traffic_control:<12} {seeds:<12} "
              f"{run.avg_travel_time:>8.1f}s   {run.avg_waiting_time:>6.1f}s   "
              f"{run.completion_rate * 100:>8.1f}%   {run.throughput:>8.1f}/hr")

    print("=" * 100)

    # Print summary by method
    summary = results.to_summary_dict()
    if len(summary) > 1:
        print("\nSUMMARY BY METHOD:")
        print("-" * 60)
        for method, stats in summary.items():
            print(f"  {method}:")
            print(f"    Avg Travel Time: {stats['avg_travel_time']['mean']:.1f}s "
                  f"(range: {stats['avg_travel_time']['min']:.1f}-{stats['avg_travel_time']['max']:.1f})")
            print(f"    Completion Rate: {stats['completion_rate']['mean'] * 100:.1f}%")
            print(f"    Throughput: {stats['throughput']['mean']:.1f} veh/hr")


def main() -> None:
    """Main CLI entry point - pure orchestration."""
    # Setup infrastructure
    setup_logging()
    logger = get_logger(__name__)

    try:
        # Parse and validate arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        validate_arguments(args)

        # Handle network-only generation
        if getattr(args, 'generate_network_only', False):
            from src.orchestration.comparison_runner import ComparisonRunner

            workspace = Path(getattr(args, 'workspace', '.'))
            runner = ComparisonRunner(workspace)
            network_path = runner.generate_network_only(args)
            logger.info(f"Network files saved to: {network_path}")
            return

        # Handle comparison runs
        if getattr(args, 'comparison_runs', None) or getattr(args, 'comparison_runs_file', None):
            from src.orchestration.comparison_runner import ComparisonRunner
            from src.orchestration.comparison_results import ComparisonResults

            workspace = Path(getattr(args, 'workspace', '.'))
            runner = ComparisonRunner(workspace)

            # Load existing network if specified
            if getattr(args, 'use_network_from', None):
                runner.load_existing_network(Path(args.use_network_from))

            # Parse run specs and execute comparison
            run_specs = _parse_comparison_runs(args)

            def progress_callback(status, current, total):
                logger.info(f"Progress: {status} ({current}/{total})")

            results = runner.run_comparison(run_specs, args, progress_callback)
            _print_comparison_table(results)
            return

        # Handle single run with existing network
        if getattr(args, 'use_network_from', None):
            from src.orchestration.comparison_runner import ComparisonRunner
            from src.orchestration.run_spec import RunSpec
            from src.utils.multi_seed_utils import get_private_traffic_seed, get_public_traffic_seed

            workspace = Path(getattr(args, 'workspace', '.'))
            runner = ComparisonRunner(workspace)
            runner.load_existing_network(Path(args.use_network_from))

            # Create a single run spec from args
            run_spec = RunSpec(
                traffic_control=getattr(args, 'traffic_control', 'tree_method'),
                private_traffic_seed=get_private_traffic_seed(args),
                public_traffic_seed=get_public_traffic_seed(args),
                name="single_run"
            )

            metrics = runner.run_single(run_spec, args)
            logger.info(f"Simulation completed: {metrics.name}")
            logger.info(f"  Avg Travel Time: {metrics.avg_travel_time:.1f}s")
            logger.info(f"  Completion Rate: {metrics.completion_rate * 100:.1f}%")
            logger.info(f"  Throughput: {metrics.throughput:.1f} veh/hr")
            return

        # Standard pipeline execution
        pipeline = PipelineFactory.create_pipeline(args)

        if getattr(args, 'file_generation_only', False):
            pipeline.execute_file_generation_only()
        else:
            pipeline.execute()

        # logger.info("Pipeline execution completed successfully")

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
