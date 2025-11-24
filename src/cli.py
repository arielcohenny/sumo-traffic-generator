"""
CLI interface for SUMO traffic generator.

"""

import sys

from src.args.parser import create_argument_parser
from src.validate.validate_arguments import validate_arguments
from src.validate.errors import ValidationError
from src.pipeline.pipeline_factory import PipelineFactory
from src.utils.logging import setup_logging, get_logger


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

        # Create and execute pipeline
        pipeline = PipelineFactory.create_pipeline(args)
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
