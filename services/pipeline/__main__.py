"""
Pipeline CLI entry point.

Usage:
    # Run once
    python -m services.pipeline --region=amsterdam --once

    # Run continuously (every 5 minutes)
    python -m services.pipeline --region=amsterdam

    # Custom interval
    python -m services.pipeline --region=amsterdam,utrecht --interval=600
"""

import argparse
import asyncio
import sys

import structlog


def setup_logging(level: str = "INFO", fmt: str = "console"):
    """Configure structlog for pretty console output or JSON."""
    if fmt == "console":
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(__import__("logging"), level.upper(), __import__("logging").INFO)
            ),
        )
    else:
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
        )


def main():
    parser = argparse.ArgumentParser(
        description="WoningScout Pipeline — scan, predict, alert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --once                      Run once with default regions
  %(prog)s --region=amsterdam --once   Run once for Amsterdam
  %(prog)s --region=amsterdam,utrecht  Run continuously
  %(prog)s --interval=600              Custom scan interval (seconds)
        """,
    )
    parser.add_argument(
        "--region",
        type=str,
        default="",
        help="Comma-separated regions to scan (default: from .env)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run pipeline once and exit (default: run continuously)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scans in continuous mode (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="console",
        choices=["console", "json"],
        help="Log format (default: console)",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, fmt=args.log_format)

    logger = structlog.get_logger()

    # Parse regions
    regions = None
    if args.region:
        regions = [r.strip() for r in args.region.split(",") if r.strip()]

    from services.pipeline.orchestrator import run_pipeline, run_continuous

    if args.once:
        logger.info("mode_single_run", regions=regions)
        result = asyncio.run(run_pipeline(regions=regions))
        logger.info("final_result", summary=result.summary)
        print(f"\n✓ {result.summary}")
    else:
        logger.info("mode_continuous", regions=regions, interval=args.interval)
        try:
            asyncio.run(run_continuous(interval_seconds=args.interval))
        except KeyboardInterrupt:
            logger.info("pipeline_stopped_by_user")
            print("\n✓ Pipeline stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
