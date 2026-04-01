"""
API server entry point.

Usage:
    # Development
    python -m services.api --reload

    # Production
    python -m services.api --host 0.0.0.0 --port 8000

    # Or via uvicorn directly
    uvicorn services.api.app:app --host 0.0.0.0 --port 8000
"""

import argparse

import structlog
import uvicorn


def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="WoningScout API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")

    args = parser.parse_args()
    setup_logging(level=args.log_level.upper())

    uvicorn.run(
        "services.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
