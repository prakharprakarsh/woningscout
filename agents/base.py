"""
Base agent interface.

All 5 agents inherit from this. Keeps things consistent:
- Same run() signature (takes state, returns state)
- Same logging setup
- Same metrics collection pattern
- Same error handling hooks

I considered making this more complex (retry decorators, circuit breakers)
but decided to keep the base thin and let each agent handle its own
failure modes. The retry logic lives in the orchestrator instead.
"""

import abc
import time
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

from agents.schemas import PipelineState


# ── Prometheus metrics (shared across all agents) ─────────────────────

AGENT_RUN_DURATION = Histogram(
    "woningscout_agent_run_seconds",
    "Time spent in agent.run()",
    labelnames=["agent_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

AGENT_RUN_TOTAL = Counter(
    "woningscout_agent_runs_total",
    "Total agent executions",
    labelnames=["agent_name", "status"],
)

AGENT_ERRORS = Counter(
    "woningscout_agent_errors_total",
    "Agent errors by type",
    labelnames=["agent_name", "error_type"],
)


class BaseAgent(abc.ABC):
    """Abstract base for all pipeline agents.

    Subclasses must implement:
        - name (property): human-readable agent name
        - _execute(state): the actual work

    The run() method wraps _execute() with logging, timing, and
    error counting. Agents should NOT override run() directly.
    """

    def __init__(self):
        self.logger = structlog.get_logger().bind(agent=self.name)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short name for logging and metrics (e.g. 'ingestion')."""
        ...

    @abc.abstractmethod
    async def _execute(self, state: PipelineState) -> PipelineState:
        """Do the actual work. Override this in subclasses."""
        ...

    async def run(self, state: PipelineState) -> PipelineState:
        """Public entry point — wraps _execute with observability.

        This is what LangGraph calls. Don't override this.
        """
        self.logger.info("agent_start", run_id=state.run_id)
        start = time.monotonic()

        try:
            result = await self._execute(state)
            elapsed = time.monotonic() - start

            AGENT_RUN_DURATION.labels(agent_name=self.name).observe(elapsed)
            AGENT_RUN_TOTAL.labels(agent_name=self.name, status="success").inc()

            self.logger.info(
                "agent_done",
                run_id=state.run_id,
                elapsed_s=round(elapsed, 3),
            )
            return result

        except Exception as e:
            elapsed = time.monotonic() - start
            error_type = type(e).__name__

            AGENT_RUN_TOTAL.labels(agent_name=self.name, status="error").inc()
            AGENT_ERRORS.labels(agent_name=self.name, error_type=error_type).inc()

            self.logger.error(
                "agent_failed",
                run_id=state.run_id,
                error=str(e),
                error_type=error_type,
                elapsed_s=round(elapsed, 3),
            )
            raise

    def _log_stats(self, state: PipelineState, **kwargs: Any) -> None:
        """Convenience: log stats and also write them to state.stats."""
        for k, v in kwargs.items():
            state.stats[f"{self.name}_{k}"] = v
        self.logger.info("agent_stats", **kwargs)
