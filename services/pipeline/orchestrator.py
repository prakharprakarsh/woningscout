"""
Pipeline Orchestrator

Wires the 5 agents into a LangGraph StateGraph.

The pipeline is sequential (not parallel) because each agent's output
IS the next agent's input. I considered parallel feature+prediction
but the feature agent's output IS the prediction input, so there's
no real parallelism to exploit.

The one non-trivial part: the conditional edge after scoring.
If no properties are undervalued (value_ratio > 1.05), we skip
the alert agent entirely. No point matching against user profiles
if there's nothing worth alerting about.

LangGraph also gives us:
- Checkpointed state (survives crashes mid-pipeline)
- Clean retry semantics per node
- Conditional routing without spaghetti if/else
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from agents.config import settings
from agents.schemas import PipelineState
from agents.ingestion import IngestionAgent
from agents.features import FeatureAgent
from agents.prediction import PredictionAgent
from agents.scoring import ScoringAgent
from agents.alerting import AlertAgent

logger = structlog.get_logger()


# ── Agent singletons ─────────────────────────────────────────────────
# Created once, reused across pipeline runs. Each agent is stateless
# except for cached models/indices which are read-only after init.

_ingestion = IngestionAgent()
_features = FeatureAgent()
_prediction = PredictionAgent()
_scoring = ScoringAgent()
_alerting = AlertAgent()


# ── Node functions ────────────────────────────────────────────────────
# LangGraph nodes must be sync or async callables that take state and
# return state. We wrap our agent.run() calls here.

async def ingest_node(state: PipelineState) -> PipelineState:
    return await _ingestion.run(state)


async def featurize_node(state: PipelineState) -> PipelineState:
    return await _features.run(state)


async def predict_node(state: PipelineState) -> PipelineState:
    return await _prediction.run(state)


async def score_node(state: PipelineState) -> PipelineState:
    return await _scoring.run(state)


async def alert_node(state: PipelineState) -> PipelineState:
    return await _alerting.run(state)


# ── Conditional routing ───────────────────────────────────────────────

def should_alert(state: PipelineState) -> str:
    """Decide whether to run the alert agent.

    Only alert if we found at least one undervalued property
    (value_ratio > 1.05). Skipping saves us the user-matching
    computation and avoids pointless log noise.
    """
    if state.has_undervalued:
        count = sum(1 for p in state.scored_properties if p.is_undervalued)
        logger.info("routing_to_alerts", undervalued_count=count)
        return "alert"
    else:
        logger.info("skipping_alerts_no_undervalued")
        return "end"


# ── Graph builder ─────────────────────────────────────────────────────

def build_pipeline():
    """Build the agent pipeline as a LangGraph StateGraph.

    Returns a compiled graph that can be invoked with:
        result = await graph.ainvoke(initial_state)

    Graph topology:
        ingest → featurize → predict → score → [alert | end]

    The conditional edge after 'score' checks if any properties
    are undervalued before bothering with user matching.
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(PipelineState)

        # Register nodes
        graph.add_node("ingest", ingest_node)
        graph.add_node("featurize", featurize_node)
        graph.add_node("predict", predict_node)
        graph.add_node("score", score_node)
        graph.add_node("alert", alert_node)

        # Wire edges (linear pipeline with one conditional)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "featurize")
        graph.add_edge("featurize", "predict")
        graph.add_edge("predict", "score")

        # Conditional: only alert if there are undervalued properties
        graph.add_conditional_edges(
            "score",
            should_alert,
            {
                "alert": "alert",
                "end": END,
            },
        )
        graph.add_edge("alert", END)

        compiled = graph.compile()
        logger.info("pipeline_built", engine="langgraph")
        return compiled

    except ImportError:
        logger.warning(
            "langgraph_not_installed_using_simple_pipeline",
            hint="pip install langgraph",
        )
        return None


# ── Simple fallback pipeline (no LangGraph dependency) ────────────────

async def run_simple_pipeline(state: PipelineState) -> PipelineState:
    """Run the pipeline without LangGraph.

    Same logic, just sequential async calls. Used when langgraph
    isn't installed (e.g., in minimal dev environments or CI).
    """
    state = await ingest_node(state)

    if not state.new_listing_ids:
        logger.info("pipeline_short_circuit_no_listings")
        return state

    state = await featurize_node(state)
    state = await predict_node(state)
    state = await score_node(state)

    if state.has_undervalued:
        state = await alert_node(state)
    else:
        logger.info("pipeline_skip_alerts_no_undervalued")

    return state


# ── Public API ────────────────────────────────────────────────────────

async def run_pipeline(
    regions: Optional[list[str]] = None,
    last_scan_ts: Optional[datetime] = None,
) -> PipelineState:
    """Run the full agent pipeline once.

    This is the main entry point. Call it from the scheduler,
    the CLI, or the API endpoint.

    Args:
        regions: Target regions to scan (default: from settings)
        last_scan_ts: Only fetch listings newer than this

    Returns:
        Final PipelineState with all results
    """
    initial_state = PipelineState(
        run_id=str(uuid.uuid4())[:8],
        target_regions=regions or settings.regions_list,
        last_scan_ts=last_scan_ts,
    )

    logger.info(
        "pipeline_start",
        run_id=initial_state.run_id,
        regions=initial_state.target_regions,
        mode="demo" if settings.is_demo_mode else "live",
    )

    start = datetime.now(tz=timezone.utc)

    # Try LangGraph first, fall back to simple pipeline
    graph = build_pipeline()
    if graph is not None:
        try:
            result = await graph.ainvoke(initial_state)
            # LangGraph may return a dict — convert back to PipelineState
            if isinstance(result, dict):
                result = PipelineState(**result)
        except Exception as e:
            logger.error("langgraph_pipeline_failed_falling_back", error=str(e))
            result = await run_simple_pipeline(initial_state)
    else:
        result = await run_simple_pipeline(initial_state)

    elapsed = (datetime.now(tz=timezone.utc) - start).total_seconds()

    logger.info(
        "pipeline_complete",
        run_id=result.run_id,
        elapsed_s=round(elapsed, 2),
        summary=result.summary,
    )

    return result


# ── CLI runner ────────────────────────────────────────────────────────

async def run_continuous(interval_seconds: int = 300):
    """Run the pipeline continuously with a configurable interval.

    This is what the Docker pipeline worker runs. It loops forever,
    sleeping between scans. Each run picks up where the last left off
    via last_scan_ts.
    """
    last_ts = None
    run_count = 0

    logger.info(
        "continuous_mode_started",
        interval_s=interval_seconds,
        regions=settings.regions_list,
    )

    while True:
        run_count += 1
        logger.info("continuous_run_start", run_number=run_count)

        try:
            result = await run_pipeline(last_scan_ts=last_ts)
            last_ts = result.last_scan_ts
        except Exception as e:
            logger.error("continuous_run_failed", error=str(e), run_number=run_count)

        logger.info(
            "continuous_sleeping",
            seconds=interval_seconds,
            next_run=run_count + 1,
        )
        await asyncio.sleep(interval_seconds)
